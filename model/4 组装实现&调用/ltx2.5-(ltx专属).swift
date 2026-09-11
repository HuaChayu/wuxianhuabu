//
//  视频模型调用&回执.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/20.
//
//  ============================================================
//  文件作用：LTX2.5 视频生成管线调用入口（从 ltx-test/main.swift 移植）。
//  包含：文本条件编码 → 去噪 latent（DiT 采样）→ VAE 解码 → 帧导出 →
//        mp4 合成 → 音频解码 → 音轨混流 的完整调用链。
//  依赖实现文件：ltx2.5/LTXDiT.swift、Gemma4Text.swift、Connector.swift
//  依赖算子文件：算子/Sampler.swift、VaeDecoder.swift、AudioVaeDecoder.swift
//  ============================================================
//

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXNN
import Hub
import Tokenizers
import AVFoundation
import CoreVideo
import CoreMedia
import ImageIO
import CoreGraphics
import AppKit
import Darwin
import Accelerate

// ============================================================
// 已迁移至 通用公共函数/模型公共函数.swift（2026-08-29）：
//   runOnBigStack / pipelineLog / 内存自适应策略
//   （GenerationTask·GenerationStage·MemoryModule·MemoryPolicy）/
//   loadImageBCFHW / fillPixelBuffer / writeMp4 /
//   readSafetensorsFloatArrays / archiveIfExists / muxAudio
// ============================================================

// MARK: - 主入口

/// 端到端生成：文本 → 去噪 latent（DiT）→ VAE 解码 → 音轨合成。
/// - Parameters:
///   - prompt: 视频提示词；为空时使用 GenConfig 默认提示词。
///   - audioPath: 可选音频条件（音频节点连入时），16kHz 立体声 wav 路径。
///   - imagePaths: 可选图片条件（图片节点连入时，最多 2 张），首张作 I2V 首帧。
/// - Returns: 生成视频 mp4 的绝对路径；任一步失败返回 nil。
/// 内存策略：cacheLimit 按物理内存自适应，阶段间 clearCache（与 ltx-test 验证一致）。
public func runVideoPipeline(prompt: String = "", audioPath: String? = nil, imagePaths: [String] = [], width: Int? = nil, height: Int? = nil, duration: VideoDuration = .fiveSeconds, stage2Refine: Bool = true, negativePrompt: String? = nil, isCancelled: @escaping () -> Bool = { false }) async -> String? {
    memProfilePoint("管线开始 基线")
    MemoryPolicy.report()
    MLX.Memory.cacheLimit = MemoryPolicy.bufferCacheLimit
    MLX.Memory.memoryLimit = MemoryPolicy.memoryLimit   // 总内存硬上限 = 物理内存 70%
    // 任务开始：互斥规则（≤64GB）——生成视频先卸对方（HiDream），确保本次任务独占内存；
    // 本任务链（LTX 侧）受保护集保护不被误卸；>64GB 不互斥，异模型保留
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .encoding, next: .sampling)
    await generateVideoTest(prompt: prompt, audioPath: audioPath, imagePaths: imagePaths, width: width, height: height, duration: duration, stage2Refine: stage2Refine, negativePrompt: negativePrompt, isCancelled: isCancelled)
    if isCancelled() {
        pipelineLog("⏹ 视频生成已取消，跳过 VAE 解码与落盘")
        return nil
    }
    MLX.Memory.clearCache()   // generateVideoTest 已返回，DiT/文本权重缓冲全部回收
    let outPath = vaeDecodeTest()
    MLX.Memory.clearCache()   // VAE/音频权重缓冲回收
    MemoryPolicy.unloadModelsIfNeeded()   // 廉价设备：生成完动态卸载常驻模型
    let snap = MLX.Memory.snapshot()
    let gb: (Int) -> String = { String(format: "%.2fGB", Double($0) / Double(1 << 30)) }
    stageEnter(phase: .done, detail: "✅ 端到端完成：active=\(gb(snap.activeMemory)) cache=\(gb(snap.cacheMemory)) peak=\(gb(snap.peakMemory))")
    memProfileReport()   // 一次性输出全部内存打点
    return outPath
}

// MARK: - 视频生成尺寸对齐公共函数（已迁出）

// [迁移] 视频尺寸入口计算（videoLatentAlignMultiple / videoResolution /
//        nearestRatio / videoSize）已抽为跨模型公共文件：
//        model/3模块/视频尺寸入口-通用.swift
//        共用方：LTX-2.5（ltx25Distill）与 MiniMax H3（minimaxH3）。
//        本文件内调用点（generateVideoTest）与 UI 收集阶段（startVideoGeneration）
//        均自动指向该公共实现，签名与行为保持不变。

struct GenConfig {
    var prompt: String = "A majestic golden retriever running through a sunlit meadow, slow motion, cinematic lighting"
    var numFrames: Int = 120         // 全量：5s @ 24fps
    var height: Int = 256         // 生成尺寸 = 目标 / 2（512），升频 ×2 后还原
    var width: Int = 256
    var frameRate: Float = 24.0
    var numSteps: Int = 8         // dev+CFG：参考 stage1 默认 30 步，测试先 20（蒸馏 6 步不够）
    var seed: UInt64 = 42
    var numKeyframes: Int = 7     // 生成关键帧数量（内部等距锚点，0=关闭）
}

func encodePrompt(_ prompt: String) async -> (video: MLXArray, audio: MLXArray)? {
    let modelDir = CommonPaths.gemmaDir
    let ltxDir = CommonPaths.ltxServeDir
    let t0 = Date()

    // tokenizer（常驻缓存，换模型目录才重载）
    let tokenizer: any Tokenizer
    if let cached = TextEncoderCache.shared.tokenizer(for: modelDir) {
        tokenizer = cached
        stageEnter(phase: .encoding, detail: "✅ 复用已加载 tokenizer")
    } else {
        stageEnter(phase: .encoding, detail: "加载 tokenizer...")
        let config = LanguageModelConfigurationFromHub(
            modelFolder: URL(fileURLWithPath: modelDir))
        guard let tokenizerConfig = try? await config.tokenizerConfig,
              let tokenizerData = try? await config.tokenizerData else {
            pipelineLog("❌ tokenizer 加载失败"); return nil
        }
        let t = try! AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        TextEncoderCache.shared.storeTokenizer(t, dir: modelDir)
        tokenizer = t
    }
    let rawIds = tokenizer.encode(text: prompt, addSpecialTokens: false)
    stageEnter(phase: .encoding, detail: "✅ tokenizer 就绪（\(Int(Date().timeIntervalSince(t0)))s），prompt token 数：\(rawIds.count)")

    // Gemma 12B 文本编码器（常驻缓存，权重文件未变则复用）
    let gemmaPath = "\(modelDir)/model.safetensors"
    let gemmaModel: Gemma4TextEncoder
    if let cached = TextEncoderCache.shared.gemma(for: gemmaPath) {
        gemmaModel = cached
        stageEnter(phase: .encoding, detail: "✅ 复用已加载 Gemma 权重（跳过加载）")
    } else {
        guard let gemmaWeights = try? MLX.loadArrays(url: URL(fileURLWithPath: gemmaPath)) else {
            pipelineLog("❌ gemma model.safetensors 加载失败"); return nil
        }
        // safetensors 键带 "model.language_model." 前缀，剥离后才匹配 Gemma4TextEncoder 顶层参数
        var strippedG: [String: MLXArray] = [:]
        for (k, v) in gemmaWeights {
            strippedG[k.hasPrefix("model.language_model.") ? String(k.dropFirst("model.language_model.".count)) : k] = v
        }
        let m = Gemma4TextEncoder()
        let gemmaNested = NestedDictionary<String, MLXArray>.unflattened(strippedG)
        m.update(parameters: gemmaNested)
        stageEnter(phase: .encoding, detail: "✅ Gemma 权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s）")
        TextEncoderCache.shared.storeGemma(m, path: gemmaPath)
        gemmaModel = m
    }

    // LTX 连接器（常驻缓存，权重文件未变则复用）
    let connPath = "\(ltxDir)/connector.safetensors"
    let connector: LTXConnector
    if let cached = TextEncoderCache.shared.connector(for: connPath) {
        connector = cached
        stageEnter(phase: .encoding, detail: "✅ 复用已加载 connector（跳过加载）")
    } else {
        guard let connWeights0 = try? MLX.loadArrays(url: URL(fileURLWithPath: connPath)) else {
            pipelineLog("❌ connector.safetensors 加载失败"); return nil
        }
        var connWeights: [String: MLXArray] = [:]
        for (k, v) in connWeights0 {
            // safetensors 键带 "connector." 前缀，剥离后匹配 LTXConnector 顶层参数
            var nk = k
            if nk.hasPrefix("connector.") { nk = String(nk.dropFirst("connector.".count)) }
            nk = nk.replacingOccurrences(of: "ff.net.0.proj", with: "ff.net.proj0")
            nk = nk.replacingOccurrences(of: "ff.net.2", with: "ff.net.proj2")
            nk = nk.replacingOccurrences(of: "attn1.to_out.0", with: "attn1.to_out0")
            connWeights[nk] = v
        }
        let c = LTXConnector()
        let connNested = NestedDictionary<String, MLXArray>.unflattened(connWeights)
        c.update(parameters: connNested)
        stageEnter(phase: .encoding, detail: "✅ 连接器权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s）")
        TextEncoderCache.shared.storeConnector(c, path: connPath)
        connector = c
    }

    let padID: Int32 = 0
    var ids = [Int32](repeating: padID, count: 256)
    let n = min(rawIds.count, 256)
    for i in 0..<n { ids[256 - n + i] = Int32(rawIds[i]) }   // 左 pad
    let idsArr = MLXArray(ids, [1, 256]).asType(.int32)
    stageEnter(phase: .encoding, detail: "有效 token 数：\(n)（左 pad \(256 - n)）")

    stageEnter(phase: .encoding, detail: "Gemma4 前向（48 层，约 1-3 分钟）...")
    let t1 = Date()
    let states = gemmaModel.encodeStates(idsArr)
    stageEnter(phase: .encoding, detail: "✅ 49 态捕获完成，耗时 \(Int(Date().timeIntervalSince(t1)))s")

    stageEnter(phase: .encoding, detail: "连接器变换 + 投影...")
    let t2 = Date()
    let (videoCond0, audioCond0) = connectorProject(states: states, connector: connector)
    // 后半段（对齐 mlx-serve encodeTextLtx）：投影后再经 video/audio_embeddings_connector
    // 的 128 学习寄存器 + 8 个 gated-attention block 处理，才是 DiT 的文本条件。
    // 缺这步会导致文本条件语义对齐不足，换提示词画面骨架雷同。
    let nValid = n
    let videoCond = connector.video_embeddings_connector(videoCond0, nValid: nValid)
    let audioCond = connector.audio_embeddings_connector(audioCond0, nValid: nValid)
    eval(videoCond, audioCond)
    stageEnter(phase: .encoding, detail: "✅ 条件嵌入完成（耗时 \(Int(Date().timeIntervalSince(t2)))s）：video \(videoCond.shape)，audio \(audioCond.shape)")
    let vArr = videoCond.asArray(Float.self)
    let aArr = audioCond.asArray(Float.self)
    let vMean = vArr.reduce(0, +) / Float(vArr.count)
    let aMean = aArr.reduce(0, +) / Float(aArr.count)
    let vMax = vArr.map { abs($0) }.max() ?? 0
    let aMax = aArr.map { abs($0) }.max() ?? 0
    stageEnter(phase: .encoding, detail: "cond 统计：prompt「\(prompt.prefix(40))」video mean=\(vMean) maxAbs=\(vMax) | audio mean=\(aMean) maxAbs=\(aMax)")
    return (videoCond, audioCond)
}

func makeVideoLatentAndPos(g: GenConfig, keyframePixelFrames: [Int] = []) -> (latent: MLXArray, pos: [Float]) {
    let F = (g.numFrames + 7) / 8
    let K = keyframePixelFrames.count
    let H = g.height / 32
    let W = g.width / 32
    MLXRandom.seed(g.seed)
    // 追加 K 个关键帧 slot（单帧 token，官方 VideoGeneratedKeyframeSlots）：
    // latent 序列 = [视频 F 帧, slot 帧 ×K]，slot 帧同样从噪声出发全程去噪生成。
    let noise = MLXRandom.normal([1, F + K, H, W, 128]).asType(PrecisionPolicy.defaultMainDType)
    var pos = [Float](repeating: 0, count: (F + K) * H * W * 3)
    for f in 0..<F {
        let fs = max(Float(f) * 8.0 + 1.0 - 8.0, 0.0)
        let fe = max(Float(f + 1) * 8.0 + 1.0 - 8.0, 0.0)
        let fmid = (fs + fe) / 2.0 / g.frameRate
        for h in 0..<H {
            let hmid = Float(h) * 32.0 + 16.0
            for w in 0..<W {
                let wmid = Float(w) * 32.0 + 16.0
                let n = (f * H + h) * W + w
                pos[n * 3 + 0] = fmid
                pos[n * 3 + 1] = hmid
                pos[n * 3 + 2] = wmid
            }
        }
    }
    // slot 帧位置：官方 _slot_positions 将 t 轴钉在单像素帧 [p, p+1)（除以 fps），
    // Swift RoPE 用跨度中点 → (p + 0.5) / fps；空间坐标与视频帧同网格。
    for (ki, p) in keyframePixelFrames.enumerated() {
        let f = F + ki
        let fmid = (Float(p) + 0.5) / g.frameRate
        for h in 0..<H {
            let hmid = Float(h) * 32.0 + 16.0
            for w in 0..<W {
                let wmid = Float(w) * 32.0 + 16.0
                let n = (f * H + h) * W + w
                pos[n * 3 + 0] = fmid
                pos[n * 3 + 1] = hmid
                pos[n * 3 + 2] = wmid
            }
        }
    }
    return (noise, pos)
}

/// 官方 evenly spaced keyframe 像素帧位置：linspace(0, numFrames-1, K+2).round()[1:-1]。
/// 返回内部等距像素帧索引（排除首尾），供 makeVideoLatentAndPos 追加 slot 与 makeKeyframesMLX 标记共用。
func computeKeyframePixelFrames(numFrames: Int, numKeyframes: Int) -> [Int] {
    guard numKeyframes > 0, numFrames >= numKeyframes + 2 else { return [] }
    var pixelPositions: [Int] = []
    let denom = Double(numKeyframes + 1)   // K+2 点 → K+1 段
    for i in 1...numKeyframes {
        let v = Double(i) * Double(numFrames - 1) / denom
        pixelPositions.append(Int(v.rounded()))
    }
    return pixelPositions
}

/// 构造 keyframesMLX 标记 [Nv] f32：1 = 该 latent token 是生成关键帧 slot。
/// 对齐官方 VideoGeneratedKeyframeSlots：slot 帧追加在 latent 序列末尾（makeVideoLatentAndPos
/// 已追加），此处将末尾 K 个 latent 帧（每帧 H*W token）整体置 1，作为 keyframes_mask 注入
/// keyframes_abs_pos_embedding；slot 帧 denoise_mask=1 全程去噪（与普通生成帧一致）。
/// - Parameter keyframePixelFrames: computeKeyframePixelFrames 的像素帧位置（空 = 关闭，返回 nil）
func makeKeyframesMLX(keyframePixelFrames: [Int], latentFrames: Int, height: Int, width: Int) -> MLXArray? {
    let K = keyframePixelFrames.count
    guard K > 0, latentFrames >= K else { return nil }
    let hw = height * width
    var marks = [Float](repeating: 0, count: latentFrames * hw)
    for k in 0..<K {
        let f = latentFrames - K + k
        for n in (f * hw)..<((f + 1) * hw) {
            marks[n] = 1.0
        }
    }
    pipelineLog("🎯 keyframes slot：追加 \(K) 个单帧（像素帧 \(keyframePixelFrames)）→ 末尾 \(K) 帧标记，\(marks.reduce(0) { $0 + Int($1) })/\(marks.count) tokens")
    return MLXArray(marks).asType(.float32)
}

func makeAudioLatentAndPos(g: GenConfig) -> (latent: MLXArray, pos: [Float]) {
    let Na = Int((Float(g.numFrames) / g.frameRate * 25.0).rounded())
    MLXRandom.seed(g.seed &+ 1)
    let noise = MLXRandom.normal([1, Na, 128]).asType(PrecisionPolicy.defaultMainDType)
    return (noise, makeAudioPositions(count: Na))
}

/// 音频时间位置（秒），对齐官方 audio_pos 计算：每个 token 覆盖 4 帧（25 token/秒）。
func makeAudioPositions(count: Int) -> [Float] {
    var pos = [Float](repeating: 0, count: count)
    for i in 0..<count {
        let start = max(Float(i) * 4.0 + 1.0 - 4.0, 0.0) * 160.0 / 16000.0
        let end = max(Float(i + 1) * 4.0 + 1.0 - 4.0, 0.0) * 160.0 / 16000.0
        pos[i] = (start + end) / 2.0
    }
    return pos
}

/// 读取任意音频文件（wav/mp3/m4a/flac 等）→ 16kHz 立体声 interleaved f32 PCM [-1,1]。
/// 返回 nil 表示读取失败；帧数 = count/2。
func loadAudioPCM16kStereo(path: String) -> [Float]? {
    let url = URL(fileURLWithPath: path)
    guard FileManager.default.fileExists(atPath: path),
          let file = try? AVAudioFile(forReading: url) else { return nil }
    let srcRate = file.processingFormat.sampleRate
    let srcCh = Int(file.processingFormat.channelCount)
    guard srcRate > 0, srcCh > 0, file.length > 0 else { return nil }

    // 读取源文件 → float32 非交错 buffer（统一用 converter 转 16k 立体声）
    let floatFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: srcRate,
                                    channels: AVAudioChannelCount(srcCh), interleaved: false)!
    guard let srcBuf = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: AVAudioFrameCount(file.length)),
          let fileF = try? AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false) else { return nil }
    try? fileF.read(into: srcBuf)
    let frames = Int(srcBuf.frameLength)
    guard frames > 0 else { return nil }

    // 目标格式：16kHz 立体声（即使源是单声道也双声道输出，混流用）
    let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                  channels: 2, interleaved: true)!
    let outCap = AVAudioFrameCount(Int64(frames) * 16000 / Int64(srcRate) + 16)
    guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCap) else { return nil }
    guard let converter = AVAudioConverter(from: floatFormat, to: outFormat) else { return nil }

    var inputDone = false
    var srcPos: AVAudioFramePosition = 0
    converter.convert(to: outBuf, error: nil) { _, status in
        if inputDone { status.pointee = .noDataNow; return nil }
        guard let b = AVAudioPCMBuffer(pcmFormat: floatFormat, frameCapacity: AVAudioFrameCount(frames)) else {
            status.pointee = .noDataNow; return nil
        }
        let remaining = frames - Int(srcPos)
        guard remaining > 0 else { inputDone = true; status.pointee = .noDataNow; return nil }
        let copy = min(remaining, Int(b.frameCapacity))
        for ch in 0..<srcCh {
            if let src = srcBuf.floatChannelData?[ch], let dst = b.floatChannelData?[ch] {
                memcpy(dst, src + Int(srcPos), copy * MemoryLayout<Float>.size)
            }
        }
        b.frameLength = AVAudioFrameCount(copy)
        srcPos += Int64(copy)
        if srcPos >= Int64(frames) { inputDone = true }
        status.pointee = .haveData
        return b
    }

    guard outBuf.frameLength > 0, let data = outBuf.floatChannelData?[0] else { return nil }
    // interleaved 立体声缓冲实际含 frameLength × channels 个 Float（L/R 交错）；
    // 之前只读 frameLength 导致音频条件只取到前半段（如 5.17s → 2.58s），口型后半无条件可看。
    let n = Int(outBuf.frameLength) * Int(outFormat.channelCount)
    return Array(UnsafeBufferPointer(start: data, count: n))
}

/// 读取图片文件 → [1,3,1,H,W] f32（RGB，0-1），供 vaeEncodeImage 首帧条件编码。
/// 等比缩放图片到目标宽高（默认 512×512，对齐视频分辨率），不足尺寸中心补零保持宽高比。
/// 目标宽高需为 32 的倍数（LTX-2.5 latent 对齐要求），首帧/尾帧图统一用同一目标尺寸。
// MARK: - DiT 权重持久缓存（方案 A）

/// DiT 权重常驻：首次从 safetensors 加载后复用，不随每次生成重建/释放。
/// stage2 refine 完成后 latent 是否已是全分辨率（vaeDecodeTest 据此跳过升频，直接 VAE 解码）
var videoLatentIsFullRes = false

/// 与 Sampler.swift 的 CompiledForwardCache 配合：权重不换、shape 不变时，
/// 编译图也不重建不释放，根治采样结束后的大图递归析构卡顿。
final class DiTModelCache {
    static let shared = DiTModelCache()
    private var dit: LTXVideoDiT?
    private var loadedPath: String = ""
    private var loadedModTime: Date?

    /// 路径未变且文件未改动 → 返回已加载实例；否则返回 nil（需要重新加载）。
    func get(path: String) -> LTXVideoDiT? {
        let mod = FileManager.default.fileModTime(path)
        if let d = dit, loadedPath == path, loadedModTime == mod {
            return d
        }
        return nil
    }

    func store(_ d: LTXVideoDiT, path: String) {
        dit = d
        loadedPath = path
        loadedModTime = FileManager.default.fileModTime(path)
    }

    /// 卸载 DiT 权重（须先清 CompiledForwardCache，否则编译闭包仍强持有权重）
    func clear() {
        dit = nil
        loadedPath = ""
        loadedModTime = nil
    }

    /// DiT 是否已缓存（监控面板权重加载状态用）
    var hasDiT: Bool { dit != nil }

    /// 已加载权重的文件大小（字节），未加载为 0（监控面板动态展示用）
    var loadedSizeBytes: Int {
        guard !loadedPath.isEmpty else { return 0 }
        return fileSizeBytes(loadedPath) ?? 0
    }

}

// MARK: - 文本编码器持久缓存（Gemma + connector + tokenizer）

/// Gemma 12B 文本编码器与 connector 常驻：首次加载后复用，换提示词不用重新灌权重，
/// 只重新编码 token（从几分钟降到几秒）。
final class TextEncoderCache {
    static let shared = TextEncoderCache()

    private var gemma: Gemma4TextEncoder?
    private var gemmaPath: String = ""
    private var gemmaModTime: Date?
    private var connector: LTXConnector?
    private var connectorPath: String = ""
    private var connectorModTime: Date?
    private var tokenizer: (any Tokenizer)?
    private var tokenizerDir: String = ""

    func gemma(for path: String) -> Gemma4TextEncoder? {
        guard let g = gemma, gemmaPath == path, gemmaModTime == FileManager.default.fileModTime(path) else { return nil }
        return g
    }

    func storeGemma(_ g: Gemma4TextEncoder, path: String) {
        gemma = g; gemmaPath = path; gemmaModTime = FileManager.default.fileModTime(path)
    }

    func connector(for path: String) -> LTXConnector? {
        guard let c = connector, connectorPath == path, connectorModTime == FileManager.default.fileModTime(path) else { return nil }
        return c
    }

    func storeConnector(_ c: LTXConnector, path: String) {
        connector = c; connectorPath = path; connectorModTime = FileManager.default.fileModTime(path)
    }

    /// Gemma 文本编码器是否已缓存（监控面板权重加载状态用）
    var hasGemma: Bool { gemma != nil }

    /// Connector 是否已缓存（监控面板权重加载状态用）
    var hasConnector: Bool { connector != nil }

    /// Gemma 与 connector 是否均已缓存（加载前评估水位：均已缓存 → newSize=0，永不触发卸载，直接复用）
    var hasTextEncoder: Bool {
        gemma != nil && connector != nil
    }

    func tokenizer(for dir: String) -> (any Tokenizer)? {
        guard let t = tokenizer, tokenizerDir == dir else { return nil }
        return t
    }

    func storeTokenizer(_ t: any Tokenizer, dir: String) {
        tokenizer = t; tokenizerDir = dir
    }

    /// 只卸载 connector（5.9G；可拆分时按体积从小到大单独卸，先卸小的）
    func clearConnector() {
        connector = nil; connectorPath = ""; connectorModTime = nil
    }

    /// 只卸载 Gemma（6.2G；tokenizer 体积小保留复用）
    func clearGemma() {
        gemma = nil; gemmaPath = ""; gemmaModTime = nil
    }

    /// 卸载 Gemma + connector + tokenizer
    func clear() {
        gemma = nil; gemmaPath = ""; gemmaModTime = nil
        connector = nil; connectorPath = ""; connectorModTime = nil
        tokenizer = nil; tokenizerDir = ""
    }

}

func generateVideoTest(prompt: String = "", audioPath: String? = nil, imagePaths: [String] = [], width: Int? = nil, height: Int? = nil, duration: VideoDuration = .fiveSeconds, stage2Refine: Bool = true, negativePrompt: String? = nil, isCancelled: @escaping () -> Bool = { false }) async {
    MonitorCenter.shared.taskStart("视频生成")
    defer { MonitorCenter.shared.taskEnd("视频生成") }
    pipelineLog("\n=== ⑥ 端到端视频生成（文本 → 去噪 latent）===")
    var g = GenConfig()
    if !prompt.isEmpty { g.prompt = prompt }   // UI 传入提示词覆盖默认值
    g.numFrames = duration.numFrames          // 5s → 120 帧 / 10s → 240 帧（@24fps）
    // 尺寸：UI/收集阶段传入的是目标分辨率（64 倍数表）。
    //       stage2Refine 开 → 空间升频器 ×2 已接入，生成尺寸 = 目标 / 2
    //       （64 倍数除 2 后天然 32 倍数，latent 对齐），升频后输出恰为目标分辨率，
    //       I2V 参考图同样按半尺寸编码，升频后还原；
    //       stage2Refine 关 → 直出目标分辨率（不除2/不升频/不二采），仅对齐 32。
    if let width, let height {
        let m = videoLatentAlignMultiple()
        if stage2Refine {
            g.width = max(m, (width / 2 / m) * m)
            g.height = max(m, (height / 2 / m) * m)
        } else {
            g.width = max(m, (width / m) * m)
            g.height = max(m, (height / m) * m)
        }
    } else if let first = imagePaths.first,
              let img = NSImage(contentsOfFile: first) {
        let aligned = videoSize(imageWidth: Int(img.size.width), imageHeight: Int(img.size.height), quality: .standard)
        let m = videoLatentAlignMultiple()
        if stage2Refine {
            g.width = max(m, (aligned.width / 2 / m) * m)
            g.height = max(m, (aligned.height / 2 / m) * m)
            pipelineLog("✅ 图片驱动：生成尺寸 \(g.width)×\(g.height)（目标 \(aligned.width)×\(aligned.height)，升频×2 后还原）")
        } else {
            g.width = max(m, (aligned.width / m) * m)
            g.height = max(m, (aligned.height / m) * m)
            pipelineLog("✅ 图片驱动：直出目标 \(g.width)×\(g.height)（stage2 关，不升频）")
        }
    }
    pipelineLog("提示词：\(g.prompt)")
    if stage2Refine {
        pipelineLog("配置：\(g.numFrames) 帧 @\(g.frameRate)fps，生成 \(g.height)×\(g.width)（升频×2 → 输出 \(g.height * 2)×\(g.width * 2)），\(g.numSteps) 步，seed=\(g.seed)")
    } else {
        pipelineLog("配置：\(g.numFrames) 帧 @\(g.frameRate)fps，直出 \(g.height)×\(g.width)（stage2 关，不升频不二采），\(g.numSteps) 步，seed=\(g.seed)")
    }

    // 0) 管线配置查表：骨架只读表 + 按开关调用公共函数
    //    Stage1 固定走蒸馏（dev 工厂保留不调用，留 dev 管线）；Stage2 固定走 refine（IC-LoRA 已移除）
    let stage1Config = Stage1Config.distilled(g: g)
    let stage2Config: Stage2Config? = stage2Refine ? Stage2Config.refine(g: g) : nil
    pipelineLog("管线模式：Stage1[蒸馏] \(stage1Config.negStrategy == .passThrough ? "+CFG" : "") → Stage2[\(stage2Config?.mode == .refine ? "旧升频+refine" : "关")]")

    // 1) 正向文本条件编码（蒸馏：cfg=1.0 无引导，仅正向；dev 已停用）
    memProfilePoint("文本编码前")
    async let posCond = encodePrompt(g.prompt)
    guard let cond = await posCond else { pipelineLog("❌ 文本编码失败"); return }
    memProfilePoint("文本编码后")
    let condV = cond.video
    let condA = cond.audio
    // dev 负向条件（CFG）：复用同一条 Gemma 编码管线，仅多一次 12B 前向；蒸馏 cfg=1.0 传 nil 跳过 CFG。
    // 负向编码失败时降级为无 CFG 继续（不阻塞生成）
    var negV: MLXArray? = nil
    var negA: MLXArray? = nil
    if stage1Config.negStrategy == .passThrough {
        let negText: String
        if let np = negativePrompt, !np.isEmpty {
            negText = np   // 用户显式传入非空负向词 → 优先使用
        } else {
            negText = "blurry, low quality, distorted, deformed, jpeg artifacts, watermark, text overlay, flickering, ghosting, duplicate motion, extra limbs, anatomical errors, oversaturated, overexposed"   // 默认负向词
        }
        pipelineLog("负向提示词：\(negText.prefix(80))...")
        memProfilePoint("负向编码前")
        if let ncond = await encodePrompt(negText) {
            negV = ncond.video
            negA = ncond.audio
            stageEnter(phase: .encoding, detail: "✅ 负向条件嵌入完成：video \(String(describing: negV?.shape))，audio \(String(describing: negA?.shape))")
        } else {
            pipelineLog("⚠️ 负向编码失败，降级为无 CFG") 
        }
        memProfilePoint("负向编码后")
    }
    // 2)-4) 加载/复用 DiT 权重 + 噪声 latent + 采样（方案 A：DiT 权重与编译图常驻缓存复用，
    //        不随每次生成重建/释放，避免大图一次性析构造成 app 假死）
    let ditPath = stage1Config.ditPath
    // 文本编码完成，DiT 采样前：加载前评估水位（DiT 未缓存按权重文件大小动态预估；已缓存复用传 0 不触发卸载），保护 编码+采样 阶段，预留按尺寸+时长动态
    let ditSizeBytes = fileSizeBytes(ditPath) ?? 0
    guard MemoryPolicy.ensureCapacity(for: DiTModelCache.shared.get(path: ditPath) == nil ? ditSizeBytes : 0, task: .video, current: .encoding, next: .sampling, width: g.width, height: g.height, frames: g.numFrames) else {
        pipelineLog("❌ 内存不足，无法加载该模型（DiT 文件约 \(ditSizeBytes / 1_000_000_000)G），已拦截采样，本次视频生成中止")
        return
    }
    let dit: LTXVideoDiT
    memProfilePoint("DiT加载前")
    if let cached = DiTModelCache.shared.get(path: ditPath) {
        dit = cached
        stageEnter(phase: .sampling, detail: "✅ 复用已加载 DiT 权重（跳过加载）", protect: [.dit])
    } else {
        guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: ditPath)) else {
            pipelineLog("❌ transformer 权重加载失败"); return
        }
        let t3 = Date()
        // 量化参数从 safetensors 文件头动态推断（4bit/8bit 通用，换地址即可；带缓存）
        let q = cachedQuantParams(from: ditPath) ?? (bits: 4, groupSize: 64)
        stageEnter(phase: .sampling, detail: "构建 LTXVideoDiT（48 层，bits=\(q.bits) groupSize=\(q.groupSize)）...")
        let d = LTXVideoDiT(groupSize: q.groupSize, bits: q.bits)
        // safetensors 键带 "transformer." 前缀，剥离后 unflattened 才能匹配模型顶层
        var strippedW: [String: MLXArray] = [:]
        for (k, v) in weights {
            strippedW[k.hasPrefix("transformer.") ? String(k.dropFirst("transformer.".count)) : k] = v
        }
        let nested = NestedDictionary<String, MLXArray>.unflattened(strippedW)
        d.update(parameters: nested)
        // keyframes_abs_pos_embedding 是裸 MLXArray 不在 update 范围，手动补灌（I2V 首帧位置嵌入）
        if let kf = strippedW["keyframes_abs_pos_embedding"] {
            d.keyframes_abs_pos_embedding = kf
            pipelineLog("✅ keyframes_abs_pos_embedding 补灌 \(kf.shape)")
        } else {
            pipelineLog("⚠️ 权重无 keyframes_abs_pos_embedding，保持 zeros")
        }
        stageEnter(phase: .sampling, detail: "✅ DiT 权重灌入完成（\(Int(Date().timeIntervalSince(t3)))s，\(strippedW.count) 键）", protect: [.dit])
        // 权重加载保持 lazy：loadArrays 构造的是 lazy Load 图（0s 建图 ≠ 权重已驻留），真实读盘
        // 延迟到编译窗口内首次 forward 自然发生，编译窗口起点水位最低。已移除此前“提前 eval 权重”
        // 的强制驻留机制（~11.3G 提前压到编译窗口前会与全尺寸编译峰值叠加顶穿 48G，SIGKILL）。
        memProfilePoint("DiT权重灌入后")
        DiTModelCache.shared.store(d, path: ditPath)
        dit = d
    }
    memProfilePoint("DiT就绪")

    // 3) 噪声 latent + 位置
    // 生成关键帧 slot：官方 evenly spaced 内部像素帧（0=关闭传空）。slot 帧在 makeVideoLatentAndPos
    // 中追加到 latent 序列末尾（单像素帧时间精度），keyframesMLX 标记 slot 帧。
    let kfPixelFrames = computeKeyframePixelFrames(numFrames: g.numFrames, numKeyframes: g.numKeyframes)
    let (noiseV, videoPos) = makeVideoLatentAndPos(g: g, keyframePixelFrames: kfPixelFrames)
    pipelineLog("video latent \(noiseV.shape)（Nv=\(videoPos.count / 3)）\(kfPixelFrames.isEmpty ? "" : "，+\(kfPixelFrames.count) keyframe slot：\(kfPixelFrames)")")

    // 3.5) 音频条件：若连入音频节点，编码真实音频为 frozen 条件（audio_sigma=0.0 锁嘴形）；
    //      否则维持现状（随机噪声音频生成）。
    var frozenAudio: MLXArray? = nil
    var audioPos: [Float] = []
    if let audioPath {
        pipelineLog("🎵 检测到音频条件输入：\(audioPath)")
        let tA = Date()
        guard let pcm = loadAudioPCM16kStereo(path: audioPath) else {
            pipelineLog("❌ 音频读取失败：\(audioPath)")
            return
        }
        pipelineLog("  音频 PCM 帧数 \(pcm.count / 2)（16kHz 立体声）")
        // 加载 audio_vae 权重（仅用 encoder，与解码共用同一文件）
        let audioVaePath = CommonPaths.audioVae
        guard let aw = try? MLX.loadArrays(url: URL(fileURLWithPath: audioVaePath)) else {
            pipelineLog("❌ audio_vae 权重加载失败：\(audioVaePath)")
            return
        }
        sharedAudioVAEWeights = aw   // 同轮复用：mp4 音频解码阶段不再重复读盘（102MB）
        // maxTokens：按视频时长帧数换算音频 token 预算（对齐 makeAudioLatentAndPos 的 Na）
        let naBudget = Int((Float(g.numFrames) / g.frameRate * 25.0).rounded())
        frozenAudio = encodeAudioCond(weights: aw, pcm: pcm, maxTokens: naBudget)
        audioPos = makeAudioPositions(count: frozenAudio!.shape[1])
        stageEnter(phase: .encoding, detail: "  ✅ 音频条件编码完成（\(Int(Date().timeIntervalSince(tA)))s）：latent \(frozenAudio!.shape)，Na=\(audioPos.count)")
        pipelineLog("  ℹ️ 采用 frozen 音频条件（audio_sigma=0.0）：嘴形跟随真实音轨，视频流正常去噪")
    } else {
        let (noiseA, naPos) = makeAudioLatentAndPos(g: g)
        audioPos = naPos
        pipelineLog("audio latent \(noiseA.shape)（Na=\(audioPos.count)），无音频条件（随机噪声音频）")
    }

    // 3.6) 图片条件（I2V 首帧）：连入图片节点时，VAE 编码参考图为 latent，
    //      替换噪声首帧 token，构建 initVideo/cleanV/condMask（对齐官方 VideoConditionByLatentIndex）。
    //      支持最多 2 张；首张作首帧条件，第二张暂存（蒸馏单阶段无 half/full 两阶段，仅用首张）。
    var initVideo: MLXArray? = nil
    var cleanV: MLXArray? = nil
    var condMask: [Float]? = nil
    var refFirstCheck: MLXArray? = nil   // I2V 诊断：参考首帧 latent，采样后与输出首帧对比
    if !imagePaths.isEmpty {
        let tI = Date()
        let vaeEncPath = stage1Config.vaeEncoderPath
        guard let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: vaeEncPath)) else {
            pipelineLog("❌ vae_encoder 权重加载失败：\(vaeEncPath)")
            return
        }
        guard let img = loadImageBCFHW(path: imagePaths[0], width: g.width, height: g.height) else {
            pipelineLog("❌ 图片读取失败：\(imagePaths[0])")
            return
        }
        let refLat = vaeEncodeImage(weights: ew, pixelsBCFHW: img)   // [1,128,1,H,W] BCFHW
        // I2V 诊断：参考 latent 统计（判断编码是否有效；真实图片 latent 应明显偏离噪声分布）
        let rStats = refLat.asArray(Float.self)
        let rMean = rStats.reduce(0, +) / Float(rStats.count)
        let rVar = rStats.map { ($0 - rMean) * ($0 - rMean) }.reduce(0, +) / Float(rStats.count)
        let rStd = sqrt(rVar)
        let rMax = rStats.map { abs($0) }.max() ?? 0
        pipelineLog("  I2V ref latent 统计：mean=\(rMean) std=\(rStd) maxAbs=\(rMax) NaN=\(rStats.contains { !$0.isFinite })")
        // [1,128,1,H,W] → [1,1,H,W,128]（BFHWC），与 noiseV 5D 布局对齐
        let ref5D = refLat.transposed(0, 2, 3, 4, 1)
        let t = noiseV.shape[1], h = noiseV.shape[2], w = noiseV.shape[3]
        let hw = h * w
        guard ref5D.shape[2] == h, ref5D.shape[3] == w else {
            pipelineLog("❌ 参考图 latent \(ref5D.shape) 与视频 latent \(noiseV.shape) 空间尺寸不匹配")
            return
        }
        // 首尾帧 I2V：imagePaths 按画布 y 排序（y 小=首帧，y 大=尾帧）。
        // 首帧模式：初始 latent [ref首, 其余噪声]；clean [ref首, 其余 0]；mask 前 HW 个 0 其余 1。
        // 首尾帧模式：中间帧去噪，两端钉入 —— initVideo [ref首, noise中, ref尾]；
        // cleanV [ref首, zeros, ref尾]；condMask 前 HW 个 0 + 中间 1 + 后 HW 个 0。
        // 注意 keyframe slot：noiseV 末尾 K 帧是 slot（全程去噪生成），不参与 I2V 钉入；
        // 首尾帧模式中视频尾帧是 latent 帧 F-1（而非总帧数 t-1）。
        let Fv = t - kfPixelFrames.count   // 视频帧数（不含 slot）
        let refFirst = ref5D                                            // [1,1,H,W,128]
        refFirstCheck = ref5D
        if imagePaths.count >= 2 {
            guard let imgTail = loadImageBCFHW(path: imagePaths[1], width: g.width, height: g.height) else {
                pipelineLog("❌ 尾帧图片读取失败：\(imagePaths[1])")
                return
            }
            let refTailLat = vaeEncodeImage(weights: ew, pixelsBCFHW: imgTail)   // [1,128,1,H,W]
            let refTail = refTailLat.transposed(0, 2, 3, 4, 1)          // [1,1,H,W,128]
            guard refTail.shape[2] == h, refTail.shape[3] == w else {
                pipelineLog("❌ 尾帧 latent \(refTail.shape) 与视频 latent \(noiseV.shape) 空间尺寸不匹配")
                return
            }
            guard Fv >= 2 else {
                pipelineLog("❌ 视频帧数不足，无法首尾帧 I2V（Fv=\(Fv)）")
                return
            }
            let noiseMid = noiseV[0..., 1 ..< (Fv - 1), 0..., 0..., 0...]   // [1,F-2,H,W,128] 视频中间帧
            let slotNoise = kfPixelFrames.isEmpty ? nil : noiseV[0..., Fv..., 0..., 0..., 0...]  // [1,K,H,W,128]
            let zerosMid = MLXArray.zeros([1, Fv - 2, h, w, 128]).asType(noiseV.dtype)
            let zerosSlot = kfPixelFrames.isEmpty ? nil : MLXArray.zeros([1, kfPixelFrames.count, h, w, 128]).asType(noiseV.dtype)
            var initParts: [MLXArray] = [refFirst, noiseMid, refTail]
            var cleanParts: [MLXArray] = [refFirst, zerosMid, refTail]
            if let sn = slotNoise, let zs = zerosSlot {
                initParts.append(sn)
                cleanParts.append(zs)
            }
            initVideo = concatenated(initParts, axis: 1).asType(noiseV.dtype)
            cleanV = concatenated(cleanParts, axis: 1).asType(noiseV.dtype)
            condMask = [Float](repeating: 0, count: hw)
                + [Float](repeating: 1, count: (Fv - 2) * hw)
                + [Float](repeating: 0, count: hw)
                + [Float](repeating: 1, count: kfPixelFrames.count * hw)
            stageEnter(phase: .vae, detail: "🖼️ I2V 首尾帧条件：首帧 \(imagePaths[0])，尾帧 \(imagePaths[1]) → VAE 编码完成（\(Int(Date().timeIntervalSince(tI)))s），mask 前/后各 \(hw) 个 0 钉入两端（slot \(kfPixelFrames.count) 帧保持去噪）")
        } else {
            let noiseRest = noiseV[0..., 1..., 0..., 0..., 0...]            // [1,T-1,H,W,128]（视频中间 + slot 全去噪）
            let zerosRest = MLXArray.zeros([1, t - 1, h, w, 128]).asType(noiseV.dtype)
            initVideo = concatenated([refFirst, noiseRest], axis: 1).asType(noiseV.dtype)
            cleanV = concatenated([refFirst, zerosRest], axis: 1).asType(noiseV.dtype)
            condMask = [Float](repeating: 0, count: hw) + [Float](repeating: 1, count: t * hw - hw)
            stageEnter(phase: .vae, detail: "🖼️ I2V 首帧条件：\(imagePaths[0])（共 \(imagePaths.count) 张）→ VAE 编码完成（\(Int(Date().timeIntervalSince(tI)))s），首帧 latent \(refLat.shape)（slot \(kfPixelFrames.count) 帧保持去噪）")
        }
    }

    // 4) 采样：Stage1 固定蒸馏（官方固定 sigma 表 + ancestral SDE，eta=1.0）无引导：
    //    cfgV/cfgA=1.0，negV/negA=nil；frozen 音频条件：audio_sigma=0.0，
    //    音频流固定为真实 latent，视频受其条件化锁嘴形
    //    （dev 管线保留未调用：20 步 DynamicShift + CFG 3.0/7.0，走 sampleLatentsDev）
    // 条件嵌入完成 → 采样开始前：中途水位评估，卸载已用完的 Gemma/Connector（保护 DiT）
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .sampling, next: .vae)
    memProfilePoint("采样开始前")
    // 生成关键帧 slot 标记（追加在 latent 末尾的 K 帧；0=关闭传 nil）
    let kfLatentFrames = noiseV.shape[1], kfH = noiseV.shape[2], kfW = noiseV.shape[3]
    let keyframesMLX = makeKeyframesMLX(keyframePixelFrames: kfPixelFrames,
                                        latentFrames: kfLatentFrames,
                                        height: kfH, width: kfW)
    var (vFinal, aFinal) = runOnBigStack {
        sampleLatentsDistilled(
            dit: dit,
            noiseV: noiseV,
            noiseA: frozenAudio ?? MLXRandom.normal([1, max(audioPos.count, 1), 128]).asType(PrecisionPolicy.defaultMainDType),
            condV: condV, condA: condA,
            negV: negV, negA: negA,
            frozenAudio: frozenAudio,
            initVideo: initVideo, cleanV: cleanV, condMask: condMask,
            keyframesMLX: keyframesMLX,
            videoPos: videoPos, audioPos: audioPos,
            seed: g.seed,
            isCancelled: isCancelled)
    }
    if isCancelled() {
        pipelineLog("⏹ 视频生成已取消，跳过后续解码与落盘")
        return
    }
    memProfilePoint("采样结束后")
    // 裁剪 keyframe slot：Stage1 输出含末尾 K 个 slot 帧（仅生成期辅助锚定，官方 clear_conditioning
    // 语义：解码/升频只消费视频帧 token），裁掉后只留视频 F 帧。
    if !kfPixelFrames.isEmpty {
        let videoFrames = noiseV.shape[1] - kfPixelFrames.count
        vFinal = vFinal[0..., 0 ..< videoFrames, 0..., 0..., 0...]
        stageEnter(phase: .sampling, detail: "✂️ keyframe slot 已裁剪：\(vFinal.shape)（保留视频 \(videoFrames) 帧）")
    }
    // DiT 权重与编译图常驻缓存，不在此释放；仅回收采样过程产生的空闲 buffer
    MLX.Memory.clearCache()
    // 采样完成，VAE 解码前：加载前评估水位（VAE 解码+升频器为临时加载无持久缓存，恒按 1.7G 预估），保护 采样+VAE 阶段，预留按尺寸+时长动态
    guard MemoryPolicy.ensureCapacity(for: 1_700_000_000, task: .video, current: .sampling, next: .vae, width: g.width, height: g.height, frames: g.numFrames) else {
        pipelineLog("❌ 内存不足，无法加载该模型（VAE 解码+升频器 约 1.7G），已拦截 VAE 解码，本次视频生成中止")
        return
    }

    // 4.5) Stage-2 蒸馏 refine（对齐官方两阶段：空间升频 latent ×2 → 3 步确定性 refine）
    //      官方 STAGE_2_DISTILLED_SIGMAS=[0.909375, 0.725, 0.421875, 0.0]。
    //      纯 t2v 直接 refine；I2V 时全分辨率重编码首帧并钉入（对齐 stage1 的
    //      VideoConditionByLatentIndex 语义：mask 前 HW 个 0 钉首帧，其余 1 生成）。
    videoLatentIsFullRes = false
    if let s2 = stage2Config, s2.mode == .refine {
        let tS2 = Date()
        stageEnter(phase: .refine, detail: "=== 4.5) 升频 ×2 + 3 步蒸馏 refine（\(condMask == nil ? "纯 t2v" : "I2V 首帧")）===", protect: [.dit])
        let vaeDecPath = s2.vaeDecoderPath
        let upPath = s2.upscalerPath
        // 只轻量读 mean/std（2×128 标量，几十 KB）而非整读 vae_decoder（777MB）：
        // 完整权重由 ⑦ VAE 解码阶段加载，避免同轮两次读盘。
        if let ms = loadVAEMeanStd(path: vaeDecPath),
           let upW = try? MLX.loadArrays(url: URL(fileURLWithPath: upPath)) {
            let mean = ms.mean.asType(vFinal.dtype)
            let std = ms.std.asType(vFinal.dtype)
            // 升频：denorm → spatial ×2 → norm（latent [1,T,H,W,128] → [1,T,2H,2W,128]）
            let denorm = vFinal * std + mean
            let upLat = latentUpsamplerSpatial(weights: upW, latentNDHWC: denorm)
            let normLat = (upLat - mean) / std
            stageEnter(phase: .refine, detail: "✅ 空间升频完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(vFinal.shape) → \(normLat.shape)")
            // 全分辨率位置（makeVideoLatentAndPos 的 noise 丢弃，只取 pos；像素坐标 h*32+16 语义不变）
            let gFull = GenConfig(numFrames: g.numFrames, height: g.height * 2, width: g.width * 2,
                                  frameRate: g.frameRate, numSteps: g.numSteps, seed: g.seed)
            let (_, fullPos) = makeVideoLatentAndPos(g: gFull)
            // 加噪到 σ0=0.909375（x_t = (1-s)*x0 + s*noise），3 步确定性 refine（无 CFG）
            MLXRandom.seed(g.seed &+ 0x5EED_5EED)
            let fullNoise = MLXRandom.normal(normLat.shape).asType(normLat.dtype)
            let initLat = normLat * (1 - ltx25Stage2Sigmas[0]) + fullNoise * ltx25Stage2Sigmas[0]
            // I2V：全分辨率重编码首/尾帧 → 替换 initLat 首/尾帧 token + 构造 clean/mask
            //      对齐 ComfyUI 官方 LTXV 首尾帧语义（VideoConditionByLatentIndex + cond_mask）：
            //      stage2 与 stage1 一致，首/尾各钉 HW 个 0，中间帧生成。
            var refineInit: MLXArray = initLat
            var refineClean: MLXArray? = nil
            var refineMask: [Float]? = nil
            if condMask != nil, !imagePaths.isEmpty {
                let vaeEncPath = s2.vaeEncoderPath ?? Stage2Config.vaeEncoder
                if let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: vaeEncPath)) {
                    let fh = gFull.height / 32, fw = gFull.width / 32
                    let fhw = fh * fw
                    let tCount = refineInit.shape[1]
                    if let img = loadImageBCFHW(path: imagePaths[0], width: gFull.width, height: gFull.height) {
                        let refLat = vaeEncodeImage(weights: ew, pixelsBCFHW: img)   // [1,128,1,2H,2W] BCFHW
                        let ref5D = refLat.transposed(0, 2, 3, 4, 1)                  // [1,1,2H,2W,128]
                        if imagePaths.count >= 2, tCount >= 3,
                           let imgTail = loadImageBCFHW(path: imagePaths[1], width: gFull.width, height: gFull.height) {
                            let refTailLat = vaeEncodeImage(weights: ew, pixelsBCFHW: imgTail)
                            let refTail5D = refTailLat.transposed(0, 2, 3, 4, 1)
                            let noiseMid = refineInit[0..., 1..<(tCount - 1), 0..., 0..., 0...]  // [1,T-2,2H,2W,128]
                            let zerosMid = MLXArray.zeros([1, tCount - 2, fh, fw, 128]).asType(refineInit.dtype)
                            refineInit = concatenated([ref5D, noiseMid, refTail5D], axis: 1).asType(refineInit.dtype)
                            refineClean = concatenated([ref5D, zerosMid, refTail5D], axis: 1).asType(refineInit.dtype)
                            refineMask = [Float](repeating: 0, count: fhw)
                                + [Float](repeating: 1, count: (tCount - 2) * fhw)
                                + [Float](repeating: 0, count: fhw)
                            stageEnter(phase: .refine, detail: "🖼️ stage2 I2V 首尾帧：\(imagePaths[0]) / \(imagePaths[1])（\(gFull.width)×\(gFull.height)）→ 首/尾各钉 \(fhw) 个 0，中间 \(tCount - 2) 帧生成")
                        } else {
                            let noiseRest = refineInit[0..., 1..., 0..., 0..., 0...]      // [1,T-1,2H,2W,128]
                            let zerosRest = MLXArray.zeros([1, tCount - 1, fh, fw, 128]).asType(refineInit.dtype)
                            refineInit = concatenated([ref5D, noiseRest], axis: 1).asType(refineInit.dtype)
                            refineClean = concatenated([ref5D, zerosRest], axis: 1).asType(refineInit.dtype)
                            refineMask = [Float](repeating: 0, count: fhw)
                                + [Float](repeating: 1, count: tCount * fhw - fhw)
                            stageEnter(phase: .refine, detail: "🖼️ stage2 I2V：全分辨率首帧 \(imagePaths[0])（\(gFull.width)×\(gFull.height)）→ latent \(ref5D.shape)，mask 前 \(fhw) 个 0")
                        }
                    } else {
                        pipelineLog("⚠️ stage2 I2V 首帧编码失败，refine 不加首帧条件")
                    }
                } else {
                    pipelineLog("⚠️ stage2 I2V 权重加载失败，refine 不加首帧条件")
                }
            }
            // ★ 双驻留修复：Stage2 refine 即将编译全分辨率大图（Nv=30240），先清掉 Stage1 编译图（Nv≈3780），
            //   避免两张大图同时驻留（图结构+内核缓存也是真内存）。Stage1 的 vFinal 已 eval 物化，析构安全。
            //   ⚠️ 不得改为"保最近N张缓存"再编译：上轮全清大图驻留会让编译峰值叠加 → 第二轮 stage2 压缩30G死机（实测）。
            runOnBigStack {
                CompiledForwardCache.shared.clear()
                MLX.Memory.clearCache()
            }
            let (vRef, _) = runOnBigStack {
                sampleLatentsDistilled(
                    dit: dit, noiseV: refineInit, noiseA: aFinal,
                    condV: condV, condA: condA, negV: nil, negA: nil,
                    frozenAudio: frozenAudio,
                    cleanV: refineClean, condMask: refineMask,
                    keyframesMLX: nil,   // 官方 stage_2_conditionings 无 keyframes，仅图像条件
                    videoPos: fullPos, audioPos: audioPos,
                    numSteps: ltx25Stage2Sigmas.count - 1,
                    sigmas: ltx25Stage2Sigmas,
                    ancestral: false,
                    seed: g.seed,
                    isCancelled: isCancelled)
            }
            if isCancelled() { pipelineLog("⏹ stage2 refine 已取消"); return }
            vFinal = vRef
            videoLatentIsFullRes = true
            stageEnter(phase: .refine, detail: "✅ stage2 refine 完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(vFinal.shape)")
        } else {
            pipelineLog("⚠️ stage2 refine 权重加载失败，回退单阶段（保持半分辨率 latent，VAE 解码前升频）")
        }
    } else {
        // stage2Refine=false：尺寸段已直出目标分辨率（不除2），latent 即全分辨率，
        // 标记跳过 VAE 解码前升频（不升频/不二采），落盘即目标分辨率。
        videoLatentIsFullRes = true
        pipelineLog("ℹ️ stage2 refine 已关闭（stage2Refine=false）：直出目标分辨率，解码不升频")
    }

    // 5) 落盘
    let outDir = outputVideoDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let vUrl = URL(fileURLWithPath: "\(outDir)/video_latent_final.npy")
    let aUrl = URL(fileURLWithPath: "\(outDir)/audio_latent_final.npy")
    try! save(array: vFinal.asType(.float32), url: vUrl)
    try! save(array: aFinal.asType(.float32), url: aUrl)

    let vMean = vFinal.mean().item(Float.self)
    let vMax = abs(vFinal).max().item(Float.self)
    let aMean = aFinal.mean().item(Float.self)
    pipelineLog("video latent: \(vFinal.shape) mean=\(vMean) maxAbs=\(vMax) → \(vUrl.path)")
    pipelineLog("audio latent: \(aFinal.shape) mean=\(aMean) → \(aUrl.path)")
    // I2V 诊断：输出首帧 vs 参考首帧 latent 差异（≈0 表示钉入生效，大则参考 latent 未真正进入采样）
    // 注意：stage2 refine 后 vFinal 为全分辨率 latent，首帧来自全分辨率重编码（stage2 块内
    // 已打印"🖼️ stage2 I2V 首帧"日志），与 stage1 的半分辨率 refFirstCheck 形状不同无法比较，跳过。
    if let rf = refFirstCheck, !videoLatentIsFullRes {
        let vf0 = vFinal[0..., 0 ..< 1, 0..., 0..., 0...].asType(.float32)   // [1,1,H,W,128]
        let diffMean = abs(vf0 - rf).mean().item(Float.self)
        let diffMax = abs(vf0 - rf).max().item(Float.self)
        pipelineLog("🖼️ I2V 验证：输出首帧 vs 参考 latent 差异 mean=\(diffMean) max=\(diffMax)（≈0 表示首帧钉入成功）")
    }
    stageEnter(phase: .sampling, detail: "=== ⑥ 去噪 latent 生成完成（下一步：VAE 解码 → 视频帧）===")
}

/// 升频器前向 + 回 norm 到 latent 分布。upLatent 是函数局部变量，返回后释放，
/// 避免大张量引用滞留到解码阶段。
func upscaleLatentAndNorm(
    upWeights: [String: MLXArray],
    denormLatent: MLXArray,
    mean: MLXArray,
    std: MLXArray
) -> MLXArray {
    let upLatent = latentUpsamplerSpatial(weights: upWeights, latentNDHWC: denormLatent)
    return (upLatent - mean) / std
}

// 同轮 audio_vae 权重缓存：音频条件编码（生成开头）→ mp4 音频解码（生成末尾）共用一次读盘。
// 用完即置 nil 释放，不跨生成驻留。
var sharedAudioVAEWeights: [String: MLXArray]?

func vaeDecodeTest() -> String? {
    stageEnter(phase: .vae, detail: "\n=== ⑦ VAE 解码（latent → 像素帧）===")
    memProfilePoint("VAE权重加载前")
    // 加载解码器权重前：与 DiT 加载路径一致的内存保护（参考 runVideoPipeline 采样后 VAE 解码前调用）——
    // 先中途卸载已无用模块（采样后的 DiT/Gemma/Connector 等，VAE 阶段保护集为空全卸让位），
    // 再按水位 + 动态预留评估放行（解码器+升频器约 1.7G，尺寸/时长按保守 512×512@120 帧估算）
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .vae, next: nil)
    guard MemoryPolicy.ensureCapacity(for: 1_700_000_000, task: .video, current: .vae, width: 512, height: 512, frames: 120) else {
        pipelineLog("❌ 内存不足，无法加载解码器（VAE 解码+升频器 约 1.7G），已拦截 VAE 解码")
        return nil
    }
    // 扩散视频解码器开关（偏好设置 videoUseDiffusionDecoder）：true=扩散解码器替换卷积 VAE
    let useDiffDecoder = AppSettings.shared.videoUseDiffusionDecoder
    let vaePath = useDiffDecoder ? CommonPaths.vaeDiffusionDecoder : CommonPaths.vaeDecoder
    let latentPath = outputVideoDirURL.appendingPathComponent("video_latent_final.npy").path
    guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: vaePath)) else {
        pipelineLog("❌ \(useDiffDecoder ? "ltx-2.5-video-vae-bf16.safetensors" : "vae_decoder.safetensors") 加载失败"); return nil
    }
    memProfilePoint("VAE权重加载后")
    stageEnter(phase: .vae, detail: "✅ \(useDiffDecoder ? "扩散视频解码器" : "卷积 VAE")权重张量数：\(weights.count)")
    guard let latent = loadNpy(latentPath) else { return nil }
    if videoLatentIsFullRes {
        pipelineLog("latent: \(latent.shape)（NDHWC [1,T,H,W,128]，float32；已是目标分辨率，跳过升频直接解码）")
    } else {
        pipelineLog("latent: \(latent.shape)（NDHWC [1,T,H,W,128]，float32；生成尺寸=目标/2，升频×2 后解码还原目标分辨率）")
    }

    // 空间升频器：denorm → latent 空间 ×2 → norm → VAE 解码（输出分辨率 ×2）
    // stage2 refine 已完成时 latent 已是全分辨率，跳过升频直接解码
    // mean/std：官方扩散解码器权重（ltx-2.5-video-vae-bf16.safetensors）自带 latent 统计
    // （键名 per_channel_statistics.mean-of-means / std-of-means，2×128 标量）；旧 mlx 版
    // vae_diffusion_decoder 不含统计，曾回退读卷积 VAE 的 vae_decoder.* 键。
    let meanStd: (mean: MLXArray, std: MLXArray)
    if useDiffDecoder {
        guard let stats = readSafetensorsFloatArrays(path: CommonPaths.vaeDiffusionDecoder, keys: [
            "per_channel_statistics.mean-of-means",
            "per_channel_statistics.std-of-means"
        ]),
        let mean0 = stats["per_channel_statistics.mean-of-means"],
        let std0 = stats["per_channel_statistics.std-of-means"] else {
            pipelineLog("❌ 扩散解码器模式需从官方 VAE 读取 mean/std，读取失败"); return nil
        }
        meanStd = (
            mean0.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType),
            std0.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType)
        )
    } else {
        meanStd = (
            weights["vae_decoder.per_channel_statistics.mean"]!.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType),
            weights["vae_decoder.per_channel_statistics.std"]!.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType)
        )
    }
    let mean = meanStd.mean, std = meanStd.std
    let latentB = latent.asType(PrecisionPolicy.defaultMainDType)
    let normLatent: MLXArray
    if videoLatentIsFullRes {
        normLatent = latentB
        stageEnter(phase: .vae, detail: "✅ latent 已是全分辨率（stage2 refine 产物 / stage2 关闭直出），跳过升频直接 VAE 解码")
    } else {
        let tUp = Date()
        let upPath = CommonPaths.spatialUpscaler
        guard let upWeights = try? MLX.loadArrays(url: URL(fileURLWithPath: upPath)) else {
            pipelineLog("❌ spatial_upscaler_x2_v1_1.safetensors 加载失败"); return nil
        }
        stageEnter(phase: .vae, detail: "✅ 升频器权重张量数：\(upWeights.count)")
        memProfilePoint("升频前")
        let denorm = latentB * std + mean
        // 升频器前向 + 回 norm 放在独立函数里，upLatent 函数返回后即释放，
        // 避免大张量引用滞留到解码阶段
        let upLat = upscaleLatentAndNorm(upWeights: upWeights, denormLatent: denorm, mean: mean, std: std)
        eval(upLat)
        normLatent = upLat
        memProfilePoint("升频后")
        stageEnter(phase: .vae, detail: "✅ 空间升频完成（\(Int(Date().timeIntervalSince(tUp)))s）：\(latentB.shape) → \(normLatent.shape)（NDHWC，空间 ×2）")
    }

    let t0 = Date()
    if useDiffDecoder {
        pipelineLog("解码中（扩散视频解码器：det 上采样 + 像素空间扩散去噪，官方 1 步 x0，预计 5~15 分钟）...")
        memPointLog("解码器启动")
    } else {
        pipelineLog("解码中（VAE Tiling 分块，1024 通道 3D 卷积，预计几分钟）...")
    }
    // latent f32 → bf16：权重本身 bf16，全链路 bf16 省内存提速；统计等价性由回归验证
    // tiled 解码：默认 tile 对齐官方 TileSizeConfig.default()（时间 tile=10帧/overlap=3，空间 tile=24格/overlap=2），
    // 分块 + 梯形 mask 加权融合，压低 720p 5s 场景 VAE 解码峰值内存（整片曾达 51.45GB）
    // 扩散解码器模式：latent（与卷积 VAE 同一输入）→ det_stages 上采样构建 context volume →
    // 扩散去噪（NA 注意力 + context 注入，官方 1 步 x0 / 可配多步），tiling 融合
    let pixels: MLXArray
    if useDiffDecoder {
        pixels = ltx2VideoDiffusionDecode(weights: weights, latentNDHWC: normLatent)
    } else {
        pixels = vaeDecodeTiled(weights: weights, latentNDHWC: normLatent)
    }
    memPointLog("eval(pixels) 前")
    eval(pixels)
    memPointLog("eval(pixels) 后")
    memProfilePoint("VAE解码后")
    stageEnter(phase: .vae, detail: "✅ \(useDiffDecoder ? "扩散" : "卷积")解码完成（\(Int(Date().timeIntervalSince(t0)))s）：\(pixels.shape)，期望 [1,3,113,512,512]")

    // NaN 检查
    let parr = pixels.asArray(Float.self)
    let hasBad = parr.contains { !$0.isFinite }
    stageEnter(phase: .vae, detail: "NaN/Inf：\(hasBad ? "⚠️ 有" : "✅ 无")，mean=\(parr.reduce(0, +) / Float(parr.count))")

    // 帧导出直接流式写 mp4，不再落盘 pixels_final.npy（省 ~1.4GB SSD 写入）
    let outDir = outputVideoDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    // 管线第一次落盘即用计数名，与资产库一致（进资产库是第二次传递，同名冲突在 attach 阶段加 _ 处理）
    let baseName = nextAssetName(prefix: "ltx")

    // 帧导出 + mp4
    let fCount = pixels.shape[2]
    let h = pixels.shape[3], w = pixels.shape[4]
    stageEnter(phase: .exporting, detail: "导出 \(fCount) 帧 \(w)×\(h) → mp4（流式）...")
    let silentPath = "\(outDir)/\(baseName)_silent.mp4"
    let t1 = Date()
    do {
        try writeMp4(frameCount: fCount, width: w, height: h, fps: 24, to: silentPath) { f, base, bytesPerRow, isV210 in
            let frameArr = pixels[0 ..< 1, 0 ..< 3, f ..< (f + 1), 0 ..< h, 0 ..< w]
            let squeezed = frameArr.reshaped([3, h, w])
            // 原生 LTX 直出 h264（writeMp4 未开 proRes）→ isV210 恒 false，走原 8bit 32ARGB 填充
            return fillFrameForPixelBuffer(squeezed, width: w, height: h, base: base, bytesPerRow: bytesPerRow, isV210: isV210)
        }
    } catch {
        pipelineLog("⚠️ mp4 流式写入失败：\(error.localizedDescription)")
    }
    stageEnter(phase: .exporting, detail: "✅ 帧转换完成（\(Int(Date().timeIntervalSince(t1)))s）：\(fCount) 帧")
    let mp4Path = "\(outDir)/\(baseName).mp4"

    // 音频解码：audio latent → 16kHz 立体声 PCM → 混入 mp4
    var audioPCM: [Float]? = nil
    let audioVaePath = CommonPaths.audioVae
    let vocoderPath = CommonPaths.vocoder
    let audioLatentPath = "\(outDir)/audio_latent_final.npy"
    stageEnter(phase: .exporting, detail: "加载音频权重（audio_vae + vocoder）...")
    memProfilePoint("音频权重加载前")
    let t2 = Date()
    // 复用同轮音频条件编码已加载的 audio_vae（102MB，避免重复读盘）；用完即释放
    let aw = sharedAudioVAEWeights ?? (try? MLX.loadArrays(url: URL(fileURLWithPath: audioVaePath)))
    sharedAudioVAEWeights = nil
    if let aw,
       let vw = try? MLX.loadArrays(url: URL(fileURLWithPath: vocoderPath)) {
        var allW = aw
        for (k, v) in vw { allW[k] = v }
        stageEnter(phase: .exporting, detail: "✅ 音频权重张量数：audio_vae \(aw.count) + vocoder \(vw.count)，加载耗时 \(Int(Date().timeIntervalSince(t2)))s")
        memProfilePoint("音频权重加载后")
        // bf16 权重保持 bf16（MLX bf16 conv 在 M 系列 GPU 原生支持，内存省一半、速度更快）
        var f32W: [String: MLXArray] = allW
        if let aLatent = loadNpy(audioLatentPath) {
            pipelineLog("audio latent: \(aLatent.shape) → 解码 mel + vocoder（预计 1~3 分钟）...")
            let t3 = Date()
            let (pcm, frames) = decodeAudio(weights: f32W, latent: aLatent.asType(PrecisionPolicy.defaultMainDType))
            memProfilePoint("音频解码后")
            stageEnter(phase: .exporting, detail: "✅ 音频解码完成（\(Int(Date().timeIntervalSince(t3)))s）：\(frames) 帧 @16kHz（\(Float(frames) / 16000.0)s）")
            let pcmMean = pcm.reduce(0, +) / Float(max(pcm.count, 1))
            let pcmMax = pcm.map { abs($0) }.max() ?? 0
            pipelineLog("   波形 mean=\(pcmMean) maxAbs=\(pcmMax) NaN=\(pcm.contains { !$0.isFinite })")
            let wavPath = "\(outDir)/ltx_audio.wav"
            writeWav(pcm, sampleRate: 16000, channels: 2, to: wavPath)
            stageEnter(phase: .exporting, detail: "✅ wav 已保存：\(wavPath)")
            audioPCM = pcm
        } else {
            pipelineLog("⚠️ audio_latent_final.npy 读取失败，跳过音频")
        }
    } else {
        pipelineLog("⚠️ 音频权重加载失败，跳过音频")
    }

    do {
        // 归档上一个成品，避免连续生成时覆盖"觉得不错"的版本
        archiveIfExists(mp4Path)
        stageEnter(phase: .exporting, detail: "✅ 无声 mp4 已生成：\(silentPath)")
        if let audioPCM {
            try muxAudio(videoPath: silentPath, wavPath: "\(outDir)/ltx_audio.wav", to: mp4Path)
            stageEnter(phase: .exporting, detail: "✅ 音轨已混入（AAC）：\(mp4Path)")
        } else {
            try? FileManager.default.moveItem(atPath: silentPath, toPath: mp4Path)
            stageEnter(phase: .exporting, detail: "✅ mp4（无音轨）：\(mp4Path)")
        }
    } catch {
        pipelineLog("⚠️ mp4 合成失败：\(error.localizedDescription)")
    }
    stageEnter(phase: .vae, detail: "=== ⑦ VAE 解码完成 ===")
    return mp4Path
}

// MARK: - 通用升频 + 二采入口（跨模型像素桥，H3→LTX 复用）

/// ============================================================
///  场景：外部模型（如 H3）直出半清视频后，经本模块把 LTX 的
///        「空间升频×2 + Stage2 3 步蒸馏 refine」作为通用后处理调用，
///        不改动 LTX 原生 generateVideoTest / vaeDecodeTest 任何流程。
///  链路：外部半清像素 → LTX VAE encode（整段）→ latentUpsamplerSpatial×2
///        → sampleLatentsDistilled(ltx25Stage2Sigmas, 3 步) → LTX VAE decode。
///  注意：refine 是生成式重画（DiT 3 步再采样），不是无损放大；内容会变化。
/// ============================================================

/// 读取视频全部帧 → BCFHW float32（[-1,1]）[1,3,T,H,W]。
/// 输出宽高：target 传 0 时取源自然尺寸并向 32 倍数对齐（LTX latent 对齐要求）；
/// 非 32 倍数源经等比缩放+居中 pad 到最近 32 倍数（黑边），语义同 loadImageBCFHW。
func readVideoFramesToBCFHW(videoPath: String, targetWidth: Int = 0, targetHeight: Int = 0)
    -> (pixels: MLXArray, width: Int, height: Int, fps: Double, frameCount: Int)? {
    let url = URL(fileURLWithPath: videoPath)
    guard FileManager.default.fileExists(atPath: videoPath) else {
        pipelineLog("❌ [像素桥] 视频不存在：\(videoPath)"); return nil
    }
    let asset = AVURLAsset(url: url)
    guard asset.duration.isValid, asset.duration.seconds > 0,
          let track = asset.tracks(withMediaType: .video).first else {
        pipelineLog("❌ [像素桥] 无法读取视频轨道：\(videoPath)"); return nil
    }
    let fps = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : 24.0
    let totalFrames = max(1, Int((asset.duration.seconds * fps).rounded()))
    let outW = targetWidth > 0 ? targetWidth : alignTo32(Int(track.naturalSize.width))
    let outH = targetHeight > 0 ? targetHeight : alignTo32(Int(track.naturalSize.height))
    guard outW > 0, outH > 0 else { pipelineLog("❌ [像素桥] 视频尺寸异常"); return nil }

    guard let reader = try? AVAssetReader(asset: asset) else {
        pipelineLog("❌ [像素桥] AVAssetReader 创建失败"); return nil
    }
    // 注意：AVAssetReaderTrackOutput 的 outputSettings 不允许 kCVImageBufferYCbCrMatrixKey（会抛 NSInvalidArgumentException），
    // 只请求 BGRA，解码器按文件自带 color info（writeMp4 已写 BT.709）自动做 YCbCr→RGB，不手动指定矩阵。
    if let any = track.formatDescriptions.first {
        let desc = any as! CMFormatDescription
        let ext = CMFormatDescriptionGetExtensions(desc) as? [String: Any]
        if let mx = ext?[kCMFormatDescriptionExtension_YCbCrMatrix as String] as? String {
            pipelineLog("🎨 [像素桥] 源视频 YCbCr 矩阵标记: \(mx)")
        }
    }
    let settings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else {
        pipelineLog("❌ [像素桥] reader 启动失败：\(reader.error?.localizedDescription ?? "")"); return nil
    }

    var px = [Float](repeating: 0, count: 3 * totalFrames * outW * outH)
    var got = 0
    let plane = outW * outH
    while let sb = output.copyNextSampleBuffer(), got < totalFrames {
        autoreleasepool {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
            let pw = CVPixelBufferGetWidth(pb), ph = CVPixelBufferGetHeight(pb)
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            guard pw > 0, ph > 0, let base = CVPixelBufferGetBaseAddress(pb) else { return }
            let scale = min(Float(outW) / Float(pw), Float(outH) / Float(ph))
            let dstW = max(Int((Float(pw) * scale).rounded()), 1)
            let dstH = max(Int((Float(ph) * scale).rounded()), 1)
            let offX = (outW - dstW) / 2
            let offY = (outH - dstH) / 2
            // 与 loadImageBCFHW 相同路径：CGImage(BGRA little, DeviceRGB) → 绘制到 RGBA context
            let rowData = Data(bytes: base, count: bpr * ph)
            guard let provider = CGDataProvider(data: rowData as CFData),
                  let cg = CGImage(width: pw, height: ph, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: bpr, space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                                   provider: provider, decode: nil, shouldInterpolate: false,
                                   intent: .defaultIntent) else { return }
            guard let ctx = CGContext(data: nil, width: outW, height: outH, bitsPerComponent: 8,
                                      bytesPerRow: outW * 4, space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: outW, height: outH))
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: offX, y: offY, width: dstW, height: dstH))
            guard let data = ctx.data else { return }
            let ptr = data.bindMemory(to: UInt8.self, capacity: outW * outH * 4)
            // ctx.data 为 RGBA（byte0=R,1=G,2=B,3=A）。px 布局声明为 [B,C,F,H,W]
            // （C 维 stride = totalFrames*plane），必须按 ci*totalFrames*plane + got*plane 存放，
            // 不能按 got 外层连续排（否则 C/F 维度错位，通道会取到其它帧 → 历史“色度错乱”假象）。
            for y in 0..<outH {
                for x in 0..<outW {
                    let i = (y * outW + x) * 4
                    let o = y * outW + x
                    px[0 * totalFrames * plane + got * plane + o] = Float(ptr[i + 0]) / 127.5 - 1.0
                    px[1 * totalFrames * plane + got * plane + o] = Float(ptr[i + 1]) / 127.5 - 1.0
                    px[2 * totalFrames * plane + got * plane + o] = Float(ptr[i + 2]) / 127.5 - 1.0
                }
            }
            got += 1
        }
    }
    guard got > 0 else { pipelineLog("❌ [像素桥] 未读到任何视频帧"); return nil }
    pipelineLog("✅ [像素桥] 读取视频 \(got) 帧 @\(Int(fps))fps → \(outW)×\(outH)（源 \(Int(track.naturalSize.width))×\(Int(track.naturalSize.height))）")
    return (MLXArray(px, [1, 3, got, outH, outW]).asType(.float32), outW, outH, fps, got)
}

func alignTo32(_ v: Int) -> Int {
    max(32, ((v + 31) / 32) * 32)
}

/// Stage2 refine 通用执行体：对**外部喂入**的半分辨率 latent（NDHWC，norm 域，LTX VAE 编码产物）
/// 做 空间升频×2 → σ0 加噪 → 3 步确定性蒸馏 refine。
/// 算法与 generateVideoTest 内 4.5 段一致，仅 latent 来源改为外部（像素桥）；不改 LTX 原管线。
/// - Parameters:
///   - halfLatent: [1,F,h,w,128] 半分辨率 norm latent（LTX VAE 域）
///   - fullPixelWidth/Height: 全分辨率像素尺寸（=半清×2，供位置编码）
///   - noiseA / audioPos: 音频 latent 与位置（外部生成；无 frozen 音频时传随机噪声起步）
///   - audioCondLatent / audioCondPos: 源视频音轨编码的真实音频 latent 与位置（可选）。
///     IC 模式下非 nil → frozen_a 条件化（口型跟源音轨，官方 stage2 audio 全程带音频条件语义）；
///     非 IC 模式忽略（保持外部传入的占位噪声起步）。
///   - imagePaths: 可选首/尾帧（全分辨率 I2V 钉入，需与输入视频首/尾内容一致）
func runLTXStage2RefineOnLatent(
    halfLatent: MLXArray,
    dit: LTXVideoDiT,
    condV: MLXArray,
    condA: MLXArray,
    noiseA: MLXArray,
    audioPos: [Float],
    audioCondLatent: MLXArray? = nil,   // IC frozen_a 真实音频 latent（源音轨编码；nil=随机占位）
    audioCondPos: [Float]? = nil,       // 配套音频位置（长度须=audioCondLatent Na）
    imagePaths: [String] = [],
    fullPixelWidth: Int,
    fullPixelHeight: Int,
    pixelFrames: Int,
    frameRate: Float = 24,
    seed: UInt64 = 42,
    isCancelled: @escaping () -> Bool = { false },
    sigmas: [Float] = ltx25Stage2Sigmas,
    anchorEveryNFrames: Int = 0,
    softAnchorMask: Float = 0.9,
    guideWeight: Float = 0,   // 全片 latent guide 强度（0=关闭）。>0 时以整片 normLat 为 guide，自动禁用 imagePaths 首尾硬钉
    headTailGuideBlend: Float = 0.5,  // guide>0 且传 imagePaths 时：首/尾帧的 guide 目标在「源 normLat 帧」与「外部首尾帧 latent」间的渗透系数（0=只锁源=现状；1=首尾帧 guide 完全指向外部图）。软引导语义：guide 是每步回拉而非 mask 硬钉，首尾帧不会被迫等于参考图
    tailGuideMask: Float = 0,  // 尾帧 mask（对齐官方 keyframe-guide 软语义）：0=硬钉外部图（旧语义）；>0=软引导，越大模型自由度越高、贴近参考程度越低。仅 imagePaths≥2 分支生效，首帧保持硬钉
    icLoRAEnable: Bool? = nil, // IC-LoRA Pixel Spatial Upscaler x2 官方模式：nil=默认关闭（非 IC 路径，收敛旧版 32640 量级，不再按 icLoRAPath 是否存在于 auto 启用）；true=强制启用。LTX_IC_LORA=1/0 环境变量可再覆盖
    icLoRAPath: String = ICLoRADefaultPath  // 官方 IC-LoRA 权重路径（默认见 ICLoRA-(ltx专属).swift；LTX_IC_LORA_PATH 可覆盖）
) -> MLXArray? {
    let tS2 = Date()
    pipelineLog("=== [像素桥] Stage2 通用 refine：升频 ×2 + \(sigmas.count - 1) 步（σ0=\(sigmas[0])）锚定=\(anchorEveryNFrames)帧/softM=\(softAnchorMask)/guideW=\(guideWeight)/tailM=\(tailGuideMask)===")
    let s2c = Stage2Config.refine(g: GenConfig(numFrames: pixelFrames, height: fullPixelHeight,
                                               width: fullPixelWidth, frameRate: frameRate,
                                               numSteps: 8, seed: seed))
    let vaeDecPath = s2c.vaeDecoderPath
    let upPath = s2c.upscalerPath
    guard let ms = loadVAEMeanStd(path: vaeDecPath) else {
        pipelineLog("⚠️ [像素桥] refine 权重加载失败（mean/std 或升频器）")
        return nil
    }
    // upW/denorm/upLatent 声明为 var：升频（eval(normLat)）完成后本函数后续不再引用，
    // 编译窗口前由「显式回收块」置空释放（见 ensureLoose·force 之后），让位给 DiT 编译峰值。
    guard var upW = try? MLX.loadArrays(url: URL(fileURLWithPath: upPath)) else {
        pipelineLog("⚠️ [像素桥] refine 权重加载失败（mean/std 或升频器）")
        return nil
    }
    let mean = ms.mean.asType(halfLatent.dtype)
    let std = ms.std.asType(halfLatent.dtype)
    // ⏭️ 对照开关（仅排查用）：PIX_SKIP_UPSCALE=1 时连空间升频也跳过（半清 latent 直接解码），
    //    用于区分 RGB 色边来自 VAE 往返（encode→decode）还是升频器/refine 链路。
    if ProcessInfo.processInfo.environment["PIX_SKIP_UPSCALE"] == "1" {
        pipelineLog("⏭️ [像素桥] PIX_SKIP_UPSCALE=1：跳过升频与 refine，半清 latent 直接解码（对照实验）")
        return halfLatent
    }
    var denorm = halfLatent * std + mean
    var upLatent = latentUpsamplerSpatial(weights: upW, latentNDHWC: denorm)
    let normLat = ((upLatent - mean) / std).contiguous()
    eval(normLat)
    pipelineLog("✅ [像素桥] 空间升频完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(halfLatent.shape) → \(normLat.shape)")
    // ⏭️ 对照开关（仅排查用）：PIX_SKIP_REFINE=1 时只做 encode+升频+decode，跳过 refine，
    //    用于数值拆解 RGB 色边到底来自 refine 还是 encode/升频/decode 公共链。
    if ProcessInfo.processInfo.environment["PIX_SKIP_REFINE"] == "1" {
        pipelineLog("⏭️ [像素桥] PIX_SKIP_REFINE=1：跳过 refine，仅升频后直接解码（对照实验）")
        return normLat
    }

    let gFull = GenConfig(numFrames: pixelFrames, height: fullPixelHeight, width: fullPixelWidth,
                          frameRate: frameRate, numSteps: 8, seed: seed)
    let (_, fullPos) = makeVideoLatentAndPos(g: gFull)

    // 加噪到 σ0（x_t = (1-σ0)*x0 + σ0*noise），按传入 sigmas 走确定性 refine（无 CFG）。
    // σ 表由调用方传入：像素桥入口默认 pixRefineSigmas 保守档（σ0=0.5，「只精修画面」模式，
    // 防止高噪声档把源画面口型/动作重画丢）；未传时本函数默认官方 ltx25Stage2Sigmas。
    MLXRandom.seed(seed &+ 0x5EED_5EED)
    let fullNoise = MLXRandom.normal(normLat.shape).asType(normLat.dtype)
    var refineInit = normLat * (1 - sigmas[0]) + fullNoise * sigmas[0]

    // 可选 I2V 首/尾帧钉入（全分辨率重编码，语义同 4.5）
    // 池化 + 用完即放：ew 与 2 张全清图编码的中间 buffer 都在池内，pool 结束立即释放；
    // 原实现 ew 活到 refine 结束且与 step2 编码器叠载，双 1344×768 图编码是 OOM 峰值点。
    // IC-LoRA 模式 RoPE 坐标（非 IC = fullPos 单份；IC = 主/参考双份同坐标）
    var icVideoPos = fullPos
    // IC-LoRA 模式 keyframes gate：全 0 → 主干不对参考区注入 keyframes_abs_pos_embedding
    // （官方 IC 参考区坐标同主帧，无附加位置 embedding；该注入仅 I2V/官方 keyframes 路径使用）
    var icKeyframesZeros: MLXArray? = nil
    var refineClean: MLXArray? = nil
    var refineMask: [Float]? = nil
    // guideClean 目标帧（guideWeight>0 时传给采样核心的干净 guide）：
    // 默认整片 = 源 normLat（现状：锁源精修，模型看不到外部参考）。
    // guide>0 且调用方仍传入 imagePaths 时 → 「强度锁 + 首尾帧软引导」并存：
    // 首/尾帧的 guide 目标改为 源帧与外部参考 latent 的渗透混合（headTailGuideBlend），
    // 中段保持锁源 normLat。guide 每步只是把 vx 拉回目标轨迹，不替换任何帧、
    // 无 mask 硬钉，因此参考图是「软引导」——模型有自由落笔空间，不会被迫让首帧等于参考图
    // （规避 v1.1 全分辨率硬钉导致的 RGB 色边/接缝）。
    var guideCleanTarget: MLXArray? = nil
    let useImageAnchors = !imagePaths.isEmpty && guideWeight <= 0
    if !imagePaths.isEmpty, guideWeight > 0 {
        pipelineLog("ℹ️ [像素桥] guideWeight>0 + imagePaths：首/尾帧并入 guide 软引导（渗透 headTailGuideBlend=\(headTailGuideBlend)，中段仍锁源 normLat；非 mask 硬钉）")
        autoreleasepool {
            guard let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: s2c.vaeEncoderPath ?? Stage2Config.vaeEncoder)) else {
                pipelineLog("⚠️ [像素桥] stage2 guide 软引导：encoder 加载失败，退化为纯锁源（首尾帧不渗透参考图）")
                return
            }
            let fh = fullPixelHeight / 32, fw = fullPixelWidth / 32
            let tCount = refineInit.shape[1]
            guard tCount >= 2, let img = loadImageBCFHW(path: imagePaths[0], width: fullPixelWidth, height: fullPixelHeight) else {
                pipelineLog("⚠️ [像素桥] stage2 guide 软引导：首帧图读取失败，退化为纯锁源")
                return
            }
            let refHead5D = vaeEncodeImage(weights: ew, pixelsBCFHW: img).transposed(0, 2, 3, 4, 1).asType(normLat.dtype)
            let headMix = normLat[0 ..< 1, 0 ..< 1, 0 ..< fh, 0 ..< fw, 0 ..< 128] * (1 - headTailGuideBlend)
                + refHead5D * headTailGuideBlend
            if imagePaths.count >= 2, tCount >= 3,
               let imgTail = loadImageBCFHW(path: imagePaths[1], width: fullPixelWidth, height: fullPixelHeight) {
                let refTail5D = vaeEncodeImage(weights: ew, pixelsBCFHW: imgTail).transposed(0, 2, 3, 4, 1).asType(normLat.dtype)
                let tailMix = normLat[0 ..< 1, (tCount - 1) ..< tCount, 0 ..< fh, 0 ..< fw, 0 ..< 128] * (1 - headTailGuideBlend)
                    + refTail5D * headTailGuideBlend
                let midSrc = normLat[0 ..< 1, 1 ..< (tCount - 1), 0 ..< fh, 0 ..< fw, 0 ..< 128]
                guideCleanTarget = concatenated([headMix, midSrc, tailMix], axis: 1).asType(normLat.dtype)
                pipelineLog("🖼️ [像素桥] guide 软引导：首/尾帧目标 = 源×\((1 - headTailGuideBlend)) + 参考×\(headTailGuideBlend)（\(fullPixelWidth)×\(fullPixelHeight)），中段 \(tCount - 2) 帧锁源")
            } else {
                let restSrc = normLat[0 ..< 1, 1 ..< tCount, 0 ..< fh, 0 ..< fw, 0 ..< 128]
                guideCleanTarget = concatenated([headMix, restSrc], axis: 1).asType(normLat.dtype)
                pipelineLog("🖼️ [像素桥] guide 软引导：仅首帧目标 = 源×\((1 - headTailGuideBlend)) + 参考×\(headTailGuideBlend)，其余帧锁源")
            }
            eval(guideCleanTarget!)
        }
        MLX.Memory.clearCache()
    }
    if useImageAnchors {
        autoreleasepool {
            guard let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: s2c.vaeEncoderPath ?? Stage2Config.vaeEncoder)) else {
                pipelineLog("⚠️ [像素桥] stage2 I2V 权重加载失败，refine 不加首帧条件")
                return
            }
            let fh = fullPixelHeight / 32, fw = fullPixelWidth / 32
            let fhw = fh * fw
            let tCount = refineInit.shape[1]
            if let img = loadImageBCFHW(path: imagePaths[0], width: fullPixelWidth, height: fullPixelHeight) {
                let refLat = vaeEncodeImage(weights: ew, pixelsBCFHW: img)
                let ref5D = refLat.transposed(0, 2, 3, 4, 1)
                if imagePaths.count >= 2, tCount >= 3,
                   let imgTail = loadImageBCFHW(path: imagePaths[1], width: fullPixelWidth, height: fullPixelHeight) {
                    let refTailLat = vaeEncodeImage(weights: ew, pixelsBCFHW: imgTail)
                    let refTail5D = refTailLat.transposed(0, 2, 3, 4, 1)
                    let noiseMid = refineInit[0 ..< 1, 1 ..< (tCount - 1), 0 ..< fh, 0 ..< fw, 0 ..< 128]
                    // 中间帧可选轻锚定（仅像素桥外部视频启用，anchorEveryNFrames>0）：
                    // 锚点帧 clean=升频后的干净源 normLat 对应帧、mask=softAnchorMask（每步 x0=x0*mask+clean*(1-mask)）。
                    // 【只精修画面模式】像素桥入口默认每 latent 帧软锚源画面 mask=0.8：
                    // 60% 步走极低频(1~0.55)+50% 软锚混合，保源结构不漂；后段 2 步收敛精修纹理。
                    // 经验：mask 过高(0.95)接近硬钉=只做轻微清理、缺少重采样，对升频糊边增益小；
                    //      mask 过低(0.5 以下)会让蒸馏采样不收敛，输出糊+重影。anchorEveryNFrames=0 关闭（原生路径默认）。
                    var midCleanFrames: [MLXArray] = []
                    var midMaskParts: [Float] = []
                    if anchorEveryNFrames > 0 {
                        for i in 1 ..< (tCount - 1) {
                            if (i - 1) % anchorEveryNFrames == 0 {
                                let src = normLat[0 ..< 1, i ..< (i + 1), 0 ..< fh, 0 ..< fw, 0 ..< 128]
                                    .asType(refineInit.dtype)
                                midCleanFrames.append(src)
                                midMaskParts.append(contentsOf: [Float](repeating: softAnchorMask, count: fhw))
                            } else {
                                midCleanFrames.append(MLXArray.zeros([1, 1, fh, fw, 128]).asType(refineInit.dtype))
                                midMaskParts.append(contentsOf: [Float](repeating: 1, count: fhw))
                            }
                        }
                    } else {
                        midCleanFrames.append(MLXArray.zeros([1, tCount - 2, fh, fw, 128]).asType(refineInit.dtype))
                        midMaskParts.append(contentsOf: [Float](repeating: 1, count: (tCount - 2) * fhw))
                    }
                    let midClean = concatenated(midCleanFrames, axis: 1).asType(refineInit.dtype)
                    refineInit = concatenated([ref5D, noiseMid, refTail5D], axis: 1).asType(refineInit.dtype)
                    refineClean = concatenated([ref5D, midClean, refTail5D], axis: 1).asType(refineInit.dtype)
                    // 尾帧对齐官方 keyframe-guide 软语义（helpers.py 尾帧走 VideoConditionByKeyframeIndex 追加 token，
                    // 非 latent 替换）：尾帧 mask 从 0（100% 硬钉外部图）改为 tailGuideMask 软引导——
                    // x0=x0*mask+clean*(1-mask)，tailM>0 时每步保留模型预测自由度，仅按 (1-tailM) 渗透参考，
                    // 消除"末段 ~8 像素帧锐度/色彩被高清参考图硬接管"的接缝；首帧保持官方 VideoConditionByLatentIndex
                    // 硬钉语义（mask=0）。tailM=0 时与旧版行为完全一致。
                    refineMask = [Float](repeating: 0, count: fhw) + midMaskParts
                        + [Float](repeating: tailGuideMask, count: fhw)
                } else {
                    let noiseRest = refineInit[0 ..< 1, 1 ..< tCount, 0 ..< fh, 0 ..< fw, 0 ..< 128]
                    let zerosRest = MLXArray.zeros([1, tCount - 1, fh, fw, 128]).asType(refineInit.dtype)
                    refineInit = concatenated([ref5D, noiseRest], axis: 1).asType(refineInit.dtype)
                    refineClean = concatenated([ref5D, zerosRest], axis: 1).asType(refineInit.dtype)
                    refineMask = [Float](repeating: 0, count: fhw)
                        + [Float](repeating: 1, count: tCount * fhw - fhw)
                }
                pipelineLog("🖼️ [像素桥] stage2 I2V 钉入：\(imagePaths.count) 张参考图（\(fullPixelWidth)×\(fullPixelHeight)），首帧 mask=0 硬钉 / 尾帧 mask=\(tailGuideMask)\(tailGuideMask > 0 ? "（keyframe-guide 软引导）" : "（硬钉·旧语义）")")
            } else {
                pipelineLog("⚠️ [像素桥] 参考帧编码失败，refine 不加图像条件")
            }
        }
        MLX.Memory.clearCache()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IC-LoRA 官方模式（Pixel Spatial Upscaler x2，H3 二采专用）
    // 严格按官方模型卡协议（reference_video_cond.py + model_card 两阶段采样），
    // 不再引入自创的 grid-dilate 零填充：
    //   ① 参考序列 = 半清视频整段 VAE latent（norm 域，H/2×W/2）直接 token 化
    //      追加：官方 VideoConditionByReferenceLatent 把参考 latent patchify 成
    //      F·h·w 稀疏真实 token 拼到主序列尾（无 0 填充占位），主干以 3D token 级
    //      序列处理（[1, NvMain+NvRef, C]）；main 语义同 5D 网格仅是 reshape 形态差。
    //   ② 主区加噪 x_t = normLat·(1-σ0) + noise·σ0（σ0=0.909375）；参考区噪声槽
    //      填 0（官方 latent_noise 参考区不灌噪声，per-token σ=0 恒为 clean）。
    //   ③ 参考区 per-token timestep=0（condMask 尾部 0，denoise_mask=0：参考区
    //      只提供 KV、自身 x0 恒写回 clean 参考，不参与重建）；
    //   ④ 参考区 RoPE 坐标 = 半清格像素中点 × downscale_factor(2) 映射回全清网格
    //      （官方 positions h/w × scale_factors），t 轴与主帧同一视频时间；
    //   ⑤ LoRA 全程开启（scale=1.0 预缩放、48 层全键注入），主序列从 σ0=0.909375
    //      按官方 STAGE_2_DISTILLED_SIGMAS 4 步档蒸馏采样（保留升频锚点，非 1.0 全噪声）；
    //      音频沿用像素桥占位/原声（不接 ref）；
    //   ⑥ 采样结束输出仅取主序列（前 NvMain token），参考区不落盘。
    // 外部 imagePaths / guide / anchor 与 IC 官方用法互斥：IC 启用时全部忽略
    // （官方 IC 参考即视频本身，无“外部参考图”概念；不引入 guide 混合）。
    var icModeEnabled = false
    // IC 使能后锁定官方 STAGE_2_DISTILLED_SIGMAS（4 步，σ0=0.909375）+ ancestral SDE；
    // 非 IC 保留调用方 σ 表。σ0=1.0 的 9 步 t2v 蒸馏档会丢掉升频锚点 → 网格/马赛克。
    var icEffSigmas: [Float] = sigmas
    let icModePath = ProcessInfo.processInfo.environment["LTX_IC_LORA_PATH"] ?? icLoRAPath
    // IC-LoRA 默认关闭（收敛回旧版 32640 量级非 IC 路径）：仅当 LTX_IC_LORA=1 或显式传入
    // icLoRAEnable=true 才启用。不再因磁盘存在 icLoRAPath 权重而 auto 启用——此前 auto 曾把
    // Nv 从 32640 扩到 40800（KV 参考直放 8160 + frozen_a 音频条件），拉高编译窗口峰值顶穿 48G。
    // LTX_IC_LORA=0 保持强制关语义；KV 参考 / 音频条件等 IC 专属机制随 icModeEnabled=false 一并停用。
    let icModeFlag = ProcessInfo.processInfo.environment["LTX_IC_LORA"]  // "1"=强制开 "0"=强制关 其它=默认关
    if icModeFlag == "0" {
        pipelineLog("ℹ️ [像素桥] IC-LoRA 已被 LTX_IC_LORA=0 关闭，走原 refine 路径")
    } else if icModeFlag == "1" || icLoRAEnable == true {
        guard let icBlocks = ICLoRACache.blocks(for: icModePath) else {
            pipelineLog("⚠️ [像素桥] IC-LoRA 权重加载失败（\(icModePath)），中止 refine")
            return nil
        }
        icModeEnabled = true
        // ★ IC 官方协议锁档：像素空间超分 x2 走 STAGE_2_DISTILLED_SIGMAS 4 步档
        //   （ltx25Stage2Sigmas，σ0=0.909375 保留升频锚点）+ ancestral SDE（官方 euler_ancestral，
        //   CFG=1）。9 步 t2v 蒸馏档 σ0=1.0 把主序列打成纯噪声、参考 KV 又无真实锚点可依
        //   → 整片彩色网格/马赛克；调用方 σ 表（PIX_SIGMAS/pixRefineSigmas）在 IC 模式不再生效。
        icEffSigmas = ltx25Stage2Sigmas
        // 外部尾帧 append 开关：默认关闭——外部尾图 ≠ H3 真实末帧时，append 的强可见性会拽末帧
        // 造成末尾跳变；末帧交由 IC KV 参考（半清整段 latent 含真实末帧）+ 源结构收敛。
        // LTX_IC_TAIL=1 可恢复旧 append 语义作对照。
        let icTailOn = ProcessInfo.processInfo.environment["LTX_IC_TAIL"] == "1"
        if guideWeight > 0 {
            pipelineLog("ℹ️ [像素桥] IC 模式不采用 guide 锁源软引导（官方 stage2 无 guideClean），改用下方 keyframe 首帧锚 + KV")
        }
        if !imagePaths.isEmpty {
            pipelineLog("🖼️ [像素桥] IC 模式启用外部首帧满锁 + KV 参考；外部尾帧 append \(icTailOn ? "开启（LTX_IC_TAIL=1）" : "默认关闭（外部尾图≠真实末帧会导致末尾跳变；LTX_IC_TAIL=1 可恢复）")")
        }
        // ① 外部首帧 keyframe 引导（官方 combined_image_conditionings / stage2 images 语义，
        //    IC 模式不再忽略 imagePaths）：
        //    首帧 = VideoConditionByLatentIndex（latent_cond.py）：替换主序列首帧 clean、mask=0 满锁，
        //    每步 post_process 强制写回参考图。外部尾帧仅在 LTX_IC_TAIL=1 时按
        //    VideoConditionByKeyframeIndex（keyframe_cond.py）append 恒 clean token（mask=0，只供注意力）。
        let tCount = normLat.shape[1]
        let fh = fullPixelHeight / 32, fw = fullPixelWidth / 32
        let cCh = normLat.shape[4]
        guard tCount == halfLatent.shape[1], halfLatent.shape[2] == fh / 2, halfLatent.shape[3] == fw / 2,
              normLat.shape[2] == fh, normLat.shape[3] == fw else {
            pipelineLog("❌ [像素桥] IC 网格不匹配：主 \(normLat.shape) vs 目标 \(tCount)×\(fh)×\(fw) / 半清 \(halfLatent.shape)")
            return nil
        }
        let nvMain = tCount * fh * fw
        let fhw = fh * fw
        let nvRef = halfLatent.shape[1] * halfLatent.shape[2] * halfLatent.shape[3]
        // IC KV 参考（半清整段 latent append，官方 stage1 IC 语义）默认保留；LTX_IC_KV=0 可关
        let icKVOn = ProcessInfo.processInfo.environment["LTX_IC_KV"] != "0"
        var headTokens: MLXArray? = nil
        var tailTokens: MLXArray? = nil
        if !imagePaths.isEmpty, tCount >= 2 {
            autoreleasepool {
                guard let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: s2c.vaeEncoderPath ?? Stage2Config.vaeEncoder)) else {
                    pipelineLog("⚠️ [像素桥] IC keyframe 引导：encoder 加载失败，跳过外部首尾帧（仅 KV）")
                    return
                }
                if let img = loadImageBCFHW(path: imagePaths[0], width: fullPixelWidth, height: fullPixelHeight) {
                    let head5D = vaeEncodeImage(weights: ew, pixelsBCFHW: img)
                        .transposed(0, 2, 3, 4, 1).asType(refineInit.dtype)   // [1,1,fh,fw,C] norm 域
                    eval(head5D)
                    headTokens = head5D.reshaped([1, fhw, cCh])
                }
                if icTailOn, imagePaths.count >= 2,
                   let imgTail = loadImageBCFHW(path: imagePaths[1], width: fullPixelWidth, height: fullPixelHeight) {
                    let tail5D = vaeEncodeImage(weights: ew, pixelsBCFHW: imgTail)
                        .transposed(0, 2, 3, 4, 1).asType(refineInit.dtype)
                    eval(tail5D)
                    tailTokens = tail5D.reshaped([1, fhw, cCh])
                }
            }
            MLX.Memory.clearCache()
        }
        // ② 主区 σ0 加噪（token 化）；首帧区（如有）替换为外部参考——官方 noiser 对 mask=0 区
        //    lerp 后恒为 clean，首帧实际以参考图为采样起点；参考区/尾帧 append 区噪声槽填 0
        //    （官方 latent_noise 参考区不灌噪声，per-token σ=0 恒为 clean）。
        let fullNoised3 = (normLat.reshaped([1, nvMain, cCh]) * (1 - icEffSigmas[0])
                           + fullNoise.reshaped([1, nvMain, cCh]) * icEffSigmas[0]).asType(refineInit.dtype)
        var mainInit3 = fullNoised3
        var mainClean3 = MLXArray.zeros([1, nvMain, cCh]).asType(refineInit.dtype)
        var mainMask = [Float](repeating: 1, count: nvMain)
        if let ht = headTokens {
            let rest = fullNoised3[0 ..< 1, fhw ..< nvMain, 0 ..< cCh]
            mainInit3 = concatenated([ht, rest], axis: 1).asType(refineInit.dtype)
            mainClean3 = concatenated([ht, MLXArray.zeros([1, nvMain - fhw, cCh]).asType(refineInit.dtype)],
                                      axis: 1).asType(refineInit.dtype)
            mainMask = [Float](repeating: 0, count: fhw)
                + [Float](repeating: 1, count: nvMain - fhw)
        }
        var extraNoise: [MLXArray] = []
        var extraClean: [MLXArray] = []
        var extraMask: [Float] = []
        if icKVOn {
            let refTokens = halfLatent.reshaped([1, nvRef, cCh]).asType(refineInit.dtype)
            extraClean.append(refTokens)
            extraNoise.append(MLXArray.zeros([1, nvRef, cCh]).asType(refineInit.dtype))
            extraMask.append(contentsOf: [Float](repeating: 0, count: nvRef))
        } else {
            pipelineLog("ℹ️ [像素桥] LTX_IC_KV=0：IC 参考区关闭（仅主序列 + 首尾帧锚）")
        }
        if let tt = tailTokens {
            extraClean.append(tt)
            extraNoise.append(MLXArray.zeros([1, fhw, cCh]).asType(refineInit.dtype))
            extraMask.append(contentsOf: [Float](repeating: 0, count: fhw))
        }
        refineInit = concatenated([mainInit3] + extraNoise, axis: 1).asType(refineInit.dtype)
        // ③ clean/mask：主区 mask=1（自由生成，clean 值不参与；首帧锚定区除外 mask=0 满锁）；
        //    参考区 / 尾帧 append 区 mask=0 恒写回真实参考（clean token）
        refineClean = concatenated([mainClean3] + extraClean, axis: 1).asType(refineInit.dtype)
        refineMask = mainMask + extraMask
        // ④ RoPE 坐标：主区 fullPos（全清格像素中点）；参考区 = 半清格像素中点 h/w ×2
        //    （官方 VideoConditionByReferenceLatent positions × downscale_factor=2，映射回
        //    全清网格），t 轴与主帧同一视频时间（同 makeVideoLatentAndPos 的 fmid 公式）。
        var extraPos: [Float] = []
        var extraPosCount = 0
        if icKVOn {
            extraPos.reserveCapacity(nvRef * 3)
            for f in 0..<tCount {
                let fs = max(Float(f) * 8.0 + 1.0 - 8.0, 0.0)
                let fe = max(Float(f + 1) * 8.0 + 1.0 - 8.0, 0.0)
                let fmid = (fs + fe) / 2.0 / frameRate
                for h in 0..<halfLatent.shape[2] {
                    let hmid = (Float(h) * 32.0 + 16.0) * 2.0
                    for w in 0..<halfLatent.shape[3] {
                        let wmid = (Float(w) * 32.0 + 16.0) * 2.0
                        extraPos.append(fmid)
                        extraPos.append(hmid)
                        extraPos.append(wmid)
                    }
                }
            }
            extraPosCount += nvRef
        }
        // 尾帧 append token：全清格坐标（hmid/wmid 同主区，不 ×2）+ t=末 latent 帧 fmid
        // （官方 VideoConditionByKeyframeIndex：positions t 轴 offset 到 frame_idx 附近）
        if tailTokens != nil {
            extraPos.reserveCapacity(extraPos.capacity + fhw * 3)
            let f = tCount - 1
            let fs = max(Float(f) * 8.0 + 1.0 - 8.0, 0.0)
            let fe = max(Float(f + 1) * 8.0 + 1.0 - 8.0, 0.0)
            let fmid = (fs + fe) / 2.0 / frameRate
            for h in 0..<fh {
                let hmid = Float(h) * 32.0 + 16.0
                for w in 0..<fw {
                    let wmid = Float(w) * 32.0 + 16.0
                    extraPos.append(fmid)
                    extraPos.append(hmid)
                    extraPos.append(wmid)
                }
            }
            extraPosCount += fhw
        }
        icVideoPos = fullPos + extraPos
        // keyframes gate 全 0（禁用主干对参考区 / append keyframe 区的 keyframes_abs_pos_embedding 注入；
        //    官方 VideoConditionByKeyframeIndex marked=False 同此语义）
        icKeyframesZeros = MLXArray.zeros([1, nvMain + extraPosCount, 1]).asType(.float32)
        pipelineLog("🧩 [像素桥] IC-LoRA 官方模式：KV 参考 \(icKVOn ? "\(halfLatent.shape) 直放 \(nvRef) tokens ×2" : "关闭")"
            + "，首帧锁帧 \(headTokens != nil ? "\(fhw) tokens" : "无")，尾帧 clean \(tailTokens != nil ? "\(fhw) tokens" : "无")"
            + "，Nv=\(nvMain)+\(extraPosCount)=\(refineInit.shape)，σ0=\(icEffSigmas[0]) \(icEffSigmas.count - 1) 步")
        // ⑤ 全程开 IC-LoRA 旁路（采样结束函数返回前自动关闭，原生路径不受污染）
        dit.attachICLoRA(icBlocks)
        dit.setICActive(true)
        defer { dit.setICActive(false) }
    }

    // ★ 双驻留瘦身（v2）：refine 前仅淘汰与本次采样形状不同的编译图。若上轮同形状 refine 图仍
    //   在缓存（同配置二次生成），采样内 shapeKey 命中直接复用 → 免重编译；原无条件 clear() 会把
    //   上轮同形状图一并清掉，编译窗口（峰值内存）被迫全量重编译 → low-swap jetsam 击杀（实测
    //   压缩内存 44GB）。前缀取采样实参形状（IC frozen 音频条件 Na 与占位不同会走 audioCondLatent）。
    //   ⚠️ 不得放宽为"保最近N张"再编译：上轮全清大图驻留会让第二轮 stage2 编译峰值叠加 → 压缩30G死机。
    let refineNoiseAShape = (icModeEnabled ? audioCondLatent : nil) ?? noiseA
    runOnBigStack {
        CompiledForwardCache.shared.evictExcept(keyPrefix: "\(refineInit.shape)-\(refineNoiseAShape.shape)-")
        MLX.Memory.clearCache()
    }
    // ★ 编译窗口水位兜底：IC 模式 KV 参考直放后序列暴涨（1080p：32640 主 + 8160×2 参考 + 音频条件），
    //   DiT 编译图与采样执行都是峰值窗口。此处 ensureLoose(protect:[.dit]) 压力分达标时卸载非保护模块
    //   （connector/gemma/HiDream 等本段已无用的权重），让编译与首步采样在较低水位起步。不卸载 .dit
    //   （本段正在使用），不关 IC、不关 KV 参考、不改任何采样语义。
    MemoryPolicy.ensureLoose(protect: [.dit], force: true)
    // [诊断打点] 编译窗口前水位（只读日志，不改任何内存/采样语义）：ensureLoose·force 刚完成清场。
    // 整机口径（activeBytes=物理内存−free−inactive，非进程驻留）与进程 MLX active/peak/cache 分列，
    // 与 ltx采样 内 compile-窗口前/图构建完成/预热eval完成 三点同渠道对照，区分未进编译/编译期爆。
    pipelineLog("🛠 [像素桥] Stage2 编译窗口前：整机 active=\(MemoryPolicy.activeBytes) 字节（\(String(format: "%.1fG", Double(MemoryPolicy.activeBytes) / Double(1 << 30)))），压力分=\(String(format: "%.1f", MemoryPolicy.pressureScore))")
    memPointLog("像素桥 编译窗口前(ensureLoose·force后)")
    // ★ 编译窗口前显式回收（模拟昨天真 Gemma 路径"用完即卸 connector+gemma 约12GB + 强制内存整理"
    //   给编译窗口腾余量的时机；空文本路径无模型可卸，改收 stage1 升频中间产物）：
    //   eval(normLat) 完成后，升频器权重 upW 与中间 latent denorm/upLatent 在本函数后续（加噪/编译/
    //   采样/解码）零引用，提前置空 + clearCache 归还 MLX 空闲缓存，降低进入 DiT 编译/预热峰值前
    //   的进程内驻留。不触碰 .dit / normLat / halfLatent / refineInit / KV 参考等二采仍需要的对象，
    //   不关 IC、不关 KV 参考、不改任何采样语义。
    runOnBigStack {
        autoreleasepool {
            upW = [:]
            denorm = MLXArray.zeros([0]).asType(denorm.dtype)
            upLatent = MLXArray.zeros([0]).asType(upLatent.dtype)
        }
        MLX.Memory.clearCache()
    }
    pipelineLog("🛠 [像素桥] 编译窗口前显式回收完成：置空升频器权重/升频中间 latent + clearCache，整机 active=\(MemoryPolicy.activeBytes) 字节（\(String(format: "%.1fG", Double(MemoryPolicy.activeBytes) / Double(1 << 30)))），压力分=\(String(format: "%.1f", MemoryPolicy.pressureScore))")
    memPointLog("像素桥 编译窗口前显式回收(upW/denorm/upLatent置空+clearCache)")
    // IC frozen_a：源音轨真实音频 latent 作为音频条件（口型跟 H3 音轨）。仅 IC 模式生效；
    // 非 IC 保持外部占位噪声起步（noiseA/audioPos 原样透传）。音频条件 Na 与占位不同时，
    // 编译图 shapeKey 含 noiseA.shape 自动分流；frozen 标志变更会重编译一次音频条件图。
    let effAudioLatent = icModeEnabled ? audioCondLatent : nil
    let effAudioPos: [Float] = (effAudioLatent != nil) ? (audioCondPos ?? audioPos) : audioPos
    if effAudioLatent != nil {
        pipelineLog("🎵 [像素桥] IC 模式音频条件化：frozen_a 真实源音轨 latent（Na=\(effAudioPos.count)，σ=0 锁口型）")
    }
    // ── SOL 稀疏注意力（二采 refine；默认开启，实机验证期）──
    // 开启后仅视频自注意力 attn1 走 H3 公共 SOL 内核（key 块 tau 路由 + 首帧 sink 恒精确），
    // 音频 attn1（headDim=64）与全部交叉/文本 attn 保持 dense；batch=1 / headDim=128 / 方阵等
    // 前置条件不满足时由 LTXSparseAttnBridge 运行期自动回退 dense，语义与关闭时一致。
    // 环境变量在此处（编译窗口外）读取一次并固化为常量：MLX.compile 图内禁止读 env / print / eval。
    //   PIX_SOL=0      关闭（默认 1=开启；出问题可置 0 回退 dense 基线）
    //   PIX_SOL_TAU    路由阈值倍数（默认 1.3，越小越精确越慢）
    //   PIX_SOL_SINK   首帧锚块数覆盖（默认按 1 帧 token 数自适应，覆盖完整首帧锚）
    //   PIX_SOL_TAIL=0 关闭 tail 补分母项（默认开）
    //   PIX_SOL_V6=0   退回 v3 内核（默认 v6）
    let s2Sparse: LTXSparseAttnConfig? = {
        let env = ProcessInfo.processInfo.environment
        guard env["PIX_SOL"] != "0" else { return nil }   // 默认开；仅显式 =0 关闭
        // 每帧 token 数：5D 网格直接取 H/W；3D token 级（IC 合并序列）主序列按全清格 H=像素/32
        let tpf = (refineInit.ndim >= 5 ? refineInit.shape[2] : max(1, fullPixelHeight / 32))
            * (refineInit.ndim >= 5 ? refineInit.shape[3] : max(1, fullPixelWidth / 32))
        let autoSink = (max(0, tpf) + 63) / 64        // 首帧占的 key 块数（SOL 块=64 行）
        let sink = Int(env["PIX_SOL_SINK"] ?? "") ?? autoSink
        return LTXSparseAttnConfig(
            tau: Float(env["PIX_SOL_TAU"] ?? "") ?? 1.3,
            tail: env["PIX_SOL_TAIL"] != "0",
            sinkBlocks: sink > 0 ? 0..<sink : nil,
            useV6: env["PIX_SOL_V6"] != "0")
    }()
    // 运行时探针日志已移除：[像素桥] SOL 稀疏注意力开启 一行不再输出（配置构造与传入逻辑不变）
    let (vRef, _) = runOnBigStack {
        sampleLatentsDistilled(
            dit: dit,
            noiseV: refineInit,
            noiseA: effAudioLatent ?? noiseA,
            condV: condV, condA: condA,
            negV: nil, negA: nil,
            frozenAudio: effAudioLatent,
            cleanV: refineClean, condMask: refineMask,
            // IC：全 0 keyframes gate（参考区不加 keyframes_abs_pos_embedding，仅借 condMask=0 走 σ0 时间步）
            keyframesMLX: icKeyframesZeros,
            // IC 官方模式无 guide（主序列全噪声由参考 KV 约束）；非 IC 保留原锁源/软引导语义
            guideClean: (!icModeEnabled && guideWeight > 0) ? (guideCleanTarget ?? normLat) : nil,
            guideWeight: icModeEnabled ? 0 : guideWeight,
            videoPos: icVideoPos, audioPos: effAudioPos,
            numSteps: icEffSigmas.count - 1,
            sigmas: icEffSigmas,
            // IC：官方 euler_ancestral SDE（STAGE_2 4 步档，eta=1.0）；非 IC 保持原确定性 Euler 精修
            ancestral: icModeEnabled,
            seed: seed,
            sparseVideo: s2Sparse,
            isCancelled: isCancelled)
    }
    if isCancelled() {
        pipelineLog("⏹ [像素桥] refine 已取消")
        return nil
    }
    pipelineLog("✅ [像素桥] Stage2 refine 完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(vRef.shape)")
    if icModeEnabled {
        // ⑥ 仅保留主序列 token（前 NvMain）；参考区只服务注意力，不参与解码落盘。
        // vRef 为 3D token 级 [1, NvMain+NvRef, 128]，恢复 5D 网格再交解码。
        let tC = normLat.shape[1]
        let fhC = normLat.shape[2], fwC = normLat.shape[3], cC = normLat.shape[4]
        let nvMainC = tC * fhC * fwC
        let vMain = vRef[0 ..< 1, 0 ..< nvMainC, 0 ..< cC]
            .reshaped([1, tC, fhC, fwC, cC])
        pipelineLog("✂️ [像素桥] IC 参考区已裁剪，输出主序列：\(vMain.shape)")
        return vMain
    }
    return vRef
}

/// 全分辨率 latent → 无声 mp4（流式写盘）。供外部像素桥复用（音轨由调用方/后续混流决定）。
func decodeLatentToSilentMP4(latentNDHWC: MLXArray, fps: Double, assetPrefix: String) -> String? {
    let t0 = Date()
    pipelineLog("=== [像素桥] VAE 解码（latent → ProRes 422 .mov）===")
    let vaePath = CommonPaths.vaeDecoder
    guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: vaePath)) else {
        pipelineLog("❌ [像素桥] vae_decoder 加载失败：\(vaePath)"); return nil
    }
    let latentB = latentNDHWC.asType(PrecisionPolicy.defaultMainDType)
    pipelineLog("latent: \(latentB.shape)（NDHWC，全分辨率）")
    let pixels = vaeDecodeTiled(weights: weights, latentNDHWC: latentB)
    eval(pixels)
    let parr = pixels.asArray(Float.self)
    let hasBad = parr.contains { !$0.isFinite }
    let pMean = parr.reduce(0, +) / Float(max(parr.count, 1))
    pipelineLog("✅ [像素桥] 解码完成（\(Int(Date().timeIntervalSince(t0)))s）：\(pixels.shape)，NaN/Inf=\(hasBad ? "有" : "无") mean=\(pMean)")

    let outDir = outputVideoDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let baseName = nextAssetName(prefix: assetPrefix)
    let fCount = pixels.shape[2]
    let h = pixels.shape[3], w = pixels.shape[4]
    // 二采像素桥产物统一 ProRes 422：容器 .mov（ProRes 不支持 mp4 封装）
    let silentPath = "\(outDir)/\(baseName)_silent.mov"
    do {
        try writeMp4(frameCount: fCount, width: w, height: h, fps: Int(fps.rounded()), to: silentPath, proRes: true) { f, base, bytesPerRow, isV210 in
            let frameArr = pixels[0 ..< 1, 0 ..< 3, f ..< (f + 1), 0 ..< h, 0 ..< w]
            let squeezed = frameArr.reshaped([3, h, w])
            // 像素桥二采 ProRes 422 最终输出：isV210=true → 10bit v210（BT.709 limited RGB→YCbCr 打包），
            // 使 ProRes 真正吃满 10bit（改造 A）
            return fillFrameForPixelBuffer(squeezed, width: w, height: h, base: base, bytesPerRow: bytesPerRow, isV210: isV210)
        }
        pipelineLog("✅ [像素桥] 无声 ProRes 422 已生成：\(silentPath)（音轨交由调用方沿用源视频）")
    } catch {
        pipelineLog("⚠️ [像素桥] ProRes 422 写入失败：\(error.localizedDescription)")
        return nil
    }
    return silentPath
}

/// 把源视频的音轨原样拷贝混入无声视频（沿用源音轨，不生成/不处理音频）。
/// 以无声视频时长为准，源音轨超出部分自动裁剪；源视频无音轨或混流失败返回 false（由调用方决定保留无声）。
func muxSourceAudioOnto(silentVideo: String, sourceVideo: String, to outPath: String) -> Bool {
    let silAsset = AVURLAsset(url: URL(fileURLWithPath: silentVideo))
    let srcAsset = AVURLAsset(url: URL(fileURLWithPath: sourceVideo))
    guard silAsset.duration.isNumeric, silAsset.duration.seconds > 0,
          let vidTrack = silAsset.tracks(withMediaType: .video).first else { return false }
    let srcAudio = srcAsset.tracks(withMediaType: .audio)
    let comp = AVMutableComposition()
    guard let cv = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else { return false }
    do {
        try cv.insertTimeRange(CMTimeRange(start: .zero, duration: silAsset.duration), of: vidTrack, at: .zero)
        if let aT = srcAudio.first, let ca = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
            let aDur = min(silAsset.duration, aT.timeRange.duration)
            try ca.insertTimeRange(CMTimeRange(start: aT.timeRange.start, duration: aDur), of: aT, at: .zero)
            pipelineLog("🎵 [像素桥] 沿用源音轨：\(sourceVideo)（时长裁剪到视频 \(aDur.seconds)s）")
        } else {
            pipelineLog("ℹ️ [像素桥] 源视频无音轨，输出保持无声")
        }
    } catch {
        pipelineLog("⚠️ [像素桥] 混流插入失败：\(error.localizedDescription)")
        return false
    }
    try? FileManager.default.removeItem(atPath: outPath)
    guard let exporter = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetPassthrough) else { return false }
    exporter.outputURL = URL(fileURLWithPath: outPath)
    // silent 视频为 ProRes422 .mov：passthrough 转封装到 .mov 以保留 ProRes 轨道（mp4 容器不支持 ProRes）
    exporter.outputFileType = .mov
    exporter.shouldOptimizeForNetworkUse = false
    let sem = DispatchSemaphore(value: 0)
    exporter.exportAsynchronously { sem.signal() }
    sem.wait()
    if exporter.status == .completed {
        pipelineLog("✅ [像素桥] 音轨混流完成：\(outPath)")
        return true
    }
    pipelineLog("⚠️ [像素桥] 音轨混流失败：\(exporter.error?.localizedDescription ?? "未知")")
    return false
}

/// 完全静态空文本条件加载（改造 B：二采无提示词，零模型权重）。
///
/// 文本条件只来自静态来源，任何情况下都不加载 Gemma 12B / connector / tokenizer，
/// 也不调用 encodePrompt：
/// 1. 常量文件存在：直接读 empty_text_cond.safetensors
///    （condV [1,256,4096] f32、condA [1,256,2048] f32）；
/// 2. 常量文件缺失或不可用：直接构造同形状全零张量兜底（用户实测空文本条件即可），
///    绝不自动自举、绝不启动 Gemma 编码空串。
///
/// - Returns: (videoCond, audioCond)，永不失败。
private func loadStaticEmptyTextCond() async -> (video: MLXArray, audio: MLXArray) {
    let path = "\(CommonPaths.ltxServeDir)/empty_text_cond.safetensors"
    if FileManager.default.fileExists(atPath: path),
       let arrays = try? MLX.loadArrays(url: URL(fileURLWithPath: path)),
       let v = arrays["condV"], let a = arrays["condA"],
       v.shape == [1, 256, 4096], a.shape == [1, 256, 2048] {
        return (v, a)
    }
    let zeroV = MLXArray.zeros([1, 256, 4096], dtype: .float32)
    let zeroA = MLXArray.zeros([1, 256, 2048], dtype: .float32)
    pipelineLog("ℹ️ [像素桥] 空文本常量缺失/不可用（\(path)），已用静态全零条件兜底 condV \(zeroV.shape) / condA \(zeroA.shape)；未加载 Gemma/connector，无任何模型权重调用")
    return (zeroV, zeroA)
}

/// 像素桥 σ 档（4 步，仅【非 IC】精修路径默认使用）：σ0=0.909375 官方重画档。
/// ⚠️ IC-LoRA 官方模式（权重存在自动启用）不再使用本表：已在
/// runLTXStage2RefineOnLatent 内锁定官方 STAGE_2_DISTILLED_SIGMAS
/// （ltx25Stage2Sigmas 4 步，σ0=0.909375 保留升频锚点 + ancestral SDE，CFG=1）——
/// 与下方 pixRefineSigmas 数值一致，但语义以官方 Stage2 为准；σ0=1.0 的 9 步
/// t2v 蒸馏档在 IC 下不收敛（彩色网格/马赛克）。
/// 原生 LTX 4.5 段仍走官方 ltx25Stage2Sigmas（见 840/898 行）；非 IC 路径需要其它档位时用 PIX_SIGMAS 覆盖。
private let pixRefineSigmas: [Float] = [0.909375, 0.725, 0.421875, 0]

/// 像素桥主入口：外部半清视频（任意模型直出，如 H3 672×384）→ LTX 升频 + Stage2 二采 → 全清 mp4。
/// 二采无提示词（改造 B）：refine 不再编码用户 prompt / 不再依赖 Gemma 12B + connector 常驻，
/// 文本条件统一取完全静态空条件（loadStaticEmptyTextCond：常量文件读张量，缺失时全零兜底，
/// 任何情况零模型权重调用、不启动 Gemma/connector）；
/// 画面由 stage1 音轨 IC（frozen_a 锁口型）与画面结构/guide 控制。
/// 不改动 LTX 原生管线任何代码；返回最终 mp4 路径（无声），失败返回 nil。
/// - Parameters:
///   - videoPath: 外部视频绝对路径（帧数建议为 8F-7 形；非对齐帧自动裁尾到最近 8F-7）
///   - imagePaths: 可选首/尾帧图（全分辨率参考钉入；传 2 张=首尾帧，1 张=仅首帧）
func ltxEnhanceExternalVideoWithStage2(
    videoPath: String,
    // 内存直通（方案 C）：非 nil 时二采像素直接取该内存元组（H3 stage1 VAE 解码像素），不读盘；
    // nil 则保持原读盘语义（自检入口/兼容路径）。字段语义与 readVideoFramesToBCFHW 返回值一致。
    memoryFrames: (pixels: MLXArray, width: Int, height: Int, fps: Double, frameCount: Int)? = nil,
    // 内存直通时的音轨/音频条件源文件（“给用户看”的 H3 stage1 预览 mp4，含 H3 生成音轨）；
    // nil 时回退到 videoPath 本身（原语义：沿用 stage1 落盘文件音轨）。
    sourceAudioVideoPath: String? = nil,
    // 改造 B：二采无提示词 —— 无 prompt 参数；文本条件由 loadStaticEmptyTextCond 提供
    imagePaths: [String] = [],
    negativePrompt: String? = nil,
    seed: UInt64 = 42,
    isCancelled: @escaping () -> Bool = { false },
    anchorEveryNFrames: Int? = nil,
    softAnchorMask: Float? = nil,
    tailGuideMask: Float? = nil,
    icLoRAEnable: Bool? = nil  // IC-LoRA 官方模式（见 runLTXStage2RefineOnLatent）：nil=默认关闭（非 IC 路径）；true=强制启用。LTX_IC_LORA=1/0 环境变量可覆盖
) async -> String? {
    // 中间帧锚定参数（仅本入口生效，不影响原生 LTX 4.5 段）：
    // 【只精修画面模式】默认每个 latent 帧都软锚定一次源画面（anchorEvery=1），
    // softM=0.8：每步 80% 模型结果 + 20% 源回拉；配合 σ0=0.5 保守起点，
    // 模型从头到尾都看得到源结构，只做纹理精修，不自由重画口型/动作。
    // 可用环境变量 PIX_ANCHOR_EVERY / PIX_SOFT_M 快速调参（anchorEvery=0 关闭锚定）。
    let anchorEvery: Int = anchorEveryNFrames ?? (Int(ProcessInfo.processInfo.environment["PIX_ANCHOR_EVERY"] ?? "") ?? 1)
    let softM: Float = softAnchorMask ?? (Float(ProcessInfo.processInfo.environment["PIX_SOFT_M"] ?? "") ?? 0.8)
    // 尾帧软引导（对齐官方 keyframe-guide 语义）：默认 0=硬钉（旧行为），>0 时尾帧 mask 由硬钉改软引导。
    // 可用环境变量 PIX_TAIL_M 覆盖（H3 像素桥默认 0.6，复现旧硬钉行为设 PIX_TAIL_M=0）。
    let tailM: Float = tailGuideMask ?? (Float(ProcessInfo.processInfo.environment["PIX_TAIL_M"] ?? "") ?? 0)
    if tailM > 0 {
        pipelineLog("ℹ️ [像素桥] 尾帧软引导开启（tailM=\(tailM)，对齐官方 keyframe-guide 软语义；尾帧不再 100% 硬钉外部图）")
    }
    // 全片 latent guide（官方 Detailer guiding_latents 语义）：PIX_GUIDE_W>0 时开启，
    // 每步把采样轨迹拉向 guideClean（默认=升频 clean latent normLat）的当前 σ 加噪版。
    // 与 imagePaths 首尾帧的关系（v2.0 软引导并存，不再互斥）：
    //   guideW>0 时关闭旧 mask 硬钉路径（anchorEvery/tailM 置 0），但 imagePaths 仍传入 refine：
    //   首/尾帧的 guide 目标 = 源帧×(1-blend) + 外部参考 latent×blend（blend=PIX_EDGE_M），
    //   中段仍锁源 normLat。guide 每步只是把 vx 拉回目标轨迹（非帧替换/非 mask），
    //   所以参考图是「软引导」：轮廓/构图被拉向参考但不强制首帧等于参考（规避 v1.1 RGB 色边）。
    //   PIX_EDGE_M=0 时退回纯锁源（参考图完全不参与）；=1.0 时首尾帧 guide 目标=参考图本身
    //   （guideW<1 时仍非硬钉；guideW=1 时约等于首尾帧被参考接管）。
    // 默认 guideW 1.0（满锁保轮廓）；设 PIX_GUIDE_W=0 可退回旧稀疏帧硬钉/软引导路径。
    let guideW: Float = (Float(ProcessInfo.processInfo.environment["PIX_GUIDE_W"] ?? "") ?? 1.0)
    // 首尾帧 guide 渗透系数（软引导强度）：0=不渗透（纯锁源）；1=guide 目标完全指向参考图。默认 0.5（半引导）。
    let edgeM: Float = (Float(ProcessInfo.processInfo.environment["PIX_EDGE_M"] ?? "") ?? 0.5)
    if guideW > 0 {
        if !imagePaths.isEmpty {
            pipelineLog("ℹ️ [像素桥] 全片 latent guide 开启（guideW=\(guideW) + 首尾帧软引导 edgeM=\(edgeM)；锚定/tailM 自动忽略，首尾帧按渗透系数并入 guide 目标）")
        } else {
            pipelineLog("ℹ️ [像素桥] 全片 latent guide 开启（guideW=\(guideW)，官方 Detailer 语义；锚=源视频自身 latent，无外部首尾帧）")
        }
    }
    if anchorEvery == 0 {
        pipelineLog("ℹ️ [像素桥] 中间帧锚定已关闭（anchorEvery=0）")
    }
    let tAll = Date()
    pipelineLog("\n========== [像素桥] 外部视频 → LTX 升频+二采 开始 ==========")
    // 改造 B：二采无提示词 —— 原「prompt 为空即失败」guard 删除；文本条件在步骤 4 由
    // 完全静态空条件提供（loadStaticEmptyTextCond：常量读张量 / 缺失全零兜底，零模型权重），
    // 不再要求非空 prompt / Gemma 编码。

    // 1) 读取外部视频 → 半清像素（32 对齐）
    //    内存直通（方案 C）：memoryFrames 非 nil 时直接用 H3 stage1 解码像素（不经磁盘编码），
    //    否则读盘（AVAssetReader 按源文件 BT.709 标记自动做 YCbCr→RGB）。
    guard let video = memoryFrames ?? readVideoFramesToBCFHW(videoPath: videoPath),
          video.frameCount >= 9 else {
        pipelineLog("❌ [像素桥] 视频读取失败或帧数不足")
        return nil
    }
    let halfW = video.width, halfH = video.height
    var total = video.frameCount
    let F = (total + 7) / 8
    let outFrames = 8 * F - 7
    var pixels = video.pixels
    if total > outFrames {
        pixels = pixels[0 ..< 1, 0 ..< 3, 0 ..< outFrames, 0 ..< halfH, 0 ..< halfW]
        total = outFrames
    }
    guard F >= 2, halfW % 32 == 0, halfH % 32 == 0 else {
        pipelineLog("❌ [像素桥] 尺寸/帧数异常：\(halfW)×\(halfH) \(total) 帧 → F=\(F)")
        return nil
    }
    pipelineLog("ℹ️ [像素桥] 输入 \(total) 帧 \(halfW)×\(halfH) → latent F=\(F)，输出像素 \(halfW*2)×\(halfH*2)")

    // 2) LTX VAE 整段编码（外部像素 → LTX latent 域）
    let gBase = GenConfig(numFrames: total, height: halfH, width: halfW,
                          frameRate: Float(video.fps), numSteps: 8, seed: seed)
    let stage1Config = Stage1Config.distilled(g: gBase)
    let tEnc = Date()
    // vae_encoder 用完即放：编码在池内同步完成，pool 结束后权重立即释放，
    // 否则它与后续 DiT/Gemma/refine I2V 重复加载的 encoder 叠载，是像素桥 OOM 主因之一。
    var halfLatent: MLXArray? = nil
    autoreleasepool {
        guard let ew = try? MLX.loadArrays(url: URL(fileURLWithPath: stage1Config.vaeEncoderPath)) else {
            pipelineLog("❌ [像素桥] vae_encoder 加载失败")
            return
        }
        halfLatent = vaeEncodeVideo(weights: ew, pixelsBCFHW: pixels)
    }
    MLX.Memory.clearCache()
    guard let halfLatent = halfLatent else { return nil }
    pipelineLog("✅ [像素桥] 整段编码完成（\(Int(Date().timeIntervalSince(tEnc)))s）：\(halfLatent.shape)")

    // 3) DiT 加载/复用（与原生管线同一缓存，不重复读盘）
    let ditPath = stage1Config.ditPath
    guard MemoryPolicy.ensureCapacity(for: DiTModelCache.shared.get(path: ditPath) == nil ? (fileSizeBytes(ditPath) ?? 0) : 0,
                                      task: .video, current: .encoding, next: .sampling,
                                      width: halfW, height: halfH, frames: total) else {
        pipelineLog("❌ [像素桥] 内存不足，无法加载 DiT")
        return nil
    }
    let dit: LTXVideoDiT
    let t3 = Date()
    if let cached = DiTModelCache.shared.get(path: ditPath) {
        dit = cached
        pipelineLog("✅ [像素桥] 复用已加载 DiT 权重")
    } else {
        guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: ditPath)) else {
            pipelineLog("❌ [像素桥] DiT 权重加载失败"); return nil
        }
        let q = cachedQuantParams(from: ditPath) ?? (bits: 4, groupSize: 64)
        let d = LTXVideoDiT(groupSize: q.groupSize, bits: q.bits)
        var strippedW: [String: MLXArray] = [:]
        for (k, v) in weights {
            strippedW[k.hasPrefix("transformer.") ? String(k.dropFirst("transformer.".count)) : k] = v
        }
        let nested = NestedDictionary<String, MLXArray>.unflattened(strippedW)
        d.update(parameters: nested)
        if let kf = strippedW["keyframes_abs_pos_embedding"] {
            d.keyframes_abs_pos_embedding = kf
        }
        pipelineLog("✅ [像素桥] DiT 权重灌入完成（\(Int(Date().timeIntervalSince(t3)))s，\(strippedW.count) 键）")
        // 权重加载保持 lazy：lazy Load 图 0s 建图，真实读盘延迟到编译窗口内首次 forward 自然发生，
        // 编译窗口起点水位最低。已移除此前“提前 eval 权重”的强制驻留机制（提前驻留会与全尺寸
        // 编译峰值叠加顶穿 48G）。
        DiTModelCache.shared.store(d, path: ditPath)
        dit = d
    }

    // 4) 文本条件：完全静态空文本条件（改造 B，无提示词二采；零模型权重）。
    //    优先读 empty_text_cond.safetensors 常量，缺失/不可用时函数内部直接构造
    //    同形状全零张量兜底 —— 任何情况都不启动 Gemma/connector，也不调用 encodePrompt。
    let cond = await loadStaticEmptyTextCond()
    pipelineLog("✅ [像素桥] 空文本条件已就绪：video \(cond.video.shape)，audio \(cond.audio.shape)（Gemma/connector 未加载）")
    // condV/condA 与 encodePrompt 产物形状一致（[1,256,4096]/[1,256,2048]），以 videoText/audioText
    // 参与 DiT attn2；audioCondLatent（源音轨真实条件，audio_vae 编码）链路在步骤 5 保持不动。
    // 二采全程不加载 Gemma/connector，无 MemoryPolicy 让位负担；此处仅保留通用内存策略调用。
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .sampling, next: .vae)

    // 5) 音频 latent：随机占位（DiT 前向必需；非 IC 采样用它起步，产物音轨沿用半清源视频音轨）。
    //    另尝试把源视频音轨编码为真实音频条件（audioCond*，仅 IC 模式 frozen_a 锁口型用；
    //    非 IC 不消费，行为与改造前完全一致）。失败/无音轨则 audioCond* 为 nil。
    let gFull = GenConfig(numFrames: outFrames, height: halfH * 2, width: halfW * 2,
                          frameRate: Float(video.fps), numSteps: 8, seed: seed)
    let (noiseA, audioPos) = makeAudioLatentAndPos(g: gFull)
    var audioCondLatent: MLXArray? = nil
    var audioCondPos: [Float]? = nil
    let naBudget = Int((Float(outFrames) / Float(video.fps) * 25.0).rounded())
    // 音轨源：内存直通时用 H3 预览文件（含 H3 生成音轨）抽音频条件/最终混流；原路径回退 videoPath。
    let audioSourcePath = sourceAudioVideoPath ?? videoPath
    // frozen_a 音频条件（audio_vae 编码源音轨）仅供 IC 模式消费；默认非 IC 不编码、不加载
    // audioVAE（收敛回旧版：仅随机噪声占位，产物音轨沿用源视频）。避免 audioVAE 权重驻留与
    // 编码瞬时峰值叠加编译窗口。启用规则与 runLTXStage2RefineOnLatent 保持一致：
    // LTX_IC_LORA=1 或 icLoRAEnable=true（LTX_IC_LORA=0 强制关）。
    let icAudioFlag = ProcessInfo.processInfo.environment["LTX_IC_LORA"]
    let icAudioRequested = (icAudioFlag == "1") || (icAudioFlag != "0" && icLoRAEnable == true)
    if icAudioRequested,
       let aw = try? MLX.loadArrays(url: URL(fileURLWithPath: CommonPaths.audioVae)),
       let pcm = loadAudioPCM16kStereo(path: audioSourcePath), !pcm.isEmpty {
        sharedAudioVAEWeights = aw
        let lat = encodeAudioCond(weights: aw, pcm: pcm, maxTokens: naBudget)
        audioCondLatent = lat
        audioCondPos = makeAudioPositions(count: lat.shape[1])
        pipelineLog("🎵 [像素桥] 源视频音轨已编码为音频条件 latent（Na=\(audioCondPos!.count)；IC 模式 frozen_a 锁口型）")
    }
    pipelineLog("ℹ️ [像素桥] 音频 latent 随机占位（Na=\(audioPos.count)；IC 关闭或源音轨不可读时音频条件关闭）")

    // 6) 通用 refine（升频 ×2 + Stage2 精修）
    //    外部半清视频进像素桥 Stage2 默认 pixRefineSigmas（与官方 STAGE_2_DISTILLED_SIGMAS
    //    数值一致，σ0=0.909375 保留升频锚点）。IC 模式锁定官方档 + ancestral SDE；
    //    非 IC 需要其它档位时用 PIX_SIGMAS 覆盖（逗号分隔、降序、末位必须为 0）。
    //    原生 LTX 4.5 段仍走官方 ltx25Stage2Sigmas（见 840/898 行），本改动不影响原生路径。
    let refineSigmas: [Float]
    if let raw = ProcessInfo.processInfo.environment["PIX_SIGMAS"], !raw.isEmpty {
        let parts = raw.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        if parts.count >= 2, parts.last == 0 {
            refineSigmas = parts
            pipelineLog("ℹ️ [像素桥] PIX_SIGMAS 覆盖 refine σ 表：\(parts.map { String(format: "%.4f", $0) }.joined(separator: ","))")
        } else {
            refineSigmas = pixRefineSigmas
            pipelineLog("⚠️ [像素桥] PIX_SIGMAS 格式非法（需≥2个降序浮点且末位0），回退默认 \(pixRefineSigmas)")
        }
    } else {
        refineSigmas = pixRefineSigmas
    }
    guard let vRef = runLTXStage2RefineOnLatent(
        halfLatent: halfLatent, dit: dit,
        condV: cond.video, condA: cond.audio,
        noiseA: noiseA, audioPos: audioPos,
        audioCondLatent: audioCondLatent, audioCondPos: audioCondPos,
        imagePaths: imagePaths,
        fullPixelWidth: halfW * 2, fullPixelHeight: halfH * 2,
        pixelFrames: outFrames,
        frameRate: Float(video.fps),
        seed: seed,
        isCancelled: isCancelled,
        sigmas: refineSigmas,
        anchorEveryNFrames: guideW > 0 ? 0 : anchorEvery,
        softAnchorMask: softM,
        guideWeight: guideW,
        headTailGuideBlend: edgeM,
        tailGuideMask: guideW > 0 ? 0 : tailM,
        icLoRAEnable: icLoRAEnable) else {
        pipelineLog("❌ [像素桥] refine 失败")
        return nil
    }
    // refine 已完成：DiT/编译图/IC 参考对后续解码已无用，与原生 4.5 解码前同策略卸掉让位，
    // 否则 DiT 11G+编译图与 VAE decoder 叠载（解码被迫超分块硬扛），且任务收尾后残留
    // 下一任务衔接直接背 30GB 起步（04:05 H3 DiT 加载被压死即此因）。
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .vae, next: nil)

    // 7) VAE 解码 → 无声 mp4，并把半清源视频的音轨原样沿用（不生成/不处理音频）
    guard let silentPath = decodeLatentToSilentMP4(latentNDHWC: vRef, fps: video.fps, assetPrefix: "uhd") else {
        pipelineLog("❌ [像素桥] 解码落盘失败")
        return nil
    }
    let finalPath = silentPath.replacingOccurrences(of: "_silent.mov", with: ".mov")
    archiveIfExists(finalPath)
    if muxSourceAudioOnto(silentVideo: silentPath, sourceVideo: audioSourcePath, to: finalPath) {
        try? FileManager.default.removeItem(atPath: silentPath)
    } else {
        // 源无音轨或混流失败：保留无声视频为最终产物
        try? FileManager.default.moveItem(atPath: silentPath, toPath: finalPath)
    }
    pipelineLog("✅ [像素桥] 完成（总耗时 \(Int(Date().timeIntervalSince(tAll)))s）：\(finalPath)")
    return finalPath
}

/// 内存直通入口（H3 stage1 像素帧 → LTX 像素桥，方案 C）：
/// 不落盘像素文件，直接用 H3 VAE 解码后的半清像素做二采输入；
/// 预览/音轨源仍来自 sourceAudioVideoPath（H3 另存的 h264 最高质量 mp4，含 H3 生成音轨）。
func ltxEnhanceExternalVideoWithStage2Pixels(
    pixels: MLXArray,
    pixelWidth: Int,
    pixelHeight: Int,
    pixelFps: Double,
    sourceAudioVideoPath: String,
    // 改造 B：二采无提示词 —— 无 prompt 参数；文本条件由 loadStaticEmptyTextCond 提供
    imagePaths: [String] = [],
    negativePrompt: String? = nil,
    seed: UInt64 = 42,
    isCancelled: @escaping () -> Bool = { false },
    anchorEveryNFrames: Int? = nil,
    softAnchorMask: Float? = nil,
    tailGuideMask: Float? = nil,
    icLoRAEnable: Bool? = nil
) async -> String? {
    let frames: (pixels: MLXArray, width: Int, height: Int, fps: Double, frameCount: Int) =
        (pixels, pixelWidth, pixelHeight, pixelFps, pixels.shape[2])
    pipelineLog("🎯 [像素桥] 内存直通入口：H3 stage1 像素 \(frames.pixels.shape)（\(pixelWidth)×\(pixelHeight) @\(pixelFps)fps），音轨源 \(sourceAudioVideoPath)")
    return await ltxEnhanceExternalVideoWithStage2(
        videoPath: sourceAudioVideoPath,
        memoryFrames: frames,
        sourceAudioVideoPath: sourceAudioVideoPath,
        imagePaths: imagePaths,
        negativePrompt: negativePrompt,
        seed: seed,
        isCancelled: isCancelled,
        anchorEveryNFrames: anchorEveryNFrames,
        softAnchorMask: softAnchorMask,
        tailGuideMask: tailGuideMask,
        icLoRAEnable: icLoRAEnable)
}

