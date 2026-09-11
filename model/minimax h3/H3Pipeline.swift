//
//  minimax h3.swift
//  无限画布
//
//  MiniMax H3 FL2VA 组装管线（文本编码 → DiT 去噪 → VAE 解码 → mp4）
//
//  串联流程（对齐 mlx-serve-main/src/minimax_h3.zig 的 generateOne）：
//    1. 加载 2 张 keyframe 图片 → VAE encodeImage → patchifyVideo → concat
//       → 噪声增强 r = 0.999·r + 0.001·noise（visualCondTimestep=0.999）
//    2. tokenizer 编码 prompt → H3TextEncoder.encodeItems([.text(ids)])
//       → hidden [seq, textDim]（编码器加载后立即释放）
//    3. H3DiT.load + attachLoras(turbo_lora) + precomputeAdaln + refineText
//       → Euler 采样：单遍 turbo 直出（官方 4/6 步 LoRA 调度），按 steps 完整跑到 0。
//       video_in = concat(cond_rows, video_x) axis0，
//       x += out.video·(sigmas[i+1]-sigmas[i])，每步 clearCache
//    4. video_x reshape [t,lh2,lw2,24,2,2] → transpose(3,0,1,4,2,5) → contig
//       → reshape [1,24,t,lh,lw] → H3VAE.decode → pixels [1,3,T,H,W]
//       （H3 侧不再做 latent 放大/Stage2 refine；放大与第二阶段交给 LTX 二采流程）
//    5. writeMp4 逐帧写出（复用工程模型公共函数-通用.swift）
//
//  内存管理：分阶段加载，用后立即置 nil + MLX.Memory.clearCache()。
//

import Foundation

/// H3 stage1 内存直通桥（h3+ltx 二采，方案 C）：
/// generateVideo 内部 VAE 解码出的 stage1 像素帧不落盘，直接持有在 bridge 上，
/// 队列在 generateVideo 返回后把 pixels 传给像素桥内存入口做二采（无磁盘中间编码损失）。
public final class H3Stage2MemoryBridge {
    public var pixels: MLXArray?      // [1,3,T,H,W] f32（[-1,1]，H3 VAE 解码输出，与 readVideoFramesToBCFHW 同域）
    public var width: Int = 0
    public var height: Int = 0
    public var frameCount: Int = 0
    public var fps: Int = 24
}
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import Tokenizers
import Hub

/// sigma 调度风格（对照实验开关）
/// - `.official`：官方 shift 公式 `sigmaSchedule`（默认，turbo 4-8 步标准调度）
/// - `.betaRefined`：Beta(0.6,0.6) 分布 + 余弦尾段精修（`betaRefinedSchedule`，
///   extra=1 / startAt=0.7），末段不再大步跳 0，用于同 seed 质量对照
public enum H3SigmaScheduleStyle {
    case official
    case betaRefined
}

/// 第二阶段 H3 refine 配置（官方 latent 升频 + 高分辨率加噪精修链路）。
/// stage1 采样完成后不清零落地，latent 行形式在内存中直通：
///   videoX/cond rows → zlat → 空间几何×scale → 回行形式（网格×scale²）
///   → 按 refineSigmas 加噪 → Euler 低步重采样 → 直接 VAE decode。
/// scale=1 或 nil 时不启用，行为与旧路径完全一致。
public struct H3Stage2Config {
    public var scale: Int = 2
    /// 视频域（model sigma）精修调度，官方 i2v/ref2va 工作流 3 步 refine。
    /// 首项即加噪起点，末项必须为 0。
    public var refineSigmas: [Double] = [0.9035, 0.6316, 0.3158, 0.0]
    /// refine 段加噪种子（缺省在 stage1 seed 上派生，避免与主采样同噪声退化）
    public var refineSeed: UInt64? = nil

    public init(scale: Int = 2,
                refineSigmas: [Double] = [0.9035, 0.6316, 0.3158, 0.0],
                refineSeed: UInt64? = nil) {
        self.scale = scale
        self.refineSigmas = refineSigmas
        self.refineSeed = refineSeed
    }
}

public enum H3FL2VAPipeline {

    /// FL2VA 视频生成入口（异步：tokenizer 构建需要 async）。
    /// - Parameters:
    ///   - prompt: 文本提示词
    ///   - firstImagePath: 首帧图片绝对路径
    ///   - lastImagePath: 尾帧图片绝对路径
    ///   - outPath: mp4 输出路径
    ///   - width/height: 生成分辨率（建议 256，需 32 对齐）
    ///   - steps: 采样步数（turbo 6 步）
    ///   - latentT: 潜在帧数（5 → 17 输出帧）
    ///   - seed: 随机种子
    ///   - log: 进度回调
    public static func generateVideo(
        prompt: String,
        firstImagePath: String,
        lastImagePath: String,
        outPath: String,
        width: Int = 256,
        height: Int = 256,
        steps: UInt32 = 6,
        latentT: UInt32 = 5,
        seed: UInt64 = 42,
        sparsePolicy: SparsePolicy = .mix,
        scheduleStyle: H3SigmaScheduleStyle = .official,
        // 分阶段门控已取消：默认 nil，stage1 去噪全程按 sparsePolicy 走 SOL 稀疏注意力，
        // 不再在首尾段整步回退稠密（SDPA）。保留该参数仅供显式对照实验：
        // 传 (start, end) 时 progress = step/total ∈ (start, end) 才稀疏，两端全 dense
        //（对齐 kijai Sol-Attn 的 percent_to_sigma：start=0.2 / end=0.9）；
        // 环境变量 NA_H3_SPARSE_GATE=0 仍可强制关闭门控做对照。
        sparseGatePercent: (start: Float, end: Float)? = nil,
        // PAB 跨步 attention 缓存：全程接入，按 attnBroadcastRefresh 调度执行——
        // warmup/tail 强制重算，中间步按 k 间隔刷新，其余步复用上一步 attention 输出。
        // k<=1 等价禁用；环境变量 NA_H3_PAB=0 可强制关闭做对照，NA_H3_PAB_K 可覆盖间隔扫描。
        attentionBroadcastK: UInt32 = 2,
        log: (String) -> Void = { s in print("[H3-FL2VA] \(s)"); fflush(stdout) },
        stage2: H3Stage2Config? = nil,
        proResOutput: Bool = false,
        stage2MemBridge: H3Stage2MemoryBridge? = nil
    ) async throws -> String {
        let t0 = Date()
        let modelDir = "/Users/huachayui/Downloads/minimax h3/MiniMax-H3-FL2VA-MLX-Serve-4bit"
        let wURL = { (name: String) in URL(fileURLWithPath: "\(modelDir)/\(name)") }

        // ── 内存打点（RSS phys_footprint + MLX activeMemory）──
        func memFootprintKB() -> UInt64 {
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) { p in
                p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ip, &count)
                }
            }
            return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) / 1024 : 0
        }
        func h3MemLog(_ label: String) {
            let rssMB = Double(memFootprintKB()) / 1024.0
            let mlxMB = Double(MLX.Memory.activeMemory) / 1024.0 / 1024.0
            let cacheMB = Double(MLX.Memory.cacheMemory) / 1024.0 / 1024.0
            log("[MEM] \(label)：RSS \(String(format: "%.0f", rssMB))MB，MLX active \(String(format: "%.0f", mlxMB))MB，MLX cache \(String(format: "%.0f", cacheMB))MB")
        }
        // 修复：禁止 cacheLimit = 0（释放即真正 free MTLBuffer），
        // 否则 GPU 队列尚引用的 buffer 会被提前释放 → MTLDebugCommandBuffer
        // "references deallocated object" 断言（use-after-free）。
        // H3 单步中间 buffer（50 层 DiT）远大于 LTX：全局 MemoryPolicy 的 2GB
        // 缓存池装不下 → allocator 每步 trim 归还 → 下步重分配 → 内存忽高忽低。
        // 放大到 10GB 让采样循环内中间 buffer 真正跨步复用（不可设 0，见上）。
        MLX.Memory.cacheLimit = 10_000_000_000
        // ★ H3 收尾恢复：10GB 缓存池仅供 H3 单步中间 buffer 跨步复用（50 层 DiT >> LTX），
        // 函数返回/抛错时立即恢复全局 2GB（MemoryPolicy.bufferCacheLimit），避免高水位残留到
        // 后续像素桥/LTX refine 编译窗口（实测 H3 后 10GB 未恢复 + refine 重编译 → low-swap jetsam）。
        defer { MLX.Memory.cacheLimit = 2_000_000_000 }

        // ── 0. 配置 ──
        let cfg: H3Config = (try? H3Config.load(from: wURL("config.json"))) ?? H3Config()
        log("config: \(cfg.hiddenSize)h/\(cfg.numLayers)L video_shift=\(cfg.sigmaShiftVideo) audio_shift=\(cfg.sigmaShiftAudio)")

        // ── 1. 加载 2 张 keyframe 图片 → VAE 编码 → cond rows ──
        guard let firstPx = loadImageBCFHW(path: firstImagePath, width: width, height: height),
              let lastPx = loadImageBCFHW(path: lastImagePath, width: width, height: height) else {
            throw NSError(domain: "H3FL2VA", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "图片加载失败：\(firstImagePath) / \(lastImagePath)"])
        }
        log("图片加载完成：首帧 \(firstPx.shape)，尾帧 \(lastPx.shape)")

        var condRows: MLXArray!
        var latC = 0, latT = 0, latH = 0, latW = 0, gridH = 0, gridW = 0
        autoreleasepool {
            var vaeWeights: H3Weights? = try! H3Weights(url: wURL("video_vae.safetensors"))
            vaeWeights?.cacheEnabled = false
            var vae: H3VAE? = try! H3VAE.load(vaeWeights!)
            let firstLat = vae!.encodeImage(firstPx)   // [1,24,1,h,w]
            let lastLat = vae!.encodeImage(lastPx)     // [1,24,1,h,w]
            latT = firstLat.shape[2]
            latH = firstLat.shape[3]
            latW = firstLat.shape[4]
            latC = firstLat.shape[1]
            gridH = latH / 2
            gridW = latW / 2

            func toRows(_ lat: MLXArray) -> MLXArray {
                let nhwc = lat.transposed(0, 2, 3, 4, 1).reshaped([latT, latH, latW, latC])
                return H3TensorOps.patchifyVideo(nhwc, gridH: gridH, gridW: gridW)
            }
            condRows = concatenated([toRows(firstLat), toRows(lastLat)], axis: 0)
            MLX.eval(condRows)
            log("keyframe 编码完成：latent [1,\(latC),\(latT),\(latH),\(latW)]，cond rows \(condRows.shape)")

            // 噪声增强：r = 0.999·r + 0.001·noise
            let augNoise = MLXRandom.normal(condRows.shape, key: MLXRandom.key(seed))
            let augA = H3TensorOps.scalarLike(Float(H3Const.visualCondTimestep), condRows)
            let augB = H3TensorOps.scalarLike(Float(1.0 - H3Const.visualCondTimestep), condRows)
            condRows = condRows * augA + augNoise * augB
            MLX.eval(condRows)

            // 释放 VAE 编码器（解码阶段再加载）
            vae = nil
            vaeWeights = nil
            MLX.Memory.clearCache()
            h3MemLog("VAE 编码器已释放（autoreleasepool 内）")
        }
        h3MemLog("VAE 编码器已释放（pool drain 后）")
        log("VAE 编码器已释放")

        // ── 2. tokenizer + 文本编码 ──
        let lmConfig = LanguageModelConfigurationFromHub(modelFolder: URL(fileURLWithPath: modelDir))
        guard let tokConfig = try? await lmConfig.tokenizerConfig,
              let tokData = try? await lmConfig.tokenizerData,
              let tokenizer = try? AutoTokenizer.from(tokenizerConfig: tokConfig, tokenizerData: tokData) else {
            throw NSError(domain: "H3FL2VA", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer 加载失败"])
        }
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) }
        log("prompt 编码：\(ids.count) tokens")

        var textHidden: MLXArray!
        autoreleasepool {
            var teWeights: H3Weights? = try! H3Weights(url: wURL("text_encoder.safetensors"))
            teWeights?.cacheEnabled = false
            var textEncoder: H3TextEncoder? = try! H3TextEncoder.load(teWeights!)
            let encoded = try! textEncoder!.encodeItems([.text(ids)])
            textHidden = encoded.hidden
            MLX.eval(textHidden)
            h3MemLog("文本编码完成")
            log("文本编码完成：hidden \(textHidden.shape)，tags \(encoded.tags.count)")

            textEncoder = nil
            teWeights = nil
            MLX.Memory.clearCache()
            h3MemLog("文本编码器置 nil 后（autoreleasepool 内）")
        }
        h3MemLog("文本编码器置 nil 后（pool drain 后）")
        Thread.sleep(forTimeInterval: 3)
        h3MemLog("文本编码器置 nil 后（sleep 3s）")
        log("文本编码器已释放")

        // ── 3. DiT 加载 + LoRA + AdaLN 预计算 + 采样 ──
        h3MemLog("DiT 加载前")
        log("DiT 开始加载（transformer.safetensors 18.7GB，读取+解析较慢）...")
        var ditWeights: H3Weights? = try H3Weights(url: wURL("transformer.safetensors"))
        ditWeights?.cacheEnabled = false
        var dit: H3DiT? = nil
        try autoreleasepool {
            dit = try H3DiT.load(from: ditWeights!, cfg: cfg)
        }
        h3MemLog("DiT 权重加载完成")
        log("DiT 权重加载完成，挂载 turbo LoRA...")
        // turbo LoRA：官方单遍直出统一用 minimax_h3_turbo_v4_step600_ema（6 步版，
        // 与 sigmaSchedule(steps:6) 完整 0→1 直出匹配）。可用环境变量 H3_LORA_FILE 覆盖做对照。
        // 社区二采（lightx2v 4 步 + Stage2 refine）已随 H3LatentUpscaler/H3Stage2Refine 移除。
        let loraName = ProcessInfo.processInfo.environment["H3_LORA_FILE"]
            ?? "minimax_h3_turbo_v4_step600_ema.safetensors"
        let loraURL = wURL(loraName)
        let loraExists = FileManager.default.fileExists(atPath: loraURL.path)
        let effectiveLora: URL = loraExists ? loraURL
            : wURL("minimax_h3_turbo_v4_step600_ema.safetensors")
        let patched = try dit!.attachLoras(from: [effectiveLora], scales: [1.0])
        log("turbo LoRA 挂载（\(effectiveLora.lastPathComponent)）：\(patched) 模块")

        // 布局：latentT → 实际输出帧数（h3PlanTemporal 与官方 planTemporal 互逆，37 → 124 帧）；
        // audioT 按官方 audio_t = round(frame_count * 40 / 24) 折算（124 → 207）
        let plan = h3PlanTemporal(Int(latentT))
        let frameCount = UInt32(plan.outputFrames)
        let audioT = H3Const.audioLatentT(frameCount: frameCount)
        let layout = PackedLayout(textLen: UInt32(textHidden.shape[0]),
                                  latentT: latentT, latentH: UInt32(latH), latentW: UInt32(latW),
                                  audioT: audioT,
                                  keyframes: [.first, .last],
                                  frameCount: frameCount)

        // sigma 调度：official = 官方 shift 公式；betaRefined = Beta 分布 + 余弦尾段精修
        // （betaRefinedSchedule 内部保证非零档数 = steps，末位补 0，可直接对位使用）
        let sigmas: [Double]
        switch scheduleStyle {
        case .official:
            sigmas = sigmaSchedule(steps: steps, shift: cfg.sigmaShiftVideo)
        case .betaRefined:
            sigmas = betaRefinedSchedule(totalSteps: Int(steps),
                                         shift: cfg.sigmaShiftVideo,
                                         extraSteps: 1,
                                         startAtSigma: 0.7,
                                         spacing: .cosine)
        }
        // 实际采样段数 = 档位间隔数（直出单遍路径 = steps）
        let stage1SegmentCount = sigmas.count - 1
        let tsList = collectScheduleTs(layout: layout, sigmas: sigmas,
                                       shiftV: cfg.sigmaShiftVideo, shiftA: cfg.sigmaShiftAudio,
                                       aug: CondNoiseAug())
        var refined = MLXArray(0)
        autoreleasepool {
            dit!.precomputeAdaln(ts: tsList)
            refined = dit!.refineText(textHidden)
            MLX.eval(refined)
            h3MemLog("DiT 就绪（pool 内）")
        }
        let rope = buildRope(layout: layout, invFreq: dit!.invFreq, dtype: PrecisionPolicy.defaultMainDType)
        h3MemLog("DiT 就绪（pool drain 后）")
        log("DiT 就绪：refined \(refined.shape)，schedule \(sigmas)")

        // 稀疏注意力开关（由调用方传入）：.off = 50 层全稠密 SDPA；
        // .mix = 官方策略，anchor 层 dense、其余交替 spatial/temporal。
        dit!.sparsePolicy = sparsePolicy
        let policyDesc: String
        switch sparsePolicy {
        case .off: policyDesc = "off（50 层全部稠密 SDPA）"
        case .mix: policyDesc = "mix（首尾2/中间2层 dense anchor，其余交替 spatial/temporal）"
        case .spatial: policyDesc = "spatial（全部层帧内空间注意力）"
        case .temporal: policyDesc = "temporal（全部层跨帧时间注意力）"
        }
        log("稀疏注意力策略：\(policyDesc)")
        if let gp = sparseGatePercent {
            log("稀疏门控：progress ∈ (\(gp.start), \(gp.end)) 中段稀疏，两端整步 dense（\(gp.start * 100)% 前 / \(gp.end * 100)% 后）")
        } else {
            log("稀疏门控：关闭（全程按稀疏策略执行）")
        }

        let nVideoRows = Int(latentT) * gridH * gridW
        let vpatch = cfg.videoPatchDim
        var videoX = MLXRandom.normal([nVideoRows, vpatch], key: MLXRandom.key(seed))
        let nAudioRows = Int(audioT * 2)
        var audioX = MLXRandom.normal([nAudioRows, 32], key: MLXRandom.key(seed &+ 1))
        MLX.eval(videoX, audioX)
        log("初始 latent：video \(videoX.shape)，audio \(audioX.shape)")
        let cMean: Float = condRows.mean().item()
        let cVar: Float = ((condRows * condRows).mean() - condRows.mean() * condRows.mean()).item()
        let vMean: Float = videoX.mean().item()
        let vVar: Float = ((videoX * videoX).mean() - videoX.mean() * videoX.mean()).item()
        log("STAT condRows mean=\(cMean) std=\(cVar.squareRoot()) videoX mean=\(vMean) std=\(vVar.squareRoot())")

        let sv = cfg.sigmaShiftVideo
        let sa = cfg.sigmaShiftAudio
        // PAB 全程接入：创建跨步缓存，按 k 间隔刷新调度走（复用上一步同层 attention 输出，
        // 跳过 norm1+qkv+rope+attn；warmup/tail 仍保护首尾步）。env 可关/可调做对照。
        let pabK: UInt32 = {
            if ProcessInfo.processInfo.environment["NA_H3_PAB"] == "0" { return 0 }
            if let s = ProcessInfo.processInfo.environment["NA_H3_PAB_K"], let v = UInt32(s) { return v }
            return attentionBroadcastK
        }()
        let pab1: H3AttnBroadcast? = pabK > 1 ? H3AttnBroadcast(count: dit!.blocks.count) : nil
        for i in 0..<stage1SegmentCount {
            let stepT = Date()
            // 分阶段门控（默认已取消）：sparseGatePercent 为 nil 时走 else 分支 —— 全程
            // sparsePolicy，不做任何整步 dense 回退；仅当调用方显式传入 (start, end) 时才启用
            // ComfyUI 式 sigma 门控（progress = i/steps 与 percent_to_sigma 阈值同构，
            // 0~start% 与 end~100% 整步回退全 dense SDPA，中间段才走稀疏 tau 路由）。
            var gateMode = "sol"
            var gateOn = sparseGatePercent != nil
            if let e = ProcessInfo.processInfo.environment["NA_H3_SPARSE_GATE"], e == "0" { gateOn = false }
            if gateOn, let gp = sparseGatePercent {
                let progress = Float(i) / Float(stage1SegmentCount)
                let denseStep = progress < gp.start || progress > gp.end
                dit!.sparsePolicy = denseStep ? .off : sparsePolicy
                gateMode = (denseStep || sparsePolicy == .off) ? "dense" : "sparse"
            } else {
                dit!.sparsePolicy = sparsePolicy
            }
            let sigma = sigmas[i]
            let dsigma = sigmas[i + 1] - sigmas[i]
            let plan = buildTimestepPlanGlobal(layout: layout, sigmaV: sigma, shiftV: sv, shiftA: sa,
                                               aug: CondNoiseAug(), globalTs: dit!.adalnTables!.ts)
            let videoIn = concatenated([condRows, videoX], axis: 0)
            let attnRefresh = pab1 == nil ? false : attnBroadcastRefresh(i, steps: UInt32(stage1SegmentCount), k: pabK)
            let pabMark = pab1 == nil ? "" : (attnRefresh ? " bcastRefresh" : " bcastReuse")
            let out = dit!.forward(layout: layout, plan: plan, textStates: refined,
                                   videoRows: videoIn, audioRows: audioX, rope: rope,
                                   sigmaV: sigma, shiftV: sv, shiftA: sa,
                                   attnBcast: pab1, attnRefresh: attnRefresh)
            h3DumpRowStats("step\(i)_out_video", out.video, limitRows: 64)
            h3DumpRowStats("step\(i)_out_audio", out.audio, limitRows: 64)
            // Euler：x += out.video·dsigma（turbo 音频用精确 Δσa/slope，
            // 因为 out.audio 已携带 dσa/dσv 缩放，见 Zig audioStepFactor）
            videoX = videoX + out.video * H3TensorOps.scalarLike(Float(dsigma), out.video)
            let slopeAF = timeShiftSlope(sigma, fromShift: sv, toShift: sa)
            let da = (timeShiftSigma(sigmas[i + 1], fromShift: sv, toShift: sa)
                   - timeShiftSigma(sigma, fromShift: sv, toShift: sa)) / slopeAF
            audioX = audioX + out.audio * H3TensorOps.scalarLike(Float(da), out.audio)
            MLX.eval(videoX, audioX)
            log("step \(i + 1)/\(stage1SegmentCount) sigma \(String(format: "%.4f", sigma)) dsigma \(String(format: "%.4f", dsigma)) [\(gateMode)\(pabMark)]（\(Int(-Date().timeIntervalSince(stepT)))s）")
        }
        let pvMean: Float = videoX.mean().item()
        let pvVar: Float = ((videoX * videoX).mean() - videoX.mean() * videoX.mean()).item()
        log("STAT post videoX mean=\(pvMean) std=\(pvVar.squareRoot())")

        // 诊断 dump（NA_DUMP_LATENT=1）：采样后 latent 与 cond rows 原始 float32
        if ProcessInfo.processInfo.environment["NA_DUMP_LATENT"] == "1" {
            let vx = videoX.asType(.float32)
            let cr = condRows.asType(.float32)
            MLX.eval(vx, cr)
            let vxF = vx.asArray(Float.self)
            let crF = cr.asArray(Float.self)
            let vxData = Data(bytes: vxF, count: vxF.count * MemoryLayout<Float>.size)
            let crData = Data(bytes: crF, count: crF.count * MemoryLayout<Float>.size)
            try vxData.write(to: URL(fileURLWithPath: "/tmp/h3_videox_raw.bin"))
            try crData.write(to: URL(fileURLWithPath: "/tmp/h3_condrows_raw.bin"))
            log("DIAG dumped videoX [\(videoX.shape)] \(vxF.count) floats → /tmp/h3_videox_raw.bin, condRows \(crF.count) floats → /tmp/h3_condrows_raw.bin")
        }

        log("采样完成")

        // ── 4.5 Stage2（可选）：H3 latent 空间放大 + 加噪 refine（官方两段式）──
        // stage1 已跑到 σ=0 得到 clean 行 latent；本段在 latent 域内做
        // 几何空间×scale → 按 refineSigmas 起点加噪 → 低步 Euler 精修，
        // 全程不落地像素，最终 zForDecode 直接喂 VAE decode（行数按放大网格重建）。
        var zForDecode: MLXArray
        if let s2cfg = stage2, s2cfg.scale > 1 {
            let sc = s2cfg.scale
            guard sc == 2 else {
                throw NSError(domain: "H3FL2VA", code: 12,
                              userInfo: [NSLocalizedDescriptionKey: "Stage2 目前仅支持 scale=2"])
            }
            let fRows = gridH * gridW                  // 低清每帧行数（cond 单帧 = fRows）
            let vpatch = videoX.shape[1]
            let H2 = latH * sc
            let W2 = latW * sc
            let gH2 = gridH * sc
            let gW2 = gridW * sc
            let nVideoRows2 = Int(latentT) * gH2 * gW2
            log("Stage2 开始：latent \(latC)×\(latentT)×\(latH)×\(latW) → ×\(sc) → \(latC)×\(latentT)×\(H2)×\(W2)，refine \(s2cfg.refineSigmas.count - 1) 步 sigmas \(s2cfg.refineSigmas)")

            // 0/①/② 几何双线性放大（b 基线同族：trilinear 空间 latent ×2，不加载学习型权重）
            let videoX2: MLXArray
            let condRows2: MLXArray
            do {
                // ① video 行 → zlat → 几何 ×2 放大 → 新网格行
                let cleanVLat = H3TensorOps.unpatchifyVideo(videoX, gridH: gridH, gridW: gridW) // [T,h,w,C]
                var zBig = cleanVLat.transposed(3, 0, 1, 2).reshaped([1, latC, Int(latentT), latH, latW]) // [1,C,T,h,w]
                MLX.eval(zBig)
                zBig = h3SpatialUpsample2x5D(zBig)                                           // [1,C,T,2h,2w]
                let bigVLat = zBig.reshaped([latC, Int(latentT), H2, W2]).transposed(1, 2, 3, 0)  // [T,H2,W2,C]
                videoX2 = H3TensorOps.patchifyVideo(bigVLat, gridH: gH2, gridW: gW2)          // [T*gH2*gW2, C*4]
                MLX.eval(videoX2)
                log("Stage2 视频 latent 放大完成：rows \(videoX.shape) → \(videoX2.shape)")

                // ② cond rows（2 个 keyframe 单帧 latent）各自放大 → 新网格行
                func upCondRows(_ rows: MLXArray) -> MLXArray {
                    let l = H3TensorOps.unpatchifyVideo(rows, gridH: gridH, gridW: gridW) // [1,h,w,C]
                    var zc = l.transposed(3, 0, 1, 2).reshaped([1, latC, 1, latH, latW])   // [1,C,1,h,w]
                    zc = h3SpatialUpsample2x5D(zc)                                         // [1,C,1,2h,2w]
                    let l2 = zc.reshaped([latC, 1, H2, W2]).transposed(1, 2, 3, 0)         // [1,H2,W2,C]
                    return H3TensorOps.patchifyVideo(l2, gridH: gH2, gridW: gW2)
                }
                let condA = condRows[0 ..< fRows, 0 ..< vpatch]
                let condB = condRows[fRows ..< (2 * fRows), 0 ..< vpatch]
                condRows2 = concatenated([upCondRows(condA), upCondRows(condB)], axis: 0)
                MLX.eval(condRows2)
                log("Stage2 cond rows 放大完成：\(condRows2.shape)（行数×\(sc * sc)）")
            }

            // ③ 重建 layout / rope / AdaLN 表（音频 T 不变，仅空间网格变化）
            let layout2 = PackedLayout(textLen: UInt32(textHidden.shape[0]),
                                       latentT: latentT, latentH: UInt32(H2), latentW: UInt32(W2),
                                       audioT: audioT,
                                       keyframes: [.first, .last],
                                       frameCount: frameCount)
            let sigmas2 = s2cfg.refineSigmas
            guard sigmas2.count >= 2, sigmas2.last! == 0.0 else {
                throw NSError(domain: "H3FL2VA", code: 13,
                              userInfo: [NSLocalizedDescriptionKey: "Stage2 refineSigmas 必须 ≥2 项且末项为 0"])
            }
            let ts2 = collectScheduleTs(layout: layout2, sigmas: sigmas2,
                                        shiftV: sv, shiftA: sa, aug: CondNoiseAug())
            dit!.precomputeAdaln(ts: ts2)
            let rope2 = buildRope(layout: layout2, invFreq: dit!.invFreq,
                                  dtype: PrecisionPolicy.defaultMainDType)

            // ④ 加噪：clean 行 → σ0 起点（flow：x = σ0·ε + (1-σ0)·x_clean，ε 派生种子）
            let rSeed = s2cfg.refineSeed ?? (seed &+ 777)
            let s0 = sigmas2[0]
            let epsV = MLXRandom.normal([nVideoRows2, vpatch], key: MLXRandom.key(rSeed)).asType(videoX2.dtype)
            var vx2 = videoX2 * H3TensorOps.scalarLike(Float(1.0 - s0), videoX2)
            vx2 = vx2 + epsV * H3TensorOps.scalarLike(Float(s0), videoX2)
            let sa0 = timeShiftSigma(s0, fromShift: sv, toShift: sa)
            let epsA = MLXRandom.normal(audioX.shape, key: MLXRandom.key(rSeed &+ 1)).asType(audioX.dtype)
            var ax2 = audioX * H3TensorOps.scalarLike(Float(1.0 - sa0), audioX)
            ax2 = ax2 + epsA * H3TensorOps.scalarLike(Float(sa0), audioX)
            MLX.eval(vx2, ax2)
            log("Stage2 加噪完成：video σ0=\(s0)（audio σ0=\(sa0)），seed=\(rSeed)")

            // ⑤ refine 低步 Euler（循环体与 stage1 一致，σ 走 refineSigmas；同样全程接 PAB，
            // refine 通常仅 3~4 步，调度自然几乎全 refresh——无收益也无画质风险）
            let pab2: H3AttnBroadcast? = pabK > 1 ? H3AttnBroadcast(count: dit!.blocks.count) : nil
            for i in 0 ..< (sigmas2.count - 1) {
                let stepT = Date()
                let sigma = sigmas2[i]
                let dsigma = sigmas2[i + 1] - sigmas2[i]
                let plan = buildTimestepPlanGlobal(layout: layout2, sigmaV: sigma, shiftV: sv, shiftA: sa,
                                                   aug: CondNoiseAug(), globalTs: ts2)
                let videoIn = concatenated([condRows2, vx2], axis: 0)
                let attnRefresh2 = pab2 == nil ? false : attnBroadcastRefresh(i, steps: UInt32(sigmas2.count - 1), k: pabK)
                let pabMark2 = pab2 == nil ? "" : (attnRefresh2 ? " bcastRefresh" : " bcastReuse")
                let out = dit!.forward(layout: layout2, plan: plan, textStates: refined,
                                       videoRows: videoIn, audioRows: ax2, rope: rope2,
                                       sigmaV: sigma, shiftV: sv, shiftA: sa,
                                       attnBcast: pab2, attnRefresh: attnRefresh2)
                vx2 = vx2 + out.video * H3TensorOps.scalarLike(Float(dsigma), out.video)
                let slopeAF = timeShiftSlope(sigma, fromShift: sv, toShift: sa)
                let da = (timeShiftSigma(sigmas2[i + 1], fromShift: sv, toShift: sa)
                       - timeShiftSigma(sigma, fromShift: sv, toShift: sa)) / slopeAF
                ax2 = ax2 + out.audio * H3TensorOps.scalarLike(Float(da), out.audio)
                MLX.eval(vx2, ax2)
                log("Stage2 step \(i + 1)/\(sigmas2.count - 1) sigma \(String(format: "%.4f", sigma)) dsigma \(String(format: "%.4f", dsigma)) [bcast\(pabMark2 == "" ? "off" : pabMark2)]（\(Int(-Date().timeIntervalSince(stepT)))s）")
            }
            audioX = ax2

            // ⑥ refine 后的行 → zlat [1,C,T,H2,W2]（decode 段宽高自动取新尺寸）
            let s2Mean: Float = vx2.mean().item()
            let s2Var: Float = ((vx2 * vx2).mean().asType(.float32) - vx2.mean() * vx2.mean()).item(Float.self)
            log("Stage2 refine 后 STAT：videoX mean=\(s2Mean) std=\(s2Var.squareRoot())")
            let finLat = H3TensorOps.unpatchifyVideo(vx2, gridH: gH2, gridW: gW2) // [T,H2,W2,C]
            zForDecode = finLat.transposed(3, 0, 1, 2).reshaped([1, latC, Int(latentT), H2, W2])
            MLX.eval(zForDecode)
            log("Stage2 完成：zForDecode \(zForDecode.shape)（宽高 ×\(sc)）")
        } else {
            // 原单遍路径：行形式直接 reshape 成 zlat [1,C,T,H,W] 供 VAE 解码
            let lh2 = latH / 2
            let lw2 = latW / 2
            let v6 = videoX.reshaped([Int(latentT), lh2, lw2, latC, 2, 2])
            let vp = v6.transposed(3, 0, 1, 4, 2, 5)
            zForDecode = vp.reshaped([1, latC, Int(latentT), latH, latW])
        }

        log("DiT 释放")
        dit = nil
        ditWeights = nil
        MLX.Memory.clearCache()

        // ── 5. VAE 解码 + 写无声 mp4（同一池内完成：像素大数组写完立即释放，再进音频段） ──
        var decWeights: H3Weights? = try H3Weights(url: wURL("video_vae.safetensors"))
        var decoder: H3VAE? = try H3VAE.load(decWeights!)
        var fCount = 0
        var h = 0
        var w = 0
        let silentPath = proResOutput ? (outPath + ".silent.mov") : (outPath + ".silent.mp4")
        try autoreleasepool {
            let pixels = decoder!.decode(zForDecode)
            MLX.eval(pixels)
            log("VAE 解码完成：\(pixels.shape)")
            fCount = pixels.shape[2]
            h = pixels.shape[3]
            w = pixels.shape[4]
            // 内存直通（方案 C）：stage1 像素帧不落盘，由 bridge 持有供像素桥直接 VAE 编码；
            // 落盘仅为“给用户看”的 h264 最高质量预览（writeMp4 h264MaxQuality）。
            if let bridge = stage2MemBridge {
                bridge.pixels = pixels
                bridge.width = w
                bridge.height = h
                bridge.frameCount = fCount
                bridge.fps = Int(H3Const.fps)
            }
            // 先写纯视频轨（writeMp4 本身不含音频，此处不动视频编码路径），
            // 随后单独解码 32kHz 立体声音频（audio_vae vocoder）并混入 outPath。
            try writeMp4(frameCount: fCount, width: w, height: h, fps: Int(H3Const.fps), to: silentPath, proRes: proResOutput, h264MaxQuality: !proResOutput) { f, base, bytesPerRow, isV210 in
                let frameArr = pixels[0 ..< 1, 0 ..< 3, f ..< (f + 1), 0 ..< h, 0 ..< w]
                let squeezed = frameArr.reshaped([3, h, w])
                // isV210 由 writeMp4 按 proRes 传入：ProRes → 10bit v210；h264 预览 → 原 8bit 32ARGB
                return fillFrameForPixelBuffer(squeezed, width: w, height: h, base: base, bytesPerRow: bytesPerRow, isV210: isV210)
            }
        }
        decoder = nil
        decWeights = nil
        MLX.Memory.clearCache()
        log("无声 mp4 已写出：\(silentPath)（\(fCount) 帧 \(w)×\(h)），VAE 解码器与像素数组已释放")

        // ── 5.1 音频 vocoder 解码 + 混入音轨 ──
        // 对齐 minimax_h3_audio.zig：audioRows [2T,32] → audioRowsToLatent
        // → [1,32,2,T] → decode（BigVGAN）→ 波形 [S,L] → interleave WAV → mux。
        do {
            let aT = Int(audioT)
            log("音频解码开始：latent rows \(audioX.shape)，audioT=\(aT) ...")
            let aLatent = try H3AudioVAE.audioRowsToLatent(audioX, audioT: aT)   // [1,32,2,T]
            let tA = Date()
            let audioVAE = try H3AudioVAE.load(url: wURL("audio_vae.safetensors"))
            let wave = try audioVAE.decode(aLatent)                              // [S,L] f32 [-1,1]
            let aSec = Int(Date().timeIntervalSince(tA))
            let frames = wave.shape[1]
            log("音频解码完成（\(aSec)s）：wave \(wave.shape)（\(Float(frames) / Float(H3AudioConst.sampleRate))s @32kHz）")

            // [S, L] → [L, S] interleaved f32 → wav（32kHz 立体声，RIFF f32）
            let inter = wave.transposed(1, 0).contiguous()
            let pcm = inter.asArray(Float.self)
            let pcmMean = pcm.reduce(0, +) / Float(max(pcm.count, 1))
            let pcmMax = pcm.map { abs($0) }.max() ?? 0
            log("   波形 mean=\(pcmMean) maxAbs=\(pcmMax) NaN=\(pcm.contains { !$0.isFinite })")
            let wavPath = outPath + ".wav"
            writeWav(pcm, sampleRate: H3AudioConst.sampleRate,
                     channels: H3AudioConst.stereoChannels, to: wavPath)
            log("wav 已写出：\(wavPath)")
            try muxAudio(videoPath: silentPath, wavPath: wavPath, to: outPath, proRes: proResOutput)
            log("音轨已混入：\(outPath)")
            try? FileManager.default.removeItem(atPath: wavPath)
            try? FileManager.default.removeItem(atPath: silentPath)
        } catch {
            // 音频失败不丢视频：无声 mp4 兜底保留为 outPath
            log("⚠️ 音频解码/混流失败（\(error.localizedDescription)），保留无声 mp4")
            if FileManager.default.fileExists(atPath: outPath) {
                try? FileManager.default.removeItem(atPath: outPath)
            }
            try? FileManager.default.moveItem(atPath: silentPath, toPath: outPath)
        }
        // 总耗时：分钟进制度（≥60s 显示 Xm Ys，否则仅 Xs）
        let totalSec = Int(Date().timeIntervalSince(t0))
        let totalStr = totalSec >= 60 ? "\(totalSec / 60)m \(totalSec % 60)s" : "\(totalSec)s"
        log("视频写出完成：\(outPath)（\(fCount) 帧 \(w)×\(h)，总耗时 \(totalStr)）")
        return outPath
    }
}

/// Stage2 空间几何放大：latent [1,C,T,H,W]（H/W 偶）→ [1,C,T,2H,2W]。
/// 双线性（align_corners=false 近似，与 b 基线产物同族），纯 slice/加权实现，
/// 不引入任何学习型权重。偶数输出 = 0.75·x[i]+0.25·x[i−1]，
/// 奇数输出 = 0.75·x[i]+0.25·x[i+1]（边界复制）。
private func h3SpatialUpsample2x5D(_ z: MLXArray) -> MLXArray {
    let c = z.shape[1]
    let t = z.shape[2]
    precondition(c >= 1 && t >= 1)

    // 对最后维（长度 m）做 align_corners=false 式 ×2：返回 [1, c, t, hh, m*2]
    func lin2xLast(_ x: MLXArray) -> MLXArray {
        let hh = x.shape[3]
        let m = x.shape[4]
        precondition(m >= 2)
        let xL = x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< m]
        let xLeft = concatenated([x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< 1],
                                  x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< (m - 1)]], axis: 4)
        let xRight = concatenated([x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 1 ..< m],
                                   x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, (m - 1) ..< m]], axis: 4)
        let even = xL * H3TensorOps.scalarLike(0.75, x) + xLeft * H3TensorOps.scalarLike(0.25, x)
        let odd = xL * H3TensorOps.scalarLike(0.75, x) + xRight * H3TensorOps.scalarLike(0.25, x)
        let ee = even.reshaped([1, c, t, hh, m, 1])
        let oo = odd.reshaped([1, c, t, hh, m, 1])
        return concatenated([ee, oo], axis: 5).reshaped([1, c, t, hh, m * 2])
    }

    // W 方向 ×2（最后维 = w）
    let xW = lin2xLast(z)                                        // [1,C,T,h,2w]
    // H 方向 ×2（交换最后两维后复用末轴插值）
    let xT = xW.transposed(0, 1, 2, 4, 3)                        // [1,C,T,2w,h]
    let xH = lin2xLast(xT)                                       // [1,C,T,2w,2h]
    return xH.transposed(0, 1, 2, 4, 3)                          // [1,C,T,2h,2w]
}
