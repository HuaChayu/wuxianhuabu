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

// MARK: - 管线日志（print + 模型监控浮层 + 落盘文件 三通道）

/// 管线关键日志：同时输出到 Xcode 控制台、模型监控浮层（需打开右下角 sparkles 开关）与日志文件。
/// 日志文件路径：output/视频/pipeline.log
func pipelineLog(_ message: String) {
    print(message)
    monitorLog(message)
    let logURL = outputVideoDirURL.appendingPathComponent("pipeline.log")
    let line = "[\(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium))] \(message)\n"
    if let h = try? FileHandle(forWritingTo: logURL) {
        h.seekToEndOfFile()
        h.write(line.data(using: .utf8)!)
        try? h.close()
    } else {
        try? line.data(using: .utf8)?.write(to: logURL)
    }
}

// MARK: - 内存自适应策略

/// 当前生成任务类型（决定阶段→模块映射：视频采样用 DiT，图像采样用 HiDream）
enum GenerationTask {
    case image   // HiDream 图像生成
    case video   // LTX 视频生成
}

/// 生成管线阶段：ensureCapacity 保护"当前阶段 + 下一阶段"需要的模块，
/// 只把之前阶段/之前任务用过、当前与下一阶段都不需要的模块列为卸载候选。
enum GenerationStage {
    case encoding   // 视频：文本编码（Gemma + connector）
    case sampling   // 视频：DiT 采样；图像：HiDream 采样
    case vae        // 视频：VAE 解码/升频（临时加载模块，无持久缓存候选）
}

/// 可卸载模型模块（持久缓存粒度）与体积：
/// HiDream 扩散头(72M) 已随主干灌入同一 backbone 实例，无独立缓存，不单列；
/// TextEncoderCache 内部可拆 connector / gemma，分开排序（体积从小到大先卸小的）。
enum MemoryModule: String, CaseIterable {
    case connector   // LTX connector，5.9G
    case gemma       // Gemma4 文本编码器（tokenizer 体积小随留），6.2G
    case dit         // LTX DiT 权重，11G（与编译图强绑定）
    case hiDream     // HiDream backbone（含扩散头 72M），16G

    /// 所属模型域：true = LTX 视频侧，false = HiDream 图像侧（异模型优先层级依据）
    var isLTX: Bool {
        switch self {
        case .connector, .gemma, .dit: return true
        case .hiDream: return false
        }
    }

    var sizeBytes: Int {
        switch self {
        case .connector: return 5_900_000_000
        case .gemma:     return 6_200_000_000
        case .dit:       return 11_000_000_000
        case .hiDream:   return 16_000_000_000
        }
    }

    /// 卸载动作（DiT 与编译图强绑定：先清 CompiledForwardCache 再清 DiTModelCache）
    func unload() {
        switch self {
        case .connector: TextEncoderCache.shared.clearConnector()
        case .gemma:     TextEncoderCache.shared.clearGemma()
        case .dit:
            ltxCompiledForward.clear()
            DiTModelCache.shared.clear()
        case .hiDream:   HiDreamModelCache.shared.clear()
        }
    }
}

/// 统一内存水位策略：所有机器（不区分内存大小）在 MLX active 内存逼近物理内存 90% 时，
/// 保护"当前阶段 + 下一阶段"所需模块，只卸之前用过的模块，且按权重大小从小到大卸，
/// 尽量保留大权重，减少连续任务的整体重载时间。
enum MemoryPolicy {
    /// 互斥规则：物理内存 ≤64GB 时，视频/图像两套模型互斥驻留（中途用完即卸时异模型一并卸）；
    /// >64GB 不互斥，异模型保留缓存供快速切换。
    static var mutualExclusive: Bool { SystemMemory.physicalMemoryGB <= 64 }

    static func report() {
        pipelineLog("内存策略：统一动态卸载（active ≥ 物理内存 \(Int(SystemMemory.memoryThresholdRatio * 100))% 触发），物理内存 \(SystemMemory.physicalMemoryGB)GB")
    }

    /// 阶段 → 需要的模块集合（保护集）
    private static func protectedModules(for task: GenerationTask, current: GenerationStage, next: GenerationStage?) -> Set<MemoryModule> {
        var protect: Set<MemoryModule> = []
        func add(_ stage: GenerationStage?) {
            guard let stage else { return }
            switch stage {
            case .encoding:
                protect.insert(.connector)
                protect.insert(.gemma)
            case .sampling:
                switch task {
                case .video: protect.insert(.dit)
                case .image: protect.insert(.hiDream)
                }
            case .vae:
                break   // VAE/升频器临时加载，不在持久缓存候选内
            }
        }
        add(current)
        add(next)
        return protect
    }

    /// 动态安全预留：按任务尺寸与时长计算（小任务更激进、大任务更保守）。
    /// 面积 = width × height / 1_000_000（单位：Mp，百万像素）。
    /// 图像任务：预留 = max(2G, 2_000_000_000 + 1_000_000_000 × 面积)
    ///   （2G 基准 + 1G/Mp；512² ≈2.26G，1024² ≈3.05G）
    /// 视频任务：预留 = max(2G, 1_500_000_000 + 1_500_000_000 × 面积 × (frames / 120))
    ///   （1.5G 基准 + 1.5G/Mp × 时长倍率；5s@24fps=120 帧倍率 1.0，10s=240 帧倍率 2.0；
    ///     大分辨率 + 长时长 更保守，小分辨率 + 短视频 更激进）
    static func safetyBuffer(for task: GenerationTask, width: Int, height: Int, frames: Int) -> Int {
        let area = Double(width * height) / 1_000_000
        switch task {
        case .image:
            return max(2_000_000_000, 2_000_000_000 + Int(1_000_000_000 * area))
        case .video:
            return max(2_000_000_000, 1_500_000_000 + Int(1_500_000_000 * area * (Double(frames) / 120.0)))
        }
    }

    /// 加载前评估水位：在各模型加载点【加载前】调用。
    /// 判定/停止条件：active + 本次预估载入体积 + 本次动态预留 ≤ 物理内存×0.9 时不卸载；
    /// 超过才卸载候选，每卸一个 clearCache 并重新 snapshot，直到满足条件才停。
    /// newSizeBytes=0（模型已在缓存、无需重载）时永不触发卸载，直接复用。
    /// 安全预留由 safetyBuffer(for:width:height:frames:) 按任务尺寸/时长动态计算：
    /// 图像按尺寸、视频按尺寸+时长，小任务更激进、大任务更保守。
    /// 卸载候选分两层，两层之间严格先后：
    ///   第 1 层【异模型优先】：先卸与当前任务不同模型的模块
    ///     （视频任务 → 先卸 HiDream 图像侧；图像任务 → 先卸 LTX 视频侧 connector/gemma/dit），
    ///     异模型内部按体积从小到大卸；
    ///   第 2 层【同模型内部】：第 1 层卸完仍不满足才进入，
    ///     卸当前任务模型内、不在保护集的模块（已经用不上的），按体积从小到大卸。
    /// 体积顺序：connector 5.9G → gemma 6.2G → DiT 11G → HiDream 16G（小/重载快的先卸）。
    /// 前两层卸完仍不满足时按阶梯降级：
    ///   阶梯1：让出本次动态预留（active+newSize ≤ 0.9×物理内存）放行，不再卸新模块；
    ///   阶梯2：最后防线（active+newSize ≤ 物理内存+8G）放行，允许逼近硬上限（少量 swap）；
    ///   阶梯3：active+newSize > 物理内存+8G → 模型本身装不下，返回 false 拦截加载，禁止硬加载超上限。
    /// - Parameters:
    ///   - width: 本次生成任务宽度（像素），用于动态预留计算
    ///   - height: 本次生成任务高度（像素），用于动态预留计算
    ///   - frames: 视频任务真实帧数（5s=120、10s=240，传 duration.numFrames）；图像任务不依赖帧数，可传默认 120
    /// - Returns: true=放行加载；false=内存不足拦截（调用侧应向用户输出清晰提示并中止加载）。
    @discardableResult
    static func ensureCapacity(for newSizeBytes: Int, task: GenerationTask, current: GenerationStage, next: GenerationStage? = nil, width: Int, height: Int, frames: Int = 120) -> Bool {
        let safetyBuffer = safetyBuffer(for: task, width: width, height: height, frames: frames)
        let threshold = SystemMemory.activeThresholdBytes
        let projected = SystemMemory.activeBytes + newSizeBytes + safetyBuffer
        // 压力分高（压缩/交换虚高）时同样进入卸载流程，不只看 active 水位
        let highPressure = SystemMemory.pressureScore >= SystemMemory.pressureTrigger
        guard newSizeBytes > 0, projected > threshold || highPressure else { return true }

        let protect = protectedModules(for: task, current: current, next: next)
        let candidates = MemoryModule.allCases.filter { !protect.contains($0) }
        // 异模型内部仍按体积升序
        let ltxSide = candidates.filter { $0.isLTX }.sorted { $0.sizeBytes < $1.sizeBytes }
        let hiDreamSide = candidates.filter { !$0.isLTX }.sorted { $0.sizeBytes < $1.sizeBytes }
        // 两层严格先后：视频任务第 1 层卸 HiDream 侧、第 2 层卸 LTX 侧；图像任务相反
        let layer1: [MemoryModule]
        let layer2: [MemoryModule]
        switch task {
        case .video:
            layer1 = hiDreamSide
            layer2 = ltxSide
        case .image:
            layer1 = ltxSide
            layer2 = hiDreamSide
        }

        // 每卸一个后重新评估：active + newSize + 动态预留 ≤ 阈值即满足，停止卸载
        func satisfied() -> Bool {
            SystemMemory.activeBytes + newSizeBytes + safetyBuffer <= threshold
        }

        let nextName = next.map { String(describing: $0) } ?? "无"
        pipelineLog("⚠️ 加载前评估：active=\(SystemMemory.activeBytes) + 载入=\(newSizeBytes) + 动态预留=\(safetyBuffer) > 阈值=\(threshold)（\(Int(SystemMemory.memoryThresholdRatio * 100))% 物理内存），保护 当前=\(current) 下一=\(nextName)，第1层异模型优先（\(task == .video ? "HiDream" : "LTX视频侧")）卸载...")

        // 第 1 层：异模型优先
        for m in layer1 {
            if satisfied() { break }
            m.unload()
            MLX.Memory.clearCache()
            pipelineLog("✅ [第1层·异模型优先] 卸载候选（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }
        // 第 2 层：同模型内部（第 1 层卸完仍不满足才进入）
        for m in layer2 {
            if satisfied() { break }
            m.unload()
            MLX.Memory.clearCache()
            pipelineLog("✅ [第2层·同模型内部] 卸载候选（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }

        // ── 兜底阶梯：前两层卸完仍不满足时按顺序降级 ──
        let noBuffer = SystemMemory.activeBytes + newSizeBytes   // 让出本次动态预留后的净需求
        let physical = SystemMemory.physicalMemoryBytes
        let hardLimit = physical + 8 * (1 << 30)    // 兜底硬上限：物理内存 + 8 GiB（任何设备固定 +8G）
        if noBuffer <= threshold {
            // 阶梯1：让出本次动态预留（不再卸新模块），直接放行
            pipelineLog("⚠️ 候选已全部卸载仍超 90%+动态预留，让出本次动态预留 \(safetyBuffer) 字节放行（active=\(SystemMemory.activeBytes) + 载入=\(newSizeBytes) ≤ 阈值=\(threshold)）")
            return true
        }
        if noBuffer <= hardLimit {
            // 阶梯2：最后防线，允许逼近 物理内存+8G（允许少量 swap）
            pipelineLog("⚠️ 候选已全部卸载且 90% 阈值仍不满足，最后防线放行（active=\(SystemMemory.activeBytes) + 载入=\(newSizeBytes) ≤ 物理内存+8G=\(hardLimit)），允许逼近硬上限")
            return true
        }
        // 阶梯3：模型本身装不下 → 拦截加载，禁止硬加载超上限
        pipelineLog("❌ 内存不足，无法加载该模型（当前占用 \(SystemMemory.activeBytes) 字节，需 \(newSizeBytes) 字节，合计 \(noBuffer) > 硬上限 \(hardLimit) 字节），已拦截加载")
        return false
    }

    /// 生成结束后按水位策略卸载模型缓存（保留原调用点）。
    /// 顺序重要：先清 CompiledForwardCache——其编译闭包强捕获 DiT 权重，
    /// 不清它，DiTModelCache 里的引用即使置 nil 权重也释放不掉。
    /// 触发条件：active 水位 ≥ 阈值，或内存压力分 ≥ pressureTrigger（压缩/交换虚高时兜底）。
    static func unloadModelsIfNeeded() {
        let threshold = SystemMemory.activeThresholdBytes
        let ps = SystemMemory.pressureScore
        guard SystemMemory.activeBytes >= threshold || ps >= SystemMemory.pressureTrigger else { return }
        pipelineLog("⚠️ 内存仍逼近上限（active=\(SystemMemory.activeBytes) 阈值=\(threshold)，压力分 \(String(format: "%.1f", ps))/\(SystemMemory.pressureTrigger)），动态卸载模型缓存...")
        ltxCompiledForward.clear()
        DiTModelCache.shared.clear()
        TextEncoderCache.shared.clear()
        MLX.Memory.clearCache()
        pipelineLog("✅ 模型缓存已卸载，常驻内存回落")
    }

    /// 生成中途用完即卸：在阶段切换点（条件嵌入完成→采样前、VAE 解码前）调用。
    /// 不看水位、不看队列——只要"当前+下一阶段保护集外"的模块对本次任务已无用，
    /// 立即卸载让位（Gemma/Connector 采样前丢、DiT 解码前丢，交给页缓存兜底重载）。
    /// 互斥规则：内存 ≤64GB 时视频/图像两套模型互斥，异模型一并卸（任务切换不残留）；
    /// >64GB 不互斥，只卸本任务链的模块，异模型保留缓存供快速切换。
    static func unloadIfNeededMidway(task: GenerationTask, current: GenerationStage, next: GenerationStage? = nil) {
        let protect = protectedModules(for: task, current: current, next: next)
        var candidates = MemoryModule.allCases.filter { !protect.contains($0) }
        if !mutualExclusive {
            // 不互斥（>64GB）：只卸本任务链（LTX 视频侧 / HiDream 图像侧）的模块，异模型保留
            candidates = candidates.filter { $0.isLTX == (task == .video) }
        }
        let ordered = candidates.sorted { $0.sizeBytes < $1.sizeBytes }
        guard !ordered.isEmpty else { return }
        let nextName = next.map { String(describing: $0) } ?? "无"
        pipelineLog("⚠️ 中途用完即卸（互斥=\(mutualExclusive)）：保护 当前=\(current) 下一=\(nextName)，卸载候选：\(ordered.map { $0.rawValue }.joined(separator: "、"))")
        for m in ordered {
            m.unload()
            MLX.Memory.clearCache()
            pipelineLog("✅ [用完即卸] 卸载（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }
    }
}

// MARK: - 主入口

/// 端到端生成：文本 → 去噪 latent（DiT）→ VAE 解码 → 音轨合成。
/// - Parameters:
///   - prompt: 视频提示词；为空时使用 GenConfig 默认提示词。
///   - audioPath: 可选音频条件（音频节点连入时），16kHz 立体声 wav 路径。
///   - imagePaths: 可选图片条件（图片节点连入时，最多 2 张），首张作 I2V 首帧。
/// - Returns: 生成视频 mp4 的绝对路径；任一步失败返回 nil。
/// 内存策略：cacheLimit 按物理内存自适应，阶段间 clearCache（与 ltx-test 验证一致）。
public func runVideoPipeline(prompt: String = "", audioPath: String? = nil, imagePaths: [String] = [], width: Int? = nil, height: Int? = nil, duration: VideoDuration = .fiveSeconds, stage2Refine: Bool = true, isCancelled: @escaping () -> Bool = { false }) async -> String? {
    memProfilePoint("管线开始 基线")
    let pipeT0 = Date()
    MemoryPolicy.report()
    MLX.Memory.cacheLimit = SystemMemory.bufferCacheLimit
    // 任务开始：互斥规则（≤64GB）——生成视频先卸对方（HiDream），确保本次任务独占内存；
    // 本任务链（LTX 侧）受保护集保护不被误卸；>64GB 不互斥，异模型保留
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .encoding, next: .sampling)
    await generateVideoTest(prompt: prompt, audioPath: audioPath, imagePaths: imagePaths, width: width, height: height, duration: duration, stage2Refine: stage2Refine, isCancelled: isCancelled)
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
    let pipeSec = Int(Date().timeIntervalSince(pipeT0))
    pipelineLog("✅ 端到端完成，总耗时 \(pipeSec / 60)分\(String(format: "%02d", pipeSec % 60))秒：active=\(gb(snap.activeMemory)) cache=\(gb(snap.cacheMemory)) peak=\(gb(snap.peakMemory))")
    memProfileReport()   // 一次性输出全部内存打点
    return outPath
}

// MARK: - 视频生成尺寸对齐公共函数

/// 视频生成 latent 尺寸对齐倍数：LTX-2.5 要求宽高为 32 的倍数。
func videoLatentAlignMultiple() -> Int { 32 }

/// 视频生成目标尺寸（动态表）：比例 × 清晰度档位 → 宽高对齐 64 倍数。
/// 空间升频器 ×2 已接入 VAE 解码前：生成尺寸 = 目标 / 2，除 2 后须仍为 32 倍数
/// （latent 对齐），故目标尺寸必须以 64 倍数对齐，除 2 后天然满足。
/// - standard：最长边 512 基准，等比缩放后每维取最近 64 倍数
/// - p720 / p1080：短边对齐档位（720→768、1080→1088，向上取 64 倍数），长边按比例取最近 64 倍数
func videoResolution(for ratio: CanvasStore.Ratio, quality: VideoQuality) -> (width: Int, height: Int) {
    let m = videoLatentAlignMultiple() * 2   // 64：目标须 64 倍数，除 2 生成后仍对齐 32
    let horizontal = ratio.width >= ratio.height
    let unit = min(ratio.width, ratio.height)
    let long = max(ratio.width, ratio.height)
    let nearest: (Double) -> Int = { max(m, Int(($0 / Double(m)).rounded()) * m) }
    let upward: (Double) -> Int = { max(m, Int(ceil($0 / Double(m))) * m) }
    var w: Int, h: Int
    switch quality {
    case .standard:
        let longSide = m * 8   // 512
        let shortSide = nearest(Double(longSide) * unit / long)
        w = horizontal ? longSide : shortSide
        h = horizontal ? shortSide : longSide
    case .p720:
        let shortSide = upward(720)
        let longSide = nearest(Double(shortSide) * long / unit)
        w = horizontal ? longSide : shortSide
        h = horizontal ? shortSide : longSide
    case .p1080:
        let shortSide = upward(1080)
        let longSide = nearest(Double(shortSide) * long / unit)
        w = horizontal ? longSide : shortSide
        h = horizontal ? shortSide : longSide
    }
    return (w, h)
}

/// 图片 → 动态表中最接近的比例（按宽高比距离取最近，如 1000×1500 的 2:3 命中 3:4）。
func nearestRatio(for width: Int, height: Int) -> CanvasStore.Ratio {
    guard width > 0, height > 0 else { return .ratio16_9 }
    let imgRatio = Double(width) / Double(height)
    return CanvasStore.Ratio.allCases.min { a, b in
        abs(a.width / a.height - imgRatio) < abs(b.width / b.height - imgRatio)
    } ?? .ratio16_9
}

/// 视频生成目标尺寸：有图按图最接近比例查表；无图按传入比例（nil → 16:9）。
/// 供 UI 收集阶段（startVideoGeneration）与管线兜底共用。
func videoSize(imageWidth: Int? = nil, imageHeight: Int? = nil, ratio: CanvasStore.Ratio? = nil, quality: VideoQuality = .standard) -> (width: Int, height: Int) {
    let r: CanvasStore.Ratio
    if let imageWidth, let imageHeight {
        r = nearestRatio(for: imageWidth, height: imageHeight)
    } else {
        r = ratio ?? .ratio16_9
    }
    return videoResolution(for: r, quality: quality)
}

struct GenConfig {
    var prompt: String = "A majestic golden retriever running through a sunlit meadow, slow motion, cinematic lighting"
    var numFrames: Int = 120         // 全量：5s @ 24fps
    var height: Int = 256         // 生成尺寸 = 目标 / 2（512），升频 ×2 后还原
    var width: Int = 256
    var frameRate: Float = 24.0
    var numSteps: Int = 8         // dev+CFG：参考 stage1 默认 30 步，测试先 20（蒸馏 6 步不够）
    var seed: UInt64 = 42
}

func encodePrompt(_ prompt: String) async -> (video: MLXArray, audio: MLXArray)? {
    let modelDir = "/Users/huachayui/Downloads/ltx2.5/gemma4-12b-ltx-v1"
    let ltxDir = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit"
    let t0 = Date()

    // tokenizer（常驻缓存，换模型目录才重载）
    let tokenizer: any Tokenizer
    if let cached = TextEncoderCache.shared.tokenizer(for: modelDir) {
        tokenizer = cached
        pipelineLog("✅ 复用已加载 tokenizer")
    } else {
        pipelineLog("加载 tokenizer...")
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
    pipelineLog("✅ tokenizer 就绪（\(Int(Date().timeIntervalSince(t0)))s），prompt token 数：\(rawIds.count)")

    // Gemma 12B 文本编码器（常驻缓存，权重文件未变则复用）
    let gemmaPath = "\(modelDir)/model.safetensors"
    let gemmaModel: Gemma4TextEncoder
    if let cached = TextEncoderCache.shared.gemma(for: gemmaPath) {
        gemmaModel = cached
        pipelineLog("✅ 复用已加载 Gemma 权重（跳过加载）")
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
        pipelineLog("✅ Gemma 权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s）")
        TextEncoderCache.shared.storeGemma(m, path: gemmaPath)
        gemmaModel = m
    }

    // LTX 连接器（常驻缓存，权重文件未变则复用）
    let connPath = "\(ltxDir)/connector.safetensors"
    let connector: LTXConnector
    if let cached = TextEncoderCache.shared.connector(for: connPath) {
        connector = cached
        pipelineLog("✅ 复用已加载 connector（跳过加载）")
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
        pipelineLog("✅ 连接器权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s）")
        TextEncoderCache.shared.storeConnector(c, path: connPath)
        connector = c
    }

    let padID: Int32 = 0
    var ids = [Int32](repeating: padID, count: 256)
    let n = min(rawIds.count, 256)
    for i in 0..<n { ids[256 - n + i] = Int32(rawIds[i]) }   // 左 pad
    let idsArr = MLXArray(ids, [1, 256]).asType(.int32)
    pipelineLog("有效 token 数：\(n)（左 pad \(256 - n)）")

    pipelineLog("Gemma4 前向（48 层，约 1-3 分钟）...")
    let t1 = Date()
    let states = gemmaModel.encodeStates(idsArr)
    pipelineLog("✅ 49 态捕获完成，耗时 \(Int(Date().timeIntervalSince(t1)))s")

    pipelineLog("连接器变换 + 投影...")
    let t2 = Date()
    let (videoCond0, audioCond0) = connectorProject(states: states, connector: connector)
    // 后半段（对齐 mlx-serve encodeTextLtx）：投影后再经 video/audio_embeddings_connector
    // 的 128 学习寄存器 + 8 个 gated-attention block 处理，才是 DiT 的文本条件。
    // 缺这步会导致文本条件语义对齐不足，换提示词画面骨架雷同。
    let nValid = n
    let videoCond = connector.video_embeddings_connector(videoCond0, nValid: nValid)
    let audioCond = connector.audio_embeddings_connector(audioCond0, nValid: nValid)
    eval(videoCond, audioCond)
    pipelineLog("✅ 条件嵌入完成（耗时 \(Int(Date().timeIntervalSince(t2)))s）：video \(videoCond.shape)，audio \(audioCond.shape)")
    let vArr = videoCond.asArray(Float.self)
    let aArr = audioCond.asArray(Float.self)
    let vMean = vArr.reduce(0, +) / Float(vArr.count)
    let aMean = aArr.reduce(0, +) / Float(aArr.count)
    let vMax = vArr.map { abs($0) }.max() ?? 0
    let aMax = aArr.map { abs($0) }.max() ?? 0
    pipelineLog("cond 统计：prompt「\(prompt.prefix(40))」video mean=\(vMean) maxAbs=\(vMax) | audio mean=\(aMean) maxAbs=\(aMax)")
    return (videoCond, audioCond)
}

func makeVideoLatentAndPos(g: GenConfig) -> (latent: MLXArray, pos: [Float]) {
    let F = (g.numFrames + 7) / 8
    let H = g.height / 32
    let W = g.width / 32
    MLXRandom.seed(g.seed)
    let noise = MLXRandom.normal([1, F, H, W, 128]).asType(.bfloat16)
    var pos = [Float](repeating: 0, count: F * H * W * 3)
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
    return (noise, pos)
}

func makeAudioLatentAndPos(g: GenConfig) -> (latent: MLXArray, pos: [Float]) {
    let Na = Int((Float(g.numFrames) / g.frameRate * 25.0).rounded())
    MLXRandom.seed(g.seed &+ 1)
    let noise = MLXRandom.normal([1, Na, 128]).asType(.bfloat16)
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
    let n = Int(outBuf.frameLength)
    return Array(UnsafeBufferPointer(start: data, count: n))
}

/// 读取图片文件 → [1,3,1,H,W] f32（RGB，0-1），供 vaeEncodeImage 首帧条件编码。
/// 等比缩放图片到目标宽高（默认 512×512，对齐视频分辨率），不足尺寸中心补零保持宽高比。
/// 目标宽高需为 32 的倍数（LTX-2.5 latent 对齐要求），首帧/尾帧图统一用同一目标尺寸。
func loadImageBCFHW(path: String, width: Int = 512, height: Int = 512) -> MLXArray? {
    guard let img = NSImage(contentsOfFile: path),
          var cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
    // 转 RGB 颜色空间，避免 CMYK/灰度等异常布局
    if cg.colorSpace?.model != .rgb {
        let rep = NSBitmapImageRep(cgImage: cg)
        guard let pngData = rep.representation(using: .png, properties: [:]) else { return nil }
        guard let nsImg = NSImage(data: pngData) else { return nil }
        guard let rgbImg = nsImg.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        cg = rgbImg
    }
    let srcW = cg.width
    let srcH = cg.height
    let scale = min(Float(width) / Float(srcW), Float(height) / Float(srcH))
    let dstW = max(Int((Float(srcW) * scale).rounded()), 1)
    let dstH = max(Int((Float(srcH) * scale).rounded()), 1)
    let offX = (width - dstW) / 2
    let offY = (height - dstH) / 2

    var px = [Float](repeating: 0, count: 1 * 3 * 1 * height * width)
    guard let ctx = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: offX, y: offY, width: dstW, height: dstH))
    guard let data = ctx.data else { return nil }
    let ptr = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
    // 位图上下文内存 row0 = 顶部，直接顺序填充即为正立图；按 BCFHW 平面填充，避免 HWC reshape 布局错位
    let plane = width * height
    for y in 0..<height {
        for x in 0..<width {
            let i = (y * width + x) * 4
            let o = y * width + x
            // VAE 编码器输入要求 [-1,1]（对齐 zig vaeEncode / decodeImageToBCFHW 的 v/127.5-1）
            px[0 * plane + o] = Float(ptr[i + 0]) / 127.5 - 1.0
            px[1 * plane + o] = Float(ptr[i + 1]) / 127.5 - 1.0
            px[2 * plane + o] = Float(ptr[i + 2]) / 127.5 - 1.0
        }
    }
    return MLXArray(px, [1, 3, 1, height, width]).asType(.float32)
}

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
        let mod = fileModTime(path)
        if let d = dit, loadedPath == path, loadedModTime == mod {
            return d
        }
        return nil
    }

    func store(_ d: LTXVideoDiT, path: String) {
        dit = d
        loadedPath = path
        loadedModTime = fileModTime(path)
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

    private func fileModTime(_ path: String) -> Date? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return attrs?[.modificationDate] as? Date
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
        guard let g = gemma, gemmaPath == path, gemmaModTime == fileModTime(path) else { return nil }
        return g
    }

    func storeGemma(_ g: Gemma4TextEncoder, path: String) {
        gemma = g; gemmaPath = path; gemmaModTime = fileModTime(path)
    }

    func connector(for path: String) -> LTXConnector? {
        guard let c = connector, connectorPath == path, connectorModTime == fileModTime(path) else { return nil }
        return c
    }

    func storeConnector(_ c: LTXConnector, path: String) {
        connector = c; connectorPath = path; connectorModTime = fileModTime(path)
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

    private func fileModTime(_ path: String) -> Date? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return attrs?[.modificationDate] as? Date
    }
}

func generateVideoTest(prompt: String = "", audioPath: String? = nil, imagePaths: [String] = [], width: Int? = nil, height: Int? = nil, duration: VideoDuration = .fiveSeconds, stage2Refine: Bool = true, isCancelled: @escaping () -> Bool = { false }) async {
    MonitorCenter.shared.taskStart("视频生成")
    defer { MonitorCenter.shared.taskEnd("视频生成") }
    pipelineLog("\n=== ⑥ 端到端视频生成（文本 → 去噪 latent）===")
    var g = GenConfig()
    if !prompt.isEmpty { g.prompt = prompt }   // UI 传入提示词覆盖默认值
    g.numFrames = duration.numFrames          // 5s → 120 帧 / 10s → 240 帧（@24fps）
    // 尺寸：UI/收集阶段传入的是目标分辨率（64 倍数表）；空间升频器 ×2 已接入
    //       VAE 解码前，生成尺寸 = 目标 / 2（64 倍数除 2 后天然 32 倍数，latent 对齐），
    //       升频后输出恰为目标分辨率。I2V 参考图同样按半尺寸编码，升频后还原。
    if let width, let height {
        let m = videoLatentAlignMultiple()
        g.width = max(m, (width / 2 / m) * m)
        g.height = max(m, (height / 2 / m) * m)
    } else if let first = imagePaths.first,
              let img = NSImage(contentsOfFile: first) {
        let aligned = videoSize(imageWidth: Int(img.size.width), imageHeight: Int(img.size.height), quality: .standard)
        let m = videoLatentAlignMultiple()
        g.width = max(m, (aligned.width / 2 / m) * m)
        g.height = max(m, (aligned.height / 2 / m) * m)
        pipelineLog("✅ 图片驱动：生成尺寸 \(g.width)×\(g.height)（目标 \(aligned.width)×\(aligned.height)，升频×2 后还原）")
    }
    pipelineLog("提示词：\(g.prompt)")
    pipelineLog("配置：\(g.numFrames) 帧 @\(g.frameRate)fps，生成 \(g.height)×\(g.width)（升频×2 → 输出 \(g.height * 2)×\(g.width * 2)），\(g.numSteps) 步，seed=\(g.seed)")

    // 1) 正向文本条件编码（蒸馏单阶段：cfg=1.0 无引导，无需负向 prompt，省一次 Gemma 12B 前向）
    memProfilePoint("文本编码前")
    async let posCond = encodePrompt(g.prompt)
    guard let cond = await posCond else { pipelineLog("❌ 文本编码失败"); return }
    memProfilePoint("文本编码后")
    let condV = cond.video
    let condA = cond.audio
    let negV: MLXArray? = nil     // 无 CFG：不提供负向，cond 直出
    // 2)-4) 加载/复用 DiT 权重 + 噪声 latent + 采样（方案 A：DiT 权重与编译图常驻缓存复用，
    //        不随每次生成重建/释放，避免大图一次性析构造成 app 假死）
    let ditPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/transformer-distilled.safetensors"
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
        pipelineLog("✅ 复用已加载 DiT 权重（跳过加载）")
    } else {
        guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: ditPath)) else {
            pipelineLog("❌ transformer 权重加载失败"); return
        }
        let t3 = Date()
        // 量化参数从 safetensors 文件头动态推断（4bit/8bit 通用，换地址即可；带缓存）
        let q = cachedQuantParams(from: ditPath) ?? (bits: 4, groupSize: 64)
        pipelineLog("构建 LTXVideoDiT（48 层，bits=\(q.bits) groupSize=\(q.groupSize)）...")
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
        pipelineLog("✅ DiT 权重灌入完成（\(Int(Date().timeIntervalSince(t3)))s，\(strippedW.count) 键）")
        memProfilePoint("DiT权重灌入后")
        DiTModelCache.shared.store(d, path: ditPath)
        dit = d
    }
    memProfilePoint("DiT就绪")

    // 3) 噪声 latent + 位置
    let (noiseV, videoPos) = makeVideoLatentAndPos(g: g)
    pipelineLog("video latent \(noiseV.shape)（Nv=\(videoPos.count / 3)）")

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
        let audioVaePath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/audio_vae.safetensors"
        guard let aw = try? MLX.loadArrays(url: URL(fileURLWithPath: audioVaePath)) else {
            pipelineLog("❌ audio_vae 权重加载失败：\(audioVaePath)")
            return
        }
        sharedAudioVAEWeights = aw   // 同轮复用：mp4 音频解码阶段不再重复读盘（102MB）
        // maxTokens：按视频时长帧数换算音频 token 预算（对齐 makeAudioLatentAndPos 的 Na）
        let naBudget = Int((Float(g.numFrames) / g.frameRate * 25.0).rounded())
        frozenAudio = encodeAudioCond(weights: aw, pcm: pcm, maxTokens: naBudget)
        audioPos = makeAudioPositions(count: frozenAudio!.shape[1])
        pipelineLog("  ✅ 音频条件编码完成（\(Int(Date().timeIntervalSince(tA)))s）：latent \(frozenAudio!.shape)，Na=\(audioPos.count)")
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
        let vaeEncPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/vae_encoder.safetensors"
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
            let noiseMid = noiseV[0..., 1 ..< (t - 1), 0..., 0..., 0...]   // [1,T-2,H,W,128]
            let zerosMid = MLXArray.zeros([1, t - 2, h, w, 128]).asType(.bfloat16)
            initVideo = concatenated([refFirst, noiseMid, refTail], axis: 1).asType(.bfloat16)
            cleanV = concatenated([refFirst, zerosMid, refTail], axis: 1).asType(.bfloat16)
            condMask = [Float](repeating: 0, count: hw)
                + [Float](repeating: 1, count: (t - 2) * hw)
                + [Float](repeating: 0, count: hw)
            pipelineLog("🖼️ I2V 首尾帧条件：首帧 \(imagePaths[0])，尾帧 \(imagePaths[1]) → VAE 编码完成（\(Int(Date().timeIntervalSince(tI)))s），mask 前/后各 \(hw) 个 0 钉入两端")
        } else {
            let noiseRest = noiseV[0..., 1..., 0..., 0..., 0...]            // [1,T-1,H,W,128]
            let zerosRest = MLXArray.zeros([1, t - 1, h, w, 128]).asType(.bfloat16)
            initVideo = concatenated([refFirst, noiseRest], axis: 1).asType(.bfloat16)
            cleanV = concatenated([refFirst, zerosRest], axis: 1).asType(.bfloat16)
            condMask = [Float](repeating: 0, count: hw) + [Float](repeating: 1, count: t * hw - hw)
            pipelineLog("🖼️ I2V 首帧条件：\(imagePaths[0])（共 \(imagePaths.count) 张）→ VAE 编码完成（\(Int(Date().timeIntervalSince(tI)))s），首帧 latent \(refLat.shape)")
        }
    }

    // 4) 采样（蒸馏 8 步：官方固定 sigma 表 + ancestral SDE（eta=1.0）无引导：cfgV/cfgA=1.0，negV/negA=nil；
    //    frozen 音频条件：audio_sigma=0.0，音频流固定为真实 latent，视频受其条件化锁嘴形）
    pipelineLog("=== ⑥ 采样：去噪 \(g.numSteps) 步（蒸馏 sigma 表 + ancestral SDE）===")
    let cfg = SamplerConfig(numSteps: g.numSteps, cfgV: 1.0, cfgA: 1.0, sigmas: ltx25DistilledSigmas, ancestral: true, seed: g.seed)
    // 条件嵌入完成 → 采样开始前：中途水位评估，卸载已用完的 Gemma/Connector（保护 DiT）
    MemoryPolicy.unloadIfNeededMidway(task: .video, current: .sampling, next: .vae)
    memProfilePoint("采样开始前")
    var (vFinal, aFinal) = sampleLatents(
        dit: dit,
        noiseV: noiseV,
        noiseA: frozenAudio ?? MLXRandom.normal([1, max(audioPos.count, 1), 128]).asType(.bfloat16),
        condV: condV, condA: condA,
        negV: negV, negA: nil,
        frozenAudio: frozenAudio,
        initVideo: initVideo, cleanV: cleanV, condMask: condMask,
        videoPos: videoPos, audioPos: audioPos,
        config: cfg,
        isCancelled: isCancelled)
    if isCancelled() {
        pipelineLog("⏹ 视频生成已取消，跳过后续解码与落盘")
        return
    }
    memProfilePoint("采样结束后")
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
    if stage2Refine {
        let tS2 = Date()
        pipelineLog("=== 4.5) 升频 ×2 + 3 步蒸馏 refine（\(condMask == nil ? "纯 t2v" : "I2V 首帧")）===")
        let vaeDecPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/vae_decoder.safetensors"
        let upPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/spatial_upscaler_x2_v1_1.safetensors"
        // 只轻量读 mean/std（2×128 标量，几十 KB）而非整读 vae_decoder（777MB）：
        // 完整权重由 ⑦ VAE 解码阶段加载，避免同轮两次读盘。
        if let stats = readSafetensorsFloatArrays(path: vaeDecPath, keys: [
            "vae_decoder.per_channel_statistics.mean",
            "vae_decoder.per_channel_statistics.std"
        ]),
           let mean0 = stats["vae_decoder.per_channel_statistics.mean"],
           let std0 = stats["vae_decoder.per_channel_statistics.std"],
           let upW = try? MLX.loadArrays(url: URL(fileURLWithPath: upPath)) {
            let mean = mean0.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
            let std = std0.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
            // 升频：denorm → spatial ×2 → norm（latent [1,T,H,W,128] → [1,T,2H,2W,128]）
            let denorm = vFinal.asType(.bfloat16) * std + mean
            let upLat = latentUpsamplerSpatial(weights: upW, latentNDHWC: denorm)
            let normLat = (upLat - mean) / std
            eval(normLat)
            pipelineLog("✅ 空间升频完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(vFinal.shape) → \(normLat.shape)")
            // 全分辨率位置（makeVideoLatentAndPos 的 noise 丢弃，只取 pos；像素坐标 h*32+16 语义不变）
            let gFull = GenConfig(numFrames: g.numFrames, height: g.height * 2, width: g.width * 2,
                                  frameRate: g.frameRate, numSteps: g.numSteps, seed: g.seed)
            let (_, fullPos) = makeVideoLatentAndPos(g: gFull)
            // 加噪到 σ0=0.909375（x_t = (1-s)*x0 + s*noise），3 步确定性 refine（无 CFG）
            MLXRandom.seed(g.seed &+ 0x5EED_5EED)
            let fullNoise = MLXRandom.normal(normLat.shape).asType(.bfloat16)
            let initLat = normLat * (1 - ltx25Stage2Sigmas[0]) + fullNoise * ltx25Stage2Sigmas[0]
            // I2V：全分辨率重编码首/尾帧 → 替换 initLat 首/尾帧 token + 构造 clean/mask
            //      对齐 ComfyUI 官方 LTXV 首尾帧语义（VideoConditionByLatentIndex + cond_mask）：
            //      stage2 与 stage1 一致，首/尾各钉 HW 个 0，中间帧生成。
            var refineInit: MLXArray = initLat
            var refineClean: MLXArray? = nil
            var refineMask: [Float]? = nil
            if condMask != nil, !imagePaths.isEmpty {
                let vaeEncPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/vae_encoder.safetensors"
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
                            let zerosMid = MLXArray.zeros([1, tCount - 2, fh, fw, 128]).asType(.bfloat16)
                            refineInit = concatenated([ref5D, noiseMid, refTail5D], axis: 1).asType(.bfloat16)
                            refineClean = concatenated([ref5D, zerosMid, refTail5D], axis: 1).asType(.bfloat16)
                            refineMask = [Float](repeating: 0, count: fhw)
                                + [Float](repeating: 1, count: (tCount - 2) * fhw)
                                + [Float](repeating: 0, count: fhw)
                            pipelineLog("🖼️ stage2 I2V 首尾帧：\(imagePaths[0]) / \(imagePaths[1])（\(gFull.width)×\(gFull.height)）→ 首/尾各钉 \(fhw) 个 0，中间 \(tCount - 2) 帧生成")
                        } else {
                            let noiseRest = refineInit[0..., 1..., 0..., 0..., 0...]      // [1,T-1,2H,2W,128]
                            let zerosRest = MLXArray.zeros([1, tCount - 1, fh, fw, 128]).asType(.bfloat16)
                            refineInit = concatenated([ref5D, noiseRest], axis: 1).asType(.bfloat16)
                            refineClean = concatenated([ref5D, zerosRest], axis: 1).asType(.bfloat16)
                            refineMask = [Float](repeating: 0, count: fhw)
                                + [Float](repeating: 1, count: tCount * fhw - fhw)
                            pipelineLog("🖼️ stage2 I2V：全分辨率首帧 \(imagePaths[0])（\(gFull.width)×\(gFull.height)）→ latent \(ref5D.shape)，mask 前 \(fhw) 个 0")
                        }
                    } else {
                        pipelineLog("⚠️ stage2 I2V 首帧编码失败，refine 不加首帧条件")
                    }
                } else {
                    pipelineLog("⚠️ stage2 I2V 权重加载失败，refine 不加首帧条件")
                }
            }
            let cfg2 = SamplerConfig(numSteps: ltx25Stage2Sigmas.count - 1, cfgV: 1.0, cfgA: 1.0,
                                     sigmas: ltx25Stage2Sigmas, ancestral: false, seed: g.seed)
            let (vRef, _) = sampleLatents(
                dit: dit, noiseV: refineInit.asType(.bfloat16), noiseA: aFinal,
                condV: condV, condA: condA, negV: nil, negA: nil,
                frozenAudio: frozenAudio,
                cleanV: refineClean, condMask: refineMask,
                videoPos: fullPos, audioPos: audioPos,
                config: cfg2, isCancelled: isCancelled)
            if isCancelled() { pipelineLog("⏹ stage2 refine 已取消"); return }
            vFinal = vRef
            videoLatentIsFullRes = true
            pipelineLog("✅ stage2 refine 完成（\(Int(Date().timeIntervalSince(tS2)))s）：\(vFinal.shape)")
        } else {
            pipelineLog("⚠️ stage2 refine 权重加载失败，回退单阶段（保持半分辨率 latent，VAE 解码前升频）")
        }
    } else {
        pipelineLog("ℹ️ stage2 refine 已关闭（stage2Refine=false），保持单阶段（VAE 解码前升频）")
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
    pipelineLog("=== ⑥ 去噪 latent 生成完成（下一步：VAE 解码 → 视频帧）===")
}

func frameToCGImage(_ frame: MLXArray, width: Int, height: Int) -> CGImage? {
    let floats = frame.asArray(Float.self)
    guard floats.count == 3 * width * height else { return nil }
    let hw = width * height
    // vDSP 向量化：[-1,1] f32 → (v*127.5+127.5) clip [0,255] → vDSP_vfixru8 四舍五入转 UInt8，
    // 用 stride=4 直接写入 RGBA 对应通道位（消除逐像素 Swift 循环，720p 240 帧从 ~8.6 亿次迭代降为 3 次 SIMD 通道）
    var rgba = [UInt8](repeating: 0, count: hw * 4)
    var scale: Float = 127.5
    var offset: Float = 127.5
    var lo: Float = 0
    var hi: Float = 255
    var tmp = [Float](repeating: 0, count: hw)
    floats.withUnsafeBufferPointer { src in
        rgba.withUnsafeMutableBufferPointer { dst in
            tmp.withUnsafeMutableBufferPointer { tbuf in
                for c in 0..<3 {
                    let base = src.baseAddress! + c * hw
                    vDSP_vsmsa(base, 1, &scale, &offset, tbuf.baseAddress!, 1, vDSP_Length(hw))
                    vDSP_vclip(tbuf.baseAddress!, 1, &lo, &hi, tbuf.baseAddress!, 1, vDSP_Length(hw))
                    vDSP_vfixru8(tbuf.baseAddress!, 1, dst.baseAddress! + c, 4, vDSP_Length(hw))
                }
                for i in 0..<hw { dst[4 * i + 3] = 255 }
            }
        }
    }
    guard let ctx = CGContext(
        data: &rgba, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    return ctx.makeImage()
}

func writeMp4(frameCount: Int, width: Int, height: Int, fps: Int, to path: String, frameAt: (Int) -> CGImage?) throws {
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.removeItem(at: url)
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
    writer.add(input)
    guard writer.startWriting() else {
        throw NSError(domain: "mp4", code: 1, userInfo: [NSLocalizedDescriptionKey: "startWriting 失败"])
    }
    writer.startSession(atSourceTime: .zero)
    let frameDur = CMTime(value: 1, timescale: CMTimeScale(fps))
    for i in 0..<frameCount {
        guard let img = frameAt(i) else { continue }
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.02)
        }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
        guard let pixelBuffer = pb else { continue }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        if let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) {
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        let t = CMTimeMultiply(frameDur, multiplier: Int32(i))
        adaptor.append(pixelBuffer, withPresentationTime: t)
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    guard writer.status == .completed else {
        throw NSError(domain: "mp4", code: 2, userInfo: [NSLocalizedDescriptionKey: "finishWriting 失败: \(writer.error?.localizedDescription ?? "?")"])
    }
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

/// 轻量读取 safetensors 指定 key（仅支持 F32/BF16），用于 stage2 refine 只需
/// per_channel_statistics（2×128 标量）的场景——避免整读 vae_decoder（777MB），
/// 完整权重留给 ⑦ VAE 解码阶段只读一次盘。
func readSafetensorsFloatArrays(path: String, keys: [String]) -> [String: MLXArray]? {
    guard let fh = try? FileHandle(forReadingFrom: URL(fileURLWithPath: path)) else { return nil }
    defer { try? fh.close() }
    let hdrLenData = fh.readData(ofLength: 8)
    guard hdrLenData.count == 8 else { return nil }
    let hdrLen = hdrLenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard hdrLen > 0, hdrLen < (1 << 30) else { return nil }
    guard let hdrData = try? fh.read(upToCount: Int(hdrLen)),
          let json = try? JSONSerialization.jsonObject(with: hdrData) as? [String: Any] else { return nil }
    var out: [String: MLXArray] = [:]
    for key in keys {
        guard let info = json[key] as? [String: Any],
              let dtype = info["dtype"] as? String,
              let shape = info["shape"] as? [Int],
              let offsets = info["data_offsets"] as? [Int], offsets.count == 2 else { continue }
        let len = offsets[1] - offsets[0]
        guard len > 0 else { continue }
        try? fh.seek(toOffset: UInt64(8 + Int(hdrLen) + offsets[0]))
        guard let data = try? fh.read(upToCount: len), data.count == len else { continue }
        if dtype == "F32" {
            var floats = [Float](repeating: 0, count: len / 4)
            _ = floats.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            out[key] = MLXArray(floats, shape)
        } else if dtype == "BF16" {
            var floats = [Float](repeating: 0, count: len / 2)
            data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
                let b = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< floats.count {
                    let u32 = UInt32(b[i].littleEndian) << 16
                    floats[i] = Float(bitPattern: u32)
                }
            }
            out[key] = MLXArray(floats, shape)
        }
    }
    return out.isEmpty ? nil : out
}

func vaeDecodeTest() -> String? {
    pipelineLog("\n=== ⑦ VAE 解码（latent → 像素帧）===")
    let vaeT0 = Date()
    memProfilePoint("VAE权重加载前")
    let vaePath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/vae_decoder.safetensors"
    let latentPath = outputVideoDirURL.appendingPathComponent("video_latent_final.npy").path
    guard let weights = try? MLX.loadArrays(url: URL(fileURLWithPath: vaePath)) else {
        pipelineLog("❌ vae_decoder.safetensors 加载失败"); return nil
    }
    memProfilePoint("VAE权重加载后")
    pipelineLog("✅ VAE 权重张量数：\(weights.count)")
    guard let latent = loadNpy(latentPath) else { return nil }
    pipelineLog("latent: \(latent.shape)（NDHWC [1,T,H,W,128]，float32；生成尺寸=目标/2，升频×2 后解码还原目标分辨率）")

    // 空间升频器：denorm → latent 空间 ×2 → norm → VAE 解码（输出分辨率 ×2）
    // stage2 refine 已完成时 latent 已是全分辨率，跳过升频直接解码
    let mean = weights["vae_decoder.per_channel_statistics.mean"]!.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
    let std = weights["vae_decoder.per_channel_statistics.std"]!.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
    let latentB = latent.asType(.bfloat16)
    let normLatent: MLXArray
    if videoLatentIsFullRes {
        normLatent = latentB
        pipelineLog("✅ latent 已是全分辨率（stage2 refine 产物），跳过升频直接 VAE 解码")
    } else {
        let tUp = Date()
        let upPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/spatial_upscaler_x2_v1_1.safetensors"
        guard let upWeights = try? MLX.loadArrays(url: URL(fileURLWithPath: upPath)) else {
            pipelineLog("❌ spatial_upscaler_x2_v1_1.safetensors 加载失败"); return nil
        }
        pipelineLog("✅ 升频器权重张量数：\(upWeights.count)")
        memProfilePoint("升频前")
        let denorm = latentB * std + mean
        // 升频器前向 + 回 norm 放在独立函数里，upLatent 函数返回后即释放，
        // 避免大张量引用滞留到解码阶段
        let upLat = upscaleLatentAndNorm(upWeights: upWeights, denormLatent: denorm, mean: mean, std: std)
        eval(upLat)
        normLatent = upLat
        memProfilePoint("升频后")
        pipelineLog("✅ 空间升频完成（\(Int(Date().timeIntervalSince(tUp)))s）：\(latentB.shape) → \(normLatent.shape)（NDHWC，空间 ×2）")
    }

    let t0 = Date()
    pipelineLog("解码中（VAE Tiling 分块，1024 通道 3D 卷积，预计几分钟）...")
    // latent f32 → bf16：权重本身 bf16，全链路 bf16 省内存提速；统计等价性由回归验证
    // tiled 解码：默认 tile 对齐官方 TileSizeConfig.default()（时间 tile=10帧/overlap=3，空间 tile=24格/overlap=2），
    // 分块 + 梯形 mask 加权融合，压低 720p 5s 场景 VAE 解码峰值内存（整片曾达 51.45GB）
    let pixels = vaeDecodeTiled(weights: weights, latentNDHWC: normLatent)
    eval(pixels)
    memProfilePoint("VAE解码后")
    pipelineLog("✅ 解码完成（\(Int(Date().timeIntervalSince(t0)))s）：\(pixels.shape)，期望 [1,3,113,512,512]")

    // NaN 检查
    let parr = pixels.asArray(Float.self)
    let hasBad = parr.contains { !$0.isFinite }
    pipelineLog("NaN/Inf：\(hasBad ? "⚠️ 有" : "✅ 无")，mean=\(parr.reduce(0, +) / Float(parr.count))")

    // 帧导出直接流式写 mp4，不再落盘 pixels_final.npy（省 ~1.4GB SSD 写入）
    let outDir = outputVideoDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    // 管线第一次落盘即用计数名，与资产库一致（进资产库是第二次传递，同名冲突在 attach 阶段加 _ 处理）
    let baseName = nextAssetName(prefix: "ltx")

    // 帧导出 + mp4
    let fCount = pixels.shape[2]
    let h = pixels.shape[3], w = pixels.shape[4]
    pipelineLog("导出 \(fCount) 帧 \(w)×\(h) → mp4（流式）...")
    let silentPath = "\(outDir)/\(baseName)_silent.mp4"
    let t1 = Date()
    do {
        try writeMp4(frameCount: fCount, width: w, height: h, fps: 24, to: silentPath) { f in
            let frameArr = pixels[0 ..< 1, 0 ..< 3, f ..< (f + 1), 0 ..< h, 0 ..< w]
            let squeezed = frameArr.reshaped([3, h, w])
            return frameToCGImage(squeezed, width: w, height: h)
        }
    } catch {
        pipelineLog("⚠️ mp4 流式写入失败：\(error.localizedDescription)")
    }
    pipelineLog("✅ 帧转换完成（\(Int(Date().timeIntervalSince(t1)))s）：\(fCount) 帧")
    let mp4Path = "\(outDir)/\(baseName).mp4"

    // 音频解码：audio latent → 16kHz 立体声 PCM → 混入 mp4
    var audioPCM: [Float]? = nil
    let audioVaePath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/audio_vae.safetensors"
    let vocoderPath = "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/vocoder.safetensors"
    let audioLatentPath = "\(outDir)/audio_latent_final.npy"
    pipelineLog("加载音频权重（audio_vae + vocoder）...")
    memProfilePoint("音频权重加载前")
    let t2 = Date()
    // 复用同轮音频条件编码已加载的 audio_vae（102MB，避免重复读盘）；用完即释放
    let aw = sharedAudioVAEWeights ?? (try? MLX.loadArrays(url: URL(fileURLWithPath: audioVaePath)))
    sharedAudioVAEWeights = nil
    if let aw,
       let vw = try? MLX.loadArrays(url: URL(fileURLWithPath: vocoderPath)) {
        var allW = aw
        for (k, v) in vw { allW[k] = v }
        pipelineLog("✅ 音频权重张量数：audio_vae \(aw.count) + vocoder \(vw.count)，加载耗时 \(Int(Date().timeIntervalSince(t2)))s")
        memProfilePoint("音频权重加载后")
        // bf16 权重保持 bf16（MLX bf16 conv 在 M 系列 GPU 原生支持，内存省一半、速度更快）
        var f32W: [String: MLXArray] = allW
        if let aLatent = loadNpy(audioLatentPath) {
            pipelineLog("audio latent: \(aLatent.shape) → 解码 mel + vocoder（预计 1~3 分钟）...")
            let t3 = Date()
            let (pcm, frames) = decodeAudio(weights: f32W, latent: aLatent.asType(.bfloat16))
            memProfilePoint("音频解码后")
            pipelineLog("✅ 音频解码完成（\(Int(Date().timeIntervalSince(t3)))s）：\(frames) 帧 @16kHz（\(Float(frames) / 16000.0)s）")
            let pcmMean = pcm.reduce(0, +) / Float(max(pcm.count, 1))
            let pcmMax = pcm.map { abs($0) }.max() ?? 0
            pipelineLog("   波形 mean=\(pcmMean) maxAbs=\(pcmMax) NaN=\(pcm.contains { !$0.isFinite })")
            let wavPath = "\(outDir)/ltx_audio.wav"
            writeWav(pcm, sampleRate: 16000, channels: 2, to: wavPath)
            pipelineLog("✅ wav 已保存：\(wavPath)")
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
        pipelineLog("✅ 无声 mp4 已生成：\(silentPath)")
        if let audioPCM {
            try muxAudio(videoPath: silentPath, wavPath: "\(outDir)/ltx_audio.wav", to: mp4Path)
            pipelineLog("✅ 音轨已混入（AAC）：\(mp4Path)")
        } else {
            try? FileManager.default.moveItem(atPath: silentPath, toPath: mp4Path)
            pipelineLog("✅ mp4（无音轨）：\(mp4Path)")
        }
    } catch {
        pipelineLog("⚠️ mp4 合成失败：\(error.localizedDescription)")
    }
    let vaeSec = Int(Date().timeIntervalSince(vaeT0))
    pipelineLog("✅ VAE 解码完成（共 \(vaeSec / 60)分\(String(format: "%02d", vaeSec % 60))秒）")
    return mp4Path
}

func archiveIfExists(_ path: String) {
    let fm = FileManager.default
    guard fm.fileExists(atPath: path) else { return }
    let df = DateFormatter()
    df.dateFormat = "yyyyMMdd_HHmmss"
    let stamp = df.string(from: Date())
    let ext = (path as NSString).pathExtension
    let base = (path as NSString).deletingPathExtension
    let newPath = "\(base)_\(stamp).\(ext)"
    try? fm.moveItem(atPath: path, toPath: newPath)
    pipelineLog("📦 归档旧产物：\(newPath)")
}

func muxAudio(videoPath: String, wavPath: String, to outPath: String) throws {
    let videoURL = URL(fileURLWithPath: videoPath)
    let wavURL = URL(fileURLWithPath: wavPath)
    let outURL = URL(fileURLWithPath: outPath)
    try? FileManager.default.removeItem(at: outURL)

    let videoAsset = AVURLAsset(url: videoURL)
    let audioAsset = AVURLAsset(url: wavURL)
    let comp = AVMutableComposition()
    guard let compVideo = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
          let srcVideo = videoAsset.tracks(withMediaType: .video).first else {
        throw NSError(domain: "mux", code: 1, userInfo: [NSLocalizedDescriptionKey: "视频轨缺失"])
    }
    let dur = srcVideo.timeRange
    try compVideo.insertTimeRange(dur, of: srcVideo, at: .zero)

    if let srcAudio = audioAsset.tracks(withMediaType: .audio).first,
       let compAudio = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
        try compAudio.insertTimeRange(dur, of: srcAudio, at: .zero)
    }

    guard let exporter = AVAssetExportSession(asset: comp, presetName: AVAssetExportPresetHighestQuality) else {
        throw NSError(domain: "mux", code: 2, userInfo: [NSLocalizedDescriptionKey: "export session 创建失败"])
    }
    exporter.outputURL = outURL
    exporter.outputFileType = .mp4
    let sem = DispatchSemaphore(value: 0)
    exporter.exportAsynchronously { sem.signal() }
    sem.wait()
    guard exporter.status == .completed else {
        throw NSError(domain: "mux", code: 3, userInfo: [NSLocalizedDescriptionKey: "导出失败: \(exporter.error?.localizedDescription ?? "?")"])
    }
}
