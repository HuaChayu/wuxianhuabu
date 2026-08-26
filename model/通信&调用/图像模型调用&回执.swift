//
//  图像模型调用&回执.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/19.
//
//  ============================================================
//  文件作用：HiDream-O1-Image 图像生成管线调用入口（从 ltx-test 验证包移植）。
//  包含：权重加载（Qwen3VL backbone + custom_heads 扩散头）→ T2I 采样 →
//       28 步 FlashFlowMatch 去噪 → patchify/unpatchify → PNG 落盘 的完整调用链。
//  依赖实现文件：HiDream-O1-Image/HiDreamBackbone.swift、HiDreamDiffusion.swift、
//               HiDreamPipeline.swift
//  依赖算子文件：通用算子&函数/模型通用加载.swift（loadSafetensors）、
//               通用算子&函数/模型公共函数.swift（memProfilePoint/memProfileReport）
//  ============================================================
//

import Foundation
import ImageIO
@preconcurrency import MLX
@preconcurrency import MLXNN
import MLXLMCommon
import MLXVLM
import Hub
import Tokenizers
import AppKit

// MARK: - 图像管线日志（print + 模型监控浮层 + 落盘文件 三通道）

/// 图像管线关键日志：同时输出到 Xcode 控制台、模型监控浮层（需打开右下角 sparkles 开关）与日志文件。
/// 日志文件路径：output/图像/pipeline.log
func imagePipelineLog(_ message: String) {
    print(message)
    monitorLog(message)
    let logURL = outputImageDirURL.appendingPathComponent("pipeline.log")
    let line = "[\(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium))] \(message)\n"
    if let h = try? FileHandle(forWritingTo: logURL) {
        h.seekToEndOfFile()
        h.write(line.data(using: .utf8)!)
        try? h.close()
    } else {
        try? line.data(using: .utf8)?.write(to: logURL)
    }
}

// MARK: - HiDream 权重持久缓存

/// HiDream 骨架 + 扩散头 + tokenizer 常驻：首次从 safetensors 加载后复用，
/// 不随每次生成重建/释放。权重目录未变且文件未改动时跳过加载（从几分钟降到秒级）。
final class HiDreamModelCache {
    static let shared = HiDreamModelCache()

    private var backbone: HiDreamBackbone?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var qwenConfig: Qwen3VLConfiguration?
    private var baseDir: String = ""
    private var modelModTime: Date?
    private var headsModTime: Date?

    /// 目录与权重文件未变 → 返回已加载实例；否则返回 nil（需要重新加载）。
    func get(base: String) -> (backbone: HiDreamBackbone, tokenizer: any Tokenizers.Tokenizer, qwenConfig: Qwen3VLConfiguration)? {
        let modelPath = base + "/model.safetensors"
        let headsPath = base + "/extras/custom_heads.safetensors"
        guard let b = backbone, let t = tokenizer, let q = qwenConfig,
              baseDir == base,
              modelModTime == fileModTime(modelPath),
              headsModTime == fileModTime(headsPath) else { return nil }
        return (b, t, q)
    }

    func store(_ b: HiDreamBackbone, tokenizer t: any Tokenizers.Tokenizer, qwenConfig q: Qwen3VLConfiguration, base: String) {
        backbone = b
        tokenizer = t
        qwenConfig = q
        baseDir = base
        modelModTime = fileModTime(base + "/model.safetensors")
        headsModTime = fileModTime(base + "/extras/custom_heads.safetensors")
    }

    /// backbone 是否已缓存（加载前评估水位：已缓存 → newSize=0，永不触发卸载，直接复用）
    var hasHiDream: Bool {
        backbone != nil
    }

    /// 卸载 HiDream 权重（生成完内存紧张时调用）
    func clear() {
        backbone = nil
        tokenizer = nil
        qwenConfig = nil
        baseDir = ""
        modelModTime = nil
        headsModTime = nil
    }

    private func fileModTime(_ path: String) -> Date? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return attrs?[.modificationDate] as? Date
    }
}

// MARK: - 图像生成尺寸对齐公共函数

/// HiDream-O1 图像生成目标尺寸：按画布比例从训练分辨率表选最接近项
/// （与通用 closestResolution 同源：仅返回可被 32 整除的网格尺寸）。
func imageSize(for ratio: CanvasStore.Ratio, width: Int? = nil, height: Int? = nil) -> (width: Int, height: Int) {
    let const = HiDreamDiffusionConstants.self
    var imgRatio: Float
    if let width, let height, width > 0, height > 0 {
        imgRatio = Float(width) / Float(height)
    } else {
        imgRatio = Float(ratio.width) / Float(ratio.height)
    }
    var best = const.predefinedResolutions[0]
    var bestDiff = Float.greatestFiniteMagnitude
    for r in const.predefinedResolutions {
        let diff = abs(Float(r.w) / Float(r.h) - imgRatio)
        if diff < bestDiff {
            bestDiff = diff
            best = r
        }
    }
    return (best.w, best.h)
}

/// 图像生成目标尺寸（按用户档位）：短边 = 档位目标值（64 倍数），长边 = 短边 × 比例后 64 倍数对齐。
/// 64 倍数除2后正好 32 倍数：作视频首尾帧时零缩放。
/// - 480→512、720→768、1080→1088、2K→2048（仅图像类节点有 2K 档）
func imageResolution(for ratio: CanvasStore.Ratio, quality: ImageQuality) -> (width: Int, height: Int) {
    let short = quality.targetShortSide
    let isLandscape = Float(ratio.width) >= Float(ratio.height)
    let longFloat = Float(short) * (isLandscape ? Float(ratio.width) / Float(ratio.height) : Float(ratio.height) / Float(ratio.width))
    let long = Int((longFloat / 64).rounded()) * 64
    return isLandscape ? (max(long, short), short) : (short, max(long, short))
}

// MARK: - 主入口

/// 读取图片的实际像素尺寸（不解码像素，仅读元数据）。
private func imagePixelSize(_ path: String) -> (w: Int, h: Int)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
          let w = props[kCGImagePropertyPixelWidth] as? Int,
          let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
    return (w, h)
}

/// 官方 keep_original_aspect 同款：参考图按面积缩放到 ≤ maxSize²、patch 对齐、保持原比例。
/// 对齐 py resize_pilimage：candidates 取 round/floor 组合，选面积 ≤ S_max 的最大者。
/// 保留备用：编辑模式尺寸由档位决定，当前 runImagePipeline 不调用本函数。
private func editTargetSize(
    from path: String,
    maxSize: Int = 2048,
    patch: Int = 32
) -> (w: Int, h: Int)? {
    guard let (origW, origH) = imagePixelSize(path), origW > 0, origH > 0 else { return nil }
    let sMax = maxSize * maxSize
    let scale = sqrt(Double(sMax) / Double(origW * origH))
    let r = { (v: Double) -> Int in Int(v.rounded()) / patch * patch }
    let f = { (v: Double) -> Int in Int(v.rounded(.down)) / patch * patch }
    let candidates = [
        (r(Double(origW) * scale), r(Double(origH) * scale)),
        (r(Double(origW) * scale), f(Double(origH) * scale)),
        (f(Double(origW) * scale), r(Double(origH) * scale)),
        (f(Double(origW) * scale), f(Double(origH) * scale)),
    ].sorted { $0.0 * $0.1 > $1.0 * $1.1 }
    let best = candidates.first { $0.0 * $0.1 <= sMax } ?? candidates[0]
    return (max(patch, best.0), max(patch, best.1))
}

/// 端到端生成：提示词 → T2I 采样 → 28 步去噪 → PNG 落盘。
/// - Parameters:
///   - prompt: 图像提示词；为空时使用默认提示词。
///   - referencePaths: 参考图路径（非空时进入编辑/多参考分支 hidreamGenerateEdit）。
///   - width/height: 请求尺寸（档位尺寸）；nil 时按画布比例选。
/// - Returns: 生成图片 PNG 的绝对路径；任一步失败返回 nil。
public func runImagePipeline(prompt: String = "", referencePaths: [String] = [], width: Int? = nil, height: Int? = nil, isCancelled: @escaping () -> Bool = { false }) async -> String? {
    MonitorCenter.shared.taskStart("图像生成")
    defer { MonitorCenter.shared.taskEnd("图像生成") }
    let base = "/Users/huachayui/Downloads/HiDream-O1-Image-Dev-mlx-bf16"
    let pipeT0 = Date()
    memProfilePoint("图像管线开始 基线")
    // 图像任务开始：互斥（≤64GB）时卸对方（LTX 视频侧 Gemma/Connector/DiT），保护 HiDream
    MemoryPolicy.unloadIfNeededMidway(task: .image, current: .sampling, next: nil)

    // 1) 权重 + tokenizer（常驻缓存，目录/文件未变则复用）
    let model: (backbone: HiDreamBackbone, tokenizer: any Tokenizers.Tokenizer, qwenConfig: Qwen3VLConfiguration)
    if let cached = HiDreamModelCache.shared.get(base: base) {
        model = cached
        imagePipelineLog("✅ 复用已加载 HiDream 权重（跳过加载）")
    } else {
        let t0 = Date()
        guard let qwenConfig = try? hidreamLoadQwenConfig(from: base) else {
            imagePipelineLog("❌ config.json 加载失败：\(base)")
            return nil
        }
        let backbone = HiDreamBackbone(qwenConfig: qwenConfig)

        var w1 = loadSafetensors(base + "/model.safetensors")
        guard var w1 else {
            imagePipelineLog("❌ model.safetensors 加载失败")
            return nil
        }
        guard let w2 = loadSafetensors(base + "/extras/custom_heads.safetensors") else {
            imagePipelineLog("❌ custom_heads.safetensors 加载失败")
            return nil
        }
        for (k, v) in w2 { w1[k] = v }
        hidreamSanitizeWeights(&w1)
        do {
            try backbone.update(
                parameters: ModuleParameters.unflattened(w1), verify: [.noUnusedKeys])
        } catch {
            imagePipelineLog("❌ 权重灌入失败：\(error.localizedDescription)")
            return nil
        }
        imagePipelineLog("✅ 权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s，\(w1.count) 键）")

        // tokenizer（Qwen3VL，从模型目录加载 tokenizer.json / tokenizer_config.json）
        let config = LanguageModelConfigurationFromHub(
            modelFolder: URL(fileURLWithPath: base))
        guard let tokenizerConfig = try? await config.tokenizerConfig,
              let tokenizerData = try? await config.tokenizerData else {
            imagePipelineLog("❌ tokenizer 配置加载失败")
            return nil
        }
        guard let tokenizer = try? AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData) else {
            imagePipelineLog("❌ tokenizer 构建失败")
            return nil
        }
        imagePipelineLog("✅ tokenizer 就绪（\(Int(Date().timeIntervalSince(t0)))s）")
        memProfilePoint("HiDream权重加载后")

        HiDreamModelCache.shared.store(backbone, tokenizer: tokenizer, qwenConfig: qwenConfig, base: base)
        model = (backbone, tokenizer, qwenConfig)
    }
    memProfilePoint("HiDream模型就绪")

    // 2) 尺寸：直接使用外部传入的档位尺寸（startImageGeneration 已按 64 倍数对齐 = 32 倍数）。
    //    编辑模式尺寸由档位决定，editTargetSize 保留备用（≤2048² 面积缩放 / keep_original_aspect 同款），当前不调用。
    //    closestResolution 内部对 32 倍数尺寸直接放行，仅非 32 倍数时吸附训练表兜底。
    var gW = width ?? 1024
    var gH = height ?? 1024
    let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let finalPrompt = trimmedPrompt.isEmpty ? "A majestic golden retriever running through a sunlit meadow, cinematic lighting" : trimmedPrompt
    imagePipelineLog("提示词：\(finalPrompt)")
    if !referencePaths.isEmpty {
        imagePipelineLog("参考图：\(referencePaths.count) 张，进入编辑/多参考分支")
    }
    imagePipelineLog("配置：生成 \(gW)×\(gH)（32 倍数放行，非 32 倍数才吸附），28 步 FlashFlowMatch，seed=32")

    // 3) 生成：有参考图走编辑/多参考分支 hidreamGenerateEdit，无参考图走纯 T2I hidreamGenerate。
    memProfilePoint("图像采样开始前")
    let tGen = Date()
    var lastStepT = Date()
    let stepLog: (Float) -> Void = { p in
        let itS = Date().timeIntervalSince(lastStepT)
        lastStepT = Date()
        imagePipelineLog(String(format: "步%02d  %.2fs/it", min(Int(p * 28) + 1, 28), itS))
    }
    let img: MLXArray
    do {
        let genResult: Result<MLXArray, Error> = {
            do {
                if referencePaths.isEmpty {
                    let r = try hidreamGenerate(
                        backbone: model.backbone,
                        tokenizer: model.tokenizer,
                        qwenConfig: model.qwenConfig,
                        prompt: finalPrompt,
                        height: gH,
                        width: gW,
                        steps: 28,
                        seed: 32) { p in
                            stepLog(p)
                        } isCancelled: {
                        isCancelled()
                    }
                    return .success(r)
                } else {
                    let r = try hidreamGenerateEdit(
                        backbone: model.backbone,
                        tokenizer: model.tokenizer,
                        qwenConfig: model.qwenConfig,
                        prompt: finalPrompt,
                        referencePaths: referencePaths,
                        height: gH,
                        width: gW,
                        steps: 28,
                        seed: 32) { p in
                            stepLog(p)
                        } isCancelled: {
                        isCancelled()
                    }
                    return .success(r)
                }
            } catch {
                return .failure(error)
            }
        }()
        img = try genResult.get()
    } catch GenerationCancelError.cancelled {
        imagePipelineLog("⏹ 图像生成已取消")
        return nil
    } catch {
        imagePipelineLog("❌ 图像生成失败：\(error.localizedDescription)")
        return nil
    }
    eval(img)
    memProfilePoint("图像采样结束后")
    let stats = img.asArray(Float.self)
    let mean = stats.reduce(0, +) / Float(max(stats.count, 1))
    let maxAbs = stats.map { abs($0) }.max() ?? 0
    let genSec = Int(Date().timeIntervalSince(tGen))
    imagePipelineLog("✅ 生成完成（\(genSec / 60)分\(String(format: "%02d", genSec % 60))秒）：\(img.shape) mean=\(mean) maxAbs=\(maxAbs) NaN=\(stats.contains { !$0.isFinite })")

    // 4) 落盘 PNG（管线第一次落盘即用计数名，与资产库一致；进资产库是第二次传递，同名冲突在 attach 阶段加 _ 处理）
    let outDir = outputImageDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let outPath = "\(outDir)/\(nextAssetName(prefix: "hidream")).png"
    guard saveHiDreamPNG(img, to: outPath) else {
        imagePipelineLog("❌ PNG 编码/写入失败")
        return nil
    }
    imagePipelineLog("✅ 已保存：\(outPath)")
    MLX.Memory.clearCache()   // 去噪中间张量缓冲回收（对齐视频管线）
    MemoryPolicy.unloadModelsIfNeeded()   // 生成完按水位策略卸载常驻模型
    let snap = MLX.Memory.snapshot()
    let gb: (Int) -> String = { String(format: "%.2fGB", Double($0) / Double(1 << 30)) }
    let pipeSec = Int(Date().timeIntervalSince(pipeT0))
    imagePipelineLog("✅ 图像端到端完成，总耗时 \(pipeSec / 60)分\(String(format: "%02d", pipeSec % 60))秒：active=\(gb(snap.activeMemory)) cache=\(gb(snap.cacheMemory)) peak=\(gb(snap.peakMemory))")
    memProfileReport()
    return outPath
}

/// 保存 [H,W,3] float32（[0,1]）图像为 PNG。
func saveHiDreamPNG(_ img: MLXArray, to path: String) -> Bool {
    let h = img.dim(0), w = img.dim(1)
    guard h > 0, w > 0 else { return false }
    let floats = img.asType(.float32).flattened().asArray(Float.self)
    guard floats.count == 3 * h * w else { return false }
    #if DEBUG
    let nanCount = floats.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
    let infCount = floats.reduce(0) { $0 + ($1.isInfinite ? 1 : 0) }
    let fMin = floats.min() ?? 0, fMax = floats.max() ?? 0
    let fMean = floats.reduce(0, +) / Float(max(floats.count, 1))
    FileHandle.standardError.write(
        "DBG saveHiDreamPNG nan=\(nanCount) inf=\(infCount) min=\(fMin) max=\(fMax) mean=\(fMean)\n".data(using: .utf8)!)
    #endif
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
        bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: w * 3, bitsPerPixel: 24) else { return false }
    guard let ptr = rep.bitmapData else { return false }
    for i in 0 ..< floats.count {
        ptr[i] = UInt8(max(0, min(255, Int((floats[i] * 255).rounded()))))
    }
    guard let png = rep.representation(using: .png, properties: [:]) else { return false }
    do {
        try png.write(to: URL(fileURLWithPath: path))
        return true
    } catch {
        return false
    }
}
