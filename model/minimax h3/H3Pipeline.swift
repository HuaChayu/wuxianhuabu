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

    // ── H3-to-LTX-Latent-Adapter 直通（可选） ──
    /// adapter 直出的 LTX 归一化 latent：NDHWC [1, F, H/32, W/32, 128]。
    /// 非 nil 时，队列走「latent 直入二采」路径，跳过 LTX vaeEncodeVideo。
    /// 若生成时 adapterSkipH3Decode=true，则 pixels 为 nil（H3 完全不解码），
    /// stage1 预览视频不存在，音轨以 audioTrackPath（独立 wav）承载。
    public var ltxHalfLatent: MLXArray?
    /// adapter 模式下 H3 侧等效像素几何（= H3 latent 空间 × 16），供 LTX 侧定尺寸/预算
    public var adapterPixelWidth: Int = 0
    public var adapterPixelHeight: Int = 0
    /// adapter 跳解码模式下的音轨载体（独立 wav）；非 nil 表示无 stage1 mp4，二采音轨取此文件
    public var audioTrackPath: String?
}
import MLX
import MLXNN
import MLXRandom
import MLXLMCommon
import Tokenizers
import Hub
import ImageIO

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
        // ★ SelfLift 第三分支（H3 一采渐进式采样，见 SelfLiftH3-(h3专属).swift）：
        //   true = stage1 走「低分（×0.5）ts 次 NFE → 零 NFE 过渡块（nearest 直接 latent 提升 +
        //          VAE 像素锚点一致修正）→ 升回全分辨率 → 高分 N-ts 次 NFE」，总 NFE = N 不变；
        //   false = 原单条 stage1 循环单遍直出（默认，行为与本次接入前逐字一致，供自检/对照脚本用）。
        //   生产入口（模型生成队列-通用.swift）按偏好设置「H3 一采设置 → SelfLift 渐进采样」显式传值。
        //   ts 不写死：transitionStep = 0 → 官方 75% 规则按 N（= sigmas.count - 1，取自步数滑杆）推导。
        selfLiftEnabled: Bool = false,
        // ★ SelfLift lowOnly 模式（IC 链路用）：true = H3 一采只跑低分（半清）段，直接返回
        //   低清 latent（跳过过渡块 + H3 高分循环）；高分（升频×2 + 精修）交由 LTX 二采完成，
        //   「H3 低分 + LTX 高分」两模型组成 lift。必须与队列二采 fullResInput=false 配套。
        selfLiftLowOnly: Bool = false,
        log: (String) -> Void = { s in print("[H3-FL2VA] \(s)"); fflush(stdout) },
        stage2: H3Stage2Config? = nil,
        proResOutput: Bool = false,
        stage2MemBridge: H3Stage2MemoryBridge? = nil,
        /// ref2va 多参考图通路：非空即启用（此时首/尾帧被忽略）。
        /// 每张图按官方 reference canvas（短边 2048、32 对齐）编码为独立参考块，
        /// 文本侧生成 `<Picture i>: ` + vision block 序列，prompt 放在最后。
        referenceImagePaths: [String] = [],
        /// 参考图缩放模式：.match = 缩到与生成画面同面积（默认，序列最短）；
        /// .mid = 短边上限 1024；.max = 短边上限 2048（官方 ref2va，序列最长）
        referenceSizing: RefImageSizing = .match,
        /// ★ H3→LTX latent 直通适配器（H3-to-LTX-Latent-Adapter，见 H3-to-LTX-Latent-Adapter-(h3专属).swift）：
        /// 非 nil 时，在 H3 VAE 解码之前拦截 clean latent，直接映射为 LTX 归一化 latent 挂到 bridge，
        /// 使队列可走「latent 直入二采」路径，省去 LTX VAE encode（乃至 H3 VAE decode）两步像素往返。
        h3ToLTXAdapter: H3ToLTXLatentAdapter? = nil,
        /// adapter 生效时是否连 H3 VAE 解码一并跳过（默认 true：不写 stage1 mp4，音轨单独落 wav
        /// 由二采最终混流；false：仍解码出 stage1 预览视频，仅省 LTX encode）。
        adapterSkipH3Decode: Bool = true
    ) async throws -> String {
        let t0 = Date()
        let modelDir = "\(CommonPaths.modelRoot)/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit"
        // ★ 2026-09-19 嵌套目录兼容：下载落位历史上可能产生「根目录/同名子目录」嵌套
        //   （远端 listFiles 的 Path 自带与模型目录同名的顶层前缀）。生成侧一律按真实
        //   落位解析：根目录直拼优先，不存在时任意深度递归查找（FileFinder），保证
        //   嵌套场景也能加载；两者皆无时回退直拼路径（后续加载自然报错，便于定位）。
        let modelRootURL = URL(fileURLWithPath: modelDir, isDirectory: true)
        let wURL = { (name: String) -> URL in
            let direct = modelRootURL.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: direct.path) { return direct }
            return FileFinder.first(named: name, under: modelRootURL) ?? direct
        }

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

        // ── 1. 条件编码：fl2va 首尾帧 OR ref2va 多参考图 ──
        let useRef2VA = !referenceImagePaths.isEmpty
        var condRows: MLXArray!
        var latC = 0, latT = 0, latH = 0, latW = 0, gridH = 0, gridW = 0
        // ref2va 布局块与文本侧视觉块（顺序严格对应）
        var refBlocks: [RefBlock] = []
        var refVisionBlocks: [H3VisionBlock] = []
        var refVisionLabels: [String] = []

        if useRef2VA {
            guard referenceImagePaths.count <= Int(H3Const.maxRefImages) else {
                throw NSError(domain: "H3REF2VA", code: 1,
                              userInfo: [NSLocalizedDescriptionKey:
                                "参考图数量 \(referenceImagePaths.count) 超过上限 \(H3Const.maxRefImages)"])
            }
            log("ref2va：\(referenceImagePaths.count) 张参考图")

            try autoreleasepool {
                var vaeWeights: H3Weights? = try H3Weights(url: wURL("video_vae.safetensors"))
                vaeWeights?.cacheEnabled = false
                var vae: H3VAE? = try H3VAE.load(vaeWeights!)
                // 目标 latent 几何：用生成分辨率的探测图走同一条 VAE 通路（不做 /16 假设）
                let probe = vae!.encodeImage(MLXArray.zeros([1, 3, 1, height, width]))
                latT = probe.shape[2]
                latH = probe.shape[3]
                latW = probe.shape[4]
                latC = probe.shape[1]
                gridH = latH / 2
                gridW = latW / 2
                log("ref2va 目标 latent [1,\(latC),\(latT),\(latH),\(latW)]")
                var rowChunks: [MLXArray] = []
                for (i, p) in referenceImagePaths.enumerated() {
                    guard let dims = h3ImagePixelSize(p) else {
                        throw NSError(domain: "H3REF2VA", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "参考图尺寸读取失败：\(p)"])
                    }
                    // 官方 reference canvas：短边 2048 上限、32 对齐；再经 fitCanvas 复核对齐
                    let rc = refImageCanvas(UInt32(dims.w), UInt32(dims.h),
                                            genW: UInt32(width), genH: UInt32(height),
                                            mode: referenceSizing)
                    let vc = fitCanvas(h: rc.h, w: rc.w)
                    guard let px = loadImageBCFHW(path: p, width: Int(vc.w), height: Int(vc.h)) else {
                        throw NSError(domain: "H3REF2VA", code: 2,
                                      userInfo: [NSLocalizedDescriptionKey: "参考图加载失败：\(p)"])
                    }
                    let lat = vae!.encodeImage(px)      // [1,24,1,lh,lw]
                    let lh = lat.shape[3], lw = lat.shape[4], lc = lat.shape[1]
                    let nhwc = lat.transposed(0, 2, 3, 4, 1).reshaped([1, lh, lw, lc])
                    rowChunks.append(H3TensorOps.patchifyVideo(nhwc, gridH: lh / 2, gridW: lw / 2))
                    refBlocks.append(RefBlock(kind: .image, latentH: UInt32(lh), latentW: UInt32(lw),
                                              latentT: 1, audioT: 0))
                    // 文本侧视觉块：静帧沿 temporal patch 重复填满 2 帧
                    let plane = px.reshaped([3, Int(vc.h), Int(vc.w)])
                    let frames = concatenated([plane, plane], axis: 0)   // [2,3,H,W]
                    refVisionBlocks.append(H3VisionBlock(frames: frames, grid: gridFor(h: vc.h, w: vc.w)))
                    refVisionLabels.append(labelFor(kind: .image, ordinal: UInt32(i + 1)))
                    if (ProcessInfo.processInfo.environment["NA_H3_FL2VA_AS_REFS"].flatMap { Int($0) } ?? 0) > 0
                        && referenceImagePaths.count == 2 {
                        // ★ 2026-09-18 首尾帧 as refs：恰好 2 张参考图时把视觉块标签换成
                        //   首/尾帧语义（而非通用 "Picture 1/2"），让模型明确哪张是开头、
                        //   哪张是结尾——软参考无 keyframes 时间锚，靠文本标签补时序角色。
                        refVisionLabels[refVisionLabels.count - 1] = (i == 0 ? "<First frame>: " : "<Last frame>: ")
                    }
                    log("参考图 \(i + 1)：\(dims.w)×\(dims.h) → 画布 \(vc.w)×\(vc.h)，latent [\(lc),1,\(lh),\(lw)]，rows \((lh / 2) * (lw / 2))")
                }
                condRows = rowChunks.count == 1 ? rowChunks[0] : concatenated(rowChunks, axis: 0)
                MLX.eval(condRows)
                log("参考图编码完成：cond rows \(condRows.shape)")

                // 噪声增强：r = 0.999·r + 0.001·noise（与 fl2va keyframe 同一规则）
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
        } else {
        // ── 1. 加载 2 张 keyframe 图片 → VAE 编码 → cond rows ──
        guard let firstPx = loadImageBCFHW(path: firstImagePath, width: width, height: height),
              let lastPx = loadImageBCFHW(path: lastImagePath, width: width, height: height) else {
            throw NSError(domain: "H3FL2VA", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "图片加载失败：\(firstImagePath) / \(lastImagePath)"])
        }
        log("图片加载完成：首帧 \(firstPx.shape)，尾帧 \(lastPx.shape)")

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
        }
        h3MemLog("VAE 编码器已释放（pool drain 后）")
        log("VAE 编码器已释放")

        // ── 2. tokenizer + 文本编码 ──
        // 语言模型目录同样按真实落位解析：根目录直拼 config.json 存在则用根目录，
        // 否则递归定位 config.json 所在目录（兼容历史嵌套落位）。
        let lmFolderURL: URL = {
            if FileManager.default.fileExists(atPath: modelRootURL.appendingPathComponent("config.json").path) {
                return modelRootURL
            }
            return FileFinder.first(named: "config.json", under: modelRootURL)?.deletingLastPathComponent() ?? modelRootURL
        }()
        let lmConfig = LanguageModelConfigurationFromHub(modelFolder: lmFolderURL)
        guard let tokConfig = try? await lmConfig.tokenizerConfig,
              let tokData = try? await lmConfig.tokenizerData,
              let tokenizer = try? AutoTokenizer.from(tokenizerConfig: tokConfig, tokenizerData: tokData) else {
            throw NSError(domain: "H3FL2VA", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "tokenizer 加载失败"])
        }
        let promptIds = tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) }
        var presentItems: [H3PresentItem] = []
        var textTagsForLayout: [UInt8] = []
        // 无字幕抑制条件：简短标准写法，避免长句误伤画面内容（如广告牌/文字元素）。
        // 作为条件 token 注入文本序列末尾（与首尾帧标签同机制），默认开启，设 NA_H3_NO_SUBTITLES=0 关闭。
        let noSubtitleOn = (ProcessInfo.processInfo.environment["NA_H3_NO_SUBTITLES"].flatMap { $0 == "1" } ?? true)
        if useRef2VA {
            // 对齐官方 build_ref2va_presentation：逐张 `<Picture i>: ` + vision block，prompt 收尾
            for (i, vb) in refVisionBlocks.enumerated() {
                let labelIds = tokenizer.encode(text: refVisionLabels[i], addSpecialTokens: false).map { Int32($0) }
                presentItems.append(.text(labelIds))
                presentItems.append(.vision(vb))
            }
            presentItems.append(.text(promptIds))
            if noSubtitleOn {
                let suppressIds = tokenizer.encode(text: "no subtitles", addSpecialTokens: false).map { Int32($0) }
                presentItems.append(.text(suppressIds))
                log("ref2va 文本序列：\(refVisionBlocks.count) 个视觉块 + prompt \(promptIds.count) tokens + 无字幕抑制 \(suppressIds.count) tokens；标签：\(refVisionLabels.joined(separator: " | "))")
            } else {
                log("ref2va 文本序列：\(refVisionBlocks.count) 个视觉块 + prompt \(promptIds.count) tokens；标签：\(refVisionLabels.joined(separator: " | "))")
            }
        } else {
            if noSubtitleOn {
                let suppressIds = tokenizer.encode(text: "no subtitles", addSpecialTokens: false).map { Int32($0) }
                presentItems = [.text(promptIds), .text(suppressIds)]
                log("prompt 编码：\(promptIds.count) tokens + 无字幕抑制 \(suppressIds.count) tokens")
            } else {
                presentItems = [.text(promptIds)]
            }
        }
        log("prompt 编码：\(promptIds.count) tokens")

        var textHidden: MLXArray!
        var textHiddenNeg: MLXArray? = nil
        // CFG 开关预判（与第三分支 slCfg.cfgScale 同一公式：仅 NA_H3_SELFLIFT_CFG 显式 >0 时编码
        // negative 条件行；缺省 0=关闭，避免 turbo LoRA 下白跑双路）。此处需在文本编码前决定。
        let slCfgScaleNeeded = (ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_CFG"].flatMap(Double.init) ?? 0.0) > 0
        autoreleasepool {
            var teWeights: H3Weights? = try! H3Weights(url: wURL("text_encoder.safetensors"))
            teWeights?.cacheEnabled = false
            var textEncoder: H3TextEncoder? = try! H3TextEncoder.load(teWeights!)
            if useRef2VA { try! textEncoder!.loadVision(teWeights!) }
            let encoded = try! textEncoder!.encodeItems(presentItems)
            if useRef2VA { textTagsForLayout = encoded.tags }
            textHidden = encoded.hidden
            // ★ CFG negative 条件行：官方 SelfLiftH3Sampler 默认 negative = 空 prompt。
            //   行数通常远小于 positive，后续在 DiT refine 后 pad 到与 positive 相同 textLen，
            //   使 negative forward 可复用同一 layout/rope/plan（时间坐标严格一致）。
            if slCfgScaleNeeded {
                var negIds = tokenizer.encode(text: "", addSpecialTokens: true).map { Int32($0) }
                if negIds.isEmpty {
                    negIds = tokenizer.encode(text: " ", addSpecialTokens: true).map { Int32($0) }
                }
                if negIds.isEmpty {
                    negIds = [Int32(0)]   // 兜底：词表 id 0 必存在；负文本内容不影响 CFG 方向
                }
                let encNeg = try! textEncoder!.encodeItems([.text(negIds)])
                textHiddenNeg = encNeg.hidden
                MLX.eval(textHiddenNeg!)
                log("SelfLift CFG negative：空文本 → \(negIds.count) tokens，hidden \(encNeg.hidden.shape)")
            }
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
                                  keyframes: useRef2VA ? [] : [.first, .last],
                                  frameCount: frameCount,
                                  refs: refBlocks)
        if useRef2VA {
            layout.textTags = textTagsForLayout
            log("ref2va 布局：text \(textHidden.shape[0]) + \(refBlocks.count) 参考块 + video/audio 目标")
        }

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
        var refinedNeg: MLXArray? = nil
        autoreleasepool {
            dit!.precomputeAdaln(ts: tsList)
            refined = dit!.refineText(textHidden)
            // ★ CFG negative 经同一 refineText 变换后 pad 到 positive 相同行数（= textLen），
            //   复用同一 layout/rope/plan：PackedLayout 的 text 段固定占 textLen 行，negative
            //   行数不足会平移后续 video/audio 行的时间坐标，两路 forward 不在同一坐标系。
            if var neg = textHiddenNeg {
                MLX.eval(neg)
                if neg.shape[0] < refined.shape[0] {
                    let pad = MLXArray.zeros([refined.shape[0] - neg.shape[0], neg.shape[1]], dtype: neg.dtype)
                    neg = concatenated([neg, pad], axis: 0)
                } else if neg.shape[0] > refined.shape[0] {
                    neg = neg[0..<refined.shape[0], 0..<neg.shape[1]]
                }
                refinedNeg = dit!.refineText(neg)
                MLX.eval(refinedNeg!)
                log("SelfLift CFG negative refined：\(refined.shape) → pad → \(refinedNeg!.shape)")
            }
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
        // ── 3.1 SelfLift 第三分支（H3 一采渐进式采样，实现在 SelfLiftH3-(h3专属).swift）─────
        // 低分（×0.5）跑 ts 次 NFE → 零 NFE 过渡块（官方 H3 路径 nearest 直接 latent 提升 +
        // VAE 像素锚点一致修正，ρ=0.6）→ 升回全分辨率 → 高分跑 N-ts 次 NFE；
        // 不变量：低分 NFE + 高分 NFE = N（总 NFE 与单遍路径一致，加速来自低分单步算力下降）。
        // ts 不写死：transitionStep = 0 → 官方 75% 规则按 N 推导（N 来自面板第一阶段总步数滑杆）。
        // 过渡块的像素锚点需要在采样中途做一次 VAE decode→encode，故此处临时加载 video_vae，
        // 过渡块跑完立即置 nil + clearCache（与 1./2. 段 keyframe 编码同一「用后即释」范式）。
        if selfLiftEnabled, stage1SegmentCount >= 2 {
            // 显式覆盖仅 transitionStep = 0（自动）；lowResScale/rho/wMin/wMax 取 SelfLiftConfig 默认＝官方 H3 建议起点；
            // NA_H3_SELFLIFT_RHO / NA_H3_SELFLIFT_WMIN / NA_H3_SELFLIFT_WMAX 可覆盖。
            // 默认 rho=0.0（官方默认）：learned upscaler 纯 z_lat 提升，不混合像素锚点。
            // 决定性实验（NA_H3TEST=26，1344×768×39 帧同 seed 同场景）：learned+rho=0 前后段逐帧全干净，
            // 而 rho=0.25/0.6/0.9 均有 z_pix VAE 往返伪影注入（运动剧烈后段背景角色双轮廓/鬼影）；
            // 官方 SelfLiftH3Sampler 默认即 rho=0，像素锚点为可选增强（默认关闭）。
            // NA_H3_SELFLIFT_RHO / NA_H3_SELFLIFT_WMIN / NA_H3_SELFLIFT_WMAX 可覆盖。
            let envRho = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_RHO"].flatMap(Double.init)
            let envWMin = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_WMIN"].flatMap(Float.init)
            let envWMax = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_WMAX"].flatMap(Float.init)
            let slCfgBase = SelfLiftConfig(enabled: true,
                                           transitionStep: 0,
                                           rho: envRho ?? 0.0,
                                           wMin: envWMin ?? 1.0,
                                           wMax: envWMax ?? 1.0)
            // 解耦模式（本工程调试扩展，A/B 用；默认关闭，不触碰默认值）：
            //   NA_H3_SELFLIFT_DECOUPLE=1              开启：低清/高清两条独立曲线，σ_k 与 σ_next 解耦
            //   NA_H3_SELFLIFT_DECOUPLE_LOW_STEPS=3    固定低清步数 L（>0 时不再由 σ_k 反推，低清终点直接取
            //                                          L 步曲线实际终点 σ_{L-1} 作为「过渡用的值」，无需过渡补 Euler）
            //   NA_H3_SELFLIFT_DECOUPLE_LOW_K=0.9     低清终点 σ_k（仅 LOW_STEPS 缺省时用于反推 L；2026-09-17 由 0.7 上调至浅区消除深跑固化双影，0.9 → L=2、σ_k≈0.9231）
            //   NA_H3_SELFLIFT_DECOUPLE_HIGH_START=…   高清起点 σ_next（缺省 = 低清终点，浅起点直接重加噪）
            //   NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS=3   高清独立步数（2026-09-17 v2 由 6 改回 3，等距曲线下 3 步即达安全残留）
            var slCfg = slCfgBase
            let decoupleEnv = ProcessInfo.processInfo.environment
            if decoupleEnv["NA_H3_SELFLIFT_DECOUPLE"] == "1" {
                slCfg.decoupleSigmaK = decoupleEnv["NA_H3_SELFLIFT_DECOUPLE_LOW_K"].flatMap(Double.init) ?? 0.7
                slCfg.decoupleLowSteps = decoupleEnv["NA_H3_SELFLIFT_DECOUPLE_LOW_STEPS"].flatMap(Int.init) ?? 0
                slCfg.decoupleSigmaNext = decoupleEnv["NA_H3_SELFLIFT_DECOUPLE_HIGH_START"].flatMap(Double.init)
                slCfg.decoupleHighSteps = decoupleEnv["NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS"].flatMap(Int.init) ?? 6
                log("SelfLift 解耦开关：σ_k=\(slCfg.decoupleSigmaK!)"
                    + (slCfg.decoupleSigmaNext.map { ", σ_next=\($0)" } ?? "（σ_next=低清终点）")
                    + (slCfg.decoupleLowSteps > 0 ? "，固定低清步数=\(slCfg.decoupleLowSteps)（终点即过渡值）" : "")
                    + "，高清步数=\(slCfg.decoupleHighSteps)")
            }
            // ★ CFG 引导（2026-09-18 重影根因修复，对齐官方 SelfLiftH3Sampler cfg=5.0）：
            //   NA_H3_SELFLIFT_CFG 显式覆盖；缺省 0（关闭）——2026-09-18 实测 cfg=5.0 对 600 turbo
            //   LoRA（蒸馏模型，训练目标 cfg=1）过强：高对比度马赛克/过曝/网格噪点，画面崩坏；
            //   官方 cfg=5.0 面向原版 H3（非 turbo）。保留机制供显式实验（小值 1.5~3.0 可试）。
            let envCfgScale = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_CFG"].flatMap(Double.init) ?? 0.0
            slCfg.cfgScale = envCfgScale
            if envCfgScale > 0 {
                log("SelfLift CFG：cfg=\(envCfgScale)（每步 positive+negative 双 forward；注意 turbo LoRA 蒸馏模型 cfg 过大会过冲）")
            } else {
                log("SelfLift CFG：关闭（cfg=0，仅 positive 单路；NA_H3_SELFLIFT_CFG=1.5~3.0 可显式实验）")
            }
            let slTs = slCfg.resolvedTransitionStep(totalSteps: stage1SegmentCount)
            if slCfg.decoupleSigmaK != nil {
                log("SelfLift 第三分支：开启（解耦模式，低清/高清独立曲线见 Runner 调度日志；面板参考 N=\(stage1SegmentCount)，原调度 ts=\(slTs) 已绕开）")
            } else {
                log("SelfLift 第三分支：开启（N=\(stage1SegmentCount) → ts=\(slTs)：低分 NFE=\(slTs) + 高分 NFE=\(stage1SegmentCount - slTs)，总 NFE=\(stage1SegmentCount)；低分倍率 \(slCfg.lowResScale)、ρ=\(slCfg.rho)、wMin=\(slCfg.wMin)/wMax=\(slCfg.wMax)）")
            }
            // 像素锚点 VAE（decoder + encoder，bf16 实测 ≈5.3GB）只服务过渡块的
            // decode→encode 往返；rho=0（needsPixelAnchor=false）时 liftPixelAnchor 为
            // nil、不会调用 hooks 的 decode/encode，故按需加载，rho=0 直接省掉这 5.3GB。
            // lowOnly 模式（IC 链路）连过渡块都不跑，同样跳过 VAE 加载。
            var slVaeWeights: H3Weights?
            var slVae: H3VAE?
            if slCfg.needsPixelAnchor && !selfLiftLowOnly {
                slVaeWeights = try H3Weights(url: wURL("video_vae.safetensors"))
                slVaeWeights?.cacheEnabled = false
                slVae = try H3VAE.load(slVaeWeights!)
                h3MemLog("SelfLift 像素锚点 VAE 已加载（≈5.3GB，仅过渡块使用）")
            } else {
                h3MemLog("SelfLift ρ=0：跳过像素锚点 VAE 加载（省 ≈5.3GB）")
            }
            let slHooks = SelfLiftH3Hooks(
                decodeToPixels: { px in
                    guard let vae = slVae else {
                        fatalError("SelfLift: needsPixelAnchor=false（rho=0）时像素锚点解码不应被调用")
                    }
                    return vae.decoder.decode(px)
                },
                encodeToLatent: { z in
                    guard let vae = slVae else {
                        fatalError("SelfLift: needsPixelAnchor=false（rho=0）时像素锚点编码不应被调用")
                    }
                    return vae.encoder.encodeVideo(z)
                },
                // ★ 像素锚点 VAE（decoder + encoder，bf16 实测 ≈5.3GB）只服务过渡块；
                //   Runner 在过渡块出口屏障（已 Stream.gpu.synchronize()）后回调此处提前置 nil，
                //   把这 5.3GB 从高分段（全分辨率 ~39k token × 高分 NFE）基线里摘掉。
                //   只放手：**不 clearCache、不动 cacheLimit**（撤销依据见 SelfLiftH3 入口屏障注释 ①②）；
                //   放掉的权重 buffer 落进空闲池，被高分段每步中间量命中复用。
                releasePixelVAE: { slVae = nil; slVaeWeights = nil })
            let slOut = try runH3Stage1WithSelfLift(
                dit: dit!,
                textStates: refined,
                condRowsFull: condRows,
                refBlocks: refBlocks,
                textTags: textTagsForLayout,
                latT: Int(latentT),
                latH: latH,
                latW: latW,
                latC: latC,
                audioT: audioT,
                textLen: UInt32(textHidden.shape[0]),
                frameCount: frameCount,
                sigmas: sigmas,
                shiftV: sv,
                shiftA: sa,
                seed: seed,
                sparsePolicy: sparsePolicy,
                attentionBroadcastK: pabK,
                cfg: slCfg,
                hooks: slHooks,
                log: log,
                lowOnly: selfLiftLowOnly,
                textStatesNeg: refinedNeg)
            videoX = slOut.videoX
            audioX = slOut.audioX
            MLX.eval(videoX, audioX)
            slVae = nil
            slVaeWeights = nil
            MLX.Memory.clearCache()
            h3MemLog("SelfLift 像素锚点 VAE 已释放（回到全分辨率 \(latW)×\(latH) latent）")
        } else {
            if !selfLiftEnabled {
                log("SelfLift 第三分支：关闭，走原单条 stage1 循环（单遍直出）")
            } else {
                log("SelfLift 第三分支：N=\(stage1SegmentCount) < 2，无法切分低分/高分两段，走原单条 stage1 循环")
            }
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
        }   // ← 原单条 stage1 循环结束（SelfLift 第三分支的开/else 收口）
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
        if selfLiftLowOnly, selfLiftEnabled, stage1SegmentCount >= 2 {
            // SelfLift lowOnly：runner 已直接返回低清 NCDHW latent [1,C,T,latHl,latWl]（半清网格），
            // 不再是全清 rows——stage2 放大与 rows→zlat reshape 全部跳过；
            // 高分（升频×2 + 精修）在 LTX 二采侧完成（fullResInput=false）。
            zForDecode = videoX
            MLX.eval(zForDecode)
            log("SelfLift lowOnly：直接采用低清 zForDecode \(zForDecode.shape)（半清，高分交由 LTX 二采升频×2 + IC 精修）")
        } else if let s2cfg = stage2, s2cfg.scale > 1 {
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

        // ── 4.9 H3→LTX latent 直通适配器（可选）：在 VAE 解码之前拦截 clean latent ──
        // 官方 H3-to-LTX-Latent-Adapter：H3 归一化 latent →［时间线性重采样 + 最近邻打包(3 槽)
        // → 2× pixel-unshuffle → Conv3D 残差主干］→ LTX 归一化 latent，可直接进 LTX Stage2 refine。
        // 一步替代「H3 VAE decode → 像素 → LTX VAE encode」两步像素往返。
        // 适配器不改变采样结果，失败时自动回退原像素桥路径（不影响出片）。
        var adapterActive = false
        var adapterSkipDecode = false
        var silentPath = proResOutput ? (outPath + ".silent.mov") : (outPath + ".silent.mp4")
        var fCount = 0
        var h = 0
        var w = 0
        if let adapter = h3ToLTXAdapter, let bridge = stage2MemBridge {
            do {
                let tA = Date()
                let latShape = zForDecode.shape
                let ltxLatent = try adapter.convert(h3LatentNCDHW: zForDecode, pixelFrames: Int(frameCount))
                bridge.ltxHalfLatent = ltxLatent
                bridge.adapterPixelWidth = latShape[4] * H3ToLTXAdapterConst.h3SpatialCompression
                bridge.adapterPixelHeight = latShape[3] * H3ToLTXAdapterConst.h3SpatialCompression
                adapterActive = true
                adapterSkipDecode = adapterSkipH3Decode
                let aSec = String(format: "%.1f", Date().timeIntervalSince(tA))
                log("★ H3→LTX 适配器直通完成（\(aSec)s）：H3 latent \(latShape) → LTX latent \(ltxLatent.shape)"
                    + "（等效像素 \(bridge.adapterPixelWidth)×\(bridge.adapterPixelHeight) @ \(frameCount) 帧）"
                    + "，已省去 LTX VAE 编码\(adapterSkipH3Decode ? "与 H3 VAE 解码" : "")")
            } catch {
                adapterActive = false
                adapterSkipDecode = false
                stage2MemBridge?.ltxHalfLatent = nil
                log("⚠️ H3→LTX 适配器不可用（\(error.localizedDescription)），回退像素桥路径（H3 decode → LTX encode）")
            }
        }

        // ── 5. VAE 解码 + 写无声视频（proResOutput=true → .mov/ProRes422，否则 .mp4/h264；同一池内完成：像素大数组写完立即释放，再进音频段） ──
        // adapter 直通且要求跳解码时不进此段：无 stage1 mp4，音轨改由 5.1 单独落 wav 承载。
        if adapterActive && adapterSkipDecode {
            h = stage2MemBridge?.adapterPixelHeight ?? 0
            w = stage2MemBridge?.adapterPixelWidth ?? 0
            fCount = Int(frameCount)
            // 像素为空，但尺寸/帧率元数据仍需回填：二采据此推输出帧率与日志几何。
            stage2MemBridge?.width = w
            stage2MemBridge?.height = h
            stage2MemBridge?.frameCount = fCount
            stage2MemBridge?.fps = Int(H3Const.fps)
            log("⏭️ adapter 直通模式：跳过 H3 VAE 解码，不写 stage1 视频文件"
                + "（像素几何 \(w)×\(h) @ \(fCount) 帧，仅供 LTX 侧尺寸推导）")
        } else {
        var decWeights: H3Weights? = try H3Weights(url: wURL("video_vae.safetensors"))
        var decoder: H3VAE? = try H3VAE.load(decWeights!)
        try autoreleasepool {
            let pixels = decoder!.decode(zForDecode)
            MLX.eval(pixels)
            log("VAE 解码完成：\(pixels.shape)")
            fCount = pixels.shape[2]
            h = pixels.shape[3]
            w = pixels.shape[4]
            // 内存直通（方案 C）：stage1 像素帧不落盘，由 bridge 持有供像素桥直接 VAE 编码；
            // 落盘为 ProRes 422 .mov（proResOutput=true，10bit v210；无二采时即最终产物，
            // 有二采时作高质量预览/音轨源）。
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
        log("无声视频已写出：\(silentPath)（\(fCount) 帧 \(w)×\(h)），VAE 解码器与像素数组已释放")
        }

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
            if adapterActive && adapterSkipDecode {
                // adapter 直通：无 stage1 视频可混流，保留 wav 作为音轨载体，
                // 由 LTX 二采在最终出片时混流（见 bridge.audioTrackPath）。
                stage2MemBridge?.audioTrackPath = wavPath
                log("音轨载体已保留（adapter 直通，无 stage1 视频文件）：\(wavPath)")
            } else {
                try muxAudio(videoPath: silentPath, wavPath: wavPath, to: outPath, proRes: proResOutput)
                log("音轨已混入：\(outPath)")
                try? FileManager.default.removeItem(atPath: wavPath)
                try? FileManager.default.removeItem(atPath: silentPath)
            }
        } catch {
            // 音频失败不丢视频：无声视频兜底保留为 outPath
            log("⚠️ 音频解码/混流失败（\(error.localizedDescription)），保留无声视频")
            if adapterActive && adapterSkipDecode {
                log("   （adapter 直通模式无 stage1 视频，需由 LTX 侧无声产物兜底）")
            } else {
                if FileManager.default.fileExists(atPath: outPath) {
                    try? FileManager.default.removeItem(atPath: outPath)
                }
                try? FileManager.default.moveItem(atPath: silentPath, toPath: outPath)
            }
        }
        // 总耗时：分钟进制度（≥60s 显示 Xm Ys，否则仅 Xs）
        let totalSec = Int(Date().timeIntervalSince(t0))
        let totalStr = totalSec >= 60 ? "\(totalSec / 60)m \(totalSec % 60)s" : "\(totalSec)s"
        if adapterActive && adapterSkipDecode {
            let latShape = stage2MemBridge?.ltxHalfLatent?.shape ?? []
            log("H3 阶段完成（adapter 直通，未解码像素）：LTX latent \(latShape)，"
                + "像素几何 \(w)×\(h) @ \(fCount) 帧，音轨载体 \(stage2MemBridge?.audioTrackPath ?? "无")，总耗时 \(totalStr)")
        } else {
            log("视频写出完成：\(outPath)（\(fCount) 帧 \(w)×\(h)，总耗时 \(totalStr)）")
        }
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

// MARK: - ref2va 参考图工具

/// 读取图片像素尺寸（不解码整图，供 reference canvas 计算）。
func h3ImagePixelSize(_ path: String) -> (w: Int, h: Int)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
          let w = props[kCGImagePropertyPixelWidth] as? Int,
          let h = props[kCGImagePropertyPixelHeight] as? Int,
          w > 0, h > 0 else { return nil }
    return (w, h)
}
