//
//  HiDreamPipeline.swift
//  无限画布
//
//  HiDream-O1-Image 扩散推理管线：
//  顶层容器 HiDreamBackbone（组装骨架的 VisionModel/LanguageModel + 扩散头）、
//  T2I 文本采样、rope fix-point 位置编码、4D 注意力掩码、forward_generation、
//  28 步 FlashFlowMatch 去噪主循环与图像还原。
//
//  权重键名对齐：
//    model.safetensors          → vision_tower.* / language_model.*
//    extras/custom_heads.safetensors → t_embedder1.* / x_embedder.* / final_layer2.*
//

import Foundation
import AppKit
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXVLM
import Tokenizers

//  [改名] 自 HiDreamPipeline.swift（方案 b 类，HiDream 专用）。
//  [拆分] HiDreamBackbone 顶层容器 + hidreamSanitizeWeights 已移至 HiDreamBackbone（HiDream专用）.swift。
//  [提升] hidreamGetRopeIndexFixPoint / hidreamGetRopeIndexFixPointMulti 已移至 2网络算子/网络算子-位置编码.swift。
//  [提升] hidreamLoadCGImage / hidreamResizeCropRGBA 已上移 通用公共函数/模型公共函数.swift（与 loadImageBCFHW 重复，未来合并 loadImageResizeCrop）。
//  [规划] HiDreamCompiledForwardCache 未来可泛型化 CompiledForwardCache<Owner>（方案 c 类，本次不做）。
// MARK: - T2I 文本采样

struct HiDreamT2ISample {
    var inputIds: [Int32]           // 文本 token（含 boi + tms）
    var inputIdsPad: [Int32]        // 文本 + vision_start + image 占位
    var positionIds: MLXArray       // [3, 1, S] int32（fix-point 版）
    var mask4d: MLXArray            // [1, 1, S, S] float32 加性掩码
    var tgtIdx: [Int32]             // vinput 在 x_pred 中的索引（txt_seq_len .. S-1）
}

/// 与 py build_t2i_text_sample 对齐（T2I，单图）。
/// tokenizer 须为 Qwen3VL tokenizer；prompt 为纯文本。
func hidreamBuildT2ISample(
    prompt: String,
    height: Int,
    width: Int,
    tokenizer: any Tokenizers.Tokenizer,
    qwenConfig: Qwen3VLConfiguration
) throws -> HiDreamT2ISample {
    let const = HiDreamDiffusionConstants.self

    // 与 py processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 等价
    let templateText = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        + "<|boi_token|>"
        + String(repeating: "<|tms_token|>", count: const.timestepTokenNum)

    let inputIds = tokenizer.encode(text: templateText, addSpecialTokens: false)
    let txtSeqLen = inputIds.count

    // 分辨率对齐（与 py 一致，网格以 32 patch 换算）
    let (h, w) = closestResolution(height: height, width: width, predefined: HiDreamDiffusionConstants.predefinedResolutions)
    let hPatches = h / const.patchSize
    let wPatches = w / const.patchSize
    let imageLen = hPatches * wPatches

    // vision 占位 token：首位置 vision_start，其余 image_token
    var visionTokens = [Int32](repeating: Int32(qwenConfig.imageTokenId), count: imageLen)
    visionTokens[0] = Int32(qwenConfig.visionStartTokenId)

    let inputIdsPad = inputIds.map { Int32($0) } + visionTokens
    let allSeqLen = inputIdsPad.count

    // fix-point 位置编码（T2I 单图，skip_vision_start_token=[1]）
    let positionIds = hidreamGetRopeIndexFixPoint(
        inputIdsPad: inputIdsPad,
        imageGridTHW: (t: 1, h: hPatches, w: wPatches),
        spatialMergeSize: 1,
        skipVisionStartToken: [1],
        fixPoint: 4096)   // [3, 1, S]

    // token_types：tms 位置起标 1，tms 单独标 3
    var tokenTypes = [Int32](repeating: 0, count: allSeqLen)
    let bgn = txtSeqLen - const.timestepTokenNum
    if bgn >= 0, bgn + imageLen + const.timestepTokenNum <= allSeqLen {
        for i in bgn ..< bgn + imageLen + const.timestepTokenNum {
            tokenTypes[i] = 1
        }
        for i in txtSeqLen - const.timestepTokenNum ..< txtSeqLen {
            tokenTypes[i] = 3
        }
    }

    let mask4d = hidreamBuildAttentionMaskFast(tokenTypesBin: tokenTypes.map { $0 > 0 })

    // vinput 索引 = token_types == 1 的位置（txt_seq_len .. S-1）
    var tgtIdx: [Int32] = []
    for (i, tt) in tokenTypes.enumerated() where tt == 1 {
        tgtIdx.append(Int32(i))
    }

    return HiDreamT2ISample(
        inputIds: inputIds.map { Int32($0) },
        inputIdsPad: inputIdsPad,
        positionIds: positionIds,
        mask4d: mask4d,
        tgtIdx: tgtIdx)
}
// MARK: - forward_generation

/// 与 py forward_generation 对齐：t_emb scatter 到 tms 位置、v_emb 拼接后走语言模型主干
/// （不含 lm_head），最后 final_layer2 出 x_pred。返回 [1, S, 3072]。
/// tms 掩码与 v 嵌入均由调用方在采样循环外预构造（只依赖 inputIds / z 或 refPatches 的常量部分）。
func hidreamForwardGeneration(
    backbone: HiDreamBackbone,
    inputIds: [Int32],
    inputsEmbeds: MLXArray,      // [1, txt_seq_len, 4096]（precompute 常量）
    tmsMaskBroadcast: MLXArray,  // [1, txt_seq_len, 4096] tms 注入掩码（循环外预构造）
    positionIds: MLXArray,       // [3, 1, S]
    mask4d: MLXArray,            // [1, 1, S, S]
    vEmb: MLXArray,              // [1, image_len(+ref), 4096] 已嵌入（xEmbedder 由调用方拆分）
    timestepPixelDiT: Float,     // 1 - step_t/1000
    visualMask: MLXArray? = nil,         // deepstack 注入位置（编辑模式）
    deepstackEmbeds: [MLXArray]? = nil,  // 视觉塔中间层特征（编辑模式）
    tgtIdx: MLXArray? = nil              // 非 nil 时 finalLayer2 仅算目标位置（省全序列计算）
) -> MLXArray {
    let const = HiDreamDiffusionConstants.self

    // t_emb: [1, 4096]
    let tEmb = backbone.tEmbedder1(MLXArray(timestepPixelDiT))

    // tms 注入：掩码循环外预构造，此处仅 broadcast t_emb 并 where 替换
    let tEmbExpanded = tEmb
        .expandedDimensions(axis: 1)          // [1, 1, 4096]
    let tEmbBroadcast = broadcast(tEmbExpanded, to: inputsEmbeds.shape) // [1, txt_len, 4096]
    let newEmbeds = MLX.where(tmsMaskBroadcast .> 0.5, tEmbBroadcast, inputsEmbeds)

    // v 嵌入已由调用方构造（T2I：xEmbedder(z)；编辑：xEmbedder(z) + 预计算 refEmb）
    let fullEmbeds = concatenated([newEmbeds, vEmb], axis: 1)              // [1, S, 4096]
    let S = fullEmbeds.dim(1)
    let placeholder = MLXArray.zeros([1, S], dtype: .int32)

    // 语言模型主干（不含 lm_head），传 4D 掩码 + fix-point 位置
    let h = backbone.languageModel.model(
        placeholder,
        cache: nil,
        inputEmbeddings: fullEmbeds,
        mask: mask4d,
        positionIds: positionIds,
        visualMask: visualMask,
        deepstackEmbeds: deepstackEmbeds)

    // finalLayer2 仅算目标位置（tgt 通常远小于全序列，省大量 MLP/linear 计算）
    let finalInput = tgtIdx != nil ? h.take(tgtIdx!, axis: 1) : h
    return backbone.finalLayer2(finalInput)                               // [1, N, 3072]
}

// MARK: - 编译单步前向（MLX.compile，对齐 ltx2.5/Sampler.swift 模式）

/// 把 HiDream 单步前向（z + t → x_pred）编译为融合内核（shape 每步固定）。
/// 常量（inputsEmbeds / tmsMaskBroadcast / positionIds / mask4d / refEmb /
/// visualMask / deepstackEmbeds / tgtIdx）由闭包捕获，运行时只换 z 与 timestep。
typealias CompiledHiDreamForward = @Sendable (MLXArray, MLXArray) -> MLXArray  // (z, timestepPixelDiT)

/// T2I 专用编译前向：恒无参考图、无视觉塔注入，路径固定（tgtIdx 恒取目标位置）。
/// 所有常量均为非 Optional，闭包捕获后确定性执行；无 if let 分支、无强制解包，
/// 风格对齐 ltx2.5/Sampler.swift 的 makeCompiledDitForward。
func makeCompiledHiDreamT2IForward(
    backbone: HiDreamBackbone,
    inputsEmbeds: MLXArray,
    tmsMaskBroadcast: MLXArray,
    positionIds: MLXArray,
    mask4d: MLXArray,
    tgtIdx: MLXArray
) -> CompiledHiDreamForward {
    MLX.compile { (z: MLXArray, tEmbScalar: MLXArray) -> MLXArray in
        let tEmb = backbone.tEmbedder1(tEmbScalar)
        let tEmbExpanded = tEmb.expandedDimensions(axis: 1)          // [1, 1, 4096]
        let tEmbBroadcast = broadcast(tEmbExpanded, to: inputsEmbeds.shape)
        let newEmbeds = MLX.where(tmsMaskBroadcast .> 0.5, tEmbBroadcast, inputsEmbeds)
        let vEmb = backbone.xEmbedder(z).asType(inputsEmbeds.dtype)  // T2I：仅 z 嵌入，无 ref
        let fullEmbeds = concatenated([newEmbeds, vEmb], axis: 1)    // [1, S, 4096]
        let S = fullEmbeds.dim(1)
        let placeholder = MLXArray.zeros([1, S], dtype: .int32)
        let h = backbone.languageModel.model(
            placeholder,
            cache: nil,
            inputEmbeddings: fullEmbeds,
            mask: mask4d,
            positionIds: positionIds,
            visualMask: nil,
            deepstackEmbeds: nil)
        let finalInput = h.take(tgtIdx, axis: 1)                     // T2I 恒取目标位置
        return backbone.finalLayer2(finalInput)
    }
}

/// 编辑专用编译前向：恒有参考图 / 视觉塔特征，路径固定（直接 concat refEmb、
/// 直接传 deepstack、恒取目标位置）。无 Optional、无 if let / `!` 分支。
func makeCompiledHiDreamEditForward(
    backbone: HiDreamBackbone,
    inputsEmbeds: MLXArray,
    tmsMaskBroadcast: MLXArray,
    positionIds: MLXArray,
    mask4d: MLXArray,
    refEmb: MLXArray,
    visualMask: MLXArray,
    deepstackEmbeds: [MLXArray],
    tgtIdx: MLXArray
) -> CompiledHiDreamForward {
    MLX.compile { (z: MLXArray, tEmbScalar: MLXArray) -> MLXArray in
        let tEmb = backbone.tEmbedder1(tEmbScalar)
        let tEmbExpanded = tEmb.expandedDimensions(axis: 1)          // [1, 1, 4096]
        let tEmbBroadcast = broadcast(tEmbExpanded, to: inputsEmbeds.shape)
        let newEmbeds = MLX.where(tmsMaskBroadcast .> 0.5, tEmbBroadcast, inputsEmbeds)
        let vEmb = concatenated(                                     // 编辑：z 嵌入 + ref 嵌入
            [backbone.xEmbedder(z).asType(inputsEmbeds.dtype), refEmb], axis: 1)
        let fullEmbeds = concatenated([newEmbeds, vEmb], axis: 1)    // [1, S, 4096]
        let S = fullEmbeds.dim(1)
        let placeholder = MLXArray.zeros([1, S], dtype: .int32)
        let h = backbone.languageModel.model(
            placeholder,
            cache: nil,
            inputEmbeddings: fullEmbeds,
            mask: mask4d,
            positionIds: positionIds,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds)
        let finalInput = h.take(tgtIdx, axis: 1)                     // 编辑恒取目标位置
        return backbone.finalLayer2(finalInput)
    }
}

/// 编译图持久缓存：同一 backbone 实例 + 同一 shape 直接复用编译产物，
/// 跨次生成（同 shape）不重编译（对齐 LTX CompiledForwardCache / CompiledGraphCache 多槽）。
final class HiDreamCompiledForwardCache {
    static let shared = HiDreamCompiledForwardCache()
    private let cache = CompiledGraphCache<HiDreamBackbone, CompiledHiDreamForward>()

    /// T2I 编译前向入口（shapeKey 前缀 hidream-t2i-）
    func getT2I(backbone: HiDreamBackbone, key: String) -> CompiledHiDreamForward? {
        cache.get(owner: backbone, key: key)
    }

    func storeT2I(_ f: @escaping CompiledHiDreamForward, backbone: HiDreamBackbone, key: String) {
        cache.store(f, owner: backbone, key: key)
    }

    /// 编辑编译前向入口（shapeKey 前缀 hidream-edit-）
    func getEdit(backbone: HiDreamBackbone, key: String) -> CompiledHiDreamForward? {
        cache.get(owner: backbone, key: key)
    }

    func storeEdit(_ f: @escaping CompiledHiDreamForward, backbone: HiDreamBackbone, key: String) {
        cache.store(f, owner: backbone, key: key)
    }

    func clear() {
        cache.clear()
    }
}

// MARK: - 主循环（generate）

/// HiDream-O1-Image T2I 生成入口。
/// - Parameters:
///   - prompt: 文本提示
///   - height/width: 请求尺寸（内部对齐到预定义分辨率）
///   - steps: 去噪步数（默认 28）
///   - seed: 随机种子（默认 32）
///   - onProgress: 每步回调（0..1）
/// - Returns: 归一化 [0,1] 的 [H, W, 3] float32 图像
func hidreamGenerate(
    backbone: HiDreamBackbone,
    tokenizer: any Tokenizers.Tokenizer,
    qwenConfig: Qwen3VLConfiguration,
    prompt: String,
    height: Int,
    width: Int,
    steps: Int = 28,
    seed: UInt64 = 32,
    onProgress: ((Float) -> Void)? = nil,
    isCancelled: () -> Bool = { false }   // 每步前检查，true → 抛取消错误（队列取消用）
) throws -> MLXArray {
    let const = HiDreamDiffusionConstants.self

    let sample = try hidreamBuildT2ISample(
        prompt: prompt, height: height, width: width,
        tokenizer: tokenizer, qwenConfig: qwenConfig)
    let txtSeqLen = sample.inputIds.count
    let imageLen = sample.tgtIdx.count
    let (h, w) = closestResolution(height: height, width: width, predefined: HiDreamDiffusionConstants.predefinedResolutions)
    let hPatches = h / const.patchSize
    let wPatches = w / const.patchSize

    // 文本嵌入常量（T2I：无 vision 注入）
    let inputsEmbeds = backbone.languageModel.model.embedTokens(
        MLXArray(sample.inputIds)[.newAxis, 0...])                          // [1, txt_len, 4096]

    // tms 注入掩码常量（inputIds 固定，循环外预构造一次）
    let tmsMaskBroadcast = hidreamBuildTmsMask(
        inputIds: sample.inputIds,
        tmsTokenID: Int32(const.tmsTokenID),
        shape: inputsEmbeds.shape)

    // 初始噪声 → patchify（官方：noise_scale_start × N(0,1)）
    let rngKey = MLXRandom.key(seed)
    let noise = MLXRandom.normal([3, h, w], key: rngKey) * MLXArray(const.noiseScaleDefault)
    var z = patchifySpatial2DBCHW(noise).expandedDimensions(axis: 0)        // [1, N, 3072]

    // scheduler
    let scheduler = FlashFlowMatchScheduler()
    scheduler.setTimesteps(numInferenceSteps: steps, customTimesteps: const.defaultTimesteps)

    let tgtIdx = MLXArray(sample.tgtIdx)
    let sigmaT = Float(const.tEps)
    let noiseClipStd: Float = 2.5

    // MLX.compile 单步前向：shape 固定则复用编译图（对齐 LTX Sampler.swift 编译缓存模式）
    let shapeKey =
        "hidream-t2i-\(inputsEmbeds.shape)-\(sample.positionIds.shape)-\(sample.mask4d.shape)-\(tgtIdx.shape)-v\(imageLen)-h\(h)-w\(w)"
    var forward: CompiledHiDreamForward? = nil
    var preWarmXPred: MLXArray? = nil   // 编译预热前向输出（首步 x0 预测），首步直接复用
    if let cached = HiDreamCompiledForwardCache.shared.getT2I(backbone: backbone, key: shapeKey) {
        stageEnter(phase: .sampling,
                   detail: "复用已编译 HiDream 前向（shape 不变，跳过编译）",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        forward = cached
    } else {
        stageEnter(phase: .sampling,
                   detail: "MLX.compile 编译 HiDream 前向中（首次约 30-90s）...",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        let tC = Date()
        let cf = makeCompiledHiDreamT2IForward(
            backbone: backbone,
            inputsEmbeds: inputsEmbeds,
            tmsMaskBroadcast: tmsMaskBroadcast,
            positionIds: sample.positionIds,
            mask4d: sample.mask4d,
            tgtIdx: tgtIdx)
        // 预热：用初始噪声 latent + 第一步 timestep 触发图构建 + 内核编译
        let warmT = Date()
        let wz = cf(z, MLXArray(1.0 - Float(scheduler.timestepsNP[0]) / 1000.0))
        eval(wz)
        stageEnter(phase: .sampling,
                   detail: "  编译完成（\(Int(Date().timeIntervalSince(tC)))s，含预热 \(Int(Date().timeIntervalSince(warmT)))s）",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        HiDreamCompiledForwardCache.shared.storeT2I(cf, backbone: backbone, key: shapeKey)
        // 预热复用：预热输入=初始噪声 z+首步 timestep，其输出即首步 x0 预测，直接存为 preWarmXPred
        preWarmXPred = wz
        forward = cf
    }

    // TeaCache：相邻步 sigma 差小于阈值时复用上一步 x0，跳过整次前向（对齐 LTX Sampler）
    let teaCache = TeaCacheLegacy()
    teaCache.threshold = 0.02

    for (i, stepT) in scheduler.timestepsNP.enumerated() {
        if isCancelled() {
            throw GenerationCancelError.cancelled
        }
        // 每步独立作用域：本步中间张量（xPred/v/modelOutput/多层前向激活）在步末释放，
        // 避免整轮采样激活攒到函数结束一次性析构导致峰值虚高（对齐 LTX Sampler 每步 autoreleasepool）
        autoreleasepool {
        let tPixelDiT: Float = 1.0 - stepT / 1000.0
        let sigma = max(stepT / 1000.0, sigmaT)

        let xPred: MLXArray
        var cachedV: MLXArray? = nil
        var cachedA: MLXArray? = nil
        if i == 0, let pw = preWarmXPred {
            // 预热复用：编译预热前向（初始噪声 z+首步 t）输出即首步 x0 预测，跳过首步 DiT 前向
            xPred = pw
            stageEnter(phase: .sampling,
                       step: i + 1, total: steps,
                       detail: "    [预热复用] 第 1 步复用编译预热输出，跳过前向",
                       logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
            teaCache.store(v: pw, a: pw)
        } else if teaCache.tryReuse(sigma: sigma, outV: &cachedV, outA: &cachedA), let rv = cachedV {
            xPred = rv
            stageEnter(phase: .sampling,
                       step: i + 1, total: steps,
                       detail: "    [TeaCache] 步 \(i + 1) σ=\(sigma) 命中缓存，跳过前向",
                       logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        } else {
            // 采样循环内内存兜底：真实重前向入口先查压力分（TeaCache 命中步直出缓存不经此）
            MemoryPolicy.ensureLoose(protect: [.hiDream])
            if let f = forward {
                xPred = f(z, MLXArray(tPixelDiT))
            } else {
                xPred = hidreamForwardGeneration(
                    backbone: backbone,
                    inputIds: sample.inputIds,
                    inputsEmbeds: inputsEmbeds,
                    tmsMaskBroadcast: tmsMaskBroadcast,
                    positionIds: sample.positionIds,
                    mask4d: sample.mask4d,
                    vEmb: backbone.xEmbedder(z).asType(inputsEmbeds.dtype),
                    timestepPixelDiT: tPixelDiT,
                    tgtIdx: tgtIdx)                                    // [1, N, 3072]
            }
            teaCache.store(v: xPred, a: xPred)
        }

        let genPatches = xPred.asType(.float32)                             // [1, N, 3072]
        let zf = z.asType(.float32)
        let v = (genPatches - zf) / MLXArray(sigma)
        let modelOutput = -v

        z = scheduler.step(
            modelOutput: modelOutput,
            timestep: stepT,
            sample: z,
            sNoise: const.noiseScaleDefault,
            noiseClipStd: noiseClipStd,
            seed: seed)
            z = z.asType(.float32)
            eval(z)

            onProgress?(Float(i + 1) / Float(steps))
        }
    }

    // (z + 1) / 2 → [0,1]，还原 CHW → HWC
    let img = (z.asType(.float32) + 1.0) * 0.5                              // [1, N, 3072]
    let imgCHW = unpatchifySpatialBCHW(
        img.squeezed(axis: 0), hPatches: hPatches, wPatches: wPatches)      // [3, H, W]
    return imgCHW.transposed(1, 2, 0)                                       // [H, W, 3]
}

/// 从 config.json 解码 Qwen3VLConfiguration（供骨架与管线共用）。
func hidreamLoadQwenConfig(from dir: String) throws -> Qwen3VLConfiguration {
    let url = URL(fileURLWithPath: dir).appendingPathComponent("config.json")
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Qwen3VLConfiguration.self, from: data)
}

// MARK: - 编辑模式（图像参考 / 多图参考）

/// 编辑模式采样样本（对应 py build_edit_text_sample 返回结构）。
struct HiDreamEditSample {
    var inputIds: [Int32]        // 文本 token（processor 展开 image_pad + boi + tms）
    var inputIdsPad: [Int32]     // 文本 + tgt span + ref spans
    var positionIds: MLXArray    // [3, 1, S]（多段 fix-point）
    var mask4d: MLXArray         // [1, 1, S, S]
    var tgtIdx: [Int32]          // token_types==1（tgt，不含 tms）
    var refPatches: MLXArray     // [1, sum(N_ref), 3*32*32]（clean ref patches，扩散侧条件）
    var pixelValues: MLXArray    // [N_vision, 1536]（视觉塔输入：16 patch + temporal merge 2）
    var imageGridTHW: [THW]      // [K] 视觉塔网格（16 patch，未除 merge）
}

enum HiDreamEditError: Error {
    case invalidReferenceImage(String)
    case visionTokenMismatch(Int, Int)
    case ropeSegmentMismatch(Int, Int)
}
/// 归一化 + patchify 32（扩散侧参考条件）：RGBA → [N, 3*32*32]，值域 [-1,1]。
func hidreamPatchify32(rgba: [UInt8], w: Int, h: Int) -> [Float] {
    let hP = h / 32, wP = w / 32
    var out = [Float](repeating: 0, count: hP * wP * 3072)
    for pi in 0 ..< hP {
        for pj in 0 ..< wP {
            let dst = (pi * wP + pj) * 3072
            var o = 0
            for c in 0 ..< 3 {
                for i in 0 ..< 32 {
                    for j in 0 ..< 32 {
                        let src = ((pi * 32 + i) * w + (pj * 32 + j)) * 4 + c
                        out[dst + o] = Float(rgba[src]) / 127.5 - 1.0
                        o += 1
                    }
                }
            }
        }
    }
    return out
}

/// Qwen3VL 视觉塔预处理（等价 Qwen2VLImageProcessorFast 静态图）：
/// normalize → 16×16 patch → temporal merge（t=1 复制成 2 帧，帧内 patch 相同）。
/// RGBA → [hP*wP, 2*16*16*3]，每 patch = [t0(768) + t1(768)]。
func hidreamVisionProcess(rgba: [UInt8], w: Int, h: Int) -> [Float] {
    // 对齐 mlx-vlm Qwen3VL processor：rescale（v/127.5-1 即 (v/255-0.5)/0.5）
    // + 帧复制（tps=2）+ 2×2 interleave 块序。
    // 输出行数 = (h/16)*(w/16)，行序为 2×2 块连续（PatchMerger 的
    // reshape(-1, hidden*4) 依赖该序；posEmbeds 也按同序生成，逐行对齐）。
    // 每行布局 [C][tps][ps][ps]，与 PatchEmbed reshape(-1, 3, 2, 16, 16) 一致。
    let ps = 16, ms = 2, C = 3, tps = 2
    let hP = h / ps, wP = w / ps
    let hBlocks = hP / ms, wBlocks = wP / ms
    var out = [Float](repeating: 0, count: hP * wP * C * tps * ps * ps)
    var dst = 0
    for bh in 0 ..< hBlocks {
        for bw in 0 ..< wBlocks {
            for bi in 0 ..< ms {
                for bj in 0 ..< ms {
                    let pi = bh * ms + bi
                    let pj = bw * ms + bj
                    for c in 0 ..< C {
                        for _ in 0 ..< tps {
                            for i in 0 ..< ps {
                                for j in 0 ..< ps {
                                    let src = ((pi * ps + i) * w + (pj * ps + j)) * 4 + c
                                    out[dst] = Float(rgba[src]) / 127.5 - 1.0
                                    dst += 1
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}
// MARK: - 编辑采样构造（py build_edit_text_sample 移植）

/// 构造编辑/多参考采样：统一 token 序列 + fix-point 位置 + 掩码 + 参考条件。
/// - Parameters:
///   - height/width: 目标图尺寸（需 32 倍数；外部已按 64 对齐）。
///   - referencePaths: 参考图路径（1 张 = 单图编辑，多张 = 多参考组合）。
func hidreamBuildEditSample(
    prompt: String,
    referencePaths: [String],
    height: Int,
    width: Int,
    tokenizer: any Tokenizers.Tokenizer,
    qwenConfig: Qwen3VLConfiguration
) throws -> HiDreamEditSample {
    let const = HiDreamDiffusionConstants.self
    let patch32 = const.patchSize
    let K = referencePaths.count
    guard K > 0 else { throw HiDreamEditError.invalidReferenceImage("无参考图") }

    // max_size 规则（py build_edit_text_sample）
    let m = max(height, width)
    let maxSize: Int
    if K == 1 { maxSize = m }
    else if K == 2 { maxSize = m * 48 / 64 }
    else if K <= 4 { maxSize = m / 2 }
    else if K <= 8 { maxSize = m * 24 / 64 }
    else { maxSize = m / 4 }

    // 视觉塔条件图尺寸（cond_img_size）
    let condImgSize: Int
    if K <= 4 { condImgSize = 384 }
    else if K <= 8 { condImgSize = 384 * 48 / 64 }
    else { condImgSize = 384 / 2 }

    let spatialMerge = qwenConfig.visionConfiguration.spatialMergeSize

    // 每张参考图：diffusion 侧 patchify32 + 视觉塔像素/网格
    var refPatchFlat = [Float]()
    var refImageLens: [Int] = []
    var refVisionLens: [Int] = []
    var pixelFlat = [Float]()
    var visualGrid: [(t: Int, h: Int, w: Int)] = []
    var refGrid32: [(h: Int, w: Int)] = []   // ref 扩散 patch 的 32 网格，供 igthwAll 第三段使用

    for path in referencePaths {
        guard let cg = hidreamLoadCGImage(path) else {
            throw HiDreamEditError.invalidReferenceImage(path)
        }
        let srcW = cg.width, srcH = cg.height
        guard srcW > 0, srcH > 0 else {
            throw HiDreamEditError.invalidReferenceImage(path)
        }
        let (nw, nh) = hidreamResizeTarget(srcW: srcW, srcH: srcH, imageSize: maxSize, patch: patch32)
        guard let rgba32 = hidreamResizeCropRGBA(cg, to: nw, newH: nh) else {
            throw HiDreamEditError.invalidReferenceImage(path)
        }
        refPatchFlat.append(contentsOf: hidreamPatchify32(rgba: rgba32, w: nw, h: nh))
        refImageLens.append((nh / 32) * (nw / 32))
        refGrid32.append((h: nh / 32, w: nw / 32))

        // 视觉塔条件图：从 resize 后比例再缩到 (cw, ch)（py 二次 resize）
        let (cw, ch) = hidreamVisionSize(maxSize: condImgSize, ratio: Float(nw) / Float(nh))
        guard let rgba16 = hidreamResizeCropRGBA(cg, to: cw, newH: ch) else {
            throw HiDreamEditError.invalidReferenceImage(path)
        }
        pixelFlat.append(contentsOf: hidreamVisionProcess(rgba: rgba16, w: cw, h: ch))
        let vh = ch / 16, vw = cw / 16
        // 视觉塔 merger(spatial merge 2×2) 后每图特征行数 = (vh/2)*(vw/2)，
        // 模板 image_pad 展开数必须与之相等，否则 precompute 抛 visionTokenMismatch。
        let merge = qwenConfig.visionConfiguration.spatialMergeSize
        refVisionLens.append((vh / merge) * (vw / merge))
        visualGrid.append((t: 1, h: vh, w: vw))
    }

    let totalRefLen = refImageLens.reduce(0, +)
    let refPatches = MLXArray(refPatchFlat).reshaped([totalRefLen, 3072])
        .expandedDimensions(axis: 0)                                // [1, sum(N), 3072]
    let pixelValues = MLXArray(pixelFlat)                            // [sum(Nv), 1536]

    // 目标 span 长度（height/width 已 32 对齐）
    let hPatches = height / patch32
    let wPatches = width / patch32
    let tgtImageLen = hPatches * wPatches

    // chat template：image 占位按每张视觉 token 数展开（py processor.apply_chat_template 展开 image_pad）
    var imageBlocks = ""
    for nv in refVisionLens {
        imageBlocks += "<|vision_start|>"
            + String(repeating: "<|image_pad|>", count: nv)
            + "<|vision_end|>"
    }
    let templateText = "<|im_start|>user\n" + imageBlocks + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
        + "<|boi_token|>"
        + String(repeating: "<|tms_token|>", count: const.timestepTokenNum)

    let inputIds = tokenizer.encode(text: templateText, addSpecialTokens: false).map { Int32($0) }
    let txtSeqLen = inputIds.count

    // 视觉占位：tgt span（首个 vision_start）+ ref spans（各首个 vision_start）
    var visionTokens = [Int32(qwenConfig.visionStartTokenId)]
        + [Int32](repeating: Int32(qwenConfig.imageTokenId), count: tgtImageLen - 1)
    for rl in refImageLens {
        visionTokens.append(Int32(qwenConfig.visionStartTokenId))
        visionTokens.append(contentsOf: [Int32](repeating: Int32(qwenConfig.imageTokenId), count: rl - 1))
    }
    let inputIdsPad = inputIds + visionTokens
    let allSeqLen = inputIdsPad.count

    // LLM 网格：refs（proc 网格已除 merge）+ tgt（32 网格）+ refs（32 网格）。
    // 第三段必须用 ref 扩散 patch 的 32 网格（对齐 refImageLens / visionTokens 展开数），
    // 不能用视觉塔 cond 网格（cond 图尺寸≠ref 图尺寸，会导致 ref span rope 位置错位→花屏）。
    var igthwAll: [(t: Int, h: Int, w: Int)] = []
    for g in visualGrid {
        igthwAll.append((t: g.t, h: g.h / spatialMerge, w: g.w / spatialMerge))
    }
    igthwAll.append((t: 1, h: hPatches, w: wPatches))
    for (rh, rw) in refGrid32 {
        igthwAll.append((t: 1, h: rh, w: rw))
    }
    var skipList = [Int](repeating: 0, count: K)
    skipList.append(1)
    skipList.append(contentsOf: [Int](repeating: 1, count: K))

    let positionIds = hidreamGetRopeIndexFixPointMulti(
        inputIdsPad: inputIdsPad,
        imageGridTHW: igthwAll,
        spatialMergeSize: 1,
        skipVisionStartToken: skipList,
        fixPoint: 4096)

    // token_types：tms 起标 1，tgt span + tms = 1，ref spans = 2，tms 单独 3
    var tokenTypes = [Int32](repeating: 0, count: allSeqLen)
    let bgn = txtSeqLen - const.timestepTokenNum
    let end = bgn + tgtImageLen + const.timestepTokenNum
    if bgn >= 0, end <= allSeqLen {
        for i in bgn ..< end { tokenTypes[i] = 1 }
        for i in txtSeqLen - const.timestepTokenNum ..< txtSeqLen { tokenTypes[i] = 3 }
        let refEnd = min(end + totalRefLen, allSeqLen)
        if refEnd > end {
            for i in end ..< refEnd { tokenTypes[i] = 2 }
        }
    }

    let mask4d = hidreamBuildAttentionMaskFast(tokenTypesBin: tokenTypes.map { $0 > 0 })

    // tgt 索引 = token_types == 1（不含 tms）
    var tgtIdx: [Int32] = []
    for (i, tt) in tokenTypes.enumerated() where tt == 1 {
        tgtIdx.append(Int32(i))
    }

    let imageGridTHW = visualGrid.map { THW($0.t, $0.h, $0.w) }

    return HiDreamEditSample(
        inputIds: inputIds,
        inputIdsPad: inputIdsPad,
        positionIds: positionIds,
        mask4d: mask4d,
        tgtIdx: tgtIdx,
        refPatches: refPatches,
        pixelValues: pixelValues,
        imageGridTHW: imageGridTHW)
}

// MARK: - 编辑模式文本嵌入预计算（py precompute_text_embeds_with_vision 移植）

/// DBG 日志：统一报告出口 pipelineLog（print + monitorLog 面板 + 落盘），
/// 保留极高频数值诊断的独立文件语义：独立落盘 output/图像/hidream_edit_dbg.log，不并入主 pipeline.log。
/// 已移除 stderr 旁路与废弃会话硬编码路径（原 hidreamDbgPath）。
private func dbgLog(_ s: String) {
    pipelineLog(s, logURL: outputImageDirURL.appendingPathComponent("hidream_edit_dbg.log"))
}

/// 全量标准差（Float）。
private func std(_ a: MLXArray) -> Float {
    let arr = a.asArray(Float.self)
    guard !arr.isEmpty else { return 1.0 }
    let n = Float(arr.count)
    let mean = arr.reduce(0, +) / n
    let ss = arr.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
    return sqrt(ss / n)
}

/// 文本嵌入 + 视觉塔特征注入 image_token 位置（仅在编辑模式使用；去噪循环前调用一次）。
/// - Returns: [1, txt_seq_len, 4096]，与 dtype 同文本嵌入。
func hidreamPrecomputeEditEmbeds(
    backbone: HiDreamBackbone,
    inputIds: [Int32],
    pixelValues: MLXArray,
    imageGridTHW: [THW]
) throws -> (MLXArray, [MLXArray], MLXArray) {   // (inputsEmbeds, deepstackEmbeds, visualMask)
    let const = HiDreamDiffusionConstants.self
    let embeds = backbone.languageModel.model.embedTokens(
        MLXArray(inputIds)[.newAxis, 0...])                       // [1, txtLen, 4096]

    let (vtOut, deepstackOuts) = backbone.visionTower(pixelValues, gridTHW: imageGridTHW)
    let imageFeatures = vtOut                                      // [N_vision, 4096]

    #if DEBUG
    let vts = imageFeatures.asArray(Float.self)
    let vtNan = vts.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
    let vtMaxAbs = vts.map { abs($0) }.max() ?? 0
    let vtMean = vts.reduce(0, +) / Float(max(vts.count, 1))
    dbgLog("DBG edit vtOut nan=\(vtNan)/\(vts.count) mean=\(vtMean) maxAbs=\(vtMaxAbs)\n")
    #endif

    let imgPositions = inputIds.enumerated()
        .filter { $0.element == Int32(const.imageTokenID) }
        .map { $0.offset }
    guard imgPositions.count == imageFeatures.dim(0) else {
        throw HiDreamEditError.visionTokenMismatch(imgPositions.count, imageFeatures.dim(0))
    }

    let S = inputIds.count
    let H = embeds.dim(2)
    var aligned = MLXArray.zeros([1, S, H], dtype: .float32)
    let feats = imageFeatures.asType(.float32)

    // 官方 masked_scatter 原样注入，无缩放（与 pipeline.py cond_image_embeds 对齐）
    for (i, pos) in imgPositions.enumerated() {
        aligned[0, pos, 0...] = feats[i ..< i + 1].squeezed(axis: 0)
    }

    let mask2d = MLXArray(inputIds.map {
        $0 == Int32(const.imageTokenID) ? Float(1.0) : Float(0.0)
    })
    let mask3d = broadcast(mask2d.expandedDimensions(axis: 0).expandedDimensions(axis: -1), to: [1, S, H])
    let inputsEmbeds = MLX.where(mask3d .> 0.5, aligned, embeds.asType(.float32)).asType(embeds.dtype)

    // deepstack 特征转同 dtype（对齐官方 deepstack_visual_embeds → inputs_embeds.dtype）
    let dsEmbeds = deepstackOuts.map { $0.asType(embeds.dtype) }
    return (inputsEmbeds, dsEmbeds, mask2d)   // mask2d: [S] 视觉位置掩码（deepstack 注入位置）
}

// MARK: - 编辑生成主循环（py 编辑分支移植）

/// HiDream-O1-Image 编辑/多参考生成入口。
/// - Parameters:
///   - referencePaths: 参考图路径（编辑/多参考组合，调度与 T2I 同款官方默认 7.5/2.5）。
///   - height/width: 请求尺寸（内部对齐到预定义分辨率）。
/// - Returns: 归一化 [0,1] 的 [H, W, 3] float32 图像
func hidreamGenerateEdit(
    backbone: HiDreamBackbone,
    tokenizer: any Tokenizers.Tokenizer,
    qwenConfig: Qwen3VLConfiguration,
    prompt: String,
    referencePaths: [String],
    height: Int,
    width: Int,
    steps: Int = 28,
    seed: UInt64 = 32,
    onProgress: ((Float) -> Void)? = nil,
    isCancelled: () -> Bool = { false }
) throws -> MLXArray {
    let const = HiDreamDiffusionConstants.self
    let (h, w) = closestResolution(height: height, width: width, predefined: const.predefinedResolutions)

    let sample = try hidreamBuildEditSample(
        prompt: prompt, referencePaths: referencePaths, height: h, width: w,
        tokenizer: tokenizer, qwenConfig: qwenConfig)

    // 文本嵌入预计算（含视觉特征注入 + deepstack 中间层特征；去噪循环外只算一次）
    let (inputsEmbeds, dsEmbeds, visualMask) = try hidreamPrecomputeEditEmbeds(
        backbone: backbone,
        inputIds: sample.inputIds,
        pixelValues: sample.pixelValues,
        imageGridTHW: sample.imageGridTHW)

    // tms 注入掩码常量（inputIds 固定，循环外预构造一次）
    let tmsMaskBroadcast = hidreamBuildTmsMask(
        inputIds: sample.inputIds,
        tmsTokenID: Int32(const.tmsTokenID),
        shape: inputsEmbeds.shape)

    // ref patches 嵌入常量（refPatches 固定，循环外只算一次，循环内仅嵌入 z）
    let refEmb = backbone.xEmbedder(sample.refPatches).asType(inputsEmbeds.dtype)   // [1, N_ref, 4096]

    // 初始噪声（py 对齐：noise_scale_start × N(0,1) → patchify）
    let rngKey = MLXRandom.key(seed)
    let noise = MLXRandom.normal([3, h, w], key: rngKey) * MLXArray(const.noiseScaleDefault)
    var z = patchifySpatial2DBCHW(noise).expandedDimensions(axis: 0)   // [1, N_tgt, 3072]

    // scheduler：官方默认，编辑与 T2I 共用 s_noise=7.5 + noise_clip_std=2.5
    let scheduler = FlashFlowMatchScheduler()
    scheduler.setTimesteps(numInferenceSteps: steps, customTimesteps: const.defaultTimesteps)
    let noiseClipStd: Float = 2.5

    let tgtIdx = MLXArray(sample.tgtIdx)
    let sigmaT = Float(const.tEps)

    // MLX.compile 单步前向：shape 固定则复用编译图（对齐 LTX Sampler.swift 编译缓存模式）
    let shapeKey =
        "hidream-edit-\(inputsEmbeds.shape)-\(sample.positionIds.shape)-\(sample.mask4d.shape)-\(tgtIdx.shape)-\(refEmb.shape)-ds\(dsEmbeds.count)-h\(h)-w\(w)"
    var forward: CompiledHiDreamForward? = nil
    var preWarmXPred: MLXArray? = nil   // 编译预热前向输出（首步 x0 预测），首步直接复用
    if let cached = HiDreamCompiledForwardCache.shared.getEdit(backbone: backbone, key: shapeKey) {
        stageEnter(phase: .sampling,
                   detail: "复用已编译 HiDream 编辑前向（shape 不变，跳过编译）",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        forward = cached
    } else {
        stageEnter(phase: .sampling,
                   detail: "MLX.compile 编译 HiDream 编辑前向中（首次约 30-90s）...",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        let tC = Date()
        let cf = makeCompiledHiDreamEditForward(
            backbone: backbone,
            inputsEmbeds: inputsEmbeds,
            tmsMaskBroadcast: tmsMaskBroadcast,
            positionIds: sample.positionIds,
            mask4d: sample.mask4d,
            refEmb: refEmb,
            visualMask: visualMask,
            deepstackEmbeds: dsEmbeds,
            tgtIdx: tgtIdx)
        // 预热：用初始噪声 latent + 第一步 timestep 触发图构建 + 内核编译
        let warmT = Date()
        let wz = cf(z, MLXArray(1.0 - Float(scheduler.timestepsNP[0]) / 1000.0))
        eval(wz)
        stageEnter(phase: .sampling,
                   detail: "  编译完成（\(Int(Date().timeIntervalSince(tC)))s，含预热 \(Int(Date().timeIntervalSince(warmT)))s）",
                   logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        HiDreamCompiledForwardCache.shared.storeEdit(cf, backbone: backbone, key: shapeKey)
        // 预热复用：预热输入=初始噪声 z+首步 timestep，其输出即首步 x0 预测，直接存为 preWarmXPred
        preWarmXPred = wz
        forward = cf
    }

    // TeaCache：相邻步 sigma 差小于阈值时复用上一步 x0，跳过整次前向（对齐 LTX Sampler）
    let teaCache = TeaCacheLegacy()
    teaCache.threshold = 0.02

    for (i, stepT) in scheduler.timestepsNP.enumerated() {
        if isCancelled() {
            throw GenerationCancelError.cancelled
        }
        // 每步独立作用域：本步中间张量（xPred/v/modelOutput/多层前向激活）在步末释放，
        // 避免整轮采样激活攒到函数结束一次性析构导致峰值虚高（对齐 LTX Sampler 每步 autoreleasepool）
        autoreleasepool {
        let tPixelDiT: Float = 1.0 - stepT / 1000.0
        let sigma = max(stepT / 1000.0, sigmaT)

        let xPred: MLXArray
        var cachedV: MLXArray? = nil
        var cachedA: MLXArray? = nil
        if i == 0, let pw = preWarmXPred {
            // 预热复用：编译预热前向（初始噪声 z+首步 t）输出即首步 x0 预测，跳过首步 DiT 前向
            xPred = pw
            stageEnter(phase: .sampling,
                       step: i + 1, total: steps,
                       detail: "    [预热复用] 第 1 步复用编译预热输出，跳过前向",
                       logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
            teaCache.store(v: pw, a: pw)
        } else if teaCache.tryReuse(sigma: sigma, outV: &cachedV, outA: &cachedA), let rv = cachedV {
            xPred = rv
            stageEnter(phase: .sampling,
                       step: i + 1, total: steps,
                       detail: "    [TeaCache] 步 \(i + 1) σ=\(sigma) 命中缓存，跳过前向",
                       logURL: outputImageDirURL.appendingPathComponent("pipeline.log"), protect: [.hiDream])
        } else {
            // 采样循环内内存兜底：真实重前向入口先查压力分（TeaCache 命中步直出缓存不经此）
            MemoryPolicy.ensureLoose(protect: [.hiDream])
            if let f = forward {
                xPred = f(z, MLXArray(tPixelDiT))
            } else {
                // 编辑模式：目标 z 嵌入 + 预计算 ref 嵌入拼接（refs 双向注意力）
                let vEmb = concatenated(
                    [backbone.xEmbedder(z).asType(inputsEmbeds.dtype), refEmb], axis: 1)

                xPred = hidreamForwardGeneration(
                    backbone: backbone,
                    inputIds: sample.inputIds,
                    inputsEmbeds: inputsEmbeds,
                    tmsMaskBroadcast: tmsMaskBroadcast,
                    positionIds: sample.positionIds,
                    mask4d: sample.mask4d,
                    vEmb: vEmb,
                    timestepPixelDiT: tPixelDiT,
                    visualMask: visualMask,                                    // deepstack 注入位置
                    deepstackEmbeds: dsEmbeds,                                 // [1, S_total, 3072]
                    tgtIdx: tgtIdx)                                            // [1, N_tgt, 3072]
            }
            teaCache.store(v: xPred, a: xPred)
        }

        #if DEBUG
        if i == 0 || i == 1 || i == 5 || i == 13 || i == 27 {
            func dbgStats(_ arr: MLXArray, _ name: String) {
                let f = arr.asArray(Float.self)
                var sum: Float = 0, mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude, nan = 0
                for v in f {
                    if v.isNaN { nan += 1; continue }
                    sum += v
                    if v < mn { mn = v }
                    if v > mx { mx = v }
                }
                let n = Float(max(f.count - nan, 1))
                let mean = sum / n
                var ss: Float = 0
                for v in f where !v.isNaN { ss += (v - mean) * (v - mean) }
                let std = sqrt(ss / n)
                dbgLog("DBG edit step\(i) sigma=\(sigma) \(name) mean=\(mean) std=\(std) min=\(mn) max=\(mx) nan=\(nan)/\(f.count)\n")
            }
            dbgStats(z, "z")
            dbgStats(sample.refPatches, "refPatches")
            dbgStats(xPred.asType(.float32), "xPred_tgt")
            dbgStats(xPred.asType(.float32), "xPred_all")
            if i == 0 {
            let S = sample.positionIds.dim(2)
            let pid = sample.positionIds.asArray(Int32.self)
            let kt = sample.inputIds.count
            let tgtImageLen = (h / const.patchSize) * (w / const.patchSize)
            let kr = kt + tgtImageLen
            if kt < S {
                let tH = kr < S ? String(pid[kr]) : "?"
                let hH = kr < S ? String(pid[S + kr]) : "?"
                let wH = kr < S ? String(pid[2 * S + kr]) : "?"
                dbgLog(
                    "DBG edit step0 posIds tgtHead=(\(pid[kt]),\(pid[S + kt]),\(pid[2 * S + kt])) refHead=(\(tH),\(hH),\(wH)) S=\(S) txt=\(kt) tgt=\(tgtImageLen) ref=\(sample.refPatches.dim(1))\n")
            }
            }
        }
        #endif

        let genPatches = xPred.asType(.float32)  // [1, N_tgt, 3072]（finalLayer2 已裁剪 tgt）
        let zf = z.asType(.float32)
        let v = (genPatches - zf) / MLXArray(sigma)
        let modelOutput = -v

        // 官方默认调度：s_noise=7.5 / noise_clip_std=2.5（公式 σ_next·noise·s_noise + (1-σ_next)·denoised）
        z = scheduler.step(
            modelOutput: modelOutput,
            timestep: stepT,
            sample: z,
            sNoise: const.noiseScaleDefault,
            noiseClipStd: noiseClipStd,
            seed: seed)
            z = z.asType(.float32)
            eval(z)

            onProgress?(Float(i + 1) / Float(steps))
        }
    }

    let img = (z.asType(.float32) + 1.0) * 0.5                              // [1, N_tgt, 3072]
    let imgCHW = unpatchifySpatialBCHW(
        img.squeezed(axis: 0), hPatches: h / const.patchSize, wPatches: w / const.patchSize) // [3, H, W]
    // 采样结束清一次 MLX 缓冲，避免峰值后内存长期滞留
    MLX.Memory.clearCache()
    return imgCHW.transposed(1, 2, 0)                                       // [H, W, 3]
}
