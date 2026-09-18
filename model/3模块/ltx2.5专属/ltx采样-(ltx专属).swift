//
//  Sampler.swift
//  ltx-test — LTX-2.5 流匹配采样器（对齐 mlx-serve src/ltx_video.zig）
//
//  ============================================================
//  作用：纯 t2v 的 guided Euler 采样循环：
//    dynamicShiftSchedule → 噪声 latent → 逐 sigma 步：
//      cond 前向（+ neg 前向做 CFG）→ x0 = x - vel*sigma
//      → CFG + norm-preserving rescale → Euler 步 → 下一 sigma
//  参考：ltx_video.zig 的 dynamicShiftSchedule / eulerStep /
//        guiderCalculate / ditSampleCfg（仅 CFG 子集，无 STG/modality/I2V）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 采样参数

struct SamplerConfig {
    var numSteps: Int = 50
    var cfgV: Float = 3.0         // 视频 CFG（对齐参考 vp.cfg=3.0；蒸馏单阶段传 1.0=无引导）
    var cfgA: Float = 7.0         // 音频 CFG（对齐参考 ap.cfg=7.0，音频引导远强于视频）
    var cfgRescaleV: Float = 0.7  // 视频 norm-preserving rescale（参考 vp.rescale）
    var cfgRescaleA: Float = 0.7  // 音频 norm-preserving rescale（参考 ap.rescale）
    var baseShift: Double = 0.95
    var maxShift: Double = 2.05
    var baseTokens: Double = 1024
    var maxTokens: Double = 4096
    var stretch: Bool = true
    var terminal: Double = 0.1
    var sigmas: [Float]? = nil     // 非 nil → 固定 sigma 表（蒸馏模型专用，跳过 dynamicShiftSchedule）
    var ancestral: Bool = false    // true → ancestral Euler 步进（蒸馏 stage1 SDE 噪声注入）
    var seed: UInt64 = 42
    var useCompile: Bool = true    // MLX.compile 编译 DiT 单步前向（大幅减少 host 调度开销）
    var teaCache: Bool = true              // TeaCache 特征缓存（官方 rel-L1 判据，相邻步特征相似时跳 blocks，非整步前向）
    var teaCacheThreshold: Float = 0.06    // TeaCache 触发阈值（官方 LTX-Video 推荐 rel_l1_thresh=0.06；调大更激进）
    // ── MultiModalGuider 扩展引导（官方 dev 默认：STG=1.0 @ block28，modality=3.0）──
    // 引导公式：pred = cond + (cfg-1)(cond-neg) + stgScale*(cond-ptb) + (modalityScale-1)*(cond-mod)
    // 每步额外 2 次前向：ptb（STG 扰动，指定 block 自注意力 passthrough）、mod（模态隔离，跳 AV 交叉）。
    // 默认关闭（0 / 1 / nil），仅 dev 单阶段参数表显式开启，蒸馏/Stage2 不受影响。
    var stgScaleV: Float = 0          // STG 引导强度·视频（官方 1.0；0=关闭）
    var stgScaleA: Float = 0          // STG 引导强度·音频（官方 1.0；0=关闭）
    var modalityScaleV: Float = 1     // Modality 引导强度·视频（官方 3.0；1=关闭）
    var modalityScaleA: Float = 1     // Modality 引导强度·音频（官方 3.0；1=关闭）
    var stgBlocksV: [Int] = [28]      // STG 扰动 block 索引·视频（官方 stg_blocks=[28]）
    var stgBlocksA: [Int] = [28]      // STG 扰动 block 索引·音频（官方 stg_blocks=[28]）
    // ── SOL 稀疏注意力（LTX 二采 refine 专用；nil = 关闭，退回 dense 全量注意力）──
    // 仅视频自注意力 attn1 生效：headDim=128 / batch=1 / 方阵 的前置条件由 LTXSparseAttnBridge 校验，
    // 不满足（音频 attn1 headDim=64、文本与跨模态交叉 attn 非方阵）自动回退 dense，语义与关闭时一致。
    var sparseVideo: LTXSparseAttnConfig? = nil
}
// MARK: - 编译前向（MLX.compile）

/// 把 DiT 单步前向编译为融合内核（shape 每步固定，完美契合）。
/// 捕获 dit 权重 + 预构造 rope + 位置数组（常量），运行时只换 latent/timestep/text 输入。
/// 返回 (video, audio, videoResidual, audioResidual)。STG 分段主图（emitMidAt 启用）时
/// 第 3/4 槽为 [残差行; mid 行] 沿 batch 维拼接（MLX.compile 返回元组上限 4 元素），运行期拆回。
typealias CompiledDitForward = @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray)

func makeCompiledDitForward(
    dit: LTXVideoDiT, rope: BlockRope, videoPos: [Float], audioPos: [Float],
    audioTimesteps: MLXArray? = nil,
    condMask: [Float]? = nil,
    maskMLX: MLXArray? = nil,
    maskMLXInv: MLXArray? = nil,
    keyframesMLX: MLXArray? = nil,
    skipAVCross: Bool = false,       // Modality 隔离：跳 AV 交叉注意力（mod 前向）
    stgBlocksV: [Int]? = nil,        // STG 扰动 block 索引（ptb 前向；nil=不扰动）
    stgBlocksA: [Int]? = nil,        // STG 扰动 block 索引（ptb 前向；nil=不扰动）
    fromLayer: Int = 0,              // STG 分段：>0 时输入为第 fromLayer 层激活（跳过 patchify），ptb 图从该层继续
    emitMidAt: Int? = nil,           // STG 分段：非 nil 时主图跑完该层额外返回 (video, audio) 中间激活
    sparseVideo: LTXSparseAttnConfig? = nil  // 非 nil → 视频自注意力走 SOL 稀疏（常量被固化进编译图，缓存 key 必须带标签区分）
) -> CompiledDitForward {
    // I2V：per-token timestep = mask*sigma 与 keyframes gate 均为纯张量运算，
    // 常量（condMask 标志 + 预构造 mask/maskMLXInv/keyframesMLX）由闭包捕获，可整体编译。
    MLX.compile { (vx: MLXArray, ax: MLXArray, ts: MLXArray, vText: MLXArray, aText: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) in
        let out = dit.forwardWithResidual(
            videoLatent: vx, audioLatent: ax, timesteps: ts,
            audioTimesteps: audioTimesteps,
            videoText: vText, audioText: aText,
            videoPos: videoPos, audioPos: audioPos,
            rope: rope,
            skipAVCross: skipAVCross,
            stgBlocksV: stgBlocksV, stgBlocksA: stgBlocksA,
            condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX,
            fromLayer: fromLayer, emitMidAt: emitMidAt,
            sparseVideo: sparseVideo)
        // MLX.compile 返回元组上限 4 元素：STG 分段主图需同时带出残差（TeaCache）与 mid 激活（ptb 输入），
        // 将 mid 沿 batch 维拼在残差之后（[残差行; mid 行]），运行期按行数拆回。
        if let mv = out.midVideo, let ma = out.midAudio, emitMidAt != nil {
            return (out.video, out.audio,
                    concatenated([out.videoResidual, mv], axis: 0),
                    concatenated([out.audioResidual, ma], axis: 0))
        }
        return (out.video, out.audio, out.videoResidual, out.audioResidual)
    }
}

// MARK: - TeaCache 编译图（官方 rel-L1 语义）

/// TeaCache 特征探针编译图：每步算第一个 transformer block 的视频调制输入
/// （官方 teacache_ltxvmodel_forward 的 modulated_inp）。cond/uncond 共用同一 latent/sigma，
/// probe 相同，每步只跑一次；轻量（无注意力/FFN）。
typealias TeaCacheProbeForward = @Sendable (MLXArray, MLXArray) -> MLXArray

func makeTeaCacheProbeForward(
    dit: LTXVideoDiT,
    condMask: [Float]? = nil,
    maskMLX: MLXArray? = nil,
    maskMLXInv: MLXArray? = nil,
    keyframesMLX: MLXArray? = nil
) -> TeaCacheProbeForward {
    MLX.compile { (vx: MLXArray, ts: MLXArray) -> MLXArray in
        dit.teaCacheProbe(
            videoLatent: vx, timesteps: ts,
            condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX)
    }
}

/// TeaCache 跳步编译图：blocks 输入 + 上一步输出调制残差 → 输出投影
/// （省掉 48 层 blocks + norm+modulate，官方 x += previous_residual 语义）。
typealias TeaCacheSkipForward = @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)

func makeTeaCacheSkipForward(
    dit: LTXVideoDiT,
    condMask: [Float]? = nil,
    maskMLXInv: MLXArray? = nil,
    keyframesMLX: MLXArray? = nil
) -> TeaCacheSkipForward {
    MLX.compile { (vx: MLXArray, ax: MLXArray, vRes: MLXArray, aRes: MLXArray) -> (MLXArray, MLXArray) in
        let out = dit.forwardSkipResidual(
            videoLatent: vx, audioLatent: ax,
            videoResidual: vRes, audioResidual: aRes,
            condMask: condMask, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX)
        return (out.video, out.audio)
    }
}

// MARK: - 编译图持久缓存（方案 A）

/// 编译图常驻复用：同一 DiT 实例 + 同一 shape 时直接复用上次编译产物，
/// 采样结束后不释放编译图，根治 "Compiled 大图一次性递归析构" 造成的 app 假死。
/// 析构只发生在：换 shape（重新编译时旧图释放）或 app 退出。
final class CompiledForwardCache {
    static let shared = CompiledForwardCache()
    /// 多槽缓存：stage1/stage2 等不同 shape 的编译图可同时驻留，避免交替重编译。
    private let cache = CompiledGraphCache<LTXVideoDiT, CompiledDitForward>()

    func get(dit: LTXVideoDiT, key: String) -> CompiledDitForward? {
        cache.get(owner: dit, key: key)
    }

    func store(_ f: @escaping CompiledDitForward, dit: LTXVideoDiT, key: String) {
        cache.store(f, owner: dit, key: key)
    }

    /// 卸载编译图（释放闭包对 DiT 权重的强捕获，供廉价设备动态卸载用）
    func clear() {
        cache.clear()
    }

    /// refine/stage 切换前前缀瘦身：保留与当前将编译形状同 key 前缀的编译图
    /// （同配置二次生成命中复用、免重编译），淘汰其余大图。
    /// 注意：不得放宽为"保最近N张"——上轮全清大图驻留会让下一轮 stage2 编译峰值叠加爆内存。
    func evictExcept(keyPrefix: String) {
        cache.evictExcept(keyPrefix: keyPrefix)
    }

    /// 编译图是否已缓存（监控面板权重加载状态用）
    var isCompiled: Bool { !cache.isEmpty }
}

// MARK: - in-context LoRA 旁路通道标签（编译缓存隔离）

/// 当前生效的 in-context LoRA 旁路通道标签，参与 MLX 编译图缓存 key。
/// 背景：编译图会把「旁路权重常量 + 各层 icActive 状态」一并固化进图结构，故：
///   · 原生路径（无旁路）与 IC 二采 / CQ 清晰度增强 必须分开编译槽位；
///   · IC 与 CQ 虽共用同一批槽位、序列布局也相同（若 σ 步数一并相同则 shape key 完全一致），
///     但注入的 LoRA 权重完全不同 → 绝不能互相复用编译图；
///   · 同一通道换权重文件 / 换 strength 同样需要重新编译。
/// 取值："off" | "ic-<权重文件名>[-s<强度>]" | "cq-<权重文件名>-s<强度>"。
/// 由 runLTXStage2RefineOnLatent 在挂载旁路前设置、函数级 defer 复位为 "off"。
enum LoRABypassTag {
    static var current: String = "off"
}

// MARK: - 单步 x0 预测（含可选 CFG 负向）

/// 一次 sigma 步：cond 前向；若提供 neg 条件再做 neg 前向做 CFG。
/// 返回 x0（bf16，与参考 ditX0 一致）。forward 非 nil 时 cond/neg 走编译内核。
/// - frozenAudio: 非 nil 时启用音频条件（frozen_a 语义）：音频流以 timestep 0 运行、
///   音频 x0 pin 为 frozen latent（自身不重建，仅通过注意力条件化视频），
///   视频流仍正常去噪。
/// - condMask/cleanV: 非 nil 时启用 I2V 首帧条件（对齐 ditX0Guided 的 cond_mask/clean_v）：
///   视频前向走 per-token timestep（mask*sigma），采样后 applyDenoiseMask 把干净首帧
///   latent 钉入（x0 = x0*mask + clean*(1-mask)）。
func ditX0Guided(
    dit: LTXVideoDiT,
    forward: CompiledDitForward?,
    forwardBatch: CompiledDitForward? = nil, // CFG 分支 batch 合并编译图（cond/uncondV/uncondA 文本条件拼 batch 一次前向，严格数值等价）
    stgForward: CompiledDitForward? = nil,  // STG 扰动编译图（stgScale 开启且走编译路径时非 nil）
    stgSegStart: Int? = nil,                // STG 分段启动：非 nil 且 >0 → cond 主图输出 segStart-1 层激活，ptb 图从 segStart 层继续（省公共前缀重复计算）
    modForward: CompiledDitForward? = nil,  // Modality 隔离编译图（modalityScale 开启且走编译路径时非 nil）
    vx: MLXArray,
    ax: MLXArray,
    sigma: Float,
    condV: MLXArray,
    condA: MLXArray,
    negV: MLXArray?,
    negA: MLXArray?,
    frozenAudio: MLXArray?,
    condMask: [Float]?,
    cleanV: MLXArray?,
    maskTensor: MLXArray? = nil,   // I2V：预构造 mask [1,T,H,W,1] bf16（采样循环外构造一次，避免每步 CPU→GPU 拷贝）
    maskMLX: MLXArray? = nil,      // I2V：预构造 mask [Nv] f32（DiT per-token timestep 复用）
    maskMLXInv: MLXArray? = nil,   // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
    keyframesMLX: MLXArray? = nil, // IC-LoRA Stage2：独立 keyframes 标记 [Nv] f32（slots 生成+标记并存）
    videoPos: [Float],
    audioPos: [Float],
    rope: BlockRope,
    cfg: SamplerConfig,
    teaCache: TeaCache? = nil,             // 非 nil → 官方 TeaCache（rel-L1 判据 + 残差跳步）
    branch: TeaCache.BranchKind = .cond,   // 当前主前向分支（cond/uncond 独立状态）
    probe: MLXArray? = nil,                // 采样循环已算好的共享特征（cond/uncond 共用，每步一次）
    skipForward: TeaCacheSkipForward? = nil
) -> (v: MLXArray, a: MLXArray) {
    // 采样循环内内存兜底：真实重前向入口先查压力分，超标即清缓存/卸非保护模型（防采样途中 OOM）
    MemoryPolicy.ensureLoose(protect: [.dit])
    let ts = MLXArray([sigma])
    // 音频条件模式下音频流走独立时间步 0（audio_sigma=0.0，clean/frozen）
    let audioTs = frozenAudio == nil ? nil : MLXArray([Float(0.0)])

    // ── cond 前向（不单独 eval：与 euler 步合并为一次物化）──
    var vx0: MLXArray
    var ax0: MLXArray
    var uvx0: MLXArray? = nil
    var uax0: MLXArray? = nil
    // STG 分段：cond 主图（batch/单分支）跑到 segStart-1 层输出中间激活，供 ptb 分段图作为输入
    var condMidV: MLXArray? = nil
    var condMidA: MLXArray? = nil
    // 主图是否输出 mid（STG 分段启用 → 主图 emitMidAt 激活，残差槽为 [残差; mid] 拼接）
    let emitMid = (stgSegStart != nil && stgSegStart! > 0)
    /// 主图返回 4 元组；emitMid 时第 3/4 槽为 [残差行; mid 行] 沿 batch 维拼接。
    /// rows = 残差行数（batch 图 = nB，单分支 = 1），返回 (residualV, residualA, midV?, midA?)。
    func unpackMain(_ rv: MLXArray, _ ra: MLXArray, rows: Int)
        -> (MLXArray, MLXArray, MLXArray?, MLXArray?) {
        if emitMid {
            return (rv[0 ..< rows], ra[0 ..< rows], rv[rows ..< rows + 1], ra[rows ..< rows + 1])
        }
        return (rv, ra, nil, nil)
    }
    if let forward {
        if let bf = forwardBatch {
            // ── CFG 分支 batch 合并路径：cond/uncondV/uncondA 文本条件拼 batch，一次前向替代三次独立调用。
            // 数值严格等价（matmul 按行独立），kernel 打满 + 启动开销降 2/3；vx/ax 三个分支输入相同，显式拼行复制。
            // 行布局：0=cond，1=uncondV（needV），2=uncondA（needA）。
            let needV = negV != nil && cfg.cfgV > 1.0
            let needA = frozenAudio == nil && negA != nil && cfg.cfgA > 1.0
            let nB = 1 + (needV ? 1 : 0) + (needA ? 1 : 0)
            precondition(nB > 1, "forwardBatch 仅 CFG 分支数 > 1 时使用")
            var vTexts: [MLXArray] = [condV]
            var aTexts: [MLXArray] = [condA]
            if needV { vTexts.append(negV!); aTexts.append(condA) }
            if needA { vTexts.append(condV); aTexts.append(negA!) }
            let vTextBatch = concatenated(vTexts, axis: 0)          // [nB, Nt, D]
            let aTextBatch = concatenated(aTexts, axis: 0)
            let vxBatch = concatenated([MLXArray](repeating: vx, count: nB), axis: 0)   // [nB, Nv, C]
            let axBatch = concatenated([MLXArray](repeating: ax, count: nB), axis: 0)   // [nB, Na, C]
            if let tc = teaCache, let probeVal = probe, let sf = skipForward {
                // 判据三分支共享同一 probe（shouldCalc 一致）：真算全真算（一次 batch 前向，拆行 store），
                // 跳步全跳步（三次轻量 skipForward，各用各自 previous_residual）。
                tc.update(&tc.cond, probe: probeVal)
                tc.update(&tc.uncondV, probe: probeVal)
                tc.update(&tc.uncondA, probe: probeVal)
                if tc.cond.shouldCalc {
                    // batch 图 video/audio batch=nB，timesteps 必须同步扩为 [nB]（AdaLN 调制表 B 维随 vx 走，
                    // 传 [1] 会导致 block 内 vSST/aSST reshaped([b,9,d]) 源数组仅 1 行 → Cannot reshape）
                    let tsBatch = MLXArray([Float](repeating: sigma, count: nB))
                    let (cvb, cab, rvb, rab) = bf(vxBatch, axBatch, tsBatch, vTextBatch, aTextBatch)
                    let (vrB, arB, midVb, midAb) = unpackMain(rvb, rab, rows: nB)
                    let cv = cvb[0 ..< 1]
                    let ca = cab[0 ..< 1]
                    vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                    ax0 = (ax - ca * sigma).asType(ax.dtype)
                    tc.store(&tc.cond, residualV: vrB[0 ..< 1], residualA: arB[0 ..< 1])
                    if let mv = midVb, let ma = midAb {
                        condMidV = mv[0 ..< 1]
                        condMidA = ma[0 ..< 1]
                    }
                    if needV {
                        let uv = cvb[1 ..< 2]
                        uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                        tc.store(&tc.uncondV, residualV: vrB[1 ..< 2], residualA: arB[1 ..< 2])
                    }
                    if needA {
                        let rowA = needV ? 2 : 1
                        let ua = cab[rowA ..< rowA + 1]
                        uax0 = (ax - ua * sigma).asType(ax.dtype)
                        tc.store(&tc.uncondA, residualV: vrB[rowA ..< rowA + 1], residualA: arB[rowA ..< rowA + 1])
                    }
                } else {
                    let (cv, ca) = sf(vx, ax, tc.cond.previousResidualV!, tc.cond.previousResidualA!)
                    vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                    ax0 = (ax - ca * sigma).asType(ax.dtype)
                    if needV {
                        let (uv, _) = sf(vx, ax, tc.uncondV.previousResidualV!, tc.uncondV.previousResidualA!)
                        uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                    }
                    if needA {
                        let (_, ua) = sf(vx, ax, tc.uncondA.previousResidualV!, tc.uncondA.previousResidualA!)
                        uax0 = (ax - ua * sigma).asType(ax.dtype)
                    }
                }
            } else {
                let tsBatch = MLXArray([Float](repeating: sigma, count: nB))
                let (cvb, cab, rvb, rab) = bf(vxBatch, axBatch, tsBatch, vTextBatch, aTextBatch)
                let (_, _, midVb, midAb) = unpackMain(rvb, rab, rows: nB)
                let cv = cvb[0 ..< 1]
                let ca = cab[0 ..< 1]
                vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                ax0 = (ax - ca * sigma).asType(ax.dtype)
                if let mv = midVb, let ma = midAb {
                    condMidV = mv[0 ..< 1]
                    condMidA = ma[0 ..< 1]
                }
                if needV {
                    let uv = cvb[1 ..< 2]
                    uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                }
                if needA {
                    let rowA = needV ? 2 : 1
                    let ua = cab[rowA ..< rowA + 1]
                    uax0 = (ax - ua * sigma).asType(ax.dtype)
                }
            }
        } else if let tc = teaCache, let probeVal = probe, let sf = skipForward {
            // 官方 TeaCache：先 update 判据，再决定走主图（真算+存残差）还是跳步图（x += residual 后 proj_out）。
            // probe 每步一次（各分支共用），此处同步 update 三分支判据（probe 相同，shouldCalc 一致），
            // 负向前向（negV/negA）只读各自 uncond 槽的 shouldCalc 与残差，不再重复累加。
            tc.update(&tc.cond, probe: probeVal)
            tc.update(&tc.uncondV, probe: probeVal)
            tc.update(&tc.uncondA, probe: probeVal)
            if tc.cond.shouldCalc {
                let (cv, ca, rv, ra) = forward(vx, ax, ts, condV, condA)
                let (vr, ar, midV, midA) = unpackMain(rv, ra, rows: 1)
                tc.store(&tc.cond, residualV: vr, residualA: ar)
                vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)   // cv [B,Nv,C] → 5D 广播
                ax0 = (ax - ca * sigma).asType(ax.dtype)
                if let mv = midV, let ma = midA {
                    condMidV = mv
                    condMidA = ma
                }
            } else {
                // 跳步：x += previous_residual 后直接 proj_out（残差已含 norm+modulate）
                let (cv, ca) = sf(vx, ax, tc.cond.previousResidualV!, tc.cond.previousResidualA!)
                vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                ax0 = (ax - ca * sigma).asType(ax.dtype)
            }
        } else {
            let (cv, ca, rv, ra) = forward(vx, ax, ts, condV, condA)
            let (_, _, midV, midA) = unpackMain(rv, ra, rows: 1)
            vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)   // cv [B,Nv,C] → 5D 广播
            ax0 = (ax - ca * sigma).asType(ax.dtype)
            if let mv = midV, let ma = midA {
                condMidV = mv
                condMidA = ma
            }
        }
    } else {
        let (cv, ca) = dit(
            videoLatent: vx, audioLatent: ax, timesteps: ts,
            audioTimesteps: audioTs,
            videoText: condV, audioText: condA,
            videoPos: videoPos, audioPos: audioPos, rope: rope,
            condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX,
            sparseVideo: cfg.sparseVideo)
        vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(vx.dtype)
        ax0 = (ax - ca * sigma).asType(ax.dtype)
    }

    // ── neg 前向（CFG，视频/音频分模态独立）──
    // 纯 t2v 无音频条件时传 negA=nil：音频分支不做负向前向，无条件直出（省一次前向）
    // cfgV==1.0 时负向零贡献（guiderCombine 乘 0 丢弃），短路跳过整次全尺寸负向前向
    // batch 合并路径已在上方 cond 段完成负向（uvx0/uax0 非 nil），此处仅非 batch 路径执行
    if uvx0 == nil, let nv = negV, cfg.cfgV > 1.0 {
        if let forward {
            if let tc = teaCache, let sf = skipForward {
                // 判据已由 cond 分支同步 update（probe 相同），此处只读 shouldCalc；
                // 真算时 store 到独立的 uncondV 槽（negV+condA），与 negA 分支（uncondA 槽）互不覆盖，
                // 避免跨模态残差错位导致 CFG 负向分支输出污染（网格感/音频怪异）
                if tc.uncondV.shouldCalc {
                    let (uv, _, rv, ra) = forward(vx, ax, ts, nv, condA)
                    let (vr, ar, _, _) = unpackMain(rv, ra, rows: 1)
                    tc.store(&tc.uncondV, residualV: vr, residualA: ar)
                    uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                } else {
                    let (uv, _) = sf(vx, ax, tc.uncondV.previousResidualV!, tc.uncondV.previousResidualA!)
                    uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
                }
            } else {
                let (uv, _, _, _) = forward(vx, ax, ts, nv, condA)
                uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
            }
        } else {
            let (uv, _) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: nv, audioText: condA,
                videoPos: videoPos, audioPos: audioPos, rope: rope,
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                keyframesMLX: keyframesMLX,
                sparseVideo: cfg.sparseVideo)
            uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(vx.dtype)
        }
    }
    // 音频 neg：frozen 条件时音频不重建（跳过负向与引导合成，x0 直接 pin 为 clean latent）
    if uax0 == nil, frozenAudio == nil, let na = negA, cfg.cfgA > 1.0 {
        if let forward {
            if let tc = teaCache, let sf = skipForward {
                // 同 negV：只读 uncondA 槽判据；真算时 store 到独立的 uncondA 槽（condV+negA），
                // 与 negV 分支（uncondV 槽）互不覆盖
                if tc.uncondA.shouldCalc {
                    let (_, ua, rv, ra) = forward(vx, ax, ts, condV, na)
                    let (vr, ar, _, _) = unpackMain(rv, ra, rows: 1)
                    tc.store(&tc.uncondA, residualV: vr, residualA: ar)
                    uax0 = (ax - ua * sigma).asType(ax.dtype)
                } else {
                    let (_, ua) = sf(vx, ax, tc.uncondA.previousResidualV!, tc.uncondA.previousResidualA!)
                    uax0 = (ax - ua * sigma).asType(ax.dtype)
                }
            } else {
                let (_, ua, _, _) = forward(vx, ax, ts, condV, na)
                uax0 = (ax - ua * sigma).asType(ax.dtype)
            }
        } else {
            let (_, ua) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: condV, audioText: na,
                videoPos: videoPos, audioPos: audioPos, rope: rope,
                sparseVideo: cfg.sparseVideo)
            uax0 = (ax - ua * sigma).asType(ax.dtype)
        }
    }

    // ── 扰动前向（官方 MultiModalGuider：ptb=STG 扰动 + mod=模态隔离，各自独立一次前向）──
    // 仅 cond 条件，不做 TeaCache 跳步（官方无此语义；残差来自 cond 分支不适用扰动输入）。
    // 两个引导独立开关：stgScaleX≠0 → 跑 STG；modalityScaleX≠1 → 跑 modality。
    let stgVOn = cfg.stgScaleV != 0 && !cfg.stgBlocksV.isEmpty
    let stgAOn = cfg.stgScaleA != 0 && !cfg.stgBlocksA.isEmpty
    let modVOn = cfg.modalityScaleV != 1
    let modAOn = cfg.modalityScaleA != 1
    var ptx0: MLXArray? = nil   // STG 扰动 x0（视频）
    var pta0: MLXArray? = nil   // STG 扰动 x0（音频）
    var modx0: MLXArray? = nil  // modality 隔离 x0（视频）
    var moda0: MLXArray? = nil  // modality 隔离 x0（音频）
    if stgVOn || stgAOn {
        if let sf = stgForward {
            // STG 分段：主图真算时已产出 segStart-1 层激活（condMidV/A），ptb 图从 segStart 层继续，
            // 省掉引导分支前 segStart 层重复计算（每步引导省 segStart/48 前向）。
            // 主图跳步（TeaCache）时无中间激活，退回全量 ptb（数值语义不变）。
            if let seg = stgSegStart, seg > 0, let mv = condMidV, let ma = condMidA {
                let (pv, pa, _, _) = sf(mv, ma, ts, condV, condA)
                if stgVOn { ptx0 = (vx - pv.reshaped(vx.shape) * sigma).asType(vx.dtype) }
                if stgAOn { pta0 = (ax - pa * sigma).asType(ax.dtype) }
            } else {
                let (pv, pa, _, _) = sf(vx, ax, ts, condV, condA)
                if stgVOn { ptx0 = (vx - pv.reshaped(vx.shape) * sigma).asType(vx.dtype) }
                if stgAOn { pta0 = (ax - pa * sigma).asType(ax.dtype) }
            }
        } else {
            let (pv, pa) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: condV, audioText: condA,
                videoPos: videoPos, audioPos: audioPos, rope: rope,
                skipAVCross: false,
                stgBlocksV: cfg.stgBlocksV, stgBlocksA: cfg.stgBlocksA,
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                keyframesMLX: keyframesMLX,
                sparseVideo: cfg.sparseVideo)
            if stgVOn { ptx0 = (vx - pv.reshaped(vx.shape) * sigma).asType(vx.dtype) }
            if stgAOn { pta0 = (ax - pa * sigma).asType(ax.dtype) }
        }
    }
    if modVOn || modAOn {
        if let mf = modForward {
            let (mv, ma, _, _) = mf(vx, ax, ts, condV, condA)
            if modVOn { modx0 = (vx - mv.reshaped(vx.shape) * sigma).asType(vx.dtype) }
            if modAOn { moda0 = (ax - ma * sigma).asType(ax.dtype) }
        } else {
            let (mv, ma) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: condV, audioText: condA,
                videoPos: videoPos, audioPos: audioPos, rope: rope,
                skipAVCross: true,
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                keyframesMLX: keyframesMLX,
                sparseVideo: cfg.sparseVideo)
            if modVOn { modx0 = (vx - mv.reshaped(vx.shape) * sigma).asType(vx.dtype) }
            if modAOn { moda0 = (ax - ma * sigma).asType(ax.dtype) }
        }
    }

    // ── 统一引导合成（官方 MultiModalGuider：CFG + STG + modality，单次 norm-preserving rescale）──
    // 引导公式：pred = cond + (cfg-1)(cond-neg) + stgScale*(cond-ptb) + (modalityScale-1)*(cond-mod)
    vx0 = guiderCombineMulti(
        cond: vx0, neg: uvx0, cfg: cfg.cfgV,
        ptb: ptx0, stgScale: cfg.stgScaleV,
        mod: modx0, modalityScale: cfg.modalityScaleV,
        rescale: cfg.cfgRescaleV)
    if let frozen = frozenAudio {
        // 音频条件（frozen_a）：音频不重建，x0 直接 pin 为 clean latent，跳过引导合成与 rescale
        ax0 = frozen
    } else {
        ax0 = guiderCombineMulti(
            cond: ax0, neg: uax0, cfg: cfg.cfgA,
            ptb: pta0, stgScale: cfg.stgScaleA,
            mod: moda0, modalityScale: cfg.modalityScaleA,
            rescale: cfg.cfgRescaleA)
    }

    // ── I2V：把干净首帧 latent 钉入（applyDenoiseMask：x0 = x0*mask + clean*(1-mask)）──
    if let mask = condMask, let clean = cleanV {
        // vx0 [1,T,H,W,128]；mask [Nv] → [1,T,H,W,1] 广播（maskTensor 采样循环外已预构造）
        let m = maskTensor ?? maskAsTensor(mask, like: vx)
        vx0 = vx0 * m + clean * (1 - m)
    }
    return (vx0, ax0)
}

// MARK: - 完整采样循环（对齐 ditSampleCfg，纯 t2v / 音频条件）

/// 输入噪声 latent → 输出去噪后的 x0 latent（bf16）。
/// - frozenAudio: 非 nil 时启用音频条件（frozen_a 语义）：噪声音频直接用 clean
///   frozen token，每步音频 x0 pin 回 frozen（只 euler 收敛不重建），视频受音频条件化。
/// - initVideo/cleanV/condMask: 非 nil 时启用 I2V 首帧条件：initVideo 为已替换首帧的
///   初始 latent（ref_tokens + noise[HW:]），cleanV 为干净首帧 latent（ref_tokens + zeros），
///   condMask 前 HW 个 0（干净）其余 1（生成）；每步 applyDenoiseMask 钉入首帧。
func sampleLatentsCore(
    dit: LTXVideoDiT,
    noiseV: MLXArray,       // [1, T, H, W, 128]
    noiseA: MLXArray,       // [1, Na, 128]（frozenAudio 非 nil 时即 frozen token 本身）
    condV: MLXArray,        // [1, 256, 4096]
    condA: MLXArray,        // [1, 256, 2048]
    negV: MLXArray?,        // nil → 跳过 CFG（纯 cond 前向）
    negA: MLXArray?,
    frozenAudio: MLXArray?, // nil → 纯 t2v；非 nil → 音频条件（frozen_a）
    initVideo: MLXArray? = nil, // I2V：初始 latent（含首帧替换）；nil → noiseV
    cleanV: MLXArray? = nil,    // I2V：干净首帧 latent；nil → 无
    condMask: [Float]? = nil,   // I2V：前 HW 个 0（干净）其余 1；nil → 纯 t2v
    keyframesMLX: MLXArray? = nil, // IC-LoRA Stage2：独立 keyframes 标记 [Nv] f32（slots 生成+标记并存，与 denoise mask 解耦）
    guideClean: MLXArray? = nil, // 全片 latent guide（对齐官方 Detailer guiding_latents 语义）：干净 guide latent，σ0 域、shape 同 noiseV
    guideWeight: Float = 0,     // guide 强度（0=关闭）。每步前向把 vx 拉向 guide 的当前 σ 加噪版：vx=(1-w)*vx + w*(guide*(1-σ)+noise*σ)
    videoPos: [Float],      // [Nv*3]
    audioPos: [Float],      // [Na]
    config: SamplerConfig,
    isCancelled: @escaping () -> Bool = { false }   // 每步前检查，true → 提前结束采样（队列取消用）
) -> (video: MLXArray, audio: MLXArray) {
    let Nv = videoPos.count / 3
    let isI2V = condMask != nil
    let sigmas = config.sigmas ?? dynamicShiftSchedule(
        numSteps: config.numSteps, numTokens: Nv,
        baseShift: config.baseShift, maxShift: config.maxShift,
        baseTokens: config.baseTokens, maxTokens: config.maxTokens,
        stretch: config.stretch, terminal: config.terminal)
    stageEnter(
        phase: .sampling,
        detail: "采样计划：\(sigmas.count - 1) 步，Nv=\(Nv)，sigma[0]=\(sigmas[0])，调度=\(config.sigmas == nil ? "DynamicShift" : "蒸馏固定表")，步进=\(config.ancestral ? "ancestral-SDE" : "确定性Euler")，音频条件=\(frozenAudio != nil)，I2V=\(isI2V)",
        protect: [.dit])

    // RoPE 预构造一次（只依赖位置），48 层复用
    let rope = buildBlockRope(config: dit.config, videoPos: videoPos, audioPos: audioPos)

    // I2V mask 预构造一次（仅依赖 Nv/shape）：编译图捕获 + 采样循环每步复用，省重复 CPU→GPU 拷贝
    let maskMLX: MLXArray? = condMask.map { MLXArray($0).asType(.float32) }                     // [Nv] f32
    let maskMLXInv: MLXArray? = condMask.map { MLXArray($0.map { 1.0 - $0 }).asType(.float32) } // [Nv] f32
    let maskTensor: MLXArray? = condMask.map { mask in
        maskAsTensor(mask, like: noiseV)
    }

    // 编译 DiT 单步前向（可选，默认开；编译产物持久缓存复用）
    // 音频条件模式需捕获 audioTimesteps=[0]（frozen），与纯 t2v 编译图分离
    // I2V：per-token timestep = mask*sigma 与 keyframes gate 均为纯张量运算，
    // 常量由闭包捕获进编译图；shapeKey 带 isI2V 标志防止与 T2V 图复用。
    var forward: CompiledDitForward? = nil
    // 扰动分支编译图（官方 MultiModalGuider 的 ptb/mod 前向）：
    // - forwardStg：STG 扰动（stgBlocksV/A 生效，自注意力 passthrough）
    // - forwardMod：模态隔离（skipAVCross=true，跳 AV 交叉）
    // 与主 cond 图仅 host 控制流参数不同，需独立编译；仅当对应引导开启时构建。
    let stgOn = (config.stgScaleV != 0 || config.stgScaleA != 0)
        && (!config.stgBlocksV.isEmpty || !config.stgBlocksA.isEmpty)
    let modOn = (config.modalityScaleV != 1 || config.modalityScaleA != 1)
    var forwardStg: CompiledDitForward? = nil
    var forwardMod: CompiledDitForward? = nil
    // 编译预热复用：首次编译的预热前向（输入 noiseV + σ0）本身就是首步的 cond 前向，
    // 折算为 x0 后供 i==0 复用，避免首次运行白跑一次全尺寸 forward（stage2 全分辨率约 180s）。
    // 前提：vx==noiseV（非 initVideo 替换）且无任何引导生效（CFG/STG/modality 均关闭，扰动分支与
    // 预热折算路径数值不等价——有引导时首步需要 neg/ptb/mod 分支，预热只覆盖 cond 分支）。
    var preWarmX0: (v: MLXArray, a: MLXArray)? = nil
    let negEffective = (negV != nil && config.cfgV > 1.0) || (negA != nil && config.cfgA > 1.0)
    let guidanceEffective = negEffective || stgOn || modOn
    // CFG 分支 batch 合并：负向有效（至少一个模态 CFG>1）时，cond/uncondV/uncondA 文本条件
    // 拼 batch 一次前向替代三次独立调用（详见 ditX0Guided）。数值严格等价；与 TeaCache 兼容
    // （判据三分支共享同一 probe，真算/跳步三分支一致）。batch 图与单分支图分开编译缓存。
    let needV = negV != nil && config.cfgV > 1.0
    let needA = frozenAudio == nil && negA != nil && config.cfgA > 1.0
    let batchCount = 1 + (needV ? 1 : 0) + (needA ? 1 : 0)
    let useForwardBatch = batchCount > 1
    var forwardBatch: CompiledDitForward? = nil
    // STG 分段启动层：stgBlocks（视频+音频并集）的最小值。>0 才分段（STG 从 block0 扰动时无公共前缀可省）；
    // 分段后主图额外输出 segStart-1 层激活（emitMidAt），ptb 图从 segStart 层继续，每步引导省 segStart/48 前向。
    let stgSegStart: Int? = stgOn ? {
        guard let m = (config.stgBlocksV + config.stgBlocksA).min(), m > 0 else { return nil }
        return m
    }() : nil
    if config.useCompile {
        // 编译图依赖 latent/text 的 shape（rope/位置常量由 shape 隐含决定）+ 是否音频条件 + 是否 I2V + 是否 keyframes 标记 + 扰动参数
        // + in-context LoRA 旁路通道标签（LoRABypassTag：原生 off / IC / CQ 及权重·强度差异必须分槽，
        //   否则同 shape 下会命中他通道的编译图 → LoRA 不注入或串味。仅标签变化，不改变任何数值语义）
        let shapeKey = "\(noiseV.shape)-\(noiseA.shape)-\(condV.shape)-\(condA.shape)-frozen\(frozenAudio != nil)-i2v\(isI2V)-kf\(keyframesMLX != nil)-stg\(stgOn ? "\(config.stgBlocksV)|\(config.stgBlocksA)" : "off")-mod\(modOn ? "on" : "off")-sol\(config.sparseVideo?.cacheTag ?? "off")-lora\(LoRABypassTag.current)"
        // 主图 emitMidAt 改变编译图结构，缓存 key 带 mid 后缀区分
        let midSuffix = (stgSegStart != nil) ? "-mid\(stgSegStart!)" : ""
        let fwdKey = (useForwardBatch ? shapeKey + "-batch\(batchCount)" : shapeKey) + midSuffix
        if let cached = CompiledForwardCache.shared.get(dit: dit, key: fwdKey) {
            pipelineLog("🛠 [compile] 复用已编译 DiT 前向（shape 不变，跳过编译）key=\(fwdKey)")
            forward = cached
        } else {
            // 编译+预热窗口水位保护：全尺寸图编译是峰值内存区（紧邻 H3 大缓存残留或压力分偏高时
            // 编译分配即触发 jetsam SIGKILL）。ensureLoose 按压力分门槛触发，达标时卸载非保护模块，
            // 让编译+预热在较低水位起步。
            // [诊断打点] 编译窗口前水位：ensureLoose/clearCache 之前打，反映进入编译区的原始水位。
            memPointLog("compile 窗口前(ensureLoose前)")
            MemoryPolicy.ensureLoose(protect: [.dit])
            MLX.Memory.clearCache()
            pipelineLog("🧩 [compile] MLX.compile 编译 DiT 前向中（首次约 30-90s）... key=\(fwdKey)")
            let tC = Date()
            let cf = makeCompiledDitForward(
                dit: dit, rope: rope, videoPos: videoPos, audioPos: audioPos,
                audioTimesteps: frozenAudio == nil ? nil : MLXArray([Float](repeating: 0.0, count: useForwardBatch ? batchCount : 1)),
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                keyframesMLX: keyframesMLX,
                emitMidAt: stgSegStart.map { $0 - 1 },
                sparseVideo: config.sparseVideo)
            // 预热：跑一次编译（触发图构建 + 内核编译），用噪声 latent 同 shape。
            // batch 合并图用拼好的 batch 输入预热（触发 B 维编译），wv/wa 取 cond 行。
            let warmT = Date()
            // [诊断打点] 编译图已构建、预热 eval 尚未执行：此处能打出→已安全过图构建期，崩点在 eval 内；
            // 若在上一行之后中断且无此条→死在 makeCompiledDitForward 图构建分配。
            memPointLog("compile 图构建完成(eval前)")
            let (wv, wa, _, _): (MLXArray, MLXArray, MLXArray, MLXArray)
            if useForwardBatch {
                var wvTexts: [MLXArray] = [condV]
                var waTexts: [MLXArray] = [condA]
                if needV { wvTexts.append(negV!); waTexts.append(condA) }
                if needA { wvTexts.append(condV); waTexts.append(negA!) }
                let wvx = concatenated([MLXArray](repeating: noiseV, count: batchCount), axis: 0)
                let wax = concatenated([MLXArray](repeating: noiseA, count: batchCount), axis: 0)
                let (bv, ba, _, _) = cf(wvx, wax, MLXArray([Float](repeating: sigmas[0], count: batchCount)),
                                        concatenated(wvTexts, axis: 0), concatenated(waTexts, axis: 0))
                eval(bv, ba)
                wv = bv[0 ..< 1]
                wa = ba[0 ..< 1]
            } else {
                let (sv, sa, _, _) = cf(noiseV, noiseA, MLXArray([sigmas[0]]), condV, condA)
                eval(sv, sa)
                wv = sv
                wa = sa
            }
            pipelineLog("✅ [compile] 编译+预热完成（\(Int(Date().timeIntervalSince(tC)))s，含预热 \(Int(Date().timeIntervalSince(warmT)))s）")
            // [诊断打点] 预热 eval 返回后：能打出此条=编译期未爆，已进入首步采样；此条缺失=死在预热 eval 分配内。
            memPointLog("compile 预热eval完成")
            CompiledForwardCache.shared.store(cf, dit: dit, key: fwdKey)
            forward = cf
            // 预热复用折算：非 I2V 替换输入且无任何引导时，把预热前向输出（=首步 cond 前向 velocity）
            // 折算为首步 x0（对齐 ditX0Guided 无引导路径：x0 = x − vel·σ → guider rescale → I2V 钉入）
            if initVideo == nil, !guidanceEffective {
                let warmVx0 = (noiseV - wv.reshaped(noiseV.shape) * sigmas[0]).asType(noiseV.dtype)
                let warmAx0 = (noiseA - wa * sigmas[0]).asType(noiseA.dtype)
                let condVx0 = guiderCombineMulti(cond: warmVx0, neg: nil, cfg: config.cfgV, ptb: nil, stgScale: 0, mod: nil, modalityScale: 1, rescale: config.cfgRescaleV)
                let condAx0: MLXArray
                if let frozen = frozenAudio {
                    condAx0 = frozen
                } else {
                    condAx0 = guiderCombineMulti(cond: warmAx0, neg: nil, cfg: config.cfgA, ptb: nil, stgScale: 0, mod: nil, modalityScale: 1, rescale: config.cfgRescaleA)
                }
                var firstVx0 = condVx0
                if let mask = condMask, let clean = cleanV {
                    let m = maskTensor ?? maskAsTensor(mask, like: noiseV)
                    firstVx0 = firstVx0 * m + clean * (1 - m)
                }
                preWarmX0 = (firstVx0, condAx0)
            }
        }
        // 扰动分支编译图：与主图同 shape/条件，仅扰动参数不同；各自独立编译 + 预热。
        // 复用缓存 key 带扰动后缀（stg/mod），主图缓存不含扰动参数，互不串用。
        // STG 分段：ptb 图 fromLayer=segStart（输入为激活，跳过 patchify），key 带 seg 后缀。
        if stgOn {
            let segSuffix = (stgSegStart != nil) ? "-seg\(stgSegStart!)" : ""
            let stgKey = shapeKey + "-ptbstg" + segSuffix
            if let cached = CompiledForwardCache.shared.get(dit: dit, key: stgKey) {
                forwardStg = cached
            } else {
                print("MLX.compile 编译 STG 扰动前向（stgBlocksV=\(config.stgBlocksV) stgBlocksA=\(config.stgBlocksA)\(stgSegStart != nil ? " 分段启动=\(stgSegStart!)层" : "")）...")
                let tC = Date()
                let cf = makeCompiledDitForward(
                    dit: dit, rope: rope, videoPos: videoPos, audioPos: audioPos,
                    audioTimesteps: frozenAudio == nil ? nil : MLXArray([Float(0.0)]),
                    condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                    keyframesMLX: keyframesMLX,
                    skipAVCross: false,
                    stgBlocksV: config.stgBlocksV, stgBlocksA: config.stgBlocksA,
                    fromLayer: stgSegStart ?? 0,
                    sparseVideo: config.sparseVideo)
                // 预热：fromLayer>0 时输入为 [B,Nv,D]/[B,Na,DA] 激活（主图 emitMidAt 输出），
                // 用同 shape 占位（zeros）触发内核编译；运行期首步真算时实际喂入主图 mid。
                if let seg = stgSegStart, seg > 0 {
                    let nv = noiseV.ndim == 3 ? noiseV.shape[1] : noiseV.shape[1] * noiseV.shape[2] * noiseV.shape[3]
                    let na = noiseA.shape[1]
                    let d = dit.config.embedDim
                    let da = dit.config.audioEmbedDim
                    let warmV3 = MLXArray.zeros([noiseV.shape[0], nv, d], dtype: noiseV.dtype)
                    let warmA3 = MLXArray.zeros([noiseA.shape[0], na, da], dtype: noiseA.dtype)
                    let (wv, wa, _, _) = cf(warmV3, warmA3, MLXArray([sigmas[0]]), condV, condA)
                    eval(wv, wa)
                } else {
                    let (wv, wa, _, _) = cf(noiseV, noiseA, MLXArray([sigmas[0]]), condV, condA)
                    eval(wv, wa)
                }
                print("  STG 扰动图编译完成（\(Int(Date().timeIntervalSince(tC)))s）")
                CompiledForwardCache.shared.store(cf, dit: dit, key: stgKey)
                forwardStg = cf
            }
        }
        if modOn {
            let modKey = shapeKey + "-ptbmod"
            if let cached = CompiledForwardCache.shared.get(dit: dit, key: modKey) {
                forwardMod = cached
            } else {
                print("MLX.compile 编译 Modality 隔离前向（skipAVCross）...")
                let tC = Date()
                let cf = makeCompiledDitForward(
                    dit: dit, rope: rope, videoPos: videoPos, audioPos: audioPos,
                    audioTimesteps: frozenAudio == nil ? nil : MLXArray([Float(0.0)]),
                    condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                    keyframesMLX: keyframesMLX,
                    skipAVCross: true,
                    sparseVideo: config.sparseVideo)
                let (wv, wa, _, _) = cf(noiseV, noiseA, MLXArray([sigmas[0]]), condV, condA)
                eval(wv, wa)
                print("  Modality 隔离图编译完成（\(Int(Date().timeIntervalSince(tC)))s）")
                CompiledForwardCache.shared.store(cf, dit: dit, key: modKey)
                forwardMod = cf
            }
        }
    }

    var vx = initVideo ?? noiseV
    var ax = noiseA
    // TeaCache：官方 rel-L1 判据（特征相似跳 blocks、复用残差），cond/uncond 分支独立状态。
    // 仅在编译路径开启（probe/skip 均需编译图）；非编译路径自动退化为逐步真算。
    let teaCache = TeaCache()
    teaCache.threshold = config.teaCacheThreshold
    var probeForward: TeaCacheProbeForward? = nil
    var skipForward: TeaCacheSkipForward? = nil
    if config.teaCache, config.useCompile {
        let pf = makeTeaCacheProbeForward(
            dit: dit, condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX)
        let sf = makeTeaCacheSkipForward(
            dit: dit, condMask: condMask, maskMLXInv: maskMLXInv,
            keyframesMLX: keyframesMLX)
        // 预热：触发 probe/skip 图编译（轻量，数秒内完成）
        let nv = noiseV.ndim == 3 ? noiseV.shape[1] : noiseV.shape[1] * noiseV.shape[2] * noiseV.shape[3]
        let na = noiseA.shape[1]
        let d = dit.config.embedDim
        let da = dit.config.audioEmbedDim
        let _ = pf(noiseV, MLXArray([sigmas[0]]))
        let (sv, sa) = sf(noiseV, noiseA,
                          MLXArray.zeros([noiseV.shape[0], nv, d], dtype: noiseV.dtype),
                          MLXArray.zeros([noiseA.shape[0], na, da], dtype: noiseA.dtype))
        eval(sv, sa)
        print("  TeaCache 编译完成（官方 rel-L1 语义，阈值 \(teaCache.threshold)，probe/skip 图）")
        probeForward = pf
        skipForward = sf
    }
    let t0 = Date()
    for i in 0..<(sigmas.count - 1) {
        if isCancelled() {
            print("⏹ 采样已取消（第 \(i + 1) 步前），提前结束")
            MLX.Memory.clearCache()
            return (vx, ax)
        }
        let sigma = sigmas[i]
        let sigmaNext = sigmas[i + 1]
        let stepT0 = Date()
        // 每步打点/报告统一由循环尾 stageEnter 接管，不再单独打"步前"点（避免收集密度翻倍）
        // 每步独立作用域：x0/中间张量在本轮结束时立即释放，
        // 避免 48 层前向的中间结果攒到采样结束一次性析构（12GB 级大爆炸）。
        autoreleasepool {
            // 全片 latent guide（对齐官方 Detailer guiding_latents 语义）：每步前向把 vx 拉向
            // guide 的当前 σ 加噪版，模型始终能看到原片结构，只在锁定结构上精修。
            // i==0 起点本身已是 clean guide 的 σ0 混合（refineInit），无需再混（且避免破坏 preWarm 折算）。
            if i > 0, guideWeight > 0, let gc = guideClean {
                MLXRandom.seed(config.seed &+ UInt64(i) &+ 0x6E49_DE51) // guide 噪声独立于轨迹噪声
                let gNoise = MLXRandom.normal(gc.shape).asType(gc.dtype)
                let gNoised = (gc * (1 - sigma) + gNoise * sigma).asType(vx.dtype)
                vx = vx * (1 - guideWeight) + gNoised * guideWeight
                eval(vx)
            }
            // 每步特征探针一次（cond/uncond 共用：仅依赖 vx+sigma，与文本条件无关）
            var sharedProbe: MLXArray? = nil
            if let pf = probeForward {
                let p = pf(vx, MLXArray([sigma]))
                eval(p)
                sharedProbe = p
            }
            let x0: (v: MLXArray, a: MLXArray)
            if i == 0, let pw = preWarmX0 {
                // 首次编译时首步前向已被预热执行，直接复用折算的 x0（省 1 次全尺寸 DiT forward，stage2 约 180s）
                // 预热图不返回残差：TeaCache.update 后残差缺失会强制下一步真算并补存
                x0 = pw
                print("    [预热复用] 第 1 步复用编译预热输出，跳过本次 DiT 前向")
                if let p = sharedProbe {
                    teaCache.update(&teaCache.cond, probe: p)
                    teaCache.update(&teaCache.uncondV, probe: p)
                    teaCache.update(&teaCache.uncondA, probe: p)
                }
            } else if config.teaCache, probeForward != nil, skipForward != nil {
                x0 = ditX0Guided(
                    dit: dit, forward: forward,
                    forwardBatch: forwardBatch,
                    stgForward: stgOn ? forwardStg : nil,
                    stgSegStart: stgSegStart,
                    modForward: modOn ? forwardMod : nil,
                    vx: vx, ax: ax, sigma: sigma,
                    condV: condV, condA: condA,
                    negV: negV, negA: negA,
                    frozenAudio: frozenAudio,
                    condMask: condMask, cleanV: cleanV,
                    maskTensor: maskTensor, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                    keyframesMLX: keyframesMLX,
                    videoPos: videoPos, audioPos: audioPos,
                    rope: rope,
                    cfg: config,
                    teaCache: teaCache, branch: .cond,
                    probe: sharedProbe, skipForward: skipForward)
            } else {
                x0 = ditX0Guided(
                    dit: dit, forward: forward,
                    forwardBatch: forwardBatch,
                    stgForward: stgOn ? forwardStg : nil,
                    stgSegStart: stgSegStart,
                    modForward: modOn ? forwardMod : nil,
                    vx: vx, ax: ax, sigma: sigma,
                    condV: condV, condA: condA,
                    negV: negV, negA: negA,
                    frozenAudio: frozenAudio,
                    condMask: condMask, cleanV: cleanV,
                    maskTensor: maskTensor, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                    keyframesMLX: keyframesMLX,
                    videoPos: videoPos, audioPos: audioPos,
                    rope: rope,
                    cfg: config)
            }
            var nvx: MLXArray
            let nax: MLXArray
            if config.ancestral && sigmaNext > 0 {
                // 每步独立可复现噪声（seed 派生：seed + 步号 + 固定偏移，避免与初始 latent 噪声同源）
                MLXRandom.seed(config.seed &+ UInt64(i) &+ 0x9E37_79B9)
                let noiseV = MLXRandom.normal(vx.shape).asType(vx.dtype)
                let noiseA = MLXRandom.normal(ax.shape).asType(ax.dtype)
                nvx = ancestralEulerStep(vx, x0.v, sigma: sigma, sigmaNext: sigmaNext, noise: noiseV)
                nax = ancestralEulerStep(ax, x0.a, sigma: sigma, sigmaNext: sigmaNext, noise: noiseA)
            } else {
                nvx = eulerStep(vx, x0.v, sigma: sigma, sigmaNext: sigmaNext)
                nax = eulerStep(ax, x0.a, sigma: sigma, sigmaNext: sigmaNext)
            }
            // I2V：采样轨迹每步强制写回干净首帧（对齐官方 denoise_mask=0 语义：
            // post_process_latent 于 x_next，SDE 路径首帧 per-token sigma=0 不注入噪声）。
            // 不加此行时 ancestral 步的 renoise 会持续污染首帧，导致"首帧只闪一下"。
            if let mask = condMask, let clean = cleanV {
                let m = maskTensor ?? maskAsTensor(mask, like: nvx)
                nvx = nvx * m + clean * (1 - m)
            }
            eval(nvx, nax)
            vx = nvx
            ax = nax
        }
        // 每步报告/打点统一收敛为 stageEnter（替换原 步后打点 + 进度 print 组合；
        // 文本格式保持不变：pipelineLog 第一通道即 print，控制台行为不变）
        let elapsed = Int(Date().timeIntervalSince(stepT0))
        let total = Int(Date().timeIntervalSince(t0))
        stageEnter(
            phase: .sampling,
            step: i + 1,
            total: sigmas.count - 1,
            detail: String(format: "  [%2d/%d] σ=%.4f→%.4f  本步 %ds  累计 %ds", i + 1, sigmas.count - 1, sigma, sigmaNext, elapsed, total),
            protect: [.dit])
    }
    // 采样完成：编译图常驻缓存复用，不释放（避免 Compiled 大图一次性析构卡死）。
    // 仅回收采样过程中产生的空闲 buffer 缓存。
    MLX.Memory.clearCache()
    stageEnter(
        phase: .sampling,
        detail: "✅ 采样完成，总耗时 \(Int(Date().timeIntervalSince(t0)))s",
        protect: [.dit])
    return (vx, ax)
}

// MARK: - 蒸馏 / dev 公开采样入口（dev 与蒸馏拆分，共同部分在 sampleLatentsCore）

/// 蒸馏采样入口：固定蒸馏配置（无引导 cfgV/A=1.0、teaCache=false、默认 8 步
/// ltx25DistilledSigmas + ancestral SDE）。允许调用方覆盖 numSteps/sigmas/ancestral，
/// 以便 Stage2 refine（3 步确定性）复用同一入口。
/// - Parameter keyframesMLX: 保留参数（默认 nil）；IC-LoRA slots 已移除，I2V 首帧条件
///   仍可经 initVideo/cleanV/condMask 传入，本入口原样透传给核心。
func sampleLatentsDistilled(
    dit: LTXVideoDiT,
    noiseV: MLXArray,
    noiseA: MLXArray,
    condV: MLXArray,
    condA: MLXArray,
    negV: MLXArray? = nil,
    negA: MLXArray? = nil,
    frozenAudio: MLXArray? = nil,
    initVideo: MLXArray? = nil,
    cleanV: MLXArray? = nil,
    condMask: [Float]? = nil,
    keyframesMLX: MLXArray? = nil,
    guideClean: MLXArray? = nil, // 全片 latent guide（官方 Detailer 语义），原样透传 core
    guideWeight: Float = 0,
    videoPos: [Float],
    audioPos: [Float],
    numSteps: Int = 8,
    sigmas: [Float]? = ltx25DistilledSigmas,
    ancestral: Bool = true,
    seed: UInt64 = 42,
    // SOL 稀疏注意力（可选，默认 nil=关闭 → dense 全量注意力，行为与历史版本逐位一致）：
    // 仅视频自注意力 attn1 走 H3 公共 SOL 内核；调用方在编译窗口外解析开关后按常量传入
    // （MLX.compile 图内禁止读 env / print / eval）。
    sparseVideo: LTXSparseAttnConfig? = nil,
    isCancelled: @escaping () -> Bool = { false }
) -> (video: MLXArray, audio: MLXArray) {
    let config = SamplerConfig(
        numSteps: numSteps, cfgV: 1.0, cfgA: 1.0,
        sigmas: sigmas, ancestral: ancestral, seed: seed,
        teaCache: false,
        sparseVideo: sparseVideo)
    return sampleLatentsCore(
        dit: dit, noiseV: noiseV, noiseA: noiseA,
        condV: condV, condA: condA,
        negV: negV, negA: negA,
        frozenAudio: frozenAudio,
        initVideo: initVideo, cleanV: cleanV, condMask: condMask,
        keyframesMLX: keyframesMLX,
        guideClean: guideClean, guideWeight: guideWeight,
        videoPos: videoPos, audioPos: audioPos,
        config: config, isCancelled: isCancelled)
}

/// dev 采样入口：固定 dev 配置（20 步、cfgV 3.0/cfgA 7.0、teaCache=true、
/// STG/modality 引导、sigmas=nil 触发 DynamicShift）；负向条件由调用方传入。
/// 当前保留不调用（留 dev 管线），后续启用时按 Stage1Config.dev 参数表传入负向词。
func sampleLatentsDev(
    dit: LTXVideoDiT,
    noiseV: MLXArray,
    noiseA: MLXArray,
    condV: MLXArray,
    condA: MLXArray,
    negV: MLXArray?,
    negA: MLXArray?,
    frozenAudio: MLXArray? = nil,
    initVideo: MLXArray? = nil,
    cleanV: MLXArray? = nil,
    condMask: [Float]? = nil,
    keyframesMLX: MLXArray? = nil,
    videoPos: [Float],
    audioPos: [Float],
    numSteps: Int = 20,
    seed: UInt64 = 42,
    isCancelled: @escaping () -> Bool = { false }
) -> (video: MLXArray, audio: MLXArray) {
    let config = SamplerConfig(
        numSteps: numSteps, cfgV: 3.0, cfgA: 7.0,
        sigmas: nil, ancestral: false, seed: seed,
        teaCache: true,
        stgScaleV: 1.0, stgScaleA: 1.0,
        modalityScaleV: 3.0, modalityScaleA: 3.0,
        stgBlocksV: [28], stgBlocksA: [28])
    return sampleLatentsCore(
        dit: dit, noiseV: noiseV, noiseA: noiseA,
        condV: condV, condA: condA,
        negV: negV, negA: negA,
        frozenAudio: frozenAudio,
        initVideo: initVideo, cleanV: cleanV, condMask: condMask,
        keyframesMLX: keyframesMLX,
        videoPos: videoPos, audioPos: audioPos,
        config: config, isCancelled: isCancelled)
}
