// LTXDiT.swift — LTX-2.5 视频 DiT 主干（结构对齐真实 4bit 权重）
//
// 本文件在 ltx-test 内自包含实现 LTX-2.5 的 48 层双流 DiT 结构，
// 属性命名**完全对齐** transformer-distilled.safetensors 的权重键名，
// 使 `model.update(parameters:)` 可以零 remap 直接灌入全部权重。
//
// 依据：~/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/ 权重实际键结构 +
// embedded_config.json（AVTransformer3DModel, ltx25）。
//
// 与无限画布工程 model/ 下实现的差异点（均已按真实权重修正）：
//   · attention 用 to_q/to_k/to_v/to_out/to_gate_logits（gated attention），
//     全部带 bias（attention_bias: true）
//   · q_norm/k_norm 为无参 RMSNorm，作用在投影后 inner 维（[inner]）
//   · FFN 为 proj_in → gelu → proj_out 双投影（非 GEGLU 三投影）
//   · patchify_proj / audio_patchify_proj 是 Linear（128→dim），非 3D 卷积
//   · 主干含 prompt_adaln_single / keyframes_abs_pos_embedding
//   · 块级 norm 全部无参（norm_elementwise_affine: false）
//
// TODO(数值对齐): RoPE 精确实现、prompt 调制输入源、AV 交叉调制输入，
// 待 Phase B 对照参考实现（diffusers / mlx-serve）核对。

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
@preconcurrency import MLXNN

/// 调试用：打印每个 attention 的输入形状。
var debugShapePrint = false
/// 块级 RoPE 表（对齐参考 BlockRope）：v=视频自注意力、a=音频自注意力、vx/ax=AV 交叉。
struct BlockRope {
    let vcos: MLXArray
    let vsin: MLXArray
    let acos: MLXArray
    let asin: MLXArray
    let vxcos: MLXArray
    let vxsin: MLXArray
    let axcos: MLXArray
    let axsin: MLXArray
}

/// 预构造块级 RoPE 表（只依赖位置，与 timestep 无关）。
/// 采样循环外构造一次，48 层复用，避免每步重建 4 组频率表。
func buildBlockRope(config: LTXDiTConfig, videoPos: [Float], audioPos: [Float]) -> BlockRope {
    let nv = videoPos.count / 3
    let na = audioPos.count
    let maxV: [Float] = [20, 2048, 2048]
    let max1: [Float] = [20]
    let rv = ditRopeV2(pos: videoPos, N: nv, A: 3, numHeads: config.numHeads, headDim: config.headDim, maxPos: maxV)
    let ra = ditRopeV2(pos: audioPos, N: na, A: 1, numHeads: config.numAudioHeads, headDim: config.audioHeadDim, maxPos: max1)
    var vtmp = [Float](repeating: 0, count: nv)
    for n in 0..<nv { vtmp[n] = videoPos[n * 3] }
    let rvx = ditRopeV2(pos: vtmp, N: nv, A: 1, numHeads: config.numAudioHeads, headDim: config.audioHeadDim, maxPos: max1)
    let rax = ditRopeV2(pos: audioPos, N: na, A: 1, numHeads: config.numAudioHeads, headDim: config.audioHeadDim, maxPos: max1)
    // 统一转 bf16：ditRopeV2 产出 f32 cos/sin，而 q/k 为 bf16，
    // 直接相乘会把 q/k 提升到 f32，导致 fast.SDPA 收到 f32 q/k + bf16 v，
    // 触发隐式 cast（实测可拖慢 ~2×）。转 bf16 后 q/k/v 全 bf16，消除 cast。
    return BlockRope(
        vcos: rv.cos.asType(PrecisionPolicy.defaultMainDType), vsin: rv.sin.asType(PrecisionPolicy.defaultMainDType),
        acos: ra.cos.asType(PrecisionPolicy.defaultMainDType), asin: ra.sin.asType(PrecisionPolicy.defaultMainDType),
        vxcos: rvx.cos.asType(PrecisionPolicy.defaultMainDType), vxsin: rvx.sin.asType(PrecisionPolicy.defaultMainDType),
        axcos: rax.cos.asType(PrecisionPolicy.defaultMainDType), axsin: rax.sin.asType(PrecisionPolicy.defaultMainDType))
}


// MARK: - 时间步嵌入 + AdaLN（命名对齐 adaln_single.emb.timestep_embedder.*）

/// timestep_embedder：sinusoidal(256) → linear1 → SiLU → linear2。
final class LTXTimestepEmbedder: Module {
    let linear1: Linear
    let linear2: Linear
    let dim: Int

    init(dim: Int) {
        self.dim = dim
        self.linear1 = Linear(256, dim, bias: true)
        self.linear2 = Linear(dim, dim, bias: true)
        super.init()
    }

    /// t：[B] float32 → [B, dim]
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let emb = sinusoidalEmbedding(t, dim: 256)
        return linear2(silu(linear1(emb)))
    }
}

/// adaln_single：emb.timestep_embedder + linear（→ numEmbeddings × dim）。
final class LTXAdaLN: Module {
    let emb: LTXAdaLNEmb
    let linear: Linear
    let numEmbeddings: Int
    let dim: Int

    init(dim: Int, numEmbeddings: Int, timestepScaleMultiplier: Float = 1000.0) {
        self.numEmbeddings = numEmbeddings
        self.dim = dim
        self.emb = LTXAdaLNEmb(dim: dim, timestepScaleMultiplier: timestepScaleMultiplier)
        self.linear = Linear(dim, numEmbeddings * dim, bias: true)
        super.init()
    }

    /// t：[B] → (ada: [B, numEmbeddings, dim], embedded: [B, dim])
    func callAsFunction(_ t: MLXArray) -> (ada: MLXArray, embedded: MLXArray) {
        let embedded = emb(t)
        let ada = linear(silu(embedded)).reshaped([t.shape[0], numEmbeddings, dim])
        return (ada, embedded)
    }
}

/// 中间容器：属性名 emb 下再挂 timestep_embedder，对齐权重路径
/// `adaln_single.emb.timestep_embedder.linear1.weight`。
final class LTXAdaLNEmb: Module {
    let timestep_embedder: LTXTimestepEmbedder
    let timestepScaleMultiplier: Float

    init(dim: Int, timestepScaleMultiplier: Float) {
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.timestep_embedder = LTXTimestepEmbedder(dim: dim)
        super.init()
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        precondition(t.dtype == .float32, "时间步需为 float32")
        let scaled = t * timestepScaleMultiplier
        return timestep_embedder(scaled)
    }
}

// MARK: - Gated Attention（对齐 to_q/to_k/to_v/to_out/to_gate_logits + q_norm/k_norm）

/// LTX gated attention：
///   q = q_norm(to_q(x))，k = k_norm(to_k(ctx))，v = to_v(ctx)
///   按 head 计算注意力 → 门控 softmax 按 head 加权 → to_out 投影。
/// 门控 logits 由 **query 输入** 计算：gate = softmax(to_gate_logits(x))。
final class LTXGatedAttention: Module {
    // fusedQKV=true 时 Q/K/V 合并为单个投影 to_qkv（同源自注意力，权重文件已离线融合），
    // to_q/to_k/to_v 为 nil；否则保持三个独立投影（跨模态注意力的 Q/KV 异源，不可融合）。
    let to_qkv: QuantizedLinear?
    let to_q: QuantizedLinear?
    let to_k: QuantizedLinear?
    let to_v: QuantizedLinear?
    let to_out: QuantizedLinear
    let to_gate_logits: QuantizedLinear
    let q_norm: RMSNorm
    let k_norm: RMSNorm

    // IC-LoRA 旁路（官方 IC-LoRA 覆盖 to_q/to_k/to_v/to_out，视频流 attn1/attn2）。
    // 默认 nil 且 icActive=false：原生路径零参与、零开销。
    // 数值：delta(x) = (x·Aᵀ)·Bᵀ，融合等价 W' = W + B@A（官方 scale=1.0 预缩放，直接相加）。
    var icLoraQ: ICLoRAPair? = nil
    var icLoraK: ICLoRAPair? = nil
    var icLoraV: ICLoRAPair? = nil
    var icLoraO: ICLoRAPair? = nil
    var icActive: Bool = false

    let queryDim: Int
    let contextDim: Int
    let innerDim: Int
    let heads: Int
    let headDim: Int
    let gateDim: Int

    init(
        queryDim: Int, contextDim: Int, innerDim: Int,
        heads: Int, headDim: Int, gateDim: Int,
        bias: Bool = true, normEps: Float = 1e-6,
        groupSize: Int = 64, bits: Int = 4,
        fusedQKV: Bool = false
    ) {
        self.queryDim = queryDim
        self.contextDim = contextDim
        self.innerDim = innerDim
        self.heads = heads
        self.headDim = headDim
        self.gateDim = gateDim
        if fusedQKV {
            self.to_qkv = QuantizedLinear(queryDim, innerDim * 3, bias: bias, groupSize: groupSize, bits: bits)
            self.to_q = nil
            self.to_k = nil
            self.to_v = nil
        } else {
            self.to_qkv = nil
            self.to_q = QuantizedLinear(queryDim, innerDim, bias: bias, groupSize: groupSize, bits: bits)
            self.to_k = QuantizedLinear(contextDim, innerDim, bias: bias, groupSize: groupSize, bits: bits)
            self.to_v = QuantizedLinear(contextDim, innerDim, bias: bias, groupSize: groupSize, bits: bits)
        }
        self.to_out = QuantizedLinear(innerDim, queryDim, bias: bias, groupSize: groupSize, bits: bits)
        self.to_gate_logits = QuantizedLinear(queryDim, heads, bias: true, groupSize: groupSize, bits: bits)
        self.q_norm = RMSNorm(dim: innerDim, eps: normEps)
        self.k_norm = RMSNorm(dim: innerDim, eps: normEps)
        super.init()
    }

    /// x：[B, Tq, queryDim]，context：[B, Tk, contextDim]
    /// - Returns: [B, Tq, queryDim]
    /// - perturbed: STG 时空扰动（官方 perturbation_mask 全 0 语义）。仅对自注意力使用：
    ///   跳过 Q/K 投影、RMSNorm、RoPE 与注意力计算，只做 value 投影直通
    ///   （out = to_v(context)），随后照常走 per-head gate 与 to_out。
    ///   对应官方 ``out = attn_out * mask + v * (1 - mask)`` 在 mask=0 时的退化路径。
    func callAsFunction(
        _ x: MLXArray, context: MLXArray,
        mask: MLXArray? = nil,
        qCos: MLXArray? = nil, qSin: MLXArray? = nil,
        kCos: MLXArray? = nil, kSin: MLXArray? = nil,
        perturbed: Bool = false,
        sparse: LTXSparseAttnConfig? = nil   // 非 nil → 自注意力走 H3 公共 SOL 稀疏内核（前置条件不满足自动回退 dense）
    ) -> MLXArray {
        let b = x.shape[0]
        let tq = x.shape[1]
        let tk = context.shape[1]

        // STG 扰动直通（自注意力专用）：跳过 q/k 投影与注意力，仅 value 投影 + gate + to_out。
        // 对齐官方 attention.py：all_perturbed → out = v（不做 q/k/rope/attention），
        // 仍走 gated_attention_function（per-head gate）与 to_out。
        if perturbed {
            let vProjP: MLXArray
            if let to_qkv {
                // 融合投影（自注意力 Q=KV=context）：取 V 段
                let qkvP = to_qkv(context)
                let dimP = innerDim
                vProjP = qkvP[0..., 0..., (2 * dimP)...]
            } else {
                vProjP = to_v!(context)
            }
            let vPT = vProjP.reshaped([b, tq, heads, headDim]).transposed(0, 2, 1, 3)  // [B,H,Tq,hd]
            let gateP = 2 * sigmoid(to_gate_logits(x))                                     // [B, Tq, H]
            let mergedP = vPT.transposed(0, 2, 1, 3)                                       // [B, Tq, H, hd]
            let weightedP = mergedP * gateP.expandedDimensions(axis: -1)
            let summedP = weightedP.reshaped([b, tq, heads * headDim])
            let outP = to_out(summedP)
            return outP
        }

        if debugShapePrint {
            print("  [attn] x \(x.shape) ctx \(context.shape) to_q \(to_qkv?.shape ?? to_q!.shape) to_k \(to_qkv != nil ? "fused" : "\(to_k!.shape)") gate \(to_gate_logits.shape)")
        }

        var qProj: MLXArray
        var kProj: MLXArray
        var vProj: MLXArray
        if let to_qkv {
            // 融合投影：Q/K/V 同源于 x，单次 matmul 后沿最后一维拆三段
            let qkv = to_qkv(x)                       // [B, Tq, 3*innerDim]
            let dim = innerDim
            qProj = qkv[0..., 0..., 0..<dim]
            kProj = qkv[0..., 0..., dim..<(2 * dim)]
            vProj = qkv[0..., 0..., (2 * dim)...]
        } else {
            qProj = to_q!(x)
            kProj = to_k!(context)
            vProj = to_v!(context)
            // IC-LoRA 旁路（仅分离投影路径；fusedQKV 权重无对应 LoRA 键，跳过）
            if icActive {
                if let d = icLoraQ?.delta(x) { qProj = qProj + d }
                if let d = icLoraK?.delta(context) { kProj = kProj + d }
                if let d = icLoraV?.delta(context) { vProj = vProj + d }
            }
        }
        // toQ/toK/toV 分别对 x / context 投影；自注意力（attn1）传 context==x，
        // 跨模态（attn2）传文本/其它。
        if debugShapePrint {
            print("    q \(qProj.shape) k \(kProj.shape) v \(vProj.shape) gate \(to_gate_logits(x).shape)")
        }
        var q = q_norm(qProj).reshaped([b, tq, heads, headDim])
        var k = k_norm(kProj).reshaped([b, tk, heads, headDim])
        let v = vProj.reshaped([b, tk, heads, headDim])

        // [B, heads, T, headDim]（RoPE 在 transpose 后按 split 方式应用，对齐参考）
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        if let qCos, let qSin {
            q = applyRopeSplit(q, cosF: qCos, sinF: qSin)
        }
        if let kCos, let kSin {
            k = applyRopeSplit(k, cosF: kCos, sinF: kSin)
        }
        let vT = v.transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(headDim))
        // 对齐参考：fast scaled_dot_product_attention（flash 内核，与 Zig mlx_fast 一致）
        // SOL 稀疏（可选，仅二采 refine 开启）：q/k/v 满足 [1,H,S,128] 时走 H3 公共 SOL 内核；
        // 跨模态/文本交叉（context≠self、headDim=64、非方阵）由桥接层自动回退 dense，语义不变。
        let ctxOut: MLXArray
        if let sparse, let sol = LTXSparseAttnBridge.attention(q, k, vT, scale: scale, cfg: sparse) {
            ctxOut = sol
        } else {
            ctxOut = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: vT, scale: scale, mask: nil)
        }

        // 门控：gate = 2*sigmoid(to_gate_logits(x)) [B, Tq, heads]
        let gate = 2 * sigmoid(to_gate_logits(x))          // [B, Tq, H]
        let merged = ctxOut.transposed(0, 2, 1, 3)         // [B, Tq, H, headDim]
        let weighted = merged * gate.expandedDimensions(axis: -1)
        let summed = weighted.reshaped([b, tq, heads * headDim])  // [B, Tq, innerDim]
        if debugShapePrint {
            eval(summed)
            print("    summed \(summed.shape) to_out \(to_out.shape)")
        }

        let out = to_out(summed)                           // [B, Tq, queryDim]
        // IC-LoRA to_out 旁路：作用于 gate 后合并向量（输入 dim = innerDim）
        if icActive, let d = icLoraO?.delta(summed) {
            return out + d
        }
        return out
    }
}

// MARK: - SOL 稀疏注意力适配层（LTX → H3 公共内核，公共侧零改动）

/// LTX 侧 SOL 稀疏注意力配置（仅二采 refine 可选开启；nil = 关闭，行为与 dense 完全一致）。
struct LTXSparseAttnConfig: Equatable {
    /// tau：每 (head, query 块) 自适应 sigma 阈值倍数（越小越精确、越慢；越大越稀疏）
    var tau: Float = 1.3
    /// 未选中块的 tail pooled 项（补 softmax 分母，抑制稀疏化后的能量漂移）
    var tail: Bool = true
    /// 恒精确的 key 块区间（首帧锚块，形象身份引导不可被路由淘汰）；超出实际块数自动收敛
    var sinkBlocks: Range<Int>? = nil
    /// v6 内核（BQ64 兄弟合并，吞吐更高；与 v3 语义一致，不做 topk 合并）
    var useV6: Bool = true

    /// 编译图缓存 key 标签：任一数值参数变化都必须触发重新编译（MLX.compile 图对常量固化）
    var cacheTag: String {
        let sink = sinkBlocks.map { "\($0.lowerBound)_\($0.upperBound)" } ?? "off"
        return "tau\(tau)-tail\(tail ? 1 : 0)-sink\(sink)-v6\(useV6 ? 1 : 0)"
    }
}

/// LTX → H3SolAttn 桥接层：把 LTX 自注意力张量 [B,H,S,HD] 直接交给 H3 公共 SOL 内核。
/// - 不重排行序：SOL 路由只按 key 块质心判相关性，与帧/像素的行序无关；
///   LTX 行序 (f*H+h)*W+w 可原样使用（首帧锚靠 sinkBlocks 恒精确保护，而非行偏移）。
/// - 任一前置条件不满足（batch≠1 / headDim≠128 / 非方阵 / S<block / 形态不符）返回 nil，
///   调用方回退 dense SDPA —— 保证功能等价、无花屏风险。
/// - traceSafe：SOL 内部读 env / print / MLX.eval 全部关闭，可安全进入 MLX.compile 图。
enum LTXSparseAttnBridge {
    static let supportedHeadDim = 128
    static let block = 64

    /// 是否满足 SOL 内核前置条件（纯元数据判断，无副作用）
    static func canApply(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> Bool {
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4 else { return false }
        guard q.shape[0] == 1, k.shape[0] == 1, v.shape[0] == 1 else { return false }
        guard q.shape[1] == k.shape[1], q.shape[1] == v.shape[1] else { return false }
        guard q.shape[3] == supportedHeadDim, k.shape[3] == supportedHeadDim, v.shape[3] == supportedHeadDim else { return false }
        guard q.shape[2] == k.shape[2], q.shape[2] == v.shape[2] else { return false }
        guard q.shape[2] >= block else { return false }
        return true
    }

    /// SOL 稀疏注意力；不可用时返回 nil（调用方回退 dense）
    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                          scale: Float, cfg: LTXSparseAttnConfig) -> MLXArray? {
        guard canApply(q, k, v) else { return nil }
        // sink 块区间按实际 key 块数收敛（首帧锚块数不得超过总块数）
        let s = q.shape[2]
        let nBlocks = (s + block - 1) / block
        var sink: Range<Int>? = nil
        if let r = cfg.sinkBlocks {
            let lo = max(0, min(r.lowerBound, nBlocks))
            let hi = max(lo, min(r.upperBound, nBlocks))
            if hi > lo { sink = lo..<hi }
        }
        let solCfg = H3SolAttnConfig(topkRatio: nil, tau: cfg.tau, sinkBlocks: sink, tail: cfg.tail)
        // g = 0：LTX 无 strip 条件行，全行按 tau 路由（首帧锚已由 sinkBlocks 恒精确保护）
        return H3SolAttn.forward(q: q, k: k, v: v, g: 0, scale: scale,
                                 cfg: solCfg, useV6: cfg.useV6, traceSafe: true)
    }
}

// MARK: - FFN（proj_in → gelu → proj_out）

final class LTXFFN: Module {
    let proj_in: QuantizedLinear
    let proj_out: QuantizedLinear
    let dim: Int
    let ffnDim: Int

    // IC-LoRA FFN 旁路：net.0.proj → proj_in、net.2 → proj_out（官方键 ff.net.0.proj / ff.net.2）
    var icLoraIn: ICLoRAPair? = nil
    var icLoraOut: ICLoRAPair? = nil
    var icActive: Bool = false

    init(dim: Int, ffnDim: Int, bias: Bool, groupSize: Int = 64, bits: Int = 4) {
        self.dim = dim
        self.ffnDim = ffnDim
        self.proj_in = QuantizedLinear(dim, ffnDim, bias: bias, groupSize: groupSize, bits: bits)
        self.proj_out = QuantizedLinear(ffnDim, dim, bias: bias, groupSize: groupSize, bits: bits)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mid = proj_in(x)
        // IC-LoRA proj_in 旁路：作用于调制后输入（官方 ff.net.0.proj）
        var g: MLXArray
        if icActive, let d = icLoraIn?.delta(x) {
            g = geluApprox(mid + d)
        } else {
            g = geluApprox(mid)
        }
        let out = proj_out(g)
        // IC-LoRA proj_out 旁路：作用于 gelu 后（官方 ff.net.2）
        if icActive, let d = icLoraOut?.delta(g) {
            return out + d
        }
        return out
    }
}

// MARK: - DiT 块

final class LTXVideoDiTBlock: Module {
    let layerIndex: Int

    // 视频流
    let norm1: RMSNormNoAffine
    let attn1: LTXGatedAttention
    let norm2: RMSNormNoAffine
    let attn2: LTXGatedAttention
    let norm3: RMSNormNoAffine
    let ff: LTXFFN
    let scale_shift_table: MLXArray          // [9, 4096]
    let prompt_scale_shift_table: MLXArray   // [2, 4096]

    // 音频流
    let audio_norm1: RMSNormNoAffine
    let audio_attn1: LTXGatedAttention
    let audio_norm2: RMSNormNoAffine
    let audio_attn2: LTXGatedAttention
    let audio_norm3: RMSNormNoAffine
    let audio_ff: LTXFFN
    let audio_scale_shift_table: MLXArray          // [9, 2048]
    let audio_prompt_scale_shift_table: MLXArray   // [2, 2048]

    // 跨模态
    let audio_to_video_norm: RMSNormNoAffine
    let audio_to_video_attn: LTXGatedAttention
    let video_to_audio_norm: RMSNormNoAffine
    let video_to_audio_attn: LTXGatedAttention
    let scale_shift_table_a2v_ca_video: MLXArray   // [5, 4096]
    let scale_shift_table_a2v_ca_audio: MLXArray   // [5, 2048]

    init(layerIndex: Int, config: LTXDiTConfig, groupSize: Int = 64, bits: Int = 4) {
        self.layerIndex = layerIndex
        let d = config.embedDim
        let da = config.audioEmbedDim
        let heads = config.numHeads
        let headDim = config.headDim
        let aHeads = config.numAudioHeads
        let aHeadDim = config.audioHeadDim
        let eps = config.normEps

        // 视频流（自注意力 Q=KV=视频，已离线融合 QKV）
        self.norm1 = RMSNormNoAffine(eps: eps)
        self.attn1 = LTXGatedAttention(
            queryDim: d, contextDim: d, innerDim: d, heads: heads, headDim: headDim, gateDim: d,
            groupSize: groupSize, bits: bits, fusedQKV: false)
        self.norm2 = RMSNormNoAffine(eps: eps)
        self.attn2 = LTXGatedAttention(
            queryDim: d, contextDim: d, innerDim: d, heads: heads, headDim: headDim, gateDim: d,
            groupSize: groupSize, bits: bits, fusedQKV: false)
        self.norm3 = RMSNormNoAffine(eps: eps)
        self.ff = LTXFFN(dim: d, ffnDim: config.ffnDim, bias: config.ffBias, groupSize: groupSize, bits: bits)
        self.scale_shift_table = MLXArray.zeros([9, d])
        self.prompt_scale_shift_table = MLXArray.zeros([2, d])

        // 音频流（自注意力 Q=KV=音频，已离线融合 QKV）
        self.audio_norm1 = RMSNormNoAffine(eps: eps)
        self.audio_attn1 = LTXGatedAttention(
            queryDim: da, contextDim: da, innerDim: da, heads: aHeads, headDim: aHeadDim, gateDim: da,
            groupSize: groupSize, bits: bits, fusedQKV: false)
        self.audio_norm2 = RMSNormNoAffine(eps: eps)
        self.audio_attn2 = LTXGatedAttention(
            queryDim: da, contextDim: da, innerDim: da, heads: aHeads, headDim: aHeadDim, gateDim: da,
            groupSize: groupSize, bits: bits, fusedQKV: false)
        self.audio_norm3 = RMSNormNoAffine(eps: eps)
        self.audio_ff = LTXFFN(dim: da, ffnDim: config.audioFfnDim, bias: config.audioFfBias, groupSize: groupSize, bits: bits)
        self.audio_scale_shift_table = MLXArray.zeros([9, da])
        self.audio_prompt_scale_shift_table = MLXArray.zeros([2, da])

        // 跨模态
        self.audio_to_video_norm = RMSNormNoAffine(eps: eps)
        self.audio_to_video_attn = LTXGatedAttention(
            queryDim: d, contextDim: da, innerDim: da, heads: heads, headDim: aHeadDim, gateDim: d,
            groupSize: groupSize, bits: bits)
        self.video_to_audio_norm = RMSNormNoAffine(eps: eps)
        self.video_to_audio_attn = LTXGatedAttention(
            queryDim: da, contextDim: d, innerDim: da, heads: aHeads, headDim: aHeadDim, gateDim: da,
            groupSize: groupSize, bits: bits)
        self.scale_shift_table_a2v_ca_video = MLXArray.zeros([5, d])
        self.scale_shift_table_a2v_ca_audio = MLXArray.zeros([5, da])

        super.init()
    }

    /// 块前向（对齐参考 ditBlock）。
    /// - Parameters:
    ///   - vx/ax: 视频 [B, Nv, D] / 音频 [B, Na, DA]
    ///   - videoTemb/audioTemb: 主干 adaLN 调制表 [B, 1, 9, dim]
    ///   - promptTemb/audioPromptTemb: prompt 调制表 [B, 1, 2, dim]
    ///   - crossVideoSS/crossAudioSS: AV 交叉调制 [B, 1, 5, dim]（per-token 时 [1, Nv, 5, dim]）
    ///   - videoText/audioText: 文本条件嵌入 [B, Nt, D] / [B, Nt, DA]（attn2 的 KV）
    ///   - rope: 视频/音频自注意力与 AV 交叉的 3D RoPE 表
    ///   - skipAVCross: true 时跳过 AV 交叉注意力（Modality Guidance severed 前向用）
    ///   - stgVideo/stgAudio: true 时本块视频/音频自注意力走 STG 扰动直通
    ///     （官方 SKIP_VIDEO_SELF_ATTN/SKIP_AUDIO_SELF_ATTN 扰动，mask=0 → value 投影 passthrough）
    ///   - perTokenVideo: true 时 video 侧调制表为 per-token（I2V cond_mask，clean 帧时间步 0）
    func callAsFunction(
        _ vx: MLXArray, _ ax: MLXArray,
        videoTemb: MLXArray, audioTemb: MLXArray,
        promptTemb: MLXArray, audioPromptTemb: MLXArray,
        crossVideoSS: MLXArray, crossAudioSS: MLXArray,
        videoText: MLXArray, audioText: MLXArray,
        rope: BlockRope,
        skipAVCross: Bool = false,
        stgVideo: Bool = false,
        stgAudio: Bool = false,
        perTokenVideo: Bool = false,
        sparseVideo: LTXSparseAttnConfig? = nil   // 非 nil → 视频自注意力 attn1 走 SOL 稀疏（音频/交叉 attn 保持 dense）
    ) -> (MLXArray, MLXArray) {
        let b = vx.shape[0]
        let d = vx.shape[2]
        let da = ax.shape[2]

        // 9 通道调制表：0,1,2=自注意力 / 3,4,5=FF / 6,7,8=文本交叉
        // per-token（I2V）：videoTemb [1,Nv,9,d] → vSST [B,Nv,9,d]；否则 [B,1,9,d]+ → [B,9,d]
        let vSST: MLXArray
        if perTokenVideo {
            vSST = (scale_shift_table.reshaped([1, 1, 1, 9, d]) + videoTemb).reshaped([b, videoTemb.shape[1], 9, d])
        } else {
            vSST = (scale_shift_table.reshaped([1, 1, 9, d]) + videoTemb).reshaped([b, 9, d])
        }
        let aSST = (audio_scale_shift_table.reshaped([1, 1, 9, da]) + audioTemb).reshaped([b, 9, da])
        // 2 通道 prompt 调制表（shift=0, scale=1）
        let vPST = (prompt_scale_shift_table.reshaped([1, 1, 2, d]) + promptTemb).reshaped([b, 2, d])
        let aPST = (audio_prompt_scale_shift_table.reshaped([1, 1, 2, da]) + audioPromptTemb).reshaped([b, 2, da])
        // 5 通道 AV 交叉调制（SCALE-first）：0,1=A2V scale/shift，2,3=V2A scale/shift，4=gate
        // per-token：crossVideoSS [1,Nv,5,d] → vCA [B,Nv,5,d]；否则 [B,1,5,d]+ → [B,5,d]
        let vCA: MLXArray
        if perTokenVideo {
            vCA = (scale_shift_table_a2v_ca_video.reshaped([1, 1, 1, 5, d]) + crossVideoSS).reshaped([b, crossVideoSS.shape[1], 5, d])
        } else {
            vCA = (scale_shift_table_a2v_ca_video.reshaped([1, 1, 5, d]) + crossVideoSS).reshaped([b, 5, d])
        }
        let aCA = (scale_shift_table_a2v_ca_audio.reshaped([1, 1, 5, da]) + crossAudioSS).reshaped([b, 5, da])

        /// modulate：norm(x) * (1 + scale) + shift（scale=row sc, shift=row si）
        func mod9(_ norm: RMSNormNoAffine, _ x: MLXArray, _ sst: MLXArray, _ si: Int, _ sc: Int) -> MLXArray {
            let shift: MLXArray
            let scale: MLXArray
            if sst.ndim == 4 {
                // [B, Nv, 9, d]：取每个 token 的通道 → [B, Nv, d]（3D，对齐 zig adalnRowN）
                shift = sst[0..., 0..., si, 0...]
                scale = sst[0..., 0..., sc, 0...]
            } else {
                shift = sst[0..., si, 0...].expandedDimensions(axis: 1)
                scale = sst[0..., sc, 0...].expandedDimensions(axis: 1)
            }
            return norm(x) * (1 + scale) + shift
        }
        func gate9(_ sst: MLXArray, _ gi: Int) -> MLXArray {
            if sst.ndim == 4 {
                return sst[0..., 0..., gi, 0...]
            }
            return sst[0..., gi, 0...].expandedDimensions(axis: 1)
        }
        /// modulate 直接作用于已 norm 输入：x * (1 + scale) + shift
        /// 4D（per-token）：[1,Nv,P,d] → 取 token 维通道 → [1,Nv,d]（3D，对齐 zig adalnRowN）
        func modNormed(_ x: MLXArray, _ sst: MLXArray, _ si: Int, _ sc: Int) -> MLXArray {
            if sst.ndim == 4 {
                let shift = sst[0..., 0..., si, 0...]
                let scale = sst[0..., 0..., sc, 0...]
                return x * (1 + scale) + shift
            }
            let shift = sst[0..., si, 0...].expandedDimensions(axis: 1)
            let scale = sst[0..., sc, 0...].expandedDimensions(axis: 1)
            return x * (1 + scale) + shift
        }

        var video = vx
        var audio = ax

        // 1. 视频自注意力 attn1（idx 0,1,2；Q=KV=视频，RoPE rv）
        // STG：stgVideo=true 时本块走扰动直通（仅 value 投影，跳过注意力）
        let vNormed1 = mod9(norm1, video, vSST, 0, 1)
        let vAttn1 = attn1(
            vNormed1, context: vNormed1,
            qCos: rope.vcos, qSin: rope.vsin, kCos: rope.vcos, kSin: rope.vsin,
            perturbed: stgVideo,
            sparse: sparseVideo)
        video = video + vAttn1 * gate9(vSST, 2)

        // 2. 音频自注意力 audio_attn1（idx 0,1,2；Q=KV=音频，RoPE ra）
        let aNormed1 = mod9(audio_norm1, audio, aSST, 0, 1)
        let aAttn1 = audio_attn1(
            aNormed1, context: aNormed1,
            qCos: rope.acos, qSin: rope.asin, kCos: rope.acos, kSin: rope.asin,
            perturbed: stgAudio)
        audio = audio + aAttn1 * gate9(aSST, 2)

        // 3. 视频文本交叉注意力 attn2（idx 6,7,8；Q=视频，KV=prompt 调制后的 videoText，无 RoPE）
        let vNormed2 = mod9(norm2, video, vSST, 6, 7)
        // prompt 调制作用于 text 输入：shift=row0, scale=row1
        let vpScale = vPST[0..., 1, 0...].expandedDimensions(axis: 1)
        let vpShift = vPST[0..., 0, 0...].expandedDimensions(axis: 1)
        let vText = videoText * (1 + vpScale) + vpShift
        video = video + attn2(vNormed2, context: vText) * gate9(vSST, 8)

        // 4. 音频文本交叉注意力 audio_attn2（idx 6,7,8；Q=音频，KV=调制后的 audioText，无 RoPE）
        let aNormed2 = mod9(audio_norm2, audio, aSST, 6, 7)
        let apScale = aPST[0..., 1, 0...].expandedDimensions(axis: 1)
        let apShift = aPST[0..., 0, 0...].expandedDimensions(axis: 1)
        let aText = audioText * (1 + apScale) + apShift
        audio = audio + audio_attn2(aNormed2, context: aText) * gate9(aSST, 8)

        // 5-6. 跨模态（共用 A2V 前的 norm 结果 vn3/an3）
        // Modality 隔离：skipAVCross=true 时整段跳过（官方 SKIP_A2V_CROSS_ATTN/SKIP_V2A_CROSS_ATTN
        // 扰动，对应 cross_attn_perturbation_mask=0 → AV 交叉注意力贡献置 0）
        if !skipAVCross {
            let vn3 = audio_to_video_norm(video)   // 视频流 rmsAF
            let an3 = video_to_audio_norm(audio)   // 音频流 rmsAF

            // A2V：Q=视频(vn3+a2v 调制)，KV=音频(an3+a2v 调制)；RoPE q=rvx, k=rax
            let vq = modNormed(vn3, vCA, 1, 0)
            let akv = modNormed(an3, aCA, 1, 0)
            let a2v = audio_to_video_attn(
                vq, context: akv,
                qCos: rope.vxcos, qSin: rope.vxsin, kCos: rope.axcos, kSin: rope.axsin)
            // A2V gate：per-token 时 vCA 4D [1,Nv,5,d] 取 [1,Nv,d]（3D）；否则 3D [1,5,d] 取 [1,1,d]
            let a2vGate = perTokenVideo
                ? vCA[0..., 0..., 4, 0...]
                : vCA[0..., 4, 0...].expandedDimensions(axis: 1)
            video = video + a2v * a2vGate

            // V2A：Q=音频(an3+v2a 调制)，KV=视频(vn3+v2a 调制，A2V 前 norm)；RoPE q=rax, k=rvx
            let aq = modNormed(an3, aCA, 3, 2)
            let vkv = modNormed(vn3, vCA, 3, 2)
            let v2a = video_to_audio_attn(
                aq, context: vkv,
                qCos: rope.axcos, qSin: rope.axsin, kCos: rope.vxcos, kSin: rope.vxsin)
            audio = audio + v2a * aCA[0..., 4, 0...].expandedDimensions(axis: 1)
        }

        // 7. 视频 FFN（idx 3,4,5）
        video = video + ff(mod9(norm3, video, vSST, 3, 4)) * gate9(vSST, 5)

        // 8. 音频 FFN（idx 3,4,5）
        audio = audio + audio_ff(mod9(audio_norm3, audio, aSST, 3, 4)) * gate9(aSST, 5)

        return (video, audio)
    }
}

// MARK: - DiT 主干

struct LTXDiTConfig {
    var numLayers: Int = 48
    var embedDim: Int = 4096
    var numHeads: Int = 32
    var headDim: Int = 128
    var ffnDim: Int = 16384
    var audioEmbedDim: Int = 2048
    var numAudioHeads: Int = 32
    var audioHeadDim: Int = 64
    var audioFfnDim: Int = 8192
    var inChannels: Int = 128
    var outChannels: Int = 128
    var audioInChannels: Int = 128
    var audioOutChannels: Int = 128
    var timestepScaleMultiplier: Float = 1000.0
    var normEps: Float = 1e-6
    var ffBias: Bool = false
    var audioFfBias: Bool = true
}

final class LTXVideoDiT: Module {
    let config: LTXDiTConfig

    // 输入投影（Linear，未量化）
    let patchify_proj: Linear
    let audio_patchify_proj: Linear

    // 时间步 / prompt / AV 交叉调制（主干级）
    let adaln_single: LTXAdaLN
    let audio_adaln_single: LTXAdaLN
    let prompt_adaln_single: LTXAdaLN
    let audio_prompt_adaln_single: LTXAdaLN
    let av_ca_video_scale_shift_adaln_single: LTXAdaLN
    let av_ca_audio_scale_shift_adaln_single: LTXAdaLN
    let av_ca_a2v_gate_adaln_single: LTXAdaLN
    let av_ca_v2a_gate_adaln_single: LTXAdaLN

    // 输出投影（Linear，未量化）
    let proj_out: Linear
    let audio_proj_out: Linear

    // 主干调制表
    let scale_shift_table: MLXArray               // [2, 4096]
    let audio_scale_shift_table: MLXArray         // [2, 2048]
    var keyframes_abs_pos_embedding: MLXArray     // [1, 4096]

    // DiT 块
    let transformer_blocks: [LTXVideoDiTBlock]

    init(config: LTXDiTConfig = LTXDiTConfig(), groupSize: Int = 64, bits: Int = 4) {
        self.config = config
        let d = config.embedDim
        let da = config.audioEmbedDim
        let mult = config.timestepScaleMultiplier

        self.patchify_proj = Linear(config.inChannels, d, bias: true)
        self.audio_patchify_proj = Linear(config.audioInChannels, da, bias: true)

        self.adaln_single = LTXAdaLN(dim: d, numEmbeddings: 9, timestepScaleMultiplier: mult)
        self.audio_adaln_single = LTXAdaLN(dim: da, numEmbeddings: 9, timestepScaleMultiplier: mult)
        self.prompt_adaln_single = LTXAdaLN(dim: d, numEmbeddings: 2, timestepScaleMultiplier: mult)
        self.audio_prompt_adaln_single = LTXAdaLN(dim: da, numEmbeddings: 2, timestepScaleMultiplier: mult)
        self.av_ca_video_scale_shift_adaln_single = LTXAdaLN(dim: d, numEmbeddings: 4, timestepScaleMultiplier: mult)
        self.av_ca_audio_scale_shift_adaln_single = LTXAdaLN(dim: da, numEmbeddings: 4, timestepScaleMultiplier: mult)
        // AV gate 用 sigma×1000（config av_ca_timestep_scale_multiplier=1000）
        self.av_ca_a2v_gate_adaln_single = LTXAdaLN(dim: d, numEmbeddings: 1, timestepScaleMultiplier: 1.0)
        self.av_ca_v2a_gate_adaln_single = LTXAdaLN(dim: da, numEmbeddings: 1, timestepScaleMultiplier: 1.0)

        self.proj_out = Linear(d, config.outChannels, bias: true)
        self.audio_proj_out = Linear(da, config.audioOutChannels, bias: true)

        self.scale_shift_table = MLXArray.zeros([2, d])
        self.audio_scale_shift_table = MLXArray.zeros([2, da])
        self.keyframes_abs_pos_embedding = MLXArray.zeros([1, d])

        self.transformer_blocks = (0..<config.numLayers).map {
            LTXVideoDiTBlock(layerIndex: $0, config: config, groupSize: groupSize, bits: bits)
        }

        super.init()
    }

    /// 主干前向（对齐参考 ditX0）。
    /// videoLatent [B, T, H, W, C]，audioLatent [B, TA, CA]，
    /// timesteps/audioTimesteps [B] float32（sigma），
    /// videoText/audioText 文本条件嵌入 [B, Nt, D]/[B, Nt, DA]，
    /// videoPos [Nv*3]（t,h,w 展平）/ audioPos [Na]（t）。
    /// - skipAVCross: true 时跳过全部 AV 交叉注意力（Modality Guidance severed 前向）
    /// - condMask: 非 nil 时启用 I2V 首帧条件：mask[n]=1 生成、0 干净首帧；
    ///   video AdaLN/AV 交叉走 per-token timestep（mask*sigma），clean token 时间步 0，
    ///   并在 patchify 后注入 keyframes_abs_pos_embedding（gate=1-mask）。
    func callAsFunction(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timesteps: MLXArray,
        audioTimesteps: MLXArray? = nil,
        videoText: MLXArray,
        audioText: MLXArray,
        videoPos: [Float],
        audioPos: [Float],
        rope: BlockRope? = nil,
        skipAVCross: Bool = false,
        stgBlocksV: [Int]? = nil,   // STG 扰动 block 索引（视频自注意力，官方 stg_blocks=[28]）
        stgBlocksA: [Int]? = nil,   // STG 扰动 block 索引（音频自注意力）
        condMask: [Float]? = nil,
        maskMLX: MLXArray? = nil,      // I2V：预构造 mask [Nv] f32（采样循环外构造一次，避免每步 CPU→GPU 拷贝）
        maskMLXInv: MLXArray? = nil,    // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
        keyframesMLX: MLXArray? = nil,  // 独立 keyframes 标记 [Nv] f32（1=注入 keyframes_abs_pos_embedding）
                                       // 优先于 maskMLXInv：IC-LoRA Stage2 keyframe slots 需要
                                       // denoise_mask=1 生成 + 标记并存，官方 keyframes_mask 与 denoise_mask 解耦
        sparseVideo: LTXSparseAttnConfig? = nil  // 非 nil → 视频自注意力走 SOL 稀疏（二采 refine 开）
    ) -> (video: MLXArray, audio: MLXArray) {
        let b = videoLatent.shape[0]
        // 支持 3D token 级输入 [B, Nv, C]（IC-LoRA Stage2 低清直放合并序列）与 5D [B, T, H, W, C] 两种形态
        let t: Int
        let h: Int
        let w: Int
        if videoLatent.ndim == 3 {
            t = 1; h = 1; w = 1
        } else {
            t = videoLatent.shape[1]
            h = videoLatent.shape[2]
            w = videoLatent.shape[3]
        }
        let nv = videoLatent.ndim == 3 ? videoLatent.shape[1] : t * h * w
        let d = config.embedDim
        let da = config.audioEmbedDim

        // 输入投影（Linear patchify）
        var video = patchify_proj(videoLatent).reshaped([b, nv, d])
        var audio = audio_patchify_proj(audioLatent)     // [B, TA, DA]

        // I2V：注入 keyframes_abs_pos_embedding（gate=1-mask，只加在干净首帧 token）
        // IC-LoRA Stage2 keyframe slots：gate 用独立 keyframesMLX（slots mask=1 生成 + 标记并存，
        // 对齐官方 keyframes_mask 与 denoise_mask 解耦；guide reference 不再被误标记）
        // 注意：保持 sigma0 为 MLXArray（不调 .item()），避免编译 tracing 阶段同步求值死锁
        let sigma0 = timesteps[0 ..< 1]   // [1] f32，标量视图（单轴切片）
        if keyframesMLX != nil || (condMask != nil && condMask!.count == nv) {
            let gate: MLXArray
            if let kfGate = keyframesMLX {
                gate = kfGate.reshaped([1, nv, 1]).asType(video.dtype)
            } else if let mask = condMask, mask.count == nv {
                gate = (maskMLXInv ?? MLXArray(mask.map { 1.0 - $0 }).asType(.float32)).reshaped([1, nv, 1]).asType(video.dtype)
            } else {
                gate = MLXArray.zeros([1, nv, 1]).asType(video.dtype)
            }
            let kf = keyframes_abs_pos_embedding.reshaped([1, 1, d])
            video = video + gate * kf
        }

        // 主干调制表
        let audioTs = audioTimesteps ?? timesteps
        // I2V：video AdaLN 用 per-token timestep（mask*sigma，clean token→0）
        // 纯 tensor 运算（maskMLX * sigma0 广播），避免 Swift map 内同步求值
        let (videoAda, videoEmbedded) = condMask.map { mask in
            let perToken = ((maskMLX ?? MLXArray(mask).asType(.float32)) * sigma0).asType(.float32)
            return adaln_single(perToken)
        } ?? adaln_single(timesteps)
        // I2V per-token：videoAda [Nv,9,d] → [1,Nv,9,d]（axis 0）；否则 [B,9,d] → [B,1,9,d]（axis 1）
        let videoTemb = (condMask != nil)
            ? videoAda.expandedDimensions(axis: 0)
            : videoAda.expandedDimensions(axis: 1)
        let (audioAda, audioEmbedded) = audio_adaln_single(audioTs)
        let audioTemb = audioAda.expandedDimensions(axis: 1)          // [B, 1, 9, da]

        // prompt 调制（prompt_adaln_single，同 t_sin；audio_prompt 也用全局 sigma）
        let (vP, _) = prompt_adaln_single(timesteps)
        let promptTemb = vP.expandedDimensions(axis: 1)               // [B, 1, 2, d]
        let (aP, _) = audio_prompt_adaln_single(timesteps)
        let audioPromptTemb = aP.expandedDimensions(axis: 1)          // [B, 1, 2, da]

        // AV 交叉调制：video 侧 scale/shift 用全局 sigma（t_sin）、audio 侧用 audio sigma（t_sin_a），
        // 两个 gate 均用全局 sigma×1.0（t_sin_gate，multiplier=1.0 已内置于 LTXAdaLN）
        // I2V：video 侧 av_ca_video_scale_shift 走 per-token（与 video AdaLN 同 timestep）
        let vSS: MLXArray
        let vGate: MLXArray
        if let mask = condMask {
            let perToken = ((maskMLX ?? MLXArray(mask).asType(.float32)) * sigma0).asType(.float32)
            vSS = av_ca_video_scale_shift_adaln_single(perToken).ada
            vGate = av_ca_a2v_gate_adaln_single(timesteps).ada
        } else {
            vSS = av_ca_video_scale_shift_adaln_single(timesteps).ada
            vGate = av_ca_a2v_gate_adaln_single(timesteps).ada
        }
        let crossVideoSS: MLXArray
        if condMask != nil {
            // per-token：scale/shift [Nv,4,d] + gate 广播到每 token [1,1,d] → [1,Nv,5,d]
            crossVideoSS = concatenated(
                [vSS.expandedDimensions(axis: 0),
                 MLX.broadcast(vGate.expandedDimensions(axis: 0), to: [1, nv, 1, d])], axis: 2)  // [1,Nv,5,d]
        } else {
            crossVideoSS = concatenated(
                [vSS.expandedDimensions(axis: 1), vGate.expandedDimensions(axis: 1)], axis: 2)  // [B, 1, 5, d]
        }

        let (aSS, _) = av_ca_audio_scale_shift_adaln_single(audioTs)
        let (aGate, _) = av_ca_v2a_gate_adaln_single(timesteps)
        let crossAudioSS = concatenated(
            [aSS.expandedDimensions(axis: 1), aGate.expandedDimensions(axis: 1)], axis: 2)  // [B, 1, 5, da]

        // RoPE 3D（外部可预构造注入；默认现算）
        let rope = rope ?? buildBlockRope(config: config, videoPos: videoPos, audioPos: audioPos)

        // 48 层
        for (idx, block) in transformer_blocks.enumerated() {
            let stgV = stgBlocksV?.contains(idx) ?? false
            let stgA = stgBlocksA?.contains(idx) ?? false
            (video, audio) = block(
                video, audio,
                videoTemb: videoTemb, audioTemb: audioTemb,
                promptTemb: promptTemb, audioPromptTemb: audioPromptTemb,
                crossVideoSS: crossVideoSS, crossAudioSS: crossAudioSS,
                videoText: videoText, audioText: audioText,
                rope: rope,
                skipAVCross: skipAVCross,
                stgVideo: stgV, stgAudio: stgA,
                perTokenVideo: condMask != nil,
                sparseVideo: sparseVideo)
        }

        // 输出 head：无参 LayerNorm + modulate（row0=shift, row1=scale）+ proj
        // per-token（I2V）：videoEmbedded [Nv,d] → [1,Nv,1,d]，表 [2,d] → [1,1,2,d] → [1,Nv,2,d]
        let vSSOut: MLXArray
        if condMask != nil {
            vSSOut = (scale_shift_table.reshaped([1, 1, 2, d]) + videoEmbedded.reshaped([1, videoEmbedded.shape[0], 1, d]))
        } else {
            vSSOut = (scale_shift_table.reshaped([1, 1, 2, d]) + videoEmbedded.reshaped([b, 1, 1, d]))
        }
        let vShift = vSSOut[0..., 0..., 0, 0...]
        let vScale = vSSOut[0..., 0..., 1, 0...]
        let videoOut = proj_out(normOut(video) * (1 + vScale) + vShift)

        let aSSOut = (audio_scale_shift_table.reshaped([1, 1, 2, da]) + audioEmbedded.reshaped([b, 1, 1, da]))
        let aShift = aSSOut[0..., 0..., 0, 0...]
        let aScale = aSSOut[0..., 0..., 1, 0...]
        let audioOut = audio_proj_out(audioNormOut(audio) * (1 + aScale) + aShift)

        return (videoOut, audioOut)
    }

    /// 完整前向 + TeaCache 中间量：返回 (video, audio, videoResidual, audioResidual)。
    /// videoResidual/audioResidual = 输出调制（norm_out + modulate）后 − blocks 输入，
    /// 供官方 TeaCache 跳步复用（x += residual 后直接 proj_out，残差已含 norm+modulate）。
    /// 与 callAsFunction 计算结果一致，仅多带残差；采样主图编译本方法。
    func forwardWithResidual(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timesteps: MLXArray,
        audioTimesteps: MLXArray? = nil,
        videoText: MLXArray,
        audioText: MLXArray,
        videoPos: [Float],
        audioPos: [Float],
        rope: BlockRope? = nil,
        skipAVCross: Bool = false,
        stgBlocksV: [Int]? = nil,   // STG 扰动 block 索引（视频自注意力，官方 stg_blocks=[28]）
        stgBlocksA: [Int]? = nil,   // STG 扰动 block 索引（音频自注意力）
        condMask: [Float]? = nil,
        maskMLX: MLXArray? = nil,       // I2V：预构造 mask [Nv] f32（采样循环外构造一次，避免每步 CPU→GPU 拷贝）
        maskMLXInv: MLXArray? = nil,     // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
        keyframesMLX: MLXArray? = nil,    // 独立 keyframes 标记 [Nv] f32（1=注入 keyframes_abs_pos_embedding）
        fromLayer: Int = 0,             // >0：跳过 patchify/keyframes 注入，videoLatent/audioLatent 直接作为第 fromLayer 层输入激活（[B,Nv,D]/[B,Na,DA]）——STG 分段 ptb 图用
        emitMidAt: Int? = nil,          // 非 nil：跑完该层后把 (video, audio) 激活作为 midVideo/midAudio 额外返回——STG 分段主图用
        sparseVideo: LTXSparseAttnConfig? = nil  // 非 nil → 视频自注意力走 SOL 稀疏（二采 refine 开）
    ) -> (video: MLXArray, audio: MLXArray, videoResidual: MLXArray, audioResidual: MLXArray, midVideo: MLXArray?, midAudio: MLXArray?) {
        let b = videoLatent.shape[0]
        // 支持 3D token 级输入 [B, Nv, C]（IC-LoRA Stage2 低清直放合并序列）与 5D [B, T, H, W, C] 两种形态
        let t: Int
        let h: Int
        let w: Int
        if videoLatent.ndim == 3 {
            t = 1; h = 1; w = 1
        } else {
            t = videoLatent.shape[1]
            h = videoLatent.shape[2]
            w = videoLatent.shape[3]
        }
        let nv = videoLatent.ndim == 3 ? videoLatent.shape[1] : t * h * w
        let d = config.embedDim
        let da = config.audioEmbedDim

        // 输入投影（Linear patchify）。STG 分段（fromLayer>0）：输入已是第 fromLayer 层激活，直接进入 block 循环。
        var video: MLXArray
        var audio: MLXArray
        if fromLayer > 0 {
            video = videoLatent       // [B, Nv, D]
            audio = audioLatent       // [B, Na, DA]
        } else {
            video = patchify_proj(videoLatent).reshaped([b, nv, d])
            audio = audio_patchify_proj(audioLatent)     // [B, TA, DA]
        }

        // I2V：注入 keyframes_abs_pos_embedding（gate=1-mask，只加在干净首帧 token）
        // IC-LoRA Stage2 keyframe slots：gate 用独立 keyframesMLX（slots mask=1 生成 + 标记并存，
        // 对齐官方 keyframes_mask 与 denoise_mask 解耦；guide reference 不再被误标记）
        // 注意：保持 sigma0 为 MLXArray（不调 .item()），避免编译 tracing 阶段同步求值死锁
        let sigma0 = timesteps[0 ..< 1]   // [1] f32，标量视图（单轴切片）
        if fromLayer == 0, keyframesMLX != nil || (condMask != nil && condMask!.count == nv) {
            let gate: MLXArray
            if let kfGate = keyframesMLX {
                gate = kfGate.reshaped([1, nv, 1]).asType(video.dtype)
            } else if let mask = condMask, mask.count == nv {
                gate = (maskMLXInv ?? MLXArray(mask.map { 1.0 - $0 }).asType(.float32)).reshaped([1, nv, 1]).asType(video.dtype)
            } else {
                gate = MLXArray.zeros([1, nv, 1]).asType(video.dtype)
            }
            let kf = keyframes_abs_pos_embedding.reshaped([1, 1, d])
            video = video + gate * kf
        }

        // 主干调制表
        let audioTs = audioTimesteps ?? timesteps
        // I2V：video AdaLN 用 per-token timestep（mask*sigma，clean token→0）
        // 纯 tensor 运算（maskMLX * sigma0 广播），避免 Swift map 内同步求值
        let (videoAda, videoEmbedded) = condMask.map { mask in
            let perToken = ((maskMLX ?? MLXArray(mask).asType(.float32)) * sigma0).asType(.float32)
            return adaln_single(perToken)
        } ?? adaln_single(timesteps)
        // I2V per-token：videoAda [Nv,9,d] → [1,Nv,9,d]（axis 0）；否则 [B,9,d] → [B,1,9,d]（axis 1）
        let videoTemb = (condMask != nil)
            ? videoAda.expandedDimensions(axis: 0)
            : videoAda.expandedDimensions(axis: 1)
        let (audioAda, audioEmbedded) = audio_adaln_single(audioTs)
        let audioTemb = audioAda.expandedDimensions(axis: 1)          // [B, 1, 9, da]

        // prompt 调制（prompt_adaln_single，同 t_sin；audio_prompt 也用全局 sigma）
        let (vP, _) = prompt_adaln_single(timesteps)
        let promptTemb = vP.expandedDimensions(axis: 1)               // [B, 1, 2, d]
        let (aP, _) = audio_prompt_adaln_single(timesteps)
        let audioPromptTemb = aP.expandedDimensions(axis: 1)          // [B, 1, 2, da]

        // AV 交叉调制：video 侧 scale/shift 用全局 sigma（t_sin）、audio 侧用 audio sigma（t_sin_a），
        // 两个 gate 均用全局 sigma×1.0（t_sin_gate，multiplier=1.0 已内置于 LTXAdaLN）
        // I2V：video 侧 av_ca_video_scale_shift 走 per-token（与 video AdaLN 同 timestep）
        let vSS: MLXArray
        let vGate: MLXArray
        if let mask = condMask {
            let perToken = ((maskMLX ?? MLXArray(mask).asType(.float32)) * sigma0).asType(.float32)
            vSS = av_ca_video_scale_shift_adaln_single(perToken).ada
            vGate = av_ca_a2v_gate_adaln_single(timesteps).ada
        } else {
            vSS = av_ca_video_scale_shift_adaln_single(timesteps).ada
            vGate = av_ca_a2v_gate_adaln_single(timesteps).ada
        }
        let crossVideoSS: MLXArray
        if condMask != nil {
            // per-token：scale/shift [Nv,4,d] + gate 广播到每 token [1,1,d] → [1,Nv,5,d]
            crossVideoSS = concatenated(
                [vSS.expandedDimensions(axis: 0),
                 MLX.broadcast(vGate.expandedDimensions(axis: 0), to: [1, nv, 1, d])], axis: 2)  // [1,Nv,5,d]
        } else {
            crossVideoSS = concatenated(
                [vSS.expandedDimensions(axis: 1), vGate.expandedDimensions(axis: 1)], axis: 2)  // [B, 1, 5, d]
        }

        let (aSS, _) = av_ca_audio_scale_shift_adaln_single(audioTs)
        let (aGate, _) = av_ca_v2a_gate_adaln_single(timesteps)
        let crossAudioSS = concatenated(
            [aSS.expandedDimensions(axis: 1), aGate.expandedDimensions(axis: 1)], axis: 2)  // [B, 1, 5, da]

        // RoPE 3D（外部可预构造注入；默认现算）
        let rope = rope ?? buildBlockRope(config: config, videoPos: videoPos, audioPos: audioPos)

        // TeaCache 残差基准：blocks 输入（patchify + keyframes 注入后）
        let oriVideo = video
        let oriAudio = audio

        // 48 层（STG 分段：从 fromLayer 开始，前面层由主图激活输入承接）
        var midVideo: MLXArray? = nil
        var midAudio: MLXArray? = nil
        for idx in fromLayer..<transformer_blocks.count {
            let block = transformer_blocks[idx]
            let stgV = stgBlocksV?.contains(idx) ?? false
            let stgA = stgBlocksA?.contains(idx) ?? false
            (video, audio) = block(
                video, audio,
                videoTemb: videoTemb, audioTemb: audioTemb,
                promptTemb: promptTemb, audioPromptTemb: audioPromptTemb,
                crossVideoSS: crossVideoSS, crossAudioSS: crossAudioSS,
                videoText: videoText, audioText: audioText,
                rope: rope,
                skipAVCross: skipAVCross,
                stgVideo: stgV, stgAudio: stgA,
                perTokenVideo: condMask != nil,
                sparseVideo: sparseVideo)
            if let em = emitMidAt, idx == em {
                midVideo = video
                midAudio = audio
            }
        }

        // 输出 head：无参 LayerNorm + modulate（row0=shift, row1=scale）+ proj
        // per-token（I2V）：videoEmbedded [Nv,d] → [1,Nv,1,d]，表 [2,d] → [1,1,2,d] → [1,Nv,2,d]
        let vSSOut: MLXArray
        if condMask != nil {
            vSSOut = (scale_shift_table.reshaped([1, 1, 2, d]) + videoEmbedded.reshaped([1, videoEmbedded.shape[0], 1, d]))
        } else {
            vSSOut = (scale_shift_table.reshaped([1, 1, 2, d]) + videoEmbedded.reshaped([b, 1, 1, d]))
        }
        let vShift = vSSOut[0..., 0..., 0, 0...]
        let vScale = vSSOut[0..., 0..., 1, 0...]
        let vModOut = normOut(video) * (1 + vScale) + vShift

        let aSSOut = (audio_scale_shift_table.reshaped([1, 1, 2, da]) + audioEmbedded.reshaped([b, 1, 1, da]))
        let aShift = aSSOut[0..., 0..., 0, 0...]
        let aScale = aSSOut[0..., 0..., 1, 0...]
        let aModOut = audioNormOut(audio) * (1 + aScale) + aShift

        return (proj_out(vModOut), audio_proj_out(aModOut),
                vModOut - oriVideo, aModOut - oriAudio,
                midVideo, midAudio)
    }

    /// TeaCache 特征探针：只算到第一个 transformer block 的视频调制输入
    /// （官方 teacache_ltxvmodel_forward 的 modulated_inp，即 block0 img_attn_norm 调制结果）。
    /// 与 forwardWithResidual 前段完全一致（patchify + keyframes 注入 + video AdaLN 调制表），
    /// 无注意力/FFN，作为独立轻量编译图供每步跳步判据使用（cond/uncond 共用同一 latent/sigma）。
    func teaCacheProbe(
        videoLatent: MLXArray,
        timesteps: MLXArray,
        condMask: [Float]? = nil,
        maskMLX: MLXArray? = nil,       // I2V：预构造 mask [Nv] f32
        maskMLXInv: MLXArray? = nil,     // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
        keyframesMLX: MLXArray? = nil    // 独立 keyframes 标记 [Nv] f32
    ) -> MLXArray {
        let b = videoLatent.shape[0]
        let t: Int
        let h: Int
        let w: Int
        if videoLatent.ndim == 3 {
            t = 1; h = 1; w = 1
        } else {
            t = videoLatent.shape[1]
            h = videoLatent.shape[2]
            w = videoLatent.shape[3]
        }
        let nv = videoLatent.ndim == 3 ? videoLatent.shape[1] : t * h * w
        let d = config.embedDim

        var video = patchify_proj(videoLatent).reshaped([b, nv, d])

        let sigma0 = timesteps[0 ..< 1]   // [1] f32，标量视图（单轴切片）
        if keyframesMLX != nil || (condMask != nil && condMask!.count == nv) {
            let gate: MLXArray
            if let kfGate = keyframesMLX {
                gate = kfGate.reshaped([1, nv, 1]).asType(video.dtype)
            } else if let mask = condMask, mask.count == nv {
                gate = (maskMLXInv ?? MLXArray(mask.map { 1.0 - $0 }).asType(.float32)).reshaped([1, nv, 1]).asType(video.dtype)
            } else {
                gate = MLXArray.zeros([1, nv, 1]).asType(video.dtype)
            }
            let kf = keyframes_abs_pos_embedding.reshaped([1, 1, d])
            video = video + gate * kf
        }

        let (videoAda, _) = condMask.map { mask in
            let perToken = ((maskMLX ?? MLXArray(mask).asType(.float32)) * sigma0).asType(.float32)
            return adaln_single(perToken)
        } ?? adaln_single(timesteps)
        let videoTemb = (condMask != nil)
            ? videoAda.expandedDimensions(axis: 0)
            : videoAda.expandedDimensions(axis: 1)

        // 官方 modulated_inp = rms_norm(x) * (1 + scale_msa) + shift_msa（第一个 block 自注意力调制输入）
        let block0 = transformer_blocks[0]
        let vSST: MLXArray
        if condMask != nil {
            vSST = (block0.scale_shift_table.reshaped([1, 1, 1, 9, d]) + videoTemb).reshaped([b, videoTemb.shape[1], 9, d])
        } else {
            vSST = (block0.scale_shift_table.reshaped([1, 1, 9, d]) + videoTemb).reshaped([b, 9, d])
        }
        let shift: MLXArray
        let scale: MLXArray
        if vSST.ndim == 4 {
            shift = vSST[0..., 0..., 0, 0...]
            scale = vSST[0..., 0..., 1, 0...]
        } else {
            shift = vSST[0..., 0, 0...].expandedDimensions(axis: 1)
            scale = vSST[0..., 1, 0...].expandedDimensions(axis: 1)
        }
        return block0.norm1(video) * (1 + scale) + shift
    }

    /// TeaCache 跳步前向：blocks 输入 + 上一步输出调制残差 → 输出投影。
    /// 官方跳步语义：x += previous_residual（残差已含 norm_out+modulate），
    /// 直接 proj_out，省掉 48 层 blocks + norm+modulate 的全部计算。
    func forwardSkipResidual(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        videoResidual: MLXArray,
        audioResidual: MLXArray,
        condMask: [Float]? = nil,
        maskMLXInv: MLXArray? = nil,     // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
        keyframesMLX: MLXArray? = nil    // 独立 keyframes 标记 [Nv] f32
    ) -> (video: MLXArray, audio: MLXArray) {
        let b = videoLatent.shape[0]
        let t: Int
        let h: Int
        let w: Int
        if videoLatent.ndim == 3 {
            t = 1; h = 1; w = 1
        } else {
            t = videoLatent.shape[1]
            h = videoLatent.shape[2]
            w = videoLatent.shape[3]
        }
        let nv = videoLatent.ndim == 3 ? videoLatent.shape[1] : t * h * w
        let d = config.embedDim

        var video = patchify_proj(videoLatent).reshaped([b, nv, d])
        if keyframesMLX != nil || (condMask != nil && condMask!.count == nv) {
            let gate: MLXArray
            if let kfGate = keyframesMLX {
                gate = kfGate.reshaped([1, nv, 1]).asType(video.dtype)
            } else if let mask = condMask, mask.count == nv {
                gate = (maskMLXInv ?? MLXArray(mask.map { 1.0 - $0 }).asType(.float32)).reshaped([1, nv, 1]).asType(video.dtype)
            } else {
                gate = MLXArray.zeros([1, nv, 1]).asType(video.dtype)
            }
            let kf = keyframes_abs_pos_embedding.reshaped([1, 1, d])
            video = video + gate * kf
        }
        video = video + videoResidual
        let audio = audio_patchify_proj(audioLatent) + audioResidual
        return (proj_out(video), audio_proj_out(audio))
    }

    /// 输出 norm（无参 LayerNorm，对齐参考 layerNormAF）
    let normOut = LayerNormNoAffine()
    let audioNormOut = LayerNormNoAffine()
}
