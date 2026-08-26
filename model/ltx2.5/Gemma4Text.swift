//
//  Gemma4Text.swift — LTX-2.5 文本编码器（Gemma-4-12B-LTX）
//
//  移植自 mlx-serve 的 gemma4-12b-ltx-v1 参考（transformer.zig / model.zig）：
//  - 48 层 Gemma4，layer_types 每 6 层第 6 层为 full_attention（索引 5,11,...,47）
//  - 4bit group64 量化（QuantizedLinear / QuantizedEmbedding，键名零 remap 对齐）
//  - global 层：16 heads × 512，kv 1 head，proportional RoPE（dims=512，freqs=[64 real + 192 inf]）
//  - sliding 层：16 heads × 256，kv 8 heads，标准 RoPE（base=10000，dims=256）
//  - attention_k_eq_v 逐层生效：global 层 V 复用 K 投影（无 v_proj 权重），sliding 层独立 v_proj + 两者都过 parameter-free v_norm
//  - embedding × sqrt(3840)（scale_embeddings=true）
//  - 四重 RMSNorm：input / post_attention(1e-8) / pre_feedforward / post_feedforward(1e-8)
//  - layer_scalar 在 attn 与 MLP 残差后乘到 h
//  - 输出：49 个 [1, T, 3840] 残差态（states[0] = embedding 输出，states[i+1] = 第 i 层输出）
//
//  权重键前缀：model.language_model.（embed_tokens / norm / layers.N.*）
//
//  Created by 花茶鱼i on 2026/8/19.
//

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
@preconcurrency import MLXNN

// MARK: - Gemma4 常量

enum Gemma4Config {
    static let hiddenSize = 3840
    static let intermediateSize = 15360
    static let numLayers = 48
    static let numHeads = 16
    static let kvHeads = 8
    static let globalKvHeads = 1
    static let headDim = 256          // sliding 层 head_dim
    static let globalHeadDim = 512    // global 层 head_dim
    static let rmsNormEps: Float = 1e-6
    static let postNormEps: Float = 1e-8
    static let vocabSize = 262144
    static let padTokenID: Int32 = 0
    static let groupSize = 64
    static let bits = 4
    static let slidingRopeTheta: Float = 10000.0
    static let globalRopeTheta: Float = 1_000_000.0
    static let partialRotaryFactor: Float = 0.25   // global 层旋转 512*0.25=128 维（64 对），余 192 维 pad inf
}

/// 判断层是否为 full_attention（layer_types 每 6 层第 6 个 full）。
func gemma4IsGlobal(_ layerIndex: Int) -> Bool {
    layerIndex % 6 == 5
}

/// global 层 proportional RoPE freqs：[64 real + 192 inf] = 256 = globalHeadDim/2。
func gemma4GlobalRopeFreqs() -> MLXArray {
    let ghd = Float(Gemma4Config.globalHeadDim)
    let nRot = Int(Float(Gemma4Config.globalHeadDim) * Gemma4Config.partialRotaryFactor / 2)
    let nPad = Gemma4Config.globalHeadDim / 2 - nRot
    var freqs = [Float](repeating: 0, count: nRot + nPad)
    for i in 0..<nRot {
        let exponent = Float(2 * i) / ghd
        freqs[i] = pow(Gemma4Config.globalRopeTheta, exponent)
    }
    for i in nRot..<(nRot + nPad) {
        freqs[i] = .infinity
    }
    return MLXArray(freqs)
}

// MARK: - Gemma4 单层注意力

final class Gemma4Attention: Module {
    @ModuleInfo var q_proj: QuantizedLinear
    @ModuleInfo var k_proj: QuantizedLinear
    // 逐层 k_eq_v：global（full_attention）层 V 复用 K 投影（权重无 v_proj）；
    // sliding 层有独立 v_proj，必须单独声明加载。
    @ModuleInfo var v_proj: QuantizedLinear?
    @ModuleInfo var o_proj: QuantizedLinear
    @ModuleInfo var q_norm: RMSNorm
    @ModuleInfo var k_norm: RMSNorm

    let isGlobal: Bool
    let headDim: Int
    let kvHeads: Int

    init(isGlobal: Bool) {
        self.isGlobal = isGlobal
        self.headDim = isGlobal ? Gemma4Config.globalHeadDim : Gemma4Config.headDim
        self.kvHeads = isGlobal ? Gemma4Config.globalKvHeads : Gemma4Config.kvHeads
        let hd = self.headDim
        let kvh = self.kvHeads
        self.q_proj = QuantizedLinear(
            Gemma4Config.hiddenSize, Gemma4Config.numHeads * hd,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        self.k_proj = QuantizedLinear(
            Gemma4Config.hiddenSize, kvh * hd,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        // sliding 层才创建 v_proj；global 层 k_eq_v，权重文件无 v_proj 键，保持 nil
        if !isGlobal {
            self.v_proj = QuantizedLinear(
                Gemma4Config.hiddenSize, kvh * hd,
                bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        }
        self.o_proj = QuantizedLinear(
            Gemma4Config.numHeads * hd, Gemma4Config.hiddenSize,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        self.q_norm = RMSNorm(dim: hd, eps: Gemma4Config.rmsNormEps)
        self.k_norm = RMSNorm(dim: hd, eps: Gemma4Config.rmsNormEps)
        super.init()
    }

    /// 前向：[1,T,3840] → [1,T,3840]。mask [1,1,T,T] bf16 additive。
    func callAsFunction(
        _ h: MLXArray, mask: MLXArray,
        globalFreqs: MLXArray?, slidingFreqs: MLXArray?
    ) -> MLXArray {
        let T = h.shape[1]
        let qHeads = Gemma4Config.numHeads
        let hd = self.headDim
        let kvh = self.kvHeads

        // QKV 投影：kRaw 保留 pre-norm 供 V 复用；V 分支见下
        let qRaw = q_proj(h).reshaped([1, T, qHeads, hd])
        let kRaw = k_proj(h).reshaped([1, T, kvh, hd])   // [1,T,kvh,hd]
        let k = kRaw.transposed(0, 2, 1, 3)   // [1,kvh,T,hd]

        // q/k RMSNorm（qRaw [1,T,H,hd] → [1,H,T,hd]；k 已在 [1,kvh,T,hd] 直接 norm）
        let qNormed = q_norm(qRaw).transposed(0, 2, 1, 3)
        let kNormed = k_norm(k)

        // RoPE（global：proportional freqs；sliding：标准 base=10000）
        let qRot: MLXArray
        let kRot: MLXArray
        if isGlobal {
            qRot = MLXFast.RoPE(
                qNormed, dimensions: hd, traditional: false,
                base: nil, scale: 1.0, offset: 0, freqs: globalFreqs)
            kRot = MLXFast.RoPE(
                kNormed, dimensions: hd, traditional: false,
                base: nil, scale: 1.0, offset: 0, freqs: globalFreqs)
        } else {
            // sliding 层：标准 RoPE，仅 base（freqs 必须为 nil，否则与 base 冲突）
            qRot = MLXFast.RoPE(
                qNormed, dimensions: hd, traditional: false,
                base: Gemma4Config.slidingRopeTheta, scale: 1.0, offset: 0, freqs: nil)
            kRot = MLXFast.RoPE(
                kNormed, dimensions: hd, traditional: false,
                base: Gemma4Config.slidingRopeTheta, scale: 1.0, offset: 0, freqs: nil)
        }

        // V 分支（对齐 mlx-serve / MLXLLM）：
        //   sliding 层：v_proj(h) 独立投影；
        //   global 层 k_eq_v：V 复用 K 投影输出（kRaw，未过 k_norm、未 RoPE）。
        // 两者都要再过 parameter-free v_norm（RMSNorm，weight=none），且 V 不过 RoPE。
        let vPre: MLXArray
        if let vp = v_proj {
            vPre = vp(h).reshaped([1, T, kvh, hd])
        } else {
            vPre = kRaw
        }
        let v = MLXFast.rmsNorm(vPre, weight: .mlxNone, eps: Gemma4Config.rmsNormEps)
            .transposed(0, 2, 1, 3)   // [1,kvh,T,hd]

        // SDPA（scale=1.0，参考 attn_scale=1）
        FileHandle.standardError.write("DBG layer\(isGlobal ? "G" : "S") q=\(qRot.shape) k=\(kRot.shape) v=\(v.shape)\n".data(using: .utf8)!)
        let attn = MLXFast.scaledDotProductAttention(
            queries: qRot, keys: kRot, values: v, scale: 1.0, mask: mask)
        let attnReshaped = attn.transposed(0, 2, 1, 3).reshaped([1, T, qHeads * hd])
        return o_proj(attnReshaped)
    }
}

// MARK: - Gemma4 单层

final class Gemma4MLP: Module {
    @ModuleInfo var gate_proj: QuantizedLinear
    @ModuleInfo var up_proj: QuantizedLinear
    @ModuleInfo var down_proj: QuantizedLinear

    override init() {
        self.gate_proj = QuantizedLinear(
            Gemma4Config.hiddenSize, Gemma4Config.intermediateSize,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        self.up_proj = QuantizedLinear(
            Gemma4Config.hiddenSize, Gemma4Config.intermediateSize,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        self.down_proj = QuantizedLinear(
            Gemma4Config.intermediateSize, Gemma4Config.hiddenSize,
            bias: false, groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        super.init()
    }

    /// GeGLU：gate 用 tanh 近似 gelu
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = geluApprox(gate_proj(x))
        let up = up_proj(x)
        return down_proj(gate * up)
    }
}

final class Gemma4Layer: Module {
    @ModuleInfo var input_layernorm: RMSNorm
    @ModuleInfo var self_attn: Gemma4Attention
    @ModuleInfo var post_attention_layernorm: RMSNorm
    @ModuleInfo var pre_feedforward_layernorm: RMSNorm
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo var post_feedforward_layernorm: RMSNorm
    @ModuleInfo var layer_scalar: MLXArray   // [1]，权重文件里的逐层缩放

    init(layerIndex: Int) {
        let isGlobal = gemma4IsGlobal(layerIndex)
        self.input_layernorm = RMSNorm(dim: Gemma4Config.hiddenSize, eps: Gemma4Config.rmsNormEps)
        self.self_attn = Gemma4Attention(isGlobal: isGlobal)
        self.post_attention_layernorm = RMSNorm(dim: Gemma4Config.hiddenSize, eps: Gemma4Config.postNormEps)
        self.pre_feedforward_layernorm = RMSNorm(dim: Gemma4Config.hiddenSize, eps: Gemma4Config.rmsNormEps)
        self.mlp = Gemma4MLP()
        self.post_feedforward_layernorm = RMSNorm(dim: Gemma4Config.hiddenSize, eps: Gemma4Config.postNormEps)
        self.layer_scalar = MLXArray.ones([1])
        super.init()
    }

    /// 前向：[1,T,3840] → [1,T,3840]
    func callAsFunction(
        _ h: MLXArray, mask: MLXArray,
        globalFreqs: MLXArray?, slidingFreqs: MLXArray?
    ) -> MLXArray {
        // 注意力残差
        let normed = input_layernorm(h)
        let attnOut = self_attn(normed, mask: mask, globalFreqs: globalFreqs, slidingFreqs: slidingFreqs)
        var out = h + post_attention_layernorm(attnOut)

        // MLP 残差（GeGLU 在 Gemma4MLP 内部）
        let ffNormed = pre_feedforward_layernorm(out)
        let ffOut = mlp(ffNormed)
        out = out + post_feedforward_layernorm(ffOut)

        // layer_scalar
        return out * layer_scalar
    }
}

// MARK: - Gemma4 文本编码器（49 态捕获）

final class Gemma4TextEncoder: Module {
    @ModuleInfo var embed_tokens: QuantizedEmbedding
    @ModuleInfo var norm: RMSNorm   // 最终 norm（49 态捕获不过它，仅吸收权重键）
    let layers: [Gemma4Layer]

    override init() {
        self.embed_tokens = QuantizedEmbedding(
            embeddingCount: Gemma4Config.vocabSize, dimensions: Gemma4Config.hiddenSize,
            groupSize: Gemma4Config.groupSize, bits: Gemma4Config.bits)
        self.norm = RMSNorm(dim: Gemma4Config.hiddenSize, eps: Gemma4Config.rmsNormEps)
        self.layers = (0..<Gemma4Config.numLayers).map { Gemma4Layer(layerIndex: $0) }
        super.init()
    }

    /// causal + left-padding 的 additive mask：[1,1,T,T] bf16。
    static func causalPadMask(ids: MLXArray, padTokenID: Int32) -> MLXArray {
        let T = ids.shape.last ?? ids.shape[0]
        // 位置 i 可见 j<=i；pad token 列被屏蔽（j 是 pad → -inf）
        var data = [Float](repeating: 0, count: T * T)
        let idsArr = ids.asArray(Int32.self)
        for i in 0..<T {
            for j in 0..<T {
                var v: Float = 0
                if j > i { v = -1e9 }
                if idsArr[j] == padTokenID { v = -1e9 }
                data[i * T + j] = v
            }
        }
        let arr = MLXArray(data, [1, 1, T, T])
        return arr.asType(.bfloat16)
    }

    /// 前向：ids [1,T] int32 → 49 个 [1,T,3840] 残差态。
    /// T 应等于 256（LTX 固定 prompt 长度，不足左侧 pad）。
    func encodeStates(_ ids: MLXArray) -> [MLXArray] {
        let T = ids.shape.last ?? ids.shape[0]
        let mask = Self.causalPadMask(ids: ids, padTokenID: Gemma4Config.padTokenID)

        // embedding × sqrt(hidden_size)（scale_embeddings=true）
        var h = embed_tokens(ids) * sqrt(Float(Gemma4Config.hiddenSize))
        eval(h, mask)

        var states = [MLXArray]()
        states.append(h)
        let gFreqs = gemma4GlobalRopeFreqs()
        for layer in layers {
            h = layer(h, mask: mask, globalFreqs: gFreqs, slidingFreqs: nil)
            eval(h)
            states.append(h)
        }
        return states
    }

    // MARK: - 自回归生成（Gemma 对话）

    /// 前向：ids [1,T] → final norm 后的 hidden state [1,T,3840]（生成用，避免 49 态拷贝）。
    func forwardLast(_ ids: MLXArray) -> MLXArray {
        let mask = Self.causalPadMask(ids: ids, padTokenID: Gemma4Config.padTokenID)
        var h = embed_tokens(ids) * sqrt(Float(Gemma4Config.hiddenSize))
        eval(h, mask)
        let gFreqs = gemma4GlobalRopeFreqs()
        for layer in layers {
            h = layer(h, mask: mask, globalFreqs: gFreqs, slidingFreqs: nil)
            eval(h)
        }
        return norm(h)
    }

    /// 自回归生成：lm_head 为 tied embedding，复用 embed_tokens 量化参数做量化 matmul。
    /// 每步取最后位置 logits，temperature + top-k + top-p 采样下一个 token。
    /// - Parameters:
    ///   - repeatPenalty: 重复惩罚（对齐 mlx-serve applyRepeatPenalty），已生成 token 的 logit>0 除以 penalty、logit<0 乘 penalty；1.0 关闭
    ///   - presencePenalty: 存在惩罚，已生成 token 的 logit 直接减去该值；0.0 关闭
    ///   - suppressTokenIDs: 始终禁止采样的 token（如多模态结束标记），采样前 logit 置 -inf
    /// - Returns: 完整 token 序列（prompt + 生成部分），遇任一 eosTokenIDs 提前停止。
    func generate(
        _ promptIds: [Int32], maxNewTokens: Int,
        temperature: Float = 1.0, topK: Int = 64, topP: Float = 0.95,
        eosTokenIDs: [Int32] = [1, 106, 50],
        suppressTokenIDs: [Int32] = [],
        repeatPenalty: Float = 1.15, presencePenalty: Float = 0.0
    ) -> [Int32] {
        var ids = promptIds
        var generated = [Int32]()
        for step in 0..<maxNewTokens {
            let arr = MLXArray(ids, [1, ids.count]).asType(.int32)
            let h = forwardLast(arr)                                              // [1,T,3840]
            let last = ids.count - 1
            let lastH = h[0, last, 0..<Gemma4Config.hiddenSize].expandedDimensions(axis: 0)  // [1,3840]
            var logits = quantizedMatmul(
                lastH, embed_tokens.weight,
                scales: embed_tokens.scales,
                biases: embed_tokens.biases,
                transpose: true,
                groupSize: Gemma4Config.groupSize,
                bits: Gemma4Config.bits)                                          // [1,vocab]
            logits = logits[0]                                                    // [vocab]
            // final_logit_softcapping（官方 gemma4 必需）：logits 裁剪到 [-30,30]，否则 softmax 塌缩成 one-hot
            logits = tanh(logits / 30.0) * 30.0

            // 抑制 token：多模态结束标记等永不采样
            if !suppressTokenIDs.isEmpty {
                var maskData = [Bool](repeating: true, count: logits.size)
                for sid in suppressTokenIDs where sid >= 0 && Int(sid) < logits.size {
                    maskData[Int(sid)] = false
                }
                logits = MLX.where(MLXArray(maskData), logits, MLXArray(-1e9, dtype: logits.dtype))
            }
            if temperature > 0, temperature != 1.0 {
                logits = logits / temperature
            }
            // 重复/存在惩罚：只对已生成的 token 生效（不含 prompt），对齐 mlx-serve applyRepeatPenalty
            if (repeatPenalty != 1.0 || presencePenalty != 0.0), !generated.isEmpty {
                logits = Self.applyPenalties(
                    logits: logits,
                    generatedIds: generated,
                    vocabSize: Gemma4Config.vocabSize,
                    repeatPenalty: repeatPenalty,
                    presencePenalty: presencePenalty)
            }
            if topK > 0, logits.size > topK {
                let sortedVals = sorted(logits)                                   // 升序
                let threshold = sortedVals[sortedVals.size - topK]                // 第 topK 大
                logits = MLX.where(logits .>= threshold, logits, MLXArray(-1e9, dtype: logits.dtype))
            }
            // top-p（nucleus）过滤：保留累计概率达 topP 的最小 token 集（至少 1 个）
            if topP > 0, topP < 1.0 {
                let probs = softmax(logits, axis: -1)                             // [vocab]
                let idx = argSort(probs, axis: -1)                                // 升序索引
                let sortedProbs = takeAlong(probs, idx, axis: -1)                 // 升序概率
                // 反向累计：revCum[i] = sum(sortedProbs[i..])，从高概率端往前累计
                let revCum = sortedProbs.cumsum(axis: -1, reverse: true)
                // 保留高概率尾部：升序位置 i 保留 ⟺ i >= j，其中 j = 最后一个 revCum >= topP 的位置
                // revCum 单调递减，revCum >= topP 构成前缀 [0..j]，j = count - 1
                let count = (revCum .>= MLXArray(topP)).sum().item(Int32.self)
                let j = max(count - 1, 0)
                var keepData = [Float](repeating: 0, count: logits.size)
                if j < logits.size {
                    for i in Int(j)..<logits.size { keepData[i] = 1.0 }
                }
                let keepDesc = MLXArray(keepData)                                 // 升序位置保留标志
                let mask = putAlong(MLXArray.zeros([logits.size]), idx, values: keepDesc, axis: 0)  // 映射回原 vocab
                logits = MLX.where(mask .> MLXArray(0.5), logits, MLXArray(-1e9, dtype: logits.dtype))
            }
            // ---- DIAG: 前 5 步打印 logits top-5 与概率特征（诊断用，可删） ----
            if step < 5 {
                let probs = softmax(logits, axis: -1)
                let topKIdx = argSort(probs, axis: -1)
                let n = min(5, logits.size)
                var topIds: [Int32] = []
                var topPs: [Float] = []
                for i in 0..<n {
                    let idx = Int(topKIdx[logits.size - 1 - i].item(Int32.self))
                    topIds.append(Int32(idx))
                    topPs.append(probs[idx].item(Float.self))
                }
                pipelineLog("[DIAG] step=\(step) logitsMax=\(logits.max().item(Float.self)) logitsMin=\(logits.min().item(Float.self)) top1=\(topIds[0])(p=\(topPs[0])) top2=\(topIds[1])(p=\(topPs[1])) top3=\(topIds[2])(p=\(topPs[2])) top4=\(topIds[3])(p=\(topPs[3])) top5=\(topIds[4])(p=\(topPs[4]))")
            }
            // ---- END DIAG ----
            let next = MLXRandom.categorical(logits, axis: -1).item(Int32.self)
            ids.append(next)
            generated.append(next)
            if eosTokenIDs.contains(next) { break }
            // 每 16 步清理一次 MLX 内存缓存，防止生成期间缓存无限累积撑爆内存
            if step % 16 == 15 {
                MLX.Memory.clearCache()
            }
        }
        return ids
    }

    /// 重复/存在惩罚（对齐 mlx-serve `applyRepeatPenalty`）：
    /// 对已生成的 unique token：logit>0 除以 repeatPenalty、logit<0 乘 repeatPenalty；再整体减去 presencePenalty。
    /// 只软性压低，不做任何强制截断。
    private static func applyPenalties(
        logits: MLXArray, generatedIds: [Int32], vocabSize: Int,
        repeatPenalty: Float, presencePenalty: Float
    ) -> MLXArray {
        // 收集 unique token（忽略越界 id）
        let seen = Set(generatedIds.filter { $0 >= 0 && Int($0) < vocabSize })
        guard !seen.isEmpty else { return logits }

        // 构建 bool mask [vocab]，true = 已生成 token
        var maskData = [Bool](repeating: false, count: vocabSize)
        for id in seen { maskData[Int(id)] = true }
        let mask = MLXArray(maskData)   // [vocab] bool

        var current = logits
        if repeatPenalty != 1.0 {
            let rp = MLXArray(repeatPenalty)
            let invRp = MLXArray(1.0 / repeatPenalty)
            let zero = MLXArray(0.0)
            let positiveMask = current .> zero                    // [vocab] bool
            let penPos = current * invRp                          // logit>0 → 压低
            let penNeg = current * rp                             // logit<0 → 拉近 0
            let signSelected = MLX.where(positiveMask, penPos, penNeg)
            current = MLX.where(mask, signSelected, current)      // 只对已生成 token 生效
        }
        if presencePenalty != 0.0 {
            let pp = MLXArray(presencePenalty)
            let subtract = mask.asType(.float32) * pp
            current = current - subtract
        }
        return current
    }
}
