//
//  Connector.swift — LTX-2.5 文本连接器（Embeddings1DConnector + projection）
//
//  移植自 mlx-serve 的 ltx_video.zig：
//  - connectorProject：49 态 RMSNorm 聚合 → [1,T,188160] → video/audio 投影（scale=sqrt(dim/3840)）
//  - connectorTransform：128 学习寄存器替换左 pad → 8 个 gated-attention block（split-RoPE、2*sigmoid 门控、GELU FF、无参 RMSNorm）
//  - 权重键前缀 connector.（video/audio_embeddings_connector / text_embedding_projection）
//
//  Created by 花茶鱼i on 2026/8/19.
//

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
@preconcurrency import MLXNN
// MARK: - 连接器单 block（gated attention + GELU FF）

final class ConnectorBlock: Module {
    @ModuleInfo var attn1: ConnectorAttention
    @ModuleInfo var ff: ConnectorFF

    init(dim: Int, headDim: Int) {
        self.attn1 = ConnectorAttention(dim: dim, headDim: headDim)
        self.ff = ConnectorFF(dim: dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cosF: MLXArray, sinF: MLXArray) -> MLXArray {
        let dim = x.shape[2]
        // 注意力
        let normed = rmsNormNoAffine(x, dim: dim, eps: 1e-6)
        let attnOut = attn1(normed, cosF: cosF, sinF: sinF)
        var out = x + attnOut
        // FF（GELU tanh）
        let normed2 = rmsNormNoAffine(out, dim: dim, eps: 1e-6)
        out = out + ff(normed2)
        return out
    }
}

final class ConnectorAttention: Module {
    @ModuleInfo var to_q: Linear
    @ModuleInfo var to_k: Linear
    @ModuleInfo var to_v: Linear
    @ModuleInfo var to_out0: Linear   // 权重键 attn1.to_out.0 → remap
    @ModuleInfo var to_gate_logits: Linear
    @ModuleInfo var q_norm: RMSNorm
    @ModuleInfo var k_norm: RMSNorm

    let headDim: Int
    let heads: Int

    init(dim: Int, headDim: Int) {
        self.headDim = headDim
        self.heads = dim / headDim
        self.to_q = Linear(dim, dim, bias: true)
        self.to_k = Linear(dim, dim, bias: true)
        self.to_v = Linear(dim, dim, bias: true)
        self.to_out0 = Linear(dim, dim, bias: true)
        self.to_gate_logits = Linear(dim, self.heads, bias: true)
        // q_norm/k_norm 权重形状 [dim]（作用在完整 inner dim），eps=1e-5
        self.q_norm = RMSNorm(dim: dim, eps: 1e-5)
        self.k_norm = RMSNorm(dim: dim, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cosF: MLXArray, sinF: MLXArray) -> MLXArray {
        let T = x.shape[1]
        let dim = x.shape[2]
        let hd = headDim
        let nh = heads
        let scale = 1.0 / sqrt(Float(hd))

        var q = q_norm(to_q(x))
        var k = k_norm(to_k(x))
        let v = to_v(x)

        let qh = q.reshaped([1, T, nh, hd]).transposed(0, 2, 1, 3)
        let kh = k.reshaped([1, T, nh, hd]).transposed(0, 2, 1, 3)
        let vh = v.reshaped([1, T, nh, hd]).transposed(0, 2, 1, 3)

        let qr = applyRopeSplit(qh, cosF: cosF, sinF: sinF)
        let kr = applyRopeSplit(kh, cosF: cosF, sinF: sinF)

        var attn = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: vh, scale: scale, mask: nil)

        // per-head gate：2*sigmoid(to_gate_logits(x)) [1,T,nh] → [1,nh,T,1]
        let gate = 2.0 * sigmoid(to_gate_logits(x))
        let gateT = gate.transposed(0, 2, 1).reshaped([1, nh, T, 1])
        attn = attn * gateT

        let ao = attn.transposed(0, 2, 1, 3).reshaped([1, T, dim])
        return to_out0(ao)
    }
}

final class ConnectorFF: Module {
    @ModuleInfo var net: ConnectorFFNet
    init(dim: Int) {
        self.net = ConnectorFFNet(dim: dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        net(x)
    }
}

/// ff.net：Sequential(0.proj → gelu → 2)，键 ff.net.0.proj / ff.net.2 → remap 为 proj0/proj2
final class ConnectorFFNet: Module {
    @ModuleInfo var proj0: Linear   // ff.net.0.proj
    @ModuleInfo var proj2: Linear   // ff.net.2
    let hidden: Int
    init(dim: Int) {
        self.hidden = dim * 4
        self.proj0 = Linear(dim, hidden, bias: true)
        self.proj2 = Linear(hidden, dim, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj2(geluApprox(proj0(x)))
    }
}

// MARK: - 连接器（projection + transformer）

final class LTXConnector: Module {
    @ModuleInfo var video_embeddings_connector: ConnectorBody
    @ModuleInfo var audio_embeddings_connector: ConnectorBody
    @ModuleInfo var text_embedding_projection: ConnectorProjection

    override init() {
        self.video_embeddings_connector = ConnectorBody(dim: 4096, headDim: 128)
        self.audio_embeddings_connector = ConnectorBody(dim: 2048, headDim: 64)
        self.text_embedding_projection = ConnectorProjection()
        super.init()
    }
}

/// video/audio 两个 transformer 主干（128 寄存器 + 8 blocks）。
final class ConnectorBody: Module {
    @ModuleInfo var learnable_registers: MLXArray
    @ModuleInfo var transformer_1d_blocks: [ConnectorBlock]
    let dim: Int
    let headDim: Int
    let numRegisters = 128

    init(dim: Int, headDim: Int) {
        self.dim = dim
        self.headDim = headDim
        self.learnable_registers = MLXArray.zeros([128, dim])
        self.transformer_1d_blocks = (0..<8).map { _ in ConnectorBlock(dim: dim, headDim: headDim) }
        super.init()
    }

    /// 前向：input [1,T,dim]，nValid=有效（非 pad）token 数 → [1,T,dim]。
    /// 对齐 mlx-serve ltx_video.zig 的 connectorTransform：输入统一 bf16，
    /// 寄存器替换左 pad（valid 在前、registers 填充剩余）→ 8 个 gated-attention block。
    func callAsFunction(_ input: MLXArray, nValid: Int) -> MLXArray {
        let T = input.shape[1]
        let dim = self.dim
        let inputBf = input.asType(PrecisionPolicy.defaultMainDType)

        // register 替换左 pad：valid（右 nValid 个）在前，registers 填充剩余
        let reg3 = learnable_registers.asType(inputBf.dtype).reshaped([1, numRegisters, dim])
        var tiles = [MLXArray]()
        let numTiles = T / numRegisters
        for _ in 0..<numTiles { tiles.append(reg3) }
        let tiled = concatenated(tiles, axis: 1)   // [1,T,dim]
        let valid = inputBf[0..., (T - nValid)...]
        let regPart = tiled[0..., nValid...]
        let x = concatenated([valid, regPart], axis: 1)  // [1,T,dim]

        let (cosF, sinF) = connectorRopeFreqsV2(T: T, dim: dim, headDim: headDim)
        let cosB = cosF.asType(inputBf.dtype)
        let sinB = sinF.asType(inputBf.dtype)
        var h = x
        for block in transformer_1d_blocks {
            h = block(h, cosF: cosB, sinF: sinB)
            eval(h)
        }
        return rmsNormNoAffine(h, dim: dim, eps: 1e-6)
    }
}

/// text_embedding_projection：video/audio aggregate embed（Linear，scale 已在 projOne 处理）。
final class ConnectorProjection: Module {
    @ModuleInfo var video_aggregate_embed: Linear
    @ModuleInfo var audio_aggregate_embed: Linear

    override init() {
        self.video_aggregate_embed = Linear(188160, 4096, bias: true)
        self.audio_aggregate_embed = Linear(188160, 2048, bias: true)
        super.init()
    }
}
// MARK: - 49 态 → 文本条件

/// connectorProject：states[49×[1,T,3840]] → (video [1,T,4096], audio [1,T,2048])
func connectorProject(states: [MLXArray], connector: LTXConnector) -> (video: MLXArray, audio: MLXArray) {
    // stack → [1,T,3840,49]
    var enc = stacked(states, axis: 3)
    // RMSNorm over axis 2（hidden）
    let meanSq = (enc * enc).mean(axis: 2, keepDims: true)
    enc = enc * rsqrt(meanSq + 1e-6)
    // [1,T,188160]
    let T = enc.shape[1]
    let stackedFlat = enc.reshaped([1, T, 188160])

    let proj = connector.text_embedding_projection
    let videoScale = sqrt(4096.0 / 3840.0)
    let audioScale = sqrt(2048.0 / 3840.0)
    let video = proj.video_aggregate_embed(stackedFlat * videoScale)
    let audio = proj.audio_aggregate_embed(stackedFlat * audioScale)
    return (video, audio)
}
