//
//  模型通用算子-卷积.swift
//  无限画布 — 通用卷积算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用 3D/2D/1D 卷积实现（纯数学，无模型结构依赖，可跨模型复用）：
//    · conv3dNDHWC：3D 卷积（时间轴 padding 模式参数化，覆盖 decoder 对称 replicate /
//      encoder causal 前向 replicate / upscaler 对称 zero 三语义）—— 使用方：LTX-2.5（VaeDecoder / SpatialUpscaler）
//    · conv2dNHWC / conv1dNLC：通用 2D/1D 卷积—— 使用方：LTX-2.5（AudioVaeDecoder）
//  布局约定：NDHWC（视频）/ NHWC / NLC，与 MLX 原生布局一致。
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 3D 卷积（NDHWC，时间轴 padding 模式参数化）

/// 3D conv（NDHWC）：把 3D conv（kD,kH,kW）等价转化为 2D conv——每个输出帧 t 的窗口
/// [t..t+kD] 的通道拼成 [H,W,kD*C]，全部 t 沿 batch 维堆叠后一次 conv2d。
/// MLX conv2d 内核远快于 conv3d；按 chunk 分块控制峰值内存。
///
/// 由原 decoderConv3dZig / encoderConv3dV2 / upscalerConv3dV2 合并而来，
/// 三实现函数体逐行一致，仅以下参数不同（合并后由参数承载）：
///   - timePad：时间轴 padding 方式（见 Conv3dTimePad）
///   - chunkBudget：分块内存预算（decoder 1<<28；encoder/upscaler 1<<30）
///   - evalChunks：是否对每个 chunk 的 conv2d 结果立即 eval()（避免大 chunk lazy 图堆积）
enum Conv3dTimePad {
    /// decoder 语义：时间轴首尾各 replicate (k-1)/2 帧（对齐 ltx_video.zig decoderConv3d）
    case replicateSymmetric
    /// encoder 语义：时间轴前向 replicate kernel-1 帧（causal，只复制首帧在前）
    case replicateCausal
    /// upscaler 语义：时间轴首尾各 zero pad (k-1)/2 帧（对齐 torch Conv3d padding=1）
    case zeroSymmetric
}

func conv3dNDHWC(
    _ x: MLXArray,
    weight: MLXArray,   // [C_out, kD, kH, kW, C_in]（MLX 布局）
    bias: MLXArray?,
    kernel: Int = 3,
    spatialPad: Int = 1,
    timePad: Conv3dTimePad = .replicateSymmetric,
    chunkBudget: Int = 1 << 28,
    evalChunks: Bool = false
) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
    let cout = weight.shape[0]
    let ps = (kernel - 1) / 2

    // 时间轴 padding
    var t = x
    switch timePad {
    case .replicateSymmetric:
        if ps > 0 {
            let first = x[0 ..< 1, axis: 1]
            let last = x[(d - 1) ..< d, axis: 1]
            let fp = MLX.concatenated([MLXArray](repeating: first, count: ps), axis: 1)
            let lp = MLX.concatenated([MLXArray](repeating: last, count: ps), axis: 1)
            t = MLX.concatenated([fp, t, lp], axis: 1)
        }
    case .replicateCausal:
        let frontPad = kernel - 1
        if frontPad > 0 {
            let first = x[0 ..< 1, axis: 1]
            let fp = MLX.concatenated([MLXArray](repeating: first, count: frontPad), axis: 1)
            t = MLX.concatenated([fp, t], axis: 1)
        }
    case .zeroSymmetric:
        if ps > 0 {
            let zf = MLXArray.zeros([b, ps, h, w, cin], dtype: x.dtype)
            let zb = MLXArray.zeros([b, ps, h, w, cin], dtype: x.dtype)
            t = MLX.concatenated([zf, t, zb], axis: 1)
        }
    }

    // 权重 [Cout, kD, kH, kW, Cin] → [Cout, kH, kW, kD*Cin]（kd 挪到通道维）
    let w2 = weight.transposed(axes: [0, 2, 3, 1, 4])
        .reshaped([cout, kernel, kernel, kernel * cin])

    // 输出空间尺寸
    let ho = h + 2 * spatialPad - kernel + 1
    let wo = w + 2 * spatialPad - kernel + 1

    // 分块：窗口数组 ≤ chunkBudget，避免大空间尺寸时峰值内存爆炸
    let windowBytesPerFrame = max(h * w * kernel * cin * t.dtype.size, 1)
    let chunkSize = max(1, min(d, chunkBudget / windowBytesPerFrame))

    var outputs: [MLXArray] = []
    outputs.reserveCapacity((d + chunkSize - 1) / chunkSize)
    var i = 0
    while i < d {
        let e = min(i + chunkSize, d)
        var wins: [MLXArray] = []
        wins.reserveCapacity(e - i)
        for j in i..<e {
            let win = t[0 ..< b, j ..< (j + kernel), 0 ..< h, 0 ..< w, 0 ..< cin]
                .transposed(axes: [0, 2, 3, 1, 4])          // [B, H, W, kD, C]
                .reshaped([b, h, w, kernel * cin])
            wins.append(win)
        }
        let stacked = MLX.concatenated(wins, axis: 1)        // [B, chunk, H, W, kD*C]
            .reshaped([(e - i) * b, h, w, kernel * cin])
        let out = MLX.conv2d(stacked, w2, stride: IntOrPair((1, 1)), padding: IntOrPair((spatialPad, spatialPad)), dilation: IntOrPair((1, 1)))
        if evalChunks {
            out.eval()  // 立即落地，避免大 chunk 下 lazy 图堆积
        }
        outputs.append(out.reshaped([b, e - i, ho, wo, cout]))
        i = e
    }
    let out5d = MLX.concatenated(outputs, axis: 1)           // [B, D, H', W', Cout]
    if let bias {
        return out5d + bias
    }
    return out5d
}

// MARK: - 2D 卷积（NHWC）

/// 2D conv（NHWC）。weight [O, kh, kw, I]。
func conv2dNHWC(_ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
                stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0)) -> MLXArray {
    var out = MLX.conv2d(
        x, weight,
        stride: IntOrPair((stride.0, stride.1)),
        padding: IntOrPair((padding.0, padding.1)),
        dilation: IntOrPair((1, 1)), groups: 1)
    if let bias { out = out + bias }
    return out
}

// MARK: - 1D 卷积（NLC）

/// 1D conv（NLC）。weight [O, k, I]。
func conv1dNLC(_ x: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
               stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1) -> MLXArray {
    var out = MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
    if let bias { out = out + bias }
    return out
}

// MARK: - Causal 2D 卷积

/// Causal 2D conv（HEIGHT=time 轴 causal，对齐 causalConv2d）。
/// 对 k=1（nin_shortcut）不加 pad；否则 time(axis1) lo=kh-1 hi=0、freq(axis2) lo=kw/2 hi=kw/2 零填充。
/// 注：签名沿用原模型侧 weights dict 约定（base 前缀取 .weight/.bias），未来可参数化为 weight/bias 直传。
/// 使用方：LTX-2.5（AudioVaeDecoder）。
func causalConv2d(_ weights: [String: MLXArray], base: String, _ x: MLXArray) -> MLXArray {
    guard let w = weights[base + ".weight"] else {
        fatalError("缺键：\(base).weight")
    }
    let b = weights[base + ".bias"]
    let kh = w.shape[1]
    let kw = w.shape[2]
    if kh == 1 {
        return conv2dNHWC(x, weight: w, bias: b)
    }
    let padded = MLX.padded(x, widths: [
        IntOrPair((0, 0)),
        IntOrPair((kh - 1, 0)),
        IntOrPair((kw / 2, kw / 2)),
        IntOrPair((0, 0)),
    ], mode: .constant)
    return conv2dNHWC(padded.contiguous(), weight: w, bias: b)
}
