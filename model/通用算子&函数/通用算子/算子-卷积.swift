//
//  模型通用算子-卷积.swift
//  无限画布 — 通用卷积算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用 3D/2D/1D 卷积实现（纯数学，无模型结构依赖，可跨模型复用）：
//    · decoderConv3dZig：3D 解码卷积（时间对称 replicate pad）—— 使用方：LTX-2.5（VaeDecoder）
//    · encoderConv3d：3D 编码卷积（causal 时间 pad）—— 使用方：LTX-2.5（VaeDecoder）
//    · upscalerConv3d：3D 卷积（时间 zero pad）—— 使用方：LTX-2.5（SpatialUpscaler）
//    · conv2dNHWC / conv1dNLC：通用 2D/1D 卷积—— 使用方：LTX-2.5（AudioVaeDecoder）
//  布局约定：NDHWC（视频）/ NHWC / NLC，与 MLX 原生布局一致。
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 3D 解码卷积（时间对称 replicate pad + 空间 zero pad）

/// 对齐 ltx_video.zig decoderConv3d：时间轴首尾各 replicate (k-1)/2 帧，
/// 空间轴 zero pad spatialPad，然后 conv3d（stride 1、无额外 padding）。
///
/// 实现：把 3D conv（kD,kH,kW）等价转化为 2D conv——每个输出帧 t 的窗口
/// [t..t+kD] 的通道拼成 [H,W,kD*C]，全部 t 沿 batch 维堆叠后一次 conv2d。
/// MLX conv2d 内核远快于 conv3d；按 chunk 分块控制峰值内存（窗口数组 ≤256MB）。
func decoderConv3dZig(
    _ x: MLXArray,
    weight: MLXArray,   // [C_out, kD, kH, kW, C_in]（MLX 布局）
    bias: MLXArray?,
    kernel: Int = 3,
    spatialPad: Int = 1
) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
    let cout = weight.shape[0]
    let ps = (kernel - 1) / 2

    // 时间轴 replicate pad
    var t = x
    if ps > 0 {
        let first = x[0 ..< 1, axis: 1]
        let last = x[(d - 1) ..< d, axis: 1]
        let fp = MLX.concatenated([MLXArray](repeating: first, count: ps), axis: 1)
        let lp = MLX.concatenated([MLXArray](repeating: last, count: ps), axis: 1)
        t = MLX.concatenated([fp, t, lp], axis: 1)
    }

    // 权重 [Cout, kD, kH, kW, Cin] → [Cout, kH, kW, kD*Cin]（kd 挪到通道维）
    let w2 = weight.transposed(axes: [0, 2, 3, 1, 4])
        .reshaped([cout, kernel, kernel, kernel * cin])

    // 输出空间尺寸
    let ho = h + 2 * spatialPad - kernel + 1
    let wo = w + 2 * spatialPad - kernel + 1

    // 分块：窗口数组 ≤ ~256MB，避免大空间尺寸时峰值内存爆炸
    let windowBytesPerFrame = max(h * w * kernel * cin * t.dtype.size, 1)
    let maxChunkBytes = 1 << 28
    let chunkSize = max(1, min(d, maxChunkBytes / windowBytesPerFrame))

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
        out.eval()  // 立即落地，避免 conv2d im2col 中间量与 lazy 图堆积
        outputs.append(out.reshaped([b, e - i, ho, wo, cout]))
        i = e
    }
    let out5d = MLX.concatenated(outputs, axis: 1)           // [B, D, H', W', Cout]
    if let bias {
        return out5d + bias
    }
    return out5d
}

// MARK: - 3D 编码卷积（causal 时间 pad + 空间 zero pad）

/// 对齐 ltx_video.zig encoderConv3d(600)。权重布局 [C_out, kD, kH, kW, C_in]。
func encoderConv3d(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray?,
    kernel: Int = 3,
    spatialPad: Int = 1
) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
    let cout = weight.shape[0]
    let ps = (kernel - 1) / 2

    // 时间轴前向 replicate pad（causal：只复制首帧在前；对齐 zig 前向 pad k-1 帧）
    var t = x
    let frontPad = kernel - 1
    if frontPad > 0 {
        let first = x[0 ..< 1, axis: 1]
        let fp = MLX.concatenated([MLXArray](repeating: first, count: frontPad), axis: 1)
        t = MLX.concatenated([fp, t], axis: 1)
    }

    // 权重 [Cout, kD, kH, kW, Cin] → [Cout, kH, kW, kD*Cin]
    let w2 = weight.transposed(axes: [0, 2, 3, 1, 4])
        .reshaped([cout, kernel, kernel, kernel * cin])

    let ho = h + 2 * spatialPad - kernel + 1
    let wo = w + 2 * spatialPad - kernel + 1

    // 分块控制峰值内存（同解码器）
    let windowBytesPerFrame = max(h * w * kernel * cin * t.dtype.size, 1)
    let maxChunkBytes = 1 << 30
    let chunkSize = max(1, min(d, maxChunkBytes / windowBytesPerFrame))

    var outputs: [MLXArray] = []
    outputs.reserveCapacity((d + chunkSize - 1) / chunkSize)
    var i = 0
    while i < d {
        let e = min(i + chunkSize, d)
        var wins: [MLXArray] = []
        wins.reserveCapacity(e - i)
        for j in i..<e {
            let win = t[0 ..< b, j ..< (j + kernel), 0 ..< h, 0 ..< w, 0 ..< cin]
                .transposed(axes: [0, 2, 3, 1, 4])
                .reshaped([b, h, w, kernel * cin])
            wins.append(win)
        }
        let stacked = MLX.concatenated(wins, axis: 1)
            .reshaped([(e - i) * b, h, w, kernel * cin])
        let out = MLX.conv2d(stacked, w2, stride: IntOrPair((1, 1)), padding: IntOrPair((spatialPad, spatialPad)), dilation: IntOrPair((1, 1)))
        outputs.append(out.reshaped([b, e - i, ho, wo, cout]))
        i = e
    }
    let out5d = MLX.concatenated(outputs, axis: 1)
    if let bias {
        return out5d + bias
    }
    return out5d
}

// MARK: - 3D 卷积（时间轴 zero pad + 空间 zero pad，对齐 torch Conv3d padding=1）

/// 对齐 torch Conv3d(k=3, padding=1)：时间轴首尾各 zero pad (k-1)/2 帧，
/// 空间轴 zero pad spatialPad，然后 conv3d（stride 1）。
/// 与 decoderConv3dZig 唯一差异：时间轴是 zero pad 而非
/// replicate（普通 torch Conv3d 语义）。
func upscalerConv3d(
    _ x: MLXArray,
    weight: MLXArray,   // [C_out, kD, kH, kW, C_in]（MLX 布局）
    bias: MLXArray?,
    kernel: Int = 3,
    spatialPad: Int = 1
) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
    let cout = weight.shape[0]
    let ps = (kernel - 1) / 2

    // 时间轴 zero pad（torch Conv3d padding 语义）
    var t = x
    if ps > 0 {
        let zf = MLXArray.zeros([b, ps, h, w, cin], dtype: x.dtype)
        let zb = MLXArray.zeros([b, ps, h, w, cin], dtype: x.dtype)
        t = MLX.concatenated([zf, t, zb], axis: 1)
    }

    // 权重 [Cout, kD, kH, kW, Cin] → [Cout, kH, kW, kD*Cin]（kd 挪到通道维）
    let w2 = weight.transposed(axes: [0, 2, 3, 1, 4])
        .reshaped([cout, kernel, kernel, kernel * cin])

    let ho = h + 2 * spatialPad - kernel + 1
    let wo = w + 2 * spatialPad - kernel + 1

    // 分块控制峰值内存（同 VaeDecoder）
    let windowBytesPerFrame = max(h * w * kernel * cin * t.dtype.size, 1)
    let maxChunkBytes = 1 << 30
    let chunkSize = max(1, min(d, maxChunkBytes / windowBytesPerFrame))

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
