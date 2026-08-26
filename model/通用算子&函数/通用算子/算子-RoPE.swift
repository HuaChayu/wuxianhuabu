//
//  模型通用算子-RoPE.swift
//  无限画布 — 通用 RoPE 频率/应用算子（纯数学，可跨模型复用）
//
//  ============================================================
//  作用：通用 RoPE 频率/应用算子（纯数学，可跨模型复用）：
//    · RopeCS：一组 cos/sin 频率表（[1, numHeads, N, headDim/2]）
//    · ditRope：3D 位置 RoPE 频率生成（视频/音频通用，maxPos 参数化）—— 使用方：LTX-2.5（LTXDiT）
//    · connectorRopeFreqs：连接器 1D RoPE 频率（maxPos 4096）—— 使用方：LTX-2.5（Connector）
//    · applyRopeSplit：旋转一半维度（split-half rope）—— 使用方：LTX-2.5（Connector、LTXDiT）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - RoPE 频率表

/// 一组 cos/sin 频率表，形状 [1, numHeads, N, headDim/2]（已 transpose + contiguous）。
struct RopeCS {
    let cos: MLXArray
    let sin: MLXArray
}

/// 通用 3D 位置 RoPE：pos 为 N×A 位置（A=轴数），逐轴按 maxPos 归一化，
/// 生成 [1, numHeads, N, headDim/2] 的 cos/sin 表（SPLIT 风格）。
func ditRope(pos: [Float], N: Int, A: Int, numHeads: Int, headDim: Int, maxPos: [Float]) -> RopeCS {
    let theta: Float = 10000.0
    let innerDim = numHeads * headDim
    let numFreqs = innerDim / (2 * A)          // 整数除法
    let expected = innerDim / 2
    let pad = expected - numFreqs * A
    let hdHalf = headDim / 2

    var fi = [Float](repeating: 0, count: numFreqs)
    let denom = Float(numFreqs - 1)
    for j in 0..<numFreqs {
        fi[j] = pow(theta, Float(j) / denom) * (Float.pi / 2.0)
    }

    var cosBuf = [Float](repeating: 0, count: N * expected)
    var sinBuf = [Float](repeating: 0, count: N * expected)
    for n in 0..<N {
        let base = n * expected
        for c in 0..<pad {
            cosBuf[base + c] = 1.0
            sinBuf[base + c] = 0.0
        }
        for j in 0..<numFreqs {
            for i in 0..<A {
                let frac = pos[n * A + i] / maxPos[i]
                let sc = frac * 2.0 - 1.0
                let ang = fi[j] * sc
                let idx = base + pad + j * A + i
                cosBuf[idx] = cos(ang)
                sinBuf[idx] = sin(ang)
            }
        }
    }
    // [1, N, numHeads, hdHalf] → transpose(0,2,1,3) → [1, numHeads, N, hdHalf]
    let cos = floatArray(cosBuf).reshaped([1, N, numHeads, hdHalf]).transposed(0, 2, 1, 3)
    let sin = floatArray(sinBuf).reshaped([1, N, numHeads, hdHalf]).transposed(0, 2, 1, 3)
    eval(cos, sin)  // 物化为 contiguous，避免惰性 strided 视图
    return RopeCS(cos: cos, sin: sin)
}

/// 连接器 1D RoPE 频率：位置 t 用 maxPos=4096 归一化（split 风格）。
func connectorRopeFreqs(T: Int, dim: Int, headDim: Int) -> (cos: MLXArray, sin: MLXArray) {
    let theta: Float = 10000.0
    let maxPos: Float = 4096.0
    let numFreqs = dim / 2
    let hdHalf = headDim / 2
    let heads = dim / headDim
    let denom = Float(numFreqs - 1)

    // fi[j] = theta^(j/(numFreqs-1)) * π/2
    var fi = [Float](repeating: 0, count: numFreqs)
    for j in 0..<numFreqs {
        let e = Float(j) / denom
        fi[j] = pow(theta, e) * (Float.pi / 2.0)
    }
    // freqs[t,j] = fi[j] * (t/maxPos*2 - 1)
    var freqs = [Float](repeating: 0, count: T * numFreqs)
    for t in 0..<T {
        let frac = Float(t) / maxPos
        let sc = frac * 2.0 - 1.0
        for j in 0..<numFreqs {
            freqs[t * numFreqs + j] = fi[j] * sc
        }
    }
    let freqArr = MLXArray(freqs, [1, T, numFreqs])
    let cosRaw = cos(freqArr)
    let sinRaw = sin(freqArr)
    let cosR = cosRaw.reshaped([1, T, heads, hdHalf])
    let sinR = sinRaw.reshaped([1, T, heads, hdHalf])
    return (cosR.transposed(0, 2, 1, 3), sinR.transposed(0, 2, 1, 3))
}

/// 旋转一半维度（split-half rope）：x [..., headDim] 对半切，rot 应用 cos/sin。
func applyRopeSplit(
    _ x: MLXArray, cosF: MLXArray, sinF: MLXArray
) -> MLXArray {
    let hd = x.shape[3]
    let half = hd / 2
    let x1 = x[0..., 0..., 0..., 0..<half]
    let x2 = x[0..., 0..., 0..., half...]
    let lo = x1 * cosF - x2 * sinF
    let hi = x1 * sinF + x2 * cosF
    return concatenated([lo, hi], axis: 3)
}
