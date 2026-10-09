//
//  模型通用算子-激活.swift
//  无限画布 — 通用激活函数（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用激活函数（纯数学，无模型结构依赖，可跨模型复用）：
//    · geluApprox：GELU tanh 近似（gelu-approximate）—— 使用方：LTX-2.5（Gemma4Text、Connector、LTXDiT）
//    · snakeBeta：SnakeBeta（BigVGAN 反走样激活，alpha/beta log 空间参数）—— 使用方：LTX-2.5（AudioVaeDecoder）
//    · antiAliasSnakeBeta：反走样 SnakeBeta（up×2 → snake → down÷2）—— 使用方：LTX-2.5（AudioVaeDecoder）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - GELU（tanh 近似）

/// GELU tanh 近似（activation_fn: gelu-approximate）。
func geluApprox(_ x: MLXArray) -> MLXArray {
    0.5 * x * (1 + tanh(sqrt(2.0 / .pi) * (x + 0.044715 * x * x * x)))
}

// MARK: - SnakeBeta 激活（BigVGAN）

/// SnakeBeta 激活。x 为 NLC；alpha/beta 为 log 空间参数 [C]。
func snakeBeta(_ x: MLXArray, alpha: MLXArray, beta: MLXArray) -> MLXArray {
    let c = alpha.shape[0]
    let ar = alpha.reshaped([1, 1, c])
    let br = beta.reshaped([1, 1, c])
    let ea = ar.exp()
    let eb = br.exp()
    let xa = x * ea
    let sn2 = xa.sin().square()
    let inv = 1.0 / (eb + 1e-9)
    return x + sn2 * inv
}

/// 把存储的 [1,1,k]（或 [1,k,1]）反走样核转成 depthwise MLX 布局 [C, k, 1]。
func depthwiseFilter(_ stored: MLXArray, channels c: Int) -> MLXArray {
    let sh = stored.shape
    let k = sh[1] * sh[2]
    let flat = stored.reshaped([1, k, 1])
    return MLX.broadcast(flat, to: [c, k, 1]).contiguous()
}

/// 反走样激活（Activation1d，ratio 2，k=12）：up×2（edge pad 5 + depthwise convT + ×2 +
/// crop 15/15）→ snakeBeta → down÷2（edge pad 5/6 + depthwise strided conv）。长度保持。
func antiAliasSnakeBeta(_ x: MLXArray, alpha: MLXArray, beta: MLXArray,
                        upFilt: MLXArray, downFilt: MLXArray) -> MLXArray {
    let c = x.shape[2]

    // ── upsample ×2 ──
    let upW = depthwiseFilter(upFilt, channels: c)
    let paddedUp = MLX.padded(x, widths: [
        IntOrPair((0, 0)), IntOrPair((5, 5)), IntOrPair((0, 0)),
    ], mode: .edge)
    var ct = MLX.convTransposed1d(paddedUp.contiguous(), upW, stride: 2, padding: 0, outputPadding: 0, groups: c)
    ct = ct * 2.0
    let lup = ct.shape[1]
    let cropped = ct[15 ..< (lup - 15), axis: 1].contiguous()

    // ── activation ──
    let act = snakeBeta(cropped, alpha: alpha, beta: beta)

    // ── downsample ÷2 ──
    let downW = depthwiseFilter(downFilt, channels: c)
    let paddedDown = MLX.padded(act, widths: [
        IntOrPair((0, 0)), IntOrPair((5, 6)), IntOrPair((0, 0)),
    ], mode: .edge)
    return MLX.conv1d(paddedDown.contiguous(), downW, stride: 2, padding: 0, dilation: 1, groups: c)
}

// MARK: - SiLU（sigmoid 门控线性单元）

/// SiLU 激活：x * sigmoid(x)。实现与 DDVNA.silu 原实现逐字一致
///（one / (one + exp(-x)) 形式，避免依赖不确定的 helper）。
func siluActivation(_ x: MLXArray) -> MLXArray {
    let one = MLXArray.ones(like: x)
    return x * (one / (one + (-x).exp()))
}
