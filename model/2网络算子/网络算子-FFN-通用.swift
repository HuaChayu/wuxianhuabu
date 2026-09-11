//
//  网络算子-FFN-通用.swift
//  无限画布 — 通用 FFN 网络层算子（网络层结构，无模型专属权重键依赖）
//
//  ============================================================
//  作用：通用前馈网络层算子：
//    · swigluMLP：SwiGLU MLP（up/gate/down 投影 + SiLU 门控），按 token 分块执行
//      对齐官方 _SWIGLU_TILE_SIZE=16384，消除 3×hidden 宽度整段中间张量；
//      带单块编译图缓存（compile 图内禁用 shape 依赖 op，动态 tile 切片留在图外）。
//  ============================================================

import Foundation
import MLX

// MARK: - SwiGLU MLP

/// SwiGLU MLP 单块编译图：8 层同构（权重作输入参数、shape 固定），编译一次全层复用。
/// 注意：compile 图内禁用 shape 依赖 op（slice/split 在 tracer 编译期无法推断输出 shape 会崩），
/// 故 up/gate 用两次独立 matmul 而非合并 [C,2*mid] 后切列；代价是多一次 matmul kernel 启动，
/// 换取整块图编译。动态 tile 切片留在图外（Swift 层）。
/// shapeless=true：单块 token 数变化（最后一块不足 tileSize）不触发重编译。
/// 实现与 DDVNA.compiledSwigluBlock 原实现逐字一致（silu 引用改为通用 siluActivation）。
fileprivate let compiledSwigluBlock: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { args in
    let xb = args[0], upWt = args[1], upBv = args[2], gateWt = args[3], gateBv = args[4], downWt = args[5]
    let up = xb.matmul(upWt) + upBv                                   // (ts,mid)
    let gate = xb.matmul(gateWt) + gateBv                             // (ts,mid)
    let down = up * siluActivation(gate)                              // SwiGLU：up * silu(gate)
    return [down.matmul(downWt)]
}

/// SwiGLU MLP 分块执行（对齐官方 _SWIGLU_TILE_SIZE=16384）：按 token 分块做 up/gate/down，
/// 消除 3×hidden 宽度的整段中间张量（up/gate/down 各一份），峰值降为 tile 级。
/// x 为任意形状 (…, C)，返回同形状；bias 可空（权重包无 bias 时传 nil 自动补 0）。
/// 实现与 DDVNA.swigluMLP 原实现逐字一致。
/// 使用方：LTX-2.5（扩散视频解码器 DDVNA.naBlock / DDVDecoder）。
func swigluMLP(
    _ x: MLXArray,
    upW: MLXArray, upB: MLXArray?,
    gateW: MLXArray, gateB: MLXArray?,
    downW: MLXArray
) -> MLXArray {
    let shape = x.shape
    let C = shape.last!
    let n = shape.dropLast().reduce(1, *)   // token 数 = 不含 C 维的乘积
    let flat = x.reshaped([n, C])
    let mid = upW.shape[0]
    // 权重转置与 bias 补零外提（compile 图外，仅算一次）
    let upWt = upW.transposed(1, 0)                                    // [C, mid]
    let upBv = upB ?? MLXArray.zeros([mid], dtype: x.dtype)
    let gateWt = gateW.transposed(1, 0)                                // [C, mid]
    let gateBv = gateB ?? MLXArray.zeros([mid], dtype: x.dtype)
    let downWt = downW.transposed(1, 0)                                // [mid, C]
    // 分块 token 数：16384→32768 减半启动次数（stage5 每层 185 万 token：113→57 块），
    // 中间张量峰值仅 +128MB 且一次只驻留一块。
    let tileSize = 16384
    var outs: [MLXArray] = []
    var start = 0
    while start < n {
        let end = Swift.min(start + tileSize, n)
        let xb = flat[start..<end]
        outs.append(compiledSwigluBlock([xb, upWt, upBv, gateWt, gateBv, downWt])[0])
        start = end
    }
    return concatenated(outs, axis: 0).reshaped(shape)
}
