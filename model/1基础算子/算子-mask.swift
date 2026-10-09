//
//  模型通用算子-mask.swift
//  无限画布 — 通用注意力掩码/广播算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用 mask 构造与广播（LTX-2.5 视频链路 + HiDream-O1-Image 图像 + Gemma4 文本）：
//    · maskAsTensor：condMask [Nv] 广播为与 latent 同秩 mask 张量 —— 使用方：LTX-2.5（Sampler）
//    · causalPadMask：causal + left-padding additive mask —— 使用方：LTX-2.5（Gemma4TextEncoder）
//    · makeAttentionMask：视觉塔下三角 block 保持掩码 —— 使用方：HiDream-O1-Image（HiDreamVision）
//    · hidreamBuildAttentionMask：HiDream 2D causal + gen 行双向掩码 —— 使用方：HiDream-O1-Image（HiDreamPipeline）
//    · hidreamBuildTmsMask：tms token 注入 0/1 掩码广播 —— 使用方：HiDream-O1-Image（HiDreamPipeline）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 掩码广播

/// 把 [Nv] condMask 广播为与 latent 同秩的 mask 张量：
/// 5D [1,T,H,W,1]（常规网格）/ 3D [1,Nv,1]（token 级合并序列，IC-LoRA Stage2 低清直放）
func maskAsTensor(_ mask: [Float], like v: MLXArray) -> MLXArray {
    if v.ndim == 3 {
        return MLXArray(mask).reshaped([1, v.shape[1], 1]).asType(.bfloat16)
    }
    return MLXArray(mask).reshaped([1, v.shape[1], v.shape[2], v.shape[3], 1]).asType(.bfloat16)
}

// MARK: - Causal + Padding 掩码

/// causal + left-padding 的 additive mask：[1,1,T,T] bf16。
/// 位置 i 可见 j<=i；pad token 列被屏蔽（j 是 pad → -inf）。
func causalPadMask(ids: MLXArray, padTokenID: Int32) -> MLXArray {
    let T = ids.shape.last ?? ids.shape[0]
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

// MARK: - 视觉塔 block 保持掩码

/// 构造视觉塔下三角 block 保持掩码（非对角置 -1e9）。
/// 只依赖 sequenceLength + cuSeqlens，视觉塔各层共享同一张 mask；
/// 由采样前/视觉塔前向开头构造一次，Attention 内不再逐层重建。
func makeAttentionMask(sequenceLength: Int, cuSeqlens: MLXArray, dtype: MLX.DType) -> MLXArray {
    var mask = ones([1, sequenceLength, sequenceLength], dtype: dtype)
    mask = mask * MLXArray(-1e9, dtype: dtype)

    let seqlens = cuSeqlens.asArray(Int.self)
    for idx in 1 ..< seqlens.count {
        let start = seqlens[idx - 1]
        let end = seqlens[idx]
        mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: dtype)
    }
    return mask
}

// MARK: - HiDream 2D 注意力掩码

/// [c 类-规划] 命名带 hidream 前缀但本质通用，未来可参数化（掩码 scale、双向行集合来源、注入 token id）
/// 为通用 buildCausalGenMask / buildTokenInjectionMask 并保留 hidream 别名兼容；本次仅注释标注，不改名不改实现。
/// HiDream 2D causal 掩码：上三角（j > i）为 -1e4；gen 行（tokenTypesBin=true）全部置 0 → 双向可见。
func hidreamBuildAttentionMask(tokenTypesBin: [Bool]) -> MLXArray {
    let S = tokenTypesBin.count
    let dtypeMin: Float = -1e4

    let rows = MLXArray(Array(0 ..< S).map { Int32($0) }).expandedDimensions(axis: 1)  // [S,1]
    let cols = MLXArray(Array(0 ..< S).map { Int32($0) }).expandedDimensions(axis: 0)  // [1,S]
    let causal2d = MLX.where(cols .> rows, MLXArray(dtypeMin), MLXArray(0.0))          // [S,S]

    var mask = causal2d.expandedDimensions(axis: 0).expandedDimensions(axis: 0)        // [1,1,S,S]
    for i in 0 ..< S where tokenTypesBin[i] {
        mask[0, 0, i, 0...] = MLXArray(0.0)
    }
    return mask
}

/// hidreamBuildAttentionMaskFast：与 hidreamBuildAttentionMask 数学等价，
/// gen 行置 0 由逐行切片赋值改为一次广播 where（gen 行多、S 大时显著减少
/// 逐行 GPU 写与中间行数）。旧版 hidreamBuildAttentionMask 保留不动。
/// [c 类-规划] 同 hidreamBuildAttentionMask，本次仅注释标注不改名。
func hidreamBuildAttentionMaskFast(tokenTypesBin: [Bool]) -> MLXArray {
    let S = tokenTypesBin.count
    let dtypeMin: Float = -1e4

    let rows = MLXArray(Array(0 ..< S).map { Int32($0) }).expandedDimensions(axis: 1)  // [S,1]
    let cols = MLXArray(Array(0 ..< S).map { Int32($0) }).expandedDimensions(axis: 0)  // [1,S]
    let causal2d = MLX.where(cols .> rows, MLXArray(dtypeMin), MLXArray(0.0))          // [S,S]

    // gen 行整行置 0（含对角，双向可见），一次广播 + where 完成
    let genRow = MLXArray(tokenTypesBin.map { $0 ? Float(1.0) : Float(0.0) })          // [S]
    let gen2d = broadcast(genRow.expandedDimensions(axis: 1), to: [S, S])              // [S,S]
    let final2d = MLX.where(gen2d .> 0, MLXArray(0.0), causal2d)
    return final2d.expandedDimensions(axis: 0).expandedDimensions(axis: 0)             // [1,1,S,S]
}

/// tms 注入掩码常量：[1, txt_len, 4096] 0/1（tms token 位置为 1）。
/// 只依赖 inputIds，采样循环外构造一次，避免每步重复 map/broadcast。
/// [c 类-规划] 命名带 hidream 前缀但本质通用（tms token 注入 0/1 掩码），未来可参数化注入 token id 并改名；
/// 本次仅注释标注不改名不改实现。
func hidreamBuildTmsMask(inputIds: [Int32], tmsTokenID: Int32, shape: [Int]) -> MLXArray {
    let tmsMask = MLXArray(inputIds.map { $0 == tmsTokenID ? Float(1.0) : Float(0.0) })   // [txt_len]
    let tmsMask3d = tmsMask
        .expandedDimensions(axis: 0)          // [1, txt_len]
        .expandedDimensions(axis: -1)         // [1, txt_len, 1]
    return broadcast(tmsMask3d, to: shape)    // [1, txt_len, 4096]
}
