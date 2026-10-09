//
//  模型通用算子-归一化.swift
//  无限画布 — 通用归一化算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用归一化实现（纯数学，无模型结构依赖，可跨模型复用）：
//    · pixelNormFast：PixelNorm（last axis 通道，无 affine）—— 使用方：LTX-2.5（VaeDecoder）
//    · groupNormNDHWC：GroupNorm（NDHWC 布局，内部 f32 计算防 bf16 数值灾难）—— 使用方：LTX-2.5（SpatialUpscaler）
//    · rmsNormNoAffine：RMSNorm（无参，x * rsqrt(mean(x²)+eps)）—— 使用方：LTX-2.5（Connector）
//    · audioPixelNorm：Audio 侧 PixelNorm（eps 1e-6），已并入 pixelNormFast（传 eps: 1e-6）—— 使用方：LTX-2.5（AudioVaeDecoder）
//  ============================================================

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - PixelNorm（last axis 通道归一化，无 affine）

/// PixelNorm over channels（last axis），eps 默认 1e-8（对齐 VaeDecoder 原实现）。
/// 原 audioPixelNorm（eps 1e-6）与之函数体逐行一致、仅默认 eps 不同，
/// 已合并去重：音频侧调用方统一改用本函数并显式传 eps: 1e-6。
func pixelNormFast(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    let sq = x * x
    let mean = sq.mean(axis: -1, keepDims: true)
    let denom = (mean + eps).sqrt()
    return x / denom
}

// MARK: - GroupNorm（NDHWC 布局）

/// GroupNorm（NDHWC 布局）。weight/bias 形状 [channels]。
/// 对 NDHWC [B,F,H,W,C] 按通道分组归一化：每组 c/groups 个通道，
/// 统计范围 = 组内通道 × F × H × W（biased var，与 torch GroupNorm 一致）。
/// 内部提升 f32 计算，输出转回原 dtype——修复 bf16 下 GroupNorm 数值灾难
///（减均值后 diff² 相消，8 位精度爆掉；权重 bf16→f32 无损失）。
func groupNormNDHWC(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray,
    groups: Int = 32,
    eps: Float = 1e-5
) -> MLXArray {
    let origDtype = x.dtype
    let xf = x.asType(.float32)
    let wf = weight.asType(.float32)
    let bf = bias.asType(.float32)
    let b = xf.shape[0], f = xf.shape[1], h = xf.shape[2], w = xf.shape[3], c = xf.shape[4]
    let cpg = c / groups
    let r = xf.reshaped([b, f, h, w, groups, cpg])
    let mean = MLX.mean(r, axes: [1, 2, 3, 5], keepDims: true)          // [B,1,1,1,G,1]
    let diff = r - mean
    let variance = MLX.mean(diff * diff, axes: [1, 2, 3, 5], keepDims: true)
    let norm = diff / (variance + eps).sqrt()
    let w2 = wf.reshaped([1, 1, 1, 1, groups, cpg])
    let b2 = bf.reshaped([1, 1, 1, 1, groups, cpg])
    return (norm * w2 + b2).reshaped([b, f, h, w, c]).asType(origDtype)
}

// MARK: - RMSNorm（无参）

/// 无参 RMSNorm：x * rsqrt(mean(x², last axis) + eps)。
/// 注意：dim 参数保留为调用兼容，实际按源实现固定 last axis（-1）。
func rmsNormNoAffine(_ x: MLXArray, dim: Int, eps: Float) -> MLXArray {
    x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps)
}

// MARK: - Module 类封装（兼容原模型侧实例化调用点）

/// 无参 RMSNorm（norm_elementwise_affine = false，无 weight 参数）。
/// 数学与 rmsNormNoAffine 等价，Module 类封装供模型侧以模块方式实例化。
/// 使用方：LTX-2.5（LTXDiT LTXGatedAttention）。
final class RMSNormNoAffine: Module, UnaryLayer {
    let eps: Float
    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps)
    }
}

/// 无参 LayerNorm（输出 head 用，对应参考 layerNormAF / mx.fast.layer_norm(weight=None, bias=None)）。
/// 使用方：LTX-2.5（LTXDiT）。
final class LayerNormNoAffine: Module, UnaryLayer {
    let eps: Float
    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let diff = x - mean
        let variance = (diff * diff).mean(axis: -1, keepDims: true)
        return diff * rsqrt(variance + eps)
    }
}

/// 有参 RMSNorm（attention q_norm/k_norm 用，weight 形状 [inner]）。
/// 使用方：LTX-2.5（LTXDiT LTXGatedAttention、Gemma4TextEncoder、Connector）。
final class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(dim: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dim])
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps) * weight
    }
}

// MARK: - 有参 RMSNorm（自由函数，MLXFast 融合 kernel 版）

/// 有参 RMSNorm 自由函数：沿最后一维归一化（MLXFast 融合 Metal kernel，
/// 单 kernel 完成 square→mean→rsqrt→scale；公式与手写一致 mean(x²)+eps 再 rsqrt）。
/// 实现与 DDVNA.rmsNorm 原实现逐字一致（eps 默认 1e-6）。
/// 与上方 `RMSNorm` Module 类数学等价，但保留 MLXFast 融合 kernel 数值路径；
/// 使用方：LTX-2.5（扩散视频解码器 DDVDecoder / DDVNA）。
func rmsNormWeighted(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: weight, eps: eps)
}
