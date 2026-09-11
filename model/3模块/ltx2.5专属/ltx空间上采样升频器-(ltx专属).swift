//
//  SpatialUpscaler.swift
//  无限画布 — LTX-2.5 空间升频器（LatentUpsampler，spatial ×2）
//
//  ============================================================
//  作用：把去噪后的 video latent [B,F,H,W,128]（NDHWC）在潜在空间
//  做 2 倍空间升采样 → [B,F,2H,2W,128]，再交给 VAE 解码得到
//  2 倍分辨率的视频帧。官方 multiscale 工作流："低分辨率生成 +
//  latent upscale ×2 + 解码"。
//  参考：Lightricks LTX-Video latent_upsampler.py（LatentUpsampler，
//        dims=3, spatial_upsample=true）。
//  结构：initial_conv(128→1024,Conv3d k3 pad1) → GroupNorm(32)
//        → SiLU → res_blocks×4 → Conv2d(1024→4096)+PixelShuffle2D(2)
//        → post_upsample_res_blocks×4 → final_conv(1024→128,Conv3d)。
//  权重：spatial_upscaler_x2_v1_1.safetensors，键前缀
//        spatial_upscaler_x2_v1_1.，MLX 布局 [C_out, kD, kH, kW, C_in]。
//  布局：全链路 NDHWC，与 VaeDecoder.swift 一致。
//  ============================================================

import Foundation
import MLX
import MLXNN
// MARK: - ResBlock（官方 LatentUpsampler 结构：post-activation residual）

/// conv1 → norm1 → SiLU → conv2 → norm2 → SiLU(x + residual)
/// 注意与 VaeDecoder.vaeResBlock3d（pre-activation）不同，这是官方
/// LatentUpsampler 的 ResBlock 顺序。
func upResBlock(
    _ x: MLXArray,
    weights: [String: MLXArray],
    base: String
) -> MLXArray {
    let residual = x
    let c1 = conv3dNDHWC(x, weight: weights[base + ".conv1.weight"]!, bias: weights[base + ".conv1.bias"], timePad: .zeroSymmetric, chunkBudget: 1 << 30, evalChunks: true)
    let n1 = groupNormNDHWC(c1, weight: weights[base + ".norm1.weight"]!, bias: weights[base + ".norm1.bias"]!)
    let a1 = silu(n1)
    let c2 = conv3dNDHWC(a1, weight: weights[base + ".conv2.weight"]!, bias: weights[base + ".conv2.bias"], timePad: .zeroSymmetric, chunkBudget: 1 << 30, evalChunks: true)
    let n2 = groupNormNDHWC(c2, weight: weights[base + ".norm2.weight"]!, bias: weights[base + ".norm2.bias"]!)
    return silu(n2 + residual)
}

// MARK: - 空间升频器主前向

/// 对已 denormalize 的 latent [B,F,H,W,128]（NDHWC）做空间 ×2 升采样，
/// 返回 [B,F,2H,2W,128]（NDHWC，仍在 latent 分布，尚未 normalize）。
/// weights 直接传 spatial_upscaler_x2_v1_1.safetensors 的原始字典
/// （键含 spatial_upscaler_x2_v1_1. 前缀）。
func latentUpsamplerSpatial(
    weights: [String: MLXArray],
    latentNDHWC: MLXArray
) -> MLXArray {
    let b = latentNDHWC.shape[0], f = latentNDHWC.shape[1]
    let prefix = "spatial_upscaler_x2_v1_1"

    // initial_conv 128→1024 + GroupNorm(32) + SiLU
    var x = conv3dNDHWC(latentNDHWC,
                           weight: weights[prefix + ".initial_conv.weight"]!,
                           bias: weights[prefix + ".initial_conv.bias"],
                           timePad: .zeroSymmetric, chunkBudget: 1 << 30, evalChunks: true)
    x = groupNormNDHWC(x,
                       weight: weights[prefix + ".initial_norm.weight"]!,
                       bias: weights[prefix + ".initial_norm.bias"]!)
    x = silu(x)
    eval(x)

    // res_blocks ×4（1024 通道 3D 卷积，分块跑，峰值内存可控）
    for i in 0..<4 {
        x = upResBlock(x, weights: weights, base: prefix + ".res_blocks.\(i)")
        eval(x)
    }

    // spatial upsample：帧合并进 batch → Conv2d(1024→4096) → PixelShuffle2D(2)
    let h = x.shape[2], w = x.shape[3]
    let x2 = x.reshaped([b * f, h, w, x.shape[4]])
    let uw = weights[prefix + ".upsampler.0.weight"]!      // [4096, 3, 3, 1024]
    let ub = weights[prefix + ".upsampler.0.bias"]!
    var u = MLX.conv2d(x2, uw, stride: IntOrPair((1, 1)), padding: IntOrPair((1, 1)), dilation: IntOrPair((1, 1))) + ub
    u = pixelShuffle2dNDHWC(u, r: 2)                        // [B*F, 2H, 2W, 1024]
    x = u.reshaped([b, f, u.shape[1], u.shape[2], u.shape[3]])
    eval(x)

    // post_upsample_res_blocks ×4
    for i in 0..<4 {
        x = upResBlock(x, weights: weights, base: prefix + ".post_upsample_res_blocks.\(i)")
        eval(x)
    }

    // final_conv 1024→128
    x = conv3dNDHWC(x,
                       weight: weights[prefix + ".final_conv.weight"]!,
                       bias: weights[prefix + ".final_conv.bias"],
                       timePad: .zeroSymmetric, chunkBudget: 1 << 30, evalChunks: true)
    eval(x)
    return x
}
