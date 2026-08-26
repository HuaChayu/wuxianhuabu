//
//  模型通用算子-张量变换.swift
//  无限画布 — 通用张量重排算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用 reshape/transpose 变换（LTX-2.5 视频链路 + HiDream-O1-Image 图像）：
//    · patchifySpatial4：空间 patchify（space-to-depth，H/W → 通道）—— 使用方：LTX-2.5（VaeDecoder）
//    · spaceToDepth：通用 space-to-depth（含时间维）—— 使用方：LTX-2.5（VaeDecoder）
//    · pixelShuffle3dNDHWC：3D depth-to-space（时间/空间上采样）—— 使用方：LTX-2.5（VaeDecoder）
//    · pixelShuffle2dNDHWC：2D depth-to-space（空间上采样）—— 使用方：LTX-2.5（SpatialUpscaler）
//    · unpatchifySpatialNDHWC：最终空间 unpatchify（逆 patchify）—— 使用方：LTX-2.5（VaeDecoder）
//    · patchifySpatial2DBCHW / unpatchifySpatialBCHW：2D BCHW 图像版 patch—— 使用方：HiDream-O1-Image
//  布局约定：NDHWC（视频）/ BCHW（图像），通道 split order 见各函数注释。
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - Patchify（空间 → 通道）

/// 空间 patchify（space-to-depth）：(B,F,H,W,C) → (B,F,H/ps,W/ps,C·ps·ps)。
/// Channel split order (c, r=W, q=H)——对齐参考 patchifySpatial4（unpatchify 的逆）。
func patchifySpatial4(_ x: MLXArray, ps: Int) -> MLXArray {
    let b = x.shape[0], f = x.shape[1], h = x.shape[2], w = x.shape[3], c = x.shape[4]
    let r1 = x.reshaped([b, f, h / ps, ps, w / ps, ps, c])
    // (B,F,Hp,q_H,Wp,r_W,C) → (B,F,Hp,Wp,C,r_W,q_H)
    let t = r1.transposed(axes: [0, 1, 2, 4, 6, 5, 3])
    return t.reshaped([b, f, h / ps, w / ps, c * ps * ps])
}

/// Space-to-depth：(B,D,H,W,C) → (B,D/st,H/sh,W/sw,C·st·sh·sw)。
/// Channel order (c, st, sh, sw)——c 最外层（参考 space_to_depth）。
func spaceToDepth(_ x: MLXArray, st: Int, sh: Int, sw: Int) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], c = x.shape[4]
    let r1 = x.reshaped([b, d / st, st, h / sh, sh, w / sw, sw, c])
    // (B,D/st,st,H/sh,sh,W/sw,sw,C) → (B,D/st,H/sh,W/sw,C,st,sh,sw)
    let t = r1.transposed(axes: [0, 1, 3, 5, 7, 2, 4, 6])
    return t.reshaped([b, d / st, h / sh, w / sw, c * st * sh * sw])
}

// MARK: - Pixel Shuffle（通道 → 空间）

/// depth-to-space：(B,D,H,W, C·tf·sf·sf) → (B, D·tf, H·sf, W·sf, C)。
/// Channel split (c, p1=temporal, p2=H, p3=W)，c 最外层。
func pixelShuffle3dNDHWC(_ x: MLXArray, sf: Int, tf: Int) -> MLXArray {
    let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3]
    let ct = x.shape[4]
    let c = ct / (sf * sf * tf)
    let r1 = x.reshaped([b, d, h, w, c, tf, sf, sf])
    let t = r1.transposed(axes: [0, 1, 5, 2, 6, 3, 7, 4])
    return t.reshaped([b, d * tf, h * sf, w * sf, c])
}

/// 2D depth-to-space：(B,H,W, C·r·r) → (B, H·r, W·r, C)。
/// Channel split (c, rH, rW)，c 最外层——与 torch pixel_shuffle 一致。
func pixelShuffle2dNDHWC(_ x: MLXArray, r: Int) -> MLXArray {
    let b = x.shape[0], h = x.shape[1], w = x.shape[2]
    let ct = x.shape[3]
    let c = ct / (r * r)
    let r1 = x.reshaped([b, h, w, c, r, r])
    let t = r1.transposed(axes: [0, 1, 4, 2, 5, 3])   // [B, H, rH, W, rW, C]
    return t.reshaped([b, h * r, w * r, c])
}

/// 最终空间 unpatchify：(B,F,H,W, C·ps·ps) → (B,F, H·ps, W·ps, C)。
/// Channel split (c, r=W, q=H) — width before height。
func unpatchifySpatialNDHWC(_ x: MLXArray, ps: Int) -> MLXArray {
    let b = x.shape[0], f = x.shape[1], h = x.shape[2], w = x.shape[3]
    let ct = x.shape[4]
    let c = ct / (ps * ps)
    let r1 = x.reshaped([b, f, h, w, c, ps, ps])
    let t = r1.transposed(axes: [0, 1, 2, 6, 3, 5, 4])
    return t.reshaped([b, f, h * ps, w * ps, c])
}

// MARK: - Patchify（2D BCHW 图像版）

/// [C, H, W] → [H/ps · W/ps, C·ps·ps]（2D 图像 patchify，BCHW 布局）。
/// 与 patchifySpatial4（5D NDHWC 视频版）布局不同，仅用于图像扩散模型。
/// 使用方：HiDream-O1-Image（hidreamGenerate 初始噪声 patchify，ps=32）。
func patchifySpatial2DBCHW(_ imgCHW: MLXArray, patch: Int = 32) -> MLXArray {
    let c = imgCHW.dim(0), h = imgCHW.dim(1), w = imgCHW.dim(2)
    precondition(h % patch == 0 && w % patch == 0)
    let hP = h / patch, wP = w / patch
    // reshape(C, hP, patch, wP, patch) → transpose(1,3,0,2,4) → reshape(hP*wP, C*patch*patch)
    let x = imgCHW.reshaped([c, hP, patch, wP, patch])
    let xt = x.transposed(1, 3, 0, 2, 4)
    return xt.reshaped([hP * wP, c * patch * patch])
}

/// [N, C·ps·ps] → [C, H, W]（2D 图像 unpatchify，BCHW 布局，patchifySpatial2DBCHW 的逆）。
/// 使用方：HiDream-O1-Image（hidreamGenerate 最终还原，ps=32, C=3）。
func unpatchifySpatialBCHW(
    _ patches: MLXArray, hPatches: Int, wPatches: Int, patch: Int = 32, channels: Int = 3
) -> MLXArray {
    let x = patches.reshaped([hPatches, wPatches, channels, patch, patch])
    let xt = x.transposed(2, 0, 3, 1, 4)
    return xt.reshaped([channels, hPatches * patch, wPatches * patch])
}
