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

// MARK: - 分块 Tile 规划（VAE 滑动窗口）

/// split_by_size：一维切成 size 长的重叠 tile，返回 (start, end, left_ramp, right_ramp)。
/// 使用方：LTX-2.5（VaeDecoder）。
func vaeSplitBySize(
    dim: Int, size: Int, overlap: Int
) -> [(start: Int, end: Int, leftRamp: Int, rightRamp: Int)] {
    if dim <= size { return [(0, dim, 0, 0)] }
    let amount = (dim + size - 2 * overlap - 1) / (size - overlap)
    var out: [(Int, Int, Int, Int)] = [(0, size, 0, overlap)]
    if amount > 2 {
        for i in 1 ..< (amount - 1) {
            let s = i * (size - overlap)
            out.append((s, s + size, overlap, overlap))
        }
    }
    out.append(((amount - 1) * (size - overlap), dim, overlap, 0))
    return out
}

/// split_temporal_causal：非首块 start-1、left_ramp+1。
/// 使用方：LTX-2.5（VaeDecoder）。
func vaeSplitTemporalCausal(
    dim: Int, size: Int, overlap: Int
) -> [(start: Int, end: Int, leftRamp: Int, rightRamp: Int)] {
    if dim <= size { return [(0, dim, 0, 0)] }
    let ivs = vaeSplitBySize(dim: dim, size: size, overlap: overlap)
    if ivs.count <= 1 { return ivs }
    return [ivs[0]] + ivs.dropFirst().map {
        ($0.start - 1, $0.end, $0.leftRamp + 1, $0.rightRamp)
    }
}

/// 1D 梯形 blend mask（对齐 compute_trapezoidal_mask_1d）。
/// 使用方：LTX-2.5（VaeDecoder）。
func vaeTrapezoidMask1D(
    length: Int, rampLeft: Int, rampRight: Int, leftStartsFromZero: Bool
) -> [Float] {
    var mask = [Float](repeating: 1.0, count: length)
    let rl = max(0, min(rampLeft, length))
    let rr = max(0, min(rampRight, length))
    if rl > 0 {
        for i in 0 ..< rl {
            // leftStartsFromZero=False: 1/(r+1)…r/(r+1)；True: 0…(r-1)/r
            let x = leftStartsFromZero ? Float(i) / Float(rl) : Float(i + 1) / Float(rl + 1)
            mask[i] *= x
        }
    }
    if rr > 0 {
        for i in 0 ..< rr {
            let x = Float(rr - i) / Float(rr + 1)   // r/(r+1)…1/(r+1)
            mask[length - rr + i] *= x
        }
    }
    return mask
}

/// latent interval → 输出像素 slice + ramp（参数化合并 vaeMapTemporal / vaeMapSpatial）。
/// - temporal=true：对齐 map_temporal_slice（scale=8），e/lr 采用时间轴公式
///   （e = 1+(end-1)*scale；lr = leftRamp==0 ? 0 : 1+(leftRamp-1)*scale）。
/// - temporal=false：对齐 map_spatial_slice（scale=32），e/lr 直接乘以 scale。
/// 使用方：LTX-2.5（VaeDecoder，时间轴传 temporal: true、空间轴传 temporal: false）。
func vaeMap(
    _ iv: (start: Int, end: Int, leftRamp: Int, rightRamp: Int), scale: Int = 8, temporal: Bool = false
) -> (start: Int, end: Int, leftRamp: Int, rightRamp: Int) {
    let s = iv.start * scale
    let e: Int
    let lr: Int
    if temporal {
        e = 1 + (iv.end - 1) * scale
        lr = iv.leftRamp == 0 ? 0 : 1 + (iv.leftRamp - 1) * scale
    } else {
        e = iv.end * scale
        lr = iv.leftRamp * scale
    }
    let rr = iv.rightRamp * scale
    return (s, e, lr, rr)
}

// MARK: - 图像尺寸规划

/// py resize_pilimage 的面积约束 + patch 对齐目标尺寸（box 降采样循环已省略）。
/// - Returns: (newW, newH)，均为 patch 倍数且面积 ≤ imageSize²。
/// 使用方：HiDream-O1-Image（HiDreamPipeline）。
func hidreamResizeTarget(srcW: Int, srcH: Int, imageSize: Int, patch: Int) -> (Int, Int) {
    let sMax = Float(imageSize * imageSize)
    let scale = sqrt(sMax / Float(srcW * srcH))
    let rw = Float(srcW) * scale
    let rh = Float(srcH) * scale
    var candidates: [(Float, Int, Int)] = []
    let pairs: [(Float, Float)] = [
        (rw.rounded(), rh.rounded()),
        (rw.rounded(), rh.rounded(.down)),
        (rw.rounded(.down), rh.rounded()),
        (rw.rounded(.down), rh.rounded(.down)),
    ]
    for (cw, ch) in pairs {
        let nw = Int(cw) / patch * patch
        let nh = Int(ch) / patch * patch
        if nw > 0, nh > 0 {
            candidates.append((Float(nw * nh), nw, nh))
        }
    }
    guard !candidates.isEmpty else { return (patch, patch) }
    candidates.sort { $0.0 > $1.0 }
    let best = candidates.first(where: { $0.0 <= sMax }) ?? candidates[candidates.count - 1]
    return (best.1, best.2)
}

/// py calculate_dimensions：maxSize 内、patch 对齐、保持比例。
/// 输入 ratio 非有限或结果非有限/小于一个 patch 时兜底返回 (32, 32)，
/// 避免 Int(Float) 因 inf/NaN 触发 Swift fatal error。
/// 使用方：HiDream-O1-Image（HiDreamPipeline）。
func hidreamVisionSize(maxSize: Int, ratio: Float) -> (Int, Int) {
    guard ratio.isFinite, ratio > 0 else { return (32, 32) }
    var width = sqrt(Float(maxSize * maxSize) * ratio)
    var height = width / ratio
    guard width.isFinite, height.isFinite, width >= 32, height >= 32 else {
        return (32, 32)
    }
    width = Float(Int(width / 32)) * 32
    height = Float(Int(height / 32)) * 32
    return (Int(width), Int(height))
}

// MARK: - 便捷构造

/// [Float] → MLXArray（自动推断 shape；一维数组便捷写法）。
func floatArray(_ values: [Float]) -> MLXArray {
    MLXArray(values)
}
