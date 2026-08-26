//
//  VaeDecoder.swift
//  ltx-test — LTX-2.5 视频 VAE 解码器（对齐 mlx-serve src/ltx_video.zig vaeDecode）
//
//  ============================================================
//  作用：把去噪后的 video latent [B,F,H,W,128]（NDHWC）解码为像素
//  [B,3,8F-7,32H,32W]（BCFHW，[-1,1]）。
//  参考：ltx_video.zig decoderConv3d(264) / pixelShuffle3d(420) /
//        resBlock3d / vaeDecode(480)，全部 NDHWC 布局。
//  权重：vae_decoder.safetensors，键前缀 vae_decoder.，MLX 布局
//        [C_out, kD, kH, kW, C_in]。
//  ============================================================

import Foundation
import MLX
import MLXNN
import Accelerate
/// Encoder ResBlock（pre-activation residual，同解码器结构）：
/// conv2(silu(pn(conv1(silu(pn(x)))))) + x
func vaeEncResBlock(_ x: MLXArray, weights: [String: MLXArray], base: String) -> MLXArray {
    func convKey(_ k: String) -> (w: MLXArray, b: MLXArray?) {
        (weights[k + ".weight"]!, weights[k + ".bias"])
    }
    let pn1 = pixelNormFast(x, eps: 1e-8)
    let a1 = silu(pn1)
    let (w1, b1) = convKey(base + ".conv1.conv")
    let c1 = encoderConv3d(a1, weight: w1, bias: b1)
    let pn2 = pixelNormFast(c1, eps: 1e-8)
    let a2 = silu(pn2)
    let (w2, b2) = convKey(base + ".conv2.conv")
    let c2 = encoderConv3d(a2, weight: w2, bias: b2)
    return c2 + x
}

/// Encoder down 块（skip+conv 双分支 space-to-depth downsample，对齐 s2dDownsample）。
func s2dDownsample(_ x: MLXArray, weights: [String: MLXArray], downIdx: Int,
                   inCh: Int, outCh: Int, st: Int, sh: Int, sw: Int) -> MLXArray {
    let prod = st * sh * sw
    let groupSize = inCh * prod / outCh

    // 时间下采样时 causal prepend 首帧
    var xt = x
    if st == 2 {
        let first = x[0 ..< 1, axis: 1]
        xt = MLX.concatenated([first, x], axis: 1)
    }

    // Skip 分支：space-to-depth → 可选 group-mean
    var xSkip = spaceToDepth(xt, st: st, sh: sh, sw: sw)
    if groupSize > 1 {
        let b = xSkip.shape[0], d = xSkip.shape[1], h = xSkip.shape[2], w = xSkip.shape[3]
        let rg = xSkip.reshaped([b, d, h, w, outCh, groupSize])
        xSkip = rg.mean(axis: -1)
    }

    // Conv 分支：causal conv → space-to-depth
    let (wc, bc) = (weights["vae_encoder.down_blocks.\(downIdx).conv.conv.weight"]!, weights["vae_encoder.down_blocks.\(downIdx).conv.conv.bias"])
    let cv = encoderConv3d(xt, weight: wc, bias: bc)
    let xConv = spaceToDepth(cv, st: st, sh: sh, sw: sw)

    return (xConv + xSkip).asType(.float32).asType(.bfloat16).contiguous()
}
// MARK: - ResBlock（pre-activation residual：conv2(silu(pn(conv1(silu(pn(x))))))+x）

func vaeResBlock3d(
    _ x: MLXArray,
    conv1W: MLXArray, conv1B: MLXArray?,
    conv2W: MLXArray, conv2B: MLXArray?
) -> MLXArray {
    // 逐步 eval + var 复用：每步只保留 1~2 个全尺寸张量，
    // 避免 pn/a1/c1/pn2/a2/c2 全部 let 存活叠成瞬态峰值
    var a = silu(pixelNormFast(x))
    a.eval()
    var c = decoderConv3dZig(a, weight: conv1W, bias: conv1B)
    c.eval()
    a = silu(pixelNormFast(c))     // 复用 a，释放第一段中间量
    a.eval()
    c = decoderConv3dZig(a, weight: conv2W, bias: conv2B)  // 复用 c
    c.eval()
    let out = c + x
    out.eval()
    return out
}

// MARK: - 完整解码器

/// 解码：latent [B,F,H,W,128]（NDHWC）→ pixels [B,3,8F-7,32H,32W]（BCFHW）。
/// weights 直接传 vae_decoder.safetensors 的原始字典（键含 vae_decoder. 前缀）。
func vaeDecode(
    weights: [String: MLXArray],
    latentNDHWC: MLXArray
) -> MLXArray {
    func convKey(_ base: String) -> (w: MLXArray, b: MLXArray?) {
        guard let w = weights[base + ".weight"] else {
            fatalError("缺键：\(base).weight\n全部键：\(weights.keys.sorted().joined(separator: ", "))")
        }
        return (w, weights[base + ".bias"])
    }

    var x = latentNDHWC

    // denormalize: x*std + mean
    // mean/std 权重是 f32，若不转 bf16 会把整条解码链路的激活张量抬成 f32
    // （128×128×1024 层 f32=7.6GB vs bf16=3.8GB），转 bf16 保持全链路 bf16。
    let mean = weights["vae_decoder.per_channel_statistics.mean"]!.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
    let std = weights["vae_decoder.per_channel_statistics.std"]!.reshaped([1, 1, 1, 1, 128]).asType(.bfloat16)
    x = x * std + mean

    // conv_in 128→1024
    let cin = convKey("vae_decoder.conv_in.conv")
    x = decoderConv3dZig(x, weight: cin.w, bias: cin.b)

    // up_blocks 交替：even=ResStage，odd=DepthToSpace conv + pixel_shuffle
    let resStages: [(up: Int, blocks: Int)] = [
        (0, 2), (2, 2), (4, 4), (6, 6), (8, 4),
    ]
    let ups: [(up: Int, sf: Int, tf: Int)] = [
        (1, 2, 2), (3, 2, 2), (5, 1, 2), (7, 2, 1),
    ]
    var i = 0
    while i <= 8 {
        if i % 2 == 0 {
            let stage = resStages.first { $0.up == i }!
            for b in 0..<stage.blocks {
                let c1 = convKey("vae_decoder.up_blocks.\(i).res_blocks.\(b).conv1.conv")
                let c2 = convKey("vae_decoder.up_blocks.\(i).res_blocks.\(b).conv2.conv")
                x = vaeResBlock3d(x, conv1W: c1.w, conv1B: c1.b, conv2W: c2.w, conv2B: c2.b)
            }
        } else {
            let spec = ups.first { $0.up == i }!
            let ck = convKey("vae_decoder.up_blocks.\(i).conv.conv")
            var cv = decoderConv3dZig(x, weight: ck.w, bias: ck.b)
            x = cv
            var ps = pixelShuffle3dNDHWC(x, sf: spec.sf, tf: spec.tf)
            if spec.tf > 1 {
                let d = ps.shape[1]
                ps = ps[1 ..< d, axis: 1]      // drop first frame after temporal upsample
            }
            x = ps.contiguous()
            eval(x)                            // keep lazy graph bounded
        }
        i += 1
    }

    // conv_out(silu(pixel_norm(x))) 128→48
    let pn = pixelNormFast(x)
    let a = silu(pn)
    let co = convKey("vae_decoder.conv_out.conv")
    x = decoderConv3dZig(a, weight: co.w, bias: co.b)

    // final spatial unpatchify 48→3（4× spatial）
    x = unpatchifySpatialNDHWC(x, ps: 4)

    // NDHWC → BCFHW 并连续化
    x = x.transposed(axes: [0, 4, 1, 2, 3]).contiguous()
    return x
}

// MARK: - 完整编码器（图片条件编码，I2V 用）

/// 编码单张图片 [1,3,1,H,W]（BCFHW，[-1,1]）→ latent [1,128,1,H',W']（BCFHW）。
/// 对齐 ltx_video.zig vaeEncode(784) + patchifySpatial4(662)。权重传 vae_encoder.safetensors。
func vaeEncodeImage(weights: [String: MLXArray], pixelsBCFHW: MLXArray) -> MLXArray {
    func stat(_ x: MLXArray, _ tag: String) {
        eval(x)
        let a = x.asArray(Float.self)
        let m = a.reduce(0, +) / Float(a.count)
        let mx = a.map { abs($0) }.max() ?? 0
        print("  [ENC:\(tag)] shape=\(x.shape) mean=\(m) maxAbs=\(mx)")
    }
    // BCFHW → BFHWC
    var h = pixelsBCFHW.transposed(axes: [0, 2, 3, 4, 1])       // [B,1,H,W,3]
    // 空间 patchify ps=4：48 = 3*4*4
    h = patchifySpatial4(h, ps: 4)                              // [B,1,H/4,W/4,48]
    stat(h, "patchify")
    // conv_in 48→128
    h = encoderConv3d(h, weight: weights["vae_encoder.conv_in.conv.weight"]!,
                      bias: weights["vae_encoder.conv_in.conv.bias"])
    stat(h, "conv_in")
    // down_blocks：偶数=resStage，奇数=downsample；res 块数 [4,6,4,2,2]
    let resCounts = [4, 6, 4, 2, 2]
    let downSpecs: [(idx: Int, inCh: Int, outCh: Int, st: Int, sh: Int, sw: Int)] = [
        (1, 128, 256, 1, 2, 2),
        (3, 256, 512, 2, 1, 1),
        (5, 512, 1024, 2, 2, 2),
        (7, 1024, 1024, 2, 2, 2),
    ]
    var stage = 0
    for i in 0..<4 {
        let nBlocks = resCounts[stage]
        for blk in 0..<nBlocks {
            h = vaeEncResBlock(h, weights: weights, base: "vae_encoder.down_blocks.\(stage * 2).res_blocks.\(blk)")
        }
        stat(h, "res_down\(stage * 2)")
        stage += 1
        let ds = downSpecs[i]
        h = s2dDownsample(h, weights: weights, downIdx: ds.idx, inCh: ds.inCh, outCh: ds.outCh,
                          st: ds.st, sh: ds.sh, sw: ds.sw)
        stat(h, "s2d_down\(ds.idx)")
    }
    // 最后一个 resStage（idx 8，2 块）
    for blk in 0..<resCounts[4] {
        h = vaeEncResBlock(h, weights: weights, base: "vae_encoder.down_blocks.8.res_blocks.\(blk)")
    }
    stat(h, "res_down8")
    // conv_out：silu(pixel_norm(x)) → conv 1024→129，取前 128 通道
    let pn = pixelNormFast(h, eps: 1e-8)
    let a = silu(pn)
    let out = encoderConv3d(a, weight: weights["vae_encoder.conv_out.conv.weight"]!,
                            bias: weights["vae_encoder.conv_out.conv.bias"])
    stat(out, "conv_out")
    let latent = out[0..., 0..., 0..., 0..., 0 ..< 128]          // [B,1,H',W',128]
    stat(latent, "slice128")
    // per-channel 归一化：(x-mean)/std，stats [128] → [1,1,1,1,128]
    let mean = weights["vae_encoder.per_channel_statistics._mean_of_means"]!.reshaped([1, 1, 1, 1, 128])
    let std = weights["vae_encoder.per_channel_statistics._std_of_means"]!.reshaped([1, 1, 1, 1, 128])
    let normed = (latent - mean) / std
    stat(normed, "normalized")
    // BFHWC → BCFHW
    return normed.transposed(axes: [0, 4, 1, 2, 3])
}

// MARK: - VAE 解码 Tiling（对照官方 ltx_core/tiling.py split_by_size / split_temporal_causal）
//
// 目的：把全分辨率 latent 按时间+空间分块解码，压低 720p 5s 场景 VAE 解码峰值内存
// （当前整片解码峰值 ~51.45GB）。语义与官方一致：
//   - 时间轴用 split_temporal_causal：后续块 start-1、left_ramp+1，保证因果连续
//   - 空间轴用 split_by_size：重叠区梯形 mask 加权融合
//   - 每块独立跑 vaeDecode，输出乘分离 1D mask 后按 out_coords 累加，最后除以权重
// 默认 tile 参数对齐官方 TileSizeConfig.default()（latent 网格单位）：
//   时间 tile=10 帧、overlap=3；空间 tile=24 格、overlap=2
//   （官方像素/帧 80/24 与 768/64 ÷ scale 8/32 得来）

/// split_by_size：一维切成 size 长的重叠 tile，返回 (start, end, left_ramp, right_ramp)
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

/// split_temporal_causal：非首块 start-1、left_ramp+1
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

/// 1D 梯形 blend mask（对齐 compute_trapezoidal_mask_1d）
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

/// 时间轴 latent interval → 输出帧 slice + ramp（对齐 map_temporal_slice，scale=8）
func vaeMapTemporal(
    _ iv: (start: Int, end: Int, leftRamp: Int, rightRamp: Int), scale: Int = 8
) -> (start: Int, end: Int, leftRamp: Int, rightRamp: Int) {
    let s = iv.start * scale
    let e = 1 + (iv.end - 1) * scale
    let lr = iv.leftRamp == 0 ? 0 : 1 + (iv.leftRamp - 1) * scale
    let rr = iv.rightRamp * scale
    return (s, e, lr, rr)
}

/// 空间轴 latent interval → 输出像素 slice + ramp（对齐 map_spatial_slice，scale=32）
func vaeMapSpatial(
    _ iv: (start: Int, end: Int, leftRamp: Int, rightRamp: Int), scale: Int = 32
) -> (start: Int, end: Int, leftRamp: Int, rightRamp: Int) {
    let s = iv.start * scale
    let e = iv.end * scale
    let lr = iv.leftRamp * scale
    let rr = iv.rightRamp * scale
    return (s, e, lr, rr)
}

/// 按内存压力分动态计算 VAE 空间分块（连续插值，无离散跳变）：
/// raw = 48 − 0.75 × max(0, 压力分 − 8)，clamp [24, 48]；
/// overlap 按格数映射（≥44→4，≥32→3，否则 2）；
/// 小 latent（maxDim > 28）再过空间分块下限 cap = max(16, maxDim/2 + overlap)，防止拆太碎。
func vaeSpatialBudget(pressureScore: Double, latentH: Int, latentW: Int) -> (tileSizeSpatial: Int, overlapSpatial: Int) {
    let raw = 48 - 0.75 * max(0, pressureScore - 8)
    var t = Int(raw.rounded())
    t = min(48, max(24, t))
    let overlap = t >= 44 ? 4 : (t >= 32 ? 3 : 2)
    let maxDim = max(latentH, latentW)
    if maxDim > 28 {
        let cap = max(16, maxDim / 2 + overlap)
        t = min(t, cap)
    }
    return (t, overlap)
}

/// Tiled VAE 解码：latent [B,F,H,W,128]（NDHWC）→ pixels [B,3,8F-7,32H,32W]（BCFHW）。
/// 默认 tile 参数对齐官方 TileSizeConfig.default() 换算到 latent 网格。
/// 空间分块按内存压力分动态连续插值：入口定一次初始档，每个时间块动态重查压力分重算格数，
/// 压力大自动拆小块（省内存），压力小自动合并（提速），全程无离散跳变。
func vaeDecodeTiled(
    weights: [String: MLXArray],
    latentNDHWC: MLXArray,
    tileSizeFrames: Int = 10,
    tileSizeSpatial: Int = 24,
    overlapFrames: Int = 3,
    overlapSpatial: Int = 2
) -> MLXArray {
    let B = latentNDHWC.shape[0]
    let F = latentNDHWC.shape[1]
    let H = latentNDHWC.shape[2]
    let W = latentNDHWC.shape[3]
    let C = latentNDHWC.shape[4]
    let outT = 8 * F - 7
    let outH = 32 * H
    let outW = 32 * W

    let tIvs = vaeSplitTemporalCausal(dim: F, size: tileSizeFrames, overlap: overlapFrames)
    // 入口：按当前压力分查初始空间分块（连续插值），时间档保持入参固定
    var tileSizeSpatial = tileSizeSpatial
    var overlapSpatial = overlapSpatial
    let entryBudget = vaeSpatialBudget(pressureScore: SystemMemory.pressureScore, latentH: H, latentW: W)
    tileSizeSpatial = entryBudget.tileSizeSpatial
    overlapSpatial = entryBudget.overlapSpatial
    var hIvs = vaeSplitBySize(dim: H, size: tileSizeSpatial, overlap: overlapSpatial)
    var wIvs = vaeSplitBySize(dim: W, size: tileSizeSpatial, overlap: overlapSpatial)
    print("[VAE-Tiled] 初始空间分块 压力分 \(String(format: "%.1f", SystemMemory.pressureScore)) → \(tileSizeSpatial) 格（overlap \(overlapSpatial)），时间档 \(tileSizeFrames)/ov\(overlapFrames)")

    // CPU f32 累加 buffer（像素）+ 权重 buffer（mask），最后归一化
    var buffer = [Float](repeating: 0, count: 3 * outT * outH * outW)
    var wbuf = [Float](repeating: 0, count: 3 * outT * outH * outW)

    var total = tIvs.count * hIvs.count * wIvs.count
    var tileCount = 0
    for tIv in tIvs {
        let tm = vaeMapTemporal(tIv)
        let tMask = vaeTrapezoidMask1D(
            length: tm.end - tm.start, rampLeft: tm.leftRamp, rampRight: tm.rightRamp,
            leftStartsFromZero: true
        )
        // 每个时间块动态重查压力分，连续插值重算空间分块（压力变化时调整格数，无跳变）
        let ps = SystemMemory.pressureScore
        let spatial = vaeSpatialBudget(pressureScore: ps, latentH: H, latentW: W)
        if spatial.tileSizeSpatial != tileSizeSpatial || spatial.overlapSpatial != overlapSpatial {
            tileSizeSpatial = spatial.tileSizeSpatial
            overlapSpatial = spatial.overlapSpatial
            hIvs = vaeSplitBySize(dim: H, size: tileSizeSpatial, overlap: overlapSpatial)
            wIvs = vaeSplitBySize(dim: W, size: tileSizeSpatial, overlap: overlapSpatial)
            total = tIvs.count * hIvs.count * wIvs.count
            print("[VAE-Tiled] 时间块 \(tIv.start)~\(tIv.end)：压力分 \(String(format: "%.1f", ps)) → 空间 \(tileSizeSpatial) 格（overlap \(overlapSpatial)），总块数 \(total)")
        }
        for hIv in hIvs {
            let hm = vaeMapSpatial(hIv)
            let hMask = vaeTrapezoidMask1D(
                length: hm.end - hm.start, rampLeft: hm.leftRamp, rampRight: hm.rightRamp,
                leftStartsFromZero: false
            )
            for wIv in wIvs {
                let wm = vaeMapSpatial(wIv)
                let wMask = vaeTrapezoidMask1D(
                    length: wm.end - wm.start, rampLeft: wm.leftRamp, rampRight: wm.rightRamp,
                    leftStartsFromZero: false
                )

                let tileLatent = latentNDHWC[
                    0 ..< B, tIv.start ..< tIv.end,
                    hIv.start ..< hIv.end, wIv.start ..< wIv.end, 0 ..< C
                ]
                let px = vaeDecode(weights: weights, latentNDHWC: tileLatent)
                eval(px)
                let pxa = px.asArray(Float.self)
                let tLen = px.shape[2]
                let hLen = px.shape[3]
                let wLen = px.shape[4]

                // 预计算 tile 权重 mask3 = tMask ⊗ hMask ⊗ wMask（连续 [tLen,hLen,wLen]）
                var mask3 = [Float](repeating: 0, count: tLen * hLen * wLen)
                for t in 0 ..< tLen {
                    let mt = tMask[t]
                    for h in 0 ..< hLen {
                        let mth = mt * hMask[h]
                        let rowOff = (t * hLen + h) * wLen
                        for w in 0 ..< wLen {
                            mask3[rowOff + w] = mth * wMask[w]
                        }
                    }
                }
                // 向量化累加：vDSP_vma 逐行乘加（src×mask + buffer → buffer），wbuf 同步 vDSP_vadd，
                // 替代原四层 for 逐元素累加（CPU 主瓶颈），行内 w 段一次 vDSP 调用处理。
                // 用 withUnsafe* 缓冲指针规避同一数组多重 inout 借用的 exclusivity 冲突。
                buffer.withUnsafeMutableBufferPointer { buf in
                    wbuf.withUnsafeMutableBufferPointer { wb in
                        pxa.withUnsafeBufferPointer { pb in
                            mask3.withUnsafeBufferPointer { mb in
                                for c in 0 ..< 3 {
                                    for t in 0 ..< tLen {
                                        let gt = tm.start + t
                                        for h in 0 ..< hLen {
                                            let gh = hm.start + h
                                            let srcBase = ((c * tLen + t) * hLen + h) * wLen
                                            let dstBase = ((c * outT + gt) * outH + gh) * outW + wm.start
                                            let maskOff = (t * hLen + h) * wLen
                                            vDSP_vma(pb.baseAddress! + srcBase, 1, mb.baseAddress! + maskOff, 1,
                                                     buf.baseAddress! + dstBase, 1, buf.baseAddress! + dstBase, 1,
                                                     vDSP_Length(wLen))
                                            vDSP_vadd(wb.baseAddress! + dstBase, 1, mb.baseAddress! + maskOff, 1,
                                                      wb.baseAddress! + dstBase, 1, vDSP_Length(wLen))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                tileCount += 1
                print("  [VAE-Tiled] tile \(tileCount)/\(total) done（\(tLen)帧×\(hLen)×\(wLen)）")
            }
        }
    }

    for i in 0 ..< buffer.count {
        buffer[i] = wbuf[i] > 1e-8 ? buffer[i] / wbuf[i] : 0
    }
    let out = MLXArray(buffer, [B, 3, outT, outH, outW])
    return out.asType(.bfloat16)
}