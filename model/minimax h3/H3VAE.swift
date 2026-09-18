//
//  H3VAE.swift
//  无限画布
//
//  MiniMax-H3 video VAE (Encoder + Decoder) 移植自 mlx-serve-main/src/minimax_h3_vae.zig。
//  Encoder: conv ResNet (FCN3D)，输出归一化 latent [1,24,T_lat,H/16,W/16]。
//  Decoder: 36-block ViT3DDecoder，temporal chunk + spatial tile + blend，
//           输出像素 [1,3,frames,H*16,W*16] in [-1,1]。
//

import MLX
import Foundation

// MARK: - 常量（与 minimax_h3_vae.zig 完全一致）

public enum H3VAEConst {
    public static let vaeRatio: Int = 16
    public static let vaeRatioT: Int = 4
    public static let clipLength: Int = 17
    public static let tokenDrop: Int = 3
    public static let tileSize: Int = 256
    /// splitTiles 的最小重叠（像素）。
    public static let tileOverlapMin: Int = 64

    public static let framePrePadding: Int = (vaeRatioT - (clipLength % vaeRatioT)) % vaeRatioT  // 3
    public static let tokensChunkSize: Int = (clipLength + vaeRatioT - 1) / vaeRatioT          // 5
    public static let tokenOverlap: Int = (tokensChunkSize - (tokenDrop % tokensChunkSize)) % tokensChunkSize  // 2
    public static let frameOverlap: Int = tokenOverlap * vaeRatioT > framePrePadding
        ? tokenOverlap * vaeRatioT - framePrePadding : 0                                        // 5

    public static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    public static let pixelStd: [Float] = [0.229, 0.224, 0.225]
    public static let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imagenetStd: [Float] = [0.229, 0.224, 0.225]

    // Encoder geometry (EncoderFCN3D: ch=128, ch_mult (1,2,2,4,4,8))
    public static let encLevels: Int = 6
    public static let encCh: Int = 128
    public static let encChMult: [Int] = [1, 2, 2, 4, 4, 8]
    public static let encSpaceDown: [Int] = [2, 2, 2, 2, 1, 1]
    public static let encTimeDown: [Int] = [1, 2, 2, 1, 1, 1]
    public static let encResBlocks: Int = 2
    public static func encBlockMid(_ level: Int) -> Int { encCh * encChMult[level] }
    public static func encBlockIn(_ level: Int) -> Int { level == 0 ? encBlockMid(0) : encCh * encChMult[level - 1] }
    public static func encHasDown(_ level: Int) -> Bool { encSpaceDown[level] * encTimeDown[level] > 1 }

    // Decoder geometry (ViT3DDecoder defaults)
    public static let decLayers: Int = 36
    public static let decHeads: Int = 32
    public static let decHeadDim: Int = 64
    public static let decInChannels: Int = 24
    public static let decOutChannels: Int = 3
    public static let decRopeTheta: Double = 100.0
    public static let decRopeDimRatio: Double = 0.75
    public static let decRegisterTokens: Int = 4
    public static let decEps: Float = 1e-5
    public static var decDim: Int { decHeads * decHeadDim }
    public static var decRopeFreqs: Int { Int((Double(decHeadDim) * decRopeDimRatio + 5) / 6) }
    public static var decRotDim: Int { decRopeFreqs * 3 * 2 }
    public static var decOutPatchDim: Int { decOutChannels * vaeRatioT * vaeRatio * vaeRatio }

    public static func fitsSingleTile(_ pixelExtent: Int) -> Bool { tileSize >= pixelExtent }
}

// MARK: - 辅助算子

/// 任意轴切片，返回视图（不复制）。axis 支持 0..<rank。
public func h3SliceAxis(_ x: MLXArray, _ start: Int, _ end: Int, axis: Int) -> MLXArray {
    switch axis {
    case 0: return x[start..<end]
    case 1: return x[0..., start..<end]
    case 2: return x[0..., 0..., start..<end]
    case 3: return x[0..., 0..., 0..., start..<end]
    case 4: return x[0..., 0..., 0..., 0..., start..<end]
    default: fatalError("h3SliceAxis: unsupported axis \(axis)")
    }
}

/// [O, I, kt, kh, kw] conv 权重 → [O, kt, kh, kw, I]（MLX conv3d 布局），cast 到 f32。
public func h3ConvTap(_ raw: MLXArray) -> MLXArray {
    raw.transposed(0, 2, 3, 4, 1).asType(.float32)
}

/// 1x1x1 conv 权重 [O, I, 1, 1, 1] → 预转置 f32 [I, O]。
public func h3OneByOne(_ raw: MLXArray) -> MLXArray {
    let sq = raw.reshaped(raw.shape[0], raw.shape[1])
    return sq.transposed(1, 0).asType(.float32)
}

/// 线性权重 [out, in] → 预转置 [in, out]，cast 到 dtype。
public func h3LoadLinT(_ raw: MLXArray, dtype: DType) -> MLXArray {
    raw.transposed(1, 0).asType(dtype)
}

/// conv3d + bias（已手动 pad，padding=0）。
public func h3Conv3dBias(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray?,
                         timeStride: Int = 1, spaceStride: Int = 1) -> MLXArray {
    var out = MLX.conv3d(x, w, stride: IntOrTriple((timeStride, spaceStride, spaceStride)), padding: 0)
    if let b { out = out + b }
    return out
}

/// 在 [1, T, H, W, C] 上对 H/W 做 reflect pad 1（镜像时不包含边缘样本）。
public func h3ReflectPad1HW(_ x: MLXArray) -> MLXArray {
    let h = x.shape[2], w = x.shape[3]
    let top = h3SliceAxis(x, 1, 2, axis: 2)
    let bot = h3SliceAxis(x, h - 2, h - 1, axis: 2)
    let xv = MLX.concatenated([top, x, bot], axis: 2)
    let left = h3SliceAxis(xv, 1, 2, axis: 3)
    let right = h3SliceAxis(xv, w - 2, w - 1, axis: 3)
    return MLX.concatenated([left, xv, right], axis: 3)
}

/// Downsample3D 的 (0,1,0,1) pad：仅 bottom+right reflect。
public func h3ReflectPadBR(_ x: MLXArray) -> MLXArray {
    let h = x.shape[2], w = x.shape[3]
    let bot = h3SliceAxis(x, h - 2, h - 1, axis: 2)
    let xv = MLX.concatenated([x, bot], axis: 2)
    let right = h3SliceAxis(xv, w - 2, w - 1, axis: 3)
    return MLX.concatenated([xv, right], axis: 3)
}

/// 时间轴前向零填充 n 帧。
public func h3CausalPadFront(_ x: MLXArray, _ n: Int) -> MLXArray {
    if n == 0 { return x }
    let shape = x.shape
    let z = MLXArray.zeros([shape[0], n, shape[2], shape[3], shape[4]], dtype: x.dtype)
    return MLX.concatenated([z, x], axis: 1)
}

/// 时间隔离 GroupNorm：per-frame 统计，axes [2,3,5]。
public func h3GroupNorm32(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let shp = x.shape
    let t = shp[1], h = shp[2], wd = shp[3], c = shp[4]
    let g = x.reshaped(1, t, h, wd, 32, c / 32)
    let mean = g.mean(axes: [2, 3, 5], keepDims: true)
    let diff = g - mean
    let vr = (diff * diff).mean(axes: [2, 3, 5], keepDims: true)
    let nrm = diff / (vr + eps).sqrt()
    let back = nrm.reshaped(1, t, h, wd, c)
    return back * w + b
}

/// 分片规划：与 Zig splitTiles 完全一致。返回 (starts, overlaps)。
public func h3SplitTiles(_ inputLen: Int) -> (starts: [Int], overlaps: [Int]) {
    let T = H3VAEConst.tileSize, OM = H3VAEConst.tileOverlapMin, R = H3VAEConst.vaeRatio
    if T >= inputLen { return ([0], []) }
    var n = (inputLen + T - 1) / T
    var remaining = 0
    while true {
        remaining = T * n - OM * (n - 1) - inputLen
        if remaining < 0 { n += 1 } else { break }
    }
    var overlaps = [Int](repeating: OM, count: n - 1)
    let units = remaining / R
    for i in 0..<units { overlaps[i % overlaps.count] += R }
    var starts = [Int](repeating: 0, count: n)
    starts[0] = 0
    for i in 0..<(n - 1) { starts[i + 1] = starts[i] + T - overlaps[i] }
    return (starts, overlaps)
}

/// 时间解码规划：与 Zig planTemporal 完全一致。
public struct H3TemporalPlan {
    public let padTokens: Int
    public let numChunks: Int
    public let outputFrames: Int
    public let paddedLen: Int
}

public func h3PlanTemporal(_ latentT: Int) -> H3TemporalPlan {
    let CS = H3VAEConst.tokensChunkSize, TD = H3VAEConst.tokenDrop
    var pseudo = latentT + TD
    var pad = 0
    let rem = pseudo % CS
    if rem != 0 {
        pad = CS - rem
        pseudo += pad
    }
    var numChunks = pseudo / CS - (TD > 0 ? 1 : 0)
    if numChunks < 1 {
        pad += CS
        numChunks += 1
    }
    let paddedLen = latentT + pad
    return H3TemporalPlan(padTokens: pad, numChunks: numChunks,
                          outputFrames: h3FramePlan(zLen: paddedLen, numChunks: numChunks, padTokens: pad),
                          paddedLen: paddedLen)
}

func h3FramePlan(zLen: Int, numChunks: Int, padTokens: Int) -> Int {
    let CS = H3VAEConst.tokensChunkSize, RT = H3VAEConst.vaeRatioT, TD = H3VAEConst.tokenDrop
    let chunkDec = CS * RT
    let splitCount = (TD > 0 ? 1 : 0) + 1
    var total = 0
    var finalOverlap = 0
    for i in 0..<numChunks {
        let tStart = i * CS
        let tEnd = min(tStart + CS + H3VAEConst.tokenOverlap, zLen)
        let clipTokens = min(tEnd, zLen) - min(tStart, zLen)
        let clipFrames = clipTokens * RT
        for j in 0..<splitCount {
            let fStart = j * chunkDec
            let fEnd = min(fStart + chunkDec, clipFrames)
            let frames = max(fEnd - fStart - H3VAEConst.framePrePadding, 0)
            if j == 0 { total += frames } else { finalOverlap = frames }
        }
    }
    total += finalOverlap
    return total - h3PadFrames(zLen: zLen, padTokens: padTokens)
}

func h3PadFrames(zLen: Int, padTokens: Int) -> Int {
    if padTokens == 0 { return 0 }
    let CL = H3VAEConst.clipLength, RT = H3VAEConst.vaeRatioT
    let intraTail = CL % RT
    if intraTail == 0 { return padTokens * RT }
    let before = zLen - padTokens
    var sum = 0
    for k in 0..<padTokens {
        sum += (before + k) % H3VAEConst.tokensChunkSize == 0 ? intraTail : RT
    }
    return sum
}

/// 线性交叉淡入：a 的尾部与 b 的头部沿 axis 混合 extent 长度。
public func h3BlendAxis(_ a: MLXArray, _ b: MLXArray, extent: Int, axis: Int, dtype: DType) -> MLXArray {
    let an = a.shape[axis], bn = b.shape[axis]
    let e = min(min(an, bn), extent)
    if e == 0 { return b }
    var wbuf = [Float](repeating: 0, count: e)
    for i in 0..<e { wbuf[i] = Float(i) / Float(e) }
    var wshape = [1, 1, 1, 1, 1]
    wshape[axis] = e
    let wb = MLXArray(wbuf, wshape).asType(dtype)
    let wa = (1.0 - wb)
    let aTail = h3SliceAxis(a, an - e, an, axis: axis)
    let bHead = h3SliceAxis(b, 0, e, axis: axis)
    let mixed = aTail * wa + bHead * wb
    if bn == e { return mixed }
    let rest = h3SliceAxis(b, e, bn, axis: axis)
    return MLX.concatenated([mixed, rest], axis: axis)
}

// MARK: - Encoder 权重块

public final class H3EncResBlock {
    public var norm1W: MLXArray, norm1B: MLXArray
    public var conv1W: MLXArray, conv1B: MLXArray
    public var norm2W: MLXArray, norm2B: MLXArray
    public var conv2W: MLXArray, conv2B: MLXArray
    /// 1x1 shortcut [in, out]；in == out 时为 nil。
    public var ninW: MLXArray?
    public var ninB: MLXArray?

    public init(norm1W: MLXArray, norm1B: MLXArray, conv1W: MLXArray, conv1B: MLXArray,
                norm2W: MLXArray, norm2B: MLXArray, conv2W: MLXArray, conv2B: MLXArray,
                ninW: MLXArray? = nil, ninB: MLXArray? = nil) {
        self.norm1W = norm1W; self.norm1B = norm1B
        self.conv1W = conv1W; self.conv1B = conv1B
        self.norm2W = norm2W; self.norm2B = norm2B
        self.conv2W = conv2W; self.conv2B = conv2B
        self.ninW = ninW; self.ninB = ninB
    }
}

public final class H3EncLevel {
    public var blocks: [H3EncResBlock]
    public var downW: MLXArray?
    public var downB: MLXArray?
    public init(blocks: [H3EncResBlock], downW: MLXArray? = nil, downB: MLXArray? = nil) {
        self.blocks = blocks; self.downW = downW; self.downB = downB
    }
}

// MARK: - Encoder

public final class H3VAEEncoder {
    public let convInW: MLXArray
    public let convInB: MLXArray
    public let levels: [H3EncLevel]
    public let normOutW: MLXArray
    public let normOutB: MLXArray
    public let convOutW: MLXArray
    public let convOutB: MLXArray
    /// [24, 48] 预转置（通道轴线性）。
    public let quantWt: MLXArray
    public let quantB: MLXArray
    public let latentsMean: MLXArray   // f32 [24]
    public let latentsStd: MLXArray    // f32 [24]

    public init(convInW: MLXArray, convInB: MLXArray, levels: [H3EncLevel],
                normOutW: MLXArray, normOutB: MLXArray, convOutW: MLXArray, convOutB: MLXArray,
                quantWt: MLXArray, quantB: MLXArray, latentsMean: MLXArray, latentsStd: MLXArray) {
        self.convInW = convInW; self.convInB = convInB
        self.levels = levels
        self.normOutW = normOutW; self.normOutB = normOutB
        self.convOutW = convOutW; self.convOutB = convOutB
        self.quantWt = quantWt; self.quantB = quantB
        self.latentsMean = latentsMean; self.latentsStd = latentsStd
    }

    /// 全部权重/统计量数组（load 末尾一次性 eval 物化，避免 lazy 权重树悬垂）。
    func allWeightArrays() -> [MLXArray] {
        var a: [MLXArray] = [convInW, convInB, normOutW, normOutB, convOutW, convOutB,
                             quantWt, quantB, latentsMean, latentsStd]
        for lv in levels {
            for rb in lv.blocks {
                a.append(contentsOf: [rb.norm1W, rb.norm1B, rb.conv1W, rb.conv1B,
                                      rb.norm2W, rb.norm2B, rb.conv2W, rb.conv2B])
                if let w = rb.ninW { a.append(w) }
                if let b = rb.ninB { a.append(b) }
            }
            if let w = lv.downW { a.append(w) }
            if let b = lv.downB { a.append(b) }
        }
        return a
    }

    public static func load(_ w: H3Weights) throws -> H3VAEEncoder {
        func tap(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return h3ConvTap(raw)
        }
        func vec(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return raw.asType(.float32)
        }
        func o1o(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return h3OneByOne(raw)
        }

        let convInW = try tap("encoder.conv_in.weight")
        let convInB = try vec("encoder.conv_in.bias")

        var levels: [H3EncLevel] = []
        levels.reserveCapacity(H3VAEConst.encLevels)
        for lv in 0..<H3VAEConst.encLevels {
            var blocks: [H3EncResBlock] = []
            blocks.reserveCapacity(H3VAEConst.encResBlocks)
            for bi in 0..<H3VAEConst.encResBlocks {
                let inCh = bi == 0 ? H3VAEConst.encBlockIn(lv) : H3VAEConst.encBlockMid(lv)
                var blk = H3EncResBlock(
                    norm1W: try vec("encoder.down.\(lv).block.\(bi).norm1.weight"),
                    norm1B: try vec("encoder.down.\(lv).block.\(bi).norm1.bias"),
                    conv1W: try tap("encoder.down.\(lv).block.\(bi).conv1.weight"),
                    conv1B: try vec("encoder.down.\(lv).block.\(bi).conv1.bias"),
                    norm2W: try vec("encoder.down.\(lv).block.\(bi).norm2.weight"),
                    norm2B: try vec("encoder.down.\(lv).block.\(bi).norm2.bias"),
                    conv2W: try tap("encoder.down.\(lv).block.\(bi).conv2.weight"),
                    conv2B: try vec("encoder.down.\(lv).block.\(bi).conv2.bias"))
                if inCh != H3VAEConst.encBlockMid(lv) {
                    blk.ninW = try o1o("encoder.down.\(lv).block.\(bi).nin_shortcut.weight")
                    blk.ninB = try vec("encoder.down.\(lv).block.\(bi).nin_shortcut.bias")
                }
                blocks.append(blk)
            }
            var level = H3EncLevel(blocks: blocks)
            if H3VAEConst.encHasDown(lv) {
                level.downW = try tap("encoder.down.\(lv).downsample.conv.weight")
                level.downB = try vec("encoder.down.\(lv).downsample.conv.bias")
            }
            levels.append(level)
        }

        let encoder = H3VAEEncoder(
            convInW: convInW, convInB: convInB, levels: levels,
            normOutW: try vec("encoder.norm_out.weight"),
            normOutB: try vec("encoder.norm_out.bias"),
            convOutW: try tap("encoder.conv_out.weight"),
            convOutB: try vec("encoder.conv_out.bias"),
            quantWt: try o1o("quant_conv.weight"),
            quantB: try vec("quant_conv.bias"),
            latentsMean: try vec("latents_mean"),
            latentsStd: try vec("latents_std"))
        // 一次性物化所有权重：避免首次大图提交时 lazy 权重树悬垂触发 Metal use-after-free。
        MLX.eval(encoder.allWeightArrays())
        return encoder
    }

    /// One CausalConv3d：空间 reflect pad 1 + 时间前向零 pad 2 + conv3d stride 1。
    func causalConv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
        let sp = h3ReflectPad1HW(x)
        let tp = h3CausalPadFront(sp, 2)
        return h3Conv3dBias(tp, w, b)
    }

    func resBlock(_ blk: H3EncResBlock, _ x: MLXArray) -> MLXArray {
        let n1 = h3GroupNorm32(x, blk.norm1W, blk.norm1B)
        let a1 = silu(n1)
        let h1 = causalConv(a1, blk.conv1W, blk.conv1B)
        let n2 = h3GroupNorm32(h1, blk.norm2W, blk.norm2B)
        let a2 = silu(n2)
        let h2 = causalConv(a2, blk.conv2W, blk.conv2B)
        let sc: MLXArray
        if let ninW = blk.ninW {
            sc = H3TensorOps.denseLinear(x, ninW, blk.ninB)
        } else {
            sc = x
        }
        return h2 + sc
    }

    /// 单 tile [1, T, th, tw, 3]（已像素归一化）→ moments [1, 48, T_lat, th/16, tw/16]。
    func encodeMoments(_ x: MLXArray) -> MLXArray {
        var h = causalConv(x, convInW, convInB)
        for lv in 0..<H3VAEConst.encLevels {
            let level = levels[lv]
            for bi in 0..<H3VAEConst.encResBlocks {
                h = resBlock(level.blocks[bi], h)
            }
            if let downW = level.downW, let downB = level.downB {
                let pd = H3VAEConst.encSpaceDown[lv] == 2 ? h3ReflectPadBR(h) : h
                let tp = h3CausalPadFront(pd, 2)
                h = h3Conv3dBias(tp, downW, downB,
                                 timeStride: H3VAEConst.encTimeDown[lv],
                                 spaceStride: H3VAEConst.encSpaceDown[lv])
            }
        }
        h = h3GroupNorm32(h, normOutW, normOutB)
        h = silu(h)
        h = causalConv(h, convOutW, convOutB)
        h = H3TensorOps.denseLinear(h, quantWt, quantB)
        // [1, T_lat, lh, lw, 48] → [1, 48, T_lat, lh, lw]
        return h.transposed(0, 4, 1, 2, 3)
    }

    /// 单帧 keyframe [1,3,1,H,W]（[-1,1]）→ 归一化 latent [1,24,1,H/16,W/16]。
    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        encodeVideo(pixels)
    }

    /// 视频 [1,3,T,H,W]（[-1,1]）→ 归一化 latent [1,24,T_lat,H/16,W/16]。
    /// T==1 是 fl2va keyframe；T>1 按 17 帧 clip 编码，尾部 repeat-pad，丢弃 TOKEN_DROP 帧。
    public func encodeVideo(_ pixels: MLXArray) -> MLXArray {
        let shp = pixels.shape
        let pt = shp[2], ph = shp[3], pw = shp[4]

        // [1,3,T,H,W] → [1,T,H,W,3]；[-1,1] → ImageNet 归一化。
        let t5 = pixels.transposed(0, 2, 3, 4, 1).asType(.float32)
        let unit01 = t5 * 0.5 + 0.5
        let pm = MLXArray(H3VAEConst.pixelMean, [1, 1, 1, 1, 3])
        let ps = MLXArray(H3VAEConst.pixelStd, [1, 1, 1, 1, 3])
        let nthwc = (unit01 - pm) / ps

        let moments = pt == 1 ? adaptiveEncode(nthwc, ph: ph, pw: pw)
                              : encodeTemporal(nthwc, pt: pt, ph: ph, pw: pw)

        // mean = 前 24 通道；按 latent 统计量归一化。
        let mean24 = h3SliceAxis(moments, 0, 24, axis: 1)
        let lm = latentsMean.reshaped(1, 24, 1, 1, 1)
        let ls = latentsStd.reshaped(1, 24, 1, 1, 1)
        return (mean24 - lm) / ls
    }

    func adaptiveEncode(_ x: MLXArray, ph: Int, pw: Int) -> MLXArray {
        if H3VAEConst.fitsSingleTile(ph) && H3VAEConst.fitsSingleTile(pw) {
            return encodeMoments(x)
        }
        return encodeTiled(x)
    }

    /// 17 帧 clip 编码（VAE 训练片段长度）。
    func encodeTemporal(_ x: MLXArray, pt: Int, ph: Int, pw: Int) -> MLXArray {
        let CL = H3VAEConst.clipLength
        var padded = x
        var total = pt
        let rem = pt % CL
        if rem != 0 {
            let padN = CL - rem
            let last = h3SliceAxis(padded, pt - 1, pt, axis: 1)
            var parts: [MLXArray] = [padded]
            parts.append(contentsOf: [MLXArray](repeating: last, count: padN))
            padded = MLX.concatenated(parts, axis: 1)
            total = pt + padN
        }
        let chunks = total / CL
        var outs: [MLXArray] = []
        outs.reserveCapacity(chunks)
        for i in 0..<chunks {
            let lo = i * CL
            let clip = h3SliceAxis(padded, lo, lo + CL, axis: 1)
            outs.append(adaptiveEncode(clip, ph: ph, pw: pw))
        }
        let cat = chunks == 1 ? outs[0] : MLX.concatenated(outs, axis: 2)
        let nLat = cat.shape[2]
        return h3SliceAxis(cat, 0, nLat - H3VAEConst.tokenDrop, axis: 2)
    }

    /// tiled encode：逐 tile 编码，在 latent 粒度与 raw 邻居 blend，裁剪组装。
    func encodeTiled(_ nhwc: MLXArray) -> MLXArray {
        let ph = nhwc.shape[2], pw = nhwc.shape[3]
        let yp = h3SplitTiles(ph)
        let xp = h3SplitTiles(pw)
        let ny = yp.starts.count, nx = xp.starts.count

        var raw: [MLXArray] = []
        raw.reserveCapacity(ny * nx)
        for (i, y0) in yp.starts.enumerated() {
            for (j, x0) in xp.starts.enumerated() {
                // Input [1, T, H, W, C]：空间轴 2/3。tile 长度 TILE_SIZE，尾块越界由
                // slice clamp（与 Zig 一致）。
                let th = h3SliceAxis(nhwc, y0, min(y0 + H3VAEConst.tileSize, ph), axis: 2)
                _ = i
                let tile = h3SliceAxis(th, x0, min(x0 + H3VAEConst.tileSize, pw), axis: 3)
                raw.append(encodeMoments(tile))
                _ = j
            }
        }
        // 物化全部 tile moments：raw 是局部数组，若保持 lazy，函数返回后中间
        // buffer 被释放，外部 eval 提交 GPU 命令时引用悬垂对象 → preCommit UAF。
        MLX.eval(raw)

        let R = H3VAEConst.vaeRatio
        var rows: [MLXArray] = []
        rows.reserveCapacity(ny)
        for i in 0..<ny {
            var cols: [MLXArray] = []
            cols.reserveCapacity(nx)
            for j in 0..<nx {
                var tile = raw[i * nx + j]
                if i > 0 {
                    tile = h3BlendAxis(raw[(i - 1) * nx + j], tile,
                                       extent: yp.overlaps[i - 1] / R, axis: 3, dtype: .float32)
                }
                if j > 0 {
                    tile = h3BlendAxis(raw[i * nx + (j - 1)], tile,
                                       extent: xp.overlaps[j - 1] / R, axis: 4, dtype: .float32)
                }
                if i + 1 < ny {
                    let h2 = tile.shape[3]
                    tile = h3SliceAxis(tile, 0, h2 - yp.overlaps[i] / R, axis: 3)
                }
                if j + 1 < nx {
                    let w2 = tile.shape[4]
                    tile = h3SliceAxis(tile, 0, w2 - xp.overlaps[j] / R, axis: 4)
                }
                cols.append(tile)
            }
            rows.append(MLX.concatenated(cols, axis: 4))
        }
        return MLX.concatenated(rows, axis: 3)
    }
}

// MARK: - Decoder 权重块

public final class H3DecBlock {
    public var norm1W: MLXArray, norm2W: MLXArray
    public var scale1: MLXArray, scale2: MLXArray
    public var qkvW: MLXArray, qkvB: MLXArray
    public var outW: MLXArray, outB: MLXArray
    public var w1W: MLXArray, w1B: MLXArray
    public var w2W: MLXArray, w2B: MLXArray

    public init(norm1W: MLXArray, norm2W: MLXArray, scale1: MLXArray, scale2: MLXArray,
                qkvW: MLXArray, qkvB: MLXArray, outW: MLXArray, outB: MLXArray,
                w1W: MLXArray, w1B: MLXArray, w2W: MLXArray, w2B: MLXArray) {
        self.norm1W = norm1W; self.norm2W = norm2W
        self.scale1 = scale1; self.scale2 = scale2
        self.qkvW = qkvW; self.qkvB = qkvB
        self.outW = outW; self.outB = outB
        self.w1W = w1W; self.w1B = w1B
        self.w2W = w2W; self.w2B = w2B
    }
}

// MARK: - Decoder

public final class H3VAEDecoder {
    /// 解码线性层是否走 bf16 快速 matmul（权重/激活加载即 bf16，跳过 denseLinear 逐层 fp32 cast）。
    /// 默认开启；设 NA_H3_VAE_DEC_BF16=0 或 NA_H3_VAE_DEC_F32=1 时回退 fp32 dense island（与参考实现完全一致）。
    public static let fastBF16Linear: Bool = {
        if ProcessInfo.processInfo.environment["NA_H3_VAE_DEC_F32"] == "1" { return false }
        if ProcessInfo.processInfo.environment["NA_H3_VAE_DEC_BF16"] == "0" { return false }
        return true
    }()

    public let dtype: DType
    public let xEmbedW: MLXArray   // [dim, dim]
    public let xEmbedB: MLXArray
    public let registerTokens: MLXArray
    public let blocks: [H3DecBlock]
    public let normOutW: MLXArray
    public let normOutB: MLXArray
    public let projOutW: MLXArray  // [dim, 3072]
    public let projOutB: MLXArray
    public let pqW: MLXArray       // [24, 24]（post_quant_conv）
    public let pqB: MLXArray
    public let latentsMean: MLXArray  // f32 [24]
    public let latentsStd: MLXArray   // f32 [24]
    /// rope 几何缓存：同一 (t,hh,ww,nSuffix) 的 tokenIds/cos/sin 复用，避免每 tile 重建。
    private var ropeMemo: [String: (cos: MLXArray, sin: MLXArray)] = [:]
    /// register_tokens + 1 个零 token 的预拼接（与 decodePixels 拼接顺序一致）。
    private lazy var suffixTokens: MLXArray = {
        let zeros = MLXArray.zeros([1, 1, H3VAEConst.decDim], dtype: dtype)
        return MLX.concatenated([registerTokens, zeros], axis: 1)
    }()

    /// 解码线性层：bf16 快速路径（权重/激活保持 dtype，不升 fp32）；回退时与 denseLinear fp32 island 一致。
    func linDense(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray?) -> MLXArray {
        if Self.fastBF16Linear {
            var out = x.matmul(w)
            if let b { out = out + b }
            return out
        }
        return H3TensorOps.denseLinear(x, w, b)
    }

    /// rope 按几何取缓存；未命中时构造后缓存。
    func ropeFor(t: Int, h: Int, wd: Int, nSuffix: Int, nRows: Int) -> (cos: MLXArray, sin: MLXArray) {
        let key = "\(t)-\(h)-\(wd)-\(nSuffix)"
        if let hit = ropeMemo[key] { return hit }
        let ids = tokenIds(t: t, h: h, wd: wd, nSuffix: nSuffix)
        let rope = buildRope(ids: ids, nRows: nRows)
        ropeMemo[key] = rope
        return rope
    }

    public init(dtype: DType, xEmbedW: MLXArray, xEmbedB: MLXArray, registerTokens: MLXArray,
                blocks: [H3DecBlock], normOutW: MLXArray, normOutB: MLXArray,
                projOutW: MLXArray, projOutB: MLXArray, pqW: MLXArray, pqB: MLXArray,
                latentsMean: MLXArray, latentsStd: MLXArray) {
        self.dtype = dtype
        self.xEmbedW = xEmbedW; self.xEmbedB = xEmbedB
        self.registerTokens = registerTokens
        self.blocks = blocks
        self.normOutW = normOutW; self.normOutB = normOutB
        self.projOutW = projOutW; self.projOutB = projOutB
        self.pqW = pqW; self.pqB = pqB
        self.latentsMean = latentsMean; self.latentsStd = latentsStd
    }

    /// 全部权重/统计量数组（load 末尾一次性 eval 物化，避免 lazy 权重树悬垂）。
    func allWeightArrays() -> [MLXArray] {
        var a: [MLXArray] = [xEmbedW, xEmbedB, registerTokens, normOutW, normOutB,
                             projOutW, projOutB, pqW, pqB, latentsMean, latentsStd]
        for db in blocks {
            a.append(contentsOf: [db.norm1W, db.norm2W, db.scale1, db.scale2,
                                  db.qkvW, db.qkvB, db.outW, db.outB,
                                  db.w1W, db.w1B, db.w2W, db.w2B])
        }
        return a
    }

    public static func load(_ w: H3Weights, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3VAEDecoder {
        func linT(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return h3LoadLinT(raw, dtype: dtype)
        }
        func own(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return raw.asType(dtype)
        }
        func ownF32(_ key: String) throws -> MLXArray {
            guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
            return raw.asType(.float32)
        }

        let xEmbedW = try linT("decoder.x_embedder.weight")
        let xEmbedB = try own("decoder.x_embedder.bias")
        let registerTokens = try own("decoder.register_tokens")
        let normOutW = try own("decoder.norm_out.weight")
        let normOutB = try own("decoder.norm_out.bias")
        let projOutW = try linT("decoder.proj_out.weight")
        let projOutB = try own("decoder.proj_out.bias")

        // post_quant_conv [24,24,1,1,1] → [24,24] matmul（通道轴）。
        let pqRaw = try own("post_quant_conv.weight")
        let pq2 = pqRaw.reshaped(24, 24)
        let pqW = pq2.transposed(1, 0).asType(dtype)

        var blocks: [H3DecBlock] = []
        blocks.reserveCapacity(H3VAEConst.decLayers)
        for i in 0..<H3VAEConst.decLayers {
            let p = "decoder.transformer_blocks.\(i)"
            blocks.append(H3DecBlock(
                norm1W: try own("\(p).norm1.weight"),
                norm2W: try own("\(p).norm2.weight"),
                scale1: try own("\(p).scale1"),
                scale2: try own("\(p).scale2"),
                qkvW: try linT("\(p).attn.to_qkv.weight"),
                qkvB: try own("\(p).attn.to_qkv.bias"),
                outW: try linT("\(p).attn.to_out.weight"),
                outB: try own("\(p).attn.to_out.bias"),
                w1W: try linT("\(p).ff.w1.weight"),
                w1B: try own("\(p).ff.w1.bias"),
                w2W: try linT("\(p).ff.w2.weight"),
                w2B: try own("\(p).ff.w2.bias")))
        }

        let decoder = H3VAEDecoder(
            dtype: dtype, xEmbedW: xEmbedW, xEmbedB: xEmbedB, registerTokens: registerTokens,
            blocks: blocks, normOutW: normOutW, normOutB: normOutB,
            projOutW: projOutW, projOutB: projOutB,
            pqW: pqW, pqB: try own("post_quant_conv.bias"),
            latentsMean: try ownF32("latents_mean"), latentsStd: try ownF32("latents_std"))
        // 一次性物化所有权重：避免首次大图提交时 lazy 权重树悬垂触发 Metal use-after-free。
        MLX.eval(decoder.allWeightArrays())
        return decoder
    }

    /// 归一化坐标：每个轴 (arange(0.5,n)/n)*2-1，suffix tokens 全零。
    func tokenIds(t: Int, h: Int, wd: Int, nSuffix: Int) -> [Float] {
        let n = t * h * wd
        var out = [Float](repeating: 0, count: (n + nSuffix) * 3)
        var i = 0
        for ti in 0..<t {
            let tv = (Float(ti) + 0.5) / Float(t) * 2 - 1
            for hi in 0..<h {
                let hv = (Float(hi) + 0.5) / Float(h) * 2 - 1
                for wi in 0..<wd {
                    let wv = (Float(wi) + 0.5) / Float(wd) * 2 - 1
                    out[i * 3 + 0] = tv
                    out[i * 3 + 1] = hv
                    out[i * 3 + 2] = wv
                    i += 1
                }
            }
        }
        return out
    }

    /// 由位置 ids 构建 rope cos/sin [1, nRows, 1, 48]。
    func buildRope(ids: [Float], nRows: Int) -> (cos: MLXArray, sin: MLXArray) {
        let nf = H3VAEConst.decRopeFreqs
        let half = nf * 3
        var ang = [Float](repeating: 0, count: nRows * half)
        let step = 2.0 * 3.0 / (Double(H3VAEConst.decHeadDim) * H3VAEConst.decRopeDimRatio)
        for r in 0..<nRows {
            for ax in 0..<3 {
                let p = Double(ids[r * 3 + ax])
                for j in 0..<nf {
                    let inv = 1.0 / pow(H3VAEConst.decRopeTheta, Double(j) * step)
                    ang[r * half + ax * nf + j] = Float(2.0 * Double.pi * p * inv)
                }
            }
        }
        let arr = MLXArray(ang, [nRows, half])
        let c = MLX.cos(arr)
        let sn = MLX.sin(arr)
        let cb = c.reshaped(1, nRows, 1, half).asType(dtype)
        let sb = sn.reshaped(1, nRows, 1, half).asType(dtype)
        return (cb, sb)
    }

    /// 单 block：x += attn(rms(x))·scale1；x += ff(rms(x))·scale2。
    func blockForward(_ b: H3DecBlock, _ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.shape[1]
        let heads = H3VAEConst.decHeads
        let hd = H3VAEConst.decHeadDim
        let eps = H3VAEConst.decEps

        let n1 = rmsNormLast(x, weight: b.norm1W, eps: eps)
        let qkv = linDense(n1, b.qkvW, b.qkvB).asType(dtype)
        // [1, n, heads, 3*hd] 切 LAST 轴（per-head interleave，与 DiT 不同）。
        let v4 = qkv.reshaped(1, n, heads, 3 * hd)
        let parts = H3TensorOps.splitEqual(v4, 3, axis: 3)

        var t: [MLXArray] = []
        t.reserveCapacity(3)
        for i in 0..<3 {
            var cur = parts[i]
            if i < 2 {
                // norm_q/norm_k 无 affine：纯 RMS。
                let xf = cur.asType(.float32)
                let variance = xf.square().mean(axis: -1, keepDims: true)
                cur = (xf * (variance + MLXArray(eps)).rsqrt()).asType(cur.dtype)
                cur = applyRopePub(cur, cos: cos, sin: sin, rot: H3VAEConst.decRotDim)
            }
            let tr = cur.transposed(0, 2, 1, 3)
            t.append(tr)
        }
        let scale: Float = 1.0 / sqrt(Float(hd))
        let attn = scaledDotProductAttention(queries: t[0], keys: t[1], values: t[2], scale: scale, mask: nil)
        let at = attn.transposed(0, 2, 1, 3)
        let af = at.reshaped(1, n, heads * hd)
        let ao = linDense(af, b.outW, b.outB).asType(dtype)
        let h1 = x + ao * b.scale1

        let n2 = rmsNormLast(h1, weight: b.norm2W, eps: eps)
        let y = linDense(n2, b.w1W, b.w1B).asType(dtype)
        let halves = H3TensorOps.splitEqual(y, 2, axis: 2)
        let ff = linDense(silu(halves[0]) * halves[1], b.w2W, b.w2B).asType(dtype)
        return h1 + ff * b.scale2
    }

    /// 单次 untiled ViT：latent [1,C,t,h,w] → pixels [1,3,t*4,h*16,w*16]。
    public func decodePixels(_ z: MLXArray) -> MLXArray {
        let shp = z.shape
        let t = shp[2], hh = shp[3], ww = shp[4]
        let nPatches = t * hh * ww
        let nSuffix = 1 + H3VAEConst.decRegisterTokens

        // [1,C,t,h,w] → [1, t*h*w, C]
        let flat = z.reshaped(1, H3VAEConst.decInChannels, nPatches)
        let trc = flat.transposed(0, 2, 1).asType(dtype)
        let h0 = linDense(trc, xEmbedW, xEmbedB).asType(dtype)

        // register tokens + 1 zero token（位置 id 全零）。
        var h = MLX.concatenated([h0, suffixTokens], axis: 1)

        let nRows = nPatches + nSuffix
        let rope = ropeFor(t: t, h: hh, wd: ww, nSuffix: nSuffix, nRows: nRows)

        for blk in blocks {
            h = blockForward(blk, h, cos: rope.cos, sin: rope.sin)
        }

        let no = layerNormLast(h, weight: normOutW, bias: normOutB, eps: H3VAEConst.decEps)
        let po = linDense(no, projOutW, projOutB).asType(dtype)
        let kept = h3SliceAxis(po, 0, nPatches, axis: 1)

        // [1, t*h*w, C*pt*ph*pw] → [1, C, t*pt, h*ph, w*pw]
        let pt = H3VAEConst.vaeRatioT
        let ps = H3VAEConst.vaeRatio
        let oc = H3VAEConst.decOutChannels
        let v = kept.reshaped(1, t, hh, ww, oc, pt, ps, ps)
        let perm = v.transposed(0, 4, 1, 5, 2, 6, 3, 7)
        return perm.reshaped(1, oc, t * pt, hh * ps, ww * ps)
    }

    /// 单 latent chunk → 像素，画布超过 256px 时 spatial tiled + blend。
    func decodeSpatial(_ z: MLXArray) -> MLXArray {
        let shp = z.shape
        let latH = shp[3], latW = shp[4]
        let pxH = latH * H3VAEConst.vaeRatio
        let pxW = latW * H3VAEConst.vaeRatio
        if H3VAEConst.fitsSingleTile(pxH) && H3VAEConst.fitsSingleTile(pxW) {
            return decodePixels(z)
        }

        let yp = h3SplitTiles(pxH)
        let xp = h3SplitTiles(pxW)
        let R = H3VAEConst.vaeRatio

        // 上一行各列的未 blend 底条（先于本行 blend 捕获）。
        var rowTails: [MLXArray?] = [MLXArray?](repeating: nil, count: xp.starts.count)

        var rows: [MLXArray] = []
        rows.reserveCapacity(yp.starts.count)
        for (i, y0) in yp.starts.enumerated() {
            let zi = y0 / R
            let zl = yp.starts[0] == 0 ? min(yp.starts[0] + H3VAEConst.tileSize, pxH) / R : H3VAEConst.tileSize / R
            _ = zl
            let zLenH = (yp.starts[0] + H3VAEConst.tileSize) / R
            var newTails: [MLXArray?] = [MLXArray?](repeating: nil, count: xp.starts.count)
            var cols: [MLXArray] = []
            cols.reserveCapacity(xp.starts.count)
            var leftTail: MLXArray?

            for (j, x0) in xp.starts.enumerated() {
                let zj = x0 / R
                let zLenW = (xp.starts[0] + H3VAEConst.tileSize) / R
                let subH = h3SliceAxis(z, zi, min(zi + zLenH, z.shape[3]), axis: 3)
                let sub = h3SliceAxis(subH, zj, min(zj + zLenW, z.shape[4]), axis: 4)
                var tile = decodePixels(sub)

                let th = tile.shape[3]
                let tw = tile.shape[4]
                if i + 1 < yp.starts.count {
                    newTails[j] = h3SliceAxis(tile, th - yp.overlaps[i], th, axis: 3)
                }
                var nextLeft: MLXArray? = nil
                if j + 1 < xp.starts.count {
                    nextLeft = h3SliceAxis(tile, tw - xp.overlaps[j], tw, axis: 4)
                }
                if i > 0, let prev = rowTails[j] {
                    tile = h3BlendAxis(prev, tile, extent: yp.overlaps[i - 1], axis: 3, dtype: dtype)
                }
                if j > 0, let prev = leftTail {
                    tile = h3BlendAxis(prev, tile, extent: xp.overlaps[j - 1], axis: 4, dtype: dtype)
                }
                leftTail = nextLeft
                if i + 1 < yp.starts.count {
                    let h2 = tile.shape[3]
                    tile = h3SliceAxis(tile, 0, h2 - yp.overlaps[i], axis: 3)
                }
                if j + 1 < xp.starts.count {
                    let w2 = tile.shape[4]
                    tile = h3SliceAxis(tile, 0, w2 - xp.overlaps[j], axis: 4)
                }
                cols.append(tile)
            }
            rows.append(MLX.concatenated(cols, axis: 4))
            rowTails = newTails
        }
        return MLX.concatenated(rows, axis: 3)
    }

    /// 完整解码：归一化 latent [1,24,T,H,W] → pixels [1,3,frames,H*16,W*16] in [-1,1]。
    public func decode(_ zNorm: MLXArray) -> MLXArray {
        let shp = zNorm.shape
        let latentT = shp[2], latH = shp[3], latW = shp[4]

        // 反归一化 + post_quant_conv（通道轴）。
        let zf = zNorm.asType(.float32)
        let lm = latentsMean.reshaped(1, H3VAEConst.decInChannels, 1, 1, 1)
        let ls = latentsStd.reshaped(1, H3VAEConst.decInChannels, 1, 1, 1)
        let zden = zf * ls + lm
        let nAll = latentT * latH * latW
        let zflat = zden.reshaped(1, H3VAEConst.decInChannels, nAll)
        let ztrc = zflat.transposed(0, 2, 1).asType(dtype)
        let mixed = linDense(ztrc, pqW, pqB).asType(dtype)
        let backc = mixed.transposed(0, 2, 1)
        var z = backc.reshaped(1, H3VAEConst.decInChannels, latentT, latH, latW)

        let plan = h3PlanTemporal(latentT)
        if plan.padTokens > 0 {
            let last = h3SliceAxis(z, latentT - 1, latentT, axis: 2)
            var pieces: [MLXArray] = [z]
            pieces.append(contentsOf: [MLXArray](repeating: last, count: plan.padTokens))
            z = MLX.concatenated(pieces, axis: 2)
        }

        let chunkDec = H3VAEConst.tokensChunkSize * H3VAEConst.vaeRatioT
        let splitCount = (H3VAEConst.tokenDrop > 0 ? 1 : 0) + 1
        var parts: [MLXArray] = []
        var overlap: MLXArray?

        let zLen = plan.paddedLen
        for i in 0..<plan.numChunks {
            let tStart = i * H3VAEConst.tokensChunkSize
            let tEnd = min(tStart + H3VAEConst.tokensChunkSize + H3VAEConst.tokenOverlap, zLen)
            if tStart >= tEnd { continue }
            let clipZ = h3SliceAxis(z, tStart, tEnd, axis: 2)
            let clipDec = decodeSpatial(clipZ)
            // 逐 chunk 物化：避免整段视频所有 tile 的 36 层 lazy 图叠加到最终一次性 eval，
            // 显著降低峰值内存；后续 slice/blend 只做轻量视图运算。
            MLX.eval(clipDec)
            let clipFrames = clipDec.shape[2]

            for j in 0..<splitCount {
                let fStart = j * chunkDec
                if fStart >= clipFrames { continue }
                let fEnd = min(fStart + chunkDec, clipFrames)
                if fEnd - fStart <= H3VAEConst.framePrePadding { continue }
                let piece = h3SliceAxis(clipDec, fStart + H3VAEConst.framePrePadding, fEnd, axis: 2)
                if j == 0 {
                    if let o = overlap {
                        overlap = nil
                        parts.append(h3BlendAxis(o, piece, extent: H3VAEConst.frameOverlap, axis: 2, dtype: dtype))
                    } else {
                        parts.append(piece)
                    }
                } else {
                    overlap = piece
                }
            }
            if i == plan.numChunks - 1 {
                if let o = overlap {
                    parts.append(o)
                    overlap = nil
                }
            }
        }
        if parts.isEmpty { fatalError("H3VAEDecoder.decode: empty temporal output") }

        var decAll = MLX.concatenated(parts, axis: 2)
        let have = decAll.shape[2]
        if have > plan.outputFrames {
            decAll = h3SliceAxis(decAll, 0, plan.outputFrames, axis: 2)
        }

        // 像素反归一化：ImageNet 统计，clamp [0,1]，再 [-1,1]。
        let df = decAll.asType(.float32)
        let pm = MLXArray(H3VAEConst.imagenetMean, [1, 3, 1, 1, 1])
        let ps = MLXArray(H3VAEConst.imagenetStd, [1, 3, 1, 1, 1])
        let a1 = df * ps + pm
        let cl = MLX.minimum(MLX.maximum(a1, MLXArray.scalar(0.0, like: a1)), MLXArray.scalar(1.0, like: a1))
        let out = cl * 2.0 - 1.0
        // 返回边界物化（与 encodeTiled 的 `MLX.eval(raw)` 同款纪律）：parts / decAll / df / a1 / cl
        // 全是本函数局部量，末尾这段懒图若直接交给调用方，则函数返回后局部中间 buffer 已无宿主，
        // 外部 eval 提交 GPU 命令时才引用的对象可能已被分配器回收 → preCommit UAF。
        MLX.eval(out)
        return out
    }
}

// MARK: - 聚合入口

/// MiniMax-H3 视频 VAE：Encoder.encodeImage + Decoder.decode。
/// 权重键名前缀 encoder.* / decoder.* / quant_conv / post_quant_conv / latents_*。
public final class H3VAE {
    public let encoder: H3VAEEncoder
    public let decoder: H3VAEDecoder
    public let dtype: DType

    public init(encoder: H3VAEEncoder, decoder: H3VAEDecoder, dtype: DType) {
        self.encoder = encoder
        self.decoder = decoder
        self.dtype = dtype
    }

    public static func load(_ w: H3Weights, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3VAE {
        H3VAE(encoder: try H3VAEEncoder.load(w),
              decoder: try H3VAEDecoder.load(w, dtype: dtype),
              dtype: dtype)
    }

    /// 单帧图片 [1,3,1,H,W]（[-1,1]）→ 归一化 latent [1,24,1,H/16,W/16]。
    public func encodeImage(_ pixels: MLXArray) -> MLXArray {
        encoder.encodeImage(pixels)
    }

    /// 归一化 latent → 像素 [-1,1]。
    public func decode(_ zNorm: MLXArray) -> MLXArray {
        decoder.decode(zNorm)
    }
}
