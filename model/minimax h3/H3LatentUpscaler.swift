//
//  H3LatentUpscaler.swift
//  无限画布
//
//  MiniMax H3 latent upscaler（官方 3D conv 版）MLX Swift 移植。
//  对齐 Comfyui_Minimax_h3_latent_Upscaler nodes/minimax_h3_latent_upscaler_3d.py：
//    LatentResizer3D：conv_in → in_blocks(12 ResBlockEmb3D + 6 TemporalConv 交错)
//    → 空间 trilinear ×2 → out_blocks(同构) → norm_out → conv_out
//  权重：minimax_h3_latent_upscaler_3d_bf16.safetensors（345M，keys 无前缀）。
//  激活一律 channels-last [1, T, H, W, C]（MLX conv3d 布局）；输入输出 [1, C, T, H, W]。
//  输入期望为 VAE latent 原始域（未归一化），内部按 LATENTS_MEAN/STD 归一化再还原。
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// 官方 upscaler 用到的 24 通道归一化统计（与 ComfyUI 节点同表）。
public let h3UpLatentsMean: [Float] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
public let h3UpLatentsStd: [Float] = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

/// 跨整卷（T·H·W 一起）的 GroupNorm(32, C)：channels-last [1,T,H,W,C]。
func h3GroupNormFull32(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let shp = x.shape
    let B = shp[0], T = shp[1], H = shp[2], W = shp[3], C = shp[4]
    let G = 32
    precondition(C % G == 0)
    let g = C / G
    let gx = x.reshaped(B, T, H, W, G, g)
    let mu = gx.mean(axes: [1, 2, 3, 5], keepDims: true)
    let diff = gx - mu
    let vr = (diff * diff).mean(axes: [1, 2, 3, 5], keepDims: true)
    let nrm = diff / (vr + eps).sqrt()
    let back = nrm.reshaped(B, T, H, W, C)
    return back * w + b
}

/// [1,C,T,h,w] → 空间 ×2（trilinear align_corners=false 2x 的 H/W 双线性等价，
/// 权重 0.75/0.25，边界复制）。与 H3Pipeline.h3SpatialUpsample2x5D 同核。
func h3UpInterp2xSpatial(_ z: MLXArray) -> MLXArray {
    let c = z.shape[1]
    let t = z.shape[2]
    precondition(c >= 1 && t >= 1)

    func lin2xLast(_ x: MLXArray) -> MLXArray {
        let hh = x.shape[3]
        let m = x.shape[4]
        precondition(m >= 2)
        let xL = x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< m]
        let xLeft = concatenated([x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< 1],
                                  x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 0 ..< (m - 1)]], axis: 4)
        let xRight = concatenated([x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, 1 ..< m],
                                   x[0 ..< 1, 0 ..< c, 0 ..< t, 0 ..< hh, (m - 1) ..< m]], axis: 4)
        let even = xL * H3TensorOps.scalarLike(0.75, x) + xLeft * H3TensorOps.scalarLike(0.25, x)
        let odd = xL * H3TensorOps.scalarLike(0.75, x) + xRight * H3TensorOps.scalarLike(0.25, x)
        let ee = even.reshaped([1, c, t, hh, m, 1])
        let oo = odd.reshaped([1, c, t, hh, m, 1])
        return concatenated([ee, oo], axis: 5).reshaped([1, c, t, hh, m * 2])
    }

    let xW = lin2xLast(z)                                        // W ×2
    let xT = xW.transposed(0, 1, 2, 4, 3)                        // H 挪到末轴
    let xH = lin2xLast(xT)                                       // H ×2
    return xH.transposed(0, 1, 2, 4, 3)
}

/// 3D conv 权重 [O,I,kt,kh,kw]（PyTorch NCDHW）→ MLX [O,kt,kh,kw,I] f32。
func h3UpConvTap(_ raw: MLXArray) -> MLXArray {
    raw.transposed(0, 2, 3, 4, 1).asType(.float32)
}

/// depthwise 沿时间维 3D conv（MLX conv3d 不支持 groups>1 的手工替代）。
/// x/w/b 均为 channels-last [1,T,H,W,C]；w 为 [C,K,1,1,1]（O 前置），仅时间维 kernel K。
/// 语义与 PyTorch Conv3d(groups=C, kernel=(K,1,1), padding=(padT,0,0)) 对齐（相关不翻转）。
func h3DepthwiseTimeConv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, padT: Int = 2) -> MLXArray {
    let shp = x.shape
    let B = shp[0], T = shp[1], H = shp[2], W = shp[3], C = shp[4]
    let K = w.shape[1]
    let zL = MLXArray.zeros([B, padT, H, W, C])
    let zR = MLXArray.zeros([B, padT, H, W, C])
    let xp = concatenated([zL, x, zR], axis: 1)          // [B,T+2p,H,W,C]
    var out = x * 0
    for k in 0 ..< K {
        let wk = w[0 ..< C, k ..< (k + 1), 0 ..< 1, 0 ..< 1, 0 ..< 1].reshaped(1, 1, 1, 1, C)
        let xk = xp[0 ..< B, k ..< (k + T), 0 ..< H, 0 ..< W, 0 ..< C]
        out = out + xk * wk
    }
    out = out + b.reshaped(1, 1, 1, 1, C)
    return out
}

public final class H3LatentUpscaler {
    // ── 顶层权重（f32 / MLX 布局）──
    var convInW: MLXArray!; var convInB: MLXArray!
    var convOutW: MLXArray!; var convOutB: MLXArray!
    var normOutW: MLXArray!; var normOutB: MLXArray!
    var embed1W: MLXArray!; var embed1B: MLXArray!
    var embed2W: MLXArray!; var embed2B: MLXArray!
    var mean: MLXArray!; var std: MLXArray!

    // 每个模块 = ResBlock(emb) 或 TemporalConv。
    // 布局（in/out 同构，Python 每 b：RB，且 b%2==0 追加 TC）：
    //   idx0 RB, 1 TC, 2 RB, 3 RB, 4 TC, 5 RB, 6 RB, 7 TC, 8 RB, 9 RB,
    //   10 TC, 11 RB, 12 RB, 13 TC, 14 RB, 15 RB, 16 TC, 17 RB
    struct RBW {
        var inNormW: MLXArray; var inNormB: MLXArray
        var convInW: MLXArray; var convInB: MLXArray
        var embW: MLXArray; var embB: MLXArray
        var outNormW: MLXArray; var outNormB: MLXArray
        var convOutW: MLXArray; var convOutB: MLXArray
    }
    struct TCW {
        var normW: MLXArray; var normB: MLXArray
        var dwW: MLXArray; var dwB: MLXArray
        var pwW: MLXArray; var pwB: MLXArray
    }
    enum Block { case rb(RBW); case tc(TCW) }
    var inBlocks: [Block] = []
    var outBlocks: [Block] = []

    public init(weights: H3Weights, prefix: String = "") throws {
        func key(_ k: String) -> String { prefix.isEmpty ? k : prefix + "." + k }
        func w(_ k: String) throws -> MLXArray {
            // 一次性加载 345M bf16 → 逐 tensor 读出立即物化，禁用缓存防滞留
            guard let a = weights.getUncached(key(k)) else {
                throw H3Error.missingWeight(key(k))
            }
            return a
        }
        mean = MLXArray(h3UpLatentsMean).asType(.float32)
        std = MLXArray(h3UpLatentsStd).asType(.float32)

        convInW = h3UpConvTap(try w("conv_in.weight"))
        convInB = (try w("conv_in.bias")).asType(.float32)
        convOutW = h3UpConvTap(try w("conv_out.weight"))
        convOutB = (try w("conv_out.bias")).asType(.float32)
        normOutW = (try w("norm_out.weight")).asType(.float32)
        normOutB = (try w("norm_out.bias")).asType(.float32)
        embed1W = h3LoadLinT(try w("embed.0.weight"), dtype: .float32)
        embed1B = (try w("embed.0.bias")).asType(.float32)
        embed2W = h3LoadLinT(try w("embed.2.weight"), dtype: .float32)
        embed2B = (try w("embed.2.bias")).asType(.float32)

        func loadBlocks(_ side: String) throws -> [Block] {
            let rbIdx = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17]
            let tcIdx = [1, 4, 7, 10, 13, 16]
            var list: [Block] = []
            var ri = 0, ti = 0
            for idx in 0..<18 {
                if ri < rbIdx.count && rbIdx[ri] == idx {
                    let p = side + ".\(idx)"
                    let r = RBW(
                        inNormW: (try w(p + ".in_layers.0.weight")).asType(.float32),
                        inNormB: (try w(p + ".in_layers.0.bias")).asType(.float32),
                        convInW: h3UpConvTap(try w(p + ".in_layers.2.weight")),
                        convInB: (try w(p + ".in_layers.2.bias")).asType(.float32),
                        embW: h3LoadLinT(try w(p + ".emb_layers.1.weight"), dtype: .float32),
                        embB: (try w(p + ".emb_layers.1.bias")).asType(.float32),
                        outNormW: (try w(p + ".out_norm.weight")).asType(.float32),
                        outNormB: (try w(p + ".out_norm.bias")).asType(.float32),
                        convOutW: h3UpConvTap(try w(p + ".out_layers.2.weight")),
                        convOutB: (try w(p + ".out_layers.2.bias")).asType(.float32))
                    list.append(.rb(r))
                    ri += 1
                } else if ti < tcIdx.count && tcIdx[ti] == idx {
                    let p = side + ".\(idx)"
                    let t = TCW(
                        normW: (try w(p + ".norm.weight")).asType(.float32),
                        normB: (try w(p + ".norm.bias")).asType(.float32),
                        dwW: h3UpConvTap(try w(p + ".dwconv.weight")),
                        dwB: (try w(p + ".dwconv.bias")).asType(.float32),
                        pwW: h3UpConvTap(try w(p + ".pwconv.weight")),
                        pwB: (try w(p + ".pwconv.bias")).asType(.float32))
                    list.append(.tc(t))
                    ti += 1
                } else {
                    throw H3Error.missingWeight(side + ".\(idx)")
                }
            }
            return list
        }
        inBlocks = try loadBlocks("in_blocks")
        outBlocks = try loadBlocks("out_blocks")
        MLX.eval(allWeightArrays())
    }

    public func allWeightArrays() -> [MLXArray] {
        var arr: [MLXArray] = [convInW, convInB, convOutW, convOutB,
                               normOutW, normOutB,
                               embed1W, embed1B, embed2W, embed2B]
        for blk in inBlocks { arr.append(contentsOf: blkArrays(blk)) }
        for blk in outBlocks { arr.append(contentsOf: blkArrays(blk)) }
        return arr
    }

    func blkArrays(_ b: Block) -> [MLXArray] {
        switch b {
        case .rb(let r):
            return [r.inNormW, r.inNormB, r.convInW, r.convInB, r.embW, r.embB,
                    r.outNormW, r.outNormB, r.convOutW, r.convOutB]
        case .tc(let t):
            return [t.normW, t.normB, t.dwW, t.dwB, t.pwW, t.pwB]
        }
    }

    // MARK: - 单块

    func runRB(_ r: RBW, _ x: MLXArray, _ eBase: MLXArray) -> MLXArray {
        let n1 = h3GroupNormFull32(x, r.inNormW, r.inNormB)
        let a1 = silu(n1)
        var h = MLX.conv3d(a1, r.convInW, padding: IntOrTriple((1, 1, 1)))
        h = h + r.convInB
        let embIn = silu(eBase)                                    // [1,64]（Python emb_layers: SiLU, Linear）
        let embOut = H3TensorOps.denseLinear(embIn, r.embW, r.embB) // [1,1024]
        let emb5 = embOut.reshaped(1, 1, 1, 1, 1024)
        let half = 1024 / 2
        let scale = emb5[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< half]
        let shift = emb5[0 ..< 1, 0 ..< 1, 0 ..< 1, 0 ..< 1, half ..< 1024]
        let n2 = h3GroupNormFull32(h, r.outNormW, r.outNormB)
        let ns = n2 * (H3TensorOps.scalarLike(1.0, n2) + scale) + shift
        let a2 = silu(ns)
        var h2 = MLX.conv3d(a2, r.convOutW, padding: IntOrTriple((1, 1, 1)))
        h2 = h2 + r.convOutB
        return x + h2
    }

    func runTC(_ t: TCW, _ x: MLXArray) -> MLXArray {
        let n = h3GroupNormFull32(x, t.normW, t.normB)
        let a = silu(n)
        // dwconv 是 depthwise(512) 沿时间维 kernel 5、空间 1×1 的 3D conv；
        // MLX conv3d 不支持 groups>1，改用手工时间滑窗相关（与 PyTorch conv 对齐）。
        var dw = h3DepthwiseTimeConv(a, t.dwW, t.dwB, padT: 2)
        var pw = MLX.conv3d(dw, t.pwW, padding: 0)
        pw = pw + t.pwB
        return x + pw
    }

    // MARK: - 前向

    /// 输入 z：[1, C, T, h, w] 原始 VAE latent（未归一化）→ [1, C, T, H2, W2]（×scale，已还原域）。
    public func apply(_ z: MLXArray, scale: Float = 2.0) -> MLXArray {
        var x = z.transposed(0, 2, 3, 4, 1).asType(.float32)      // [1,T,h,w,C]
        MLX.eval(x)
        x = (x - mean) / std                                      // 通道归一化（末维广播）
        x = MLX.conv3d(x, convInW, padding: IntOrTriple((1, 1, 1))) + convInB
        MLX.eval(x)

        // embed：Python embed = Linear(1→64) → SiLU → Linear(64→64)，输入 [scale-1]
        let scaleIn = MLXArray(scale - 1).reshaped(1, 1).asType(.float32)
        let e1 = silu(H3TensorOps.denseLinear(scaleIn, embed1W, embed1B))
        let eBase = H3TensorOps.denseLinear(e1, embed2W, embed2B)  // [1,64]

        for blk in inBlocks {
            switch blk {
            case .rb(let r): x = runRB(r, x, eBase)
            case .tc(let t): x = runTC(t, x)
            }
            MLX.eval(x)
        }
        // 中间特征空间 ×scale（trilinear align_corners=false 等价）
        x = x.transposed(0, 4, 1, 2, 3)                          // [1,C,T,h,w]
        x = h3UpInterp2xSpatial(x)                               // [1,C,T,2h,2w]
        x = x.transposed(0, 2, 3, 4, 1)                          // [1,T,H2,W2,C]
        MLX.eval(x)
        for blk in outBlocks {
            switch blk {
            case .rb(let r): x = runRB(r, x, eBase)
            case .tc(let t): x = runTC(t, x)
            }
            MLX.eval(x)
        }
        var h = silu(h3GroupNormFull32(x, normOutW, normOutB))
        h = MLX.conv3d(h, convOutW, padding: IntOrTriple((1, 1, 1))) + convOutB
        h = h * std + mean
        MLX.eval(h)
        return h.transposed(0, 4, 1, 2, 3)                       // [1,C,T,H2,W2]
    }
}
