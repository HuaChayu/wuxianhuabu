//
//  H3-to-LTX-Latent-Adapter-(h3专属).swift
//  无限画布 · MiniMax H3 → LTX-2.5 latent 直通适配器
//
//  ============================================================================
//  为什么需要它
//  ============================================================================
//  原「H3 一采 + LTX 二采」链路在阶段交界处要走一段像素往返：
//
//      H3 clean latent [1,24,T,H/16,W/16]
//          └─ H3VAE.decode ──→ 像素 [1,3,Tf,H,W] ──→（内存桥 BCFHW）
//              └─ LTX vaeEncodeVideo ──→ LTX latent [1,F,h,w,128]
//                  └─ runLTXStage2RefineOnLatent（升频×2 + Stage2 refine）
//
//  其中 H3 VAE decode 与 LTX VAE encode 两步都只为了"换个 latent 域"，
//  既昂贵（两套 3D VAE 全序列前向）又会在两次重建里丢细节、漂色。
//
//  本适配器把这一步替换成一次轻量 Conv3D 映射：
//
//      H3 clean latent ──H3-to-LTX-Latent-Adapter──→ LTX 归一化 latent（直进 refine）
//
//  即：跳过 H3 解码、跳过 LTX 编码，latent → latent 直接衔接。
//
//  ============================================================================
//  权重与官方出处
//  ============================================================================
//  权重：H3-to-LTX-Latent-Adapter.safetensors（BF16，194,759,504 参数，228 个张量）
//  官方仓库：Efficient-Large-Model/H3-to-LTX-Latent-Adapter
//  消费方：NVlabs/Sana（runtime/stage2_ops/h3_ltx_adapter/{constants,geometry,model,adapter}.py）
//
//  官方推理形状契约（h3_upscale.py 中固定）：
//      H3_INPUT        = (1, 24, 37, 24, 42)     ← 像素 384×672 @124 帧（H3 16× 压缩）
//      H3_UPSCALED     = (1, 24, 37, 48, 84)     ← 先经 LBH H3 latent ×2 放大（本项目不接 LBH，
//                                                   仍用它自己的几何 ×2 / 或原尺寸直送）
//      ADAPTER_OUTPUT  = (1, 128, 17, 24, 42)    ← 注意 T=17
//      REFINER_INPUT   = (1, 128, 16, 24, 42)    ← refine 前须裁掉末帧（LTX 时间压缩 8）
//
//  ============================================================================
//  归一化域（关键，勿重复归一化）
//  ============================================================================
//  适配器的输入契约是 **已归一化** 的 H3 latent，即 constants.py 中的
//  (raw - H3_LATENTS_MEAN) / H3_LATENTS_STD。本项目 H3 VAE 权重 video_vae.safetensors
//  内自带的 latents_mean / latents_std 与该 constants 数值一致，采样结束拿到的
//  zForDecode 就已是该归一化域（用 (x-mean)/std 与乘 std 加 mean 互逆验证过），
//  因此本项目走「normalized 分支」：**不重复归一化**，直接几何对齐 → 模型。
//  （官方 adapter.convert(input_normalization="raw_h3") 才会做 normalize_raw_h3。）
//
//  ============================================================================
//  数值/实现注意事项
//  ============================================================================
//  · GroupNorm 的统计跨 T/H/W（不是 per-frame！），故不能复用 h3GroupNorm32；
//    另在内部升 float32 累积均值/方差后降回原 dtype，避免 bf16 统计精度损失。
//  · 时间 depthwise 卷积（k=(3,1,1), groups=C）在 MLX 下绕开 groups conv，
//    直接展开成 3 个时间 tap 的加权和，等价且更快。
//  · 空间/主干卷积走项目既有的 conv3dNDHWC 同款「时间窗拼通道 + conv2d」技巧。
//  · 1×1×1 卷积走 matmul 快路径。
//  · Swift 规则：函数新增参数一律置于声明末位（实参顺序须与声明一致）。
//  ============================================================================

import Foundation
import MLX

// MARK: - 常量（逐条对齐官方 constants.py / config.json，勿改）

public enum H3ToLTXAdapterConst {

    // ── 几何（constants.py） ──
    public static let h3LatentChannels = 24
    public static let h3SpatialCompression = 16
    public static let h3TemporalCompression = 4
    public static let h3ClipLength = 17
    public static let h3TokenDrop = 3

    public static let ltxLatentChannels = 128
    public static let ltxSpatialCompression = 32
    public static let ltxTemporalCompression = 8

    /// 时间打包槽位（config.json → geometry.temporal_pack_slots）
    public static let temporalPackSlots = 3

    // ── H3 per-channel 统计量（constants.py，训练期冻结） ──
    // 本项目 latent 已是该归一化域，此处仅作校验/外部调用方备用，不在主路径重复归一化。
    public static let h3LatentsMean: [Float] = [
        0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
        -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
        -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
        -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
        -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
    ]

    public static let h3LatentsStd: [Float] = [
        1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
        1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
        0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
        0.7996809482574463, 0.4498890042304993, 0.7197399735450745, 0.6936293244361877,
        2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
        3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
    ]

    // ── 网络结构（config.json → model_config，model_type = tiny） ──
    public static let inChannels = 384      // 24（H3）× 4（linear 1 + packed 3）
    public static let outChannels = 128     // LTX latent 通道
    public static let width = 752
    public static let numBlocks = 22
    public static let expansion = 2         // hidden = 752 × 2 = 1504
    public static let groups = 16           // GroupNorm 组数
    public static let normEps: Float = 1e-6
}

// MARK: - 错误

public enum H3ToLTXAdapterError: LocalizedError {
    case shapeMismatch(String)
    case missingWeight(String)
    case unsupportedVariant(String)

    public var errorDescription: String? {
        switch self {
        case .shapeMismatch(let s): return "H3-to-LTX Adapter 形状不符：\(s)"
        case .missingWeight(let s): return "H3-to-LTX Adapter 权重缺失：\(s)"
        case .unsupportedVariant(let s): return "H3-to-LTX Adapter 不支持的变体：\(s)"
        }
    }
}

// MARK: - 几何对齐（逐行复刻官方 geometry.py）

public enum H3ToLTXGeometry {

    /// H3 latent 的每个时间 token 对应的"像素帧位置"。
    /// tokens_per_chunk = ceil(17/4) = 5；positions = min(chunk*17 + token*4, pf-1)；
    /// 末尾丢弃 H3_TOKEN_DROP=3 个（H3 VAE 尾部冗余 token）。
    /// 例：pf=124 → 8 chunks × 5 = 40 → 丢 3 → 37，与官方 H3_INPUT 的 T=37 一致。
    public static func h3TemporalPositions(_ numPixelFrames: Int) -> [Float] {
        precondition(numPixelFrames > 0, "numPixelFrames must be positive")
        let clip = H3ToLTXAdapterConst.h3ClipLength
        let tc = H3ToLTXAdapterConst.h3TemporalCompression
        let tokensPerChunk = (clip + tc - 1) / tc
        let numChunks = (numPixelFrames + clip - 1) / clip
        var positions: [Int] = []
        positions.reserveCapacity(numChunks * tokensPerChunk)
        for chunk in 0 ..< numChunks {
            for token in 0 ..< tokensPerChunk {
                positions.append(min(chunk * clip + token * tc, numPixelFrames - 1))
            }
        }
        let drop = H3ToLTXAdapterConst.h3TokenDrop
        if drop > 0, positions.count > drop {
            positions = Array(positions.dropLast(drop))
        }
        return positions.map(Float.init)
    }

    /// LTX 侧为对齐 8 帧时间压缩所需的"补齐后像素帧数"。
    public static func paddedLTXPixelFrames(_ numPixelFrames: Int) -> Int {
        precondition(numPixelFrames > 0, "numPixelFrames must be positive")
        let tc = H3ToLTXAdapterConst.ltxTemporalCompression
        return ((numPixelFrames - 1 + tc - 1) / tc) * tc + 1
    }

    /// 补齐后像素帧序列对应的 LTX latent 时间位置。
    public static func ltxTemporalPositions(_ numPixelFrames: Int) -> [Float] {
        let tc = H3ToLTXAdapterConst.ltxTemporalCompression
        let latentFrames = (numPixelFrames - 1) / tc + 1
        return (0 ..< latentFrames).map { Float(min($0 * tc, numPixelFrames - 1)) }
    }

    /// 线性时间重采样（等价 torch.searchsorted(right=False) + lerp）。
    /// 输入输出均为 NDHWC [B, T, H, W, C]。
    public static func temporalResample(_ latent: MLXArray,
                                        sourcePositions: [Float],
                                        targetPositions: [Float]) -> MLXArray {
        var leftIdx = [Int32](), rightIdx = [Int32](), weights = [Float]()
        leftIdx.reserveCapacity(targetPositions.count)
        rightIdx.reserveCapacity(targetPositions.count)
        weights.reserveCapacity(targetPositions.count)

        let n = sourcePositions.count
        for t in targetPositions {
            // searchsorted(source, t, right=False)：首个 source[i] >= t
            var r = n - 1
            for i in 0 ..< n where sourcePositions[i] >= t { r = i; break }
            let l = max(r - 1, 0)
            let den = sourcePositions[r] - sourcePositions[l]
            let w = den > 0 ? min(max((t - sourcePositions[l]) / den, 0), 1) : 0
            leftIdx.append(Int32(l))
            rightIdx.append(Int32(r))
            weights.append(w)
        }

        let left = MLX.take(latent, MLXArray(leftIdx, [leftIdx.count]), axis: 1)
        let right = MLX.take(latent, MLXArray(rightIdx, [rightIdx.count]), axis: 1)
        let wv = MLXArray(weights, [1, weights.count, 1, 1, 1])
        return left + (right - left) * wv
    }

    /// 最近邻时间打包（等价 torch 的 argmin 分配 + slots 槽位填充 + 通道维 permute）。
    /// 空槽位以 0 填充并按 occupied 掩码置零，输出 NDHWC 且通道顺序为 slots-major。
    public static func temporalNearestPack(_ latent: MLXArray,
                                           sourcePositions: [Float],
                                           targetPositions: [Float],
                                           slots: Int) throws -> MLXArray {
        // 1) 每个 source 位置归到最近的 target 位置
        var assignment = [Int](repeating: 0, count: sourcePositions.count)
        for (s, sv) in sourcePositions.enumerated() {
            var best = 0
            var bestDist = Float.greatestFiniteMagnitude
            for (t, tv) in targetPositions.enumerated() {
                let d = abs(sv - tv)
                if d < bestDist { bestDist = d; best = t }
            }
            assignment[s] = best
        }

        // 2) 反挂到各自 target（保持 source 升序，与 torch.nonzero 顺序一致）
        var perTarget = [[Int]](repeating: [], count: targetPositions.count)
        for (s, t) in assignment.enumerated() { perTarget[t].append(s) }

        let requiredSlots = perTarget.map(\.count).max() ?? 0
        guard requiredSlots <= slots else {
            throw H3ToLTXAdapterError.shapeMismatch(
                "temporal packing needs \(requiredSlots) slots, checkpoint provides \(slots)")
        }

        // 3) 构造 [Tt × slots] 的索引与占位掩码
        var flat = [Int32](), occ = [Float]()
        flat.reserveCapacity(targetPositions.count * slots)
        occ.reserveCapacity(targetPositions.count * slots)
        for t in 0 ..< targetPositions.count {
            let list = perTarget[t]
            for k in 0 ..< slots {
                flat.append(k < list.count ? Int32(list[k]) : 0)
                occ.append(k < list.count ? 1 : 0)
            }
        }

        let b = latent.shape[0], h = latent.shape[2], w = latent.shape[3], c = latent.shape[4]
        let tt = targetPositions.count
        let selected = MLX.take(latent, MLXArray(flat, [flat.count]), axis: 1) // [B, Tt*slots, H, W, C]
        var out = selected.reshaped([b, tt, slots, h, w, c])
        out = out * MLXArray(occ, [1, tt, slots, 1, 1, 1])
        // [B, Tt, slots, H, W, C] → [B, Tt, H, W, slots, C] → [B, Tt, H, W, slots*C]
        out = out.transposed(axes: [0, 1, 3, 4, 2, 5]).reshaped([b, tt, h, w, slots * c])
        return out
    }

    /// NDHWC 版 pixel_unshuffle(2)：通道维顺序 c-major × h_sub × w_sub（与 torch 一致）。
    public static func pixelUnshuffleNDHWC(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], h = x.shape[2], w = x.shape[3], c = x.shape[4]
        let h2 = h / 2, w2 = w / 2
        return x
            .reshaped([b, t, h2, 2, w2, 2, c])                 // [B,T,H2,hs,W2,ws,C]
            .transposed(axes: [0, 1, 2, 4, 6, 3, 5])           // [B,T,H2,W2,C,hs,ws]
            .reshaped([b, t, h2, w2, c * 4])
    }

    /// 官方 align_h3_to_ltx：时间线性重采样 + 最近邻打包（沿通道拼 96）→ 空间 pixel_unshuffle(2) → 384 通道。
    /// 输入 NDHWC [B, T_h3, H, W, 24]，输出 NDHWC [B, T_ltx, H/2, W/2, 384]。
    public static func align(_ latentNDHWC: MLXArray,
                             pixelFrames: Int,
                             targetHeight: Int,
                             targetWidth: Int,
                             slots: Int = H3ToLTXAdapterConst.temporalPackSlots) throws -> MLXArray {
        guard latentNDHWC.ndim == 5 else {
            throw H3ToLTXAdapterError.shapeMismatch("expect NDHWC 5D, got \(latentNDHWC.shape)")
        }
        let sourcePositions = h3TemporalPositions(pixelFrames)
        guard sourcePositions.count == latentNDHWC.shape[1] else {
            throw H3ToLTXAdapterError.shapeMismatch(
                "H3 latent T=\(latentNDHWC.shape[1]) 与 pixelFrames=\(pixelFrames) 不匹配，期望 T=\(sourcePositions.count)")
        }

        let paddedFrames = paddedLTXPixelFrames(pixelFrames)
        let targetPositions = ltxTemporalPositions(paddedFrames)

        let linear = temporalResample(latentNDHWC, sourcePositions: sourcePositions, targetPositions: targetPositions)
        let packed = try temporalNearestPack(latentNDHWC, sourcePositions: sourcePositions,
                                             targetPositions: targetPositions, slots: slots)
        let aligned = MLX.concatenated([linear, packed], axis: 4)

        let sh = aligned.shape[2], sw = aligned.shape[3]
        guard sh == targetHeight * 2, sw == targetWidth * 2 else {
            throw H3ToLTXAdapterError.shapeMismatch(
                "冻结适配器要求 2× pixel-unshuffle：source=(\(sh),\(sw)) target=(\(targetHeight),\(targetWidth))")
        }
        return pixelUnshuffleNDHWC(aligned)
    }
}

// MARK: - 网络（复刻官方 model.py 的 H3ToLTXConvAdapter，NDHWC 布局）

/// 全部张量按"权重是什么 dtype 就用什么 dtype"，中间激活 dtype 跟随输入。
public final class H3ToLTXConvAdapterModel {

    struct Block {
        let normW: MLXArray, normB: MLXArray
        let spatialW: MLXArray, spatialB: MLXArray
        let temporalW: MLXArray, temporalB: MLXArray
        let inProjW: MLXArray, inProjB: MLXArray
        let outProjW: MLXArray, outProjB: MLXArray
    }

    private let skipW: MLXArray, skipB: MLXArray
    private let stemW: MLXArray, stemB: MLXArray
    private let blocks: [Block]
    private let finalNormW: MLXArray, finalNormB: MLXArray
    private let headW: MLXArray, headB: MLXArray
    private let groups: Int

    private init(skipW: MLXArray, skipB: MLXArray,
                 stemW: MLXArray, stemB: MLXArray,
                 blocks: [Block],
                 finalNormW: MLXArray, finalNormB: MLXArray,
                 headW: MLXArray, headB: MLXArray,
                 groups: Int) {
        self.skipW = skipW; self.skipB = skipB
        self.stemW = stemW; self.stemB = stemB
        self.blocks = blocks
        self.finalNormW = finalNormW; self.finalNormB = finalNormB
        self.headW = headW; self.headB = headB
        self.groups = groups
    }

    /// PyTorch Conv3d 权重 [Cout, Cin, kD, kH, kW] → MLX [Cout, kD, kH, kW, Cin]
    private static func permuteConvWeight(_ w: MLXArray) -> MLXArray {
        w.transposed(axes: [0, 2, 3, 4, 1])
    }

    /// 从已加载的 safetensors 字典构建。
    /// 键名与官方 checkpoint 一致：skip / stem / blocks.N.{norm,spatial,temporal,in_proj,out_proj} / final_norm / head
    public static func build(from arrays: [String: MLXArray],
                             numBlocks: Int = H3ToLTXAdapterConst.numBlocks,
                             groups: Int = H3ToLTXAdapterConst.groups) throws -> H3ToLTXConvAdapterModel {
        func need(_ k: String) throws -> MLXArray {
            guard let v = arrays[k] else { throw H3ToLTXAdapterError.missingWeight(k) }
            return v
        }

        let skipW = try permuteConvWeight(need("skip.weight"))
        let skipB = try need("skip.bias")
        let stemW = try permuteConvWeight(need("stem.weight"))
        let stemB = try need("stem.bias")

        var blocks: [Block] = []
        blocks.reserveCapacity(numBlocks)
        for i in 0 ..< numBlocks {
            let p = "blocks.\(i)."
            blocks.append(Block(
                normW: try need(p + "norm.weight"), normB: try need(p + "norm.bias"),
                spatialW: try permuteConvWeight(need(p + "spatial.weight")), spatialB: try need(p + "spatial.bias"),
                temporalW: try need(p + "temporal.weight"), temporalB: try need(p + "temporal.bias"),
                inProjW: try permuteConvWeight(need(p + "in_proj.weight")), inProjB: try need(p + "in_proj.bias"),
                outProjW: try permuteConvWeight(need(p + "out_proj.weight")), outProjB: try need(p + "out_proj.bias")
            ))
        }

        let finalNormW = try need("final_norm.weight")
        let finalNormB = try need("final_norm.bias")
        let headW = try permuteConvWeight(need("head.weight"))
        let headB = try need("head.bias")

        return H3ToLTXConvAdapterModel(
            skipW: skipW, skipB: skipB, stemW: stemW, stemB: stemB,
            blocks: blocks, finalNormW: finalNormW, finalNormB: finalNormB,
            headW: headW, headB: headB, groups: groups)
    }

    // MARK: 前向

    public func callAsFunction(_ aligned: MLXArray) -> MLXArray {
        // aligned: NDHWC [B, T, h, w, 384]
        let skip = conv1x1x1(aligned, weight: skipW, bias: skipB)

        var h = convTimeStack2D(aligned, weight: stemW, bias: stemB, kD: 3, kH: 3, kW: 3)
        for (i, blk) in blocks.enumerated() {
            h = applyBlock(blk, to: h)
            // 每 4 块落地一次，避免 22 层 lazy 图堆积造成峰值内存尖峰
            if i % 4 == 3 { MLX.eval(h) }
        }
        h = groupNorm(h, weight: finalNormW, bias: finalNormB)
        h = silu(h)
        h = conv1x1x1(h, weight: headW, bias: headB)
        return skip + h
    }

    private func applyBlock(_ blk: Block, to x: MLXArray) -> MLXArray {
        var h = groupNorm(x, weight: blk.normW, bias: blk.normB)
        h = silu(h)
        h = convTimeStack2D(h, weight: blk.spatialW, bias: blk.spatialB, kD: 1, kH: 3, kW: 3)
        h = depthwiseTemporal3x1x1(h, weight: blk.temporalW, bias: blk.temporalB)
        h = silu(h)

        let proj = conv1x1x1(h, weight: blk.inProjW, bias: blk.inProjB) // [B,T,H,W, 2*hidden]
        let cout = proj.shape[4]
        let half = cout / 2
        let b = proj.shape[0], t = proj.shape[1], hh = proj.shape[2], ww = proj.shape[3]
        let value = proj[0 ..< b, 0 ..< t, 0 ..< hh, 0 ..< ww, 0 ..< half]
        let gate = proj[0 ..< b, 0 ..< t, 0 ..< hh, 0 ..< ww, half ..< cout]
        let gated = value * silu(gate)
        let out = conv1x1x1(gated, weight: blk.outProjW, bias: blk.outProjB)
        return x + out
    }

    // MARK: 算子

    /// GroupNorm（NDHWC，统计跨 T/H/W/Cg，与 PyTorch NCDHW 语义一致）。
    /// 内部升 float32 累积均值/方差，避免 bf16 下的统计精度损失。
    private func groupNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], h = x.shape[2], w = x.shape[3], c = x.shape[4]
        let cg = c / groups
        let r = x.reshaped([b, t, h, w, groups, cg])
        let rf = r.asType(.float32)
        let mean = rf.mean(axes: [1, 2, 3, 5], keepDims: true)
        let varc = rf.variance(axes: [1, 2, 3, 5], keepDims: true)
        let normed = ((rf - mean) / MLX.sqrt(varc + H3ToLTXAdapterConst.normEps)).asType(x.dtype)
        let y = normed.reshaped([b, t, h, w, c])
        return y * weight + bias
    }

    /// 1×1×1 卷积 → matmul 快路径（NDHWC 输入输出）。
    private func conv1x1x1(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
        let cout = weight.shape[0]
        let wm = weight.reshaped([cout, cin])
        let y = MLX.matmul(x.reshaped([-1, cin]), wm.transposed()) + bias
        return y.reshaped([b, t, h, w, cout])
    }

    /// 通用（groups=1）3D 卷积的 "时间窗拼通道 + conv2d" 实现，支持各向异性 kernel。
    /// 时间维按 torch 对称 zero padding：(kD-1)/2 两侧各补。
    private func convTimeStack2D(_ x: MLXArray, weight: MLXArray, bias: MLXArray?,
                                 kD: Int, kH: Int, kW: Int) -> MLXArray {
        let b = x.shape[0], d = x.shape[1], h = x.shape[2], w = x.shape[3], cin = x.shape[4]
        let cout = weight.shape[0]
        let pD = (kD - 1) / 2
        let pH = (kH - 1) / 2
        let pW = (kW - 1) / 2

        var t = x
        if pD > 0 {
            let zf = MLXArray.zeros([b, pD, h, w, cin], dtype: x.dtype)
            let zb = MLXArray.zeros([b, pD, h, w, cin], dtype: x.dtype)
            t = MLX.concatenated([zf, t, zb], axis: 1)
        }

        // 权重 [Cout, kD, kH, kW, Cin] → [Cout, kH, kW, kD*Cin]（kD-major，与窗口拼接顺序一致）
        let w2 = weight.transposed(axes: [0, 2, 3, 1, 4]).reshaped([cout, kH, kW, kD * cin])

        var wins: [MLXArray] = []
        wins.reserveCapacity(d)
        for j in 0 ..< d {
            let win = t[0 ..< b, j ..< (j + kD), 0 ..< h, 0 ..< w, 0 ..< cin]
                .transposed(axes: [0, 2, 3, 1, 4])          // [B, H, W, kD, Cin]
                .reshaped([b, h, w, kD * cin])
            wins.append(win)
        }
        let stacked = MLX.concatenated(wins, axis: 1)
            .reshaped([d * b, h, w, kD * cin])              // batch 维 = (原 batch, 时间) 行优先

        var out = conv2dNHWC(stacked, weight: w2, bias: nil,
                             stride: (1, 1), padding: (pH, pW))
        if let bias { out = out + bias }
        return out.reshaped([b, d, h, w, cout])
    }

    /// 时间 depthwise 卷积 k=(3,1,1), groups=C：展开为 3 个时间 tap 的加权和。
    /// weight 为 PyTorch 布局 [C, 1, 3, 1, 1]（dim2 = kD）。
    private func depthwiseTemporal3x1x1(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], h = x.shape[2], w = x.shape[3], c = x.shape[4]
        let wc = weight.reshaped([c, 3])
        let zero = MLXArray.zeros([b, 1, h, w, c], dtype: x.dtype)
        let padded = MLX.concatenated([zero, x, zero], axis: 1) // [B, T+2, H, W, C]

        var acc = MLXArray.zeros([b, t, h, w, c], dtype: x.dtype)
        for j in 0 ..< 3 {
            let wj = wc[0 ..< c, j ..< (j + 1)].reshaped([1, 1, 1, 1, c])
            let xj = padded[0 ..< b, j ..< (j + t), 0 ..< h, 0 ..< w, 0 ..< c]
            acc = acc + xj * wj
        }
        return acc + bias.reshaped([1, 1, 1, 1, c])
    }
}

// MARK: - 适配器门面

/// H3 → LTX latent 直通适配器。
///
/// 用法（本项目接入点：H3Pipeline 直出段，zForDecode 之后、H3VAE.decode 之前）：
/// ```swift
/// let adapter = try H3ToLTXLatentAdapter.load(
///     weightsURL: URL(fileURLWithPath: "/…/H3-to-LTX-Latent-Adapter.safetensors"))
/// let ltxLatent = try adapter.convert(h3LatentNCDHW: zForDecode, pixelFrames: Int(frameCount))
/// // ltxLatent: NDHWC [1, F, h, w, 128]（LTX 归一化域）→ 直接喂 runLTXStage2RefineOnLatent
/// ```
public final class H3ToLTXLatentAdapter {

    private let model: H3ToLTXConvAdapterModel
    public let weightDType: DType

    private init(model: H3ToLTXConvAdapterModel, weightDType: DType) {
        self.model = model
        self.weightDType = weightDType
    }

    /// 从 .safetensors 加载（BF16 权重，推理用）。
    public static func load(weightsURL: URL) throws -> H3ToLTXLatentAdapter {
        let arrays = try MLX.loadArrays(url: weightsURL)
        let model = try H3ToLTXConvAdapterModel.build(from: arrays)
        // 权重统一按其自身 dtype（官方为 bfloat16）；若个别张量为 f16/f32 也照用
        let dtype = arrays["stem.weight"]?.dtype ?? .bfloat16
        return H3ToLTXLatentAdapter(model: model, weightDType: dtype)
    }

    /// H3 clean latent → LTX 归一化 latent。
    ///
    /// - Parameters:
    ///   - h3LatentNCDHW: `[B, 24, T_h3, H/16, W/16]`（NCDHW，即项目 zForDecode 的原样布局），
    ///     必须是**已归一化**域（(raw-mean)/std），适配器不会再做归一化。
    ///   - pixelFrames: 该 latent 对应的**输出像素帧数**（H3 对齐后帧数，例如 124）。
    ///     用于时间对齐；h3TemporalPositions(pixelFrames).count 必须等于 T_h3，否则抛错。
    /// - Returns: NDHWC `[1, F, H/32, W/32, 128]`，F = (pixelFrames + 7) / 8
    ///   （适配器原生多出末帧，此处已按官方 REFINER_INPUT 约定裁掉，可直接进 LTX refine）。
    public func convert(h3LatentNCDHW: MLXArray, pixelFrames: Int) throws -> MLXArray {
        guard h3LatentNCDHW.ndim == 5 else {
            throw H3ToLTXAdapterError.shapeMismatch("H3 latent 期望 5D NCDHW，实际 \(h3LatentNCDHW.shape)")
        }
        let b = h3LatentNCDHW.shape[0]
        let c = h3LatentNCDHW.shape[1]
        let latH = h3LatentNCDHW.shape[3]
        let latW = h3LatentNCDHW.shape[4]
        guard b == 1, c == H3ToLTXAdapterConst.h3LatentChannels else {
            throw H3ToLTXAdapterError.shapeMismatch("H3 latent 期望 [1,24,T,H,W]，实际 \(h3LatentNCDHW.shape)")
        }
        guard latH % 2 == 0, latW % 2 == 0 else {
            throw H3ToLTXAdapterError.shapeMismatch(
                "空间维需为偶数才能 2× pixel-unshuffle：\(latH)×\(latW)")
        }

        // 输出像素几何 = H3 空间压缩比 16；LTX 侧为 32，正好 2× 关系
        let pixelHeight = latH * H3ToLTXAdapterConst.h3SpatialCompression
        let pixelWidth = latW * H3ToLTXAdapterConst.h3SpatialCompression
        let targetHeight = pixelHeight / H3ToLTXAdapterConst.ltxSpatialCompression
        let targetWidth = pixelWidth / H3ToLTXAdapterConst.ltxSpatialCompression

        // NCDHW → NDHWC
        let ndhwc = h3LatentNCDHW
            .transposed(axes: [0, 2, 3, 4, 1])
            .asType(weightDType)

        let aligned = try H3ToLTXGeometry.align(ndhwc,
                                                 pixelFrames: pixelFrames,
                                                 targetHeight: targetHeight,
                                                 targetWidth: targetWidth)

        var out = model(aligned)                       // [1, T_ltx, h, w, 128]

        // 官方：ADAPTER_OUTPUT T=17 → REFINER_INPUT T=16（裁掉末帧）
        let ltxFrames = (pixelFrames + H3ToLTXAdapterConst.ltxTemporalCompression - 1)
            / H3ToLTXAdapterConst.ltxTemporalCompression
        if out.shape[1] > ltxFrames {
            out = out[0 ..< 1, 0 ..< ltxFrames, 0 ..< out.shape[2], 0 ..< out.shape[3], 0 ..< out.shape[4]]
        }
        MLX.eval(out)
        return out
    }

    /// 便捷版：直接给出转换后 LTX latent 的像素几何（供上层的 fullPixelWidth/Height 计算）。
    public static func ltxPixelSize(forH3LatentSpace latH: Int, _ latW: Int) -> (width: Int, height: Int) {
        let ph = latH * H3ToLTXAdapterConst.h3SpatialCompression
        let pw = latW * H3ToLTXAdapterConst.h3SpatialCompression
        return (pw, ph)
    }
}
