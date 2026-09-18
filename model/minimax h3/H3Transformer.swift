//
//  H3Transformer.swift
//  MiniMax H3 (mlx-serve-main) DiT transformer移植
//
//  对应源码: /Users/huachayui/Downloads/mlx-serve-main/src/minimax_h3.zig
//  模块:  Model / AttnW / MlpW / AdalnW / BlockW / RefinerBlockW
//         timeEmbed / refineText / precomputeAdaln / forward / finalHead
//  权重:  transformer.safetensors (affine 4-bit group64 量化 + fp32 岛) + turbo_lora.safetensors
//
//  依赖:  H3Common.swift (H3Weights / MfLinear / H3LoraFile / rmsNormLast / applyRopePub ...)
//         H3Layout.swift (PackedLayout / buildRope / buildTimestepPlan / ModRun / SparseSpec ...)
//

import Foundation
import MLX
import MLXFast
import MLXRandom
import Darwin

// NA_H3ATTCHK sparse-vs-dense probe layer counter (debug only).
private var sparseLayerNo = 0
private var profLayerNo = 0

// NA_H3CDF dense-attention locality/CDF probe (debug only).
private var cdfLayerNo = 0
private var cdfGeo: (g: Int, t: Int, f: Int)?

/// Probe how a video row's DENSE attention mass is distributed over the video
/// key grid (strip | same-frame | neighbor-frames | far-frames) plus top-k
/// cumulative weight. Answers: what key-set shape can approximate dense best?
private func h3RunCDF(q: MLXArray, k: MLXArray, hd: Int, S: Int, geo: (g: Int, t: Int, f: Int), layer: Int) {
    let targets = [0, 12, 25, 49]
    guard targets.contains(layer) else { return }
    let g = geo.g, t = geo.t, f = geo.f
    let scaleF = Float(1.0 / sqrt(Double(hd)))
    let probeRows = [g, g + (t / 2) * f + f / 2]   // first video row + middle-frame/middle-token row
    for row in probeRows {
        let qr = q[0, 0, row..<(row + 1), 0..<hd]
        let kr = k[0, 0, 0..<S, 0..<hd]
        let sc = matmul(qr, kr.transposed()).reshaped([S]).asType(.float32)
        MLX.eval(sc)
        let smax = sc.max().item(Float.self)
        let fr0 = (row - g) / f
        let fi0 = (row - g) % f
        var wStrip = 0.0, wSelf = 0.0, wSLoc = 0.0, wFLoc = 0.0, wFNear = 0.0, wFMid = 0.0, wFFar = 0.0
        // candidate key-set mass coverage: W1=±1帧 W4=±4帧 ; W4+S2 / W4+S3 add strided far frames
        var cW1 = 0.0, cW4 = 0.0, cW4S2 = 0.0, cW4S3 = 0.0
        var total = 0.0
        // top-512 min-heap kept as simple sorted window (debug only)
        var top: [Double] = Array(repeating: -1, count: 512)
        var topMin = -1.0
        for i in 0..<S {
            let w = exp(Double((sc[i].item(Float.self) - smax) * scaleF))
            total += w
            if w > topMin {
                if let mi = top.firstIndex(of: topMin) { top[mi] = w }
                topMin = top.min()!
            }
            if i < g { wStrip += w; continue }
            let idx = i - g
            guard idx < t * f else { continue }   // beyond video grid (audio etc.)
            let fr = idx / f
            let dfr = abs(fr - fr0)
            if dfr == 0 {
                let dfi = abs(idx % f - fi0)
                if dfi == 0 { wSelf += w } else if dfi <= 2 { wSLoc += w }
            } else if dfr <= 1 { wFLoc += w }
            else if dfr <= 4 { wFNear += w }
            else if dfr <= 12 { wFMid += w }
            else { wFFar += w }
            // candidate key-set coverage (frames fully sampled)
            if dfr <= 1 { cW1 += w }
            if dfr <= 4 { cW4 += w }
            if dfr <= 4 || (fr % 2 == fr0 % 2) { cW4S2 += w }
            if dfr <= 4 || (fr % 3 == fr0 % 3) { cW4S3 += w }
        }
        let rs = total > 0 ? 1.0 / total : 1.0
        let sumTop = { (n: Int) -> Double in top.sorted(by: >).prefix(n).reduce(0, +) }
        // count sampled frames for each candidate set (edges truncated)
        var n1 = 0, n4 = 0, n42 = 0, n43 = 0
        for fr in 0..<t {
            let dfr = abs(fr - fr0)
            if dfr <= 1 { n1 += 1 }
            if dfr <= 4 { n4 += 1 }
            if dfr <= 4 || fr % 2 == fr0 % 2 { n42 += 1 }
            if dfr <= 4 || fr % 3 == fr0 % 3 { n43 += 1 }
        }
        print(String(format: "[H3CDF] L%02d row(fr=%d,fi=%d) S=%d g=%d | strip=%.3f self=%.3f sLoc=%.3f fLoc=%.3f fNear=%.3f fMid=%.3f fFar=%.3f | top8=%.3f top64=%.3f top512=%.3f",
                     layer, fr0, fi0, S, g,
                     wStrip * rs, wSelf * rs, wSLoc * rs, wFLoc * rs, wFNear * rs, wFMid * rs, wFFar * rs,
                     sumTop(8) * rs, sumTop(64) * rs, sumTop(512) * rs))
        print(String(format: "[H3CDF]   coverage W1(%d帧)=%.3f W4(%d帧)=%.3f W4+S2(%d帧)=%.3f W4+S3(%d帧)=%.3f",
                     n1, (cW1 + wStrip) * rs, n4, (cW4 + wStrip) * rs, n42, (cW4S2 + wStrip) * rs, n43, (cW4S3 + wStrip) * rs))
        fflush(stdout)
    }
    if layer >= 49 {
        print("[H3CDF] probe complete at layer 49, exiting.")
        fflush(stdout)
        exit(0)
    }
}

// MARK: - Tensor helpers (mirror minimax_h3.zig top-level helpers)

/// 诊断钩子：NA_DUMP_FWD=1 时打印张量统计（eager eval）。
public func h3DumpRowStats(_ name: String, _ x: MLXArray, limitRows: Int = 0) {
    guard ProcessInfo.processInfo.environment["NA_DUMP_FWD"] == "1" else { return }
    MLX.eval(x)
    if ProcessInfo.processInfo.environment["NA_DUMP_LATENT"] == "1",
       let p = ProcessInfo.processInfo.environment["NA_BIN_\(name.replacingOccurrences(of: "[", with: "_").replacingOccurrences(of: "]", with: "_"))"],
       !FileManager.default.fileExists(atPath: p) {
        let f = x.asType(.float32).asArray(Float.self)
        let data = f.withUnsafeBufferPointer { Data(buffer: $0) }
        try? data.write(to: URL(fileURLWithPath: p))
    }
    let f = x.asType(.float32)
    let n = f.shape.reduce(1, *)
    var sub = f
    if limitRows > 0, f.shape.count == 2 {
        let rows = min(f.shape[0], limitRows)
        sub = f[0..<rows]
    }
    let vals = sub.asArray(Float.self)
    var s: Double = 0
    for v in vals { s += Double(v) }
    let mn = Float(s / Double(vals.count))
    var ss: Double = 0
    var mx: Float = 0
    for v in vals {
        let d = Double(v - mn)
        ss += d * d
        if abs(v) > mx { mx = abs(v) }
    }
    let sd = Float((ss / Double(vals.count)).squareRoot())
    print("[NA_DUMP_FWD] \(name) shape=\(x.shape) n=\(n) mean=\(mn) std=\(sd) maxAbs=\(mx)")
    fflush(stdout)
}

public enum H3TensorOps {
    /// Rows [start, end) along axis 0. Keeps a view (no copy).
    public static func sliceRows(_ x: MLXArray, _ start: Int, _ end: Int) -> MLXArray {
        x[start..<end]
    }

    /// Slice along a leading axis (0..<rank). Generic helper mirroring Zig sliceAxis.
    public static func sliceAxis(_ x: MLXArray, _ start: Int, _ end: Int, axis: Int) -> MLXArray {
        switch axis {
        case 0: return x[start..<end]
        case 1: return x[0..., start..<end]
        case 2: return x[0..., 0..., start..<end]
        case 3: return x[0..., 0..., 0..., start..<end]
        case 4: return x[0..., 0..., 0..., 0..., start..<end]
        default: fatalError("H3TensorOps.sliceAxis: unsupported axis \(axis)")
        }
    }

    /// 0-d scalar like the reference's scalarLike(v, ref).
    public static func scalarLike(_ v: Float, _ ref: MLXArray) -> MLXArray {
        MLXArray(v) // dtype is float32; callers cast where needed
    }

    public static func meanAbs(_ x: MLXArray) -> MLXArray {
        x.abs().mean()
    }

    /// x[..., in] @ W[in, out] + b, all in fp32 (dense island). Matches denseLinear.
    public static func denseLinear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray?) -> MLXArray {
        let xf = x.asType(.float32)
        let wf = w.asType(.float32)
        var out = xf.matmul(wf)
        if let b {
            out = out + b.asType(.float32)
        }
        return out
    }

    /// Split [..., n*k, ...] into n equal pieces along `axis` (contiguous order).
    public static func splitEqual(_ x: MLXArray, _ n: Int, axis: Int) -> [MLXArray] {
        let size = x.shape[axis]
        let step = size / n
        var parts: [MLXArray] = []
        parts.reserveCapacity(n)
        for i in 0..<n {
            let s = i * step
            let e = (i + 1) * step
            parts.append(sliceAxis(x, s, e, axis: axis))
        }
        return parts
    }

    /// Audio latent [T, 2h, 2w, C] → rows [T*2h*2w, C] (row-major (t,y,x)).
    public static func audioLatentToRows(_ lat: MLXArray, w: Int, h: Int) -> MLXArray {
        lat.reshaped(-1, lat.shape[3])
    }

    /// Rows [T*2h*2w, C] → latent [T, 2h, 2w, C].
    public static func audioRowsToLatent(_ rows: MLXArray, w: Int, h: Int) -> MLXArray {
        let c = rows.shape[1]
        let n = rows.shape[0]
        let t = n / (2 * h * 2 * w)
        return rows.reshaped(t, 2 * h, 2 * w, c)
    }

    /// Still image [1, 3, H, W] → 2-frame patch block [2, 3, H, W] (temporal repeat).
    public static func stillAsFramePair(_ img: MLXArray) -> MLXArray {
        concatenated([img, img], axis: 0)
    }

    /// Latent video [T, H, W, C] → patch rows [T·gh·gw, C·patchH·patchW] (patch 2×2).
    /// 行序 (t, gh, gw)；行内 [c, ph, pw] 展平，与 Zig patchifyVideo
    /// （n c t r h p w q -> n t h w c r p q）完全一致。
    public static func patchifyVideo(_ lat: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let t = lat.shape[0]
        let c = lat.shape[3]
        let p = 2
        // [T, gh, p, gw, p, C] → [T, gh, gw, C, p, p]（行内 C 外层、patch 内层）
        let pv = lat.reshaped(t, gridH, p, gridW, p, c)
        let pvt = pv.transposed(0, 1, 3, 5, 2, 4)
        return pvt.reshaped(t * gridH * gridW, c * p * p)
    }

    /// Patch rows [T·gh·gw, C·4]（行内 [c, ph, pw]）→ latent [T, gh·2, gw·2, C]。
    public static func unpatchifyVideo(_ rows: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let t = rows.shape[0] / (gridH * gridW)
        let c = rows.shape[1] / 4
        let r = rows.reshaped(t, gridH, gridW, c, 2, 2)
        return r.transposed(0, 1, 4, 2, 5, 3).reshaped(t, gridH * 2, gridW * 2, c)
    }

    /// dσa/dσv — the sampler folds this into the audio velocity (forward() returns -slope·aout).
    public static func audioStepFactor(_ sigmaV: Double, _ shiftV: Double, _ shiftA: Double) -> Double {
        timeShiftSlope(sigmaV, fromShift: shiftV, toShift: shiftA)
    }

    /// Step-cache warmup: k<=1 disables broadcast; the first two and the last two steps
    /// always recompute (mirrors attnBroadcastRefresh in H3Layout).
    public static func stepCacheWarmup(step: Int, total: Int) -> Bool {
        attnBroadcastRefresh(step, steps: UInt32(total), k: 2)
    }
}

// MARK: - Time embedding (fp32 island)

public final class H3TimeEmbedder {
    public let inputDim: Int
    public let outDim: Int
    public var projInW: MLXArray   // [inputDim, outDim] fp32
    public var projInB: MLXArray?  // [outDim]
    public var projOutW: MLXArray  // [outDim, outDim] fp32
    public var projOutB: MLXArray? // [outDim]

    public init(inputDim: Int, outDim: Int,
                projInW: MLXArray, projInB: MLXArray?,
                projOutW: MLXArray, projOutB: MLXArray?) {
        self.inputDim = inputDim
        self.outDim = outDim
        self.projInW = projInW
        self.projInB = projInB
        self.projOutW = projOutW
        self.projOutB = projOutB
    }

    public static func load(_ w: H3Weights, inputDim: Int = 256, outDim: Int = 2688) throws -> H3TimeEmbedder {
        guard let inW = w.get("time_embedder.proj_in.weight"),
              let outW = w.get("time_embedder.proj_out.weight") else {
            throw H3Error.missingWeight("time_embedder.proj_in.weight / proj_out.weight")
        }
        return H3TimeEmbedder(
            inputDim: inputDim,
            outDim: outDim,
            projInW: inW,
            projInB: w.get("time_embedder.proj_in.bias"),
            projOutW: outW,
            projOutB: w.get("time_embedder.proj_out.bias"))
    }

    /// t_vals → [m, outDim] (fp32 inside, cast to bf16 by the caller).
    public func forward(_ tVals: [Double]) -> MLXArray {
        let m = tVals.count
        let half = inputDim / 2
        let log10k = log(10000.0)
        var buf = [Float](repeating: 0, count: m * inputDim)
        for (i, t) in tVals.enumerated() {
            for j in 0..<half {
                let freq = exp(-log10k * Double(j) / Double(half))
                let arg = t * freq
                buf[i * inputDim + j] = Float(cos(arg))
                buf[i * inputDim + half + j] = Float(sin(arg))
            }
        }
        let emb = MLXArray(buf, [m, inputDim])
        let h1 = H3TensorOps.denseLinear(emb, projInW, projInB)
        let a1 = silu(h1)
        return H3TensorOps.denseLinear(a1, projOutW, projOutB)
    }
}

// MARK: - Lookup AdaLN (pruned / fused checkpoints)

/// 剪枝/融合权重的查表式时间调制，替代 `time_embedder + adaln_proj` 大矩阵通路。
///
/// 社区 pruned/fused 权重把时刻嵌入投影到 rank 维正交基上：
///     silu(tEmb(t)) ≈ adaln_mean + adaln_t_table[idx] · adaln_basis,  idx = round(t·1024)
/// 并把 `adaln_proj.linear` 折叠为 [out, rank]，其 bias 已吸收 `W·adaln_mean` 项。
/// 于是调制可直接由查表系数得到：
///     mod = W[out, rank] · c + b[out]
/// 该式已与原版权重逐层端到端比对，余弦 ≥ 0.999（层 0/25/49 × t∈[0,1]）。
public final class H3AdalnLUT {
    public let basis: MLXArray    // [rank, timeEmbedDim] fp32
    public let mean: MLXArray     // [timeEmbedDim] fp32（存档用，bias 已折叠）
    public let tTable: MLXArray   // [1025, rank] fp32
    public let rank: Int

    public init(basis: MLXArray, mean: MLXArray, tTable: MLXArray) {
        self.basis = basis
        self.mean = mean
        self.tTable = tTable
        self.rank = basis.shape[0]
    }

    /// 存在 `adaln_basis` + `adaln_t_table` 时启用；否则返回 nil（走原版通路）。
    public static func load(_ w: H3Weights) -> H3AdalnLUT? {
        guard let b = w.get("adaln_basis"), let t = w.get("adaln_t_table") else { return nil }
        let m = w.get("adaln_mean") ?? MLXArray.zeros([b.shape[1]])
        return H3AdalnLUT(basis: b, mean: m, tTable: t)
    }

    /// 时间步 t ∈ [0,1]（即 `TimestepPlan.uniqueT`）→ 查表系数 [m, rank]。
    public func coeffs(_ ts: [Double]) -> MLXArray {
        let n = tTable.shape[0]
        var idx = [Int32](repeating: 0, count: ts.count)
        for (i, t) in ts.enumerated() {
            let r = Int((t * Double(n - 1)).rounded())
            idx[i] = Int32(min(max(r, 0), n - 1))
        }
        return tTable.take(MLXArray(idx, [ts.count]), axis: 0)
    }
}

// MARK: - AdaLN

public final class H3AdalnW {
    public var linear: MfLinear   // [timeEmbedDim, expand·modalities·hidden]
    public var bias: MLXArray?    // [out]
    public let expand: Int        // 6 block / 2 final
    public let modalities: Int    // 3 block / 1 final

    public init(linear: MfLinear, bias: MLXArray?, expand: Int, modalities: Int) {
        self.linear = linear
        self.bias = bias
        self.expand = expand
        self.modalities = modalities
    }

    public static func load(_ w: H3Weights, prefix: String, timeEmbedDim: Int,
                            expand: Int, modalities: Int, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3AdalnW {
        // 剪枝/融合权重把 adaln_proj 折叠成 [out, rank]（rank ≪ timeEmbedDim）；
        // dense 分支的入维以权重视图为准，量化分支仍需 timeEmbedDim 解析打包几何。
        let resolvedIn: Int
        if w.contains(prefix + ".linear.scales") {
            resolvedIn = timeEmbedDim
        } else if let rw = w.get(prefix + ".linear.weight") {
            resolvedIn = rw.shape[1]
        } else {
            resolvedIn = timeEmbedDim
        }
        let lin = try MfLinear.load(w, prefix: prefix + ".linear", inFeatures: resolvedIn, dtype: dtype)
        // MfLinear.load 已加载 prefix+".linear.bias" 并在 forward 内部加一次，
        // 此处若再传同一 bias 会双加，导致 AdaLN 调制整体偏移、输出全噪点。
        return H3AdalnW(linear: lin, bias: nil, expand: expand, modalities: modalities)
    }

    /// t_emb [m, timeEmbedDim] → `expand` arrays, each [m·modalities, hidden].
    /// Row t_row·modalities + tag is the modulation for (timestep, stream).
    /// y layout: [m, expand·modalities·hidden] → [m·modalities, expand·hidden].
    /// 权重行序为 [mod, e]（mod 慢 e 快），与 Zig/ComfyUI AdalnProj.view 一致：
    /// 行 r = t_row*modalities + tag，块 k 即 expand 维的第 k 个 hidden 块。
    public func forward(_ tEmb: MLXArray, applySilu: Bool = true) -> [MLXArray] {
        let inp = applySilu ? silu(tEmb) : tEmb
        // MfLinear.forward 内部已应用 lora（loraAdd），此处不得重复应用。
        var y = linear.forward(inp)
        if let b = bias {
            y = y + b.asType(y.dtype)
        }
        let m = tEmb.shape[0]
        let hidden = linear.outDim / (expand * modalities)
        let rows = m * modalities
        let cols = expand * hidden
        let v = y.reshaped(rows, cols)
        return H3TensorOps.splitEqual(v, expand, axis: 1)
    }
}

/// Precomputed per-block modulation tables (from precomputeAdaln).
public struct H3AdalnTables {
    public let ts: [Double]
    public let blocks: [[MLXArray]] // [layer][6] each [t·modalities, hidden]
    public let final: [MLXArray]      // [2] each [t, hidden]
}

// MARK: - Attention / MLP weights

public final class H3AttnW {
    public var qkv: MfLinear   // [S, hidden] → [S, 3·inner]
    public var qNorm: MLXArray
    public var kNorm: MLXArray
    public var out: MfLinear   // [S, inner] → [S, hidden]
    public var qkvLora = LoraSlot()
    public var outLora = LoraSlot()

    public init(qkv: MfLinear, qNorm: MLXArray, kNorm: MLXArray, out: MfLinear) {
        self.qkv = qkv
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.out = out
    }

    public static func load(_ w: H3Weights, prefix: String, cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3AttnW {
        let qkv = try MfLinear.load(w, prefix: prefix + ".qkv_proj", inFeatures: cfg.hiddenSize, dtype: dtype)
        guard let qn = w.get(prefix + ".q_norm.weight"), let kn = w.get(prefix + ".k_norm.weight") else {
            throw H3Error.missingWeight(prefix + ".q_norm.weight")
        }
        let out = try MfLinear.load(w, prefix: prefix + ".out_proj", inFeatures: cfg.innerDim, dtype: dtype)
        return H3AttnW(qkv: qkv, qNorm: qn.asType(dtype), kNorm: kn.asType(dtype), out: out)
    }

    /// SDPA attention. x [S, hidden]; rope == nil disables rotary (refiner).
    public func forward(_ x: MLXArray, cfg: H3Config, rope: RopeTables?, sparse: SparseSpec?) -> MLXArray {
        let n = x.shape[0]
        let h = cfg.numHeads
        let hd = cfg.headDim
        let rotHalf = cfg.rotDim / 2
        let inner = cfg.innerDim
        let prof = getenv("NA_H3PROF") != nil
        let t0p = CFAbsoluteTimeGetCurrent()

        let qkv = qkv.forward(x)   // MfLinear.forward 内部已应用 qkvLoRA
        if prof { MLX.eval(qkv) }
        let t1p = CFAbsoluteTimeGetCurrent()
        h3DumpRowStats("attn_qkv", qkv, limitRows: 0)
        let parts = H3TensorOps.splitEqual(qkv, 3, axis: 1) // [S, inner] ×3

        var qkvn: [MLXArray] = []
        for (i, p) in parts.enumerated() {
            let v4 = p.reshaped(1, n, h, hd)
            if i == 2 {
                qkvn.append(v4)
                continue
            }
            let nw = i == 0 ? qNorm : kNorm
            let normed = rmsNormLast(v4, weight: nw, eps: cfg.normEps)
            if let rope {
                qkvn.append(applyRopePub(normed, cos: rope.cos, sin: rope.sin, rot: rotHalf * 2))
            } else {
                qkvn.append(normed)
            }
        }

        let q = qkvn[0].transposed(0, 2, 1, 3)
        let k = qkvn[1].transposed(0, 2, 1, 3)
        let v = qkvn[2].transposed(0, 2, 1, 3)
        h3DumpRowStats("attn_q_rope", q, limitRows: 0)
        if prof { MLX.eval(q) }
        let t2p = CFAbsoluteTimeGetCurrent()

        let scale = Float(1.0 / sqrt(Double(hd)))
        let S = q.shape[2]
        let useSol = (sparse?.mode ?? .dense) != .dense
        let gFull = Int(sparse?.videoStart ?? 0)
        // ---- 按 query 行分块：稠密 / 稀疏统一入口（不再只在稠密分支生效）----
        // mask=nil 非因果 => 每行 softmax 独立 => 稠密路径分块与整段 SDPA 数值完全一致。
        // 分块只切 query 维，key/value 保持全量；稀疏路径按块内实际 strip 行数传 g，
        // strip 条件行仍走 dense 精确分支，video 行走 SOL tau 路由。
        // 触发阈值 S>35000（15s≈57k 的 scores 会打到 48G 上限；10s≈38k 亦一并分块降内存）。
        // 块长 1792 = 28×64：query 块起点恒为 64 的整数倍，稀疏路径的 video 64 行分块网格
        // 在各块之间相位一致（不再随块号漂移）；稠密/稀疏两条路径统一用同一块长。
        let cs = 1792
        let attn: MLXArray
        if S > 35_000 {
            var chunkOuts: [MLXArray] = []
            chunkOuts.reserveCapacity((S + cs - 1) / cs)
            if useSol {
                // （历史方案记录，已被 v7 分块取代）本块实际包含的 strip 条件行数：整条序列的 g
                // 只对 [0,g) 生效，块内按交集折算（gBlk = max(0, min(hi, gFull) - lo)）。
                // 严禁把 gFull 直接套到每一块——否则 video 块的前 g 行会被误判为 strip 行走 dense，
                // 稀疏语义被破坏。该折算只在「q 只切一块、K/V 仍全量」时成立，而 v6 内核的
                // K/V 行数由 q 行数反推，这样调用 K/V 会行错位并触发 reshape 断言，故 v7 改为
                // 下方 strip 段 / video 段分开切。
                // 块内含 video 行（gBlk==0 即整块 video）：SOL 内核支持 g==0
                // 非稀疏路径，或整块都是 strip 行（SOL 对 strip 行恒为 dense，等价直算）
                // v7 分块（SOL）：query 块必须落在 video 段的 64 行块边界上，故 strip 段与
                // video 段分开切——strip 段仍走 dense 精确分支，video 段交给 v7 分块内核
                // （K/V 全量、prepareKV 整层只算一次、attendVideo 按块复用）。
                // 顺序拼接即还原整条序列，故 attn 与整段 SOL 数值一致（selfTest Case S 断言）。
                if gFull > 0 {
                    for lo in stride(from: 0, to: gFull, by: cs) {
                        let hi = min(lo + cs, gFull)
                        // strip 条件行恒为 dense 精确（SOL 对 strip 行同语义，等价直算）
                        chunkOuts.append(scaledDotProductAttention(
                            queries: q[0..., 0..., lo..<hi, 0...], keys: k, values: v,
                            scale: scale, mask: nil))
                    }
                }
                let Lvid = S - gFull
                if Lvid > 0 {
                    let kvPrepared = H3SolAttn.prepareKV(k: k, v: v, gFull: gFull)
                    var off = 0
                    while off < Lvid {
                        let len = min(cs, Lvid - off)
                        chunkOuts.append(H3SolAttn.attendVideo(
                            q: q[0..., 0..., (gFull + off)..<(gFull + off + len), 0...],
                            kv: kvPrepared, qOffVideo: off, scale: scale))
                        off += len
                    }
                }
            } else {
                // 稠密：分块只切 query 维，key/value 保持全量；mask=nil 非因果 => 每行
                // softmax 独立 => 分块与整段 SDPA 数值完全一致。
                for lo in stride(from: 0, to: S, by: cs) {
                    let hi = min(lo + cs, S)
                    let qc = q[0..., 0..., lo..<hi, 0...]
                    chunkOuts.append(scaledDotProductAttention(
                        queries: qc, keys: k, values: v, scale: scale, mask: nil))
                }
            }
            attn = concatenated(chunkOuts, axis: 2)
        } else if useSol {
            // Sol-Attn（网络算子-稀疏注意力-SOL通用.swift）: 块质心 tau 自适应路由 + tail pooled 分母（v6 默认内核）
            attn = H3SolAttn.forward(q: q, k: k, v: v,
                                     g: gFull, scale: scale,
                                     useV6: true)   // h3sol_flash_v6: BQ64 兄弟合并（256 threads），数值与 v3 逐位一致，tau1-tail 156ms vs v3 196ms
        } else {
            attn = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        }
        if useSol, getenv("NA_H3ATTCHK") != nil, let sparse {
            let refA = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
            let d = abs(attn - refA).asType(.float32)      // [1,H,S,hd]
            let gI = Int(sparse.videoStart)
            let stripSeg = d[0, 0..<cfg.numHeads, 0..<gI, 0..<hd]
            let vidSeg = d[0, 0..<cfg.numHeads, gI..<S, 0..<hd]
            let stripMean = stripSeg.mean().item(Float.self)
            let vidMean = vidSeg.mean().item(Float.self)
            let vidMax = vidSeg.max().item(Float.self)
            print(String(format: "[ATTCHK] SOL L=%02d stripMean=%.6f videoMean=%.5f videoMax=%.3f S=%d g=%d", sparseLayerNo, stripMean, vidMean, vidMax, S, gI))
            sparseLayerNo += 1
            fflush(stdout)
            if getenv("NA_H3ATTCHK_ABORT") != nil {
                // value = max layers to probe before exiting (default 1)
                let lim = ProcessInfo.processInfo.environment["NA_H3ATTCHK_ABORT"].flatMap(Int.init) ?? 1
                if sparseLayerNo >= lim { exit(0) }
            }
        }
        if getenv("NA_H3CDF") != nil, let geo = cdfGeo {
            cdfLayerNo += 1
            h3RunCDF(q: q, k: k, hd: hd, S: q.shape[2], geo: geo, layer: cdfLayerNo)
        }
        if prof { MLX.eval(attn) }
        let t3p = CFAbsoluteTimeGetCurrent()

        let back = attn.transposed(0, 2, 1, 3).reshaped(n, inner)
        h3DumpRowStats("attn_sdpa_out", back, limitRows: 0)
        let outv = out.forward(back)   // MfLinear.forward 内部已应用 outLoRA
        h3DumpRowStats("attn_out_proj", outv, limitRows: 0)
        if prof {
            profLayerNo += 1
            MLX.eval(outv)
            let t4p = CFAbsoluteTimeGetCurrent()
            let dq = (t1p - t0p) * 1000, dn = (t2p - t1p) * 1000
            let da = (t3p - t2p) * 1000, do2 = (t4p - t3p) * 1000
            let strMode: String
            if let s = sparse, s.mode != .dense { strMode = "sol" }
            else { strMode = "dense" }
            print(String(format: "[PROF] L%02d xDt=%@ qkv=%.0fms ropeN=%.0fms attn(%@)=%.0fms out=%.0fms tot=%.0fms",
                         profLayerNo, String(describing: x.dtype), dq, dn, strMode, da, do2, (t4p - t0p) * 1000))
            fflush(stdout)
            if let lim = ProcessInfo.processInfo.environment["NA_H3PROF_ABORT"].flatMap(Int.init) {
                if profLayerNo >= lim { exit(0) }
            }
        }
        return outv
    }
}

public final class H3MlpW {
    public var fc1: MfLinear // [S, hidden] → [S, 2·ffn] (gate+up fused)
    public var fc2: MfLinear // [S, ffn] → [S, hidden]
    public var fc1Lora = LoraSlot()
    public var fc2Lora = LoraSlot()

    public init(fc1: MfLinear, fc2: MfLinear) {
        self.fc1 = fc1
        self.fc2 = fc2
    }

    public static func load(_ w: H3Weights, prefix: String, cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3MlpW {
        let fc1 = try MfLinear.load(w, prefix: prefix + ".fc1", inFeatures: cfg.hiddenSize, dtype: dtype)
        let fc2 = try MfLinear.load(w, prefix: prefix + ".fc2", inFeatures: cfg.ffnHidden, dtype: dtype)
        return H3MlpW(fc1: fc1, fc2: fc2)
    }

    /// SwiGLU: silu(gate)·up, gate and up fused in fc1's output.
    public func forward(_ x: MLXArray) -> MLXArray {
        // fc1/fc2 的 MfLinear.forward 内部已应用各自 LoRA，这里不再重复应用。
        let y = fc1.forward(x)
        h3DumpRowStats("mlp_fc1", y, limitRows: 0)
        let halves = H3TensorOps.splitEqual(y, 2, axis: 1)
        let gated = silu(halves[0]) * halves[1]
        h3DumpRowStats("mlp_gated", gated, limitRows: 0)
        let o = fc2.forward(gated)
        h3DumpRowStats("mlp_fc2_out", o, limitRows: 0)
        return o
    }
}

public final class H3BlockW {
    public var norm1: MLXArray
    public var norm2: MLXArray
    public var attn: H3AttnW
    public var mlp: H3MlpW
    public var adaln: H3AdalnW?

    public init(norm1: MLXArray, norm2: MLXArray, attn: H3AttnW, mlp: H3MlpW, adaln: H3AdalnW?) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.attn = attn
        self.mlp = mlp
        self.adaln = adaln
    }

    public static func load(_ w: H3Weights, idx: Int, cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3BlockW {
        let pfx = "blocks.\(idx)"
        guard let n1 = w.get(pfx + ".norm1.weight"), let n2 = w.get(pfx + ".norm2.weight") else {
            throw H3Error.missingWeight(pfx + ".norm1.weight")
        }
        let attn = try H3AttnW.load(w, prefix: pfx + ".attn", cfg: cfg, dtype: dtype)
        let mlp = try H3MlpW.load(w, prefix: pfx + ".mlp", cfg: cfg, dtype: dtype)
        let adaln = try H3AdalnW.load(w, prefix: pfx + ".adaln_proj", timeEmbedDim: cfg.timeEmbedDim,
                                      expand: 6, modalities: 3, dtype: dtype)
        return H3BlockW(norm1: n1.asType(dtype), norm2: n2.asType(dtype), attn: attn, mlp: mlp, adaln: adaln)
    }
}

public final class H3RefinerBlockW {
    public var norm1: MLXArray
    public var norm2: MLXArray
    public var attn: H3AttnW
    public var mlp: H3MlpW

    public init(norm1: MLXArray, norm2: MLXArray, attn: H3AttnW, mlp: H3MlpW) {
        self.norm1 = norm1
        self.norm2 = norm2
        self.attn = attn
        self.mlp = mlp
    }

    public static func load(_ w: H3Weights, idx: Int, cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3RefinerBlockW {
        let pfx = "token_refiner.blocks.\(idx)"
        guard let n1 = w.get(pfx + ".norm1.weight"), let n2 = w.get(pfx + ".norm2.weight") else {
            throw H3Error.missingWeight(pfx + ".norm1.weight")
        }
        let attn = try H3AttnW.load(w, prefix: pfx + ".attn", cfg: cfg, dtype: dtype)
        let mlp = try H3MlpW.load(w, prefix: pfx + ".mlp", cfg: cfg, dtype: dtype)
        return H3RefinerBlockW(norm1: n1.asType(dtype), norm2: n2.asType(dtype), attn: attn, mlp: mlp)
    }
}

// MARK: - DiT (Model)

public struct H3DiTOutput {
    public let video: MLXArray // [nVideoRows, videoPatchDim] (negative velocity)
    public let audio: MLXArray // [nAudioRows, audioLatentDim] (negative velocity, slope-scaled)
}

public final class H3DiT {
    public let cfg: H3Config
    public let dtype: DType

    // fp32 islands
    public var videoPatchW: MLXArray  // [videoPatchDim, hidden]
    public var videoPatchB: MLXArray? // [hidden]
    public var audioPatchW: MLXArray  // [audioLatentDim, hidden]
    public var audioPatchB: MLXArray? // [hidden]
    public var conditionProj: MfLinear? // [textDim → hidden]
    public var conditionBias: MLXArray?
    public var teInW: MLXArray
    public var teInB: MLXArray?
    public var teOutW: MLXArray
    public var teOutB: MLXArray?
    public var invFreq: MLXArray     // [16] fp32
    public var finalNorm: MLXArray   // [hidden]
    public var videoOutW: MLXArray   // [hidden, videoPatchDim]
    public var videoOutB: MLXArray?
    public var audioOutW: MLXArray   // [hidden, audioLatentDim]
    public var audioOutB: MLXArray?

    public var refiner: [H3RefinerBlockW]
    public var blocks: [H3BlockW]
    public var finalAdaln: H3AdalnW?
    public var refinerFinalNorm: MLXArray?

    public var adalnTables: H3AdalnTables?
    /// 剪枝/融合权重的查表式时间调制（非 nil 时替代 time_embedder 通路）。
    public var adalnLUT: H3AdalnLUT?
    public var sparsePolicy: SparsePolicy = .off


    public init(cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) {
        self.cfg = cfg
        self.dtype = dtype
        self.videoPatchW = MLXArray(0)
        self.videoPatchB = nil
        self.audioPatchW = MLXArray(0)
        self.audioPatchB = nil
        self.conditionProj = nil
        self.conditionBias = nil
        self.teInW = MLXArray(0)
        self.teInB = nil
        self.teOutW = MLXArray(0)
        self.teOutB = nil
        self.invFreq = MLXArray(0)
        self.finalNorm = MLXArray(0)
        self.videoOutW = MLXArray(0)
        self.videoOutB = nil
        self.audioOutW = MLXArray(0)
        self.audioOutB = nil
        self.refiner = []
        self.blocks = []
        self.finalAdaln = nil
    }

    public var usesPrecomputedAdaln: Bool { adalnTables != nil }

    // MARK: load

    public static func load(from w: H3Weights, cfg: H3Config, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3DiT {
        let m = H3DiT(cfg: cfg, dtype: dtype)

        guard let vpw = w.get("video_patch_proj.weight"),
              let apw = w.get("audio_patch_proj.weight"),
              let inv = w.get("rope.inv_freq"),
              let fnorm = w.get("final_layer.norm.weight") ?? w.get("final_norm.weight"),
              let vout = w.get("final_layer.video_out.weight") ?? w.get("video_out_proj.weight"),
              let aout = w.get("final_layer.audio_out.weight") ?? w.get("audio_out_proj.weight") else {
            throw H3Error.badFile("transformer.safetensors missing core projections")
        }
        // 时间调制两条通路：原版 time_embedder（sin/cos + MLP，2688 维输入）
        // 或剪枝/融合权重的查表（8 维系数，见 H3AdalnLUT）。
        m.adalnLUT = H3AdalnLUT.load(w)
        let teIn = w.get("time_embedder.proj_in.weight")
        let teOut = w.get("time_embedder.proj_out.weight")
        if teIn == nil || teOut == nil {
            guard m.adalnLUT != nil else {
                throw H3Error.badFile("transformer.safetensors: neither time_embedder nor adaln_t_table present")
            }
        }
        // safetensors 中 dense 投影权重为 [out, in]，denseLinear 按 x.matmul(w) 需 [in, out]，统一转置。
        m.videoPatchW = vpw.transposed(1, 0)
        m.videoPatchB = w.get("video_patch_proj.bias")
        m.audioPatchW = apw.transposed(1, 0)
        m.audioPatchB = w.get("audio_patch_proj.bias")
        if let teIn, let teOut {
            m.teInW = teIn.transposed(1, 0)
            m.teInB = w.get("time_embedder.proj_in.bias")
            m.teOutW = teOut.transposed(1, 0)
            m.teOutB = w.get("time_embedder.proj_out.bias")
        }
        m.invFreq = inv
        m.finalNorm = fnorm.asType(dtype)
        m.videoOutW = vout.transposed(1, 0)
        m.videoOutB = w.get("final_layer.video_out.bias") ?? w.get("video_out_proj.bias")
        m.audioOutW = aout.transposed(1, 0)
        m.audioOutB = w.get("final_layer.audio_out.bias") ?? w.get("audio_out_proj.bias")

        if w.contains("condition_proj.weight") {
            m.conditionProj = try MfLinear.load(w, prefix: "condition_proj", inFeatures: cfg.textDim, dtype: dtype)
            m.conditionBias = w.get("condition_proj.bias")
        }

        // Refiner (dense, unquantized in the repack)
        m.refiner = []
        var ri = 0
        while w.contains("token_refiner.blocks.\(ri).attn.qkv_proj.weight") ||
              w.contains("token_refiner.blocks.\(ri).attn.qkv_proj.scales") {
            m.refiner.append(try H3RefinerBlockW.load(w, idx: ri, cfg: cfg, dtype: dtype))
            ri += 1
        }
        if w.contains("token_refiner.final_norm.weight") {
            m.refinerFinalNorm = w.get("token_refiner.final_norm.weight")?.asType(dtype)
        }

        m.blocks = []
        for i in 0..<cfg.numLayers {
            m.blocks.append(try H3BlockW.load(w, idx: i, cfg: cfg, dtype: dtype))
            if i == 0 {
                let b = m.blocks[0]
                b.attn.qkv.diagName = "b0.qkv"
                b.attn.out.diagName = "b0.out"
                b.mlp.fc1.diagName = "b0.fc1"
                b.mlp.fc2.diagName = "b0.fc2"
            }
            if (i + 1) % 5 == 0 || i == cfg.numLayers - 1 {
                var info = task_vm_info_data_t()
                var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
                let kr = withUnsafeMutablePointer(to: &info) { p in
                    p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
                        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ip, &count)
                    }
                }
                let rssMB = kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1024.0 / 1024.0 : 0
                let mlxMB = Double(MLX.Memory.activeMemory) / 1024.0 / 1024.0
                print("[H3DiT.load] layers \(i + 1)/\(cfg.numLayers)：RSS \(String(format: "%.0f", rssMB))MB，MLX active \(String(format: "%.0f", mlxMB))MB")
                fflush(stdout)
            }
        }

        if w.contains("final_layer.adaln_proj.linear.weight") || w.contains("final_adaln_proj.linear.weight") {
            let fap = w.contains("final_layer.adaln_proj.linear.weight") ? "final_layer.adaln_proj" : "final_adaln_proj"
            m.finalAdaln = try H3AdalnW.load(w, prefix: fap, timeEmbedDim: cfg.timeEmbedDim,
                                             expand: 2, modalities: 1, dtype: dtype)
        }
        return m
    }

    // MARK: LoRA (turbo_lora.safetensors)

    /// Attach LoRA adapters from files (Kohya-flat or diffusion_model prefixes).
    /// Returns the number of modules patched. Reuses H3Common's H3LoraFile parser
    /// and pushes each adapter onto the target MfLinear's LoraSlot (module naming:
    /// blocks.{i}.attn.qkv_proj | attn.out_proj | mlp.fc1 | mlp.fc2).
    /// 仅当 LoRA 适配器的输入维与目标 linear 一致时才挂载。
    /// 剪枝/融合权重把 adaln_proj 折叠成 [out, rank]（rank ≪ timeEmbedDim），
    /// 与原版 LoRA（输入维 = timeEmbedDim）不兼容，硬挂会在前向 matmul 处形状崩溃。
    @discardableResult
    static func attachLoRA(_ lin: MfLinear, _ refs: [LoraRef]) -> Bool {
        if let a = refs.first?.a, a.shape.count >= 2, a.shape[0] != lin.inDim {
            return false
        }
        lin.lora.set(refs)
        return true
    }

    @discardableResult
    public func attachLoras(from urls: [URL], scales: [Double] = []) throws -> Int {
        var patched = 0
        var skipped = 0
        for (i, url) in urls.enumerated() {
            let scale = scales.indices.contains(i) ? Float(scales[i]) : 1.0
            let f = try H3LoraFile.load(from: url)
            for e in f.entries {
                let refs = [LoraRef(a: e.a.asType(dtype), b: e.b.asType(dtype), scale: scale)]
                // 1) blocks.{i}.attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2 / adaln_proj.linear
                if let parsed = Self.parseLoraModule(e.module), parsed.layer < blocks.count {
                    let b = blocks[parsed.layer]
                    let ok: Bool
                    switch parsed.kind {
                    case .qkv: ok = Self.attachLoRA(b.attn.qkv, refs)
                    case .out: ok = Self.attachLoRA(b.attn.out, refs)
                    case .fc1: ok = Self.attachLoRA(b.mlp.fc1, refs)
                    case .fc2: ok = Self.attachLoRA(b.mlp.fc2, refs)
                    case .adaln: ok = b.adaln.map { Self.attachLoRA($0.linear, refs) } ?? false
                    }
                    if ok { patched += 1 } else { skipped += 1 }
                    continue
                }
                // 2) token_refiner.blocks.{i}.attn.qkv_proj / attn.out_proj / mlp.fc1 / mlp.fc2
                if let rp = Self.parseRefinerLoraModule(e.module), rp < refiner.count {
                    let rb = refiner[rp]
                    let ok: Bool
                    switch Self.refinerKind(of: e.module) {
                    case .qkv: ok = Self.attachLoRA(rb.attn.qkv, refs)
                    case .out: ok = Self.attachLoRA(rb.attn.out, refs)
                    case .fc1: ok = Self.attachLoRA(rb.mlp.fc1, refs)
                    case .fc2: ok = Self.attachLoRA(rb.mlp.fc2, refs)
                    case .adaln: ok = false
                    }
                    if ok { patched += 1 } else { skipped += 1 }
                    continue
                }
                // 3) final_layer.adaln_proj.linear (or final_adaln_proj.linear)
                if e.module.hasPrefix("final_layer.adaln_proj") || e.module.hasPrefix("final_adaln_proj") {
                    if let fa = finalAdaln, Self.attachLoRA(fa.linear, refs) { patched += 1 } else { skipped += 1 }
                    continue
                }
                // 4) legacy flat "adaln_proj" catch-all (rare)
                if e.module == "adaln_proj" {
                    for b in blocks {
                        if let ad = b.adaln, Self.attachLoRA(ad.linear, refs) { patched += 1 }
                    }
                }
            }
        }
        if skipped > 0 {
            print("[H3DiT.attachLoras] 跳过 \(skipped) 个形状不匹配的 LoRA 模块（剪枝后折叠的 adaln_proj 通路）"); fflush(stdout)
        }
        return patched
    }

    private static func parseLoraModule(_ module: String) -> (layer: Int, kind: LoRAKind)? {
        // "blocks.3.attn.qkv_proj" / "blocks.3.mlp.fc1" / "blocks.3.adaln_proj.linear"
        let parts = module.split(separator: ".")
        guard parts.count >= 4, parts[0] == "blocks", let layer = Int(parts[1]) else { return nil }
        switch (parts[2], parts.count >= 4 ? parts[3] : "") {
        case ("attn", "qkv_proj"): return (layer, .qkv)
        case ("attn", "out_proj"): return (layer, .out)
        case ("mlp", "fc1"): return (layer, .fc1)
        case ("mlp", "fc2"): return (layer, .fc2)
        case ("adaln_proj", "linear"): return (layer, .adaln)
        default: return nil
        }
    }

    private static func parseRefinerLoraModule(_ module: String) -> Int? {
        // "token_refiner.blocks.3.attn.qkv_proj"
        let parts = module.split(separator: ".")
        guard parts.count == 5, parts[0] == "token_refiner", parts[1] == "blocks",
              let layer = Int(parts[2]), parts[3] == "attn" || parts[3] == "mlp" else { return nil }
        let tail = "\(parts[3]).\(parts[4])"
        guard ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"].contains(tail) else { return nil }
        return layer
    }

    private static func refinerKind(of module: String) -> LoRAKind {
        let parts = module.split(separator: ".")
        switch (parts[3], parts[4]) {
        case ("attn", "qkv_proj"): return .qkv
        case ("attn", "out_proj"): return .out
        case ("mlp", "fc1"): return .fc1
        case ("mlp", "fc2"): return .fc2
        default: return .adaln
        }
    }

    private enum LoRAKind { case qkv, out, fc1, fc2, adaln }

    // MARK: AdaLN precompute

    /// Fold AdaLN modulation into tables once per (ts, plan). After this the
    /// per-block adaln weights are no longer touched in the sampling loop.
    public func precomputeAdaln(ts: [Double]) {
        // 时间调制两条通路：查表（pruned/fused，8 维系数直接喂折叠后的 adaln_proj）
        // 或原版 time_embedder（2688 维嵌入，需 silu）。
        let tEmb: MLXArray
        let applySilu: Bool
        if let lut = adalnLUT {
            tEmb = lut.coeffs(ts).asType(dtype)
            applySilu = false
        } else {
            let embedder = H3TimeEmbedder(inputDim: 256, outDim: cfg.timeEmbedDim,
                                          projInW: teInW, projInB: teInB,
                                          projOutW: teOutW, projOutB: teOutB)
            tEmb = embedder.forward(ts).asType(dtype)
            applySilu = true
        }

        var blocksMods: [[MLXArray]] = []
        for b in blocks {
            if let ad = b.adaln {
                blocksMods.append(ad.forward(tEmb, applySilu: applySilu).map { $0.asType(dtype) })
            } else {
                // Ablated/absent: zeros.
                let zero = MLXArray.zeros([ts.count, cfg.hiddenSize]).asType(dtype)
                blocksMods.append((0..<6).map { _ in zero })
            }
        }
        let finalMods = finalAdaln?.forward(tEmb, applySilu: applySilu) ?? []
        adalnTables = H3AdalnTables(ts: ts, blocks: blocksMods, final: finalMods)
        if getenv("NA_H3PROF") != nil {
            print(String(format: "[PROF] adaln: tEmb=%@ blk0_0=%@ zeroDt=%@", String(describing: tEmb.dtype), String(describing: blocksMods[0][0].dtype), String(describing: MLXArray.zeros([1, cfg.hiddenSize]).dtype))); fflush(stdout)
        }
    }

    // MARK: Text refine

    /// Qwen states [L, textDim] → refined [L, hidden]. Identity when dims match.
    public func refineText(_ textStates: MLXArray) -> MLXArray {
        let hidden = textStates.shape[1]
        if hidden == cfg.hiddenSize { return textStates }
        var h: MLXArray
        if let cp = conditionProj {
            h = cp.forward(textStates)
            if let b = conditionBias {
                h = h + b.asType(h.dtype)
            }
        } else {
            h = textStates
        }
        for b in refiner {
            let n1 = rmsNormLast(h, weight: b.norm1, eps: cfg.normEps)
            let at = b.attn.forward(n1, cfg: cfg, rope: nil, sparse: nil)
            h = h + at
            let n2 = rmsNormLast(h, weight: b.norm2, eps: cfg.normEps)
            let mo = b.mlp.forward(n2)
            h = h + mo
        }
        if let rn = refinerFinalNorm {
            h = rmsNormLast(h, weight: rn, eps: cfg.normEps)
        }
        return h
    }

    // MARK: Forward

    /// Full packed forward. textStates [textLen, hidden]; videoRows/audioRows are
    /// fp32 latent rows (cond rows already concatenated in order by the caller).
    public func forward(layout: PackedLayout,
                        plan: TimestepPlan,
                        textStates: MLXArray,
                        videoRows: MLXArray,
                        audioRows: MLXArray,
                        rope: RopeTables,
                        sigmaV: Double,
                        shiftV: Double,
                        shiftA: Double,
                        attnBcast: H3AttnBroadcast?,
                        attnRefresh: Bool) -> H3DiTOutput {
        let cfg = self.cfg

        // fp32 latent streams → bf16 embeddings
        let ve32 = H3TensorOps.denseLinear(videoRows, videoPatchW, videoPatchB)
        let videoEmbed = ve32.asType(dtype)
        let ae32 = H3TensorOps.denseLinear(audioRows, audioPatchW, audioPatchB)
        let audioEmbed = ae32.asType(dtype)

        // Assemble the packed stream segment by segment.
        var pieces: [MLXArray] = []
        var voff = 0
        var aoff = 0
        for seg in layout.segments {
            let n = Int(seg.end - seg.start)
            switch seg.kind.stream {
            case .text:
                // The reference emits the whole refined text block contiguously.
                pieces.append(textStates.asType(dtype))
            case .video:
                pieces.append(H3TensorOps.sliceRows(videoEmbed, voff, voff + n))
                voff += n
            case .audio:
                pieces.append(H3TensorOps.sliceRows(audioEmbed, aoff, aoff + n))
                aoff += n
            }
        }
        for (pi, p) in pieces.enumerated() {
            h3DumpRowStats("h_piece[\(pi)]", p, limitRows: 0)
        }
        var h = concatenated(pieces, axis: 0)
        h3DumpRowStats("h_assembled", h, limitRows: 0)
        if getenv("NA_H3PROF") != nil {
            print(String(format: "[PROF] asm: text=%@ vEmbed=%@ h=%@", String(describing: textStates.dtype), String(describing: videoEmbed.dtype), String(describing: h.dtype))); fflush(stdout)
        }
        if ProcessInfo.processInfo.environment["NA_DUMP_LATENT"] == "1", !FileManager.default.fileExists(atPath: "/tmp/h3_h_assembled.bin") {
            MLX.eval(h)
            let hf = h.asType(.float32).asArray(Float.self)
            let data = hf.withUnsafeBufferPointer { Data(buffer: $0) }
            try? data.write(to: URL(fileURLWithPath: "/tmp/h3_h_assembled.bin"))
        }

        for (bi, b) in blocks.enumerated() {
            // Modulations [t·modalities, hidden]; rows addressed by run.modRow.
            let mods: [MLXArray]? = adalnTables?.blocks[bi]
            let shiftMSA: MLXArray
            let scaleMSA: MLXArray
            let gateMSA: MLXArray
            let shiftMLP: MLXArray
            let scaleMLP: MLXArray
            let gateMLP: MLXArray
            if let mods {
                if bi == 0 {
                    h3DumpRowStats("mod_b0_shiftMSA", mods[0], limitRows: 0)
                    h3DumpRowStats("mod_b0_scaleMSA", mods[1], limitRows: 0)
                    h3DumpRowStats("mod_b0_gateMSA", mods[2], limitRows: 0)
                    h3DumpRowStats("mod_b0_gateMLP", mods[5], limitRows: 0)
                    if getenv("NA_H3PROF") != nil {
                        print(String(format: "[PROF] bi0 dtype h=%@ mods0=%@ n1=%@", String(describing: h.dtype), String(describing: mods[0].dtype), String(describing: rmsNormLast(h, weight: b.norm1, eps: cfg.normEps).dtype))); fflush(stdout)
                    }
                }
                shiftMSA = mods[0]
                scaleMSA = mods[1]
                gateMSA = mods[2]
                shiftMLP = mods[3]
                scaleMLP = mods[4]
                gateMLP = mods[5]
            } else {
                fatalError("H3DiT.forward requires precomputed AdaLN tables; call precomputeAdaln first")
            }

            // Attention branch (PAB cache)
            var at: MLXArray
            let cacheable = attnBcast != nil
            if cacheable, !attnRefresh, attnBcast?.blocks[bi] != nil {
                at = attnBcast!.blocks[bi]!
            } else {
                let n1 = rmsNormLast(h, weight: b.norm1, eps: cfg.normEps)
                let m1 = modScaleShift(n1, shift: shiftMSA, scale: scaleMSA, runs: plan.runs)
                if bi == 0 { h3DumpRowStats("m1_b0", m1, limitRows: 0) }
                let smode = sparseModeForLayer(bi, nLayers: blocks.count, policy: sparsePolicy)
                let fDim = UInt32((Int(layout.latentH) / Int(H3Const.patchH)) * (Int(layout.latentW) / Int(H3Const.patchH)))
                if getenv("NA_H3CDF") != nil, cdfGeo == nil {
                    cdfGeo = (Int(layout.videoSegment.start), Int(layout.latentT), Int(fDim))
                }
                let spec: SparseSpec? = smode == .dense ? nil :
                    SparseSpec(mode: smode,
                                        t: layout.latentT,
                                        f: fDim,
                                        videoStart: layout.videoSegment.start)
                at = b.attn.forward(m1, cfg: cfg, rope: rope, sparse: spec)
                if cacheable {
                    attnBcast?.blocks[bi] = at
                }
            }
            h = modGate(h, gate: gateMSA, other: at, runs: plan.runs)

            let n2 = rmsNormLast(h, weight: b.norm2, eps: cfg.normEps)
            let m2 = modScaleShift(n2, shift: shiftMLP, scale: scaleMLP, runs: plan.runs)
            let mo = b.mlp.forward(m2)
            h = modGate(h, gate: gateMLP, other: mo, runs: plan.runs)
            if bi == 0 || bi == 13 || bi == blocks.count - 1 {
                h3DumpRowStats("h_block\(bi)", h, limitRows: 0)
            }
        }

        // Final layer: single modality, rows = t_row.
        var fmods: [MLXArray] = []
        if let tables = adalnTables {
            fmods = tables.final // [2] each [t, hidden]
        }
        let fShift = fmods.isEmpty ? MLXArray.zeros([1, cfg.hiddenSize]) : fmods[0]
        let fScale = fmods.isEmpty ? MLXArray.ones([1, cfg.hiddenSize]) : fmods[1]

        let vseg = layout.videoSegment
        let aseg = layout.audioSegment
        func rowFor(_ start: UInt32) -> Int {
            for r in plan.runs where r.start <= start && start < r.end {
                return Int(r.modRow / 3)
            }
            return plan.runs.last.map { Int($0.modRow / 3) } ?? 0
        }
        let vRow = rowFor(vseg.start)
        let aRow = rowFor(aseg.start)

        let vout = finalHead(h, seg: vseg, row: vRow, shift: fShift, scale: fScale,
                             w: videoOutW, b: videoOutB)
        h3DumpRowStats("h_final_pre", h, limitRows: 0)
        h3DumpRowStats("vout_raw", vout, limitRows: 0)
        let aoutRaw = finalHead(h, seg: aseg, row: aRow, shift: fShift, scale: fScale,
                                w: audioOutW, b: audioOutB)

        // Negated velocities; audio scaled by dσa/dσv.
        let slope = timeShiftSlope(sigmaV, fromShift: shiftV, toShift: shiftA)
        return H3DiTOutput(video: -vout, audio: aoutRaw * MLXArray.scalar(Float(-slope), like: aoutRaw))
    }

    private func finalHead(_ h: MLXArray, seg: Segment, row: Int,
                           shift: MLXArray, scale: MLXArray,
                           w: MLXArray, b: MLXArray?) -> MLXArray {
        let part = H3TensorOps.sliceRows(h, Int(seg.start), Int(seg.end))
        let nn = rmsNormLast(part, weight: finalNorm, eps: cfg.normEps)
        let sc = H3TensorOps.sliceRows(scale, row, row + 1)
        let sh = H3TensorOps.sliceRows(shift, row, row + 1)
        let m = nn * (sc + MLXArray(1.0)) + sh
        let f = m.asType(.float32)
        return H3TensorOps.denseLinear(f, w, b)
    }
}

// MARK: - Attention broadcast cache
// H3AttnBroadcast 类与 PAB 刷新调度已抽至独立文件 H3AttnBroadcast.swift，此处不再定义。

// MARK: - Modulation helpers (mirror Zig modScaleShift / modGate)

public func modScaleShift(_ x: MLXArray, shift: MLXArray, scale: MLXArray,
                          runs: [ModRun]) -> MLXArray {
    var pieces: [MLXArray] = []
    pieces.reserveCapacity(runs.count)
    for r in runs {
        let seg = H3TensorOps.sliceRows(x, Int(r.start), Int(r.end))
        let sc = H3TensorOps.sliceRows(scale, Int(r.modRow), Int(r.modRow + 1))
        let sh = H3TensorOps.sliceRows(shift, Int(r.modRow), Int(r.modRow + 1))
        pieces.append(seg * (sc + MLXArray.scalar(1.0, like: scale)) + sh)
    }
    return concatenated(pieces, axis: 0)
}

public func modGate(_ x: MLXArray, gate: MLXArray, other: MLXArray,
                    runs: [ModRun]) -> MLXArray {
    var pieces: [MLXArray] = []
    pieces.reserveCapacity(runs.count)
    for r in runs {
        let xs = H3TensorOps.sliceRows(x, Int(r.start), Int(r.end))
        let os = H3TensorOps.sliceRows(other, Int(r.start), Int(r.end))
        let g = H3TensorOps.sliceRows(gate, Int(r.modRow), Int(r.modRow + 1))
        pieces.append(xs + os * g)
    }
    return concatenated(pieces, axis: 0)
}
