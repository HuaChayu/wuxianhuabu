// H3TextEncoder.swift
// MiniMax H3 — Qwen3-VL-32B text encoder + vision tower.
// Ported from mlx-serve-main/src/minimax_h3.zig (TextEncoder, lines ~2561-3060)
// and src/mage_flow.zig (VisionTower, lines ~2930-3560).
//
// The encoder runs the raw prompt ids (NO special tokens) through 50 decoder
// layers and returns the layer-50 hidden state WITHOUT a final norm (the
// checkpoint ships no `model.norm`; the reference sets
// layer_norm_hidden_state=False). With vision blocks present the LM positions
// become INTERLEAVED 3-axis mRoPE, the tower's three DeepStack taps are ADDED
// into the first three LM layers at the vision rows, and vision rows carry
// adaLN modality tag 0 (widened over the flanking delimiters).
//
// Weight prefix (H3 checkpoint): "model.embed_tokens.weight",
// "model.layers.{i}.{self_attn.q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|
//   input_layernorm|post_attention_layernorm}",
// "model.layers.{i}.mlp.{gate_proj|up_proj|down_proj}",
// vision: "visual.{patch_embed.proj|pos_embed|blocks.{i}...|merger|deepstack_merger_list.{k}}"

import Foundation
import MLX
import MLXFast
import MLXRandom

// MARK: - Config & item types

public struct H3TeConfig {
    public var hidden = 5120
    public var heads = 64
    public var kvHeads = 8
    public var headDim = 128
    public var intermediate = 25600
    public var layers = 50
    public var theta = 5_000_000.0
    public var eps: Float = 1e-6
    public init() {}
    public static let qwen3vl32b = H3TeConfig()
}

/// One 2-frame patch block entering the ViT: [2, 3, H, W] in [-1, 1] at the
/// block's Qwen canvas. A still image repeats itself to fill the temporal
/// patch; a video pair carries two distinct frames.
public struct H3VisionBlock {
    public let frames: MLXArray
    public let grid: Grid
    public init(frames: MLXArray, grid: Grid) {
        self.frames = frames
        self.grid = grid
    }
}

public enum H3PresentItem {
    case text([Int32])
    case vision(H3VisionBlock)
}

public struct H3EncodedPrompt {
    /// [seq, hidden] layer-50 hidden state (NOT final-normed).
    public let hidden: MLXArray
    /// adaLN modality tags per position: 1 = text, 0 = video.
    public let tags: [UInt8]
    public init(hidden: MLXArray, tags: [UInt8]) {
        self.hidden = hidden
        self.tags = tags
    }
}

// MARK: - TextEncoder per-layer weights

public final class H3TeLayerW {
    public let inputLn: MLXArray   // f32
    public let postLn: MLXArray    // f32
    public let qNorm: MLXArray     // f32
    public let kNorm: MLXArray     // f32
    public let qw: MfLinear
    public let kw: MfLinear
    public let vw: MfLinear
    public let ow: MfLinear
    public let gateW: MfLinear
    public let upW: MfLinear
    public let downW: MfLinear

    public init(inputLn: MLXArray, postLn: MLXArray, qNorm: MLXArray, kNorm: MLXArray,
                qw: MfLinear, kw: MfLinear, vw: MfLinear, ow: MfLinear,
                gateW: MfLinear, upW: MfLinear, downW: MfLinear) {
        self.inputLn = inputLn
        self.postLn = postLn
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.qw = qw
        self.kw = kw
        self.vw = vw
        self.ow = ow
        self.gateW = gateW
        self.upW = upW
        self.downW = downW
    }
}

// MARK: - TextEncoder

/// 文本编码器 trace 开关（环境变量 H3_TE_TRACE=1 时逐层 eval 打印）。
public enum H3TE {
    public static var trace: Bool { ProcessInfo.processInfo.environment["H3_TE_TRACE"] == "1" }
}

public final class H3TextEncoder {
    public let cfg: H3TeConfig
    public let dtype: DType
    public let embedTable: MLXArray
    public let layers: [H3TeLayerW]
    /// Loaded lazily (529 tensors, ~1 GB resident) — only for vision requests.
    public var vision: H3VisionTower?

    public static func load(_ w: H3Weights, cfg: H3TeConfig = .qwen3vl32b,
                            dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3TextEncoder {
        guard let raw = w.get("model.embed_tokens.weight") else {
            throw H3Error.missingWeight("model.embed_tokens.weight")
        }
        let embed = raw.asType(dtype)
        var layers: [H3TeLayerW] = []
        layers.reserveCapacity(Int(cfg.layers))
        for i in 0..<Int(cfg.layers) {
            let p = "model.layers.\(i)"
            let lin = try (0..<7).map { j -> MfLinear in
                let (name, inDim): (String, Int)
                switch j {
                case 0: (name, inDim) = ("self_attn.q_proj", cfg.hidden)
                case 1: (name, inDim) = ("self_attn.k_proj", cfg.hidden)
                case 2: (name, inDim) = ("self_attn.v_proj", cfg.hidden)
                case 3: (name, inDim) = ("self_attn.o_proj", cfg.heads * cfg.headDim)
                case 4: (name, inDim) = ("mlp.gate_proj", cfg.hidden)
                case 5: (name, inDim) = ("mlp.up_proj", cfg.hidden)
                default: (name, inDim) = ("mlp.down_proj", cfg.intermediate)
                }
                return try MfLinear.load(w, prefix: "\(p).\(name)", inFeatures: inDim, dtype: dtype)
            }
            let inputLn = w.get("\(p).input_layernorm.weight")?.asType(.float32)
                ?? MLXArray.ones([cfg.hidden], dtype: .float32)
            let postLn = w.get("\(p).post_attention_layernorm.weight")?.asType(.float32)
                ?? MLXArray.ones([cfg.hidden], dtype: .float32)
            let qNorm = w.get("\(p).self_attn.q_norm.weight")?.asType(.float32)
                ?? MLXArray.ones([cfg.headDim], dtype: .float32)
            let kNorm = w.get("\(p).self_attn.k_norm.weight")?.asType(.float32)
                ?? MLXArray.ones([cfg.headDim], dtype: .float32)
            layers.append(H3TeLayerW(inputLn: inputLn, postLn: postLn, qNorm: qNorm, kNorm: kNorm,
                                     qw: lin[0], kw: lin[1], vw: lin[2], ow: lin[3],
                                     gateW: lin[4], upW: lin[5], downW: lin[6]))
        }
        return H3TextEncoder(cfg: cfg, dtype: dtype, embedTable: embed, layers: layers)
    }

    private init(cfg: H3TeConfig, dtype: DType, embedTable: MLXArray, layers: [H3TeLayerW]) {
        self.cfg = cfg
        self.dtype = dtype
        self.embedTable = embedTable
        self.layers = layers
    }

    /// Load the Qwen3-VL vision tower out of the SAME weight map. Separate from
    /// `load` because a text-only request never touches it.
    public func loadVision(_ w: H3Weights) throws {
        if vision != nil { return }
        vision = try H3VisionTower.loadFrom(w, dtype: dtype)
    }

    // MARK: text-only encode

    /// Raw prompt ids -> [seq, hidden] layer-50 hidden state.
    public func encode(_ ids: [Int32]) throws -> MLXArray {
        let seq = ids.count
        let emb = embedTable.take(MLXArray(ids), axis: 0)          // [seq, hidden]
        var x = emb.reshaped(1, seq, cfg.hidden)
        let rope = teBuildRope(seq: seq)
        let mask = teCausalMask(seq: seq, dtype: dtype)
        for layer in layers {
            x = teLayerForward(layer, x: x, mask: mask, rope: rope, seq: seq)
        }
        return x.reshaped(seq, cfg.hidden)
    }

    // MARK: vision-conditioned encode

    /// Raw prompt ids + vision blocks -> conditioning + adaLN tags.
    public func encodeItems(_ items: [H3PresentItem]) throws -> H3EncodedPrompt {
        let cfg = self.cfg
        let nVision = items.filter { if case .vision = $0 { return true } else { return false } }.count
        if nVision > 0 && vision == nil {
            throw H3Error.missingWeight("vision tower not loaded")
        }

        // 1. Walk the stream once: sequence length and the vision spans.
        var spans: [Span] = []
        var seqU: UInt32 = 0
        for item in items {
            switch item {
            case .text(let ids):
                seqU += UInt32(ids.count)
            case .vision(let vb):
                let size = vb.grid.mergedTokens
                spans.append(Span(index: seqU + 1, size: size, grid: vb.grid))
                seqU += size + 2
            }
        }
        let seq = Int(seqU)

        // 2. Run the tower per block, assemble embeds by concatenation.
        var merged: [MLXArray] = []
        var dsBlocks: [[MLXArray]] = []
        for item in items {
            if case .vision(let vb) = item {
                let rows = h3VisionPatchRows(frames: vb.frames, grid: vb.grid)
                let g = [vb.grid.t, vb.grid.gh, vb.grid.gw]
                let out = try vision!.forward(rows, grids: [g])
                merged.append(out.merged)
                dsBlocks.append(out.deepstack)
            }
        }

        var pieces: [MLXArray] = []
        var vi = 0
        for item in items {
            switch item {
            case .text(let ids):
                pieces.append(embedTable.take(MLXArray(ids), axis: 0))
            case .vision:
                pieces.append(embedTable.take(MLXArray([H3VisionConst.visionStart]), axis: 0))
                pieces.append(merged[vi].asType(dtype))
                pieces.append(embedTable.take(MLXArray([H3VisionConst.visionEnd]), axis: 0))
                vi += 1
            }
        }
        let flat = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
        var x = flat.reshaped(1, seq, cfg.hidden)

        // 3. Positions: plain 1-D when text-only, interleaved mRoPE otherwise.
        let positions = mropePositions(seqLen: seqU, spans: spans)
        let rope = teBuildRopeFrom(seq: seq, positions: positions)
        let mask = teCausalMask(seq: seq, dtype: dtype)
        if H3TE.trace { MLX.eval(x, mask); print("[H3TE] embeds+mask ok \(x.shape) \(mask.shape) mask.dtype=\(mask.dtype)") }

        // 4. Layers, with DeepStack taps folded in after the first three.
        for (li, layer) in layers.enumerated() {
            x = teLayerForward(layer, x: x, mask: mask, rope: rope, seq: seq)
            if H3TE.trace { MLX.eval(x); print("[H3TE] layer \(li) ok \(x.shape)") }
            if li < 3 && !spans.isEmpty && nVision > 0 {
                let pad = teDeepstackPadded(seq: seq, spans: spans, ds: dsBlocks, li: li)
                x = x + pad
            }
        }
        let out = x.reshaped(seq, cfg.hidden)
        return H3EncodedPrompt(hidden: out, tags: tokenTags(seqLen: seqU, spans: spans))
    }

    // MARK: internals

    private func teBuildRope(seq: Int) -> (cos: MLXArray, sin: MLXArray) {
        teBuildRopeFrom(seq: seq, positions: nil)
    }

    /// mRoPE table [3·seq] axis-major (or nil) -> [1, 1, seq, head_dim/2]
    /// interleaved cos/sin tables. `positions == nil` collapses to plain 1-D.
    private func teBuildRopeFrom(seq: Int, positions: [Double]?) -> (cos: MLXArray, sin: MLXArray) {
        let half = cfg.headDim / 2
        let ang = ropeAngles(seqLen: UInt32(seq), positions: positions,
                             headDim: cfg.headDim, theta: cfg.theta)
        var cosBuf = [Float](repeating: 0, count: seq * half)
        var sinBuf = [Float](repeating: 0, count: seq * half)
        for i in 0..<seq {
            for j in 0..<half {
                let a = ang[i * half + j]
                cosBuf[i * half + j] = Float(cos(a))
                sinBuf[i * half + j] = Float(sin(a))
            }
        }
        let cos = MLXArray(cosBuf).reshaped(1, 1, seq, half).asType(dtype)
        let sin = MLXArray(sinBuf).reshaped(1, 1, seq, half).asType(dtype)
        return (cos, sin)
    }

    /// [1, seq, hidden] that is ZERO everywhere except the vision rows, where
    /// it carries DeepStack tap `li`. Spans are contiguous -> concat of zero
    /// blocks and tap slices (no scatter needed).
    private func teDeepstackPadded(seq: Int, spans: [Span], ds: [[MLXArray]], li: Int) -> MLXArray {
        var parts: [MLXArray] = []
        var cursor: Int = 0
        for (i, sp) in spans.enumerated() {
            if Int(sp.index) > cursor {
                parts.append(MLXArray.zeros([Int(sp.index) - cursor, cfg.hidden], dtype: dtype))
            }
            parts.append(ds[i][li])
            cursor = Int(sp.end)
        }
        if cursor < seq {
            parts.append(MLXArray.zeros([seq - cursor, cfg.hidden], dtype: dtype))
        }
        let cat = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
        return cat.reshaped(1, seq, cfg.hidden)
    }

    private func teLayerForward(_ layer: H3TeLayerW, x: MLXArray, mask: MLXArray,
                                rope: (cos: MLXArray, sin: MLXArray), seq: Int) -> MLXArray {
        let cfg = self.cfg
        let xn = rmsNormLast(x, weight: layer.inputLn, eps: cfg.eps)

        // q/k/v projections with q/k per-head RMSNorm + interleaved mRoPE.
        let qp = layer.qw.forward(xn).reshaped(1, seq, cfg.heads, cfg.headDim)
        let kp = layer.kw.forward(xn).reshaped(1, seq, cfg.kvHeads, cfg.headDim)
        let vp = layer.vw.forward(xn).reshaped(1, seq, cfg.kvHeads, cfg.headDim)

        let qn = rmsNormLast(qp, weight: layer.qNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
        let kn = rmsNormLast(kp, weight: layer.kNorm, eps: cfg.eps).transposed(0, 2, 1, 3)
        let vt = vp.transposed(0, 2, 1, 3)
        let q = applyRopeHalf(qn, cos: rope.cos, sin: rope.sin)
        let k = applyRopeHalf(kn, cos: rope.cos, sin: rope.sin)

        let scale: Float = 1.0 / sqrt(Float(cfg.headDim))
        let attn = scaledDotProductAttention(queries: q, keys: k, values: vt, scale: scale, mask: mask)
        let at = attn.transposed(0, 2, 1, 3).reshaped(1, seq, cfg.heads * cfg.headDim)
        let o = layer.ow.forward(at)
        let h1 = x + o

        // MLP: gate/up SiLU.
        let hn = rmsNormLast(h1, weight: layer.postLn, eps: cfg.eps)
        let g = silu(layer.gateW.forward(hn))
        let u = layer.upW.forward(hn)
        let d = layer.downW.forward(g * u)
        return h1 + d
    }
}

// MARK: - Causal mask

/// Additive causal mask [1, 1, seq, seq]. Built with `where` — never by
/// multiplying an indicator by -inf (0·-inf is NaN).
public func teCausalMask(seq: Int, dtype: DType = .float32) -> MLXArray {
    let rows = MLXArray(0..<seq).expandedDimensions(axis: 1)  // [seq, 1]
    let cols = MLXArray(0..<seq).expandedDimensions(axis: 0)  // [1, seq]
    let keep = greaterEqual(rows, cols)
    let m = `where`(keep, MLXArray(0), MLXArray(-Float.infinity))
    return m.reshaped(1, 1, seq, seq).asType(dtype)
}

// MARK: - Vision patch rows ([2, 3, H, W] -> [gh·gw, 1536])

/// The reference's reshape+permute verbatim: `(1, temporal=2, C=3, gh/2, 2,
/// 16, gw/2, 2, 16)` then `permute(0,3,6,4,7,2,1,5,8)`. Rows come out in
/// SPATIAL-MERGE order (lines up with pos-embed/rotary/merger), each row is
/// [C, T, ph, pw]-flattened = 1536.
public func h3VisionPatchRows(frames: MLXArray, grid: Grid) -> MLXArray {
    let p = Int(H3VisionConst.patch)
    let m = Int(H3VisionConst.merge)
    let gh = Int(grid.gh)
    let gw = Int(grid.gw)
    let x9 = frames.reshaped(1, Int(H3VisionConst.temporalPatch), 3, gh / m, m, p, gw / m, m, p)
    let tr = x9.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return tr.reshaped(gh * gw, 3 * Int(H3VisionConst.temporalPatch) * p * p)
}

// MARK: - Vision tower (Qwen3-VL-32B shape)

public struct H3VitConfig {
    public var hidden = 1152
    public var heads = 16
    public var inter = 4304
    public var depth = 27
    public var out = 5120
    public var deepstack: [Int] = [8, 16, 24]
    public var prefix = "visual"
    public init() {}
    public var headDim: Int { hidden / heads }
    /// Rotary inv_freq length: (head_dim/2)/2 — the 2-D rotary lays [h, w, h, w].
    public var rotHalf: Int { headDim / 4 }
    public var mergeHid: Int { hidden * 2 * 2 }
    public static let h3 = H3VitConfig()
}

public final class H3VitBlockW {
    public let n1w: MLXArray, n1b: MLXArray, n2w: MLXArray, n2b: MLXArray
    public let qkvW: MfLinear, qkvB: MLXArray
    public let projW: MfLinear, projB: MLXArray
    public let fc1W: MfLinear, fc1B: MLXArray
    public let fc2W: MfLinear, fc2B: MLXArray
    public init(n1w: MLXArray, n1b: MLXArray, n2w: MLXArray, n2b: MLXArray,
                qkvW: MfLinear, qkvB: MLXArray, projW: MfLinear, projB: MLXArray,
                fc1W: MfLinear, fc1B: MLXArray, fc2W: MfLinear, fc2B: MLXArray) {
        self.n1w = n1w; self.n1b = n1b; self.n2w = n2w; self.n2b = n2b
        self.qkvW = qkvW; self.qkvB = qkvB; self.projW = projW; self.projB = projB
        self.fc1W = fc1W; self.fc1B = fc1B; self.fc2W = fc2W; self.fc2B = fc2B
    }
}

public final class H3VitMergerW {
    public let normW: MLXArray, normB: MLXArray
    public let fc1W: MfLinear, fc1B: MLXArray
    public let fc2W: MfLinear, fc2B: MLXArray
    public init(normW: MLXArray, normB: MLXArray, fc1W: MfLinear, fc1B: MLXArray,
                fc2W: MfLinear, fc2B: MLXArray) {
        self.normW = normW; self.normB = normB
        self.fc1W = fc1W; self.fc1B = fc1B; self.fc2W = fc2W; self.fc2B = fc2B
    }
}

public final class H3VisionTower {
    public let cfg: H3VitConfig
    public let dtype: DType
    public let patchW: MLXArray   // [1536, hidden] pre-transposed linear
    public let patchB: MLXArray   // [hidden]
    public let posEmbed: MLXArray // [2304, hidden]
    public let blocks: [H3VitBlockW]
    public let merger: H3VitMergerW
    public let deepstack: [H3VitMergerW]

    public static func loadFrom(_ w: H3Weights, cfg: H3VitConfig = .h3,
                                dtype: DType = PrecisionPolicy.defaultMainDType) throws -> H3VisionTower {
        // Conv3d patch embed as a pre-transposed linear [1536, hidden]: raw
        // weight OITHW [hidden,3,2,16,16] -> transpose to OHWI-3d [hidden,2,16,16,3]
        // (T,H,W,I matches the reference reshape), flatten kernels, transpose.
        guard let raw = w.get("\(cfg.prefix).patch_embed.proj.weight") else {
            throw H3Error.missingWeight("\(cfg.prefix).patch_embed.proj.weight")
        }
        let t = raw.transposed(0, 2, 3, 4, 1)                       // [H,2,16,16,3]
        let flat = t.reshaped(cfg.hidden, 1536)
        let patchW = flat.transposed(1, 0).asType(dtype)            // [1536, hidden]
        let patchB = w.get("\(cfg.prefix).patch_embed.proj.bias")?.asType(dtype)
            ?? MLXArray.zeros([cfg.hidden], dtype: dtype)
        let posEmbed = w.get("\(cfg.prefix).pos_embed.weight")?.asType(dtype)
            ?? MLXArray.zeros([2304, cfg.hidden], dtype: dtype)

        var blocks: [H3VitBlockW] = []
        blocks.reserveCapacity(cfg.depth)
        for i in 0..<cfg.depth {
            let p = "\(cfg.prefix).blocks.\(i)"
            let n1 = "\(p).norm1", n2 = "\(p).norm2", qkv = "\(p).attn.qkv"
            let proj = "\(p).attn.proj", fc1 = "\(p).mlp.linear_fc1", fc2 = "\(p).mlp.linear_fc2"
            blocks.append(H3VitBlockW(
                n1w: w.get("\(n1).weight") ?? MLXArray.ones([cfg.hidden], dtype: .float32),
                n1b: w.get("\(n1).bias") ?? MLXArray.zeros([cfg.hidden], dtype: .float32),
                n2w: w.get("\(n2).weight") ?? MLXArray.ones([cfg.hidden], dtype: .float32),
                n2b: w.get("\(n2).bias") ?? MLXArray.zeros([cfg.hidden], dtype: .float32),
                qkvW: try MfLinear.load(w, prefix: qkv, inFeatures: cfg.hidden, dtype: dtype),
                qkvB: w.get("\(qkv).bias")?.asType(dtype) ?? MLXArray.zeros([3 * cfg.hidden], dtype: dtype),
                projW: try MfLinear.load(w, prefix: proj, inFeatures: cfg.hidden, dtype: dtype),
                projB: w.get("\(proj).bias")?.asType(dtype) ?? MLXArray.zeros([cfg.hidden], dtype: dtype),
                fc1W: try MfLinear.load(w, prefix: fc1, inFeatures: cfg.hidden, dtype: dtype),
                fc1B: w.get("\(fc1).bias")?.asType(dtype) ?? MLXArray.zeros([cfg.inter], dtype: dtype),
                fc2W: try MfLinear.load(w, prefix: fc2, inFeatures: cfg.inter, dtype: dtype),
                fc2B: w.get("\(fc2).bias")?.asType(dtype) ?? MLXArray.zeros([cfg.hidden], dtype: dtype)))
        }

        func loadMerger(_ prefix: String) throws -> H3VitMergerW {
            let nm = "\(prefix).norm", fc1 = "\(prefix).linear_fc1", fc2 = "\(prefix).linear_fc2"
            return H3VitMergerW(
                normW: w.get("\(nm).weight") ?? MLXArray.ones([cfg.mergeHid], dtype: .float32),
                normB: w.get("\(nm).bias") ?? MLXArray.zeros([cfg.mergeHid], dtype: .float32),
                fc1W: try MfLinear.load(w, prefix: fc1, inFeatures: cfg.mergeHid, dtype: dtype),
                fc1B: w.get("\(fc1).bias")?.asType(dtype) ?? MLXArray.zeros([cfg.hidden], dtype: dtype),
                fc2W: try MfLinear.load(w, prefix: fc2, inFeatures: cfg.mergeHid, dtype: dtype),
                fc2B: w.get("\(fc2).bias")?.asType(dtype) ?? MLXArray.zeros([cfg.out], dtype: dtype))
        }
        let merger = try loadMerger("\(cfg.prefix).merger")
        var ds: [H3VitMergerW] = []
        for k in 0..<3 {
            ds.append(try loadMerger("\(cfg.prefix).deepstack_merger_list.\(k)"))
        }
        return H3VisionTower(cfg: cfg, dtype: dtype, patchW: patchW, patchB: patchB,
                             posEmbed: posEmbed, blocks: blocks, merger: merger, deepstack: ds)
    }

    private init(cfg: H3VitConfig, dtype: DType, patchW: MLXArray, patchB: MLXArray,
                 posEmbed: MLXArray, blocks: [H3VitBlockW], merger: H3VitMergerW,
                 deepstack: [H3VitMergerW]) {
        self.cfg = cfg
        self.dtype = dtype
        self.patchW = patchW
        self.patchB = patchB
        self.posEmbed = posEmbed
        self.blocks = blocks
        self.merger = merger
        self.deepstack = deepstack
    }

    /// pixel_values [Np, 1536] with per-image grids (t, gh, gw) ->
    /// merged [Ntok, out] + 3 DeepStack feature sets (each [Ntok, out]).
    public func forward(_ pixelValues: MLXArray, grids: [[UInt32]]) throws -> (merged: MLXArray, deepstack: [MLXArray]) {
        let cfg = self.cfg
        let np = pixelValues.shape[0]

        // 1. Patch embed: reorder [Np,I,T,H,W] -> [Np,T,H,W,I], flatten, linear.
        let pv = pixelValues.asType(dtype)
        let pv5 = pv.reshaped(np, 3, 2, 16, 16)
        let pvt = pv5.transposed(0, 2, 3, 4, 1)          // [Np, T, H, W, I]
        let pvf = pvt.reshaped(np, 1536)
        var hidden = pvf.matmul(patchW) + patchB          // [Np, hidden]

        // 2. Interpolated pos-embed (merge order) + add.
        let pos = buildPosEmbeds(grids: grids)
        hidden = hidden + pos

        // 3. 2D rotary cos/sin [Np, head_dim] (f32).
        let rope = buildVisionRope(grids: grids)

        // 4. cu_seqlens: one segment per (image × frame).
        var segs: [Int] = [0]
        for g in grids {
            let frameLen = Int(g[1] * g[2])
            for _ in 0..<Int(g[0]) {
                segs.append(segs[segs.count - 1] + frameLen)
            }
        }

        // 5. Blocks (+ DeepStack taps).
        var dsOut: [MLXArray?] = [nil, nil, nil]
        for (i, blk) in blocks.enumerated() {
            hidden = vitBlockForward(blk, hidden: hidden, cos: rope.cos, sin: rope.sin, segs: segs)
            for (k, di) in cfg.deepstack.enumerated() where i == di {
                dsOut[k] = mergerForward(deepstack[k], hidden: hidden, postShuffle: true)
            }
        }

        // 6. Merger.
        let merged = mergerForward(merger, hidden: hidden, postShuffle: false)
        let deep = dsOut.map { $0! }
        return (merged, deep)
    }

    // MARK: internals

    /// Bilinear-interpolated learned position embeddings in spatial-merge order.
    private func buildPosEmbeds(grids: [[UInt32]]) -> MLXArray {
        let cfg = self.cfg
        let side = H3VisionConst.vitGridSide
        var outputs: [MLXArray] = []
        for g in grids {
            let t = Int(g[0]), gh = Int(g[1]), gw = Int(g[2])
            let n = gh * gw
            var idx = [Int32](repeating: 0, count: 4 * n)
            var wt = [Float](repeating: 0, count: 4 * n)
            for i in 0..<gh {
                let hidx = (gh == 1) ? 0.0 : Double(i) * Double(side - 1) / Double(gh - 1)
                let hf = floor(hidx)
                let hfi = Int(hf)
                let hci = min(hfi + 1, side - 1)
                let dh = hidx - hf
                for j in 0..<gw {
                    let widx = (gw == 1) ? 0.0 : Double(j) * Double(side - 1) / Double(gw - 1)
                    let wf = floor(widx)
                    let wfi = Int(wf)
                    let wci = min(wfi + 1, side - 1)
                    let dw = widx - wf
                    let nn = i * gw + j
                    let bh = hfi * side
                    let bhc = hci * side
                    idx[0 * n + nn] = Int32(bh + wfi)
                    idx[1 * n + nn] = Int32(bh + wci)
                    idx[2 * n + nn] = Int32(bhc + wfi)
                    idx[3 * n + nn] = Int32(bhc + wci)
                    wt[0 * n + nn] = Float((1 - dh) * (1 - dw))
                    wt[1 * n + nn] = Float((1 - dh) * dw)
                    wt[2 * n + nn] = Float(dh * (1 - dw))
                    wt[3 * n + nn] = Float(dh * dw)
                }
            }
            let gathered = posEmbed.take(MLXArray(idx), axis: 0)          // [4n, hidden]
            let g3 = gathered.reshaped(4, n, cfg.hidden)
            let wtArr = MLXArray(wt).reshaped(4, n, 1).asType(dtype)
            let weighted = g3 * wtArr
            let interp = weighted.sum(axis: 0)                            // [n, hidden]
            // t=1 always for H3; tile kept for fidelity.
            let tiled = t > 1 ? concatenated(Array(repeating: interp, count: t), axis: 0) : interp
            let r = tiled.reshaped(t, gh / 2, 2, gw / 2, 2, cfg.hidden)
            let rt = r.transposed(0, 1, 3, 2, 4, 5)
            outputs.append(rt.reshaped(t * n, cfg.hidden))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 0)
    }

    /// Vision 2D rotary cos/sin [Ntok, head_dim] (f32), spatial-merge order.
    /// Per token (h, w): angles interleave [h·f, w·f, h·f, w·f] over the
    /// 16-freq table (head_dim/4 frequencies).
    private func buildVisionRope(grids: [[UInt32]]) -> (cos: MLXArray, sin: MLXArray) {
        let cfg = self.cfg
        var ntok = 0
        for g in grids { ntok += Int(g[0]) * Int(g[1]) * Int(g[2]) }
        let hd = cfg.headDim
        let rotHalf = cfg.rotHalf
        var cosb = [Float](repeating: 0, count: ntok * hd)
        var sinb = [Float](repeating: 0, count: ntok * hd)
        let rotDim = Double(rotHalf * 2)
        var invFreq = [Double](repeating: 0, count: rotHalf)
        for j in 0..<rotHalf {
            invFreq[j] = 1.0 / pow(10_000.0, Double(2 * j) / rotDim)
        }
        var tok = 0
        for g in grids {
            let t = Int(g[0]), gh = Int(g[1]), gw = Int(g[2])
            let m = 2
            for frame in 0..<t {
                for ai in 0..<(gh / m) {
                    for bi in 0..<(gw / m) {
                        for ci in 0..<m {
                            for d in 0..<m {
                                let hpos = Double(ai * m + ci)
                                let wpos = Double(bi * m + d)
                                let base = tok * hd
                                for j in 0..<rotHalf {
                                    let ah = hpos * invFreq[j]
                                    let aw = wpos * invFreq[j]
                                    cosb[base + j] = Float(cos(ah))
                                    cosb[base + rotHalf + j] = Float(cos(aw))
                                    cosb[base + 2 * rotHalf + j] = Float(cos(ah))
                                    cosb[base + 3 * rotHalf + j] = Float(cos(aw))
                                    sinb[base + j] = Float(sin(ah))
                                    sinb[base + rotHalf + j] = Float(sin(aw))
                                    sinb[base + 2 * rotHalf + j] = Float(sin(ah))
                                    sinb[base + 3 * rotHalf + j] = Float(sin(aw))
                                }
                                tok += 1
                            }
                        }
                    }
                }
            }
        }
        let cf = MLXArray(cosb).reshaped(ntok, hd)
        let sf = MLXArray(sinb).reshaped(ntok, hd)
        return (cf, sf)
    }

    private func vitBlockForward(_ bw: H3VitBlockW, hidden: MLXArray, cos: MLXArray,
                                 sin: MLXArray, segs: [Int]) -> MLXArray {
        let cfg = self.cfg
        let n1 = layerNormLast(hidden, weight: bw.n1w, bias: bw.n1b, eps: 1e-6)
        let attn = visionAttn(bw, x: n1, cos: cos, sin: sin, segs: segs)
        let h1 = hidden + attn
        let n2 = layerNormLast(h1, weight: bw.n2w, bias: bw.n2b, eps: 1e-6)
        let fc1 = bw.fc1W.forward(n2, bias: bw.fc1B)
        let g = geluTanh(fc1)
        let fc2 = bw.fc2W.forward(g, bias: bw.fc2B)
        return h1 + fc2
    }

    private func visionAttn(_ bw: H3VitBlockW, x: MLXArray, cos: MLXArray, sin: MLXArray,
                            segs: [Int]) -> MLXArray {
        let cfg = self.cfg
        let np = x.shape[0]
        let qkv = bw.qkvW.forward(x, bias: bw.qkvB)         // [Np, 3·hidden]
        let qkv5 = qkv.reshaped(np, 3, cfg.heads, cfg.headDim)
        let q = applyVitRope(qkv5[0..., 0, 0..., 0...], cos: cos, sin: sin)  // [Np, heads, hd]
        let k = applyVitRope(qkv5[0..., 1, 0..., 0...], cos: cos, sin: sin)
        let v = qkv5[0..., 2, 0..., 0...]

        let scale: Float = 1.0 / sqrt(Float(cfg.headDim))
        var outs: [MLXArray] = []
        for i in 0..<(segs.count - 1) {
            outs.append(visionSegAttn(q: q, k: k, v: v, start: segs[i], end: segs[i + 1], scale: scale))
        }
        let cat = outs.count == 1 ? outs[0] : concatenated(outs, axis: 0)
        let flat = cat.reshaped(np, cfg.hidden)
        return bw.projW.forward(flat, bias: bw.projB)
    }

    /// SDPA over one segment q/k/v [Np, heads, hd] sliced to [start, end).
    private func visionSegAttn(q: MLXArray, k: MLXArray, v: MLXArray,
                               start: Int, end: Int, scale: Float) -> MLXArray {
        let cfg = self.cfg
        let qs = q[start..<end]
        let ks = k[start..<end]
        let vs = v[start..<end]
        let seg = end - start
        // [seg, heads, hd] -> [1, heads, seg, hd]
        let qt = qs.transposed(1, 0, 2).reshaped(1, cfg.heads, seg, cfg.headDim)
        let kt = ks.transposed(1, 0, 2).reshaped(1, cfg.heads, seg, cfg.headDim)
        let vt = vs.transposed(1, 0, 2).reshaped(1, cfg.heads, seg, cfg.headDim)
        let attn = scaledDotProductAttention(queries: qt, keys: kt, values: vt, scale: scale, mask: nil)
        let a3 = attn.reshaped(cfg.heads, seg, cfg.headDim)
        return a3.transposed(1, 0, 2)
    }

    /// Patch merger: (optional post-shuffle) LayerNorm -> 4-token group ->
    /// linear_fc1 -> gelu-tanh -> linear_fc2. [N, hidden] -> [N/4, out].
    private func mergerForward(_ mw: H3VitMergerW, hidden: MLXArray, postShuffle: Bool) -> MLXArray {
        let cfg = self.cfg
        let n = hidden.shape[0]
        let grouped = n / 4
        let x: MLXArray
        if postShuffle {
            let r = hidden.reshaped(grouped, cfg.mergeHid)
            x = layerNormLast(r, weight: mw.normW, bias: mw.normB, eps: 1e-6)
        } else {
            let nrm = layerNormLast(hidden, weight: mw.normW, bias: mw.normB, eps: 1e-6)
            x = nrm.reshaped(grouped, cfg.mergeHid)
        }
        let fc1 = mw.fc1W.forward(x, bias: mw.fc1B)
        let g = geluTanh(fc1)
        return mw.fc2W.forward(g, bias: mw.fc2B)
    }

    /// Rotate-half 2D rotary on x [Np, heads, hd] with cos/sin [Np, hd] (f32),
    /// computed in f32 then cast back to x's dtype (HF vision rule).
    private func applyVitRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let inDtype = x.dtype
        let xf = x.asType(.float32)
        let cosB = cos.reshaped(x.shape[0], 1, x.shape[2])
        let sinB = sin.reshaped(x.shape[0], 1, x.shape[2])
        let half = x.shape[2] / 2
        let x1 = xf[0..., 0..., ..<half]
        let x2 = xf[0..., 0..., half...]
        let rotated = concatenated([-x2, x1], axis: 2)
        let out = xf * cosB + rotated * sinB
        return out.asType(inDtype)
    }
}

// MARK: - MfLinear convenience (bias overload)

extension MfLinear {
    /// forward with an explicit bias array (VisionTower dense linears keep
    /// separate bias tensors rather than MfLinear.bias).
    public func forward(_ x: MLXArray, bias: MLXArray, stream: MLX.Stream? = nil) -> MLXArray {
        forward(x, stream: stream) + bias
    }
}
