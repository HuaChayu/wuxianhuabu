// H3Common.swift
// MiniMax H3 (Hailuo 3.0) — common layer for the Swift + MLX port.
// Ported from mlx-serve-main/src/minimax_h3.zig + mage_flow.zig (MfLinear).
//
// Contents:
//   - Constants & H3Config (transformer geometry)
//   - H3Weights: lazy safetensors reader (header index + on-demand tensor load)
//   - MfLinear: dense (pre-transposed [in,out]) OR affine group-quantized
//     (4-bit g64 U32-packed weight + scales + biases), lazy dequant + matmul
//   - LoRA slot (turbo_lora / style adapters): base + Σ scale·(x·A)·B
//   - MLX tensor primitives mirroring minimax_h3.zig's helpers

import Foundation
import MLX
import MLXRandom
import Darwin

// MARK: - Constants (asserted against the checkpoint's model_index.json)

public enum H3Const {
    /// Source frames represented by latent frame k, cycling on k % 5 (17 frames / 5 latents).
    public static let framePerToken: [UInt32] = [1, 4, 4, 4, 4]
    public static let frameRescale: Double = 5.0 / 3.0
    public static let visualCondTimestep: Double = 0.999
    public static let audioCondTimestep: Double = 1.0

    public static let canvasMultiple: UInt32 = 32
    public static let baseShortEdge: UInt32 = 768
    public static let maxPixels: UInt32 = 768 * 1344
    public static let fps: UInt32 = 24
    public static let audioLatentFPS: UInt32 = 40

    /// 续接窗口（像素帧）：前置尾段重复延续长度。
    /// 对齐 ComfyUI-H3-Motion-Context 默认 context_length=22（22 像素帧 ≈ 0.92s，
    /// 经 videoLatentT 换算 = 7 latent step / 7 个 keyframe 锚点）。
    /// 全局唯一入口：落盘窗口 / 续接总时长 / 完成后裁切均由它换算，禁止散落硬编码。
    public static let continuationContextFrames: UInt32 = 22

    /// Defaults from `model_index.json` sigma_shift_scales.
    public static let sigmaShiftVideo: Double = 12.0
    public static let sigmaShiftAudio: Double = 3.0

    /// Spatial compression: VAE 16x, then DiT 2x2 patch → 32x effective.
    public static let vaeSpatial: UInt32 = 16
    public static let patchH: UInt32 = 2
    public static let patchW: UInt32 = 2

    public static let refImageShortEdge: UInt32 = 2048
    public static let maxRefImages = 9
    public static let maxRefVideos = 3
    public static let maxRefAudios = 3
    public static let maxRefTotal = 12
    public static let minRefVideoFrames: UInt32 = 5

    // MARK: - Frame-count → latent 正向转换（对齐官方 alignFrameCount + videoLatentT）
    /// 官方 alignFrameCount(n)：把请求帧数向上对齐到 n % 17 == 5（120 → 124）。
    public static func alignFrameCount(_ n: UInt32) -> UInt32 {
        var f = n
        while f % 17 != 5 { f += 1 }
        return f
    }

    /// 官方 videoLatentT(frame_count)：对齐后帧数 → 潜在帧数（124 → 37）。
    /// frame_count <= 5 → 2；否则 ((frame_count - 5) / 17) * 5 + 2。
    public static func videoLatentT(frameCount: UInt32) -> UInt32 {
        if frameCount <= 5 { return 2 }
        return ((frameCount - 5) / 17) * 5 + 2
    }

    /// 官方 audio_t = round(frame_count * audioLatentFPS / fps)（124 → 207）。
    public static func audioLatentT(frameCount: UInt32) -> UInt32 {
        UInt32((Double(frameCount) * Double(audioLatentFPS) / Double(fps)).rounded())
    }
}

// MARK: - Config

public struct H3Config: Codable {
    public var hiddenSize: Int = 5376
    public var numLayers: Int = 50
    public var numHeads: Int = 56
    public var headDim: Int = 128
    public var ffnHidden: Int = 14336
    public var latentDim: Int = 24
    public var audioLatentDim: Int = 32
    public var textDim: Int = 5120
    public var timeEmbedDim: Int = 2688
    public var ropeInvFreqLen: Int = 16
    public var normEps: Float = 1e-5
    /// sigma_shift_scales from the checkpoint
    public var sigmaShiftVideo: Double = 12.0
    public var sigmaShiftAudio: Double = 3.0
    public var fps: UInt32 = 24

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_hidden_layers"
        case numHeads = "num_attention_heads"
        case headDim = "attention_head_dim"
        case ffnHidden = "ffn_hidden_size"
        case latentDim = "latents_dim"
        case audioLatentDim = "audio_latents_dim"
        case textDim = "text_dim"
        case timeEmbedDim = "time_embed_dim"
        case ropeInvFreqLen = "rope_inv_freq_len"
        case normEps = "norm_eps"
        case sigmaShiftVideo = "sigma_shift_video"
        case sigmaShiftAudio = "sigma_shift_audio"
        case fps
    }

    /// Attention inner width = heads × head_dim (7168 for the full model).
    public var innerDim: Int { numHeads * headDim }
    /// Rotated dims: 3 axes × 16 inv-freqs × 2 halves.
    public var rotDim: Int { ropeInvFreqLen * 3 * 2 }
    /// Patch-embed width for video latents: channels × 2 × 2.
    public var videoPatchDim: Int { latentDim * Int(H3Const.patchH) * Int(H3Const.patchW) }

    public init() {}

    public static func load(from url: URL) throws -> H3Config {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let cfg = try decoder.decode(H3Config.self, from: data)
        // sigma_shift_scales arrives as a nested object in model_index.json.
        return cfg
    }

    /// Parse the checkpoint's `sigma_shift_scales` object (video/audio keys).
    public static func sigmaShifts(fromJSON data: Data) -> (video: Double, audio: Double) {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sss = obj["sigma_shift_scales"] as? [String: Any] else {
            return (H3Const.sigmaShiftVideo, H3Const.sigmaShiftAudio)
        }
        let v = (sss["video"] as? NSNumber)?.doubleValue ?? H3Const.sigmaShiftVideo
        let a = (sss["audio"] as? NSNumber)?.doubleValue ?? H3Const.sigmaShiftAudio
        return (v, a)
    }
}

// MARK: - DType / safetensors helpers

public enum H3DType: UInt8, Sendable {
    case f32, f16, bf16, u32, u8, i64, i32

    public var mlx: DType {
        switch self {
        case .f32: return .float32
        case .f16: return .float16
        case .bf16: return .bfloat16
        case .u32: return .uint32
        case .u8: return .uint8
        case .i64: return .int64
        case .i32: return .int32
        }
    }

    public init?(safetensorsName: String) {
        switch safetensorsName {
        case "F32": self = .f32
        case "F16": self = .f16
        case "BF16": self = .bf16
        case "U32": self = .u32
        case "U8": self = .u8
        case "I64": self = .i64
        case "I32": self = .i32
        default: return nil
        }
    }

    public var bytesPerElement: Int {
        switch self {
        case .f32, .u32, .i32: return 4
        case .f16, .bf16, .u8: return 2
        case .u8: return 1
        case .i64: return 8
        }
    }
}

// MARK: - H3Weights (lazy safetensors reader)

/// Lazy safetensors container. The header is parsed once; each tensor is read
/// from the file on demand (7.8 GB transformer.safetensors never fully loads).
public final class H3Weights {
    public struct TensorMeta {
        public let dtype: H3DType
        public let shape: [Int]
        public let start: UInt64
        public let count: Int
    }

    private let handle: FileHandle
    private let dataStart: UInt64
    public private(set) var meta: [String: TensorMeta] = [:]
    private var cache: [String: MLXArray] = [:]
    /// 一次性读取阶段（DiT/VAE/text_encoder 全量加载）关闭缓存，
    /// 避免 18.7GB 原始 tensor 全部滞留内存导致压力被杀。
    public var cacheEnabled: Bool = true

    public init(url: URL) throws {
        let fh = try FileHandle(forReadingFrom: url)
        self.handle = fh
        guard let lenData = try fh.read(upToCount: 8), lenData.count == 8 else {
            throw H3Error.badFile("cannot read header length")
        }
        let headerLen = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        self.dataStart = 8 + headerLen
        guard let headerData = try fh.read(upToCount: Int(headerLen)) else {
            throw H3Error.badFile("cannot read header")
        }
        guard let obj = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw H3Error.badFile("header is not JSON")
        }
        for (key, value) in obj {
            guard key != "__metadata__", let t = value as? [String: Any],
                  let dtypeName = t["dtype"] as? String,
                  let dt = H3DType(safetensorsName: dtypeName),
                  let shape = t["shape"] as? [NSNumber],
                  let offs = t["data_offsets"] as? [NSNumber], offs.count == 2 else { continue }
            let start = UInt64(truncating: offs[0])
            let end = UInt64(truncating: offs[1])
            let count = Int(end - start)
            meta[key] = TensorMeta(dtype: dt, shape: shape.map { $0.intValue }, start: start, count: count)
        }
    }

    deinit {
        try? handle.close()
    }

    public var keys: [String] { Array(meta.keys) }

    public func contains(_ key: String) -> Bool { meta[key] != nil }

    /// Fetch a tensor (cached). Returns nil when the key is absent.
    public func get(_ key: String) -> MLXArray? {
        if cacheEnabled, let cached = cache[key] { return cached }
        guard let m = meta[key] else { return nil }
        do {
            try handle.seek(toOffset: dataStart + m.start)
            guard let data = try handle.read(upToCount: m.count) else { return nil }
            let arr = MLXArray(data, m.shape, dtype: m.dtype.mlx)
            if cacheEnabled { cache[key] = arr }
            return arr
        } catch {
            return nil
        }
    }

    /// Fetch without caching (for one-shot tensors like LoRA pairs).
    public func getUncached(_ key: String) -> MLXArray? {
        guard let m = meta[key] else { return nil }
        do {
            try handle.seek(toOffset: dataStart + m.start)
            guard let data = try handle.read(upToCount: m.count) else { return nil }
            return MLXArray(data, m.shape, dtype: m.dtype.mlx)
        } catch {
            return nil
        }
    }

    /// Quantized triple fetch: "prefix.weight" (U32 packed) + scales + biases.
    /// 兼容两套命名：mlx-serve 的 `prefix.scales/.biases`，以及 comfy/HF 风格的
    /// `prefix.weight.scales/.weight.biases`（H3 pruned/fused 权重即后者）。
    /// 缺这一层 fallback 会把 U32 打包权当稠密 BF16 读入，前向立刻形状崩溃。
    public func getQuantized(_ prefix: String) -> (weight: MLXArray, scales: MLXArray, biases: MLXArray)? {
        guard let w = get(prefix + ".weight") else { return nil }
        if let s = get(prefix + ".scales"), let b = get(prefix + ".biases") {
            return (w, s, b)
        }
        if let s = get(prefix + ".weight.scales"), let b = get(prefix + ".weight.biases") {
            return (w, s, b)
        }
        return nil
    }

    public func clearCache() { cache.removeAll() }
}

public enum H3Error: Error, LocalizedError {
    case badFile(String)
    case missingWeight(String)
    case notImplemented(String)
    public var errorDescription: String? {
        switch self {
        case .badFile(let s): return "bad file: \(s)"
        case .missingWeight(let s): return "missing weight: \(s)"
        case .notImplemented(let s): return "not implemented: \(s)"
        }
    }
}

// MARK: - MfLinear (dense or affine group-quantized linear)

/// Dense-or-quantized linear mirroring mage_flow.MfLinear.
///   dense: w stored PRE-TRANSPOSED [in, out]; forward = x·w + bias
///   quantized: w packed U32 [out, in·bits/32], scales/biases [out, in/group]
///     (affine 4-bit g64 for the H3 checkpoint); forward dequantizes lazily
///     to [out, in] bf16, transposes and matmuls.
public final class MfLinear {
    public let inDim: Int
    public var outDim: Int
    public var isQuantized: Bool = false
    public var w: MLXArray?          // dense [in, out] (compute dtype) OR packed [out, in*bits/32]
    public var scales: MLXArray?
    public var biases: MLXArray?
    public var bias: MLXArray?       // [out]
    public var bits: Int = 0
    public var groupSize: Int = 0
    /// lazily materialized dense weight [out, in] for the quantized path
    private var dequantized: MLXArray?
    public var lora = LoraSlot()
    /// 诊断名：NA_DUMP_FWD=1 且非空时打印 base/delta 分离统计
    public var diagName = ""

    public init(inDim: Int, outDim: Int) {
        self.inDim = inDim
        self.outDim = outDim
    }

    public convenience init(dense w: MLXArray, bias: MLXArray?, inDim: Int, dtype: DType) {
        self.init(inDim: inDim, outDim: w.shape[1])
        self.w = w.asType(dtype)
        self.bias = bias?.asType(dtype)
    }

    /// Load from a weights container. `prefix` + ".weight"/".scales"/".biases".
    /// inFeatures resolves the packed geometry (bits = 32·w_cols / in).
    public static func load(_ w: H3Weights, prefix: String, inFeatures: Int, dtype: DType = PrecisionPolicy.defaultMainDType) throws -> MfLinear {
        let lin = MfLinear(inDim: inFeatures, outDim: 0)
        if let q = w.getQuantized(prefix) {
            let wShape = q.weight.shape
            guard wShape.count == 2 else { throw H3Error.badFile("quantized weight rank \(wShape.count)") }
            let wCols = wShape[1]
            lin.outDim = wShape[0]
            lin.isQuantized = true
            lin.w = q.weight
            lin.bits = 32 * wCols / inFeatures
            let sCols = q.scales.shape[1]
            lin.groupSize = inFeatures / sCols
            lin.scales = q.scales.asType(dtype)
            lin.biases = q.biases.asType(dtype)
            if let b = w.get(prefix + ".bias") { lin.bias = b.asType(dtype) }
        } else {
            guard let raw = w.get(prefix + ".weight") else {
                throw H3Error.missingWeight(prefix + ".weight")
            }
            // raw [out, in] → [in, out], materialize, cast.
            lin.outDim = raw.shape[0]
            let t = raw.transposed(1, 0)
            lin.w = t.asType(dtype)
            if let b = w.get(prefix + ".bias") { lin.bias = b.asType(dtype) }
        }
        return lin
    }

    /// x[..., in] · W (+ bias). Matches MfLinear.forward semantics.
    public func forward(_ x: MLXArray, stream: MLX.Stream? = nil) -> MLXArray {
        var xc: MLXArray
        var out: MLXArray
        if isQuantized {
            // Native GPU quantized matmul — never materialize the dense weight
            // (a 4-bit text tower would balloon to ~60 GB bf16 in memory).
            guard let qw = w, let qs = scales else {
                fatalError("quantized linear missing components")
            }
            xc = x.asType(.bfloat16)
            if ProcessInfo.processInfo.environment["NA_DEQUANT"] == "1" && !diagName.isEmpty {
                // 诊断：仅诊断层反量化 dense [out,in] → x @ W^T，验证 quantizedMM 布局
                let dw = dequantizedWeight()
                let dqOut = xc.matmul(dw.transposed(1, 0))
                if !diagName.isEmpty { h3DumpRowStats("dequant_\(diagName)", dqOut, limitRows: 32) }
                out = dqOut
            } else {
                out = quantizedMM(xc, qw, scales: qs, biases: biases,
                                  transpose: true, groupSize: groupSize, bits: bits)
            }
        } else {
            guard let w = w else { fatalError("MfLinear has no weight") }
            xc = x.asType(w.dtype)
            out = xc.matmul(w)
        }
        if let b = bias {
            out = out + b
        }
        if lora.hasAdapters {
            let wantDiag = ProcessInfo.processInfo.environment["NA_DUMP_FWD"] == "1" && !diagName.isEmpty
            if wantDiag { h3DumpRowStats("lora_base_\(diagName)", out, limitRows: 32) }
            let delta = loraDelta(xc, lora)
            if wantDiag { h3DumpRowStats("lora_delta_\(diagName)", delta, limitRows: 32) }
            out = out + delta
        }
        return out
    }

    /// Dequantize affine group-quantized weight to dense [out, in] bf16 (lazy).
    /// q ∈ [0, 2^bits): w = q · scale + bias per group.
    /// GPU-vectorized: packed U32 words → nibbles via shifts, then scale+bias.
    public func dequantizedWeight() -> MLXArray {
        if let d = dequantized { return d }
        guard let qw = w, let qs = scales, let qb = biases else {
            fatalError("quantized linear missing components")
        }
        let out = outDim
        let groups = inDim / groupSize
        let nPerWord = 32 / bits            // 8 for 4-bit
        // qw [out, in/8] (4-bit U32 packed) → [out, groups, groupSize/nPerWord, 1]
        let wordArr = qw.dtype == .uint32 ? qw : qw.asType(.uint32)
        let words = wordArr.reshaped([out, groups, groupSize / nPerWord, 1])
        let shifts = MLXArray(Array(0..<nPerWord).map { UInt32($0 * bits) }, [1, 1, 1, nPerWord])
        let nibbles = bitwiseAnd(words >> shifts, 0xF)  // [out, groups, gs/npw, npw] → [out, in]
        let s = qs.asType(.float32).reshaped([out, groups, 1, 1])
        let b = qb.asType(.float32).reshaped([out, groups, 1, 1])
        let dq = nibbles.asType(.float32) * s + b
        let arr = dq.reshaped([out, inDim]).asType(.bfloat16)
        dequantized = arr
        return arr
    }

    public var dtype: DType { w?.dtype ?? .bfloat16 }
}

// MARK: - LoRA

public struct LoraRef {
    public let a: MLXArray  // [in, r]
    public let b: MLXArray  // [r, out]
    public let scale: Float
}

public struct LoraSlot {
    public var refs: [LoraRef] = []
    public var hasAdapters: Bool { !refs.isEmpty }
    public mutating func set(_ r: [LoraRef]) { refs = r }
}

/// base + Σ scale·((x·A)·B). Consumes `base` (returns a new array).
public func loraAdd(_ base: MLXArray, _ x: MLXArray, _ slot: LoraSlot) -> MLXArray {
    base + loraDelta(x, slot)
}

/// Σ scale·((x·A)·B) — LoRA 增量部分。
public func loraDelta(_ x: MLXArray, _ slot: LoraSlot) -> MLXArray {
    guard !slot.refs.isEmpty else { return MLXArray.zeros(like: x) }
    var acc: MLXArray?
    for r in slot.refs {
        let delta = x.matmul(r.a).matmul(r.b) * MLXArray(r.scale, dtype: x.dtype)
        acc = acc.map { $0 + delta } ?? delta
    }
    return acc ?? MLXArray.zeros(like: x)
}

/// Turbo LoRA file (turbo_lora.safetensors): keys
///   blocks.{i}.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2}.{lora_A,lora_B}.weight
/// or the Kohya-flat spelling. `attach` maps them onto module names.
public final class H3LoraFile {
    public struct Entry {
        public let module: String   // e.g. "blocks.3.attn.qkv_proj"
        public let a: MLXArray      // [in, r]
        public let b: MLXArray      // [r, out]
    }
    public var entries: [Entry] = []

    public static func load(from url: URL) throws -> H3LoraFile {
        let w = try H3Weights(url: url)
        let f = H3LoraFile()
        // Collect all A/B arrays first (key order is not guaranteed).
        var aByModule: [String: MLXArray] = [:]
        var bByModule: [String: MLXArray] = [:]
        for key in w.keys {
            guard let module = Self.moduleName(fromKey: key), let suffix = Self.suffix(fromKey: key) else { continue }
            guard let arr = w.getUncached(key) else { continue }
            // safetensors stores A as [r, in] and B as [out, r]; both are
            // pre-transposed to the forward order [in, r] / [r, out].
            let arrT = arr.transposed(1, 0)
            if suffix == "A" {
                aByModule[module] = arrT
            } else {
                bByModule[module] = arrT
            }
        }
        for module in aByModule.keys {
            if let b = bByModule[module], let a = aByModule[module] {
                f.entries.append(Entry(module: module, a: a, b: b))
            }
        }
        return f
    }

    /// "diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight" → "blocks.3.attn.qkv_proj"
    static func moduleName(fromKey key: String) -> String? {
        var k = key
        if k.hasPrefix("diffusion_model.") { k = String(k.dropFirst("diffusion_model.".count)) }
        if k.hasPrefix("lora_unet_") { k = String(k.dropFirst("lora_unet_".count)) }
        if let r = k.range(of: ".lora_A.weight") {
            return String(k[..<r.lowerBound])
        }
        if let r = k.range(of: ".lora_B.weight") {
            return String(k[..<r.lowerBound])
        }
        return nil
    }

    static func suffix(fromKey key: String) -> String? {
        if key.hasSuffix(".lora_A.weight") { return "A" }
        if key.hasSuffix(".lora_B.weight") { return "B" }
        if key.hasSuffix(".lora_down.weight") { return "A" }
        if key.hasSuffix(".lora_up.weight") { return "B" }
        return nil
    }

    public func find(_ module: String) -> Entry? {
        entries.first { $0.module == module }
    }
}

// MARK: - Tensor primitives (mirror minimax_h3.zig helpers)

public func rmsNormLast(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let xf = x.asType(.float32)
    let variance = xf.square().mean(axis: -1, keepDims: true)
    let normed = xf * (variance + MLXArray(eps)).rsqrt()
    let out = normed * weight.asType(.float32)
    return out.asType(x.dtype)
}

/// LayerNorm over the last axis with weight+bias (vision tower).
public func layerNormLast(_ x: MLXArray, weight: MLXArray, bias: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let xf = x.asType(.float32)
    let mean = xf.mean(axis: -1, keepDims: true)
    let variance = xf.square().mean(axis: -1, keepDims: true) - mean.square()
    let normed = (xf - mean) * (variance + MLXArray(eps)).rsqrt()
    let out = normed * weight.asType(.float32) + bias.asType(.float32)
    return out.asType(x.dtype)
}

public func silu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}

/// gelu_tanh used by the vision tower.
public func geluTanh(_ x: MLXArray) -> MLXArray {
    // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
    let coeff = MLXArray.scalar(Float(sqrt(2.0 / Double.pi)), like: x)
    let inner = x + MLXArray.scalar(0.044715, like: x) * x.pow(3)
    return MLXArray.scalar(0.5, like: x) * x * (MLXArray.scalar(1.0, like: x) + tanh(coeff * inner))
}

/// Split half pairs: x [..., 2k] → rotate-first-half form used by H3 rope.
/// Mirrors Zig ropeHalf: o1 = x1·cos − x2·sin, o2 = x2·cos + x1·sin.
public func applyRopeHalf(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let hd = x.shape[x.ndim - 1]
    let half = hd / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    let o1 = x1 * cos - x2 * sin
    let o2 = x2 * cos + x1 * sin
    return concatenated([o1, o2], axis: -1)
}

/// 1-D rotary: angles [seq, half] split into paired (θ,θ) layout.
public func ropeHalf(_ invFreq: MLXArray, seq: Int, dtype: DType = .float32) -> (cos: MLXArray, sin: MLXArray) {
    let pos = MLXArray(0..<seq).asType(.float32)
    let angles = pos.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)  // [seq, half]
    let cos = angles.cos().asType(dtype)
    let sin = angles.sin().asType(dtype)
    return (cos, sin)
}

/// H3-style split-half rotation on [1, S, H, hd] with cos/sin [S, hd].
/// dims [0, 48) and [48, 96) rotate; the tail passes through.
public func applyRopePub(_ x: MLXArray, cos: MLXArray, sin: MLXArray, rot: Int) -> MLXArray {
    let hd = x.shape[x.ndim - 1]
    let rotHalf = rot / 2
    // Separated-pair rotary: pairs (x[i], x[i+rotHalf]) share angle cos/sin of length rotHalf.
    let x1 = x[0..., 0..., 0..., ..<rotHalf]
    let x2 = x[0..., 0..., 0..., rotHalf..<rot]
    let tail = x[0..., 0..., 0..., rot...]
    let out1 = x1 * cos - x2 * sin
    let out2 = x2 * cos + x1 * sin
    let outRot = concatenated([out1, out2], axis: -1)
    if rot < hd {
        return concatenated([outRot, tail], axis: -1)
    }
    return outRot
}

/// concat helper for [a, b] along axis.
public func concat(_ arrays: [MLXArray], axis: Int) -> MLXArray {
    concatenated(arrays, axis: axis)
}

/// Convert [N, C, T, H, W] frames → patch rows for the vision tower.
public func visionPatchRows(frames: MLXArray, gridH: Int, gridW: Int, patch: Int = 16) -> MLXArray {
    // frames [N, 3, H, W] (H = gridH·16, W = gridW·16)
    let n = frames.shape[0]
    let h = gridH * patch
    let w = gridW * patch
    let pv = frames.reshaped(n, 3, gridH, patch, gridW, patch)
    let pvt = pv.transposed(0, 2, 4, 3, 5, 1)   // [N, gh, gw, ph, pw, 3]
    return pvt.reshaped(n * gridH * gridW, 3 * patch * patch)
}

// MARK: - Stream helpers

public func defaultStream() -> MLX.Stream? { nil }
