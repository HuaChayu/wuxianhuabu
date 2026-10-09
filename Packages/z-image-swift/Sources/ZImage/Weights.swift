// Weight loading for ZImageTransformer2DModel — mirrors src/utils/loader.py.
// All upstream→Swift key remapping lives HERE (skill rule: sanitize in the loader,
// never the constructor, never first forward). Each remap corresponds to a documented
// structural deviation in Transformer.swift's header.

import Foundation
import MLX
import MLXNN

public enum ZImageWeights {

    /// Precision-sensitive DiT projections kept at bf16 when quantizing (Lens doctrine:
    /// skipping in/out embeds, time embed, final layer, and per-block modulation lifts
    /// int4 quality at the same size). Everything else (attention + FFN Linears in the
    /// 30 main blocks and 4 refiners — the bulk of the 6B) is quantized.
    static func keepHiPrecision(path: String) -> Bool {
        let keepPrefixes = [
            "x_embedder",           // patch in-embed
            "final_layer",          // out projection + its AdaLN
            "t_embedder",           // timestep MLP
            "cap_embedder_linear",  // caption in-projection
        ]
        if keepPrefixes.contains(where: { path.hasPrefix($0) }) { return true }
        // per-block AdaLN modulation (small, precision-sensitive scale/gate producer)
        if path.hasSuffix("adaLN_modulation.0") { return true }
        return false
    }

    /// Quantize a loaded DiT in place (affine, group 64) skipping keepHiPrecision layers.
    public static func quantizeDiT(_ model: ZImageTransformer2DModel, bits: Int, groupSize: Int = 64) {
        quantize(model: model, filter: { path, module in
            guard module is Linear else { return nil }   // norms/params untouched
            if keepHiPrecision(path: path) { return nil }
            return (groupSize, bits, .affine)
        })
        eval(model)
    }

    /// Upstream safetensors key → Swift module path. Order matters (prefix rewrites).
    static let ditKeyRemaps: [(prefix: String, replacement: String)] = [
        ("all_x_embedder.2-1.", "x_embedder."),
        ("all_final_layer.2-1.", "final_layer."),
        ("final_layer.adaLN_modulation.1.", "final_layer.adaLN_linear."),
        ("t_embedder.mlp.0.", "t_embedder.mlp_in."),
        ("t_embedder.mlp.2.", "t_embedder.mlp_out."),
        ("cap_embedder.0.", "cap_embedder_norm."),
        ("cap_embedder.1.", "cap_embedder_linear."),
    ]

    static func remapDiTKey(_ key: String) -> String {
        var key = key
        for (prefix, replacement) in ditKeyRemaps where key.hasPrefix(prefix) {
            key = replacement + key.dropFirst(prefix.count)
            break
        }
        // final_layer remap above only fires post all_final_layer rewrite:
        if key.hasPrefix("final_layer.adaLN_modulation.1.") {
            key = "final_layer.adaLN_linear." + key.dropFirst("final_layer.adaLN_modulation.1.".count)
        }
        return key
    }

    /// Load every *.safetensors shard in a directory into one flat dict.
    static func loadShards(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        precondition(!files.isEmpty, "no safetensors shards in \(directory.path)")

        var weights: [String: MLXArray] = [:]
        for file in files {
            let shard = try MLX.loadArrays(url: file)
            weights.merge(shard) { _, new in new }
        }
        return weights
    }

    /// Load the DiT from `<snapshot>/transformer/`, strict (`verify: .all`).
    /// Turbo ships fp32 shards, Base bf16 — pass `dtype` to unify (nil = keep source).
    public static func loadTransformer(
        snapshotPath: String,
        dtype: DType? = .bfloat16
    ) throws -> ZImageTransformer2DModel {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("transformer")
        var weights = try loadShards(directory: dir)

        weights = Dictionary(uniqueKeysWithValues: weights.map { key, value in
            (remapDiTKey(key), dtype.map { value.asType($0) } ?? value)
        })

        let model = ZImageTransformer2DModel()
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    /// Load the FLUX.1-dev AE from `<snapshot>/vae/` — fp32 (config `force_upcast: true`).
    /// Conv2d weights transpose PT (O,I,kH,kW) → MLX (O,kH,kW,I) here.
    public static func loadVAE(
        snapshotPath: String,
        dtype: DType = .float32
    ) throws -> AutoencoderKL {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("vae")
        let cfg = try ZImageVAEConfig.load(dir.appendingPathComponent("config.json"))
        let vae = AutoencoderKL(cfg)

        var weights: [String: MLXArray] = [:]
        for (key, rawValue) in try loadShards(directory: dir) {
            var value = rawValue
            if key.hasSuffix(".weight"), value.ndim == 4 {
                value = value.transposed(0, 2, 3, 1)
            }
            weights[key] = value.asType(dtype)
        }
        try vae.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(vae)
        return vae
    }

    /// Load the Qwen3-4B hidden-state encoder from `<snapshot>/text_encoder/`.
    /// The encoder runs (n−1) layers for hidden_states[-2]; the checkpoint's last-layer,
    /// final-norm, and (tied) lm_head keys are intentionally dropped here.
    public static func loadTextEncoder(
        snapshotPath: String,
        dtype: DType? = .bfloat16
    ) throws -> Qwen3HiddenStateEncoder {
        let dir = URL(fileURLWithPath: snapshotPath).appendingPathComponent("text_encoder")
        let config = try Qwen3EncoderConfiguration.load(dir.appendingPathComponent("config.json"))
        let lastLayerPrefix = "model.layers.\(config.hiddenLayers - 1)."

        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadShards(directory: dir) {
            guard key.hasPrefix("model.") else { continue }        // drops lm_head.* if present
            let stripped = String(key.dropFirst("model.".count))
            if key.hasPrefix(lastLayerPrefix) || stripped.hasPrefix("norm.") { continue }
            weights[stripped] = dtype.map { value.asType($0) } ?? value
        }

        let encoder = Qwen3HiddenStateEncoder(config)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(encoder)
        return encoder
    }
}
