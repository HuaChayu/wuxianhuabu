// Init-time configuration for the Z-Image MLXEngine packages (C9). Shared by the Base
// (ZImageT2IPackage) and Turbo (ZImageTurboT2IPackage) tiers — the tier is chosen by which
// package wraps the config, which sets defaultSteps / defaultGuidanceScale / provenance.
//
// A snapshot is a diffusers-style tree: transformer/ text_encoder/ vae/ tokenizer/ scheduler/.
// `snapshotPath` is an explicit local override (validation lane); when nil, load() resolves
// against the ModelStore after auto-materializing `weightSources` (v0.19.0).

import Foundation
import MLXToolKit

public struct ZImageConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// mlx-community family base (per-quant repo derived via the quant suffix), e.g.
    /// "mlx-community/Z-Image-Turbo" → "-bf16" / "-8bit" / "-4bit".
    public var repo: String
    public var revision: String?
    /// DiT quant tier. `QuantConfigured` surfaces it so the MemoryGovernor charges the matching
    /// split QuantFootprint (not the bf16 max). int8/int4 select the pre-quantized published repo.
    public var quant: Quant
    /// Explicit local snapshot root (validation / bring-your-own). nil ⇒ auto-materialize.
    public var snapshotPath: String?
    /// Per-tier generation defaults (set by the wrapping package).
    public var defaultSteps: Int
    public var defaultGuidanceScale: Double
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Z-Image-Turbo",
        revision: String? = nil,
        quant: Quant = .bf16,
        snapshotPath: String? = nil,
        defaultSteps: Int = 8,
        defaultGuidanceScale: Double = 0.0,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.modelsRootDirectory = modelsRootDirectory
    }

    // Environment-specific fields (snapshotPath, modelsRootDirectory) are excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, defaultSteps, defaultGuidanceScale
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decode(Quant.self, forKey: .quant)
        defaultSteps = try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 8
        defaultGuidanceScale = try c.decodeIfPresent(Double.self, forKey: .defaultGuidanceScale) ?? 0.0
    }

    /// Base tier (non-distilled, ~28-step CFG). Provenance = the Z-Image family repo.
    public static func base(
        repo: String = "mlx-community/Z-Image", quant: Quant = .bf16,
        snapshotPath: String? = nil, modelsRootDirectory: URL? = nil
    ) -> ZImageConfiguration {
        ZImageConfiguration(
            repo: repo, quant: quant, snapshotPath: snapshotPath,
            defaultSteps: 28, defaultGuidanceScale: 4.0,
            modelsRootDirectory: modelsRootDirectory)
    }

    /// Turbo tier (distilled 8-step, no CFG).
    public static func turbo(
        repo: String = "mlx-community/Z-Image-Turbo", quant: Quant = .bf16,
        snapshotPath: String? = nil, modelsRootDirectory: URL? = nil
    ) -> ZImageConfiguration {
        ZImageConfiguration(
            repo: repo, quant: quant, snapshotPath: snapshotPath,
            defaultSteps: 8, defaultGuidanceScale: 0.0,
            modelsRootDirectory: modelsRootDirectory)
    }
}

// MARK: - Weight sources (auto-materialization, v0.19.0 MAT gate)

extension ZImageConfiguration: WeightSourcing {
    /// The quant-suffixed mlx-community repo the tier materializes from. Each published repo is a
    /// complete snapshot (transformer at that quant + text_encoder + vae + tokenizer + scheduler);
    /// only transformer/ (and the encoder, for the quant tiers) differs across tiers.
    public var effectiveRepo: String {
        switch quant {
        case .int8: return repo + "-8bit"
        case .int4: return repo + "-4bit"
        default: return repo + "-bf16"
        }
    }

    /// One snapshot source; the globs pull the whole diffusers tree.
    public var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: effectiveRepo, revision: revision,
                      matching: ["transformer/*", "text_encoder/*", "vae/*",
                                 "tokenizer/*", "scheduler/*", "*.json"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        // Explicit snapshot dir wins — a readable transformer/ means nothing to fetch.
        if let snapshotPath,
           fm.fileExists(atPath: URL(fileURLWithPath: snapshotPath)
               .appendingPathComponent("transformer").path) {
            return []
        }
        guard let dir = ModelStore(root: storeRoot).directory(for: effectiveRepo) else {
            return weightSources  // no store + no explicit path ⇒ everything missing
        }
        let present = fm.fileExists(atPath: dir.appendingPathComponent("transformer").path)
            && fm.fileExists(atPath: dir.appendingPathComponent("vae").path)
        return present ? [] : weightSources
    }

    /// Store-resolved snapshot directory (what load() uses after materialization).
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if let snapshotPath { return URL(fileURLWithPath: snapshotPath) }
        return ModelStore(root: storeRoot).directory(for: effectiveRepo)
    }
}
