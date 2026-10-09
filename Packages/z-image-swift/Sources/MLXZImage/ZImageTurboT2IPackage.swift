// MLXEngine `textToImage` package for **Z-Image-Turbo** — the distilled 8-step sibling.
//
// Turbo is architecture-identical to base Z-Image (same 6.15B S3-DiT, Qwen3-4B encoder,
// FLUX.1 AE); only the DiT weights + scheduler shift (3.0 vs 6.0) differ, and it runs
// distilled at guidance 0 (no CFG / negative prompt). So this is a thin variant: a distinct
// PackageID + manifest that delegates all lifecycle/inference to an inner ZImageT2IPackage.
// The engine supplies a Turbo ZImageConfiguration at admission (ZImageConfiguration.turbo()).

import Foundation
import MLXToolKit

@InferenceActor
public final class ZImageTurboT2IPackage: ModelPackage {
    public typealias Configuration = ZImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "Tongyi-MAI/Z-Image-Turbo", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Architecture-identical to base Z-Image → same split envelope (delegates to the
                // same inner ZImageT2IPackage / shared ZImageGenerator core). Resident + activation
                // measured via zimage-cli (int4 1024²/8-step: 25.7 GB peak, DiT 3.5 GB;
                // bf16: 33.9 GB peak, DiT 11.7 GB). See ZImageT2IPackage for the split rationale.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 20_000_000_000,
                                   peakActivationBytes: 14_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 15_000_000_000,
                                   peakActivationBytes: 14_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 6_000_000_000,
                                   peakActivationBytes: 20_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "z-image-turbo-t2i",
                    summary: "Z-Image-Turbo 6B text-to-image (Apache-2.0): the fast tier — "
                        + "distilled 8-step at guidance 0 (no CFG), #1-open-weights-class "
                        + "photorealism + EN/CN text; ~13 s @1024² int4 on a 16 GB Mac. "
                        + "Note: low seed variance (a model trait, not a bug).",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "z-image-turbo-img2img",
                    summary: "Z-Image-Turbo image-to-image (Apache-2.0): fast re-generate from an "
                        + "input image + prompt; metaData[\"strength\"] (0–1, default 0.75).",
                    modes: []
                )
            ]
        )
    }

    private let inner: ZImageT2IPackage

    public nonisolated init(configuration: Configuration) {
        self.inner = ZImageT2IPackage(configuration: configuration)
    }

    public func load() async throws { try await inner.load() }
    public func unload() async { await inner.unload() }
    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try await inner.run(request)
    }
}

extension ZImageTurboT2IPackage {
    /// The author one-liner the engine registers (distinct PackageID from base Z-Image).
    public nonisolated static var registration: PackageRegistration {
        .of(ZImageTurboT2IPackage.self)
    }
}
