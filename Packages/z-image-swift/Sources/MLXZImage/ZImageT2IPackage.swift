// MLXEngine `textToImage` package over the Z-Image core — the BASE tier (non-distilled,
// ~28-step CFG). Z-Image (Tongyi-MAI, Apache-2.0): Qwen3-4B thinking-template conditioning
// (hidden_states[-2]) → 6.15B single-stream S3-DiT → FLUX.1-dev AE. Swift core parity-locked
// vs PT goldens (DiT cos ≥0.9999999 · VAE 118 dB · encoder ids-exact · e2e 105–108 dB).
//
// Coexists with the Turbo tier and the other T2I backers (Lens, ERNIE-Image) via the
// multi-package registry — apps select by PackageID ("z-image-t2i" vs "z-image-turbo-t2i")
// or setDefault per device tier.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXProfiling
import MLXToolKit
import Tokenizers
import UniformTypeIdentifiers
import ZImage

public enum ZImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Z-Image snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class ZImageT2IPackage: ModelPackage {
    public typealias Configuration = ZImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Fully Apache-2.0: DiT + Qwen3-4B encoder + FLUX.1 AE weights all Apache; the
            // Swift port code is MIT. No acknowledgement gate (unlike NC models).
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "Tongyi-MAI/Z-Image", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (efficiency contract 1.14.0). Resident floor = DiT + VAE +
                // Qwen3-4B encoder (all co-resident — encoder is small enough here that per-stage
                // eviction isn't warranted, unlike ERNIE's Mistral-3B). Measured on-disk / active:
                //   bf16: DiT 11.7 GB + encoder ~8 GB + VAE ~0.3 GB ≈ 20 GB resident.
                //   int8: DiT 6.4 GB + encoder ~8 GB (bf16) + VAE 0.3 ≈ 15 GB.
                //   int4: DiT 3.5 GB + encoder ~2.3 GB (int4) + VAE 0.3 ≈ 6 GB.
                // Activation = denoise + VAE-decode scratch at 1024²; GPU peak MEASURED via
                // zimage-cli: bf16 render peak 33.9 GB (act ≈ 14 GB over the 20 GB floor),
                // int4 peak 25.7 GB (act ≈ 20 GB — decode conv scratch dominates on int4).
                // [residentBytes = measured active post-load (solid). peakActivationBytes is a
                //  GPU-smoke figure; smoke MLX-peak under-reads process phys_footprint ~2.7×
                //  (BiRefNet lesson) — FLAGGED for an in-app phys re-baseline once Z-Image is
                //  registered in the MLXEngineImage app.]
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
                chipFloor: nil  // memory admission gates; no chip-tier floor beyond it
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "z-image-t2i",
                    summary: "Z-Image 6B single-stream S3-DiT text-to-image (Apache-2.0): the "
                        + "quality/LoRA tier — non-distilled ~28-step with CFG + negative "
                        + "prompts, strong photorealism and EN/CN text rendering, 512²–2048².",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "z-image-img2img",
                    summary: "Z-Image image-to-image (Apache-2.0): re-generate from an input image "
                        + "conditioned on a prompt; metaData[\"strength\"] (0–1, default 0.75) sets "
                        + "how much of the input is preserved vs redrawn.",
                    modes: []
                )
            ]
        )
    }

    let configuration: Configuration
    private var generator: ZImageGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }

        // First-run materialization is engine-executed since mlx-engine-swift 0.32.0
        // (contract 1.24.0): resident()/prepare() downloads missing WeightSourcing
        // sources before load() runs. The guard below stays as the offline backstop —
        // absent weights with no store root still fail legibly here.
        guard let snapshot = configuration.resolvedSnapshotDirectory(
            storeRoot: configuration.modelsRootDirectory),
            FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw ZImagePackageError.unreadableSnapshot(
                configuration.snapshotPath ?? configuration.effectiveRepo)
        }

        // DiT: load bf16, quantize on load for int8/int4. (Pre-quantized published repos —
        // smaller download — are a later optimization; resident footprint is identical.)
        let transformer = try ZImageWeights.loadTransformer(
            snapshotPath: snapshot.path, dtype: .bfloat16)
        switch configuration.quant {
        case .int8: ZImageWeights.quantizeDiT(transformer, bits: 8)
        case .int4: ZImageWeights.quantizeDiT(transformer, bits: 4)
        default: break
        }
        // VAE fp32 (force_upcast); encoder bf16 (int4 encoder is a future publish optimization).
        let vae = try ZImageWeights.loadVAE(snapshotPath: snapshot.path, dtype: .float32)
        let encoder = try ZImageWeights.loadTextEncoder(
            snapshotPath: snapshot.path, dtype: .bfloat16)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: snapshot.appendingPathComponent("tokenizer"))
        let textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)

        // Scheduler shift is config-driven from the snapshot (Base 6.0, Turbo 3.0).
        let (shift, dynamic) = Self.readSchedulerConfig(snapshot: snapshot)

        generator = ZImageGenerator(
            transformer: transformer, vae: vae, textEncoder: textEncoder,
            schedulerShift: shift, schedulerDynamic: dynamic, transformerDtype: .bfloat16)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the core's denoise loop bails per step
        // (`Task.isCancelled` break in ZImagePipeline.generate), a cancelled task skips the
        // VAE decode, and the post-generate checkpoints below rethrow CancellationError
        // unchanged (never wrapped in ZImagePackageError).
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        let prof = MLXProfiler.shared

        if let t2i = request as? T2IRequest {
            let width = ((t2i.width ?? 1024) / 16) * 16
            let height = ((t2i.height ?? 1024) / 16) * 16
            let steps = t2i.steps ?? configuration.defaultSteps
            let guidance = Float(t2i.guidanceScale ?? configuration.defaultGuidanceScale)
            prof.beginRun("z-image textToImage steps=\(steps) \(width)x\(height)")
            let (pixels, w, h) = generator.generate(
                prompt: t2i.prompt, negativePrompt: t2i.negativePrompt,
                width: width, height: height, steps: steps, guidanceScale: guidance, seed: t2i.seed ?? 0)
            prof.endRun(denominators: ["step": Double(steps)])
            try Task.checkCancellation()
            return T2IResponse(image: Image(format: .png, data: try Self.encodePNG(pixels: pixels, width: w, height: h), width: w, height: h))
        }

        if let edit = request as? IEditRequest, let first = edit.images.first {
            let width = ((edit.width ?? 1024) / 16) * 16
            let height = ((edit.height ?? 1024) / 16) * 16
            let steps = edit.steps ?? configuration.defaultSteps
            let guidance = Float(edit.guidanceScale ?? configuration.defaultGuidanceScale)
            // Default strength 0.75 — a more useful midpoint than diffusers' 0.6: on the distilled
            // Turbo, 0.6 barely moves global/time-of-day edits (the preserved low-freq wins), while
            // 0.75 gives visible change on most prompts and still holds composition. Dramatic
            // environment swaps (e.g. day→night) still want ~0.9; callers override via metaData.
            var strength: Float = 0.75
            if case .double(let d)? = edit.metaData["strength"] { strength = Float(d) }
            else if case .int(let n)? = edit.metaData["strength"] { strength = Float(n) }
            let image = try Self.decodeInputImage(first, width: width, height: height)
            prof.beginRun("z-image img2img strength=\(strength) steps=\(steps) \(width)x\(height)")
            let (pixels, w, h) = generator.generateImg2Img(
                prompt: edit.prompt, negativePrompt: edit.negativePrompt, image: image,
                width: width, height: height, steps: steps, guidanceScale: guidance,
                strength: strength, seed: edit.seed ?? 0)
            prof.endRun(denominators: ["step": Double(steps)])
            try Task.checkCancellation()
            return IEditResponse(image: Image(format: .png, data: try Self.encodePNG(pixels: pixels, width: w, height: h), width: w, height: h))
        }

        throw PackageError.unsupportedCapability(request.capability)
    }

    /// Decode an input `Image` → [1,3,height,width] in [-1,1], scaled.
    nonisolated static func decodeInputImage(_ image: Image, width: Int, height: Int) throws -> MLXArray {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw ZImagePackageError.pngEncode }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let ctx = CGContext(data: &buf, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [Float](repeating: 0, count: 3 * width * height)
        for p in 0..<(width * height) {
            rgb[p] = Float(buf[p*4]) / 127.5 - 1
            rgb[width*height + p] = Float(buf[p*4+1]) / 127.5 - 1
            rgb[2*width*height + p] = Float(buf[p*4+2]) / 127.5 - 1
        }
        return MLXArray(rgb, [1, 3, height, width])
    }

    /// Read shift / use_dynamic_shifting from the snapshot's scheduler config (Base 6.0 / Turbo 3.0).
    nonisolated static func readSchedulerConfig(snapshot: URL) -> (shift: Float, dynamic: Bool) {
        struct SchedCfg: Codable { var shift: Float?; var use_dynamic_shifting: Bool? }
        let url = snapshot.appendingPathComponent("scheduler/scheduler_config.json")
        guard let data = try? Data(contentsOf: url),
              let cfg = try? JSONDecoder().decode(SchedCfg.self, from: data)
        else { return (3.0, false) }
        return (cfg.shift ?? 3.0, cfg.use_dynamic_shifting ?? false)
    }

    /// Interleaved RGB8 → PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw ZImagePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]
            buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw ZImagePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw ZImagePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ZImagePackageError.pngEncode }
        return out as Data
    }
}

extension ZImageT2IPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(ZImageT2IPackage.self)
    }
}
