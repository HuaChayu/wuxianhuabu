// P5 gate — full-pipeline numeric parity vs the fp32/CPU oracle e2e golden
// (zimage_e2e_cpu_256_4step: injected init latents, 4 steps, guidance 0).
// Gates: scheduler trace exact, final-latent cosine ≥ 0.999, decoded PSNR ≥ 40 dB.
// The 1024² MPS bf16 golden is the semantic/eyeball reference (backend trajectories
// diverge at bf16 over 8 steps); full-res GPU rendering is the CLI lane's job.
// Gated behind ZIMAGE_PARITY=1 (loads DiT + encoder + VAE, ~45 GB fp32).

import Foundation
import MLX
import Tokenizers
import XCTest

@testable import ZImage

final class P5PipelineTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["ZIMAGE_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/Z-Image-Turbo").path

    static let goldenPrompt =
        "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, "
        + "warm lamp glow, photorealistic"

    func testE2EParityCPU256() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1",
            "set ZIMAGE_PARITY=1 to run the P5 gate")
        Device.setDefault(device: .cpu)

        let g = try MLX.loadArrays(
            url: Self.goldensDir.appendingPathComponent("zimage_e2e_cpu_256_4step.safetensors"))

        let transformer = try ZImageWeights.loadTransformer(
            snapshotPath: Self.snapshotPath, dtype: .float32)
        let vae = try ZImageWeights.loadVAE(snapshotPath: Self.snapshotPath, dtype: .float32)
        let encoder = try ZImageWeights.loadTextEncoder(
            snapshotPath: Self.snapshotPath, dtype: .float32)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: Self.snapshotPath).appendingPathComponent("tokenizer"))
        let textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)
        let scheduler = FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: 1000, shift: 3.0, useDynamicShifting: false)

        let result = ZImagePipeline.generate(
            transformer: transformer, vae: vae, textEncoder: textEncoder,
            scheduler: scheduler,
            prompt: Self.goldenPrompt,
            height: 256, width: 256,
            numInferenceSteps: 4, guidanceScale: 0.0,
            initLatents: g["init_latents"]!.asType(.float32),
            transformerDtype: .float32)

        // scheduler trace exact vs the golden run
        let sigmaMaxAbs = MLX.abs(MLXArray(scheduler.sigmas) - g["sigmas"]!).max().item(Float.self)
        XCTAssertLessThan(sigmaMaxAbs, 2e-6, "sigma trace")

        // final latent
        let ours = result.latents.flattened()
        let golden = g["final_latent"]!.asType(.float32).flattened()
        let cos = MLX.sum(ours * golden).item(Float.self)
            / (Foundation.sqrt(MLX.sum(ours * ours).item(Float.self))
               * Foundation.sqrt(MLX.sum(golden * golden).item(Float.self)) + 1e-12)
        let latMaxAbs = MLX.abs(ours - golden).max().item(Float.self)
        print("  [P5] final latent cos=\(String(format: "%.6f", cos)) "
              + "max_abs=\(String(format: "%.3e", latMaxAbs))")
        XCTAssertGreaterThan(cos, 0.999, "final latent cosine")

        // decoded image PSNR
        let img = result.image!
        let goldenImg = g["decoded"]!.asType(.float32)
        XCTAssertEqual(img.shape, goldenImg.shape)
        let mse = MLX.mean(MLX.square(img - goldenImg)).item(Float.self)
        let psnr = 10 * log10(4.0 / mse)
        print("  [P5] decoded PSNR = \(String(format: "%.2f", psnr)) dB (gate ≥ 40)")
        XCTAssertGreaterThanOrEqual(psnr, 40.0)
    }
}
