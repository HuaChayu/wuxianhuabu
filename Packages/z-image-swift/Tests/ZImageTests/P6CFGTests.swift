// P6 gate — Base tier + CFG path numeric parity vs the fp32/CPU oracle golden
// (zimage_e2e_cpu_256_6step_cfg4: Base checkpoint shift 6.0, guidance 4.0, negative
// prompt, injected init latents). This is the ONLY gate that exercises the bsz=2
// unified path, pos-anchored CFG (pos + g·(pos−neg)), and cfg_truncation.
// Gated behind ZIMAGE_PARITY=1 + ZIMAGE_BASE_SNAPSHOT.

import Foundation
import MLX
import Tokenizers
import XCTest

@testable import ZImage

final class P6CFGTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    // Base snapshot: default sibling of the Turbo weights dir.
    static let baseSnapshot = ProcessInfo.processInfo.environment["ZIMAGE_BASE_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/Z-Image").path

    static let goldenPrompt =
        "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, "
        + "warm lamp glow, photorealistic"

    func testBaseCFGParityCPU() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1",
            "set ZIMAGE_PARITY=1 to run the P6 gate")
        Device.setDefault(device: .cpu)

        let g = try MLX.loadArrays(
            url: Self.goldensDir.appendingPathComponent("zimage_e2e_cpu_256_6step_cfg4.safetensors"))

        let transformer = try ZImageWeights.loadTransformer(
            snapshotPath: Self.baseSnapshot, dtype: .float32)
        let vae = try ZImageWeights.loadVAE(snapshotPath: Self.baseSnapshot, dtype: .float32)
        let encoder = try ZImageWeights.loadTextEncoder(
            snapshotPath: Self.baseSnapshot, dtype: .float32)
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: Self.baseSnapshot).appendingPathComponent("tokenizer"))
        let textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)
        // Base checkpoint scheduler: shift 6.0.
        let scheduler = FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: 1000, shift: 6.0, useDynamicShifting: false)

        let result = ZImagePipeline.generate(
            transformer: transformer, vae: vae, textEncoder: textEncoder,
            scheduler: scheduler,
            prompt: Self.goldenPrompt,
            height: 256, width: 256,
            numInferenceSteps: 6, guidanceScale: 4.0,
            negativePrompt: "blurry, low quality",
            initLatents: g["init_latents"]!.asType(.float32),
            transformerDtype: .float32)

        let ours = result.latents.flattened()
        let golden = g["final_latent"]!.asType(.float32).flattened()
        let cos = MLX.sum(ours * golden).item(Float.self)
            / (Foundation.sqrt(MLX.sum(ours * ours).item(Float.self))
               * Foundation.sqrt(MLX.sum(golden * golden).item(Float.self)) + 1e-12)
        let latMaxAbs = MLX.abs(ours - golden).max().item(Float.self)
        print("  [P6] Base+CFG final latent cos=\(String(format: "%.6f", cos)) "
              + "max_abs=\(String(format: "%.3e", latMaxAbs))")
        XCTAssertGreaterThan(cos, 0.999, "Base+CFG final latent cosine")

        let mse = MLX.mean(MLX.square(result.image! - g["decoded"]!.asType(.float32))).item(Float.self)
        let psnr = 10 * log10(4.0 / mse)
        print("  [P6] Base+CFG decoded PSNR = \(String(format: "%.2f", psnr)) dB (gate ≥ 40)")
        XCTAssertGreaterThanOrEqual(psnr, 40.0)
    }
}
