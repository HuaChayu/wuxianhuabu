// P3 gate — FLUX.1-dev AE decode parity vs fp32/CPU oracle golden (zimage_vae).
// Gate: PSNR ≥ 55 dB on decoded image + pitfall-#7 noise-path smoke (random Gaussian
// through denorm→decode must not show periodic patterns — artifact saved for eyeball).
// Gated behind ZIMAGE_PARITY=1 (loads real VAE weights).

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import ZImage

final class P3VAETests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["ZIMAGE_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/Z-Image-Turbo").path

    func requireVAE() throws -> AutoencoderKL {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1",
            "set ZIMAGE_PARITY=1 to run the P3 gate")
        Device.setDefault(device: .cpu)
        return try ZImageWeights.loadVAE(snapshotPath: Self.snapshotPath, dtype: .float32)
    }

    func testDecodeParityPSNR() throws {
        let vae = try requireVAE()
        let g = try MLX.loadArrays(
            url: Self.goldensDir.appendingPathComponent("zimage_vae.safetensors"))

        let inLatent = g["in_latent"]!.asType(.float32)
        let golden = g["decoded"]!.asType(.float32)

        // pipeline denorm: z / scaling_factor + shift_factor
        let z = inLatent / vae.scalingFactor + vae.shiftFactor
        let denormGolden = g["denormed"]!.asType(.float32)
        let denormMaxAbs = MLX.abs(z - denormGolden).max().item(Float.self)
        XCTAssertLessThan(denormMaxAbs, 1e-5, "denorm max_abs")

        let decoded = vae.decode(z)
        eval(decoded)
        XCTAssertEqual(decoded.shape, golden.shape)

        // PSNR over [-1, 1]-range images (peak 2.0)
        let mse = MLX.mean(MLX.square(decoded - golden)).item(Float.self)
        let psnr = 10 * log10(4.0 / mse)
        print("  [P3] decode PSNR = \(String(format: "%.2f", psnr)) dB (gate ≥ 55)")
        XCTAssertGreaterThanOrEqual(psnr, 55.0)
    }

    func testNoisePathSmoke() throws {
        let vae = try requireVAE()
        // Random Gaussian latent through the full post-DiT chain (pitfall #7):
        // any periodic pattern at stride 2/4/8/16 means a spatial op is broken.
        MLXRandom.seed(0)
        let noise = MLXRandom.normal([1, 16, 32, 32])
        let z = noise / vae.scalingFactor + vae.shiftFactor
        let decoded = vae.decode(z)
        eval(decoded)

        XCTAssertEqual(decoded.shape, [1, 3, 256, 256])
        let maxVal = MLX.abs(decoded).max().item(Float.self)
        XCTAssertTrue(maxVal.isFinite, "decode produced non-finite values")

        // Cheap periodicity probe: fold H into stride-8 phase bins; a broken
        // pixel-shuffle/upsample shows as large per-phase mean divergence.
        let img = decoded[0][0]  // [256, 256]
        let folded = img.reshaped(32, 8, 256)
        let phaseMeans = MLX.mean(folded, axes: [0, 2])  // [8]
        eval(phaseMeans)
        let spread = (phaseMeans.max() - phaseMeans.min()).item(Float.self)
        let overallStd = MLX.sqrt(MLX.variance(img)).item(Float.self)
        print("  [P3] noise-path phase spread \(spread) vs std \(overallStd)")
        XCTAssertLessThan(spread, overallStd * 0.5, "stride-8 phase structure detected")

        // Save for the mandatory eyeball (PORTING-SPEC P3).
        let arr = MLX.clip((decoded[0] / 2 + 0.5), min: 0, max: 1).transposed(1, 2, 0) * 255
        eval(arr)
        let out = Self.goldensDir.appendingPathComponent("p3_noise_smoke.safetensors")
        try MLX.save(arrays: ["noise_decode_rgb": arr], url: out)
        print("  [P3] noise-path artifact saved: \(out.lastPathComponent)")
    }
}
