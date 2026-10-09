// P2 gate — DiT parity vs fp32/CPU oracle goldens (zimage_dit_{aligned,unaligned}).
// Heavy (loads the full 6B DiT fp32): gated behind ZIMAGE_PARITY=1.
// Mirrors ZImageTransformer2DModel.callAsFunction stage by stage so a failure
// localizes to the exact sub-op (granular-goldens doctrine).

import Foundation
import MLX
import XCTest

@testable import ZImage

final class P2DiTParityTests: XCTestCase {

    static let goldensDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Tests/goldens")

    static let snapshotPath = ProcessInfo.processInfo.environment["ZIMAGE_SNAPSHOT"]
        ?? URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/Z-Image-Turbo").path

    // Lazy static — loads once on first access from the test thread, shared by both cases.
    nonisolated(unsafe) static let model: ZImageTransformer2DModel? = {
        guard ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1" else { return nil }
        Device.setDefault(device: .cpu)
        return try! ZImageWeights.loadTransformer(snapshotPath: snapshotPath, dtype: .float32)
    }()

    func requireModel() throws -> ZImageTransformer2DModel {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1",
            "set ZIMAGE_PARITY=1 (and optionally ZIMAGE_SNAPSHOT) to run the P2 gate")
        Device.setDefault(device: .cpu)
        return Self.model!
    }

    struct Stage {
        let name: String
        let ours: MLXArray
        let golden: MLXArray
    }

    func check(_ s: Stage, cosMin: Float = 0.99999, maxAbsMax: Float = 1e-2,
               file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(s.ours.shape, s.golden.shape, "\(s.name) shape", file: file, line: line)
        let a = s.ours.asType(.float32).flattened()
        let b = s.golden.asType(.float32).flattened()
        let cos = MLX.sum(a * b).item(Float.self)
            / (Foundation.sqrt(MLX.sum(a * a).item(Float.self))
               * Foundation.sqrt(MLX.sum(b * b).item(Float.self)) + 1e-12)
        let maxAbs = MLX.abs(a - b).max().item(Float.self)
        let name = s.name.padding(toLength: 22, withPad: " ", startingAt: 0)
        print("  [P2] \(name) cos=\(String(format: "%.7f", cos)) max_abs=\(String(format: "%.3e", maxAbs))")
        XCTAssertGreaterThan(cos, cosMin, "\(s.name) cosine", file: file, line: line)
        XCTAssertLessThan(maxAbs, maxAbsMax, "\(s.name) max_abs", file: file, line: line)
    }

    func runCase(_ caseName: String) throws {
        let model = try requireModel()
        let g = try MLX.loadArrays(
            url: Self.goldensDir.appendingPathComponent("zimage_dit_\(caseName).safetensors"))

        let inX = g["in_x"]!.asType(.float32)      // [16, 1, H, W]
        let inT = g["in_t"]!.asType(.float32)      // [1]
        let inCap = g["in_cap"]!.asType(.float32)  // [capOriLen, 2560]
        let capOriLen = inCap.shape[0]

        // --- stage: timestep embed (forward does t * t_scale first) ---
        let tEmb = model.tEmbedder(inT * model.tScale)
        check(Stage(name: "t_emb", ours: tEmb, golden: g["t_emb"]!))

        // --- stage: patchify + embedders ---
        let (xItems, capItems, sizes) = model.patchifyAndEmbed(
            allImage: [inX], allCapFeats: [inCap], patchSize: 2, fPatchSize: 1)
        let xOriLen = xItems[0].oriLen

        // golden x_embed/cap_embed were hooked on inputs padded by last-row duplication,
        // pre pad-token replacement → compare the original (un-padded) rows only.
        check(Stage(name: "x_embed",
                    ours: xItems[0].tokens[..<xOriLen],
                    golden: g["x_embed"]![..<xOriLen]))
        check(Stage(name: "cap_embed",
                    ours: capItems[0].tokens[..<capOriLen],
                    golden: g["cap_embed"]![..<capOriLen]))

        let adalnInput = tEmb.asType(xItems[0].tokens.dtype)

        // --- stage: noise refiner ---
        let xLens = [xItems[0].paddedLen]
        let xMaxLen = xLens[0]
        var xTokens = ZImageTransformer2DModel.padSequence([xItems[0].tokens], maxLen: xMaxLen)
        let xRope = model.ropeEmbedder(MLXArray(xItems[0].posIds, [xMaxLen, 3]))
        let xCos = xRope.cos[.newAxis, 0..., 0...]
        let xSin = xRope.sin[.newAxis, 0..., 0...]
        for (i, layer) in model.noiseRefiner.enumerated() {
            xTokens = layer(xTokens, attnMask: nil, ropeCos: xCos, ropeSin: xSin,
                            adalnInput: adalnInput)
            eval(xTokens)
            check(Stage(name: "noise_refiner\(i)", ours: xTokens, golden: g["noise_refiner\(i)"]!))
        }

        // --- stage: context refiner ---
        let capMaxLen = capItems[0].paddedLen
        var capTokens = ZImageTransformer2DModel.padSequence([capItems[0].tokens], maxLen: capMaxLen)
        let capRope = model.ropeEmbedder(MLXArray(capItems[0].posIds, [capMaxLen, 3]))
        let capCos = capRope.cos[.newAxis, 0..., 0...]
        let capSin = capRope.sin[.newAxis, 0..., 0...]
        for (i, layer) in model.contextRefiner.enumerated() {
            capTokens = layer(capTokens, attnMask: nil, ropeCos: capCos, ropeSin: capSin)
            eval(capTokens)
            check(Stage(name: "context_refiner\(i)", ours: capTokens, golden: g["context_refiner\(i)"]!))
        }

        // --- stage: unified main stack ---
        var unified = concatenated([xTokens[0], capTokens[0]], axis: 0)[.newAxis, 0..., 0...]
        let uCos = concatenated([xRope.cos, capRope.cos], axis: 0)[.newAxis, 0..., 0...]
        let uSin = concatenated([xRope.sin, capRope.sin], axis: 0)[.newAxis, 0..., 0...]
        let checkpoints = Set([0, 1, 15, 29])
        for (i, layer) in model.layers.enumerated() {
            unified = layer(unified, attnMask: nil, ropeCos: uCos, ropeSin: uSin,
                            adalnInput: adalnInput)
            eval(unified)
            if checkpoints.contains(i) {
                // tolerance loosens with depth — fp32 accumulation over 30 blocks
                check(Stage(name: "block\(i)_out", ours: unified, golden: g["block\(i)_out"]!),
                      cosMin: 0.9999, maxAbsMax: i >= 15 ? 5e-2 : 1e-2)
            }
        }

        // --- stage: final layer + unpatchify ---
        let finalOut = model.finalLayer(unified, adalnInput)
        check(Stage(name: "final_layer", ours: finalOut, golden: g["final_layer"]!),
              cosMin: 0.9999, maxAbsMax: 5e-2)

        let out = model.unpatchify(finalOut[0], size: sizes[0], patchSize: 2, fPatchSize: 1)
        check(Stage(name: "out_final", ours: out, golden: g["out_final"]!),
              cosMin: 0.9999, maxAbsMax: 5e-2)

        // --- whole-forward through the public API must agree with the staged path ---
        let apiOut = model([inX], t: inT, capFeats: [inCap])[0]
        check(Stage(name: "api_vs_staged", ours: apiOut, golden: out),
              cosMin: 0.999999, maxAbsMax: 1e-5)
    }

    func testDiTParityAligned() throws {
        try runCase("aligned")
    }

    func testDiTParityUnaligned() throws {
        try runCase("unaligned")
    }
}
