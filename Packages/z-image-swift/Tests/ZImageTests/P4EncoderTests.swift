// P4 gate — text conditioning vs fp32/CPU oracle goldens (zimage_encoder).
// Gates: (1) chat-template string EXACT, (2) token ids EXACT (the tokenizer gate —
// any drift here poisons everything downstream), (3) hidden_states[-2] features
// cosine ≥ 0.999 / max_abs < 1e-2 vs the fp32 golden.
// Gated behind ZIMAGE_PARITY=1 (loads the 4B encoder).

import Foundation
import MLX
import Tokenizers
import XCTest

@testable import ZImage

final class P4EncoderTests: XCTestCase {

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

    func requireEncoder() async throws -> ZImageTextEncoder {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ZIMAGE_PARITY"] == "1",
            "set ZIMAGE_PARITY=1 to run the P4 gate")
        Device.setDefault(device: .cpu)
        let encoder = try ZImageWeights.loadTextEncoder(
            snapshotPath: Self.snapshotPath, dtype: .float32)
        let tokenizerURL = URL(fileURLWithPath: Self.snapshotPath)
            .appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        return ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)
    }

    func testChatTemplateExact() throws {
        let meta = try JSONDecoder().decode(
            [String: String].self,
            from: Data(contentsOf: Self.goldensDir.appendingPathComponent("zimage_encoder_meta.json")))
        XCTAssertEqual(ZImageTextEncoder.formatPrompt(Self.goldenPrompt), meta["pos_formatted"])
        XCTAssertEqual(ZImageTextEncoder.formatPrompt(""), meta["neg_formatted"])
    }

    func testTokenIdsAndFeatures() async throws {
        let textEncoder = try await requireEncoder()
        let g = try MLX.loadArrays(
            url: Self.goldensDir.appendingPathComponent("zimage_encoder.safetensors"))

        for (tag, prompt) in [("pos", Self.goldenPrompt), ("neg", "")] {
            let goldenIdsPadded = g["\(tag)_input_ids"]!.asType(.int32).asArray(Int32.self)
            let goldenMask = g["\(tag)_attention_mask"]!.asType(.int32).asArray(Int32.self)
            let validLen = goldenMask.reduce(0) { $0 + Int($1) }
            let goldenIds = Array(goldenIdsPadded.prefix(validLen)).map(Int.init)

            // Gate 2: token ids EXACT
            let ids = textEncoder.tokenize(prompt)
            XCTAssertEqual(ids, goldenIds, "\(tag): token ids must match exactly")

            // Gate 3: features
            let features = textEncoder.encode(prompt)
            eval(features)
            let golden = g["\(tag)_features"]!.asType(.float32)
            XCTAssertEqual(features.shape, golden.shape, "\(tag) feature shape")

            let a = features.asType(.float32).flattened()
            let b = golden.flattened()
            let cos = MLX.sum(a * b).item(Float.self)
                / (Foundation.sqrt(MLX.sum(a * a).item(Float.self))
                   * Foundation.sqrt(MLX.sum(b * b).item(Float.self)) + 1e-12)
            let maxAbs = MLX.abs(a - b).max().item(Float.self)
            print("  [P4] \(tag) features cos=\(String(format: "%.7f", cos)) "
                  + "max_abs=\(String(format: "%.3e", maxAbs)) len=\(features.shape[0])")
            XCTAssertGreaterThan(cos, 0.999, "\(tag) features cosine")
            XCTAssertLessThan(maxAbs, 1e-2, "\(tag) features max_abs")
        }
    }
}
