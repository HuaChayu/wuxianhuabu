// LIVE gate for engine-executed first-run materialization (contract 1.24.0): drives a real
// `MLXServeEngine.prepare()` against an EMPTY store and asserts the engine downloaded the
// declared source before `load()` ran. The offline MAT gate proves the declaration shape;
// this proves the executor actually fetches.
//
// Skipped unless ZIMAGE_MAT_E2E=1 — it pulls ~100 MB from the public hub:
//
//   ZIMAGE_MAT_E2E=1 swift test --filter EngineMaterializationE2ETests
//
// One narrow source (the smallest real shard of the Z-Image text encoder) stands in for the
// full snapshot; the engine's executor is repo-generic, so the 12 GB tree would only make the
// gate slower, not stronger.

import Foundation
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXZImage

/// Mirrors a real package's storage shape, but declares ONE small file — and refuses to load
/// unless the engine already materialized it. `prepare()` succeeding IS the evidence.
private struct E2EConfiguration: PackageConfiguration, WeightSourcing, ModelStorable {
    var modelsRootDirectory: URL?
    var weightSources: [WeightSource] {
        [WeightSource(role: "snapshot", repo: "mlx-community/Z-Image-Turbo-bf16", revision: nil,
                      matching: ["text_encoder/model-00003-of-00003.safetensors"])]
    }
}

@InferenceActor
private final class E2EPackage: ModelPackage {
    typealias Configuration = E2EConfiguration
    nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(sourceRepo: "mlx-community/Z-Image-Turbo-bf16",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                footprints: [QuantFootprint(quant: .bf16, residentBytes: 1)],
                requiredBackends: [.metalGPU]),
            surfaces: [T2IContract.descriptor(name: "zimage-mat-e2e", summary: "live MAT gate")])
    }
    private let configuration: E2EConfiguration
    nonisolated init(configuration: E2EConfiguration) { self.configuration = configuration }
    func load() async throws {
        let missing = configuration.missingWeightSources(storeRoot: configuration.modelsRootDirectory)
        guard missing.isEmpty else { throw PackageError.notLoaded }
    }
    func unload() async {}
    func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        throw PackageError.unsupportedCapability(request.capability)
    }
}

final class EngineMaterializationE2ETests: XCTestCase {

    func testEnginePrepareMaterializesFromHub() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["ZIMAGE_MAT_E2E"] == "1",
                          "live download gate — set ZIMAGE_MAT_E2E=1")

        let root = FileManager.default.temporaryDirectory
            .appending(path: "zimage-mat-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let store = ModelStore(root: root)
        let engine = MLXServeEngine()
        await engine.useModelStore(store)
        var config = E2EConfiguration()
        config.modelsRootDirectory = root
        try await engine.register(PackageRegistration.of(E2EPackage.self), configuration: config)

        // Empty store ⇒ the engine reports the download up front.
        let needs = await engine.needsDownload(.textToImage)
        XCTAssertTrue(needs, "empty store should report needsDownload")

        // load() throws unless the file is already there, so prepare() succeeding proves the
        // engine executed materialization BEFORE constructing/loading the package.
        try await engine.prepare(.textToImage)

        let file = try XCTUnwrap(store.directory(for: "mlx-community/Z-Image-Turbo-bf16"))
            .appending(path: "text_encoder/model-00003-of-00003.safetensors")
        let size = try FileManager.default
            .attributesOfItem(atPath: file.path)[.size] as? Int ?? 0
        XCTAssertEqual(size, 99_630_640, "downloaded shard should be byte-complete")
        let after = await engine.needsDownload(.textToImage)
        XCTAssertFalse(after, "satisfied store should no longer report needsDownload")

        // Idempotency across residencies: a satisfied store must not re-download.
        await engine.evict(.textToImage)
        try await engine.prepare(.textToImage)
    }
}
