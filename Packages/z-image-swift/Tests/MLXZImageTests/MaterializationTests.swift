// MLXZImage through the engine's MAT gate (v0.19.0 auto-materialization) + WeightSourcing
// declaration shape. Offline — no network, no weights.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXZImage

final class MaterializationTests: XCTestCase {

    /// A temp snapshot dir with a `transformer/` + `vae/` so an explicit-path config reads as satisfied.
    private func satisfiedSnapshot() throws -> (dir: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "zimage-mat-\(UUID().uuidString)")
        for sub in ["transformer", "vae"] {
            try FileManager.default.createDirectory(
                at: base.appending(path: sub), withIntermediateDirectories: true)
        }
        return (base, { try? FileManager.default.removeItem(at: base) })
    }

    // MARK: - Engine MAT gate

    func testMATGatePassesTurboBF16() throws {
        let (dir, cleanup) = try satisfiedSnapshot()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: ZImageConfiguration.turbo(),
            satisfiedConfiguration: ZImageConfiguration.turbo(snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testMATGatePassesBaseInt4() throws {
        let (dir, cleanup) = try satisfiedSnapshot()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: ZImageConfiguration.base(quant: .int4),
            satisfiedConfiguration: ZImageConfiguration.base(quant: .int4, snapshotPath: dir.path))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testQuantSelectsSuffixedRepo() {
        XCTAssertEqual(ZImageConfiguration.turbo(quant: .bf16).effectiveRepo,
                       "mlx-community/Z-Image-Turbo-bf16")
        XCTAssertEqual(ZImageConfiguration.turbo(quant: .int8).effectiveRepo,
                       "mlx-community/Z-Image-Turbo-8bit")
        XCTAssertEqual(ZImageConfiguration.base(quant: .int4).effectiveRepo,
                       "mlx-community/Z-Image-4bit")
    }

    func testWeightSourcesSingleSnapshot() {
        let sources = ZImageConfiguration.turbo(quant: .int4).weightSources
        XCTAssertEqual(sources.map(\.role), ["snapshot"])
        XCTAssertEqual(sources[0].repo, "mlx-community/Z-Image-Turbo-4bit")
        XCTAssertTrue(sources[0].matching!.contains("transformer/*"))
    }

    // MARK: - Store-layout probe + explicit-path precedence

    func testStoreLayoutSatisfiesAndExplicitPathWins() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "zimage-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = ZImageConfiguration.turbo()   // bf16 → one snapshot source
        // Empty store: the snapshot is missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 1)
        // Populate the expected store layout — path from ModelStore so the fixture
        // tracks the engine's canonical models--org--name layout (contract 1.22.0)
        // instead of a stale literal.
        let dir = ModelStore(root: root).directory(for: "mlx-community/Z-Image-Turbo-bf16")!
        for sub in ["transformer", "vae"] {
            try FileManager.default.createDirectory(
                at: dir.appending(path: sub), withIntermediateDirectories: true)
        }
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: root).isEmpty)
        XCTAssertEqual(cfg.resolvedSnapshotDirectory(storeRoot: root)?.path, dir.path)

        // Explicit snapshotPath wins over the store layout.
        let (snap, cleanup) = try satisfiedSnapshot()
        defer { cleanup() }
        let explicit = ZImageConfiguration.turbo(snapshotPath: snap.path)
        XCTAssertTrue(explicit.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(explicit.resolvedSnapshotDirectory(storeRoot: nil)?.path, snap.path)
    }

    // MARK: - Codable + manifest surface (C11)

    func testCodableRoundTripCarriesTierDefaults() throws {
        let cfg = ZImageConfiguration.base(quant: .int8)
        let decoded = try JSONDecoder().decode(
            ZImageConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.quant, .int8)
        XCTAssertEqual(decoded.defaultSteps, 28)
        XCTAssertEqual(decoded.defaultGuidanceScale, 4.0)
    }

    func testManifestsAreApacheAndDistinct() {
        XCTAssertEqual(ZImageT2IPackage.manifest.license.weightLicense, .apache2)
        XCTAssertEqual(ZImageTurboT2IPackage.manifest.license.weightLicense, .apache2)
        let baseSurface = ZImageT2IPackage.manifest.surfaces[0].name
        let turboSurface = ZImageTurboT2IPackage.manifest.surfaces[0].name
        XCTAssertEqual(baseSurface, "z-image-t2i")
        XCTAssertEqual(turboSurface, "z-image-turbo-t2i")
        XCTAssertNotEqual(baseSurface, turboSurface)
    }
}
