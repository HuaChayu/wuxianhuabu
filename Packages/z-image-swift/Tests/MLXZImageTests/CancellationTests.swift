// CancellationTests.swift — Z-Image (Base + Turbo) through the engine's CAN gate (offline,
// no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled: the entry
// checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before notLoaded
// validation) fires before weights are touched, so a stub configuration suffices. CAN-3 is
// the document of record for the checkpoint cadence:
//   - denoise/step — `if Task.isCancelled { break }` at the top of the denoise loop in
//     ZImagePipeline.generate (Sources/ZImage/Pipeline.swift), the shared loop behind both
//     t2i and img2img on both tiers (non-throwing core API — sanctioned break shape).
//   - pre-decode seam — a cancelled task skips the monolithic VAE decode (the decode itself
//     is ONE MLX eval, no chunk loop, so no per-chunk cadence is claimed).
//   - the wrapper's post-generate `try Task.checkCancellation()` rethrows the
//     CancellationError UNCHANGED (no catch blocks anywhere in Sources — nothing to launder).
// Turbo delegates run() to an inner ZImageT2IPackage, so one denoise-loop cadence covers
// both PackageIDs; each still passes the gate independently below.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXZImage

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRunBase() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation or weights are touched, so this is offline-safe.
        let package = ZImageT2IPackage(configuration: ZImageConfiguration.base(quant: .int4))
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledRunTurbo() async {
        let package = ZImageTurboT2IPackage(configuration: ZImageConfiguration.turbo())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    /// Both tiers share the ZImagePipeline denoise loop: cancellation is checked once per
    /// denoise step (Task.isCancelled break, Pipeline.swift denoise loop), and a cancelled
    /// task additionally skips the monolithic VAE decode. Decode has no chunk loop, so only
    /// the real denoise/step cadence is declared.
    private var posture: CancellationConformance.CheckpointPosture {
        .cadence([
            .init(phase: .denoise, unit: .step)
        ])
    }

    func testCANCadenceDeclarationBase() {
        // Multi-GB peak activation (14–20 GB declared) implies long runs — the sub-second
        // exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: ZImageT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: ZImageT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclarationTurbo() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: ZImageTurboT2IPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: ZImageTurboT2IPackage.manifest, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }
}
