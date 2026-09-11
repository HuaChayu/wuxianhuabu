// H3Layout.swift
// MiniMax H3 (Hailuo 3.0) — packed-sequence layout, frame ladder, timestep
// plans and RoPE tables for the Swift + MLX port.
// Ported from mlx-serve-main/src/minimax_h3.zig (frame ladder §"Frame ladder",
// canvas §"Canvas", dual-sigma §"Dual sigma schedule", position grids,
// PackedLayout.initFull, buildTimestepPlan, sigmaSchedule, buildRope).
//
// Contents:
//   - Frame ladder: temporalShape / alignFrameCount / alignRefFrameCount /
//     videoLatentT / qwenSampledFrames
//   - Canvas math: adaptCanvas / refImageCanvas / refVideoCanvas
//   - Dual-sigma: timeShiftSigma / timeShiftSlope / sigmaSchedule
//   - Position grids: axisFromSqrtArea / videoTSpan / videoTGrid
//   - SegmentKind / Segment / RefBlock / PackedLayout (initFull)
//   - RopeTables / buildRope (interleaved 3-axis angles)
//   - TimestepPlan / buildTimestepPlan / buildTimestepPlanGlobal / collectScheduleTs
//   - Sparse-attention policy + attention-broadcast gating

import Foundation
import MLX
import Darwin

// MARK: - Frame ladder

public struct TemporalShape {
    public var frameCount: UInt32
    public var latentT: UInt32
    public var audioT: UInt32
    public init(frameCount: UInt32, latentT: UInt32, audioT: UInt32) {
        self.frameCount = frameCount
        self.latentT = latentT
        self.audioT = audioT
    }
}

/// Python's `round` (half-to-even). `nom_w / 32` can land exactly on .5.
public func roundHalfEven(_ x: Double) -> Double {
    let fl = floor(x)
    let frac = x - fl
    if frac > 0.5 { return fl + 1 }
    if frac < 0.5 { return fl }
    return (fl.truncatingRemainder(dividingBy: 2.0) == 0) ? fl : fl + 1
}

/// Snap UP to the model's 17k+5 grid: 40 -> 56, 200 -> 209, 363 -> 379.
public func alignFrameCount(_ n: UInt32) -> UInt32 {
    var v = n
    while v % 17 != 5 { v += 1 }
    return v
}

/// Snap DOWN to the same ladder with the model's 5-frame floor (for refs).
public func alignRefFrameCount(_ n: UInt32) -> UInt32 {
    if n < 5 { return 0 }
    var v = n
    while v % 17 != 5 { v -= 1 }
    return v
}

/// Latent frames for an aligned source-frame count.
public func videoLatentT(_ frameCount: UInt32) -> UInt32 {
    if frameCount <= 5 { return 2 }
    return ((frameCount - 5) / 17) * 5 + 2
}

public func qwenFrameStride() -> UInt32 { H3Const.fps / 2 }

public func qwenSampledFrames(_ alignedFrames: UInt32) -> UInt32 {
    let stride = qwenFrameStride()
    return (alignedFrames + stride - 1) / stride
}

/// Aligned frame count + latent/audio latent counts.
public func temporalShape(_ length: UInt32) -> TemporalShape {
    let frameCount = alignFrameCount(max(5, length))
    let fc = Double(frameCount)
    let audioT = (fc / Double(H3Const.fps) * Double(H3Const.audioLatentFPS)).rounded()
    return TemporalShape(frameCount: frameCount, latentT: videoLatentT(frameCount), audioT: UInt32(audioT))
}

// MARK: - Canvas

public struct Canvas {
    public var w: UInt32
    public var h: UInt32
    public init(w: UInt32, h: UInt32) {
        self.w = w
        self.h = h
    }
}

/// 768-short-edge canvas under the 768*1344 area cap, each axis snapped to 32.
public func adaptCanvas(_ w: UInt32, _ h: UInt32) -> Canvas {
    let fw = Double(w)
    let fh = Double(h)
    let ratio = fw / fh
    let base = Double(H3Const.baseShortEdge)
    var nomW: Double = (ratio >= 1.0) ? base * ratio : base
    var nomH: Double = (ratio >= 1.0) ? base : base / ratio
    let cap = Double(H3Const.maxPixels)
    if nomW * nomH > cap {
        let s = sqrt(cap / (nomW * nomH))
        nomW *= s
        nomH *= s
    }
    let m = Double(H3Const.canvasMultiple)
    return Canvas(
        w: max(H3Const.canvasMultiple, UInt32(roundHalfEven(nomW / m) * m)),
        h: max(H3Const.canvasMultiple, UInt32(roundHalfEven(nomH / m) * m))
    )
}

public enum RefImageSizing {
    case match, max
}

func snap32(_ v: Double) -> UInt32 {
    let m = Double(H3Const.canvasMultiple)
    let r = roundHalfEven(v / m) * m
    return max(H3Const.canvasMultiple, UInt32(r))
}

/// Reference image canvas — never upscales.
public func refImageCanvas(_ w: UInt32, _ h: UInt32, genW: UInt32, genH: UInt32, mode: RefImageSizing) -> Canvas {
    let fw = Double(w)
    let fh = Double(h)
    let scale: Double
    switch mode {
    case .match:
        scale = min(1.0, sqrt((Double(genW) * Double(genH)) / (fw * fh)))
    case .max:
        scale = min(1.0, Double(H3Const.refImageShortEdge) / Double(min(w, h)))
    }
    return Canvas(w: snap32(fw * scale), h: snap32(fh * scale))
}

/// Reference video canvas — never enlarged beyond its own pixels.
public func refVideoCanvas(_ w: UInt32, _ h: UInt32) -> Canvas {
    let c = adaptCanvas(w, h)
    if w * h < c.w * c.h {
        return Canvas(w: snap32(Double(w)), h: snap32(Double(h)))
    }
    return c
}

// MARK: - Dual sigma schedule

/// Map a sigma from one shifted schedule to another through the shared base grid.
public func timeShiftSigma(_ sigma: Double, fromShift: Double, toShift: Double) -> Double {
    let base = shiftToBase(sigma, fromShift)
    return toShift * base / (1.0 + (toShift - 1.0) * base)
}

func shiftToBase(_ sigma: Double, _ shift: Double) -> Double {
    sigma / (shift + sigma * (1.0 - shift))
}

/// d(sigma_to)/d(sigma_from) at the same base-grid point — scales the audio
/// velocity so the flat ODE integrates the audio stream's true schedule.
public func timeShiftSlope(_ sigma: Double, fromShift: Double, toShift: Double) -> Double {
    let base = shiftToBase(sigma, fromShift)
    let num = 1.0 + (fromShift - 1.0) * base
    let den = 1.0 + (toShift - 1.0) * base
    return (toShift * num * num) / (fromShift * den * den)
}

/// `sigmas[i] = shift·b/(1+(shift-1)·b)` for `b = 1 - i/steps`; length steps+1, ends at 0.
public func sigmaSchedule(steps: UInt32, shift: Double) -> [Double] {
    var out = [Double](repeating: 0, count: Int(steps) + 1)
    for i in 0...Int(steps) {
        let base = 1.0 - Double(i) / Double(steps)
        out[i] = timeShiftSigma(base, fromShift: 1.0, toShift: shift)
    }
    return out
}

// MARK: - Position grids

/// Area-normalized coordinates along one spatial axis:
/// n = dim/patch values of (arange(n)·ratio/n + (1-ratio)/2)·32.
public func axisFromSqrtArea(dim: UInt32, patch: UInt32, sqrtArea: Double) -> [Double] {
    let n = Int(dim / patch)
    let ratio = Double(dim) / sqrtArea
    let step = ratio / Double(n)
    let start = (1.0 - ratio) / 2.0
    return (0..<n).map { (Double($0) * step + start) * 32.0 }
}

/// Temporal span of latent frame k, in grid units.
public func videoTSpan(_ k: Int) -> Double {
    H3Const.frameRescale * Double(H3Const.framePerToken[k % H3Const.framePerToken.count])
}

/// Exclusive cumulative sum of the first n spans, offset by `origin`.
public func videoTGrid(_ n: UInt32, origin: Double) -> [Double] {
    var out = [Double](repeating: 0, count: Int(n))
    var acc = origin
    for i in 0..<Int(n) {
        out[i] = acc
        acc += videoTSpan(i)
    }
    return out
}

// MARK: - Segments

public enum SegmentKind: Int {
    case text, cond, refImg, refAudio, audio, video

    /// Which embedding stream feeds this segment's rows.
    public var stream: StreamKind {
        switch self {
        case .text: return .text
        case .cond, .refImg, .video: return .video
        case .refAudio, .audio: return .audio
        }
    }
    /// AdaLN modality tag: video 0, text 1, audio 2. Mod row = t_row·3 + tag.
    public var modalityTag: UInt32 {
        switch self {
        case .video, .cond, .refImg: return 0
        case .text: return 1
        case .audio, .refAudio: return 2
        }
    }
    public enum StreamKind { case text, video, audio }
}

public struct Segment {
    public var start: UInt32
    public var end: UInt32
    public var kind: SegmentKind
    public init(start: UInt32, end: UInt32, kind: SegmentKind) {
        self.start = start
        self.end = end
        self.kind = kind
    }
}

public enum KeyframeAnchor {
    case first, last
}

/// One ref2va reference block, in request order (images → videos → standalone
/// audio; a video's soundtrack packs immediately before its own rows).
public struct RefBlock {
    public enum RefKind { case image, audio, video }
    public var kind: RefKind
    public var latentH: UInt32 = 0
    public var latentW: UInt32 = 0
    public var latentT: UInt32 = 0
    public var audioT: UInt32 = 0
    public init(kind: RefKind, latentH: UInt32, latentW: UInt32, latentT: UInt32, audioT: UInt32) {
        self.kind = kind
        self.latentH = latentH
        self.latentW = latentW
        self.latentT = latentT
        self.audioT = audioT
    }
}

func refFrameRows(_ b: RefBlock) -> UInt32 {
    (b.latentH / H3Const.patchH) * (b.latentW / H3Const.patchW)
}

// MARK: - PackedLayout

/// Static packed-sequence structure. Layout is `[text | cond... | audio | video]`
/// — the target audio/video are ALWAYS the last two segments.
public final class PackedLayout {
    public var seqLen: UInt32
    public var segments: [Segment]
    /// [seq_len × 3] row-major (t, h, w), f64.
    public var positionIds: [Double]
    /// Per video-stream row, in packed order: true = denoised target row.
    public var imgUpdate: [Bool]
    public var audioUpdate: [Bool]
    /// AdaLN modality tag per TEXT row (borrowed; non-empty only with vision blocks).
    public var textTags: [UInt8]
    public var textLen: UInt32
    public var latentT: UInt32
    public var latentH: UInt32
    public var latentW: UInt32
    public var audioT: UInt32

    public init(seqLen: UInt32, segments: [Segment], positionIds: [Double],
                imgUpdate: [Bool], audioUpdate: [Bool], textTags: [UInt8],
                textLen: UInt32, latentT: UInt32, latentH: UInt32, latentW: UInt32, audioT: UInt32) {
        self.seqLen = seqLen
        self.segments = segments
        self.positionIds = positionIds
        self.imgUpdate = imgUpdate
        self.audioUpdate = audioUpdate
        self.textTags = textTags
        self.textLen = textLen
        self.latentT = latentT
        self.latentH = latentH
        self.latentW = latentW
        self.audioT = audioT
    }

    public convenience init(textLen: UInt32, latentT: UInt32, latentH: UInt32, latentW: UInt32,
                            audioT: UInt32, keyframes: [KeyframeAnchor], frameCount: UInt32) {
        self.init(textLen: textLen, latentT: latentT, latentH: latentH, latentW: latentW,
                  audioT: audioT, keyframes: keyframes, frameCount: frameCount, refs: [])
    }

    /// Full layout construction (text + fl2va keyframe rows + ref2va blocks +
    /// target audio + target video), mirroring initFull.
    public convenience init(textLen: UInt32, latentT: UInt32, latentH: UInt32, latentW: UInt32,
                audioT: UInt32, keyframes: [KeyframeAnchor], frameCount: UInt32, refs: [RefBlock]) {
        self.init(seqLen: 0, segments: [], positionIds: [], imgUpdate: [], audioUpdate: [], textTags: [],
                  textLen: textLen, latentT: latentT, latentH: latentH, latentW: latentW, audioT: audioT)
        let area = sqrt(Double(latentH) * Double(latentW))
        let hAxis = axisFromSqrtArea(dim: latentH, patch: H3Const.patchH, sqrtArea: area)
        let wAxis = axisFromSqrtArea(dim: latentW, patch: H3Const.patchW, sqrtArea: area)
        let frameRows = UInt32(hAxis.count * wAxis.count)

        let nCond = UInt32(keyframes.count)
        let nAudioRows = audioT * 2
        let nVideoRows = latentT * frameRows

        // Reference row budgets.
        var refImgRows: UInt32 = 0
        var refAudioRows: UInt32 = 0
        var refSegments: UInt32 = 0
        for b in refs {
            switch b.kind {
            case .image:
                refImgRows += refFrameRows(b)
                refSegments += 1
            case .audio:
                if b.audioT > 0 {
                    refAudioRows += b.audioT * 2
                    refSegments += 1
                }
            case .video:
                if b.audioT > 0 {
                    refAudioRows += b.audioT * 2
                    refSegments += 1
                }
                refImgRows += b.latentT * refFrameRows(b)
                refSegments += 1
            }
        }

        let seqLen = textLen + nCond * frameRows + refImgRows + refAudioRows + nAudioRows + nVideoRows
        var segments: [Segment] = []
        segments.reserveCapacity(Int(3 + nCond + refSegments))
        var positionIds = [Double](repeating: 0, count: Int(seqLen) * 3)
        var imgUpdate = [Bool](repeating: false, count: Int(nCond * frameRows + refImgRows + nVideoRows))
        var audioUpdate = [Bool](repeating: false, count: Int(refAudioRows + nAudioRows))

        var row: UInt32 = 0
        var imgRow = 0
        var audioRow = 0

        // text: t = 0..text_len-1, h = w = 0.
        segments.append(Segment(start: 0, end: textLen, kind: .text))
        for i in 0..<Int(textLen) { positionIds[(Int(row) + i) * 3] = Double(i) }
        row += textLen

        // Cursor shared by the target streams.
        var cursor = Double(textLen)

        // fl2va keyframe condition rows, sharing the target spatial grid.
        for anchor in keyframes {
            let condT: Double
            switch anchor {
            case .first:
                condT = cursor
            case .last:
                var spans: Double = 0
                for k in 0..<Int(latentT) { spans += videoTSpan(k) }
                condT = cursor + spans - H3Const.frameRescale
            }
            segments.append(Segment(start: row, end: row + frameRows, kind: .cond))
            writeFrameGrid(&positionIds, row: row, t: condT, hAxis: hAxis, wAxis: wAxis)
            for _ in 0..<Int(frameRows) {
                imgUpdate[imgRow] = false
                imgRow += 1
            }
            row += frameRows
        }
        _ = frameCount

        let targetWLow = wAxis[0]
        let targetWHigh = wAxis[wAxis.count - 1]

        // ref2va blocks, in request order; each advances the cursor.
        for b in refs {
            switch b.kind {
            case .image:
                let g = refGrid(b)
                segments.append(Segment(start: row, end: row + g.rows, kind: .refImg))
                writeFrameGrid(&positionIds, row: row, t: cursor, hAxis: g.hAxis, wAxis: g.wAxis)
                for _ in 0..<Int(g.rows) {
                    imgUpdate[imgRow] = false
                    imgRow += 1
                }
                row += g.rows
                cursor += 1.0
            case .audio:
                if b.audioT > 0 {
                    let n = b.audioT * 2
                    segments.append(Segment(start: row, end: row + n, kind: .refAudio))
                    writeAudioGrid(&positionIds, row: row, cursor: cursor, audioT: b.audioT,
                                   wLow: targetWLow, wHigh: targetWHigh)
                    for _ in 0..<Int(n) {
                        audioUpdate[audioRow] = false
                        audioRow += 1
                    }
                    row += n
                }
                cursor += Double(b.audioT)
            case .video:
                let g = refGrid(b)
                if b.audioT > 0 {
                    let n = b.audioT * 2
                    segments.append(Segment(start: row, end: row + n, kind: .refAudio))
                    // A soundtrack takes ITS OWN video's w extremes.
                    writeAudioGrid(&positionIds, row: row, cursor: cursor, audioT: b.audioT,
                                   wLow: g.wAxis[0], wHigh: g.wAxis[g.wAxis.count - 1])
                    for _ in 0..<Int(n) {
                        audioUpdate[audioRow] = false
                        audioRow += 1
                    }
                    row += n
                }
                let nRows = b.latentT * g.rows
                segments.append(Segment(start: row, end: row + nRows, kind: .refImg))
                let rtGrid = videoTGrid(b.latentT, origin: cursor)
                for (f, tv) in rtGrid.enumerated() {
                    writeFrameGrid(&positionIds, row: row + UInt32(f) * g.rows, t: tv, hAxis: g.hAxis, wAxis: g.wAxis)
                }
                for _ in 0..<Int(nRows) {
                    imgUpdate[imgRow] = false
                    imgRow += 1
                }
                row += nRows
                var spans: Double = 0
                for k in 0..<Int(b.latentT) { spans += videoTSpan(k) }
                cursor += max(Double(b.audioT), spans)
            }
        }

        // Target audio, then target video — always the last two segments.
        segments.append(Segment(start: row, end: row + nAudioRows, kind: .audio))
        writeAudioGrid(&positionIds, row: row, cursor: cursor, audioT: audioT,
                       wLow: targetWLow, wHigh: targetWHigh)
        for _ in 0..<Int(nAudioRows) {
            audioUpdate[audioRow] = true
            audioRow += 1
        }
        row += nAudioRows

        segments.append(Segment(start: row, end: row + nVideoRows, kind: .video))
        let tGrid = videoTGrid(latentT, origin: cursor)
        for (f, tv) in tGrid.enumerated() {
            writeFrameGrid(&positionIds, row: row + UInt32(f) * frameRows, t: tv, hAxis: hAxis, wAxis: wAxis)
        }
        for _ in 0..<Int(nVideoRows) {
            imgUpdate[imgRow] = true
            imgRow += 1
        }
        row += nVideoRows

        self.seqLen = seqLen
        self.segments = segments
        self.positionIds = positionIds
        self.imgUpdate = imgUpdate
        self.audioUpdate = audioUpdate
    }

    /// The target video segment — the final layer's slice (last by construction).
    public var videoSegment: Segment { segments[segments.count - 1] }
    /// The target audio segment — second to last by construction.
    public var audioSegment: Segment { segments[segments.count - 2] }
}

struct RefGrid {
    var hAxis: [Double]
    var wAxis: [Double]
    var rows: UInt32
}

func refGrid(_ b: RefBlock) -> RefGrid {
    let area = sqrt(Double(b.latentH) * Double(b.latentW))
    let h = axisFromSqrtArea(dim: b.latentH, patch: H3Const.patchH, sqrtArea: area)
    let w = axisFromSqrtArea(dim: b.latentW, patch: H3Const.patchW, sqrtArea: area)
    return RefGrid(hAxis: h, wAxis: w, rows: UInt32(h.count * w.count))
}

/// One latent frame's rows: t fixed, (h, w) the row-major meshgrid of the axes.
func writeFrameGrid(_ pos: inout [Double], row: UInt32, t: Double, hAxis: [Double], wAxis: [Double]) {
    var i = Int(row)
    for hv in hAxis {
        for wv in wAxis {
            pos[i * 3 + 0] = t
            pos[i * 3 + 1] = hv
            pos[i * 3 + 2] = wv
            i += 1
        }
    }
}

/// Stereo audio rows, CHANNEL-MAJOR: t advances per latent step, h stays 0,
/// w pinned to the spatial grid's two extremes.
func writeAudioGrid(_ pos: inout [Double], row: UInt32, cursor: Double, audioT: UInt32, wLow: Double, wHigh: Double) {
    for i in 0..<Int(audioT) {
        let t = cursor + Double(i)
        let a = (Int(row) + i) * 3
        pos[a + 0] = t
        pos[a + 1] = 0
        pos[a + 2] = wLow
        let b = (Int(row) + Int(audioT) + i) * 3
        pos[b + 0] = t
        pos[b + 1] = 0
        pos[b + 2] = wHigh
    }
}

// MARK: - RoPE tables

public struct RopeTables {
    public var cos: MLXArray   // [S, 48]
    public var sin: MLXArray
    public init(cos: MLXArray, sin: MLXArray) {
        self.cos = cos
        self.sin = sin
    }
}

/// Build cos/sin from the layout's [S,3] positions and the checkpoint's
/// inv_freq [16]. Angles = concat(t·inv, h·inv, w·inv) = 48 values; the
/// split-half pairing rotates dim i against dim i+48.
public func buildRope(layout: PackedLayout, invFreq: MLXArray, dtype: DType = PrecisionPolicy.defaultMainDType) -> RopeTables {
    let n = Int(layout.seqLen)
    let nf = invFreq.shape[0]
    let half = nf * 3

    let pos = MLXArray(layout.positionIds.map { Float($0) }, [n, 3, 1])
    let inv = invFreq.reshaped(1, 1, nf).asType(.float32)
    let perAxis = pos * inv                                   // [S,3,16]
    let ang = perAxis.reshaped(n, half)                       // [S,48]
    let cos = ang.cos()
    let sin = ang.sin()
    // Broadcast slot [1, S, 1, half] against q/k [1, S, H, hd].
    let c = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
    let s = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2).asType(dtype)
    return RopeTables(cos: c, sin: s)
}

// MARK: - Timestep plans

public struct ModRun {
    public var start: UInt32
    public var end: UInt32
    public var modRow: UInt32
    public init(start: UInt32, end: UInt32, modRow: UInt32) {
        self.start = start
        self.end = end
        self.modRow = modRow
    }
}

public let maxUniqueT = 4

public struct CondNoiseAug {
    public var visual: Double
    public var audio: Double
    public init(visual: Double = H3Const.visualCondTimestep, audio: Double = H3Const.audioCondTimestep) {
        self.visual = visual
        self.audio = audio
    }
}

/// The distinct timesteps a forward needs and the per-run modulation rows.
/// Row = t_row·3 + modality tag.
public struct TimestepPlan {
    public var uniqueT: [Double]
    public var runs: [ModRun]
    public init(uniqueT: [Double], runs: [ModRun]) {
        self.uniqueT = uniqueT
        self.runs = runs
    }
}

public func buildTimestepPlan(layout: PackedLayout, sigmaV: Double, shiftV: Double, shiftA: Double, aug: CondNoiseAug) -> TimestepPlan {
    let tV = 1.0 - sigmaV
    let tA = 1.0 - timeShiftSigma(sigmaV, fromShift: shiftV, toShift: shiftA)

    var uniq: [Double] = []
    func add(_ v: Double) {
        if uniq.contains(v) { return }
        uniq.append(v)
    }

    var hasVisCond = false
    var hasAudCond = false
    for seg in layout.segments {
        switch seg.kind {
        case .cond, .refImg: hasVisCond = true
        case .refAudio: hasAudCond = true
        default: break
        }
    }
    add(tV)
    add(tA)
    if hasVisCond { add(max(tV, aug.visual)) }
    if hasAudCond { add(max(tA, aug.audio)) }
    uniq.sort()

    func rowOf(_ v: Double) -> UInt32 {
        if let i = uniq.firstIndex(of: v) { return UInt32(i) }
        return 0
    }

    var runs: [ModRun] = []
    runs.reserveCapacity(layout.segments.count + layout.textTags.count)
    for seg in layout.segments {
        let t: Double
        switch seg.kind {
        case .text, .video: t = tV
        case .audio: t = tA
        case .cond, .refImg: t = max(tV, aug.visual)
        case .refAudio: t = max(tA, aug.audio)
        }
        let rowBase = rowOf(t) * 3
        if seg.kind == .text && layout.textTags.count == Int(seg.end - seg.start) {
            // Vision blocks interleave the text rows: resolve into tag runs.
            let tags = layout.textTags
            var runStart: UInt32 = 0
            var i: UInt32 = 1
            while i <= tags.count {
                if i == tags.count || tags[Int(i)] != tags[Int(runStart)] {
                    runs.append(ModRun(start: seg.start + runStart, end: seg.start + i,
                                       modRow: rowBase + UInt32(tags[Int(runStart)])))
                    runStart = i
                }
                i += 1
            }
            continue
        }
        runs.append(ModRun(start: seg.start, end: seg.end, modRow: rowBase + seg.kind.modalityTag))
    }
    return TimestepPlan(uniqueT: uniq, runs: runs)
}

/// Same plan but with every run's mod row remapped into a GLOBAL sorted
/// schedule table (the shape AdaLN precompute serves rows from).
public func buildTimestepPlanGlobal(layout: PackedLayout, sigmaV: Double, shiftV: Double, shiftA: Double,
                                    aug: CondNoiseAug, globalTs: [Double]) -> TimestepPlan {
    var p = buildTimestepPlan(layout: layout, sigmaV: sigmaV, shiftV: shiftV, shiftA: shiftA, aug: aug)
    for (i, r) in p.runs.enumerated() {
        let t = p.uniqueT[Int(r.modRow / 3)]
        let tag = r.modRow % 3
        guard let gi = globalTs.firstIndex(of: t) else {
            fatalError("timestep \(t) not in schedule")
        }
        p.runs[i].modRow = UInt32(gi) * 3 + tag
    }
    return p
}

/// The union of every step's unique timesteps, sorted ascending — the table
/// AdaLN precompute evaluates.
public func collectScheduleTs(layout: PackedLayout, sigmas: [Double], shiftV: Double, shiftA: Double, aug: CondNoiseAug) -> [Double] {
    var list: [Double] = []
    for sig in sigmas {
        let p = buildTimestepPlan(layout: layout, sigmaV: sig, shiftV: shiftV, shiftA: shiftA, aug: aug)
        for t in p.uniqueT where !list.contains(t) {
            list.append(t)
        }
    }
    return list.sorted()
}

// MARK: - Sparse attention policy

public enum SparseMode {
    case dense, spatial, temporal
}

public enum SparsePolicy {
    case off, mix, spatial, temporal

    public init(raw: String?) {
        guard let v = raw, !v.isEmpty, v != "0" else { self = .off; return }
        switch v {
        case "mix": self = .mix
        case "spatial": self = .spatial
        case "temporal": self = .temporal
        default: self = .off
        }
    }
}

/// Which pattern a layer runs under a policy. Anchor layers (first two,
/// middle two, last two) stay DENSE under "mix".
public func sparseModeForLayer(_ layer: Int, nLayers: Int, policy: SparsePolicy) -> SparseMode {
    switch policy {
    case .off: return .dense
    case .spatial: return .spatial
    case .temporal: return .temporal
    case .mix:
        let mid = nLayers / 2
        if layer < 2 || layer + 2 >= nLayers || layer == mid || layer == mid + 1 { return .dense }
        return (layer % 2 == 0) ? .spatial : .temporal
    }
}

/// Geometry the sparse core needs: the video segment is the LAST `t·f` rows.
public struct SparseSpec {
    public var mode: SparseMode
    public var t: UInt32
    public var f: UInt32
    public var videoStart: UInt32
    public init(mode: SparseMode, t: UInt32, f: UInt32, videoStart: UInt32) {
        self.mode = mode
        self.t = t
        self.f = f
        self.videoStart = videoStart
    }
}

// MARK: - Attention broadcast gating (PAB-style)
// H3AttnBroadcast 类与 attnBroadcastRefresh 等调度已抽至独立文件 H3AttnBroadcast.swift。
