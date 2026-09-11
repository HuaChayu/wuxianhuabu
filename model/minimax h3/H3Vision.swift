// H3Vision.swift
// MiniMax H3 (Hailuo 3.0) — vision presentation: the pure math between a
// reference pixel buffer and the Qwen3-VL sequence it becomes.
// Ported from mlx-serve-main/src/minimax_h3_vision.zig (entire module).
//
// H3 conditions on references by splicing vision blocks into the raw token
// stream (the tokenizer runs with NO special tokens):
//   fl2va:  "<Picture 1>: " <|vision_start|> …pad… <|vision_end|> <prompt>
//   ref2va: image -> "<Picture i>: " + block
//           audio -> "<Audio j>: "            (audio never enters Qwen)
//           video -> "<Video k>: " then per 2-frame pair "<T.T seconds>" + block
//
// Four load-bearing details (silent when wrong — the model conditions on
// garbage rather than erroring):
//   1. Resize policy: snap to multiple of patch·merge = 32 with HALF-TO-EVEN,
//      then push under the pixel cap / over the floor.
//   2. AdaLN tags: a vision span carries tag 0 (video) and widens by ONE on
//      each side to cover vision_start/vision_end; the mRoPE span does NOT.
//   3. mRoPE ids: the block collapses to ONE t, h/w run over the merged grid,
//      and text after it resumes at max(grid)/2 past the block start.
//   4. Qwen3-VL's INTERLEAVED mRoPE: T default, H at slots ≡1 (mod 3), W at
//      slots ≡2 (mod 3), both only below rope_dims[axis]·3.

import Foundation
import Darwin

// MARK: - Constants

public enum H3VisionConst {
    public static let visionStart: Int32 = 151652
    public static let visionEnd: Int32 = 151653

    public static let patch: UInt32 = 16
    public static let temporalPatch: UInt32 = 2
    public static let merge: UInt32 = 2
    public static let minPixels: UInt32 = 3136
    public static let maxPixels: UInt32 = 12845056

    /// Qwen3-VL normalizes to [-1, 1], not CLIP statistics.
    public static let imageMean: Float = 0.5
    public static let imageStd: Float = 0.5

    /// LM rope: theta and per-axis mRoPE section widths (T, H, W).
    public static let ropeTheta: Double = 5_000_000.0
    public static let ropeDims: [UInt32] = [24, 20, 20]

    /// Qwen3-VL-32B vision tower geometry.
    public static let vitHidden: Int = 1152
    public static let vitHeads: Int = 16
    public static let vitHeadDim: Int = 72
    public static let vitInter: Int = 4304
    public static let vitDepth: Int = 27
    public static let vitOut: Int = 5120
    public static let vitDeepstack: [Int] = [8, 16, 24]
    public static let vitGridSide: Int = 48 // sqrt(num_position_embeddings 2304)
}

// MARK: - Resize policy

public struct H3VisionCanvas {
    public var h: UInt32
    public var w: UInt32
    public init(h: UInt32, w: UInt32) {
        self.h = h
        self.w = w
    }
}

/// Qwen3-VL's `process_qwen2vl_images` sizing: snap to patch·merge = 32, then
/// scale under the pixel cap / over the floor (from ORIGINAL dims, floor/ceil).
public func fitCanvas(h: UInt32, w: UInt32) -> H3VisionCanvas {
    let factor = Double(H3VisionConst.patch * H3VisionConst.merge)
    let fh = Double(h)
    let fw = Double(w)
    var hBar = roundHalfEven(fh / factor) * factor
    var wBar = roundHalfEven(fw / factor) * factor

    let maxPx = Double(H3VisionConst.maxPixels)
    let minPx = Double(H3VisionConst.minPixels)
    if hBar * wBar > maxPx {
        let beta = sqrt((fh * fw) / maxPx)
        hBar = max(factor, floor(fh / beta / factor) * factor)
        wBar = max(factor, floor(fw / beta / factor) * factor)
    } else if hBar * wBar < minPx {
        let beta = sqrt(minPx / (fh * fw))
        hBar = ceil(fh * beta / factor) * factor
        wBar = ceil(fw * beta / factor) * factor
    }
    return H3VisionCanvas(h: UInt32(hBar), w: UInt32(wBar))
}

/// A vision block's patch grid. `t` is always 1 here: a still repeats itself
/// to fill the 2-frame patch, a video block is exactly one 2-frame pair.
public struct Grid {
    public var t: UInt32
    public var gh: UInt32
    public var gw: UInt32
    public init(t: UInt32, gh: UInt32, gw: UInt32) {
        self.t = t
        self.gh = gh
        self.gw = gw
    }
    /// Raw patches, i.e. rows entering the ViT.
    public var patches: UInt32 { t * gh * gw }
    /// Rows entering the LM after the merger's 2x2 spatial shuffle.
    public var mergedTokens: UInt32 { patches / (H3VisionConst.merge * H3VisionConst.merge) }
    public var mergedH: UInt32 { gh / H3VisionConst.merge }
    public var mergedW: UInt32 { gw / H3VisionConst.merge }
}

public func gridFor(h: UInt32, w: UInt32) -> Grid {
    let c = fitCanvas(h: h, w: w)
    return Grid(t: 1, gh: c.h / H3VisionConst.patch, gw: c.w / H3VisionConst.patch)
}

// MARK: - Spans, tags and positions

/// One vision block in the LM sequence. `index` is the first EXPANDED row
/// (one past its `<|vision_start|>`); `size` is `grid.mergedTokens()`.
public struct Span {
    public var index: UInt32
    public var size: UInt32
    public var grid: Grid
    public init(index: UInt32, size: UInt32, grid: Grid) {
        self.index = index
        self.size = size
        self.grid = grid
    }
    public var end: UInt32 { index + size }
}

/// AdaLN modality tag per LM position: 1 = text, 0 = video. The vision run
/// widens by ONE on each side to cover vision_start / vision_end.
public func tokenTags(seqLen: UInt32, spans: [Span]) -> [UInt8] {
    var out = [UInt8](repeating: 1, count: Int(seqLen))
    for sp in spans {
        let lo = (sp.index == 0) ? 0 : Int(sp.index) - 1
        let hi = min(Int(seqLen), Int(sp.end) + 1)
        if lo < hi {
            for i in lo..<hi { out[i] = 0 }
        }
    }
    return out
}

public struct TagRun {
    public var start: UInt32
    public var end: UInt32
    public var tag: UInt8
    public init(start: UInt32, end: UInt32, tag: UInt8) {
        self.start = start
        self.end = end
        self.tag = tag
    }
}

/// Maximal runs of equal tag — the DiT consumes runs, never a per-position
/// tag vector.
public func tagRuns(_ tags: [UInt8]) -> [TagRun] {
    guard !tags.isEmpty else { return [] }
    var runs: [TagRun] = []
    var runStart: UInt32 = 0
    var i: UInt32 = 1
    while i <= tags.count {
        if i == tags.count || tags[Int(i)] != tags[Int(runStart)] {
            runs.append(TagRun(start: runStart, end: i, tag: tags[Int(runStart)]))
            runStart = i
        }
        i += 1
    }
    return runs
}

/// [3·seq] mRoPE position ids, AXIS-MAJOR (`out[axis·seq + i]`), or nil when
/// there is no vision block (signal to use plain 1-D rope).
///
/// The whole block collapses to ONE t position; h/w run row-major over the
/// MERGED grid; text after a block resumes at `max(grid)/2` past the block's
/// start — so `offset` accumulates `len_max - size`.
public func mropePositions(seqLen: UInt32, spans: [Span]) -> [Double]? {
    guard !spans.isEmpty else { return nil }
    var out = [Double](repeating: 0, count: Int(seqLen) * 3)
    let sLen = Int(seqLen)

    var offset: Int64 = 0
    for (si, sp) in spans.enumerated() {
        let start = Int(sp.index)
        let end = Int(sp.end)
        if si == 0 {
            for i in 0..<start {
                let v = Double(i)
                for ax in 0..<3 { out[ax * sLen + i] = v }
            }
        }
        let lenMax = Int64(max(sp.grid.t, max(sp.grid.gh, sp.grid.gw)) / 2)
        let startNext = lenMax + Int64(start)
        var k = end
        while k < sLen {
            let v = Double(startNext + offset + Int64(k - end))
            for ax in 0..<3 { out[ax * sLen + k] = v }
            k += 1
        }

        let base = Double(Int64(start) + offset)
        let size = Int(sp.size)
        // t: one position for the whole block.
        for i in start..<end { out[0 * sLen + i] = base }
        // h: ceil(size / merged_h) consecutive repeats of each row index.
        let mh = Int(sp.grid.mergedH)
        if mh > 0 {
            let rep = (size + mh - 1) / mh
            for i in 0..<size {
                out[1 * sLen + start + i] = base + Double(i / rep)
            }
        }
        // w: the column indices cycling.
        let mw = Int(sp.grid.mergedW)
        if mw > 0 {
            for i in 0..<size {
                out[2 * sLen + start + i] = base + Double(i % mw)
            }
        }
        offset += lenMax - Int64(size)
    }
    return out
}

// MARK: - Interleaved mRoPE

/// Which position axis feeds frequency slot `j` of `head_dim/2`.
/// T is the default; H takes slots ≡1 (mod 3), W slots ≡2 (mod 3), both only
/// below rope_dims[axis]·3 — the tail past 3·min(...) stays T.
public func axisOfFreq(_ j: Int) -> Int {
    let m = j % 3
    if m == 1 && j < Int(H3VisionConst.ropeDims[1]) * 3 { return 1 }
    if m == 2 && j < Int(H3VisionConst.ropeDims[2]) * 3 { return 2 }
    return 0
}

/// Per-position rope ANGLES [seq·(head_dim/2)]. `positions` is the axis-major
/// table from `mropePositions`, or nil for the text-only case (plain 1-D).
public func ropeAngles(seqLen: UInt32, positions: [Double]?, headDim: Int, theta: Double) -> [Double] {
    let half = headDim / 2
    var out = [Double](repeating: 0, count: Int(seqLen) * half)
    for j in 0..<half {
        let exp = Double(2 * j) / Double(headDim)
        let inv = 1.0 / pow(theta, exp)
        let ax = (positions == nil) ? 0 : axisOfFreq(j)
        if let pp = positions {
            for i in 0..<Int(seqLen) {
                out[i * half + j] = pp[ax * Int(seqLen) + i] * inv
            }
        } else {
            for i in 0..<Int(seqLen) {
                out[i * half + j] = Double(i) * inv
            }
        }
    }
    return out
}

// MARK: - Presentation text

public enum RefKind {
    case image, audio, video
}

/// Label preceding a reference block. Ordinals are 1-based PER TYPE.
public func labelFor(kind: RefKind, ordinal: UInt32) -> String {
    switch kind {
    case .image: return "<Picture \(ordinal)>: "
    case .audio: return "<Audio \(ordinal)>: "
    case .video: return "<Video \(ordinal)>: "
    }
}

/// Timestamp label before each 2-frame video block, at the pair's midpoint.
/// One decimal — "<1.0 seconds>" is a different token sequence than "<1 seconds>".
public func timestampLabel(seconds: Double) -> String {
    String(format: "<%.1f seconds>", seconds)
}
