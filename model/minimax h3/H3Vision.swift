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

/// Reference mark WITHOUT trailing separator, e.g. "<Picture 2>".
/// Used both as the vision-block prefix (labelFor) and as the inline prompt
/// reference resolved by `resolveReferences(inlineRef:)`, so the two always
/// carry byte-identical token sequences.
public func refMark(kind: RefKind, ordinal: UInt32) -> String {
    switch kind {
    case .image: return "<Picture \(ordinal)>"
    case .audio: return "<Audio \(ordinal)>"
    case .video: return "<Video \(ordinal)>"
    }
}

/// 正文内联指代：`<Subject n>`，与 subject_definitions 段同源（2026-10-27 方案 B）。
///
/// 视觉块头仍然只能用 `<Picture n>: `（pipeline 按该标签排视觉 token），
/// 但**用户正文里的 @ 占位**必须解析成 `<Subject n>`：定义段已经把
/// `<Subject n>` 绑定到 `<Picture n>`，正文沿同一条链指代，模型才知道
/// “喊话的是图 2 里那个人”。此前正文写 `<Picture n>` 时，指代链在正文处
/// 断裂（定义段讲 Subject，正文讲 Picture），角色身份落空，表现为
/// “不知道谁在说话 / 随便挑一张图套上去”。
public func subjectMark(ordinal: UInt32) -> String {
    "<Subject \(ordinal)>"
}

/// Label preceding a reference block. Ordinals are 1-based PER TYPE.
public func labelFor(kind: RefKind, ordinal: UInt32) -> String {
    refMark(kind: kind, ordinal: ordinal) + ": "
}

/// Timestamp label before each 2-frame video block, at the pair's midpoint.
/// One decimal — "<1.0 seconds>" is a different token sequence than "<1 seconds>".
public func timestampLabel(seconds: Double) -> String {
    String(format: "<%.1f seconds>", seconds)
}

// MARK: - REF2VA 六段结构（subject_definitions / summary / retention_analysis
//         / detailed_description / overall_soundscape / non_diegetic_music）

/// 参考图送模提示词：完整官方六段结构（顺序固定，缺段会影响参考图与音轨生效）。
///
/// 官方依据（MiniMaxAI/MiniMax-H3 `scripts/readme/reproducible-768p-ref2va-request.sh`
/// 的 prompt 字段 + `VIDEO_PROMPT_WRITING_GUIDE_base_en.md` §4.5 / ref_guide §2.1）：
/// `subject_definitions:` → `summary:` → `retention_analysis:` → `detailed_description:`
/// → `overall_soundscape:` → `non_diegetic_music:`
///
/// **不编造任何内容**：本函数只生成「结构与代号对应」这类确定性文本，
/// 所有具体外观/服装/发色/音色/环境音一律不填（没有来源就不写）。
/// 外观交给视觉块承载，由 `retention_analysis` 锁进画面。
/// `overall_soundscape` / `non_diegetic_music` 无来源时按官方允许的 `N/A` 占位，
/// 需要具体音效时由用户正文自带 `overall_soundscape:` 段（此时整段原样透传）。
/// **段名不可翻译**：`subject_definitions:` 等六个段名是模型的解析 key，保持官方英文；
/// 段内正文全部中文（仅语言由英文替换为中文，段落结构与顺序不变）。
///
/// **编号独立**：定义段只做 `<Subject n>` ↔ `<Picture n>` 的对应，不写 `(Sn)`；
/// `(Sx)` 只出现在第四段正文，按实际发声顺序编号（官方行 278）。
///
/// **类型来自引用本身**（2026-10-29）：`subjectKinds` 按下标（序号 - 1）给出每个条件图的
/// 主体类型中文名，由调用方从「引用块快照 / 连线源节点」的节点类型（角色 / 场景 / 道具）取，
/// 本函数不做任何视觉推断；取不到类型时统一写「主体」，不按位置猜。
public func ref2vaStructuredPrompt(userText: String,
                                   subjectCount: Int,
                                   subjectKinds: [String?] = []) -> String {
    guard subjectCount > 0 else { return userText }
    let body = userText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !body.isEmpty else { return userText }
    // 用户已按官方段名手写时整段透传，不重复加工。
    if body.contains("subject_definitions:") { return body }

    // ① subject_definitions —— <Subject n> ↔ <Picture n> 的对应关系 + 该图的主体类型。
    //   类型按引用来源（角色 / 场景 / 道具）逐个写，缺失时中性兜底「主体」。
    var defs = "subject_definitions:\n"
    for ord in 1...subjectCount {
        let kind = (ord - 1 < subjectKinds.count ? subjectKinds[ord - 1] : nil) ?? "主体"
        defs += "<Subject \(ord)> 是 <Picture \(ord)> 中的\(kind)。\n"
    }

    // ② summary —— 任务类型 + 素材数量，不描述情节（情节在④）。
    //let summary = "summary:\n[参考生成] 目标视频由 \(subjectCount) 张参考图生成，具体内容见下方详细描述。\n"

    // ③ retention_analysis —— 只锁「完全保留」这一条可验证声明，外观细节不编造。
    var ret = "retention_analysis:\n"
    for ord in 1...subjectCount {
        ret += "<Subject \(ord)>: fully_preserved —— 外观完全沿用 <Picture \(ord)>。\n"
    }

    // ④ detailed_description —— 用户正文逐字原样，不补任何镜头号；不做翻译、不改写。
    let detailed = "detailed_description:\n\(body)\n"

    // ⑤⑥ 音频两段 —— ⑤ 只下「与环境一致」的委托式要求，不写死具体声源（风雪/刀剑等
    //   只能从画面与正文来，硬编码就是编造）；⑥ 官方示例即 N/A，无配乐需求时保持 N/A。
    let sound = "overall_soundscape:\n"
        + "与画面场景一致的自然环境声，以及画面中可见动作产生的音效。\n"
    let music = "non_diegetic_music:\nN/A\n"

    return [defs, ret, detailed, sound, music].joined(separator: "\n")
}

/// 六段结构前置到用户正文前。
/// 用户已手写 `subject_definitions:` 时不重复插入。
public func prependRefSubjectSections(to prompt: String,
                                     subjectCount: Int,
                                     subjectKinds: [String?] = []) -> String {
    ref2vaStructuredPrompt(userText: prompt, subjectCount: subjectCount,
                           subjectKinds: subjectKinds)
}

// MARK: - FL2VA 三段结构（integrated_multimodal_description / overall_soundscape
//         / non_diegetic_music）

/// 首尾帧（FL2VA）送模提示词：官方三段结构。
///
/// 官方依据（MiniMaxAI/MiniMax-H3 `scripts/readme/handbook-minimax-h3.md`）：
/// FL2VA 与 Ref2VA 的 prompt 结构**不可互换**。FL2VA 只认
/// `integrated_multimodal_description:` → `overall_soundscape:` → `non_diegetic_music:`
/// 三段；Ref2VA 的 `subject_definitions / summary / retention_analysis /
/// detailed_description` 段名在 FL2VA 下不被识别，其下的正文会整段落入自由
/// 叙述区被当作朗读文本念出来（表现为 `<d>` 外的中文被朗读）。
///
/// 与 ref2vaStructuredPrompt 的差别：首尾帧没有参考主体，故不生成 subject 定义 /
/// retention_analysis，也不绑定 `<Picture n>`；正文逐字原样，不补镜头号。音频两段与
/// Ref2VA 同源（不硬编码声源），官方示例无配乐时即 `N/A`。
public func fl2vaStructuredPrompt(userText: String) -> String {
    let body = userText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !body.isEmpty else { return userText }
    // 用户已按官方段名手写时整段透传，不重复加工。
    if body.contains("integrated_multimodal_description:") { return body }

    // ① integrated_multimodal_description —— 用户正文逐字原样，不补任何镜头号。
    let desc = "integrated_multimodal_description:\n\(body)\n"

    // ②③ 音频两段 —— 与 Ref2VA 同源：只下「与环境一致」的委托式要求，不写死声源。
    let sound = "overall_soundscape:\n"
        + "Natural ambient sound of the scene, plus sound effects from the visible actions.\n"
    let music = "non_diegetic_music:\nN/A\n"

    return [desc, sound, music].joined(separator: "\n")
}

// MARK: - 台词拆封（引号 → `<d>[Language] …</d>` + 说话人 `(Sx)`）
//
// 官方依据（MiniMaxAI/MiniMax-H3 `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md`）：
//   §4.4 行 129 — 说话人识别语、ID、动作、语气写在 `<d>` **外**；`<d>` 内只放语言标签
//   和台词原文，逐字保留。
//   §4.4 行 132 — `The young woman (S1) says: <d>[English] …</d>`
//   §4.4 行 136 — 旁白必须显式写 `says in an off-screen voiceover`，并声明嘴唇不动。
//   §4.5 行 146 — 屏幕上可见的文字（招牌/字幕）才用英文双引号。
//
// 结论：**引号本身不是 H3 的台词语法**。中文 `“…”` 会被当作「画面可见文字」，
// 语音轨道于是把整段描述当连续朗读文本念出来（旁白）。必须把引号内的字换成 `<d>`。
//
// 拆封规则（确定性，无 LLM）：
//   1. 引号即边界 —— 不靠句号/感叹号切台词。引号配对成功即一条台词。
//   2. 句号/感叹号/问号只用来界定**归属窗口**：说话人取引号前最近一句里的最后一个
//      `<Subject n>`；前窗口没有则看引号后同一句（`… <Subject 2> 说道`）；都没有才回退
//      到上一个说话人，仍没有则不绑定（官方允许用中性 voice 描述）。
//   3. `(Sx)` 按**实际发声顺序**编号（官方行 278），不是出场顺序；同一主体全程复用同一 ID；
//      从头到尾不发声的角色不分配 ID（官方行 125），所以「只做动作不说话的角 2」天然不需要处理。
//   4. 引号内的嵌套引号（`‘…’`）“`” 一律剥掉，只留字。
public enum H3Dialogue {
    public struct Report {
        public let text: String
        public let lineCount: Int      // 识别出的台词条数
        public let speakerCount: Int   // 分配到的 (Sx) 个数
        public let unboundLines: Int   // 未能绑定说话人的台词条数
        public var summary: String {
            "台词 \(lineCount) 条｜说话人 \(speakerCount) 个｜未绑定 \(unboundLines) 条"
        }
    }

    private static let sentenceEnds: Set<Character> = ["。", "！", "？", "；", "…", "!", "?", ";", "\n"]
    private static let quoteChars: Set<Character> = ["“", "”", "「", "」", "『", "』", "‘", "’", "\""]
    private static let openers: [Character: Character] = ["“": "”", "「": "」", "『": "』", "‘": "’", "\"": "\""]

    public static func normalize(_ input: String) -> Report {
        // 用户已手写官方标记时不重复加工。
        if input.contains("<d>") {
            return Report(text: input, lineCount: 0, speakerCount: 0, unboundLines: 0)
        }
        let chars = Array(input)
        let spans = topLevelQuotes(chars)
        guard !spans.isEmpty else {
            return Report(text: input, lineCount: 0, speakerCount: 0, unboundLines: 0)
        }

        var inserts: [Int: String] = [:]
        var speakerIDs: [Int: String] = [:]     // Subject 序号 → Sx
        var nextSpeaker = 1
        var lastSpeaker: Int?
        var unbound = 0

        var previousSpanEnd = 0
        for span in spans {
            // 归属窗口：引号前，从「上一段台词结束 / 最近句末标点」二选一更靠后者起算。
            let windowStart = max(previousSpanEnd, sentenceStart(chars, before: span.open))
            let prevText = windowStart < span.open ? String(chars[windowStart..<span.open]) : ""
            let afterEnd = sentenceEnd(chars, after: span.close)
            let nextText = (span.close + 1) < afterEnd ? String(chars[(span.close + 1)..<afterEnd]) : ""

            let prevSubject = lastSubject(in: prevText).map { (ord: $0.ordinal, end: windowStart + $0.endOffset) }
            let nextSubject = firstSubject(in: nextText).map { (ord: $0.ordinal, end: span.close + 1 + $0.endOffset) }
            previousSpanEnd = span.close + 1

            var chosen: (ord: Int, insertAt: Int, needsLead: Bool)?
            if let prevSubject, prevSubject.end <= span.open {
                // 常规前置归属：`<Subject 2> 大喊：“…”` —— 主体已在台词前，只补 (Sx)。
                chosen = (prevSubject.ord, prevSubject.end, false)
            } else if let nextSubject {
                // 后置归属：`“…” <Subject 2>` —— 引号后紧跟主体，主体就是这句的说话人。
                // 只按官方语法补 (Sx) + says，不做任何中文词法判断。
                chosen = (nextSubject.ord, span.open, true)
            } else if let lastSpeaker {
                chosen = (lastSpeaker, span.open, true)   // 连续对白：沿用上一个说话人
            }

            if let chosen {
                let id: String
                if let existing = speakerIDs[chosen.ord] {
                    id = existing
                } else {
                    id = "S\(nextSpeaker)"
                    speakerIDs[chosen.ord] = id
                    nextSpeaker += 1
                }
                lastSpeaker = chosen.ord
                if chosen.needsLead {
                    inserts[chosen.insertAt, default: ""] +=
                        "\(subjectMark(ordinal: UInt32(chosen.ord))) (\(id)) says, "
                } else {
                    inserts[chosen.insertAt, default: ""] += " (\(id))"
                }
            } else {
                // ★ 2026-10-27：归属失败时不再插入 "A voice says, "。
                //   该短语为本地自造（社区实现无此写法），插在 <d> 正前方等于给模型划出一个
                //   更大的朗读窗口，导致 <d> 之外的叙述文字（如「角色1在颤抖中…」「不要字幕。」）
                //   被一并念出。归属失败时保持原文不动，只包 <d>。
                unbound += 1
            }
        }

        let spanByOpen = Dictionary(spans.map { ($0.open, $0) }, uniquingKeysWith: { a, _ in a })
        var out = ""
        out.reserveCapacity(chars.count + spans.count * 12)
        var i = 0
        while i < chars.count {
            if let ins = inserts[i] { out += ins }
            if let span = spanByOpen[i] {
                var body = ""
                var j = span.open + 1
                while j < span.close {
                    if !quoteChars.contains(chars[j]) { body.append(chars[j]) }
                    j += 1
                }
                body = tidy(body)
                out += "<d>[\(languageTag(of: body))] \(body)</d>"
                i = span.close + 1
                continue
            }
            out.append(chars[i])
            i += 1
        }
        if let ins = inserts[chars.count] { out += ins }

        return Report(text: out,
                      lineCount: spans.count,
                      speakerCount: speakerIDs.count,
                      unboundLines: unbound)
    }

    // MARK: 引号扫描

    /// 只取最外层引号对；嵌套引号（`“他说‘好’…”`）随内容一起剥掉。
    private static func topLevelQuotes(_ chars: [Character]) -> [(open: Int, close: Int)] {
        var result: [(open: Int, close: Int)] = []
        var stack: [(index: Int, closer: Character)] = []
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if let closer = openers[c] {
                // ASCII 双引号开闭同形：栈顶已是 " 时优先当闭合。
                if c == "\"", let top = stack.last, top.closer == "\"" {
                    let popped = stack.removeLast()
                    if stack.isEmpty { result.append((popped.index, i)) }
                } else {
                    stack.append((i, closer))
                }
            } else if let top = stack.last, c == top.closer {
                let popped = stack.removeLast()
                if stack.isEmpty { result.append((popped.index, i)) }
            }
            i += 1
        }
        return result.sorted { $0.open < $1.open }
    }

    // MARK: 窗口与主体

    private static func sentenceStart(_ chars: [Character], before index: Int) -> Int {
        var i = index - 1
        while i >= 0 {
            if sentenceEnds.contains(chars[i]) { return i + 1 }
            i -= 1
        }
        return 0
    }

    private static func sentenceEnd(_ chars: [Character], after index: Int) -> Int {
        var i = index + 1
        while i < chars.count {
            if sentenceEnds.contains(chars[i]) { return i }
            i += 1
        }
        return chars.count
    }

    /// 文本里最后一个 `<Subject n>`，返回序号与其在窗口内的结束偏移。
    private static func lastSubject(in text: String) -> (ordinal: Int, endOffset: Int)? {
        allSubjects(in: text).last
    }

    private static func firstSubject(in text: String) -> (ordinal: Int, endOffset: Int)? {
        allSubjects(in: text).first
    }

    private static func allSubjects(in text: String) -> [(ordinal: Int, endOffset: Int)] {
        let chars = Array(text)
        var found: [(Int, Int)] = []
        var i = 0
        while i < chars.count {
            if chars[i] == "<" {
                var j = i + 1
                while j < chars.count, chars[j] == " " { j += 1 }
                let keyword = Array("Subject")
                if j + keyword.count <= chars.count,
                   Array(chars[j..<(j + keyword.count)]) == keyword {
                    j += keyword.count
                    while j < chars.count, chars[j] == " " { j += 1 }
                    var digits = ""
                    while j < chars.count, chars[j].isNumber {
                        digits.append(chars[j])
                        j += 1
                    }
                    if !digits.isEmpty, j < chars.count, chars[j] == ">", let ord = Int(digits) {
                        found.append((ord, j + 1))
                        i = j
                    }
                }
            }
            i += 1
        }
        return found
    }

    // MARK: 文本工具

    /// 台词正字化：去掉装饰性重复标点与 emoji（官方行 274）。
    private static func tidy(_ s: String) -> String {
        var out = ""
        var lastPunct: Character?
        for c in s {
            if c == "～" || c == "~" { continue }
            if c.unicodeScalars.first?.properties.isEmojiPresentation == true { continue }
            if "！!?？。.".contains(c) {
                if let p = lastPunct, p == c { continue }
                lastPunct = c
            } else {
                lastPunct = nil
            }
            out.append(c)
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func languageTag(of s: String) -> String {
        var cjk = 0, latin = 0
        for u in s.unicodeScalars {
            if (0x4E00...0x9FFF).contains(u.value) { cjk += 1 }
            else if (0x41...0x7A).contains(u.value) { latin += 1 }
        }
        return cjk >= latin ? "Chinese" : "English"
    }
}
