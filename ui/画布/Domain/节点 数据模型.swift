// ============================================================
//  文件作用：节点数据模型。NodeType 节点类型，CanvasNode 节点数据模型（含 Equatable），NodeContentHistory / VideoModel / ImageModel / ImageQuality / VideoQuality / VideoDuration 模型枚举，NodeConnection / DraggingConnection / AlignmentGuide / ConnectSide 连线模型。
//  互动文件：被 节点 ui.swift、节点 交互 鼠标.swift、画布状态 公共函数.swift 及画布其余文件引用；与 节点 几何.swift 同属 Domain 层。
// ============================================================

import SwiftUI
import Foundation

// MARK: - 节点类型

enum NodeType: String, CaseIterable, Identifiable, Codable {
    case image_character = "角色"
    case image_scene = "场景"
    case text = "文本"
    case image_prop = "道具"
    case video = "视频"
    case audio = "音频"
    
    var id: String { rawValue }

    /// 旧存档兼容：「图片」已更名为「道具」，读旧项目 JSON 时把 "图片" 映射到 .image_prop，
    /// 其余未知值仍按原行为抛错（与自动合成 Codable 一致）。
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        if let type = NodeType(rawValue: raw) {
            self = type
        } else if raw == "图片" {
            self = .image_prop
        } else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath,
                      debugDescription: "未知节点类型：\(raw)")
            )
        }
    }
    
    // 底部信息栏左侧小图标
    var icon: String {
        switch self {
        case .image_character: return "person"
        case .image_scene: return "photo.on.rectangle.angled"
        case .text: return "textformat"
        case .image_prop: return "photo"
        case .video: return "video"
        case .audio: return "music.note"
        }
    }
    
    // 顶部占位区图标
    var placeholderIcon: String {
        switch self {
        case .image_character: return "person.crop.circle"
        case .image_scene: return "building.2"
        case .text: return "textformat"
        case .image_prop: return "photo"
        case .video: return "video"
        case .audio: return "music.note"
        }
    }
    
    // 顶部独立文本标签（角色/场景/道具/文本/视频/音频）
    var topLabel: String {
        switch self {
        case .text: return "T 文本"
        case .image_character: return "角色"
        case .image_scene: return "场景"
        case .image_prop: return "道具"
        case .video: return "视频"
        case .audio: return "音频"
        }
    }
}

// MARK: - 节点数据模型（画布坐标 position 为画布坐标系，显示时乘 zoom + offset）

/// 图片类节点（同一张图的不同用途分类）：角色 / 场景 / 道具 三者之间可右键互转。
/// 顺序即右键菜单展示顺序（角色 → 场景 → 道具），去掉当前类型后即为可转目标。
let imageNodeTypes: [NodeType] = [.image_character, .image_scene, .image_prop]

/// 「添加节点」菜单展示顺序（不依赖 case 声明顺序）：
/// 图片三类角色/场景/道具聚在一起，其后才是文本/视频/音频。
let addNodeMenuTypes: [NodeType] = [.image_character, .image_scene, .image_prop, .text, .video, .audio]

struct CanvasNode: Identifiable, Codable {
    var id: UUID = UUID()   // var：解码时从 JSON 恢复原 id，否则加载后连线 fromID/toID 匹配不上
    var type: NodeType
    var title: String
    var subtitle: String = ""
    var needsSupplement: Bool = false   // 是否显示橙色"待补充"标签
    var position: CGPoint = .zero       // 画布坐标
    var imageFileName: String? = nil    // 资产库图片文件名（资产库/下），nil = 空节点
    var mediaFileName: String? = nil    // 原始媒体文件名（视频/音频，资产库/下），nil = 无媒体内容
    var prompt: String = ""             // 节点自身提示词（引用资产的节点以资产库 prompt 为准，未引用资产时使用本字段）
    var groupID: UUID? = nil            // 打组标记：同组节点共享同一 id（nil = 未分组）
    var ratio: CanvasStore.Ratio? = nil // 视频节点私有比例（nil = 跟随画布全局比例；Codable 对 Optional 自动 decodeIfPresent，旧档兼容）
    var duration: VideoDuration? = nil  // 视频节点私有时长（nil = 跟随画布全局秒数；Codable 自动 decodeIfPresent，旧档兼容）
    var tailFrameEnabled: Bool = true   // 视频节点「尾帧」开关（on = 播放到尾帧停住；默认开，Codable 自动 decodeIfPresent，旧档兼容）
    var model: VideoModel = .ltx25Distill   // 视频节点使用的生成模型（默认 ltx2.5 蒸馏版；Codable 自动 decodeIfPresent，旧档兼容）
    var refMode: H3RefMode? = nil   // 视频节点 H3 参考模式（UI 下拉仅「多参考/首尾帧」两选项、必选其一；旧档无该字段时 decodeIfPresent 得 nil，队列端按多参考兜底）
    var quality: VideoQuality = .standard   // 视频节点清晰度档位（用户标签；默认标准档，Codable 自动 decodeIfPresent，旧档兼容）
    var imageModel: ImageModel = .hidreamO1 // 图像/角色/场景节点使用的生成模型（默认 HiDream-O1；Codable 自动 decodeIfPresent，旧档兼容）
    var imageQuality: ImageQuality = .p720  // 图像/角色/场景节点尺寸档位（默认 720 档；Codable 自动 decodeIfPresent，旧档兼容）
    /// 节点用过的历史内容（替换/生成新内容时旧内容入列，右下角徽标可展开恢复；Codable 自动 decodeIfPresent，旧档兼容）
    var history: [NodeContentHistory]? = nil
    /// 带 U+FFFC 占位符的提示词原文（prompt 字段仍是剥掉占位符的纯文本，供生成端使用；
    /// nil = 该节点提示词里没有引用块。Codable 自动 decodeIfPresent，旧档兼容）
    var promptRich: String? = nil
    /// 引用块持久化快照（与 promptRich 占位符顺序一一对应；Codable 自动 decodeIfPresent，旧档兼容）
    var promptMentionRefs: [PromptMentionRef]? = nil
}

/// CanvasNode 全字段 Equatable：供 NodeView/ConnectionLine 做 .equatable() 剪枝。
/// 滚动/平移画布只改 offset（渲染输入全部不变）→ 子视图 body 不再重新求值 → drawingGroup 位图保持、只做位图平移。
extension CanvasNode: Equatable {
    static func == (lhs: CanvasNode, rhs: CanvasNode) -> Bool {
        lhs.id == rhs.id &&
        lhs.type == rhs.type &&
        lhs.title == rhs.title &&
        lhs.subtitle == rhs.subtitle &&
        lhs.needsSupplement == rhs.needsSupplement &&
        lhs.position == rhs.position &&
        lhs.imageFileName == rhs.imageFileName &&
        lhs.mediaFileName == rhs.mediaFileName &&
        lhs.prompt == rhs.prompt &&
        lhs.groupID == rhs.groupID &&
        lhs.ratio == rhs.ratio &&
        lhs.duration == rhs.duration &&
        lhs.tailFrameEnabled == rhs.tailFrameEnabled &&
        lhs.model == rhs.model &&
        lhs.refMode == rhs.refMode &&
        lhs.quality == rhs.quality &&
        lhs.imageModel == rhs.imageModel &&
        lhs.imageQuality == rhs.imageQuality &&
        lhs.history == rhs.history &&
        lhs.promptRich == rhs.promptRich &&
        lhs.promptMentionRefs == rhs.promptMentionRefs
    }
}

/// 节点历史内容条目：只记录标题与资产引用（提示词不随历史存储，恢复时从源读）
struct NodeContentHistory: Codable, Identifiable, Equatable {
    var id = UUID()
    var title: String
    var imageFileName: String?
    var mediaFileName: String?
    var createdAt = Date()
}

/// 视频生成模型（未来扩展：不同模型判断标准不同，输入合规规则按模型分派）
enum VideoModel: String, Codable, CaseIterable {
    case ltx25Distill = "LTX-2.5 蒸馏版"
    case minimaxH3 = "MiniMax H3"
    
    var displayName: String { rawValue }
    
    /// 视频 latent 对齐倍率：LTX-2.5 要求宽高为 32 的倍数；MiniMax H3 同要求
    /// （H3Const.canvasMultiple=32，无空间升频器，生成分辨率须 32 对齐）。
    /// 未来模型各自定义，动态表自动跟随。
    var latentAlignMultiple: Int { 32 }
}

/// H3 参考注入模式（仅 MiniMax H3 视频节点生效；nil 时队列端默认多参考）
enum H3RefMode: String, Codable, CaseIterable {
    case multiRef = "多参考"    // ref2va 软参考：条件图按软参考块 + 视觉块注入（队列端 NA_H3_FL2VA_AS_REFS=1）
    case firstLast = "首尾帧"   // fl 首尾帧：条件图按首尾帧硬锚注入（队列端 NA_H3_FL2VA_AS_REFS=0）

    var displayName: String { rawValue }

    /// 队列端分流布尔：true = 走 ref2va 软参考传参（referenceImagePaths）；false = 走首尾帧传参（firstPath/lastPath）
    var fl2vaAsRefs: Bool { self == .multiRef }
}

/// 图像生成模型（图像/角色/场景节点共用；暂时只有 HiDream-O1，未来扩展）
enum ImageModel: String, Codable, CaseIterable {
    case hidreamO1 = "HiDream-O1"
    
    var displayName: String { rawValue }
}

/// 图像生成尺寸档位（用户标签；短边实际值 = 64 倍数对齐：480→512、720→768、1080→1088、2K→2048。
/// 64 倍数除2后正好 32 倍数，作视频首尾帧时零缩放。视频节点无 2K 档，仅图像类节点多出这一档。）
enum ImageQuality: String, Codable, CaseIterable {
    case p480 = "480"
    case p720 = "720"
    case p1080 = "1080"
    case p2k = "2K"
    
    var displayName: String { rawValue }
    
    /// 档位目标短边像素（64 倍数）
    var targetShortSide: Int {
        switch self {
        case .p480: return 512
        case .p720: return 768
        case .p1080: return 1088
        case .p2k: return 2048
        }
    }
    
    /// 编辑模式（有参考图输入）实际生效档位：仅 1080/2K 有效（1080p 2MP 为编辑稳定下限），
    /// 当前档位不在有效集合时自动落到第一个有效档位（1080）；非编辑模式返回自身。
    func resolved(hasValidImageInput: Bool) -> ImageQuality {
        guard hasValidImageInput else { return self }
        return (self == .p1080 || self == .p2k) ? self : .p1080
    }
}

/// 视频生成清晰度档位（用户标签；具体实际值 = 比例 × 档位查动态表，按 32 倍数适配，不钉死标准分辨率）
enum VideoQuality: String, Codable, CaseIterable {
    case standard = "标准"
    case p720 = "720p"
    case p1080 = "1080p"
    
    var displayName: String { rawValue }
}

/// 视频时长档位（5s / 10s / 15s；@24fps → 120 / 240 / 360 帧）
public enum VideoDuration: String, Codable, CaseIterable {
    case fiveSeconds = "5s"
    case tenSeconds = "10s"
    case fifteenSeconds = "15s"
    
    public static let `default` = VideoDuration.fiveSeconds
    
    public var displayName: String { rawValue }
    
    /// 时长秒数：5s → 5，10s → 10，15s → 15
    public var seconds: Int {
        switch self {
        case .fiveSeconds: return 5
        case .tenSeconds: return 10
        case .fifteenSeconds: return 15
        }
    }
    
    /// 对应全量帧数（24fps）：5s → 120，10s → 240，15s → 360
    public var numFrames: Int { seconds * 24 }
}

// MARK: - 连线数据模型

// 节点之间的连线
struct NodeConnection: Identifiable, Codable {
    var id: UUID = UUID()   // var：解码时从 JSON 恢复原 id
    let fromID: UUID
    let toID: UUID
}

// 提示词面板顶部的「前置条件缩略图」条目（由 CanvasStore.promptInputThumbnails 计算）
/// 打开面板时收集本节点所有入边源节点：有视觉内容的取资产缩略图（图片/视频首帧），
/// 无视觉内容的（文本/音频/空节点）退回类型图标；无效连线（超限/空节点）以半透明 + 红框提示。
struct PromptInputThumbnail: Identifiable {
    /// 源节点 id（同一源节点多线去重后仍唯一）
    let id: UUID
    /// 源节点标题（悬停提示用）
    let title: String
    /// 源节点类型（图标/角标用）
    let kind: NodeType
    /// 缩略图（nil = 无视觉内容，显示 kind.placeholderIcon）
    let image: NSImage?
    /// 该连线是否被输入合规性判定为无效（超限 / 空节点）
    let isInvalid: Bool
}

/// 输入框手打 @ 时的候选节点（@ 提及面板里的一行）
struct PromptMentionCandidate: Identifiable {
    /// 被引用节点 id
    let id: UUID
    /// 节点名字
    let title: String
    /// 节点类型（分类图标/角标）
    let kind: NodeType
    /// 缩略图（nil = 无可视内容，退回类型图标）
    let image: NSImage?
}

/// @ 提及面板里的一个分类（角色/场景/图片…）
struct PromptMentionGroup: Identifiable {
    var kind: NodeType
    var items: [PromptMentionCandidate]
    var id: NodeType { kind }
}

/// @ 提及面板里可被键盘高亮/切换的一行：候选节点 或 分类行
enum PromptMentionRow: Identifiable {
    /// 可插入的候选节点
    case candidate(PromptMentionCandidate)
    /// 分类行（→ / 回车进入，展开并在右栏逐项浏览）
    case group(PromptMentionGroup)

    var id: String {
        switch self {
        case .candidate(let item): return "c-\(item.id.uuidString)"
        case .group(let group): return "g-\(group.kind.rawValue)"
        }
    }

    /// 候选节点（分类行返回 nil）
    var candidate: PromptMentionCandidate? {
        if case .candidate(let item) = self { return item }
        return nil
    }

    /// 分类（候选行返回 nil）
    var group: PromptMentionGroup? {
        if case .group(let group) = self { return group }
        return nil
    }
}

/// @ 提及面板的数据源：上游连线节点优先 + 本项目全部节点（按类型分类展开）
struct PromptMentionCatalog {
    /// 本节点入边源节点（面板顶部「素材引用」区优先展示）
    var upstream: [PromptMentionCandidate] = []
    /// 本项目全部节点（不含本节点；面板下方按类型分组展开）
    var all: [PromptMentionCandidate] = []

    static let empty = PromptMentionCatalog()

    /// 按分类顺序分组，只保留有内容的分类
    var grouped: [PromptMentionGroup] {
        let order: [NodeType] = [.image_character, .image_scene, .image_prop, .video, .text, .audio]
        var result: [PromptMentionGroup] = []
        for kind in order {
            let items = all.filter { $0.kind == kind }
            if !items.isEmpty { result.append(PromptMentionGroup(kind: kind, items: items)) }
        }
        return result
    }

    /// 名字命中过滤词的候选（空过滤词 = 全量）：上游连线节点在前，随后本项目其它节点，按 id 去重
    func searchCandidates(keyword: String) -> [PromptMentionCandidate] {
        let trimmed = keyword.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        var seen = Set<UUID>()
        var result: [PromptMentionCandidate] = []
        for item in upstream + all {
            guard !seen.contains(item.id) else { continue }
            seen.insert(item.id)
            // 与面板里「命中段标强调色」用同一套匹配规则，保证筛出来的行一定带高亮段
            if item.title.range(of: trimmed, options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive]) != nil {
                result.append(item)
            }
        }
        return result
    }

    /// 键盘 ↑↓ 的遍历序列（一层层走：左栏候选 + 分类行 → 进入某分类后为右栏该分类的节点）：
    /// - 搜索模式（过滤词非空）= 过滤结果（不分层）
    /// - 已进入某分类（focusInCategory 且 enteredKind 非空）= 该分类下的节点（右栏）
    /// - 其余（浏览根层）= 「素材引用」候选 + 分类行，能一路走到图片/视频等类别
    func navRows(keyword: String, enteredKind: NodeType?, focusInCategory: Bool) -> [PromptMentionRow] {
        let trimmed = keyword.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            return searchCandidates(keyword: trimmed).map { PromptMentionRow.candidate($0) }
        }
        if focusInCategory, let enteredKind {
            let items = all.filter { $0.kind == enteredKind }
            return items.isEmpty ? [] : items.map { PromptMentionRow.candidate($0) }
        }
        var seen = Set<UUID>()
        var rows: [PromptMentionRow] = []
        for item in upstream where seen.insert(item.id).inserted {
            rows.append(.candidate(item))
        }
        for group in grouped {
            rows.append(.group(group))
        }
        return rows
    }
}

/// 输入框内的引用块（点 @ 前置条件缩略图后插入）：
/// 记录被引用节点的 id 与名字，界面渲染成"@ + 小图 + 名字"的块；
/// 后续生成/队列端可按 nodeID 取该节点的实际内容（图/视频首帧/文本），名字与 id 同时保留。
struct PromptMention: Identifiable, Equatable {
    /// 块自身 id（同一个被引用节点可以插入多个块，各自独立）
    let id: UUID
    /// 被引用节点 id（后续取内容用；同一节点可重复引用，不做去重）
    let nodeID: UUID
    /// 被引用节点名字（块上展示）
    let title: String
    /// 被引用节点类型（块左侧图标/色点用）
    let kind: NodeType
    /// 小头像（nil = 无可视内容，用类型图标）
    let image: NSImage?
}

/// 引用块的持久化快照（PromptMention 的可编码版本，不含 NSImage）：
/// 与 CanvasNode.promptRich 里的 U+FFFC 占位符按出现顺序一一对应，
/// App 重启后据此把块与文字混排原样还原（缩略图按 nodeID 现取）。
struct PromptMentionRef: Codable, Equatable {
    /// 块自身 id（保证同一节点多个块各自可区分）
    var blockID: UUID = UUID()
    /// 被引用节点 id
    var nodeID: UUID
    /// 被引用节点名字快照（节点已删除时兜底展示）
    var title: String
    /// 被引用节点类型快照
    var kind: NodeType
}

/// 节点输入框草稿缓存（按节点 id 存放，仅内存态；跨启动由 CanvasNode.promptRich/promptMentionRefs 落盘兜底）：
/// 节点 prompt 字段只存纯文本（占位符被剥掉），这里额外保留「带 U+FFFC 占位符的原文 + 引用块顺序」，
/// 用于切换选中 / 重新点击节点后用同一份原文把引用块原样恢复，块与文字的混排关系不丢。
struct NodePromptDraftCache {
    /// 带占位符的原文（占位符个数与 mentions 顺序一一对应）
    var text: String
    /// 引用块（每个块 id 唯一；title 为展示快照，被引用节点改名时按 nodeID 就地更新）
    var mentions: [PromptMention]
}

/// 节点改名事件（画布层 → 输入框层的单向通知）：
/// 只在改名成功处发布一条，输入框层按 id 精确匹配更新引用块名字，
/// 不做「全量节点遍历 + 逐节点比对」的轮询式同步（改名低频，但画布节点可能很多）。
struct NodeTitleChange: Equatable {
    /// 被改名的节点 id
    let nodeID: UUID
    /// 新标题
    let title: String
    /// 递增序号：同一标题连续改两次也能触发一次 UI 更新
    let seq: Int
}

// 正在拖拽的连线（from 节点 + 当前鼠标画布坐标 + 起始端口方向）
struct DraggingConnection {
    let fromID: UUID
    var currentPos: CGPoint   // 画布坐标
    var side: ConnectSide = .right
}

// 对齐参考线（拖动节点吸附时显示）
struct AlignmentGuide: Identifiable {
    let id = UUID()
    let isVertical: Bool      // true=垂直参考线（对齐 x），false=水平参考线（对齐 y）
    let position: CGFloat     // 屏幕坐标
}

// 加号端口方向：左侧 = 输入，右侧 = 输出
enum ConnectSide {
    case left, right
    
    /// 左侧为输入端口
    var isInput: Bool { self == .left }
    /// 右侧为输出端口
    var isOutput: Bool { self == .right }
    /// 相反方向
    var opposite: ConnectSide { self == .left ? .right : .left }
}
