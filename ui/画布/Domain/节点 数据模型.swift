// ============================================================
//  文件作用：节点数据模型。NodeType 节点类型，CanvasNode 节点数据模型（含 Equatable），NodeContentHistory / VideoModel / ImageModel / ImageQuality / VideoQuality / VideoDuration 模型枚举，NodeConnection / DraggingConnection / AlignmentGuide / ConnectSide 连线模型。
//  互动文件：被 节点 ui.swift、节点 交互 鼠标.swift、画布状态 公共函数.swift 及画布其余文件引用；与 节点 几何.swift 同属 Domain 层。
// ============================================================

import SwiftUI
import Foundation

// MARK: - 节点类型

enum NodeType: String, CaseIterable, Identifiable, Codable {
    case character = "角色"
    case scene = "场景"
    case text = "文本"
    case image = "图片"
    case video = "视频"
    case audio = "音频"
    
    var id: String { rawValue }
    
    // 底部信息栏左侧小图标
    var icon: String {
        switch self {
        case .character: return "person"
        case .scene: return "photo.on.rectangle.angled"
        case .text: return "textformat"
        case .image: return "photo"
        case .video: return "video"
        case .audio: return "music.note"
        }
    }
    
    // 顶部占位区图标
    var placeholderIcon: String {
        switch self {
        case .character: return "person.crop.circle"
        case .scene: return "building.2"
        case .text: return "textformat"
        case .image: return "photo"
        case .video: return "video"
        case .audio: return "music.note"
        }
    }
    
    // 顶部独立文本标签（文本/图像/视频/音频）
    var topLabel: String {
        switch self {
        case .text: return "T 文本"
        case .image: return "图像"
        case .video: return "视频"
        case .audio: return "音频"
        default: return ""
        }
    }
}

// MARK: - 节点数据模型（画布坐标 position 为画布坐标系，显示时乘 zoom + offset）

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
    var quality: VideoQuality = .standard   // 视频节点清晰度档位（用户标签；默认标准档，Codable 自动 decodeIfPresent，旧档兼容）
    var imageModel: ImageModel = .hidreamO1 // 图像/角色/场景节点使用的生成模型（默认 HiDream-O1；Codable 自动 decodeIfPresent，旧档兼容）
    var imageQuality: ImageQuality = .p720  // 图像/角色/场景节点尺寸档位（默认 720 档；Codable 自动 decodeIfPresent，旧档兼容）
    /// 节点用过的历史内容（替换/生成新内容时旧内容入列，右下角徽标可展开恢复；Codable 自动 decodeIfPresent，旧档兼容）
    var history: [NodeContentHistory]? = nil
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
        lhs.quality == rhs.quality &&
        lhs.imageModel == rhs.imageModel &&
        lhs.imageQuality == rhs.imageQuality &&
        lhs.history == rhs.history
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
