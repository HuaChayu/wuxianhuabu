//
//  节点.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/14.
//
// ============================================================
//  文件作用：节点与连线的基础定义。包含 NodeType 节点类型枚举、CanvasNode /
//  NodeConnection / DraggingConnection / AlignmentGuide / ConnectSide 等数据模型、
//  NodeView 节点卡片视图、ConnectionLine / DraggingLine 连线视图，以及
//  connectionMidPoint / portPosition / hitTestPort 等全局辅助函数。
//  互动文件：被 交互.swift、画布 ui.swift、画布 状态.swift、画布 工具栏2.swift、
//  画布 菜单.swift、画布 资产面板.swift、画布 节点操作.swift 引用（提供类型与视图）。
// ============================================================

import SwiftUI
import AppKit
import AVFoundation
import Translation

// MARK: - 媒体时长缓存

/// 媒体文件时长缓存（按媒体文件名缓存秒数，避免节点渲染反复读盘解析）
private let mediaDurationCache = NSCache<NSString, NSNumber>()

/// 媒体文件时长（秒；仅音频/视频且有实际媒体内容时返回，无内容/读失败返回 nil）
func mediaDuration(for node: CanvasNode) -> TimeInterval? {
    guard node.type == .video || node.type == .audio,
          let fileName = mediaFileName(for: node) else { return nil }
    let key = fileName as NSString
    if let cached = mediaDurationCache.object(forKey: key) { return cached.doubleValue }
    let url = assetLibraryURL.appendingPathComponent(fileName)
    let duration: TimeInterval?
    if node.type == .video {
        let secs = AVAsset(url: url).duration.seconds
        duration = secs.isFinite && secs > 0 ? secs : nil
    } else {
        duration = (try? AVAudioPlayer(contentsOf: url))?.duration
    }
    if let d = duration {
        mediaDurationCache.setObject(NSNumber(value: d), forKey: key)
        return d
    }
    return nil
}

// MARK: - 资产图片缓存

/// 资产图片内存缓存（按文件名缓存解码后的 NSImage，避免移动画布等重绘场景反复从磁盘加载解码）
private let nodeImageCache = NSCache<NSString, NSImage>()

/// 从缓存加载资产图片：命中直接返回，未命中读盘解码后写入缓存。
/// 渲染侧 nodeImage 与占位高度计算共用，避免每次 body 重算都走磁盘 I/O。
func cachedAssetImage(for fileName: String) -> NSImage? {
    let key = fileName as NSString
    if let img = nodeImageCache.object(forKey: key) { return img }
    guard let img = NSImage(contentsOf: assetLibraryURL.appendingPathComponent(fileName)) else { return nil }
    nodeImageCache.setObject(img, forKey: key)
    return img
}

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
    
    var displayName: String { rawValue }
    
    /// 视频 latent 对齐倍率：LTX-2.5 要求宽高为 32 的倍数（未来模型各自定义，动态表自动跟随）
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

/// 视频时长档位（5s / 10s；@24fps → 120 / 240 帧）
public enum VideoDuration: String, Codable, CaseIterable {
    case fiveSeconds = "5s"
    case tenSeconds = "10s"
    
    public static let `default` = VideoDuration.fiveSeconds
    
    public var displayName: String { rawValue }
    
    /// 对应全量帧数（24fps）：5s → 120，10s → 240
    public var numFrames: Int { self == .tenSeconds ? 240 : 120 }
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

// MARK: - 加号端口尺寸常量（NodeView 渲染与 MouseControlNSView 命中检测共用）

/// 加号按钮尺寸
let nodePortSize: CGFloat = 24
/// 加号球容器半径（不可见命中区域，包裹加号）
let nodeBallRadius: CGFloat = 30
/// 组盒内边距（逻辑值，乘 zoom）：需容纳节点加号球完全落在盒内。
/// 节点加号球心在节点 frame 外 nodeBallRadius*zoom 处、视觉外沿约 42*zoom，故取 48*zoom 留余量。
/// 渲染（组盒 frame）、空白命中（hitTestGroupBlank）、组盒加号球心（groupPortBallCenter）三处统一引用本常量。
let groupBoxPadding: CGFloat = 48

// MARK: - 节点尺寸映射表（宽高比 + 缩放系数，不硬编码具体尺寸）

/// 节点尺寸规格：x:y 宽高比 + 缩放系数
struct NodeSizeSpec {
    var x: CGFloat           // 宽比例
    var y: CGFloat           // 高比例
    var scale: CGFloat = 1.0 // 缩放系数
}

/// 各类型节点尺寸规格（最大边 = nodeBaseWidth * scale，另一边按比例推导）
let nodeSizeSpecs: [NodeType: NodeSizeSpec] = [
    // 视频节点整体放大（空/有内容统一走本映射表：nodeWidthForType / placeholderHeightForType 共用，渲染与命中永远一致）
    .video:     NodeSizeSpec(x: 16, y: 9, scale: 1.35),
    .scene:     NodeSizeSpec(x: 16, y: 9),
    .image:     NodeSizeSpec(x: 16, y: 9),
    .character: NodeSizeSpec(x: 4, y: 5),
    .audio:     NodeSizeSpec(x: 16, y: 9, scale: 0.8),
    .text:      NodeSizeSpec(x: 16, y: 9, scale: 0.8),
]

/// 节点最大边基准宽度
let nodeBaseWidth: CGFloat = 180

/// 尺寸统一取 40 的倍数（网格间距），保证所有节点边缘落在网格周期内
func roundedToGrid(_ value: CGFloat, grid: CGFloat = 40) -> CGFloat {
    (value / grid).rounded() * grid
}

func roundedGridSize(_ size: CGSize) -> CGSize {
    CGSize(width: roundedToGrid(size.width), height: roundedToGrid(size.height))
}

// MARK: - 球中心（唯一入口）

/// 加号球中心（相对节点中心）：节点左右外侧（半宽 + ballRadius）处。
/// 渲染侧（connectButton 加号定位）与命中侧（portCircleCenter）共用，保证两处永远一致。
func ballCenterOffset(for size: CGSize, side: ConnectSide) -> CGPoint {
    let x: CGFloat = side == .left ? -(size.width / 2 + nodeBallRadius)
                                   :  (size.width / 2 + nodeBallRadius)
    return CGPoint(x: x, y: 0)
}

/// 组盒加号球中心（画布坐标，含 zoom；不含视口 offset）：
/// 组盒 frame（rect + groupBoxPadding*zoom padding）外沿再外移球半径处，与节点加号球心（节点边缘外 ballRadius）视觉一致。
/// 渲染侧（组盒加号）、命中侧（hitTestGroupPort）与临时连线起点三处共用，保证永远一致。
func groupPortBallCenter(rect: CGRect, side: ConnectSide, zoom: Double) -> CGPoint {
    let padding: CGFloat = groupBoxPadding * zoom
    let x = rect.midX + (side == .left ? -1 : 1) * (rect.width / 2 + padding + nodeBallRadius * zoom)
    return CGPoint(x: x, y: rect.midY)
}

// MARK: - 空节点占位区按钮区域（唯一入口）

/// 空节点占位区高度（映射表比例；渲染侧 NodeView.placeholderHeight 与命中侧 hitTestEmptyNodeButton 共用，保证两处永远一致）
func emptyPlaceholderHeight(for type: NodeType) -> CGFloat {
    let spec = nodeSizeSpecs[type] ?? NodeSizeSpec(x: 16, y: 9)
    let raw = spec.x >= spec.y
        ? nodeBaseWidth * spec.scale * spec.y / spec.x
        : nodeBaseWidth * spec.scale
    return roundedToGrid(raw)
}

/// 空节点占位区布局常量（渲染侧 VStack 与命中侧 emptyNodeButtonRect 共用，保证两处永远一致）
private let emptyPlaceholderIconSize: CGFloat = 28
private let emptyPlaceholderSpacing: CGFloat = 10

/// 空节点按钮组区域（相对节点中心，逻辑尺寸；仅命中侧使用）
/// 渲染侧为 VStack(图标 + 间距 + 按钮)，按钮组中心相对占位区中心 = (图标高 + 间距)/2，动态计算不硬编码
/// 按钮组宽度按实际按钮数量：有资产库按钮 = 两按钮总宽 140；无资产库（文本节点） = 单按钮 70 居中
func emptyNodeButtonRect(for size: CGSize, type: NodeType) -> CGRect {
    let ph = emptyPlaceholderHeight(for: type)
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : 18
    let buttonCenterOffset = (emptyPlaceholderIconSize + emptyPlaceholderSpacing) / 2
    let centerY = topLabelH + ph / 2 - size.height / 2 + buttonCenterOffset
    let width: CGFloat = type.allowedAssetCategories.isEmpty ? 70 : 140
    return CGRect(x: -width / 2, y: centerY - 16, width: width, height: 32)
}

// MARK: - 播放按钮区域（唯一入口）

/// 播放按钮尺寸（玻璃质感圆形按钮；渲染侧与命中侧共用）
let playButtonSize: CGFloat = 44
/// 播放按钮命中区域（略大于视觉尺寸，便于点击）
let playButtonHitSize: CGFloat = 48

// MARK: - 生成状态取消按钮区域（唯一入口）

/// 生成状态覆盖层距内容区左上角边距（渲染侧 VStack 与命中侧共用）
let generationStatusPadding: CGFloat = 8
/// 生成状态文字胶囊高度（caption 字号 + 上下 padding；渲染侧与命中侧共用）
let generationStatusTextHeight: CGFloat = 22
/// 生成状态文字与取消按钮间距（渲染侧 VStack spacing 与命中侧共用）
let generationStatusSpacing: CGFloat = 8
/// 取消按钮视觉/命中尺寸（渲染侧与命中侧共用）
let generationCancelButtonWidth: CGFloat = 52
let generationCancelButtonHeight: CGFloat = 24

/// 生成状态覆盖层"取消"按钮矩形（相对节点中心，逻辑尺寸；渲染侧与命中侧共用，保证两处永远一致）
/// 覆盖层整体贴内容区左上角（内容区顶 = topLabel 底 + padding），文字胶囊在上、取消按钮在下。
func generationCancelButtonRect(for size: CGSize, type: NodeType) -> CGRect {
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : nodeTopLabelHeight
    // 内容区左上角相对节点中心：(x: -size.width/2, y: topLabelH - size.height/2)
    let x = -size.width / 2 + generationStatusPadding
    let y = topLabelH - size.height / 2 + generationStatusPadding + generationStatusTextHeight + generationStatusSpacing
    return CGRect(x: x, y: y, width: generationCancelButtonWidth, height: generationCancelButtonHeight)
}

// MARK: - 历史徽章区域（唯一入口）

/// 历史徽章视觉尺寸（白底圆 icon，右下角数字徽章；渲染侧与命中侧共用）
let historyBadgeSize: CGFloat = 24
/// 历史徽章命中区域（略大于视觉尺寸，便于点击）
let historyBadgeHitSize: CGFloat = 32
/// 徽章距内容区右下角边距
let historyBadgePadding: CGFloat = 8

/// 历史徽章中心相对节点中心的偏移（渲染侧与命中侧共用，保证两处永远一致）
/// 位置：内容区（占位区）右下角向内 padding
/// 注意：x 用实际内容宽度而非 nodeWidthForType——视频有封面时渲染宽度走 videoNodeSize 自适应（横屏 440/竖屏 240），
/// 用空节点基准宽（240）会导致横屏视频命中区与渲染位置错位 100pt，点徽章无效
func historyBadgeCenterOffset(for size: CGSize, type: NodeType, imageFileName: String?) -> CGPoint {
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : nodeTopLabelHeight
    let ph = placeholderHeightForType(type, imageFileName: imageFileName)
    let centerY = topLabelH + ph / 2 - size.height / 2
    let contentW: CGFloat = (type == .video && imageFileName != nil)
        ? videoNodeSize(imageFileName: imageFileName).width
        : nodeWidthForType(type)
    let x = contentW / 2 - historyBadgePadding - historyBadgeSize / 2
    let y = centerY + ph / 2 - historyBadgePadding - historyBadgeSize / 2
    return CGPoint(x: x, y: y)
}

/// 节点顶部标签高度（Text(.caption2) + padding(.bottom, 4)，实测 17.0）
/// 渲染侧 / 命中侧 / 兜底估算共用，保证三处永远一致（硬编码 18/22 会导致占位区中心推算偏下）
let nodeTopLabelHeight: CGFloat = 17

/// 视频节点统一长边（横屏/竖屏视觉体量一致：长边都等于该值，另一边按视频真实宽高比推导）
/// 取 432 = 空视频节点长边(243) × 16/9，等价于把当前「竖屏视频(243×432)」作为满意基准，
/// 横屏视频放大到 432×243，横竖面积一致、观感协调；想整体放大/缩小改这一个数即可
let videoNodeLongSide: CGFloat = nodeBaseWidth * nodeSizeSpecs[.video]!.scale * 16 / 9

/// 视频节点尺寸：只区分横竖方向，统一映射到两个标准比例
/// 横屏(宽≥高) = 16:9(432×243)；竖屏(高>宽) = 9:16(243×432)，不随真实比例变化
/// 渲染侧 NodeView / 命中侧 / 兜底估算共用，保证三处永远一致
func videoNodeSize(imageFileName: String?) -> CGSize {
    let longSide = videoNodeLongSide
    if let fileName = imageFileName,
       let img = cachedAssetImage(for: fileName) {
        let w = img.size.width
        let h = img.size.height
        guard w > 0, h > 0 else { return roundedGridSize(CGSize(width: longSide, height: longSide * 9 / 16)) }
        if h <= w {   // 横屏：统一 16:9
            return roundedGridSize(CGSize(width: longSide, height: longSide * 9 / 16))
        } else {      // 竖屏：统一 9:16
            return roundedGridSize(CGSize(width: longSide * 9 / 16, height: longSide))
        }
    }
    return roundedGridSize(CGSize(width: longSide, height: longSide * 9 / 16))   // 空视频节点按 16:9
}

/// 节点展示尺寸统一入口（视频有图片 = 长边统一算法；其余走原映射表/图片比例逻辑）
/// 渲染侧 NodeView / 命中侧 / 兜底估算共用，保证三处永远一致
func nodeDisplaySize(for node: CanvasNode) -> CGSize {
    if node.type == .video, node.imageFileName != nil {
        return videoNodeSize(imageFileName: node.imageFileName)
    }
    return CGSize(width: nodeWidthForType(node.type),
                  height: placeholderHeightForType(node.type, imageFileName: node.imageFileName))
}

/// 节点宽度（有图片 = 空节点基准宽度，不随图片尺寸突然变大；空节点按映射表比例）
/// 渲染侧 NodeView 与命中侧共用，保证两处永远一致
func nodeWidthForType(_ type: NodeType) -> CGFloat {
    let spec = nodeSizeSpecs[type] ?? NodeSizeSpec(x: 16, y: 9)
    let raw = spec.x >= spec.y
        ? nodeBaseWidth * spec.scale
        : nodeBaseWidth * spec.scale * spec.x / spec.y
    return roundedToGrid(raw)
}

/// 占位区高度（有图片按图片比例自适应 = 节点宽 * 图高/图宽；空节点按映射表比例）
/// 视频节点有图片时走长边统一算法（videoNodeSize），横竖屏视觉体量一致
/// 渲染侧 NodeView 与命中侧共用，保证两处永远一致
func placeholderHeightForType(_ type: NodeType, imageFileName: String?) -> CGFloat {
    if type == .video, imageFileName != nil {
        return videoNodeSize(imageFileName: imageFileName).height
    }
    if let fileName = imageFileName,
       let img = cachedAssetImage(for: fileName) {
        let w = img.size.width
        let h = img.size.height
        guard w > 0 else { return emptyPlaceholderHeight(for: type) }
        return roundedToGrid(nodeWidthForType(type) * h / w)
    }
    return emptyPlaceholderHeight(for: type)
}

/// 播放按钮中心相对节点中心的偏移（渲染侧与命中侧共用，保证两处永远一致）
func playButtonCenterOffset(for size: CGSize, type: NodeType, imageFileName: String?) -> CGPoint {
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : nodeTopLabelHeight
    let ph = placeholderHeightForType(type, imageFileName: imageFileName)
    let y = topLabelH + ph / 2 - size.height / 2
    return CGPoint(x: 0, y: y)
}

/// 节点卡片完整高度（topLabel + 占位区 + 分割线 + 底部信息栏）
/// 节点尺寸未上报（nodeSizes 缺失）时兜底估算完整高度，避免输入框贴到占位区底边（上半部分）盖住信息栏
func nodeCardTotalHeight(for node: CanvasNode, nodeSizes: [UUID: CGSize]) -> CGFloat {
    if let s = nodeSizes[node.id] { return s.height }
    let topLabelH: CGFloat = node.type.topLabel.isEmpty ? 0 : nodeTopLabelHeight   // caption2(13) + 4 底部 padding
    let dividerH: CGFloat = 1
    // 视频有实际内容时信息栏三行（标题 + 尾帧开关行 + 副标题）：17 + 2 + 14 + 2 + 13 + paddingV 16 ≈ 64；其余节点两行 48
    let infoBarH: CGFloat = (node.type == .video && node.imageFileName != nil) ? 64 : 48
    return topLabelH + placeholderHeightForType(node.type, imageFileName: node.imageFileName) + dividerH + infoBarH
}

// MARK: - 尾帧开关命中区域（唯一入口）

/// 尾帧开关组合（「尾帧」文字 + on/off 开关）相对节点中心的命中区域（逻辑尺寸，含放大）。
/// 仅命中侧 hitTestTailFrameSwitch 使用；渲染侧为 SwiftUI 自动布局，行高/间距参数与此处保持一致：
/// 信息栏 padding(.vertical, 8) + 标题行(subheadline ≈17) + VStack spacing 2 + 尾帧行(开关 14 高)。
func tailFrameSwitchHitRect(for size: CGSize, type: NodeType, imageFileName: String?) -> CGRect {
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : nodeTopLabelHeight
    let ph = placeholderHeightForType(type, imageFileName: imageFileName)
    let dividerH: CGFloat = 1
    // 组合宽 = 「尾帧」caption2 两字估算宽 24 + spacing 6 + 开关 28；开关右缘贴信息栏右 padding(10)
    let textWidth: CGFloat = 24
    let spacing: CGFloat = 6
    let switchWidth: CGFloat = 28
    let comboWidth = textWidth + spacing + switchWidth
    let maxX = size.width / 2 - 10
    let minX = maxX - comboWidth
    // 尾帧行中心相对卡片顶：信息栏顶(8) + 标题行(17) + spacing(2) + 尾帧行半高(7)
    let titleRowH: CGFloat = 17
    let switchRowCenterFromTop = topLabelH + ph + dividerH + 8 + titleRowH + 2 + 7
    let centerY = switchRowCenterFromTop - size.height / 2
    // 命中放大：宽两侧各扩 4，高上下各扩 4
    return CGRect(x: minX - 4, y: centerY - 7 - 4, width: comboWidth + 8, height: 14 + 8)
}

// MARK: - 节点视图（圆角卡片：顶部占位区 + 底部信息栏）

struct NodeView: View {
    let node: CanvasNode
    /// 全局缩放（Command+滚轮）：节点卡片与加号整体随 zoom 缩放
    var zoom: Double = 1.0
    var isSelected: Bool = false
    var isHovered: Bool = false
    /// 当前悬停的空节点按钮动作（上传 / 资产库；nil 表示未悬停按钮，按钮不高亮）
    var hoveredEmptyButton: EmptyNodeButtonAction? = nil
    /// 节点主题色（加号 / 选中框 / 悬停高亮）
    var accentColor: Color = .blue
    /// 加号相对球中心的偏移（鼠标在球内吸附时跟随移动；nil 表示在球中心）
    var portOffset: CGPoint? = nil
    /// 当前吸附的加号方向（nil 表示在节点矩形内，加号居中；仅对应方向应用偏移）
    var portSide: ConnectSide? = nil
    /// 拖拽连线中强制显示的端口方向（目标节点自动亮出对应加号；nil 表示不强制）
    var showPortSide: ConnectSide? = nil
    /// 本节点正在拖拽连线（拖出端加号在拖拽期间保持显示）
    var isPortDragging: Bool = false
    /// 视频画面是否改由画布屏幕叠加层渲染（true 时本视图内不画 VideoPlayerView）
    /// 内容层套了 .drawingGroup() 离屏合帧，AVPlayerLayer 画面不参与 SwiftUI 离屏绘制（有声无画），
    /// 由 NodeCanvasView 屏幕叠加层按节点坐标补画真实视频画面。
    var videoSurfaceHidden: Bool = false
    var onSizeChange: ((CGSize) -> Void)?
    var onConnectDragChanged: ((UUID, CGPoint, CGSize, ConnectSide) -> Void)?
    var onConnectDragEnded: ((UUID, CGPoint, CGSize, ConnectSide) -> Void)?
    
    /// 正在拖拽连线（拖拽期间保持加号显示）
    @State private var isConnecting = false
    /// 节点实际尺寸（供加号定位）
    @State private var nodeSize: CGSize = .zero
    /// 媒体播放管理（观察播放状态刷新播放按钮）
    @ObservedObject private var playerManager = MediaPlayerManager.shared
    /// 生成队列（观察节点生成状态，内容区覆盖"正在生成/排队中"）
    @ObservedObject private var generationQueue = GenerationQueue.shared
    
    // 加号按钮尺寸（与命中检测共用常量）
    private let portSize = nodePortSize
    // 加号球容器半径（不可见命中区域，包裹加号；与命中检测共用常量）
    private let ballRadius = nodeBallRadius
    
    // 节点尺寸规格（按类型从映射表取，缺省回退 16:9；仅空节点使用）
    private var spec: NodeSizeSpec {
        nodeSizeSpecs[node.type] ?? NodeSizeSpec(x: 16, y: 9)
    }
    // 资产图片（从资产库缓存加载；nil = 空节点）
    private var nodeImage: NSImage? {
        guard let fileName = node.imageFileName else { return nil }
        return cachedAssetImage(for: fileName)
    }
    // 节点宽度（视频有图片走长边统一算法，横竖屏体量一致；其余 = 空节点基准宽度，不随图片尺寸突然变大）
    private var nodeWidth: CGFloat {
        if node.type == .video, node.imageFileName != nil {
            return videoNodeSize(imageFileName: node.imageFileName).width
        }
        return nodeWidthForType(node.type)
    }
    // 占位区高度（有图片按图片比例自适应 = 节点宽 * 图高/图宽；空节点按映射表比例，与命中侧共用）
    private var placeholderHeight: CGFloat {
        return placeholderHeightForType(node.type, imageFileName: node.imageFileName)
    }
    // 是否可播放（音频/视频且有实际媒体内容；空节点无内容不显示播放按钮，统一由 mediaFileName(for:) 判定）
    private var isPlayable: Bool {
        (node.type == .audio || node.type == .video) && mediaFileName(for: node) != nil
    }
    // 当前节点是否正在播放
    private var isNodePlaying: Bool {
        playerManager.isPlaying(nodeID: node.id)
    }

    /// 信息栏是否显示迷你进度滑块 + 尾帧按钮（仅视频有实际内容；渲染侧与命中侧条件一致）
    private var showInfoBarControls: Bool {
        node.type == .video && node.imageFileName != nil
    }
    
    var body: some View {
        // 节点卡片本体 + 加号层（overlay 默认对齐 card 中心，加号用 offset 相对节点中心定位，直接消费命中侧球心数据）
        // scaleEffect：全局缩放（Command+滚轮），节点卡片与加号整体随 zoom 缩放；锚点默认中心，与 .position 定位的中心一致
        card
            .frame(width: nodeWidth)
            .overlay {
                // 左侧加号（输入端口）：悬停 / 拖拽连线 / 作为连线目标时显示
                if isHovered || isConnecting || isPortDragging || showPortSide == .left {
                    connectButton(side: .left, size: nodeSize, isTarget: showPortSide == .left)
                        .transition(.scale.combined(with: .opacity))
                }
                // 右侧加号（输出端口）：悬停 / 拖拽连线 / 作为连线目标时显示
                if isHovered || isConnecting || isPortDragging || showPortSide == .right {
                    connectButton(side: .right, size: nodeSize, isTarget: showPortSide == .right)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .scaleEffect(zoom)
    }
    
    // 节点卡片
    private var card: some View {
        VStack(alignment: .leading, spacing: 0) {
            // 顶部独立文本标签（文本/图像/视频/音频）
            if !node.type.topLabel.isEmpty {
                Text(node.type.topLabel)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.leading, 2)
                    .padding(.bottom, 4)
            }
            
            // 节点卡片
            VStack(alignment: .leading, spacing: 0) {
                // 顶部占位区（有图片显示资产图片，空节点显示占位图标 + 上传/资产库按钮）
                ZStack {
                    // 占位区底色：音频节点用主题色梦幻淡染（对角浅渐变，随全局节点主题色换肤），
                    // 其他类型保持中性浅灰（内容为主，不给颜色）
                    Group {
                        if node.type == .audio {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(
                                    LinearGradient(
                                        colors: [accentColor.opacity(0.12), accentColor.opacity(0.04)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        } else {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.gray.opacity(0.08))
                        }
                    }
                    // 音频节点：无论空/有内容都走 SwiftUI 渲染（渐变底 + 乐符跟随全局主题色、矢量清晰），
                    // 不用静态占位图位图（不走换肤且图标发虚）；无内容时带 上传/资产库 按钮，有内容只留乐符
                    if node.type == .audio {
                        VStack(spacing: emptyPlaceholderSpacing) {
                            Image(systemName: node.type.placeholderIcon)
                                .font(.system(size: emptyPlaceholderIconSize))
                                .foregroundColor(accentColor.opacity(0.5))
                            if mediaFileName(for: node) == nil {
                                HStack(spacing: 10) {
                                    emptyNodeButton(title: "上传", icon: "arrow.up.doc", isHovered: hoveredEmptyButton == .upload) {
                                        // 点击由鼠标控制层命中按钮区域后回调处理（见 画布 交互 鼠标.swift）
                                    }
                                    if !node.type.allowedAssetCategories.isEmpty {
                                        emptyNodeButton(title: "资产库", icon: "folder", isHovered: hoveredEmptyButton == .library) {
                                            // 点击由鼠标控制层命中按钮区域后回调处理（见 画布 交互 鼠标.swift）
                                        }
                                    }
                                }
                            }
                        }
                    } else if let img = nodeImage {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: nodeWidth, height: placeholderHeight)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    } else {
                        VStack(spacing: emptyPlaceholderSpacing) {
                            Image(systemName: node.type.placeholderIcon)
                                .font(.system(size: emptyPlaceholderIconSize))
                                .foregroundColor(node.type == .audio ? accentColor.opacity(0.5) : .gray.opacity(0.5))
                            HStack(spacing: 10) {
                                emptyNodeButton(title: "上传", icon: "arrow.up.doc", isHovered: hoveredEmptyButton == .upload) {
                                    // 点击由鼠标控制层命中按钮区域后回调处理（见 画布 交互 鼠标.swift）
                                }
                                if !node.type.allowedAssetCategories.isEmpty {
                                    emptyNodeButton(title: "资产库", icon: "folder", isHovered: hoveredEmptyButton == .library) {
                                        // 点击由鼠标控制层命中按钮区域后回调处理（见 画布 交互 鼠标.swift）
                                    }
                                }
                            }
                        }
                    }
                    // 视频播放中：显示视频画面（覆盖缩略图）
                    // videoSurfaceHidden=true 时不在此层渲染：内容层 drawingGroup 离屏合帧会吞掉 AVPlayerLayer 画面，
                    // 真实画面由 NodeCanvasView 屏幕叠加层按节点坐标补画（见 画布&节点 交互 连线.swift）
                    if isNodePlaying, node.type == .video, !videoSurfaceHidden, let player = playerManager.videoPlayer {
                        VideoPlayerView(player: player)
                            .frame(width: nodeWidth, height: placeholderHeight)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    // 播放按钮（音频/视频悬停显示；播放中不常驻，避免遮挡视频画面）
                    // 显示直接由 isHovered 驱动：悬停即现，鼠标离开即隐；播放中再悬停回来显示停止按钮
                    if isPlayable {
                        playButton
                            .opacity(isHovered ? 1 : 0)
                            .allowsHitTesting(isHovered)
                    }
                    // 播放中的底部控件（不遮挡播放按钮，按钮在占位区中央）：
                    // ① 音频波形：音频播放中常显（真 PCM 波形，随播放进度滚动）
                    // ② 播放进度条：播放中且鼠标悬停内容区才出现；拖动由鼠标控制层命中处理（此处仅渲染外观）
                    VStack {
                        Spacer()
                        if isNodePlaying, node.type == .audio {
                            WaveformView(accentColor: accentColor)
                                .padding(.bottom, 14)
                        }
                        if isNodePlaying {
                            MediaProgressBarView(accentColor: accentColor)
                                .opacity(isHovered ? 1 : 0)
                                .padding(.horizontal, 10)
                                .padding(.bottom, 10)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .allowsHitTesting(false)   // 进度条点击/拖动统一由鼠标控制层命中（画布 交互 鼠标.swift hitTestProgressBar）

                    // 生成状态覆盖层：该节点有任务在生成/排队时显示在内容区左上角，提示进度 + 提供取消；
                    // 取消后队列移除该任务，此处自动消失恢复实际内容
                    // 仅"取消"按钮可命中（点击由鼠标控制层处理），文字胶囊与空白区不拦截点击（穿透到鼠标控制层）
                    if let qs = generationQueue.queueStatus(for: node.id) {
                        VStack(spacing: generationStatusSpacing) {
                            Group {
                                if qs == .running {
                                    // 生成中：省略号 1→2→3 循环动画（每 0.5s 一拍；仅 running 时存在，
                                    // 覆盖层消失即销毁，无 timer 泄漏）；"排队中"保持静态
                                    TimelineView(.periodic(from: .now, by: 0.5)) { timeline in
                                        let dotCount = Int(timeline.date.timeIntervalSinceReferenceDate / 0.5) % 3 + 1
                                        Text("正在生成" + String(repeating: "·", count: dotCount))
                                            .font(.caption)
                                            .fontWeight(.medium)
                                            .foregroundColor(accentColor)
                                    }
                                } else {
                                    Text("排队中…")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundColor(accentColor)
                                }
                            }
                            .frame(height: generationStatusTextHeight)
                            .padding(.horizontal, 10)
                            .background(
                                Capsule()
                                    .fill(Color.black.opacity(0.45))
                            )
                            .allowsHitTesting(false)
                            Button {
                                if let taskID = generationQueue.taskID(for: node.id) {
                                    generationQueue.cancel(taskID)
                                }
                            } label: {
                                Text("取消")
                                    .font(.caption)
                                    .foregroundColor(.red)
                                    .frame(width: generationCancelButtonWidth, height: generationCancelButtonHeight)
                                    .background(
                                        Capsule()
                                            .fill(Color.black.opacity(0.6))
                                    )
                            }
                            .buttonStyle(.plain)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding(generationStatusPadding)
                    }

                    // 历史徽章：有往期内容的节点右下角；数字 = 往期条数
                    // 仅当前内容（无往期）时不显示；点击展开该节点往期内容面板（命中由鼠标控制层处理）
                    if let history = node.history, !history.isEmpty {
                        historyBadge
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                            .padding(historyBadgePadding)
                    }
                }
                .frame(height: placeholderHeight)
                .frame(maxWidth: .infinity)
                
                Divider()
                
                // 底部信息栏
                HStack(spacing: 8) {
                    Image(systemName: node.type.icon)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        // 标题行：所有节点都显示（视频有内容时下方尾帧行仍在，三行布局）
                        HStack(spacing: 4) {
                            Text(node.title)
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .lineLimit(1)
                        }
                        // 视频有实际内容：秒数上方的「尾帧」on/off 开关行（文字在开关前面，右对齐紧贴；点击由鼠标控制层命中）
                        if showInfoBarControls {
                            HStack(spacing: 6) {
                                Spacer(minLength: 0)
                                Text("尾帧")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .fixedSize()
                                // on/off 开关：off = 灰轨道 + 手柄在左；on = 主题色轨道 + 手柄在右（清晰反馈）
                                ZStack(alignment: node.tailFrameEnabled ? .trailing : .leading) {
                                    Capsule()
                                        .fill(node.tailFrameEnabled ? accentColor : Color.black.opacity(0.15))
                                    Circle()
                                        .fill(Color.white)
                                        .frame(width: 10)
                                        .shadow(color: .black.opacity(0.3), radius: 1, y: 0.5)
                                        .padding(2)
                                }
                                .frame(width: 28, height: 14)
                                .overlay(
                                    Capsule()
                                        .stroke(node.tailFrameEnabled ? accentColor.opacity(0.4) : Color.clear, lineWidth: 1)
                                )
                                .allowsHitTesting(false)
                                .animation(.easeOut(duration: 0.15), value: node.tailFrameEnabled)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        // 副标题行 + 媒体时长（右下角，仅音频/视频有实际内容时显示，四舍五入取整秒）
                        if !node.subtitle.isEmpty || mediaDuration(for: node) != nil {
                            HStack(spacing: 4) {
                                if !node.subtitle.isEmpty {
                                    Text(node.subtitle)
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .lineLimit(1)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                } else {
                                    Spacer(minLength: 0)
                                }
                                if let duration = mediaDuration(for: node) {
                                    Text("\(Int(duration.rounded()))s")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .monospacedDigit()
                                }
                            }
                        }
                    }
                    
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                // 悬停命中：下半部分（底部信息栏）变主题色
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(isHovered ? accentColor.opacity(0.18) : Color.clear)
                )
            }
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color.white.opacity(0.9))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(isSelected ? accentColor : (isHovered ? accentColor.opacity(0.8) : Color.gray.opacity(0.25)),
                                    lineWidth: isSelected ? 2 : (isHovered ? 2 : 1))
                    )
            )
            .shadow(color: isHovered ? accentColor.opacity(0.25) : .clear, radius: 8)
        }
        .background(
            // 只测量卡片本体实际尺寸（不含加号球等外部元素），用于命中检测与加号定位
            GeometryReader { geo in
                Color.clear
                    .onAppear {
                        nodeSize = geo.size
                        onSizeChange?(geo.size)
                    }
                    .onChange(of: geo.size) { _, newSize in
                        nodeSize = newSize
                        onSizeChange?(newSize)
                    }
            }
        )
    }
    
    // 空节点占位区按钮（悬浮无底色，图标 + 文字浅灰，小号；鼠标命中按钮时高亮：文字加深 + 主题色浅背景）
    private func emptyNodeButton(title: String, icon: String, isHovered: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 10))
                Text(title)
                    .font(.system(size: 11))
            }
            .foregroundColor(isHovered ? Color(red: 0.2, green: 0.2, blue: 0.24) : .gray.opacity(0.6))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(isHovered ? accentColor.opacity(0.15) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
    
    // 玻璃质感播放/停止按钮（音频/视频节点悬停显示；点击切换播放/停止）
    // 位于占位区正中央（ZStack 默认居中）；点击命中由鼠标控制层处理（见 画布 交互 鼠标.swift）
    // 渲染样式与资产库格子/属性预览共用 MediaPlayButton（fun/媒体播放.swift）
    private var playButton: some View {
        MediaPlayButton(isPlaying: isNodePlaying) {
            playerManager.toggle(node: node)
        }
    }

    // 历史徽章：白底圆 icon + 右下角数字徽章（数字 = 当前内容 + 往期条数）
    // 点击展开该节点往期内容面板；命中由鼠标控制层处理（见 画布 交互 鼠标.swift hitTestHistoryBadge）
    private var historyBadge: some View {
        ZStack(alignment: .bottomTrailing) {
            // 白底圆 icon（主题色历史图标）
            Circle()
                .fill(Color.white)
                .frame(width: historyBadgeSize, height: historyBadgeSize)
                .overlay(Circle().stroke(accentColor.opacity(0.35), lineWidth: 1))
                .shadow(color: .black.opacity(0.12), radius: 2, y: 0.5)
            Image(systemName: "clock.arrow.circlepath")
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(accentColor)
                .frame(width: historyBadgeSize, height: historyBadgeSize)
            // 数字徽章：往期条数
            Text("\(node.history?.count ?? 0)")
                .font(.system(size: 9, weight: .bold))
                .foregroundColor(.white)
                .frame(minWidth: 14, minHeight: 14)
                .background(Circle().fill(accentColor))
                .offset(x: 3, y: -3)
                .shadow(color: .black.opacity(0.2), radius: 1, y: 0.5)
        }
    }
    
    // 加号按钮（悬停时出现在节点左右两侧的球容器内，仅用于拖拽连线）
    // 加号默认在球中心；鼠标进入球内时跟随鼠标吸附移动
    // isTarget：拖拽连线中本端口为可释放目标时高亮
    private func connectButton(side: ConnectSide, size: CGSize, isTarget: Bool = false) -> some View {
        // 球心相对节点中心：直接消费命中侧同款 ballCenterOffset（与 portCircleCenter 同一来源），渲染侧零独立计算
        let ballCenter = ballCenterOffset(for: size, side: side)
        // 加号相对球心的吸附偏移（仅当前吸附方向应用，左右独立）：直接消费命中侧算好的 portOffset
        let offset = (portSide == side) ? (portOffset ?? .zero) : .zero
        return Button(action: {}) {
            Image(systemName: "plus")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(isTarget ? .white : accentColor)
                .frame(width: portSize, height: portSize)
                .background(Circle().fill(isTarget ? accentColor : Color.white))
                .overlay(Circle().stroke(isTarget ? accentColor : accentColor.opacity(0.6), lineWidth: 1.5))
                .shadow(color: isTarget ? accentColor.opacity(0.5) : .black.opacity(0.15), radius: isTarget ? 6 : 3)
                .scaleEffect(isTarget ? 1.15 : 1)
        }
        .buttonStyle(.plain)
        // overlay 默认对齐 card 中心，offset 基准即节点中心：加号 = 节点中心 + 球心偏移 + 吸附偏移
        .offset(x: ballCenter.x + offset.x, y: ballCenter.y + offset.y)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    isConnecting = true
                    // 鼠标相对节点中心的偏移（供画布层换算屏幕坐标）：
                    // 加号中心相对节点中心(球心+吸附) + 鼠标相对按钮位置 - 按钮半宽；整体乘 zoom（scaleEffect 缩放后坐标）
                    let localPos = CGPoint(x: (ballCenter.x + offset.x) * zoom + value.location.x - portSize * zoom / 2,
                                           y: (ballCenter.y + offset.y) * zoom + value.location.y - portSize * zoom / 2)
                    onConnectDragChanged?(node.id, localPos, size, side)
                }
                .onEnded { value in
                    isConnecting = false
                    let localPos = CGPoint(x: (ballCenter.x + offset.x) * zoom + value.location.x - portSize * zoom / 2,
                                           y: (ballCenter.y + offset.y) * zoom + value.location.y - portSize * zoom / 2)
                    onConnectDragEnded?(node.id, localPos, size, side)
                }
        )
    }
}

/// NodeView Equatable：滚动/平移画布时（node/zoom/选中/悬停等渲染输入全部不变）剪枝 body，避免全量重建与位图重采样。
/// 闭包（onSizeChange 等）与 @State/@ObservedObject 不参与比较：闭包行为稳定，状态由 SwiftUI 独立跟踪。
extension NodeView: Equatable {
    static func == (lhs: NodeView, rhs: NodeView) -> Bool {
        // node 按渲染输入字段逐项比较（②比较瘦身）：history 只比条目数（恢复/追加才变，逐元素比较纯浪费）；
        // prompt 不在 NodeView 内渲染，不比。其余字段均影响卡片渲染。
        lhs.node.id == rhs.node.id &&
        lhs.node.type == rhs.node.type &&
        lhs.node.title == rhs.node.title &&
        lhs.node.subtitle == rhs.node.subtitle &&
        lhs.node.needsSupplement == rhs.node.needsSupplement &&
        lhs.node.position == rhs.node.position &&
        lhs.node.imageFileName == rhs.node.imageFileName &&
        lhs.node.mediaFileName == rhs.node.mediaFileName &&
        lhs.node.groupID == rhs.node.groupID &&
        lhs.node.ratio == rhs.node.ratio &&
        lhs.node.duration == rhs.node.duration &&
        lhs.node.tailFrameEnabled == rhs.node.tailFrameEnabled &&
        lhs.node.model == rhs.node.model &&
        lhs.node.quality == rhs.node.quality &&
        lhs.node.imageModel == rhs.node.imageModel &&
        lhs.node.imageQuality == rhs.node.imageQuality &&
        (lhs.node.history?.count ?? 0) == (rhs.node.history?.count ?? 0) &&
        lhs.zoom == rhs.zoom &&
        lhs.isSelected == rhs.isSelected &&
        lhs.isHovered == rhs.isHovered &&
        lhs.hoveredEmptyButton == rhs.hoveredEmptyButton &&
        lhs.accentColor == rhs.accentColor &&
        lhs.portOffset == rhs.portOffset &&
        lhs.portSide == rhs.portSide &&
        lhs.showPortSide == rhs.showPortSide &&
        lhs.isPortDragging == rhs.isPortDragging &&
        lhs.videoSurfaceHidden == rhs.videoSurfaceHidden
    }
}

// MARK: - 连线视图（画布坐标转屏幕坐标，随画布平移缩放）

// 画布视图动画（缩放适配等 offset/zoom 变化统一使用，保证节点/网格/连线同步过渡不割裂）
let canvasViewAnimation: Animation = .easeInOut(duration: 0.3)

// 连线贝塞尔曲线（Animatable：缩放/平移画布时端点插值，连线平滑过渡不跳变）
// nonisolated：项目默认主 actor 隔离，Shape/Animatable 的 conformance 要求 Sendable，需显式非隔离
nonisolated struct ConnectionCurve: Shape, Animatable {
    var start: CGPoint
    var end: CGPoint

    var animatableData: AnimatablePair<CGPoint.AnimatableData, CGPoint.AnimatableData> {
        get { AnimatablePair(start.animatableData, end.animatableData) }
        set {
            start.animatableData = newValue.first
            end.animatableData = newValue.second
        }
    }

    func path(in rect: CGRect) -> Path {
        let midX = (start.x + end.x) / 2
        var path = Path()
        path.move(to: start)
        path.addCurve(to: end,
                      control1: CGPoint(x: midX, y: start.y),
                      control2: CGPoint(x: midX, y: end.y))
        return path
    }
}

struct ConnectionLine: View {
    let from: CanvasNode
    let to: CanvasNode
    let nodeSizes: [UUID: CGSize]
    let zoom: Double
    let offset: CGPoint
    var color: Color = .blue
    var isSelected: Bool = false   // 选中态：虚线 + 渐变流动（输出→输入方向）
    var isDisabled: Bool = false   // 禁用态：灰色虚线 + 低透明度（与主题色实线区分明显）
    var isGlowing: Bool = false    // 连续视频发光态：全局色固定频率闪烁（仅发光线，动画范围最小化）
    /// 节点拖动期间的临时位移（画布坐标）与拖动节点集合：端点随拖动节点实时补偿，避免连线滞后
    var dragOffset: CGPoint = .zero
    var draggingNodeIDs: Set<UUID> = []
    
    var body: some View {
        if isGlowing {
            // 发光态：固定 1.2s 周期正弦平滑呼吸（亮→暗→亮连续渐变，非硬切换；仅发光线进入动画）
            TimelineView(.animation) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                // phase ∈ 0...1：sin 周期 1.2s，一个完整呼吸（亮→暗→亮）
                let phase = (sin(t * .pi * 2 / 1.2) + 1) / 2
                let start = compensatedPort(of: from, side: .right)
                let end = compensatedPort(of: to, side: .left)
                ConnectionCurve(start: start, end: end)
                    .stroke(color.opacity(0.35 + 0.6 * phase),
                            style: StrokeStyle(lineWidth: 2 + phase, lineCap: .round))
                    .shadow(color: color.opacity(0.1 + 0.8 * phase), radius: 2 + 6 * phase)
            }
        } else if isDisabled {
            // 禁用态：灰色虚线 + 低透明度（来源节点输入不会被加入条件输入）
            let start = compensatedPort(of: from, side: .right)
            let end = compensatedPort(of: to, side: .left)
            ConnectionCurve(start: start, end: end)
                .stroke(Color.gray.opacity(0.45),
                        style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [5, 5]))
        } else if isSelected {
            // 选中态：虚线 + 无缝流动动画（仅选中时运行 TimelineView，避免空转）
            TimelineView(.animation) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                // dash 图案周期 = 7+5 = 12；phase 每 1s 从 0 跑到 12（正好一个完整周期），
                // 循环点图案完全一致，实现无缝流动（输出→输入方向）
                let dashTotal: CGFloat = 12
                let phase = CGFloat(t.truncatingRemainder(dividingBy: 1.0)) * dashTotal
                let start = compensatedPort(of: from, side: .right)
                let end = compensatedPort(of: to, side: .left)
                ConnectionCurve(start: start, end: end)
                    .stroke(color.opacity(0.95),
                            style: StrokeStyle(lineWidth: 2.5,
                                               lineCap: .round,
                                               dash: [7, 5],
                                               dashPhase: -phase))
            }
        } else {
            // 未选中：静态实线
            let start = compensatedPort(of: from, side: .right)
            let end = compensatedPort(of: to, side: .left)
            ConnectionCurve(start: start, end: end)
                .stroke(color.opacity(0.7), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        }
    }
    
    /// 端口位置（屏幕坐标）+ 拖动中的节点临时位移补偿
    private func compensatedPort(of node: CanvasNode, side: ConnectSide) -> CGPoint {
        let p = portPosition(of: node, side: side, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
        guard draggingNodeIDs.contains(node.id) else { return p }
        return CGPoint(x: p.x + dragOffset.x * zoom, y: p.y + dragOffset.y * zoom)
    }
}

/// ConnectionLine Equatable：滚动/平移画布时（from/to/zoom/offset 等渲染输入全部不变）剪枝 body，避免连线重算与位图重采样。
extension ConnectionLine: Equatable {
    static func == (lhs: ConnectionLine, rhs: ConnectionLine) -> Bool {
        // ②比较瘦身：连线渲染只依赖两端节点的位置（端口坐标）+ 尺寸（查 nodeSizes），
        // 不比整个 CanvasNode（title/prompt/history 等与连线无关）；nodeSizes 只比两端节点的尺寸，不比全字典
        lhs.from.id == rhs.from.id &&
        lhs.from.position == rhs.from.position &&
        lhs.to.id == rhs.to.id &&
        lhs.to.position == rhs.to.position &&
        lhs.nodeSizes[lhs.from.id] == rhs.nodeSizes[rhs.from.id] &&
        lhs.nodeSizes[lhs.to.id] == rhs.nodeSizes[rhs.to.id] &&
        lhs.zoom == rhs.zoom &&
        lhs.offset == rhs.offset &&
        lhs.color == rhs.color &&
        lhs.isSelected == rhs.isSelected &&
        lhs.isDisabled == rhs.isDisabled &&
        lhs.isGlowing == rhs.isGlowing &&
        lhs.dragOffset == rhs.dragOffset &&
        lhs.draggingNodeIDs == rhs.draggingNodeIDs
    }
}

// 连线贝塞尔曲线中点（屏幕坐标），用于放置中心裁断按钮
func connectionMidPoint(from: CanvasNode, to: CanvasNode, nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint) -> CGPoint {
    let start = portPosition(of: from, side: .right, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
    let end = portPosition(of: to, side: .left, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
    return CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2)
}

// 正在拖拽的临时连线（虚线）
// startPoint：组盒加号拖出时传入组盒球心（屏幕坐标）；nil 时用 from 节点端口作为起点
struct DraggingLine: View {
    let from: CanvasNode?
    let startPoint: CGPoint?
    let to: CGPoint   // 画布坐标
    let nodeSizes: [UUID: CGSize]
    let zoom: Double
    let offset: CGPoint
    var side: ConnectSide = .right
    var color: Color = .blue
    
    init(from: CanvasNode?, startPoint: CGPoint? = nil, to: CGPoint, nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint, side: ConnectSide = .right, color: Color = .blue) {
        self.from = from
        self.startPoint = startPoint
        self.to = to
        self.nodeSizes = nodeSizes
        self.zoom = zoom
        self.offset = offset
        self.side = side
        self.color = color
    }
    
    var body: some View {
        Path { path in
            let start: CGPoint
            if let startPoint {
                start = startPoint
            } else if let from {
                start = portPosition(of: from, side: side, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
            } else {
                start = .zero
            }
            let end = CGPoint(x: to.x * zoom + offset.x, y: to.y * zoom + offset.y)
            let midX = (start.x + end.x) / 2
            path.move(to: start)
            path.addCurve(to: end,
                          control1: CGPoint(x: midX, y: start.y),
                          control2: CGPoint(x: midX, y: end.y))
        }
        .stroke(color.opacity(0.5), style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [6, 4]))
    }
}

// ============================================================

// MARK: - 节点输入框（节点下方提示词输入区 + 展开聚焦模式 + 打组）

/// 各类型节点输入框底部要求文本（不同节点显示不同要求；文本无要求栏）
func nodePromptRequirement(for type: NodeType) -> String? {
    switch type {
    case .audio: return "44100Hz"
    case .video: return "1080P 25fps"
    case .image: return "1024×1024"
    case .character: return "角色描述"
    case .scene: return "场景描述"
    case .text: return nil
    }
}

/// 各类型节点输入框占位符
func nodePromptPlaceholder(for type: NodeType) -> String {
    switch type {
    case .audio: return "输入音色描述，@引用素材..."
    case .video: return "输入画面描述，@引用素材..."
    case .image: return "输入画面描述，@引用素材..."
    case .character: return "输入角色描述，@引用素材..."
    case .scene: return "输入场景描述，@引用素材..."
    case .text: return "输入内容描述..."
    }
}

/// 节点下方输入框卡片（跟随节点显示；内容由外部持有草稿，失焦/发送/展开关闭时写回节点）
struct NodePromptBar: View {
    let node: CanvasNode
    let currentGlobalRatio: CanvasStore.Ratio
    let currentGlobalDuration: VideoDuration
    @Binding var text: String
    var onExpand: () -> Void
    var onSend: () -> Void
    var onRatioChange: (CanvasStore.Ratio) -> Void
    /// 视频节点模型选择回调
    var onModelChange: (VideoModel) -> Void = { _ in }
    /// 图像/角色/场景节点模型选择回调（暂时只有 HiDream-O1）
    var onImageModelChange: (ImageModel) -> Void = { _ in }
    /// 视频节点已连入有效图片（首帧/尾帧条件）时，私有比例选择禁用（尺寸由图决定）
    var hasValidImageInput: Bool = false
    /// 视频节点清晰度档位选择回调
    var onQualityChange: (VideoQuality) -> Void = { _ in }
    /// 视频节点时长（秒数）选择回调
    var onDurationChange: (VideoDuration) -> Void = { _ in }
    /// 图像节点尺寸档位选择回调
    var onImageQualityChange: (ImageQuality) -> Void = { _ in }
    /// 视频生成中：发送按钮转圈禁用（仅视频节点生成期间为 true）
    var isGenerating: Bool = false
    /// 输入校验错误（视频节点发送前检查连线输入，超限时由外部写入；非 nil 显示红色感叹号）
    var validationError: String? = nil
    @State private var showValidationPopover = false
    /// 翻译进行中：防重复点击
    @State private var isTranslating = false
    /// 翻译失败提示：非 nil 时在输入框下方显示红色提示（下次点击或翻译成功时清除）
    @State private var translateError: String?
    /// 翻译会话配置：非 nil 时驱动 .translationTask 执行一次翻译（点击时先置 nil 再置新值，强制触发）
    @State private var translationConfig: TranslationSession.Configuration?
    /// 待翻译文本：供 .translationTask action 读取，避免闭包捕获过期状态
    @State private var pendingTranslationText = ""
    
    var body: some View {
        VStack(spacing: 10) {
            // 第一行：输入区（浅灰圆角包裹，一眼看出是输入框）+ 展开
            HStack(alignment: .center, spacing: 8) {
                TextField(nodePromptPlaceholder(for: node.type), text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .lineLimit(3...5)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.black.opacity(0.06))
                    )
                // 翻译按钮：按字符占比判断方向（中文多→英文，英文多→中文），系统内置翻译框架离线翻译
                Button {
                    translatePrompt()
                } label: {
                    Image(systemName: "translate")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .disabled(isTranslating)
                Button(action: onExpand) {
                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            
            // 翻译失败提示：红色简短文案，点击翻译或翻译成功时自动清除
            if let translateError {
                Text(translateError)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 2)
            }
            
            // 第二行：加号（第一个）+ 模型选择 + 要求 + 信息 + 发送
            HStack(spacing: 10) {
                Image(systemName: "plus")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.secondary)
                
                // 模型选择：视频节点用视频模型，图像/角色/场景节点用图像模型（暂时只有 HiDream-O1）
                if node.type == .video {
                    compactMenu(
                        items: VideoModel.allCases,
                        isSelected: { node.model == $0 },
                        displayName: { $0.displayName },
                        onSelect: { model in
                            debugLog("节点输入框：选择模型 \(model.displayName)")
                            onModelChange(model)
                        }
                    ) {
                        Text(node.model.displayName)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                } else {
                    compactMenu(
                        items: ImageModel.allCases,
                        isSelected: { node.imageModel == $0 },
                        displayName: { $0.displayName },
                        onSelect: { model in
                            debugLog("节点输入框：选择图像模型 \(model.displayName)")
                            onImageModelChange(model)
                        }
                    ) {
                        Text(node.imageModel.displayName)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                }
                
                // 比例下拉：视频/图像/角色/场景节点通用（nil 时跟随全局比例；与画布右上角一致）
                // 编辑模式（hasValidImageInput 为 true）比例由参考图决定，禁用比例下拉（含视频 I2V 首帧）
                if node.type == .video || node.type == .image || node.type == .character || node.type == .scene {
                    compactMenu(
                        items: CanvasStore.Ratio.allCases,
                        isSelected: { node.ratio == $0 },
                        displayName: { $0.displayName },
                        disabled: hasValidImageInput,
                        onSelect: { ratio in
                            debugLog("节点输入框：选择私有比例 \(ratio.displayName)")
                            onRatioChange(ratio)
                        }
                    ) {
                        HStack(spacing: 4) {
                            Image(systemName: "crop")
                            Text(node.ratio?.displayName ?? currentGlobalRatio.displayName)
                        }
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                    }
                    // 编辑模式（hasValidImageInput 为 true）比例由参考图决定，禁用比例下拉（含视频 I2V 首帧）
                    
                    // 档位下拉：视频节点用视频档位，图像/角色/场景节点用图像档位（多 2K 档）
                    if node.type == .video {
                        compactMenu(
                            items: VideoQuality.allCases,
                            isSelected: { node.quality == $0 },
                            displayName: { $0.displayName },
                            onSelect: { q in
                                debugLog("节点输入框：视频节点选择档位 \(q.displayName)")
                                onQualityChange(q)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "rectangle.arrowtriangle.2.inward")
                                Text(node.quality.displayName)
                            }
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        }
                        
                        // 时长下拉：5s / 10s（nil 跟随全局秒数）
                        compactMenu(
                            items: VideoDuration.allCases,
                            isSelected: { node.duration == $0 },
                            displayName: { $0.displayName },
                            onSelect: { d in
                                debugLog("节点输入框：选择视频时长 \(d.displayName)")
                                onDurationChange(d)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "clock")
                                Text(node.duration?.displayName ?? currentGlobalDuration.displayName)
                            }
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        }
                    } else {
                        compactMenu(
                            items: ImageQuality.allCases.filter { !hasValidImageInput || $0 == .p1080 || $0 == .p2k },
                            isSelected: { node.imageQuality.resolved(hasValidImageInput: hasValidImageInput) == $0 },
                            displayName: { $0.displayName },
                            onSelect: { q in
                                debugLog("节点输入框：选择图像档位 \(q.displayName)")
                                onImageQualityChange(q)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "rectangle.arrowtriangle.2.inward")
                                Text(node.imageQuality.resolved(hasValidImageInput: hasValidImageInput).displayName)
                            }
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        }
                        // 编辑模式档位仅 1080/2K（1080p 2MP 为编辑稳定下限）
                    }
                } else if let req = nodePromptRequirement(for: node.type) {
                    Text(req)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button(action: { debugLog("节点输入框：点击信息") }) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                
                // 输入校验感叹号：视频节点连线输入超限时变红；点击弹出报错原因
                if node.type == .video, let validationError {
                    Button {
                        showValidationPopover = true
                    } label: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.red)
                            .help(validationError)
                    }
                    .buttonStyle(.plain)
                    .popover(isPresented: $showValidationPopover, arrowEdge: .bottom) {
                        Text(validationError)
                            .font(.system(size: 12))
                            .padding(10)
                            .frame(maxWidth: 260)
                    }
                }
                
                Button(action: onSend) {
                    if isGenerating {
                        ProgressView()
                            .controlSize(.small)
                            .frame(width: 22, height: 22)
                            .help("生成中…")
                    } else {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 22))
                            .foregroundColor(.blue)
                    }
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.96))
                .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)
        )
        .frame(width: 420)
        .translationTask(translationConfig) { session in
            // 环境注入的翻译会话：仅当有待翻译文本时执行；每次点击先置 nil 再置新配置，保证触发
            let text = pendingTranslationText
            guard !text.isEmpty else { return }
            do {
                let response = try await session.translate(text)
                let targetText = response.targetText
                await MainActor.run {
                    isTranslating = false
                    self.text = targetText
                    translateError = nil
                    debugLog("节点输入框：翻译完成")
                }
            } catch {
                await MainActor.run {
                    isTranslating = false
                    translateError = "翻译失败：模型未下载，请到系统设置-通用-翻译下载语言包"
                    debugLog("节点输入框：翻译失败：\(error.localizedDescription)")
                }
            }
        }
    }
    
    /// 翻译按钮动作：按字符占比判断方向（中文多→英文，英文多→中文），通过 .translationTask 环境注入的系统翻译会话执行
    private func translatePrompt() {
        let text = self.text
        // 统计中文字符数（Unicode 表意文字：扩展 A 区 + 基本区）
        let chineseCount = text.unicodeScalars.reduce(0) { count, scalar in
            let v = scalar.value
            return ((v >= 0x3400 && v <= 0x4DBF) || (v >= 0x4E00 && v <= 0x9FFF)) ? count + 1 : count
        }
        // 统计英文字母数（a-zA-Z）
        let englishCount = text.unicodeScalars.reduce(0) { count, scalar in
            let v = scalar.value
            return ((v >= 0x41 && v <= 0x5A) || (v >= 0x61 && v <= 0x7A)) ? count + 1 : count
        }
        // 中英字符数相等（含均为 0）：不翻译，保持原文
        guard chineseCount != englishCount else {
            debugLog("节点输入框：翻译跳过（中英字符数相等或输入为空）")
            return
        }
        let zhMore = chineseCount > englishCount
        let from = Locale.Language(identifier: zhMore ? "zh-Hans" : "en")
        let to = Locale.Language(identifier: zhMore ? "en" : "zh-Hans")
        // 清除上一次的错误提示
        translateError = nil
        pendingTranslationText = ""
        translationConfig = nil
        isTranslating = true
        // 下一拍同时写入待翻译文本与新配置，驱动 .translationTask 执行一次翻译（先置 nil 保证每次点击都触发）
        Task { @MainActor in
            pendingTranslationText = text
            translationConfig = TranslationSession.Configuration(source: from, target: to)
        }
    }
}

/// 展开聚焦模式：深色遮罩（集中精神）+ 居中白色小卡片（约 560 宽，不再铺满全屏）
struct ExpandedNodePromptView: View {
    let node: CanvasNode
    let currentGlobalRatio: CanvasStore.Ratio
    let currentGlobalDuration: VideoDuration
    @Binding var text: String
    var onClose: () -> Void
    var onSend: () -> Void
    var onRatioChange: (CanvasStore.Ratio) -> Void
    /// 视频节点模型选择回调
    var onModelChange: (VideoModel) -> Void = { _ in }
    /// 图像/角色/场景节点模型选择回调（暂时只有 HiDream-O1）
    var onImageModelChange: (ImageModel) -> Void = { _ in }
    /// 视频节点已连入有效图片（首帧/尾帧条件）时，私有比例选择禁用（尺寸由图决定）
    var hasValidImageInput: Bool = false
    /// 视频节点清晰度档位选择回调
    var onQualityChange: (VideoQuality) -> Void = { _ in }
    /// 视频节点时长（秒数）选择回调
    var onDurationChange: (VideoDuration) -> Void = { _ in }
    /// 图像节点尺寸档位选择回调
    var onImageQualityChange: (ImageQuality) -> Void = { _ in }
    /// 视频生成中：发送按钮转圈禁用
    var isGenerating: Bool = false
    /// 输入校验错误（视频节点发送前检查连线输入，超限时由外部写入；非 nil 显示红色感叹号）
    var validationError: String? = nil
    @State private var showValidationPopover = false
    
    var body: some View {
        ZStack {
            Color.black.opacity(0.65)
                .ignoresSafeArea()
                .onTapGesture { onClose() }
            
            VStack(spacing: 0) {
                // 顶栏：仅右上角收缩按钮（与展开按钮方向相悖）
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "arrow.down.left")
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("收起")
                }
                
                // 中间：多行输入区（无边框）
                TextField(nodePromptPlaceholder(for: node.type), text: $text, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 16))
                    .lineLimit(6...)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                
                // 底部工具行：全部图标化，无文字按钮
                HStack(spacing: 14) {
                    Image(systemName: "plus")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(.secondary)
                    Image(systemName: "at")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(.secondary)
                    // 模型选择：视频节点用视频模型，图像/角色/场景节点用图像模型（暂时只有 HiDream-O1）
                    if node.type == .video {
                        compactMenu(
                            items: VideoModel.allCases,
                            isSelected: { node.model == $0 },
                            displayName: { $0.displayName },
                            onSelect: { model in
                                debugLog("展开输入框：选择模型 \(model.displayName)")
                                onModelChange(model)
                            }
                        ) {
                            Text(node.model.displayName)
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                        }
                    } else {
                        compactMenu(
                            items: ImageModel.allCases,
                            isSelected: { node.imageModel == $0 },
                            displayName: { $0.displayName },
                            onSelect: { model in
                                debugLog("展开输入框：选择图像模型 \(model.displayName)")
                                onImageModelChange(model)
                            }
                        ) {
                            Text(node.imageModel.displayName)
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // 比例下拉：视频/图像/角色/场景节点通用（nil 时跟随全局比例）
                    // 编辑模式（hasValidImageInput 为 true）比例由参考图决定，禁用比例下拉（含视频 I2V 首帧）
                    if node.type == .video || node.type == .image || node.type == .character || node.type == .scene {
                        compactMenu(
                            items: CanvasStore.Ratio.allCases,
                            isSelected: { node.ratio == $0 },
                            displayName: { $0.displayName },
                            disabled: hasValidImageInput,
                            onSelect: { ratio in
                                debugLog("展开输入框：选择私有比例 \(ratio.displayName)")
                                onRatioChange(ratio)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "crop")
                                Text(node.ratio?.displayName ?? currentGlobalRatio.displayName)
                            }
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                        }
                        // 编辑模式（hasValidImageInput 为 true）比例由参考图决定，禁用比例下拉（含视频 I2V 首帧）
                        
                        // 档位下拉：视频节点用视频档位，图像/角色/场景节点用图像档位（多 2K 档）
                        if node.type == .video {
                            compactMenu(
                                items: VideoQuality.allCases,
                                isSelected: { node.quality == $0 },
                                displayName: { $0.displayName },
                                onSelect: { q in
                                    debugLog("展开输入框：视频节点选择档位 \(q.displayName)")
                                    onQualityChange(q)
                                }
                            ) {
                                HStack(spacing: 4) {
                                    Image(systemName: "rectangle.arrowtriangle.2.inward")
                                    Text(node.quality.displayName)
                                }
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                            }
                            
                            // 时长下拉：5s / 10s（nil 跟随全局秒数）
                            compactMenu(
                                items: VideoDuration.allCases,
                                isSelected: { node.duration == $0 },
                                displayName: { $0.displayName },
                                onSelect: { d in
                                    debugLog("展开输入框：选择视频时长 \(d.displayName)")
                                    onDurationChange(d)
                                }
                            ) {
                                HStack(spacing: 4) {
                                    Image(systemName: "clock")
                                    Text(node.duration?.displayName ?? currentGlobalDuration.displayName)
                                }
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                            }
                        } else {
                            compactMenu(
                                items: ImageQuality.allCases.filter { !hasValidImageInput || $0 == .p1080 || $0 == .p2k },
                                isSelected: { node.imageQuality.resolved(hasValidImageInput: hasValidImageInput) == $0 },
                                displayName: { $0.displayName },
                                onSelect: { q in
                                    debugLog("展开输入框：选择图像档位 \(q.displayName)")
                                    onImageQualityChange(q)
                                }
                            ) {
                                HStack(spacing: 4) {
                                    Image(systemName: "rectangle.arrowtriangle.2.inward")
                                    Text(node.imageQuality.resolved(hasValidImageInput: hasValidImageInput).displayName)
                                }
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                            }
                            // 编辑模式档位仅 1080/2K（1080p 2MP 为编辑稳定下限）
                        }
                    } else if let req = nodePromptRequirement(for: node.type) {
                        Text(req)
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Button(action: { debugLog("节点输入框：点击信息") }) {
                        Image(systemName: "info.circle")
                            .font(.system(size: 14))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    
                    // 输入校验感叹号：视频节点连线输入超限时变红；点击弹出报错原因
                    if node.type == .video, let validationError {
                        Button {
                            showValidationPopover = true
                        } label: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 15))
                                .foregroundColor(.red)
                                .help(validationError)
                        }
                        .buttonStyle(.plain)
                        .popover(isPresented: $showValidationPopover, arrowEdge: .bottom) {
                            Text(validationError)
                                .font(.system(size: 12))
                                .padding(10)
                                .frame(maxWidth: 280)
                        }
                    }
                    
                    Button(action: onSend) {
                        if isGenerating {
                            ProgressView()
                                .controlSize(.small)
                                .frame(width: 34, height: 34)
                                .help("生成中…")
                        } else {
                            Image(systemName: "arrow.up")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.white)
                                .frame(width: 34, height: 34)
                                .background(Circle().fill(Color.black))
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(isGenerating)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
            }
            .padding(20)
            .frame(width: 560)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.white.opacity(0.97))
                    .shadow(color: Color.black.opacity(0.3), radius: 24, x: 0, y: 10)
            )
        }
    }
}

/// 打组按钮（多选节点时显示在选中节点群上方）
struct GroupNodesButton: View {
    let count: Int
    var onGroup: () -> Void
    
    var body: some View {
        Button(action: onGroup) {
            Label("打组 (\(count))", systemImage: "square.on.square")
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    Capsule()
                        .fill(Color.white.opacity(0.96))
                        .shadow(color: Color.black.opacity(0.12), radius: 4, x: 0, y: 2)
                )
        }
        .buttonStyle(.plain)
    }
}

/// 节点集合的包围盒（画布坐标，含缩放；不含视口 offset，调用方自行叠加）
/// draggingNodeIDs + dragOffset：拖动期间让拖动中节点按「起始位置 + 临时位移」参与包围盒，
/// 组盒/打组按钮据此实时包裹节点，不必每帧写回 nodes（与节点拖动临时位移方案同构，保持性能）
func nodesBoundingRect(_ nodes: [CanvasNode], nodeSizes: [UUID: CGSize], zoom: Double,
                       draggingNodeIDs: Set<UUID> = [], dragOffset: CGPoint = .zero) -> CGRect {
    guard let first = nodes.first else { return .zero }
    func draggedPos(_ node: CanvasNode) -> CGPoint {
        draggingNodeIDs.contains(node.id)
            ? CGPoint(x: node.position.x + dragOffset.x, y: node.position.y + dragOffset.y)
            : node.position
    }
    let firstSize = nodeSizes[first.id] ?? nodeDisplaySize(for: first)
    let firstPos = draggedPos(first)
    var minX = firstPos.x * zoom - firstSize.width / 2 * zoom
    var maxX = firstPos.x * zoom + firstSize.width / 2 * zoom
    var minY = firstPos.y * zoom - firstSize.height / 2 * zoom
    var maxY = firstPos.y * zoom + firstSize.height / 2 * zoom
    for node in nodes.dropFirst() {
        let size = nodeSizes[node.id] ?? nodeDisplaySize(for: node)
        let p = draggedPos(node)
        minX = min(minX, p.x * zoom - size.width / 2 * zoom)
        maxX = max(maxX, p.x * zoom + size.width / 2 * zoom)
        minY = min(minY, p.y * zoom - size.height / 2 * zoom)
        maxY = max(maxY, p.y * zoom + size.height / 2 * zoom)
    }
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
}

// MARK: - 节点输入框通用下拉菜单（单选输入框 / 展开输入框共用）
// 封装 Menu + ForEach + checkmark 标记 + borderlessButton 样式，去掉 12 处同构样板。
// items 传入已过滤选项数组；label 尾随闭包提供触发按钮内容；onSelect 由调用方负责日志与写回。
func compactMenu<T: Hashable>(
    items: [T],
    isSelected: @escaping (T) -> Bool,
    displayName: @escaping (T) -> String,
    disabled: Bool = false,
    onSelect: @escaping (T) -> Void,
    @ViewBuilder label: () -> some View
) -> some View {
    Menu {
        ForEach(items, id: \.self) { item in
            Button {
                onSelect(item)
            } label: {
                if isSelected(item) {
                    Label(displayName(item), systemImage: "checkmark")
                } else {
                    Text(displayName(item))
                }
            }
        }
    } label: {
        label()
    }
    .menuStyle(.borderlessButton)
    .menuIndicator(.visible)
    .fixedSize()
    .disabled(disabled)
}


