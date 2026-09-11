// ============================================================
//  文件作用：节点几何计算。加号端口尺寸常量（nodePortSize / nodeBallRadius / groupBoxPadding），节点尺寸映射表（NodeSizeSpec / nodeSizeSpecs / nodeBaseWidth），几何函数（roundedToGrid / roundedGridSize / ballCenterOffset / groupPortBallCenter / emptyPlaceholderHeight / emptyNodeButtonRect / generationCancelButtonRect / historyBadgeCenterOffset / videoNodeSize / nodeDisplaySize / nodeWidthForType / placeholderHeightForType / playButtonCenterOffset / nodeCardTotalHeight / tailFrameSwitchHitRect / connectionMidPoint / nodePromptRequirement / nodePromptPlaceholder / nodesBoundingRect）。
//  互动文件：渲染侧（NodeView / ConnectionLine / NodePromptBar / GroupNodesButton）与命中侧（节点 交互 鼠标.swift）共用，保证渲染与命中永远一致。
// ============================================================

import SwiftUI
import Foundation

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
let emptyPlaceholderIconSize: CGFloat = 28
let emptyPlaceholderSpacing: CGFloat = 10

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

// 连线贝塞尔曲线中点（屏幕坐标），用于放置中心裁断按钮
func connectionMidPoint(from: CanvasNode, to: CanvasNode, nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint) -> CGPoint {
    let start = portPosition(of: from, side: .right, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
    let end = portPosition(of: to, side: .left, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
    return CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2)
}

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
