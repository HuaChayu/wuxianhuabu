//
//  节点.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/14.
//
// ============================================================
//  文件作用：节点与连线视图。NodeView 节点卡片视图、ConnectionLine /
//  DraggingLine 连线视图、NodePromptBar 输入框、ExpandedNodePromptView 展开模式、
//  GroupNodesButton 打组按钮、compactMenu 下拉菜单辅助。
//  互动文件：被 交互.swift、画布 ui.swift、画布 状态.swift、画布 工具栏2.swift、
//  画布 菜单.swift、画布 资产面板.swift、画布 节点操作.swift 引用（提供类型与视图）；
//  数据模型已拆分至 ../../Domain/节点 数据模型.swift（NodeType / CanvasNode /
//  NodeConnection / DraggingConnection / AlignmentGuide / ConnectSide），几何与常量
//  拆分至 ../../Domain/节点 几何.swift，媒体/图片缓存拆分至 ../../Data/媒体缓存.swift。
// ============================================================

import SwiftUI
import AppKit
import AVFoundation
import Translation


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
    /// 偏好设置（悬停荧光范围/强度，偏好设置页"节点默认颜色"下方可调）
    @ObservedObject private var settings = AppSettings.shared
    
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
                            UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 3, style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [accentColor.opacity(0.12), accentColor.opacity(0.04)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        } else {
                            UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 3, style: .continuous)
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
                            .clipShape(UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 3, style: .continuous))
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
                            .clipShape(UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 3, style: .continuous))
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
                // 悬停命中：下半部分（底部信息栏）变主题色（顶部直角，避免与素材区交界反衬出圆角白边）
                .background(
                    UnevenRoundedRectangle(topLeadingRadius: 0, bottomLeadingRadius: 6, bottomTrailingRadius: 6, topTrailingRadius: 0, style: .continuous)
                        .fill(isHovered ? accentColor.opacity(0.18) : Color.clear)
                )
            }
            .background(
                ZStack(alignment: .bottom) {
                    // 底层：卡片白底 + 整圈描边（描边完整绘制，由上层内容/压线带盖住内侧呈镂空细线）
                    UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 10, bottomTrailingRadius: 10, topTrailingRadius: 3, style: .continuous)
                        .fill(Color.white.opacity(0.9))
                        .overlay(
                            UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 10, bottomTrailingRadius: 10, topTrailingRadius: 3, style: .continuous)
                                .stroke(isSelected ? accentColor : (isHovered ? accentColor.opacity(0.8) : Color.gray.opacity(0.25)),
                                        lineWidth: isSelected ? 2 : (isHovered ? 2 : 1))
                        )
                    // 标签区压线带：不透明卡片底色贴边（顶部直角），从信息栏顶（Divider 下缘）盖到卡片底，
                    // 压住标签段描边内侧 → 与素材区图片压住描边一致，全局镂空细线
                    UnevenRoundedRectangle(topLeadingRadius: 0, bottomLeadingRadius: 10, bottomTrailingRadius: 10, topTrailingRadius: 0, style: .continuous)
                        .fill(Color.white)
                        .frame(maxHeight: .infinity)
                        .padding(.top, placeholderHeight + 1)
                }
            )
            .shadow(color: isHovered ? accentColor.opacity(settings.hoverGlowIntensity) : .clear, radius: settings.hoverGlowRadius)
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

/// 节点下方输入框卡片（跟随节点显示；内容由外部持有草稿，失焦/发送/展开关闭时写回节点）
struct NodePromptBar: View {
    let node: CanvasNode
    let currentGlobalRatio: CanvasStore.Ratio
    let currentGlobalDuration: VideoDuration
    @Binding var text: String
    var onExpand: () -> Void
    var onSend: () -> Void
    /// 尾帧续接链路展示信息：非 nil = 本视频节点处于尾帧续接链路（h3ChainSourceID 命中且前置
    /// 视频节点有实际像素尺寸），生成尺寸强制跟随前置视频；尺寸档位/比例下拉改为展示前置视频
    /// 实际档位/比例并禁用（延续前置视频，仅告知不可改）
    var chainDisplayInfo: CanvasStore.ChainVideoDisplayInfo? = nil
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
                        disabled: hasValidImageInput || chainDisplayInfo != nil,
                        onSelect: { ratio in
                            debugLog("节点输入框：选择私有比例 \(ratio.displayName)")
                            onRatioChange(ratio)
                        }
                    ) {
                        HStack(spacing: 4) {
                            Image(systemName: "crop")
                            Text(chainDisplayInfo?.ratio.displayName ?? node.ratio?.displayName ?? currentGlobalRatio.displayName)
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
                            disabled: chainDisplayInfo != nil,
                            onSelect: { q in
                                debugLog("节点输入框：视频节点选择档位 \(q.displayName)")
                                onQualityChange(q)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "rectangle.arrowtriangle.2.inward")
                                Text(chainDisplayInfo?.quality.displayName ?? node.quality.displayName)
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
    /// 尾帧续接链路展示信息：非 nil = 本视频节点处于尾帧续接链路（h3ChainSourceID 命中且前置
    /// 视频节点有实际像素尺寸），生成尺寸强制跟随前置视频；尺寸档位/比例下拉改为展示前置视频
    /// 实际档位/比例并禁用（延续前置视频，仅告知不可改）
    var chainDisplayInfo: CanvasStore.ChainVideoDisplayInfo? = nil
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
                            disabled: hasValidImageInput || chainDisplayInfo != nil,
                            onSelect: { ratio in
                                debugLog("展开输入框：选择私有比例 \(ratio.displayName)")
                                onRatioChange(ratio)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "crop")
                                Text(chainDisplayInfo?.ratio.displayName ?? node.ratio?.displayName ?? currentGlobalRatio.displayName)
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
                                disabled: chainDisplayInfo != nil,
                                onSelect: { q in
                                    debugLog("展开输入框：视频节点选择档位 \(q.displayName)")
                                    onQualityChange(q)
                                }
                            ) {
                                HStack(spacing: 4) {
                                    Image(systemName: "rectangle.arrowtriangle.2.inward")
                                    Text(chainDisplayInfo?.quality.displayName ?? node.quality.displayName)
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


