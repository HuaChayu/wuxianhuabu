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
                    // 素材区不透明垫底：贴边盖住素材段描边内侧，避免半透明灰底/图片 fit 留缝透出
                    // 描边淡色（深紫描边 + 透过白底的淡紫描边叠成双层轮廓）；与标签区压线带同理
                    UnevenRoundedRectangle(topLeadingRadius: 3, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 3, style: .continuous)
                        .fill(Color.white)
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
                    // 压住标签段描边内侧 → 与素材区图片压住描边一致，全局镂空细线；
                    // 左右内缩 0.7pt：描边保留 85% 露出（2pt 描边露 1.7pt），避免被压得过细
                    UnevenRoundedRectangle(topLeadingRadius: 0, bottomLeadingRadius: 10, bottomTrailingRadius: 10, topTrailingRadius: 0, style: .continuous)
                        .fill(Color.white)
                        .frame(maxHeight: .infinity)
                        .padding(.top, placeholderHeight)
                        .padding(.horizontal, 0.7)
                        .padding(.bottom, 0.7)
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
    /// 尾帧续接链路展示信息：非 nil = 本视频节点处于尾帧续接链路（链上存在上游续接源），
    /// 生成尺寸/比例/时长强制跟随链首（最上游非续接视频节点）当前参数；尺寸档位/比例/时长
    /// 下拉改为展示链首即将生效参数并禁用（延续链首参数，仅告知不可改）
    var chainDisplayInfo: CanvasStore.ChainVideoDisplayInfo? = nil
    var onRatioChange: (CanvasStore.Ratio) -> Void
    /// 视频节点模型选择回调
    var onModelChange: (VideoModel) -> Void = { _ in }
    /// H3 参考模式选择回调（多参考 / 首尾帧，仅 MiniMax H3 生效）
    var onRefModeChange: (H3RefMode) -> Void = { _ in }
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
    /// 前置视频（尾帧续接源，已开尾帧）正在生成：本节点发送按钮一并禁用，
    /// 避免手动点生成与级联传播冲突（闪退恢复后前置空闲时仍可手动生成）
    var isChainSourceGenerating: Bool = false
    /// 输入校验错误（视频节点发送前检查连线输入，超限时由外部写入；非 nil 显示红色感叹号）
    var validationError: String? = nil
    /// 前置条件缩略图：非空 = 本节点有入边（前面连了节点），面板上方显示缩略图区；
    /// 空数组 = 无前置条件，保持原「只有输入」的样式（由画布侧 store.promptInputThumbnails 计算）
    var inputThumbnails: [PromptInputThumbnail] = []
    /// 输入框内引用块（@ 前置节点）：与草稿同层持有；点缩略图展开里的 @ 追加，块内保留被引用节点的名字与 id
    var mentions: Binding<[PromptMention]> = .constant([])
    /// 待插入的引用块：点缩略图里的 @ / @ 提及面板选择时写入，行内输入控件插入后自动置空
    @State private var pendingMention: PromptMention?
    /// 手打 @ 之后的光标位置（控件用它删掉 @ 再插块）
    @State private var typedAtAnchor: Int?
    /// @ 提及面板是否展开
    @State private var showMentionPanel = false
    /// 手打 @ 之后继续输入的文字：作为本项目内节点搜索条件实时过滤面板候选项；
    /// 空字符串 = 尚未输入，面板保持原结构（素材引用 + 按类型分类展开）
    @State private var mentionQuery: String = ""
    /// @ 提及候选（上游连线节点 + 本项目全部节点分类，由画布侧计算）
    var mentionCatalog: PromptMentionCatalog = .empty
    /// 面板键盘高亮项下标（0 = 打开即高亮第一项）：↑↓ 移动，→/回车 进入类别或插入该行
    @State private var mentionHighlight: Int = 0
    /// 面板当前展开的分类：提到这里持有，键盘导航与面板渲染共用同一份可见候选清单
    @State private var mentionExpandedKind: NodeType?
    /// 键盘焦点是否已进入某分类（true = 正在右栏逐项浏览该分类的节点，← 退回分类行）
    @State private var mentionFocusInCategory = false
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
            // 前置条件缩略图区：仅当本节点前面连了节点时出现（打开面板即时计算，无连线则不占高度）
            if !inputThumbnails.isEmpty {
                // 悬停展开后点 @ → 把该前置节点作为引用块插入输入框
                PromptInputThumbnailStrip(items: inputThumbnails, onMention: { addMention($0) })
            }
            
            // 第一行：输入区（浅灰圆角包裹，一眼看出是输入框）+ 展开
            HStack(alignment: .center, spacing: 8) {
                // 输入区：引用块与文字同行混排（块 文字 块 文字），块后可直接继续输入
                InlineMentionTextView(
                    text: $text,
                    mentions: mentions,
                    pendingMention: $pendingMention,
                    typedAtAnchor: $typedAtAnchor,
                    onTypedAt: { showMentionPanel = true },
                    onMentionQueryChange: handleMentionQueryChange,
                    isMentionPanelActive: showMentionPanel,
                    onMentionKeyCommand: handleMentionKeyCommand,
                    placeholder: nodePromptPlaceholder(for: node.type),
                    fontSize: 14
                )
                .popover(isPresented: $showMentionPanel) {
                    PromptMentionPickPanel(
                        catalog: mentionCatalog,
                        query: mentionQuery,
                        highlightIndex: mentionHighlight,
                        expandedKind: $mentionExpandedKind,
                        focusInCategory: $mentionFocusInCategory
                    ) { pickMention($0) }
                }
                .onChange(of: showMentionPanel) { _, isShowing in
                    if isShowing {
                        // 打开即高亮第一项：不必先按 ↓ 就能 → / 回车
                        mentionHighlight = 0
                        mentionFocusInCategory = false
                    } else {
                        // 面板以任何方式关掉（选完 / 点外部 / Esc）→ 清掉过滤词、分类与高亮，下次打开回到原结构
                        mentionQuery = ""
                        mentionExpandedKind = nil
                        mentionFocusInCategory = false
                        mentionHighlight = 0
                    }
                }
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
                
                // H3 参考模式下拉：多参考（ref2va 软参考）/ 首尾帧（fl 硬锚），控制原 NA_H3_FL2VA_AS_REFS 分流
                // 仅 MiniMax H3 模型可切换；非 H3 模型禁用并保持节点选择（或默认多参考）
                if node.type == .video {
                    compactMenu(
                        items: H3RefMode.allCases,
                        isSelected: { (node.refMode ?? .multiRef) == $0 },
                        displayName: { $0.displayName },
                        disabled: node.model != .minimaxH3,
                        onSelect: { mode in
                            debugLog("节点输入框：选择 H3 参考模式 \(mode.displayName)")
                            onRefModeChange(mode)
                        }
                    ) {
                        HStack(spacing: 4) {
                            Image(systemName: (node.refMode ?? .multiRef) == .firstLast ? "arrow.left.and.right" : "photo.stack")
                            Text((node.refMode ?? .multiRef).displayName)
                        }
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .frame(minWidth: 58)   // 横向稍宽，与模型选择拉开层级
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
                        
                        // 时长下拉：5s / 10s / 15s（nil 跟随全局秒数；续接链路中禁用并跟随链首时长）
                        compactMenu(
                            items: VideoDuration.allCases,
                            isSelected: { node.duration == $0 },
                            displayName: { $0.displayName },
                            disabled: chainDisplayInfo != nil,
                            onSelect: { d in
                                debugLog("节点输入框：选择视频时长 \(d.displayName)")
                                onDurationChange(d)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: "clock")
                                Text(chainDisplayInfo?.duration?.displayName ?? node.duration?.displayName ?? currentGlobalDuration.displayName)
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
                .disabled(isGenerating || isChainSourceGenerating)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.96))
                .shadow(color: Color.black.opacity(0.12), radius: 6, x: 0, y: 2)
        )
        .frame(width: 460)
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
                    mentions.wrappedValue = []
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
    
    /// 点缩略图展开里的 @ → 请求在输入框光标处插入引用块（同一节点可重复插入多个块，不去重）
    private func addMention(_ thumbnail: PromptInputThumbnail) {
        pendingMention = PromptMention(
            id: UUID(),
            nodeID: thumbnail.id,
            title: thumbnail.title,
            kind: thumbnail.kind,
            image: thumbnail.image
        )
        debugLog("节点输入框：请求插入引用 \(thumbnail.title)（\(thumbnail.id.uuidString)）")
    }

    /// @ 提及面板选中某个节点 → 请求在「手打 @ 的位置」插入引用块（控件负责把那个 @ 删掉）
    private func pickMention(_ candidate: PromptMentionCandidate) {
        showMentionPanel = false
        mentionQuery = ""
        pendingMention = PromptMention(
            id: UUID(),
            nodeID: candidate.id,
            title: candidate.title,
            kind: candidate.kind,
            image: candidate.image
        )
        debugLog("节点输入框：@ 面板选择引用 \(candidate.title)（\(candidate.id.uuidString)）")
    }

    /// 手打 @ 之后继续输入的文字 → 面板实时筛选；nil = 提及输入已结束（@ 被删 / 光标离开 / 敲了空格）
    private func handleMentionQueryChange(_ query: String?) {
        guard let query else {
            mentionQuery = ""
            mentionHighlight = 0
            mentionFocusInCategory = false
            if showMentionPanel { showMentionPanel = false }
            if typedAtAnchor != nil { typedAtAnchor = nil }
            return
        }
        // 过滤词变了 → 清单长度会变，高亮回到第一项，避免停在错行上
        if mentionQuery != query {
            mentionHighlight = 0
            mentionFocusInCategory = false
        }
        mentionQuery = query
        // 不在这里补弹面板：面板由手打 @ 的 onTypedAt 负责弹出，用户点外部关掉后继续打字不该被顶起来
    }

    /// 面板当前的 ↑↓ 序列（与面板渲染共用 catalog 里的同一套计算，保证键盘选中项就是屏幕上那行）：
    /// 浏览根层 = 素材引用候选 + 分类行；已进入分类 = 该分类下节点；搜索态 = 过滤结果
    private var mentionNavRows: [PromptMentionRow] {
        mentionCatalog.navRows(
            keyword: mentionQuery,
            enteredKind: mentionExpandedKind,
            focusInCategory: mentionFocusInCategory
        )
    }

    /// 当前高亮行（下标越界时夹到最后一行，保证 → / 回车始终有目标）
    private var mentionHighlightedRow: PromptMentionRow? {
        let rows = mentionNavRows
        guard !rows.isEmpty else { return nil }
        return rows[min(max(mentionHighlight, 0), rows.count - 1)]
    }

    /// @ 面板打开时的键盘接管：↑↓ 在序列内移动（含分类行），→ / 回车进入类别或插入引用，← 退回上层，Esc 收起。
    /// 返回 false = 该键不归面板管（面板没开 / 序列为空时回车仍是换行）
    private func handleMentionKeyCommand(_ command: MentionPanelKeyCommand) -> Bool {
        guard showMentionPanel else { return false }
        let rows = mentionNavRows
        switch command {
        case .previous:
            guard !rows.isEmpty else { return false }
            mentionHighlight = max(min(mentionHighlight, rows.count - 1) - 1, 0)
            return true
        case .next:
            guard !rows.isEmpty else { return false }
            mentionHighlight = min(mentionHighlight + 1, rows.count - 1)
            return true
        case .enter:
            return enterMentionRow()
        case .back:
            return backFromMentionRow()
        case .cancel:
            showMentionPanel = false
            mentionQuery = ""
            mentionHighlight = 0
            mentionFocusInCategory = false
            return true
        }
    }

    /// → / 回车：分类行 = 进入该类别（右栏逐项浏览，高亮落到第一项）；候选项 = 插入引用块
    private func enterMentionRow() -> Bool {
        guard let row = mentionHighlightedRow else { return false }
        switch row {
        case .candidate(let candidate):
            pickMention(candidate)
            return true
        case .group(let group):
            // 空分类不进入：留在左栏，避免进到空右栏后无从下手
            guard !group.items.isEmpty else { return true }
            mentionExpandedKind = group.kind
            mentionFocusInCategory = true
            mentionHighlight = 0
            return true
        }
    }

    /// ← ：在类别内 = 一次收起该类别并把焦点退回它的分类行；在分类行 = 收起已展开的类别；根层无操作（仍消费，避免移动光标关掉面板）
    private func backFromMentionRow() -> Bool {
        if mentionFocusInCategory {
            let kind = mentionExpandedKind
            // 一次 ← 就收起：先复位焦点与展开态，再把光标落到刚才那个分类行上（不留「先退光标」的中间态）
            mentionFocusInCategory = false
            mentionExpandedKind = nil
            if let kind, let index = mentionNavRows.firstIndex(where: { $0.group?.kind == kind }) {
                mentionHighlight = index
            } else {
                mentionHighlight = 0
            }
            return true
        }
        if mentionExpandedKind != nil {
            mentionExpandedKind = nil
            mentionHighlight = min(mentionHighlight, max(mentionNavRows.count - 1, 0))
        }
        return true
    }
    
    /// 翻译按钮动作：按字符占比判断方向（中文多→英文，英文多→中文），通过 .translationTask 环境注入的系统翻译会话执行
    private func translatePrompt() {
        // 翻译只处理纯文字：引用块占位符不参与翻译（翻译结果会替换整段文字，块随之清空）
        let text = self.text.strippingMentionPlaceholders
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
    /// 尾帧续接链路展示信息：非 nil = 本视频节点处于尾帧续接链路（链上存在上游续接源），
    /// 生成尺寸/比例/时长强制跟随链首（最上游非续接视频节点）当前参数；尺寸档位/比例/时长
    /// 下拉改为展示链首即将生效参数并禁用（延续链首参数，仅告知不可改）
    var chainDisplayInfo: CanvasStore.ChainVideoDisplayInfo? = nil
    var onRatioChange: (CanvasStore.Ratio) -> Void
    /// 视频节点模型选择回调
    var onModelChange: (VideoModel) -> Void = { _ in }
    /// H3 参考模式选择回调（多参考 / 首尾帧，仅 MiniMax H3 生效）
    var onRefModeChange: (H3RefMode) -> Void = { _ in }
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
    /// 前置视频（尾帧续接源，已开尾帧）正在生成：本节点发送按钮一并禁用
    var isChainSourceGenerating: Bool = false
    /// 输入校验错误（视频节点发送前检查连线输入，超限时由外部写入；非 nil 显示红色感叹号）
    var validationError: String? = nil
    /// 前置条件缩略图：非空 = 本节点前面连了节点，展开面板上方同样展示缩略图区
    var inputThumbnails: [PromptInputThumbnail] = []
    /// 输入框内引用块（@ 前置节点）：与草稿同层持有；点缩略图展开里的 @ 追加，块内保留被引用节点的名字与 id
    var mentions: Binding<[PromptMention]> = .constant([])
    /// 待插入的引用块：点缩略图里的 @ / @ 提及面板选择时写入，行内输入控件插入后自动置空
    @State private var pendingMention: PromptMention?
    /// 手打 @ 之后的光标位置（控件用它删掉 @ 再插块）
    @State private var typedAtAnchor: Int?
    /// @ 提及面板是否展开
    @State private var showMentionPanel = false
    /// 手打 @ 之后继续输入的文字：作为本项目内节点搜索条件实时过滤面板候选项；
    /// 空字符串 = 尚未输入，面板保持原结构（素材引用 + 按类型分类展开）
    @State private var mentionQuery: String = ""
    /// @ 提及候选（上游连线节点 + 本项目全部节点分类，由画布侧计算）
    var mentionCatalog: PromptMentionCatalog = .empty
    /// 面板键盘高亮项下标（0 = 打开即高亮第一项）：↑↓ 移动，→/回车 进入类别或插入该行
    @State private var mentionHighlight: Int = 0
    /// 面板当前展开的分类：提到这里持有，键盘导航与面板渲染共用同一份可见候选清单
    @State private var mentionExpandedKind: NodeType?
    /// 键盘焦点是否已进入某分类（true = 正在右栏逐项浏览该分类的节点，← 退回分类行）
    @State private var mentionFocusInCategory = false
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
                
                // 前置条件缩略图区（有入边才出现，与节点下方输入框一致）
                if !inputThumbnails.isEmpty {
                    // 悬停展开后点 @ → 把该前置节点作为引用块插入输入框
                    PromptInputThumbnailStrip(items: inputThumbnails, cardWidth: 40, cardHeight: 40, cardCornerRadius: 6, onMention: { addMention($0) })
                        .padding(.horizontal, 24)
                        .padding(.top, 4)
                }
                
                // 中间：多行输入区（无边框），引用块与文字同行混排
                InlineMentionTextView(
                    text: $text,
                    mentions: mentions,
                    pendingMention: $pendingMention,
                    typedAtAnchor: $typedAtAnchor,
                    onTypedAt: { showMentionPanel = true },
                    onMentionQueryChange: handleMentionQueryChange,
                    isMentionPanelActive: showMentionPanel,
                    onMentionKeyCommand: handleMentionKeyCommand,
                    placeholder: nodePromptPlaceholder(for: node.type),
                    fontSize: 16,
                    minHeight: 26
                )
                .popover(isPresented: $showMentionPanel) {
                    PromptMentionPickPanel(
                        catalog: mentionCatalog,
                        query: mentionQuery,
                        highlightIndex: mentionHighlight,
                        expandedKind: $mentionExpandedKind,
                        focusInCategory: $mentionFocusInCategory
                    ) { pickMention($0) }
                }
                .onChange(of: showMentionPanel) { _, isShowing in
                    if isShowing {
                        // 打开即高亮第一项：不必先按 ↓ 就能 → / 回车
                        mentionHighlight = 0
                        mentionFocusInCategory = false
                    } else {
                        // 面板以任何方式关掉（选完 / 点外部 / Esc）→ 清掉过滤词、分类与高亮，下次打开回到原结构
                        mentionQuery = ""
                        mentionExpandedKind = nil
                        mentionFocusInCategory = false
                        mentionHighlight = 0
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
                
                // 底部工具行：全部图标化，无文字按钮
                HStack(spacing: 14) {
                    Image(systemName: "plus")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundColor(.secondary)
                    // @ 提及面板：点图标直接弹面板，选中后块插到当前光标处（文本里不追加 @）
                    Button {
                        typedAtAnchor = nil
                        mentionQuery = ""
                        mentionHighlight = 0
                        mentionFocusInCategory = false
                        mentionExpandedKind = nil
                        showMentionPanel = true
                    } label: {
                        Image(systemName: "at")
                            .font(.system(size: 15, weight: .medium))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("引用本项目节点")
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
                    
                    // H3 参考模式下拉：多参考（ref2va 软参考）/ 首尾帧（fl 硬锚），控制原 NA_H3_FL2VA_AS_REFS 分流
                    // 仅 MiniMax H3 模型可切换；非 H3 模型禁用并保持节点选择（或默认多参考）
                    if node.type == .video {
                        compactMenu(
                            items: H3RefMode.allCases,
                            isSelected: { (node.refMode ?? .multiRef) == $0 },
                            displayName: { $0.displayName },
                            disabled: node.model != .minimaxH3,
                            onSelect: { mode in
                                debugLog("展开输入框：选择 H3 参考模式 \(mode.displayName)")
                                onRefModeChange(mode)
                            }
                        ) {
                            HStack(spacing: 4) {
                                Image(systemName: (node.refMode ?? .multiRef) == .firstLast ? "arrow.left.and.right" : "photo.stack")
                                Text((node.refMode ?? .multiRef).displayName)
                            }
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                            .frame(minWidth: 58)   // 横向稍宽，与模型选择拉开层级
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
                            
                            // 时长下拉：5s / 10s / 15s（nil 跟随全局秒数；续接链路中禁用并跟随链首时长）
                            compactMenu(
                                items: VideoDuration.allCases,
                                isSelected: { node.duration == $0 },
                                displayName: { $0.displayName },
                                disabled: chainDisplayInfo != nil,
                                onSelect: { d in
                                    debugLog("展开输入框：选择视频时长 \(d.displayName)")
                                    onDurationChange(d)
                                }
                            ) {
                                HStack(spacing: 4) {
                                    Image(systemName: "clock")
                                    Text(chainDisplayInfo?.duration?.displayName ?? node.duration?.displayName ?? currentGlobalDuration.displayName)
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
                    .disabled(isGenerating || isChainSourceGenerating)
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
    
    /// 点缩略图展开里的 @ → 请求在展开输入框光标处插入引用块（同一节点可重复插入多个块，不去重）
    private func addMention(_ thumbnail: PromptInputThumbnail) {
        pendingMention = PromptMention(
            id: UUID(),
            nodeID: thumbnail.id,
            title: thumbnail.title,
            kind: thumbnail.kind,
            image: thumbnail.image
        )
        debugLog("展开输入框：请求插入引用 \(thumbnail.title)（\(thumbnail.id.uuidString)）")
    }

    /// @ 提及面板选中某个节点 → 请求在「手打 @ 的位置」插入引用块（控件负责把那个 @ 删掉）
    private func pickMention(_ candidate: PromptMentionCandidate) {
        showMentionPanel = false
        mentionQuery = ""
        pendingMention = PromptMention(
            id: UUID(),
            nodeID: candidate.id,
            title: candidate.title,
            kind: candidate.kind,
            image: candidate.image
        )
        debugLog("展开输入框：@ 面板选择引用 \(candidate.title)（\(candidate.id.uuidString)）")
    }

    /// 手打 @ 之后继续输入的文字 → 面板实时筛选；nil = 提及输入已结束（@ 被删 / 光标离开 / 敲了空格）
    private func handleMentionQueryChange(_ query: String?) {
        guard let query else {
            mentionQuery = ""
            mentionHighlight = 0
            mentionFocusInCategory = false
            if showMentionPanel { showMentionPanel = false }
            if typedAtAnchor != nil { typedAtAnchor = nil }
            return
        }
        // 过滤词变了 → 清单长度会变，高亮回到第一项，避免停在错行上
        if mentionQuery != query {
            mentionHighlight = 0
            mentionFocusInCategory = false
        }
        mentionQuery = query
        // 不在这里补弹面板：面板由手打 @ 的 onTypedAt 负责弹出，用户点外部关掉后继续打字不该被顶起来
    }

    /// 面板当前的 ↑↓ 序列（与面板渲染共用 catalog 里的同一套计算，保证键盘选中项就是屏幕上那行）：
    /// 浏览根层 = 素材引用候选 + 分类行；已进入分类 = 该分类下节点；搜索态 = 过滤结果
    private var mentionNavRows: [PromptMentionRow] {
        mentionCatalog.navRows(
            keyword: mentionQuery,
            enteredKind: mentionExpandedKind,
            focusInCategory: mentionFocusInCategory
        )
    }

    /// 当前高亮行（下标越界时夹到最后一行，保证 → / 回车始终有目标）
    private var mentionHighlightedRow: PromptMentionRow? {
        let rows = mentionNavRows
        guard !rows.isEmpty else { return nil }
        return rows[min(max(mentionHighlight, 0), rows.count - 1)]
    }

    /// @ 面板打开时的键盘接管：↑↓ 在序列内移动（含分类行），→ / 回车进入类别或插入引用，← 退回上层，Esc 收起。
    /// 返回 false = 该键不归面板管（面板没开 / 序列为空时回车仍是换行）
    private func handleMentionKeyCommand(_ command: MentionPanelKeyCommand) -> Bool {
        guard showMentionPanel else { return false }
        let rows = mentionNavRows
        switch command {
        case .previous:
            guard !rows.isEmpty else { return false }
            mentionHighlight = max(min(mentionHighlight, rows.count - 1) - 1, 0)
            return true
        case .next:
            guard !rows.isEmpty else { return false }
            mentionHighlight = min(mentionHighlight + 1, rows.count - 1)
            return true
        case .enter:
            return enterMentionRow()
        case .back:
            return backFromMentionRow()
        case .cancel:
            showMentionPanel = false
            mentionQuery = ""
            mentionHighlight = 0
            mentionFocusInCategory = false
            return true
        }
    }

    /// → / 回车：分类行 = 进入该类别（右栏逐项浏览，高亮落到第一项）；候选项 = 插入引用块
    private func enterMentionRow() -> Bool {
        guard let row = mentionHighlightedRow else { return false }
        switch row {
        case .candidate(let candidate):
            pickMention(candidate)
            return true
        case .group(let group):
            // 空分类不进入：留在左栏，避免进到空右栏后无从下手
            guard !group.items.isEmpty else { return true }
            mentionExpandedKind = group.kind
            mentionFocusInCategory = true
            mentionHighlight = 0
            return true
        }
    }

    /// ← ：在类别内 = 一次收起该类别并把焦点退回它的分类行；在分类行 = 收起已展开的类别；根层无操作（仍消费，避免移动光标关掉面板）
    private func backFromMentionRow() -> Bool {
        if mentionFocusInCategory {
            let kind = mentionExpandedKind
            // 一次 ← 就收起：先复位焦点与展开态，再把光标落到刚才那个分类行上（不留「先退光标」的中间态）
            mentionFocusInCategory = false
            mentionExpandedKind = nil
            if let kind, let index = mentionNavRows.firstIndex(where: { $0.group?.kind == kind }) {
                mentionHighlight = index
            } else {
                mentionHighlight = 0
            }
            return true
        }
        if mentionExpandedKind != nil {
            mentionExpandedKind = nil
            mentionHighlight = min(mentionHighlight, max(mentionNavRows.count - 1, 0))
        }
        return true
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


// MARK: - 输入框引用块（@ 前置节点，行内混排）
// 点缩略图里的 @ → 在光标处插入一个引用块；块与文字任意混排（块 文字 块 文字），块后可直接继续打字。
// 块内不再显示 @，只留被引用节点的头像/类型图标 + 名字；节点 id 存在数据里，供后续取内容。
// 文本中用 U+FFFC 占位符代表块，mentions 按占位符出现顺序一一对应。

/// 引用块占位符
enum PromptMentionToken {
    static let placeholder = "\u{FFFC}"
    /// 文本中的占位符个数（= 引用块个数），用于校验 mentions 与原文是否对齐
    static func count(in text: String) -> Int {
        text.reduce(into: 0) { partial, character in
            if String(character) == placeholder { partial += 1 }
        }
    }
}

extension String {
    /// 去掉引用块占位符后的纯提示词文本（写回节点、发送模型、翻译前调用）
    var strippingMentionPlaceholders: String {
        replacingOccurrences(of: PromptMentionToken.placeholder, with: "")
    }
}

/// 引用块贴图附件：附带被引用节点信息（id 供后续取内容）
final class PromptMentionAttachment: NSTextAttachment {
    let mention: PromptMention

    init(mention: PromptMention, fontSize: CGFloat) {
        self.mention = mention
        super.init(data: nil, ofType: nil)
        let chip = PromptMentionChipImage.make(for: mention, fontSize: fontSize)
        image = chip
        bounds = CGRect(x: 0, y: -3, width: chip.size.width, height: chip.size.height)
    }

    required init?(coder: NSCoder) { nil }
}

/// 把引用块画成贴图（白底圆角 + 小头像/类型图标 + 名字，不含 @）
enum PromptMentionChipImage {
    static func make(for mention: PromptMention, fontSize: CGFloat) -> NSImage {
        let titleFont = NSFont.systemFont(ofSize: max(10, fontSize - 2), weight: .medium)
        let iconSide = max(12, fontSize)
        let hPadding: CGFloat = 5
        let gap: CGFloat = 4
        let height = ceil(fontSize + 7)

        let paragraph = NSMutableParagraphStyle()
        paragraph.lineBreakMode = .byTruncatingTail
        let attributes: [NSAttributedString.Key: Any] = [
            .font: titleFont,
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: paragraph
        ]
        let title = NSAttributedString(string: mention.title, attributes: attributes)
        let textWidth = min(ceil(title.size().width), 150)
        let width = max(ceil(hPadding * 2 + iconSide + gap + textWidth), ceil(height))

        return NSImage(size: NSSize(width: width, height: height), flipped: false) { rect in
            let path = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: 4, yRadius: 4)
            NSColor.textBackgroundColor.setFill()
            path.fill()
            NSColor.separatorColor.setStroke()
            path.lineWidth = 1
            path.stroke()

            let iconRect = NSRect(x: hPadding, y: (height - iconSide) / 2, width: iconSide, height: iconSide)
            if let thumbnail = mention.image {
                NSGraphicsContext.saveGraphicsState()
                NSBezierPath(roundedRect: iconRect, xRadius: 3, yRadius: 3).addClip()
                thumbnail.draw(in: iconRect, from: .zero, operation: .sourceOver, fraction: 1)
                NSGraphicsContext.restoreGraphicsState()
            } else {
                let config = NSImage.SymbolConfiguration(pointSize: max(8, iconSide - 3), weight: .medium)
                if let symbol = NSImage(systemSymbolName: mention.kind.placeholderIcon, accessibilityDescription: nil)?
                    .withSymbolConfiguration(config) {
                    let tint = NSColor(mention.kind.thumbnailAccent)
                    let tinted = NSImage(size: symbol.size, flipped: false) { symbolRect in
                        symbol.draw(in: symbolRect)
                        tint.set()
                        symbolRect.fill(using: .sourceAtop)
                        return true
                    }
                    let drawRect = NSRect(
                        x: iconRect.midX - symbol.size.width / 2,
                        y: iconRect.midY - symbol.size.height / 2,
                        width: symbol.size.width,
                        height: symbol.size.height
                    )
                    tinted.draw(in: drawRect)
                }
            }

            let textHeight = title.size().height
            let textRect = NSRect(
                x: hPadding + iconSide + gap,
                y: (height - textHeight) / 2,
                width: textWidth,
                height: textHeight
            )
            title.draw(in: textRect)
            return true
        }
    }
}

/// 高度随内容自适应的 NSTextView：块与文字同行混排、块后继续输入都靠它（SwiftUI TextField 无法渲染行内附件）
final class InlineMentionNSTextView: NSTextView {
    var minimumHeight: CGFloat = 22
    var placeholderString: String = ""

    override var intrinsicContentSize: NSSize {
        NSSize(width: NSView.noIntrinsicMetric, height: measuredHeight(for: nil))
    }

    override func layout() {
        super.layout()
        guard let textContainer else { return }
        let width = max(bounds.width - textContainerInset.width * 2, 10)
        if abs(textContainer.containerSize.width - width) > 0.5 {
            textContainer.containerSize = NSSize(width: width, height: .greatestFiniteMagnitude)
            // 宽度变化会改变换行行数：下一拍通知外层重算高度，避免多行文本被裁
            DispatchQueue.main.async { [weak self] in
                self?.invalidateIntrinsicContentSize()
            }
        }
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
    }

    /// 按给定宽度算出的高度（宽度为 nil 时用当前 textContainer 宽度）
    func measuredHeight(for width: CGFloat?) -> CGFloat {
        guard let layoutManager, let textContainer else { return minimumHeight }
        if let width, width > 0 {
            textContainer.containerSize = NSSize(
                width: max(width - textContainerInset.width * 2, 10),
                height: .greatestFiniteMagnitude
            )
        }
        layoutManager.ensureLayout(for: textContainer)
        let contentHeight = ceil(layoutManager.usedRect(for: textContainer).height)
        return max(contentHeight + textContainerInset.height * 2, minimumHeight)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholderString.isEmpty else { return }
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font ?? NSFont.systemFont(ofSize: 14),
            .foregroundColor: NSColor.placeholderTextColor
        ]
        NSAttributedString(string: placeholderString, attributes: attributes)
            .draw(at: NSPoint(x: textContainerInset.width, y: textContainerInset.height))
    }
}

/// @ 面板打开期间从输入框接管过来的按键（输入框自己不再响应）
enum MentionPanelKeyCommand {
    /// ↑：上移一行（序列含分类行）
    case previous
    /// ↓：下移一行（序列含分类行）
    case next
    /// → / 回车：分类行 = 进入该类别逐项浏览；候选项 = 插入引用
    case enter
    /// ← ：从类别内退回类别行；在类别行 = 收起已展开的分类
    case back
    /// Esc：收起面板
    case cancel
}

/// 行内混排输入框：文字与引用块在同一行流式排列
struct InlineMentionTextView: NSViewRepresentable {
    @Binding var text: String
    @Binding var mentions: [PromptMention]
    /// 待插入的引用块（点缩略图 @ / @ 提及面板选择时写入，控件插入后自动置空）
    @Binding var pendingMention: PromptMention?
    /// 手打 @ 之后的光标位置（= @ 字符位 +1）：写入即代表面板要删掉这个 @ 再插块；
    /// 控件插入引用块时按此位置把 @ 一并删掉，面板关闭时由外部置空（@ 保留在文本里）
    @Binding var typedAtAnchor: Int?
    /// 手打 @ 时回调（面板据此弹出）
    var onTypedAt: (() -> Void)? = nil
    /// 手打 @ 之后「@ 与光标之间」的文字变化回调：
    /// - 字符串（可为空）= 当前过滤词，面板据此实时筛选本项目节点；
    /// - nil = 提及输入结束（@ 已被删除、光标移出 @ 之后的连续输入区、或敲了空格/换行）
    var onMentionQueryChange: ((String?) -> Void)? = nil
    /// @ 面板当前是否开着：用户点面板外部 / Esc 主动关掉后，@ 之后的文字不再当过滤词（也不会把面板顶起来）
    var isMentionPanelActive: Bool = false
    /// @ 面板打开时把 ↑/↓/←/→/回车/Esc 交给面板：
    /// 返回 true = 面板已消费该键（输入框不响应，光标不移动、不插换行）；false = 原样交回输入框
    var onMentionKeyCommand: ((MentionPanelKeyCommand) -> Bool)? = nil
    var placeholder: String
    var fontSize: CGFloat = 14
    var minHeight: CGFloat = 22

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> InlineMentionNSTextView {
        let textView = InlineMentionNSTextView(frame: .zero)
        textView.delegate = context.coordinator
        textView.isEditable = true
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = false
        textView.backgroundColor = .clear
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 0, height: 1)
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = false
        textView.textContainer?.heightTracksTextView = false
        textView.allowsUndo = true
        textView.font = NSFont.systemFont(ofSize: fontSize)
        textView.textColor = .labelColor
        textView.insertionPointColor = .labelColor
        textView.placeholderString = placeholder
        textView.minimumHeight = minHeight
        context.coordinator.rebuild(textView, text: text, mentions: mentions)
        return textView
    }

    func updateNSView(_ textView: InlineMentionNSTextView, context: Context) {
        context.coordinator.parent = self
        // 父级清空了 @ 锚点（面板关闭 / 引用块已插入）→ 控件同步结束「@ 后连续输入」状态。
        // 刚从 textDidChange 记下锚点、绑定还没回写的那一帧不算「已清空」，否则会把自己的锚点误清掉
        if typedAtAnchor == nil, !context.coordinator.isAnchorSyncPending {
            context.coordinator.typedAtAnchorLocation = nil
        }
        textView.placeholderString = placeholder
        textView.minimumHeight = minHeight
        if (textView.font?.pointSize ?? 0) != fontSize {
            textView.font = NSFont.systemFont(ofSize: fontSize)
        }

        if let pending = pendingMention {
            let anchor = typedAtAnchor
            context.coordinator.insert(pending, into: textView, removingTypedAt: anchor)
            // 同一帧内 updateNSView 可能被调用多次：只清掉本次这一条，
            // 避免把用户紧接着新点的另一条 pending 也清掉（丢块）
            DispatchQueue.main.async {
                if self.pendingMention?.id == pending.id { self.pendingMention = nil }
                if anchor != nil, self.typedAtAnchor == anchor { self.typedAtAnchor = nil }
            }
            return
        }

        if !context.coordinator.isDirty, !textView.hasMarkedText(),
           textView.string != text || context.coordinator.renderedMentionIDs != mentions.map(\.id) {
            // 文本相同但引用块不同（切换节点后两个节点占位符文本一样）也要重建，否则残留上个节点的块
            context.coordinator.rebuild(textView, text: text, mentions: mentions)
        }

        // 被引用节点改名：只就地重绘名字变了的块贴图（不动文本、光标与撤销栈）
        context.coordinator.refreshMentionChips(textView, mentions: mentions)
    }

    func sizeThatFits(_ proposal: ProposedViewSize, nsView: InlineMentionNSTextView, context: Context) -> CGSize? {
        guard let width = proposal.width, width > 0 else {
            return CGSize(width: max(nsView.bounds.width, 120), height: nsView.measuredHeight(for: nil))
        }
        return CGSize(width: width, height: nsView.measuredHeight(for: width))
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: InlineMentionTextView
        /// 控件内部刚改过文本，等回写绑定完成前不做外部重建
        var isDirty = false
        /// 当前文本里已渲染的引用块 id（按出现顺序）：用于识别「切换节点后文本相同但块不同」
        private(set) var renderedMentionIDs: [UUID] = []
        private var insertedStore: [UUID: PromptMention] = [:]

        init(_ parent: InlineMentionTextView) {
            self.parent = parent
        }

        /// 光标移动时把输入属性重置为正文样式，避免在引用块后就近继承附件属性导致文字样式跑掉
        func textViewDidChangeSelection(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView, !textView.hasMarkedText() else { return }
            let font = textView.font ?? NSFont.systemFont(ofSize: 14)
            textView.typingAttributes = [.font: font, .foregroundColor: NSColor.labelColor]
            // 光标若移出「@ 之后」的连续输入区，视为提及输入结束（面板收起）
            reportMentionQuery(textView)
        }

        /// @ 面板打开期间接管方向键 / 回车 / Esc：交给面板移动高亮、插入引用、收起面板。
        /// 面板没开（isMentionPanelActive == false）时一律返回 false，输入框键盘行为完全不变；
        /// 输入法组字中也不抢键（否则会把候选词选择搞乱）。
        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard parent.isMentionPanelActive, !textView.hasMarkedText() else { return false }
            let command: MentionPanelKeyCommand
            switch commandSelector {
            case #selector(NSResponder.moveUp(_:)):
                command = .previous
            case #selector(NSResponder.moveDown(_:)):
                command = .next
            case #selector(NSResponder.moveRight(_:)):
                command = .enter
            case #selector(NSResponder.moveLeft(_:)):
                command = .back
            case #selector(NSResponder.insertNewline(_:)):
                command = .enter
            case #selector(NSResponder.cancelOperation(_:)):
                command = .cancel
            default:
                return false
            }
            // 面板明确消费才拦下（如没有高亮项时回车仍走输入框换行）
            return parent.onMentionKeyCommand?(command) ?? false
        }

        /// 本次输入若是单个 @（手打引用触发符），记下它在文本里的位置（@ 之前的光标位置）
        private var typedAtLocation: Int?

        /// 手打 @ 后在文本里的锚点（= @ 字符位 + 1）；nil = 当前不处于「@ 后连续输入」状态
        var typedAtAnchorLocation: Int?

        /// 刚记下锚点、等父级绑定回写的过渡期：这期间 updateNSView 里的「锚点为空」判定不可信
        private(set) var isAnchorSyncPending = false

        /// 取「手打 @ 后正在输入的过滤词」：
        /// @ 仍紧邻锚点、光标在 @ 之后且两者之间没有空白/换行/引用块占位符时返回这段文字（可为空串，表示还没输入）；
        /// 其余情况返回 nil，表示这次 @ 提及输入已结束。
        private func mentionQuery(in textView: NSTextView) -> String? {
            guard let anchor = typedAtAnchorLocation, anchor >= 1 else { return nil }
            let ns = textView.string as NSString
            guard anchor <= ns.length else { return nil }
            guard ns.substring(with: NSRange(location: anchor - 1, length: 1)) == "@" else { return nil }
            let selection = textView.selectedRange()
            guard selection.length == 0, selection.location >= anchor, selection.location <= ns.length else { return nil }
            let typed = ns.substring(with: NSRange(location: anchor, length: selection.location - anchor))
            let hasBreak = typed.contains { character in
                character.isWhitespace || character.isNewline || String(character) == PromptMentionToken.placeholder
            }
            return hasBreak ? nil : typed
        }

        /// 把过滤词变化报给父级；没有 @ 锚点时直接返回，避免「点图标打开的面板」被误关
        private func reportMentionQuery(_ textView: NSTextView) {
            guard typedAtAnchorLocation != nil else { return }
            // 面板已被用户主动关掉（点外部 / Esc）且不是刚打 @ 的过渡帧 → 结束提及输入，不再回传
            guard parent.isMentionPanelActive || isAnchorSyncPending else {
                typedAtAnchorLocation = nil
                return
            }
            let query = mentionQuery(in: textView)
            DispatchQueue.main.async {
                if query == nil {
                    self.typedAtAnchorLocation = nil
                    if self.parent.typedAtAnchor != nil { self.parent.typedAtAnchor = nil }
                }
                self.parent.onMentionQueryChange?(query)
            }
        }

        func textView(_ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange, replacementString: String?) -> Bool {
            if replacementString == "@", !textView.hasMarkedText() {
                typedAtLocation = affectedCharRange.location
            } else {
                typedAtLocation = nil
            }
            return true
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            // 手打 @：把「@ 之后的光标位置」交回父级，由面板负责弹出并记录待删除位置
            if let loc = typedAtLocation {
                typedAtLocation = nil
                let anchor = loc + 1
                // 同步记下锚点（不等绑定回写），后续按键才能立刻算出过滤词
                typedAtAnchorLocation = anchor
                isAnchorSyncPending = true
                DispatchQueue.main.async {
                    self.parent.typedAtAnchor = anchor
                    self.parent.onTypedAt?()
                    self.isAnchorSyncPending = false
                }
            }
            // 输入法组字中不重建，只回写文本；上屏后由后续 update 正常同步
            let snapshot = textView.string
            // @ 之后继续输入的文字 → 实时回传为面板过滤词（未输入时为空串，面板保持原结构）
            reportMentionQuery(textView)
            let ids = Self.attachmentIDs(in: textView)
            var map = insertedStore
            for mention in parent.mentions { map[mention.id] = mention }
            let resolved = ids.compactMap { map[$0] }
            isDirty = true
            DispatchQueue.main.async {
                self.parent.text = snapshot
                if self.parent.mentions.map(\.id) != resolved.map(\.id) {
                    self.parent.mentions = resolved
                }
                self.isDirty = false
            }
            textView.invalidateIntrinsicContentSize()
        }

        /// 扫描文本中的附件，按出现顺序返回被引用节点 id
        static func attachmentIDs(in textView: NSTextView) -> [UUID] {
            guard let storage = textView.textStorage else { return [] }
            var ids: [UUID] = []
            storage.enumerateAttribute(.attachment, in: NSRange(location: 0, length: storage.length)) { value, _, _ in
                if let attachment = value as? PromptMentionAttachment {
                    ids.append(attachment.mention.id)
                }
            }
            return ids
        }

        /// 在光标处插入引用块（同一个被引用节点可重复插入多个块，块 id 各自唯一）
        func insert(_ mention: PromptMention, into textView: InlineMentionNSTextView, removingTypedAt anchor: Int? = nil) {
            // 同一次点击可能触发多轮 updateNSView（pending 尚未异步置空前）：
            // 块 id 是每次点击新生成的 UUID，按块 id 去重只拦「同一个块插两次」，
            // 不会影响点两次 @ 插入两个同节点块
            guard insertedStore[mention.id] == nil else { return }
            insertedStore[mention.id] = mention
            guard let storage = textView.textStorage else { return }
            let attachment = PromptMentionAttachment(mention: mention, fontSize: textView.font?.pointSize ?? 14)
            let chip = NSAttributedString(attachment: attachment)

            var range = textView.selectedRange()
            // 手打 @ 触发的插入：先把那个 @ 字符删掉，块顶在 @ 原来的位置（不留 "@块"）
            let ns = textView.string as NSString
            if let anchor, anchor >= 1, anchor <= ns.length {
                let atRange = NSRange(location: anchor - 1, length: 1)
                if ns.substring(with: atRange) == "@" {
                    // 「@ 后输入即搜索」：光标前那段被当成过滤词的文字也要一起删掉，只留下引用块
                    let cursor = textView.selectedRange()
                    var typedLength = 0
                    if cursor.length == 0, cursor.location >= anchor, cursor.location <= ns.length {
                        let typed = ns.substring(with: NSRange(location: anchor, length: cursor.location - anchor))
                        let hasBreak = typed.contains { character in
                            character.isWhitespace || character.isNewline || String(character) == PromptMentionToken.placeholder
                        }
                        if !hasBreak { typedLength = (typed as NSString).length }
                    }
                    storage.beginEditing()
                    storage.replaceCharacters(in: NSRange(location: anchor - 1, length: 1 + typedLength), with: "")
                    storage.endEditing()
                    range = NSRange(location: anchor - 1, length: 0)
                }
            }

            storage.beginEditing()
            storage.replaceCharacters(in: range, with: chip)
            storage.endEditing()
            textView.setSelectedRange(NSRange(location: range.location + chip.length, length: 0))
            // 引用块已落位：结束「@ 后连续输入」状态，之后打的字不再当过滤词
            typedAtAnchorLocation = nil
            textView.didChangeText()
            textView.invalidateIntrinsicContentSize()
            textView.window?.makeFirstResponder(textView)
            textView.typingAttributes = [
                .font: textView.font ?? NSFont.systemFont(ofSize: 14),
                .foregroundColor: NSColor.labelColor
            ]
            debugLog("节点输入框：插入引用 \(mention.title)（\(mention.id.uuidString)）")
        }

        /// 引用块里被引用的节点改名后，只把对应块的附件贴图换成新名字：
        /// 不重建整段文本（保住光标、选区与撤销栈），只替换 attachment 属性值（长度不变）。
        /// 没有名字变化时为零开销早退，可在每次 update 里安全调用。
        func refreshMentionChips(_ textView: InlineMentionNSTextView, mentions: [PromptMention]) {
            guard !mentions.isEmpty, let storage = textView.textStorage, storage.length > 0 else { return }
            var latestMap: [UUID: PromptMention] = [:]
            for mention in mentions { latestMap[mention.id] = mention }
            var pending: [(range: NSRange, latest: PromptMention)] = []
            storage.enumerateAttribute(.attachment, in: NSRange(location: 0, length: storage.length)) { value, range, _ in
                guard let attachment = value as? PromptMentionAttachment,
                      let latest = latestMap[attachment.mention.id],
                      latest.title != attachment.mention.title else { return }
                pending.append((range, latest))
            }
            guard !pending.isEmpty else { return }
            let fontSize = textView.font?.pointSize ?? 14
            storage.beginEditing()
            for item in pending {
                storage.addAttribute(
                    .attachment,
                    value: PromptMentionAttachment(mention: item.latest, fontSize: fontSize),
                    range: item.range
                )
            }
            storage.endEditing()
            textView.invalidateIntrinsicContentSize()
            debugLog("节点输入框：引用块改名重绘 \(pending.count) 个")
        }

        /// 按绑定文本重建（首次装载 / 外部改写）
        func rebuild(_ textView: InlineMentionNSTextView, text: String, mentions: [PromptMention]) {
            guard let storage = textView.textStorage else { return }
            let font = textView.font ?? NSFont.systemFont(ofSize: 14)
            let base: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.labelColor]
            let rebuilt = NSMutableAttributedString()
            var index = 0
            for character in text {
                if String(character) == PromptMentionToken.placeholder {
                    if index < mentions.count {
                        rebuilt.append(NSAttributedString(
                            attachment: PromptMentionAttachment(mention: mentions[index], fontSize: font.pointSize)
                        ))
                    }
                    index += 1
                } else {
                    rebuilt.append(NSAttributedString(string: String(character), attributes: base))
                }
            }
            let selected = textView.selectedRange()
            storage.setAttributedString(rebuilt)
            textView.setSelectedRange(NSRange(location: min(selected.location, storage.length), length: 0))
            textView.invalidateIntrinsicContentSize()

            let snapshot = rebuilt.string
            let renderedCount = index
            renderedMentionIDs = mentions.prefix(renderedCount).map(\.id)
            if snapshot != text || renderedCount != mentions.count {
                let surplus = renderedCount < mentions.count ? Array(mentions.prefix(renderedCount)) : mentions
                DispatchQueue.main.async {
                    if self.parent.text != snapshot { self.parent.text = snapshot }
                    if self.parent.mentions.count != surplus.count { self.parent.mentions = surplus }
                }
            }
        }
    }
}

// MARK: - 前置条件缩略图（节点下方输入框 / 展开输入框共用）
// 打开面板时若本节点有入边，面板上方展示上游节点缩略图；无入边则整块不渲染（调用方先判空）。

/// 条件缩略图横向条：按连线顺序排列（新连入的追加在队尾），最多 6 张，超出显示 +N
/// 输入框手打 @ 弹出的提及面板（参考系统 @ 面板）：
/// 顶部「素材引用」优先列本节点上游连线节点；下方按类型分类，点「展开」在右侧列出该类全部节点。
/// 所有内容都取自本项目当前画布的节点，点任意一行即插入引用块。
struct PromptMentionPickPanel: View {
    let catalog: PromptMentionCatalog
    /// 手打 @ 之后继续输入的过滤词：非空 = 进入搜索模式，把本项目内节点名按该词实时筛选成一份扁平清单；
    /// 空串 = 还没有输入，面板保持原结构（「素材引用」+ 按类型分类可展开）
    var query: String = ""
    /// 键盘高亮项在「当前 ↑↓ 序列」里的下标（宿主在面板打开时置 0 = 默认高亮第一项）
    var highlightIndex: Int = 0
    /// 当前展开的分类（nil = 只有左栏）：由宿主持有，键盘导航与面板渲染共用同一份可见清单
    @Binding var expandedKind: NodeType?
    /// 键盘焦点是否已进入某分类（true = 正在右栏逐项浏览该分类的节点）
    @Binding var focusInCategory: Bool
    var onPick: (PromptMentionCandidate) -> Void

    private let leftWidth: CGFloat = 212
    private let rightWidth: CGFloat = 208
    private let rowHeight: CGFloat = 27
    /// 两栏共用的最大高度（超出后各自滚动，避免面板无限长）
    private let panelMaxHeight: CGFloat = 320

    private var expandedItems: [PromptMentionCandidate] {
        guard let expandedKind else { return [] }
        return catalog.grouped.first(where: { $0.kind == expandedKind })?.items ?? []
    }

    /// 去掉首尾空白后的过滤词（全空白等同于没输入，保持原面板）
    private var keyword: String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var isSearching: Bool { !keyword.isEmpty }

    /// 搜索模式候选项：先按名字匹配本项目节点，上游连线节点排前面；同一节点只出现一次
    private var searchResults: [PromptMentionCandidate] {
        catalog.searchCandidates(keyword: keyword)
    }

    /// 面板当前的 ↑↓ 序列：与宿主键盘导航共用同一份计算
    /// （搜索态 = 过滤结果；已进入分类 = 该分类节点；浏览根层 = 素材引用候选 + 分类行）
    private var navRows: [PromptMentionRow] {
        catalog.navRows(keyword: keyword, enteredKind: expandedKind, focusInCategory: focusInCategory)
    }

    /// 当前高亮行：下标越界（清单变短）时夹到最后一行，保证回车/→ 始终有目标
    private var highlightedRow: PromptMentionRow? {
        guard !navRows.isEmpty else { return nil }
        return navRows[min(max(highlightIndex, 0), navRows.count - 1)]
    }

    /// 高亮行的稳定 id（ScrollViewReader 定位用）
    private var highlightedRowID: String? { highlightedRow?.id }

    /// 候选行是否处于键盘高亮
    private func isHighlighted(_ item: PromptMentionCandidate) -> Bool {
        highlightedRowID == PromptMentionRow.candidate(item).id
    }

    /// 分类行是否处于键盘高亮
    private func isHighlighted(group: PromptMentionGroup) -> Bool {
        highlightedRowID == PromptMentionRow.group(group).id
    }

    var body: some View {
        if isSearching {
            searchResultsColumn
        } else {
            browseColumns
        }
    }

    /// 搜索模式：单栏扁平清单，顶部沿用「素材引用」标题并给出命中数
    private var searchResultsColumn: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                searchResultsList
            }
            .frame(width: leftWidth)
            .frame(maxHeight: panelMaxHeight)
            // 键盘高亮移动时把该行滚进可视区（键盘选中与鼠标点击走同一套候选）
            .onChange(of: highlightedRowID) { _, id in
                guard let id else { return }
                proxy.scrollTo(id, anchor: .center)
            }
        }
    }

    /// 搜索模式的候选清单本体
    private var searchResultsList: some View {
        VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text("素材引用")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                    Spacer(minLength: 4)
                    Text("匹配 \(searchResults.count)")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary.opacity(0.8))
                }
                .padding(.horizontal, 9)
                .padding(.top, 7)
                .padding(.bottom, 3)

                if searchResults.isEmpty {
                    Text("没有名字含「\(keyword)」的节点")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary.opacity(0.7))
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                } else {
                    ForEach(searchResults) { item in
                        candidateRow(item)
                    }
                }
            }
            .padding(.bottom, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// 未输入过滤词时的原面板：左栏「素材引用」+ 按类型分类（点展开在右栏列该类全部节点）
    private var browseColumns: some View {
        HStack(alignment: .top, spacing: 0) {
            // 左栏整体上下滚动：前置连线节点有几条列几条，分类也按本项目实际类型罗列
            ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("素材引用")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 9)
                        .padding(.top, 7)
                        .padding(.bottom, 3)

                    if catalog.upstream.isEmpty {
                        Text("本节点还没有前置连线")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary.opacity(0.7))
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                    } else {
                        ForEach(catalog.upstream) { item in
                            candidateRow(item)
                        }
                    }

                    Divider()
                        .padding(.horizontal, 7)
                        .padding(.vertical, 4)

                    ForEach(catalog.grouped) { group in
                        Button {
                            withAnimation(.easeInOut(duration: 0.12)) {
                                expandedKind = (expandedKind == group.kind) ? nil : group.kind
                                // 鼠标点分类行 → 键盘焦点回到左栏清单上
                                focusInCategory = false
                            }
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: group.kind.icon)
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundColor(.secondary)
                                    .frame(width: 15)
                                Text(group.kind.rawValue)
                                    .font(.system(size: 12))
                                    .foregroundColor(.primary)
                                Spacer(minLength: 4)
                                Text(focusInCategory && expandedKind == group.kind ? "浏览中" : (expandedKind == group.kind ? "收起" : "进入"))
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                                Image(systemName: expandedKind == group.kind ? "chevron.down" : "chevron.right")
                                    .font(.system(size: 9, weight: .semibold))
                                    .foregroundColor(.secondary)
                            }
                            .padding(.horizontal, 9)
                            .frame(height: rowHeight)
                            .background(
                                RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .fill(isHighlighted(group: group) ? Color.accentColor.opacity(0.18) : Color.clear)
                                    .padding(.horizontal, 4)
                            )
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        // 分类行也参与 ↑↓ 序列：供 ScrollViewReader 定位，与宿主导航共用同一套行 id
                        .id(PromptMentionRow.group(group).id)
                    }

                    if catalog.grouped.isEmpty {
                        Text("本项目还没有可引用节点")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary.opacity(0.7))
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4)
                    }
                }
                .padding(.bottom, 5)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(width: leftWidth)
            .frame(maxHeight: panelMaxHeight)
            // 键盘高亮移动 → 把该行滚进可视区
            .onChange(of: highlightedRowID) { _, id in
                guard let id else { return }
                proxy.scrollTo(id, anchor: .center)
            }
            }

            if expandedKind != nil {
                Divider()
                ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    VStack(alignment: .leading, spacing: 1) {
                        if expandedItems.isEmpty {
                            Text("该分类下暂无节点")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary.opacity(0.7))
                                .padding(.horizontal, 9)
                                .padding(.vertical, 6)
                        } else {
                            ForEach(expandedItems) { item in
                                candidateRow(item)
                            }
                        }
                    }
                    .padding(.vertical, 5)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(width: rightWidth)
                .frame(maxHeight: panelMaxHeight)
                .onChange(of: highlightedRowID) { _, id in
                    guard let id else { return }
                    proxy.scrollTo(id, anchor: .center)
                }
                }
            }
        }
    }

    /// 面板里的一行：小图/类型图标 + 名字，点击即插入引用块；键盘高亮时整行加一层强调底色
    private func candidateRow(_ item: PromptMentionCandidate) -> some View {
        let isHighlighted = isHighlighted(item)
        return Button {
            onPick(item)
        } label: {
            HStack(spacing: 7) {
                thumbnail(item)
                Text(styledTitle(item.title))
                    .font(.system(size: 12))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 2)
            }
            .padding(.horizontal, 9)
            .frame(height: rowHeight)
            .background(
                RoundedRectangle(cornerRadius: 5, style: .continuous)
                    .fill(isHighlighted ? Color.accentColor.opacity(0.18) : Color.clear)
                    .padding(.horizontal, 4)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // 供 ScrollViewReader 定位（键盘 ↑↓ 时滚动跟随）；与宿主导航用同一套行 id
        .id(PromptMentionRow.candidate(item).id)
    }

    /// 行标题：搜索模式下把名字里命中的那段文字标成强调色，一眼看出为什么被筛出来
    private func styledTitle(_ title: String) -> AttributedString {
        var attributed = AttributedString(title)
        attributed.foregroundColor = .primary
        guard isSearching, !keyword.isEmpty,
              let range = attributed.range(
                  of: keyword,
                  options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive]
              )
        else { return attributed }
        attributed[range].foregroundColor = .accentColor
        attributed[range].inlinePresentationIntent = .stronglyEmphasized
        return attributed
    }

    @ViewBuilder
    private func thumbnail(_ item: PromptMentionCandidate) -> some View {
        ZStack {
            if let image = item.image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: item.kind.placeholderIcon)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(Color(item.kind.thumbnailAccent))
            }
        }
        .frame(width: 18, height: 18)
        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
    }
}

struct PromptInputThumbnailStrip: View {
    let items: [PromptInputThumbnail]
    /// 单张方块尺寸（默认极小方块，只占一条细带，不挤压下方输入区；展开面板可另行放大）
    var cardWidth: CGFloat = 32
    var cardHeight: CGFloat = 32
    var cardCornerRadius: CGFloat = 5
    /// 悬停展开后点 @ → 把该前置节点插入输入框（画布层写入 mentions）
    var onMention: (PromptInputThumbnail) -> Void = { _ in }
    
    var body: some View {
        HStack(spacing: 5) {
            ForEach(items.prefix(6)) { item in
                PromptInputThumbnailCard(
                    item: item,
                    width: cardWidth,
                    height: cardHeight,
                    cornerRadius: cardCornerRadius,
                    onMention: { onMention(item) }
                )
            }
            if items.count > 6 {
                Text("+\(items.count - 6)")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// 单张条件缩略图：折叠态是一个小方块（不展示完整画面，只作"前置条件已接入"的占位标识）；
/// 悬停横向展开成胶囊：左侧圆形 @ 按钮（引用该前置节点）+ 右侧小缩略图，底部一条类型色带；
/// 展开后再悬停右侧小缩略图 → popover 展示该图完整画面。
/// 无效连线（超限/空节点）→ 半透明 + 红框，提示这条条件不会参与生成。
struct PromptInputThumbnailCard: View {
    let item: PromptInputThumbnail
    var width: CGFloat = 32
    var height: CGFloat = 32
    /// 圆角随尺寸收缩（小方块用 5，展开面板可传更大值）
    var cornerRadius: CGFloat = 5
    /// 悬停展开后左侧 @ 圆钮直径系数（相对卡片高度）：
    /// 圆钮宽度即容器宽度（左右不留边），所以这个系数同时决定「左侧区域窄不窄、符号大不大」
    var mentionBadgeRatio: CGFloat = 0.42
    private var mentionBadgeSide: CGFloat { max(12, (height * mentionBadgeRatio).rounded()) }
    /// 点击 @ 的回调（把该前置节点作为引用插入输入框）；nil 时按钮仅作视觉标识
    var onMention: (() -> Void)? = nil
    
    @State private var isHovered = false
    @State private var showPreview = false
    @State private var previewShowTask: Task<Void, Never>? = nil
    @State private var previewHideTask: Task<Void, Never>? = nil
    
    private var accent: Color { item.kind.thumbnailAccent }
    /// 展开态的小缩略图边长（比 @ 圆钮略大，留出圆角）
    private var thumbSide: CGFloat { isHovered ? height - 4 : height }
    /// 展开态内边距与间距（收紧后整体更紧凑）
    private let expandHPadding: CGFloat = 3
    private let expandSpacing: CGFloat = 4
    /// @ 钮命中区在圆点之外的左右外扩量：0 = 容器宽度就是圆钮直径，符号左右不留黑边；
    /// 点击热区靠「卡片整高」在上下方向撑成矩形
    private let badgeHitPaddingX: CGFloat = 0
    /// 展开态胶囊宽度：@ 区按圆点实际宽度算（不再加宽），保证明显左窄右宽而非等分
    private var expandedWidth: CGFloat {
        (mentionBadgeSide + badgeHitPaddingX * 2) + thumbSide + expandSpacing + expandHPadding * 2
    }
    
    var body: some View {
        HStack(spacing: isHovered ? expandSpacing : 5) {
            if isHovered {
                mentionBadge
                    .transition(.opacity.combined(with: .scale(scale: 0.6)))
            }
            thumbnail
        }
        .padding(.horizontal, isHovered ? expandHPadding : 0)
        .frame(width: isHovered ? expandedWidth : width, height: height)
        .background(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(isHovered ? Color.black.opacity(0.82) : Color.black.opacity(0.06))
        )
        .overlay(alignment: .bottom) {
            // 底部类型色带（展开才出现，按源节点类型区分，仅视觉标识）
            if isHovered {
                Rectangle()
                    .fill(accent)
                    .frame(height: 2)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(item.isInvalid ? Color.red.opacity(0.55) : Color.black.opacity(0.08),
                              lineWidth: 1)
        )
        .opacity(item.isInvalid ? 0.5 : 1)
        .contentShape(Rectangle())
        .onHover { hovering in
            withAnimation(.spring(response: 0.22, dampingFraction: 0.82)) {
                isHovered = hovering
            }
            if !hovering { cancelPreviewShow(); schedulePreviewHide() }
        }
        .help(item.title)
        // 箭头朝下 → popover 出现在卡片上方（预览不遮挡下方输入区）
        .popover(isPresented: $showPreview, arrowEdge: .top) {
            previewContent
                .onHover { hovering in
                    if hovering { previewHideTask?.cancel() } else { schedulePreviewHide() }
                }
        }
    }
    
    /// 小缩略图本体：折叠态填满方块（fill 裁切），展开态缩成小方块作为"图片入口"
    private var thumbnail: some View {
        ZStack {
            if let image = item.image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: item.kind.placeholderIcon)
                    .font(.system(size: isHovered ? 10 : 12))
                    .foregroundColor(isHovered ? .white.opacity(0.75) : .secondary)
            }
        }
        .frame(width: thumbSide, height: thumbSide)
        .clipShape(RoundedRectangle(cornerRadius: isHovered ? 3 : cornerRadius, style: .continuous))
        .contentShape(Rectangle())
        .onHover { hovering in
            // 仅在展开态响应：悬停右侧小图 → 延迟展示完整图（避免划过时闪烁）
            guard isHovered else { return }
            if hovering { schedulePreviewShow() } else { cancelPreviewShow(); schedulePreviewHide() }
        }
    }
    
    /// 展开态左侧的 @ 引用按钮：
    /// 视觉上是小圆点，命中区是「占满卡片高度」的矩形（上下、左右都比圆点大），
    /// 不用精准点到 @ 字符本身也能触发引用。
    private var mentionBadge: some View {
        Button {
            onMention?()
        } label: {
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.16))
                    .frame(width: mentionBadgeSide, height: mentionBadgeSide)
                Text("@")
                    .font(.system(size: max(9, mentionBadgeSide * 0.72), weight: .semibold))
                    .foregroundColor(.white.opacity(0.92))
            }
            .frame(width: mentionBadgeSide + badgeHitPaddingX * 2, height: height)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("引用节点「\(item.title)」")
    }
    
    /// 完整图预览（popover，白色圆角卡片，按原比例 fit）
    @ViewBuilder
    private var previewContent: some View {
        if let image = item.image {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 340, maxHeight: 250)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .padding(8)
        } else {
            VStack(spacing: 6) {
                Image(systemName: item.kind.placeholderIcon)
                    .font(.system(size: 22))
                    .foregroundColor(.secondary)
                Text(item.title)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            .padding(16)
        }
    }
    
    // 悬停进出用短延迟 + 取消，规避快速划过时的预览闪烁
    private func schedulePreviewShow() {
        previewHideTask?.cancel()
        guard !showPreview else { return }
        previewShowTask?.cancel()
        previewShowTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 220_000_000)
            guard !Task.isCancelled else { return }
            showPreview = true
        }
    }
    
    private func cancelPreviewShow() {
        previewShowTask?.cancel()
        previewShowTask = nil
    }
    
    private func schedulePreviewHide() {
        guard showPreview else { return }
        previewHideTask?.cancel()
        previewHideTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 180_000_000)
            guard !Task.isCancelled else { return }
            showPreview = false
        }
    }
}

extension NodeType {
    /// 前置条件缩略图底部色带颜色（按节点类型区分，仅视觉标识，不影响生成）
    var thumbnailAccent: Color {
        switch self {
        case .character: return Color(red: 0.93, green: 0.45, blue: 0.60)
        case .scene: return Color(red: 0.25, green: 0.70, blue: 0.72)
        case .text: return Color(red: 0.30, green: 0.72, blue: 0.52)
        case .image: return Color(red: 0.29, green: 0.56, blue: 0.89)
        case .video: return Color(red: 0.56, green: 0.37, blue: 0.88)
        case .audio: return Color(red: 0.92, green: 0.62, blue: 0.27)
        }
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


