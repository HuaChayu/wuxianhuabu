// ============================================================
//  文件作用：CanvasView 扩展。右键菜单（节点操作/添加节点/主菜单）、添加节点面板
//  addNodePanel（右键与左侧工具栏共用）、引用该节点生成面板 generatePanel、
//  通用菜单项 contextMenuItem；
//  鼠标控制层 MouseControlView / MouseControlNSView（中键拖动移动、滚轮上下滚动、
//  Command+滚轮缩放、节点拖拽、框选、右键菜单、连线命中与加号连线拖拽）。
//  互动文件：引用 画布 状态.swift（CanvasStore）、节点ui.swift（NodeType、CanvasNode、
//  NodeView、ConnectionLine、DraggingLine、NodeConnection、ConnectSide、
//  connectionMidPoint、portPosition、hitTestPort）、画布 节点操作.swift（addNode、
//  generateNode、duplicateNodes、deleteNodes、copyNodeText、targetNodeIDs）、
//  资产管理 ui.swift（AssetStore）；
//  被 画布 ui.swift、工具栏资产库 ui.swift 引用。
// ============================================================

import SwiftUI

// ============================================================

// MARK: - NSView 坐标通用转换（左下原点 ↔ 左上原点）

extension NSView {
    /// NSView 坐标（左下原点）转 SwiftUI 左上原点
    func flippedPoint(_ event: NSEvent) -> CGPoint {
        let p = convert(event.locationInWindow, from: nil)
        return CGPoint(x: p.x, y: bounds.height - p.y)
    }

    /// 本地坐标点（左下原点）转 SwiftUI 左上原点
    func flippedPoint(_ point: NSPoint) -> CGPoint {
        CGPoint(x: point.x, y: bounds.height - point.y)
    }

    /// 由两个点构建标准矩形（自动归一化 min/max）
    func rect(from a: CGPoint, to b: CGPoint) -> CGRect {
        CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
               width: abs(b.x - a.x), height: abs(b.y - a.y))
    }
}

// ============================================================

// MARK: - 空节点占位区按钮动作

/// 空节点占位区按钮：上传 / 资产库（按钮在鼠标控制层之下，SwiftUI 命中不到，由本层命中后回调）
enum EmptyNodeButtonAction {
    case upload
    case library
}

// ============================================================

// MARK: - 右键菜单 + 引用该节点生成面板

extension CanvasView {
    // MARK: - 右键菜单（鼠标在哪面板在哪）
    
    var contextMenu: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let nodeID = contextMenuNodeID, let node = store.nodes.first(where: { $0.id == nodeID }) {
                // 节点操作菜单（右键点在节点上）
                if contextMenuFromOutline {
                    // 大纲内节点右键：只显示「改名」「复制」（大纲节点操作）
                    contextMenuItem(icon: "pencil", title: "改名") {
                        beginOutlineRename(node)
                        showContextMenu = false
                        debugLog("大纲视图：开始重命名节点「\(node.title)」")
                    }
                    contextMenuItem(icon: "plus.square.on.square", title: "复制") {
                        duplicateNodes([node.id])
                        showContextMenu = false
                        debugLog("大纲视图：复制节点「\(node.title)」")
                    }
                } else {
                    // 画布节点：原有操作 + 改名
                    contextMenuItem(icon: "pencil", title: "改名") {
                        beginOutlineRename(node)
                        showContextMenu = false
                        debugLog("画布节点：开始重命名节点「\(node.title)」")
                    }
                    
                    contextMenuItem(icon: "doc.on.doc", title: "复制文本", isEnabled: node.type == .text) {
                        copyNodeText(node)
                        showContextMenu = false
                        debugLog("右键菜单：复制文本「\(node.title)」")
                    }
                    
                    contextMenuItem(icon: "plus.square.on.square", title: "创建副本") {
                        duplicateNodes(targetNodeIDs(for: nodeID))
                        showContextMenu = false
                        debugLog("右键菜单：创建副本「\(node.title)」")
                    }
                    
                    if node.groupID != nil {
                        contextMenuItem(icon: "square.on.square.dashed", title: "从组内移出") {
                            removeNodeFromGroup(node)
                            showContextMenu = false
                            debugLog("右键菜单：从组内移出「\(node.title)」")
                        }
                    }
                    
                    Divider()
                        .padding(.vertical, 6)
                    
                    contextMenuItem(icon: "trash", title: "删除") {
                        deleteNodes(targetNodeIDs(for: nodeID))
                        showContextMenu = false
                        debugLog("右键菜单：删除节点「\(node.title)」")
                    }
                }
            } else if showAddNodePanel {
                // 添加节点子面板
                addNodePanel {
                    showContextMenu = false
                    showAddNodePanel = false
                }
            } else {
                // 主菜单
                contextMenuItem(icon: "square.and.arrow.up", title: "上传") {
                    showContextMenu = false
                    debugLog("右键菜单：上传文件")
                    handleUploadFiles()
                }
                
                contextMenuItem(icon: "plus", title: "添加节点", showArrow: true) {
                    debugLog("右键菜单：打开添加节点面板")
                    withAnimation { showAddNodePanel = true }
                }
            }
        }
        .padding(.vertical, 6)
        .frame(width: 180)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.97))
                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        )
    }
    
    // MARK: - 添加节点面板（右键菜单与左侧工具栏共用，onClose 由调用方决定关闭哪个面板）
    
    func addNodePanel(onClose: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("添加节点")
                .font(.headline)
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, 8)
            
            // 类型节点
            Text("类型节点")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.horizontal, 14)
                .padding(.bottom, 4)
            
            ForEach(NodeType.allCases) { type in
                contextMenuItem(icon: type.icon, title: type.rawValue) {
                    debugLog("添加节点：\(type.rawValue)")
                    addNode(type)
                    onClose()
                }
            }
            
            Divider()
                .padding(.vertical, 6)
            
            // 资源操作
            Text("资源操作")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.horizontal, 14)
                .padding(.bottom, 4)
            
            // 添加资源（暂不可用）
            HStack(spacing: 10) {
                Image(systemName: "plus.circle")
                    .font(.system(size: 14))
                    .frame(width: 18)
                Text("添加资源")
                    .font(.subheadline)
            }
            .foregroundColor(.secondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            
            contextMenuItem(icon: "square.and.arrow.up", title: "上传") {
                onClose()
                debugLog("添加节点面板：上传文件")
                handleUploadFiles()
            }
        }
        .padding(.vertical, 6)
        .frame(width: 180)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.97))
                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        )
    }
    
    func contextMenuItem(icon: String, title: String, showArrow: Bool = false, isEnabled: Bool = true, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 14))
                    .frame(width: 18)
                Text(title)
                    .font(.subheadline)
                Spacer()
                if showArrow {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                }
            }
            .foregroundColor(isEnabled ? .primary : .secondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
    }
    
    // MARK: - 引用该节点生成面板（拖拽连线在空白处松手时弹出）
    
    var generatePanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("引用该节点生成")
                .font(.headline)
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, 8)
            
            ForEach([NodeType.image, .video, .audio, .character, .scene]) { type in
                contextMenuItem(icon: type.icon, title: type.rawValue) {
                    generateNode(of: type)
                    showGeneratePanel = false
                }
            }
        }
        .padding(.vertical, 6)
        .frame(width: 180)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.97))
                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        )
    }
}

// ============================================================

// MARK: - 资产库框选层（覆盖在资产网格视口上，仅拦截空白处左键做框选；命中资产项则穿透给 draggable，拖拽优先）

struct AssetSelectionView: NSViewRepresentable {
    var assetFrames: [UUID: CGRect]
    var onSelectionChanged: (CGRect) -> Void
    var onSelectionEnded: ([UUID]) -> Void

    func makeNSView(context: Context) -> AssetSelectionNSView {
        let view = AssetSelectionNSView()
        view.assetFrames = assetFrames
        view.onSelectionChanged = onSelectionChanged
        view.onSelectionEnded = onSelectionEnded
        return view
    }

    func updateNSView(_ nsView: AssetSelectionNSView, context: Context) {
        nsView.assetFrames = assetFrames
        nsView.onSelectionChanged = onSelectionChanged
        nsView.onSelectionEnded = onSelectionEnded
    }
}

final class AssetSelectionNSView: NSView {
    var assetFrames: [UUID: CGRect] = [:]
    var onSelectionChanged: ((CGRect) -> Void)?
    var onSelectionEnded: (([UUID]) -> Void)?

    private var dragStart: CGPoint?
    private var currentRect: CGRect?

    override func hitTest(_ point: NSPoint) -> NSView? {
        // 资产 frame 尚未上报（为空）时一律不拦截，保证拖拽到画布正常
        guard !assetFrames.isEmpty else { return nil }
        // 命中资产项 → 返回 nil 穿透给下层 draggable（拖拽优先于框选）
        // assetFrames 为 SwiftUI 左上原点坐标，需先翻转本地点再比较
        let flipped = flippedPoint(point)
        for frame in assetFrames.values {
            if frame.contains(flipped) {
                return nil
            }
        }
        // 空白处 → 拦截，处理框选
        return super.hitTest(point)
    }

    override func mouseDown(with event: NSEvent) {
        let p = flippedPoint(event)
        dragStart = p
        currentRect = CGRect(origin: p, size: .zero)
        onSelectionChanged?(currentRect!)
    }

    override func mouseDragged(with event: NSEvent) {
        guard let start = dragStart else { return }
        let p = flippedPoint(event)
        let rect = rect(from: start, to: p)
        currentRect = rect
        onSelectionChanged?(rect)
    }

    override func mouseUp(with event: NSEvent) {
        guard let start = dragStart else { return }
        let p = flippedPoint(event)
        let rect = rect(from: start, to: p)
        dragStart = nil
        currentRect = nil
        let ids = assetFrames.filter { $0.value.intersects(rect) }.map { $0.key }
        onSelectionEnded?(ids)
    }
}

// ============================================================

// MARK: - 鼠标控制层（中键拖动移动 / 滚轮上下滚动 / Command+滚轮缩放）

struct MouseControlView: NSViewRepresentable {
    @Binding var offset: CGPoint
    @Binding var zoom: Double
    var gridSize: Double
    var snapToGrid: Bool
    /// 画布视口尺寸（Command+滚轮缩放无选中内容时的锚点取视口中心）
    var viewSize: CGSize = .zero
    /// 面板打开时底层画布完全隔离（悬停/点击/拖拽/滚轮/右键全部失效）
    var isIsolated: Bool = false
    var nodes: [CanvasNode] = []
    var nodeSizes: [UUID: CGSize] = [:]
    var selectedNodeIDs: Set<UUID> = []
    var connections: [NodeConnection] = []
    var selectedConnectionID: UUID? = nil
    /// 当前正在播放的节点 id（nil = 无节点播放；进度条拖动命中仅对播放中的节点生效）
    var playingNodeID: UUID? = nil
    var onDragEnded: (() -> Void)?
    var onRightClick: ((CGPoint, UUID?) -> Void)?
    var onNodeDragStart: (([UUID: CGPoint]) -> Void)?
    var onNodeDrag: ((UUID, CGPoint) -> Void)?
    var onNodeDragEnded: (() -> Void)?
    var onSelectionChanged: ((CGRect) -> Void)?
    var onSelectionEnded: (([UUID]) -> Void)?
    var onNodeClick: ((UUID) -> Void)?
    /// 单击组盒空白（未拖动）：选中整组（参数为组内节点 id 列表）
    var onSelectGroup: (([UUID]) -> Void)?
    var onConnectionClick: ((UUID) -> Void)?
    var onCutConnection: ((UUID) -> Void)?
    var onAlignmentGuides: (([AlignmentGuide]) -> Void)?
    var onHoverNode: ((UUID?, ConnectSide?, CGPoint?) -> Void)?
    /// 组盒加号悬停（整组选中时鼠标靠近组盒左右加号；nil 表示未悬停组盒加号）
    var onHoverGroupPort: ((UUID?, ConnectSide?, CGPoint?) -> Void)?
    var onPortDragChanged: ((UUID, CGPoint, ConnectSide) -> Void)?
    var onPortDragEnded: ((UUID, CGPoint, ConnectSide) -> Void)?
    /// 空节点占位区按钮点击（上传 / 资产库）
    var onEmptyNodeButtonClick: ((UUID, EmptyNodeButtonAction) -> Void)?
    /// 空节点占位区按钮悬停（上传 / 资产库；nil 表示未悬停按钮）
    var onEmptyNodeButtonHover: ((UUID?, EmptyNodeButtonAction?) -> Void)?
    /// 外部媒体文件拖放（Finder 拖入画布）：回调文件 URL 列表与落点（视图坐标，左上原点）
    var onExternalDrop: (([URL], CGPoint) -> Void)?
    /// 播放按钮点击（音频/视频节点悬停时；点击切换播放/停止）
    var onPlayButtonClick: ((UUID) -> Void)?
    /// 历史徽章点击（有内容节点右下角；点击展开该节点往期内容面板）
    var onHistoryBadgeClick: ((UUID) -> Void)?
    /// 尾帧开关点击（视频有内容节点信息栏；点击切换 on/off）
    var onTailFrameSwitchClick: ((UUID) -> Void)?
    /// 播放进度条拖动 seek（播放中的视频/音频节点；参数为节点 id 与 0...1 进度比例）
    var onProgressSeek: ((UUID, Double) -> Void)?
    
    func makeNSView(context: Context) -> MouseControlNSView {
        let view = MouseControlNSView()
        view.offsetBinding = $offset
        view.zoomBinding = $zoom
        view.gridSize = gridSize
        view.snapToGrid = snapToGrid
        view.viewSize = viewSize
        view.isIsolated = isIsolated
        view.nodes = nodes
        view.nodeSizes = nodeSizes
        view.selectedNodeIDs = selectedNodeIDs
        view.connections = connections
        view.selectedConnectionID = selectedConnectionID
        view.playingNodeID = playingNodeID
        view.onDragEnded = onDragEnded
        view.onRightClick = onRightClick
        view.onNodeDragStart = onNodeDragStart
        view.onNodeDrag = onNodeDrag
        view.onNodeDragEnded = onNodeDragEnded
        view.onSelectionChanged = onSelectionChanged
        view.onSelectionEnded = onSelectionEnded
        view.onNodeClick = onNodeClick
        view.onSelectGroup = onSelectGroup
        view.onConnectionClick = onConnectionClick
        view.onCutConnection = onCutConnection
        view.onAlignmentGuides = onAlignmentGuides
        view.onHoverNode = onHoverNode
        view.onHoverGroupPort = onHoverGroupPort
        view.onPortDragChanged = onPortDragChanged
        view.onPortDragEnded = onPortDragEnded
        view.onEmptyNodeButtonClick = onEmptyNodeButtonClick
        view.onEmptyNodeButtonHover = onEmptyNodeButtonHover
        view.onExternalDrop = onExternalDrop
        view.onPlayButtonClick = onPlayButtonClick
        view.onHistoryBadgeClick = onHistoryBadgeClick
        view.onTailFrameSwitchClick = onTailFrameSwitchClick
        view.onProgressSeek = onProgressSeek
        // 注册外部文件拖放类型（Finder 拖入媒体文件；SwiftUI dropDestination 被本层拦截，改由 AppKit 原生接收）
        view.registerForDraggedTypes([.fileURL])
        return view
    }
    
    func updateNSView(_ nsView: MouseControlNSView, context: Context) {
        nsView.offsetBinding = $offset
        nsView.zoomBinding = $zoom
        nsView.gridSize = gridSize
        nsView.snapToGrid = snapToGrid
        nsView.viewSize = viewSize
        // 面板打开（进入隔离）时清除画布悬停状态，避免残留高亮
        if nsView.isIsolated != isIsolated {
            nsView.isIsolated = isIsolated
            if isIsolated {
                nsView.onHoverNode?(nil, nil, nil)
                nsView.onEmptyNodeButtonHover?(nil, nil)
            }
        }
        // ⑥ 大数组内容没变时跳过赋值（缓冲区指针 O(1) 比较）；字典/集合 COW 赋值本身 O(1) 直接同步
        nsView.syncNodes(nodes)
        nsView.nodeSizes = nodeSizes
        nsView.selectedNodeIDs = selectedNodeIDs
        nsView.syncConnections(connections)
        if nsView.selectedConnectionID != selectedConnectionID {
            nsView.selectedConnectionID = selectedConnectionID
        }
        nsView.playingNodeID = playingNodeID
        nsView.onDragEnded = onDragEnded
        nsView.onRightClick = onRightClick
        nsView.onNodeDragStart = onNodeDragStart
        nsView.onNodeDrag = onNodeDrag
        nsView.onNodeDragEnded = onNodeDragEnded
        nsView.onSelectionChanged = onSelectionChanged
        nsView.onSelectionEnded = onSelectionEnded
        nsView.onNodeClick = onNodeClick
        nsView.onSelectGroup = onSelectGroup
        nsView.onConnectionClick = onConnectionClick
        nsView.onCutConnection = onCutConnection
        nsView.onAlignmentGuides = onAlignmentGuides
        nsView.onHoverNode = onHoverNode
        nsView.onHoverGroupPort = onHoverGroupPort
        nsView.onPortDragChanged = onPortDragChanged
        nsView.onPortDragEnded = onPortDragEnded
        nsView.onEmptyNodeButtonClick = onEmptyNodeButtonClick
        nsView.onEmptyNodeButtonHover = onEmptyNodeButtonHover
        nsView.onExternalDrop = onExternalDrop
        nsView.onPlayButtonClick = onPlayButtonClick
        nsView.onHistoryBadgeClick = onHistoryBadgeClick
        nsView.onTailFrameSwitchClick = onTailFrameSwitchClick
        nsView.onProgressSeek = onProgressSeek
    }
}

final class MouseControlNSView: NSView {
    var offsetBinding: Binding<CGPoint>?
    var zoomBinding: Binding<Double>?
    var gridSize: Double = 40
    var snapToGrid: Bool = true
    /// 画布视口尺寸（Command+滚轮缩放无选中内容时的锚点取视口中心）
    var viewSize: CGSize = .zero
    var onDragEnded: (() -> Void)?
    var onRightClick: ((CGPoint, UUID?) -> Void)?
    /// 播放进度条拖动 seek 回调（节点 id + 0...1 进度比例）
    var onProgressSeek: ((UUID, Double) -> Void)?
    /// 平移合并（①滚动优化）：滚动/中键拖动每事件只累积 delta，下一 runloop tick 合并写一次 offset，
    /// 避免 120Hz 触控板事件直写 @Published → 每事件触发画布全量 body 重建
    private var pendingOffsetDelta: CGPoint = .zero
    private var offsetFlushScheduled = false
    
    /// 缩放合并（同①机制，扩展到 Command+滚轮）：每事件只累积 factor，下一 runloop tick 合并写一次 zoom+offset，
    /// 避免 120Hz 触控板缩放事件直写 → 每事件触发内容层位图全量重采样（缩放卡顿的直接来源）
    private var pendingZoomFactor: CGFloat = 1.0
    private var zoomFlushScheduled = false
    
    // 节点交互
    var nodes: [CanvasNode] = []
    var nodeSizes: [UUID: CGSize] = [:]
    var selectedNodeIDs: Set<UUID> = []
    var connections: [NodeConnection] = []
    var selectedConnectionID: UUID? = nil
    /// 当前正在播放的节点 id（nil = 无节点播放；进度条拖动命中仅对播放中的节点生效）
    var playingNodeID: UUID?
    
    // ⑥ 大数组同步缓存：COW 缓冲区指针 O(1) 比较，数组内容没变就不重新赋值，
    // 避免画布平移/缩放每帧触发 updateNSView 重复同步（空数组用哨兵指针 1）
    private var syncedNodesPtr = UnsafeRawPointer(bitPattern: 1)!
    private var syncedConnectionsPtr = UnsafeRawPointer(bitPattern: 1)!

    private func bufferID<T>(_ c: UnsafeBufferPointer<T>) -> UnsafeRawPointer {
        UnsafeRawPointer(c.baseAddress) ?? UnsafeRawPointer(bitPattern: 1)!
    }
    func syncNodes(_ newValue: [CanvasNode]) {
        let ptr = newValue.withUnsafeBufferPointer { bufferID($0) }
        if ptr != syncedNodesPtr { syncedNodesPtr = ptr; nodes = newValue }
    }
    func syncConnections(_ newValue: [NodeConnection]) {
        let ptr = newValue.withUnsafeBufferPointer { bufferID($0) }
        if ptr != syncedConnectionsPtr { syncedConnectionsPtr = ptr; connections = newValue }
    }
    var onNodeDragStart: (([UUID: CGPoint]) -> Void)?
    var onNodeDrag: ((UUID, CGPoint) -> Void)?
    var onNodeDragEnded: (() -> Void)?
    var onSelectionChanged: ((CGRect) -> Void)?
    var onSelectionEnded: (([UUID]) -> Void)?
    var onNodeClick: ((UUID) -> Void)?
    /// 单击组盒空白（未拖动）：选中整组（参数为组内节点 id 列表）
    var onSelectGroup: (([UUID]) -> Void)?
    var onConnectionClick: ((UUID) -> Void)?
    var onCutConnection: ((UUID) -> Void)?
    var onAlignmentGuides: (([AlignmentGuide]) -> Void)?
    var onHoverNode: ((UUID?, ConnectSide?, CGPoint?) -> Void)?
    // 组盒加号悬停（整组选中时组盒两侧加号存在；nil 表示未悬停；与拖拽目标态共用 hoveredGroupPort* 状态）
    var onHoverGroupPort: ((UUID?, ConnectSide?, CGPoint?) -> Void)?
    // 加号连线拖拽（加号在节点外侧，SwiftUI 命中不到，由本层统一处理）
    var onPortDragChanged: ((UUID, CGPoint, ConnectSide) -> Void)?
    var onPortDragEnded: ((UUID, CGPoint, ConnectSide) -> Void)?
    // 空节点占位区按钮点击（上传 / 资产库）
    var onEmptyNodeButtonClick: ((UUID, EmptyNodeButtonAction) -> Void)?
    // 空节点占位区按钮悬停（上传 / 资产库；nil 表示未悬停按钮）
    var onEmptyNodeButtonHover: ((UUID?, EmptyNodeButtonAction?) -> Void)?
    // 外部媒体文件拖放（Finder 拖入画布）：回调文件 URL 列表与落点（视图坐标，左上原点）
    var onExternalDrop: (([URL], CGPoint) -> Void)?
    // 播放按钮点击（音频/视频节点悬停时；点击切换播放/停止）
    var onPlayButtonClick: ((UUID) -> Void)?
    // 历史徽章点击（有内容节点右下角；点击展开该节点往期内容面板）
    var onHistoryBadgeClick: ((UUID) -> Void)?
    // 尾帧开关点击（视频有内容节点信息栏；点击切换 on/off）
    var onTailFrameSwitchClick: ((UUID) -> Void)?
    
    /// 面板打开时底层画布完全隔离：所有鼠标事件入口直接 return
    var isIsolated: Bool = false
    
    // 节点拖拽状态
    private var draggingNodeID: UUID?
    private var dragStartScreen: CGPoint = .zero
    private var dragStartPositions: [(UUID, CGPoint)] = []
    private var hasDragged = false
    // 组内空白拖整组：mouseUp 单击时跳过选中（避免误触发 onNodeClick）
    private var dragFromGroupBlank = false
    // 播放进度条拖动状态：按下/拖动期间持续 seek，松手结束
    private var seekingProgressNodeID: UUID?
    
    // 加号连线拖拽状态
    private var portDragNodeID: UUID?
    private var portDragSide: ConnectSide?
    
    // 框选状态
    private var isSelecting = false
    private var selectionStart: CGPoint = .zero
    
    // 中键拖拽状态
    private var isMiddleDragging = false
    private var lastDragLocation: NSPoint = .zero
    
    // 悬停检测 tracking area
    private var hoverTrackingArea: NSTrackingArea?
    
    // 加号球容器半径（不可见命中区域，包裹加号；与 NodeView 共用常量）
    let ballRadius = nodeBallRadius
    // 加号尺寸（与 NodeView 保持一致）
    let portSize = nodePortSize
    
    override var acceptsFirstResponder: Bool { true }
    
    // 更新 tracking area（视图尺寸变化时重建）
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area = hoverTrackingArea {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.activeInKeyWindow, .mouseMoved, .mouseEnteredAndExited, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
        hoverTrackingArea = area
    }
    
    // 鼠标移动：检测悬停节点或两侧球区域，回传 (nodeID, side, 加号偏移)
    override func mouseMoved(with event: NSEvent) {
        // 面板打开：底层画布完全隔离，不响应悬停
        if isIsolated { return }
        let p = flippedPoint(event)
        // 0. 空节点占位区按钮：按钮独立高亮 + 节点悬停（容器高亮）
        if let (nodeID, action) = hitTestEmptyNodeButton(at: p) {
            debugLog("鼠标移动：命中空节点按钮(\(action == .upload ? "上传" : "资产库"))", throttle: true)
            onHoverNode?(nodeID, nil, .zero)
            onEmptyNodeButtonHover?(nodeID, action)
            return
        }
        // 1. 节点矩形内：加号在球中心
        if let nodeID = hitTestNode(at: p) {
            debugLog("鼠标移动：命中节点", throttle: true)
            onHoverNode?(nodeID, nil, .zero)
            onEmptyNodeButtonHover?(nil, nil)
            return
        }
        // 2. 左右球区域内：对应侧加号跟随鼠标（吸附）
        if let (nodeID, side, offset) = hitTestBall(at: p) {
            debugLog("鼠标移动：命中加号球(\(side == .left ? "左" : "右"))", throttle: true)
            onHoverNode?(nodeID, side, offset)
            onHoverGroupPort?(nil, nil, nil)
            onEmptyNodeButtonHover?(nil, nil)
            return
        }
        // 2.5 组盒左右加号球：仅整组选中时组盒加号可见，悬停命中要求整组选中（与渲染层显示条件一致），
        // 亮出对应侧加号并跟随鼠标（点击选中 → 出现加号 → 悬停吸附特效）
        if let (gid, side, offset) = hitTestGroupPort(at: p) {
            debugLog("鼠标移动：命中组盒加号球(\(side == .left ? "左" : "右"))", throttle: true)
            onHoverGroupPort?(gid, side, offset)
            onHoverNode?(nil, nil, nil)
            onEmptyNodeButtonHover?(nil, nil)
            return
        }
        // 3. 空白处：清除悬停
        debugLog("鼠标移动：空白", throttle: true)
        onHoverNode?(nil, nil, nil)
        onHoverGroupPort?(nil, nil, nil)
        onEmptyNodeButtonHover?(nil, nil)
    }
    
    // 鼠标离开视图：清除悬停
    override func mouseExited(with event: NSEvent) {
        onHoverNode?(nil, nil, nil)
        onHoverGroupPort?(nil, nil, nil)
        onEmptyNodeButtonHover?(nil, nil)
    }
    
    // MARK: - 外部文件拖放（Finder 拖入媒体文件）
    
    // 拖入：仅接受文件 URL 类型
    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        guard !isIsolated else { return [] }
        let hasFile = sender.draggingPasteboard.types?.contains(.fileURL) ?? false
        return hasFile ? .copy : []
    }
    
    // 松手：提取文件 URL 列表与落点（视图坐标，左上原点），回调给 SwiftUI 层导入
    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard !isIsolated else { return false }
        let urls = sender.draggingPasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        guard !urls.isEmpty else { return false }
        let p = flippedPoint(sender.draggingLocation)
        debugLog("外部文件拖放：收到 \(urls.count) 个文件，落点 \(p)")
        onExternalDrop?(urls, p)
        return true
    }
    
    override func mouseDown(with event: NSEvent) {
        // 面板打开：底层画布完全隔离，不响应点击
        if isIsolated { return }
        
        if event.buttonNumber == 0 {
            let p = flippedPoint(event)
            // 裁断按钮命中 → 移除选中的连线（按钮在鼠标控制层之上，SwiftUI 命中不到，由本层处理）
            if let connID = selectedConnectionID,
               let connection = connections.first(where: { $0.id == connID }),
               let from = nodes.first(where: { $0.id == connection.fromID }),
               let to = nodes.first(where: { $0.id == connection.toID }) {
                let mid = connectionMidPoint(from: from, to: to, nodeSizes: nodeSizes, zoom: currentZoom, offset: currentOffset)
                if distance(from: p, to: mid) <= 30 {
                    debugLog("左键点击：裁断连线")
                    onCutConnection?(connID)
                    return
                }
            }
            // 加号球容器命中 → 启动连线拖拽（加号在节点外侧，SwiftUI 命中不到，由本层处理）
            if let (nodeID, side, _) = hitTestBall(at: p) {
                debugLog("左键点击：命中加号球(\(side == .left ? "左" : "右")) 启动连线拖拽")
                portDragNodeID = nodeID
                portDragSide = side
                onPortDragChanged?(nodeID, p, side)
                return
            }
            // 组盒加号球容器命中（整组选中时组盒两侧加号存在）→ 启动组连线拖拽（fromID 用组 ID，画布层按组批量建连）
            if let (gid, side, _) = hitTestGroupPort(at: p) {
                debugLog("左键点击：命中组加号球(\(side == .left ? "左" : "右")) 启动组连线拖拽")
                portDragNodeID = gid
                portDragSide = side
                onPortDragChanged?(gid, p, side)
                return
            }
            // 尾帧开关命中 → 切换 on/off（视频有内容节点信息栏右侧；开关在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            if let nodeID = hitTestTailFrameSwitch(at: p) {
                debugLog("左键点击：命中尾帧开关")
                onTailFrameSwitchClick?(nodeID)
                return
            }
            // 播放按钮命中 → 切换播放/停止（按钮在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            if let nodeID = hitTestPlayButton(at: p) {
                debugLog("左键点击：命中播放按钮")
                onPlayButtonClick?(nodeID)
                return
            }
            // 历史徽章命中 → 展开该节点往期内容面板（徽章在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            if let nodeID = hitTestHistoryBadge(at: p) {
                debugLog("左键点击：命中历史徽章")
                onHistoryBadgeClick?(nodeID)
                return
            }
            // 生成状态覆盖层取消按钮命中 → 取消该节点活跃任务（按钮在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            // 与队列面板取消共用 GenerationQueue.shared.cancel：任务移除后覆盖层/队列面板同步消失，管线经 cancelToken 中断
            if let nodeID = hitTestGenerationCancelButton(at: p) {
                debugLog("左键点击：命中生成状态取消按钮")
                if let taskID = GenerationQueue.shared.taskID(for: nodeID) {
                    GenerationQueue.shared.cancel(taskID)
                }
                return
            }
            // 播放进度条命中 → 开始拖动跳转进度（仅播放中的视频/音频节点；进度条在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            if let (nodeID, ratio) = hitTestProgressBar(at: p) {
                debugLog("左键点击：命中播放进度条")
                seekingProgressNodeID = nodeID
                onProgressSeek?(nodeID, ratio)
                return
            }
            // 空节点占位区按钮命中 → 回调上传/资产库（按钮在鼠标控制层之下，SwiftUI 命中不到，由本层处理）
            if let (nodeID, action) = hitTestEmptyNodeButton(at: p) {
                debugLog("左键点击：命中空节点按钮(\(action == .upload ? "上传" : "资产库"))")
                onEmptyNodeButtonClick?(nodeID, action)
                return
            }
            // 命中节点 → 拖拽节点；空白处 → 框选
            if let nodeID = hitTestNode(at: p) {
                debugLog("左键点击：命中节点 开始拖拽")
                draggingNodeID = nodeID
                dragStartScreen = p
                hasDragged = false
                if selectedNodeIDs.contains(nodeID) {
                    // 节点已在选中组：整组一起移动
                    dragStartPositions = nodes.filter { selectedNodeIDs.contains($0.id) }.map { ($0.id, $0.position) }
                } else {
                    // 未选中：只移动当前节点
                    if let node = nodes.first(where: { $0.id == nodeID }) {
                        dragStartPositions = [(nodeID, node.position)]
                    }
                }
                onNodeDragStart?(Dictionary(uniqueKeysWithValues: dragStartPositions))   // 记录拖拽前状态（用于撤销）+ 拖动起始位置（临时位移方案）
            } else if let groupNodes = hitTestGroupBlank(at: p) {
                // 组盒内空白（非节点区域）：拖动整个组，逻辑与已选中整组移动一致
                debugLog("左键点击：命中组空白 开始拖动整组(\(groupNodes.count)个节点)")
                dragFromGroupBlank = true
                draggingNodeID = groupNodes.first?.id
                dragStartScreen = p
                hasDragged = false
                dragStartPositions = groupNodes.map { ($0.id, $0.position) }
                onNodeDragStart?(Dictionary(uniqueKeysWithValues: dragStartPositions))
            } else if let connID = hitTestConnection(at: p) {
                // 命中连线 → 选中该连线（虚线流动 + 中心裁断按钮）
                debugLog("左键点击：命中连线 选中")
                onConnectionClick?(connID)
            } else {
                debugLog("左键点击：空白 开始框选")
                isSelecting = true
                selectionStart = p
                onSelectionChanged?(CGRect(x: p.x, y: p.y, width: 0, height: 0))
            }
        }
    }
    
    // 中键按下（AppKit 中中键走 otherMouse 系列事件）
    override func otherMouseDown(with event: NSEvent) {
        if isIsolated { return }
        if event.buttonNumber == 2 {
            debugLog("中键按下：开始移动画布")
            isMiddleDragging = true
            lastDragLocation = event.locationInWindow
        }
    }
    
    // 右键按下：回调鼠标位置（转为 SwiftUI 左上原点坐标）与命中的节点 id
    override func rightMouseDown(with event: NSEvent) {
        if isIsolated { return }
        let pointInView = convert(event.locationInWindow, from: nil)
        let flipped = flippedPoint(pointInView)
        let hitNodeID = hitTestNode(at: flipped)
        debugLog("右键点击：\(hitNodeID != nil ? "命中节点" : "空白")")
        onRightClick?(flipped, hitNodeID)
    }
    
    override func otherMouseDragged(with event: NSEvent) {
        guard !isIsolated, isMiddleDragging else { return }
        let current = event.locationInWindow
        let dx = current.x - lastDragLocation.x
        let dy = current.y - lastDragLocation.y
        if var offset = offsetBinding?.wrappedValue {
            offset.x += dx
            offset.y -= dy
            offsetBinding?.wrappedValue = offset
        }
        lastDragLocation = current
    }
    
    override func otherMouseUp(with event: NSEvent) {
        if isIsolated { return }
        if event.buttonNumber == 2 {
            debugLog("中键释放：移动画布结束")
            isMiddleDragging = false
            onDragEnded?()
        }
    }
    
    override func mouseDragged(with event: NSEvent) {
        if isIsolated { return }
        let p = flippedPoint(event)
        if let nodeID = portDragNodeID, let side = portDragSide {
            // 加号连线拖拽：更新临时连线
            debugLog("左键拖拽：连线拖拽中(\(side == .left ? "左" : "右"))", throttle: true)
            onPortDragChanged?(nodeID, p, side)
            return
        }
        if let nodeID = seekingProgressNodeID {
            // 播放进度条拖动：实时跳转进度（鼠标 x → 0...1 比例）
            debugLog("左键拖拽：拖动播放进度", throttle: true)
            if let ratio = progressRatio(at: p, for: nodeID) {
                onProgressSeek?(nodeID, ratio)
            }
            return
        }
        if draggingNodeID != nil {
            // 拖拽移动节点（屏幕位移 → 画布位移），整组一起动
            debugLog("左键拖拽：移动节点", throttle: true)
            let zoom = currentZoom
            let offset = currentOffset
            let dx = (p.x - dragStartScreen.x) / zoom
            let dy = (p.y - dragStartScreen.y) / zoom
            if abs(p.x - dragStartScreen.x) > 3 || abs(p.y - dragStartScreen.y) > 3 {
                hasDragged = true
            }
            // 对齐吸附：开启网格吸附时，拖动节点对齐最近节点的中心/顶部/底部，并显示参考虚线
            var finalDx = dx
            var finalDy = dy
            var guides: [AlignmentGuide] = []
            if snapToGrid {
                let draggedIDs = Set(dragStartPositions.map { $0.0 })
                let threshold = 5.0 / zoom   // 屏幕 5pt 容差
                var xAdjusted = false
                var yAdjusted = false
                for (id, startPos) in dragStartPositions {
                    let pos = CGPoint(x: startPos.x + dx, y: startPos.y + dy)
                    let size = nodeSizes[id] ?? .zero
                    for other in nodes where !draggedIDs.contains(other.id) {
                        // ④ 粗筛：中心距离超过容差 + 节点半尺寸最大差异时不可能对齐，
                        // 跳过详细比较（平方距离比较，无 sqrt）；800 画布坐标远大于节点半宽/半高差
                        let coarseR = threshold + 800
                        let cdx = pos.x - other.position.x
                        let cdy = pos.y - other.position.y
                        if cdx * cdx + cdy * cdy > coarseR * coarseR { continue }
                        let otherSize = nodeSizes[other.id] ?? .zero
                        // 中心 x 对齐（垂直参考线）
                        if !xAdjusted, abs(pos.x - other.position.x) < threshold {
                            finalDx = other.position.x - startPos.x
                            xAdjusted = true
                            guides.append(AlignmentGuide(isVertical: true, position: other.position.x * zoom + offset.x))
                        }
                        // 顶部 y 对齐（水平参考线）
                        let top = pos.y - size.height / 2
                        let otherTop = other.position.y - otherSize.height / 2
                        if !yAdjusted, abs(top - otherTop) < threshold {
                            finalDy = otherTop - (startPos.y - size.height / 2)
                            yAdjusted = true
                            guides.append(AlignmentGuide(isVertical: false, position: otherTop * zoom + offset.y))
                        }
                        // 底部 y 对齐（水平参考线）
                        let bottom = pos.y + size.height / 2
                        let otherBottom = other.position.y + otherSize.height / 2
                        if !yAdjusted, abs(bottom - otherBottom) < threshold {
                            finalDy = otherBottom - (startPos.y + size.height / 2)
                            yAdjusted = true
                            guides.append(AlignmentGuide(isVertical: false, position: otherBottom * zoom + offset.y))
                        }
                        // 中心 y 对齐（水平参考线）
                        if !yAdjusted, abs(pos.y - other.position.y) < threshold {
                            finalDy = other.position.y - startPos.y
                            yAdjusted = true
                            guides.append(AlignmentGuide(isVertical: false, position: other.position.y * zoom + offset.y))
                        }
                    }
                }
            }
            onAlignmentGuides?(guides)
            for (id, startPos) in dragStartPositions {
                onNodeDrag?(id, CGPoint(x: startPos.x + finalDx, y: startPos.y + finalDy))
            }
        } else if isSelecting {
            // 更新框选矩形
            debugLog("左键拖拽：框选", throttle: true)
            onSelectionChanged?(rect(from: selectionStart, to: p))
        }
    }
    
    override func mouseUp(with event: NSEvent) {
        if isIsolated { return }
        if seekingProgressNodeID != nil {
            // 播放进度条拖动结束
            debugLog("左键释放：进度条拖动结束")
            seekingProgressNodeID = nil
            return
        }
        if let nodeID = portDragNodeID, let side = portDragSide {
            // 加号连线拖拽结束：完成连线
            debugLog("左键释放：连线拖拽结束")
            let p = flippedPoint(event)
            onPortDragEnded?(nodeID, p, side)
            portDragNodeID = nil
            portDragSide = nil
            return
        }
        if let nodeID = draggingNodeID {
            draggingNodeID = nil
            onAlignmentGuides?([])   // 释放鼠标，参考线消失
            if hasDragged {
                // 拖拽移动完成
                debugLog("左键释放：节点拖拽完成")
                onNodeDragEnded?()
            } else if dragFromGroupBlank {
                // 单点组盒空白（未拖动）：选中整组，让用户明确感知"组盒子被选中"
                debugLog("左键释放：单击选中整组")
                onSelectGroup?(dragStartPositions.map { $0.0 })
            } else {
                // 单点：选中该节点
                debugLog("左键释放：单击选中节点")
                onNodeClick?(nodeID)
            }
            dragFromGroupBlank = false
        } else if isSelecting {
            isSelecting = false
            let p = flippedPoint(event)
            let rect = rect(from: selectionStart, to: p)
            // 框选结束：选中与框选矩形相交的节点（屏幕坐标转画布坐标）
            let zoom = currentZoom
            let offset = currentOffset
            let canvasRect = CGRect(x: (rect.minX - offset.x) / zoom,
                                    y: (rect.minY - offset.y) / zoom,
                                    width: rect.width / zoom,
                                    height: rect.height / zoom)
            // 接触即选中：节点矩形与框选矩形相交
            let selected = nodes.filter { node in
                let size = nodeSizes[node.id] ?? .zero
                let nodeRect = CGRect(x: node.position.x - size.width / 2,
                                      y: node.position.y - size.height / 2,
                                      width: size.width,
                                      height: size.height)
                return canvasRect.intersects(nodeRect)
            }.map { $0.id }
            debugLog("左键释放：框选结束 命中\(selected.count)个节点")
            onSelectionEnded?(selected)
        }
    }
    
    // 命中检测：屏幕坐标是否落在某条连线上（贝塞尔曲线采样求最近距离）
    // 返回命中的连线 id；节点命中优先，故连线端点贴节点边缘不会误触发
    private func hitTestConnection(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let threshold: CGFloat = 8   // 命中容差（线宽 2.5 + 余量）
        for connection in connections {
            guard let from = nodes.first(where: { $0.id == connection.fromID }),
                  let to = nodes.first(where: { $0.id == connection.toID }) else { continue }
            let start = portPosition(of: from, side: .right, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
            let end = portPosition(of: to, side: .left, nodeSizes: nodeSizes, zoom: zoom, offset: offset)
            // ⑧ 粗筛：贝塞尔曲线不会超出两端点包围盒，鼠标点远离包围盒直接跳过，避免每条连线都做 40 段采样
            let minX = min(start.x, end.x) - threshold
            let minY = min(start.y, end.y) - threshold
            let bbox = CGRect(x: minX, y: minY,
                              width: abs(end.x - start.x) + threshold * 2,
                              height: abs(end.y - start.y) + threshold * 2)
            guard bbox.contains(screenPos) else { continue }
            let midX = (start.x + end.x) / 2
            let c1 = CGPoint(x: midX, y: start.y)
            let c2 = CGPoint(x: midX, y: end.y)
            // 采样 40 段，求点到曲线最近距离
            var minDist = CGFloat.greatestFiniteMagnitude
            var prev = start
            for i in 1...40 {
                let t = CGFloat(i) / 40
                let mt = 1 - t
                let mt2 = mt * mt
                let t2 = t * t
                let x = mt2 * mt * start.x + 3 * mt2 * t * c1.x + 3 * mt * t2 * c2.x + t2 * t * end.x
                let y = mt2 * mt * start.y + 3 * mt2 * t * c1.y + 3 * mt * t2 * c2.y + t2 * t * end.y
                let p = CGPoint(x: x, y: y)
                // 点到线段 prev-p 的距离
                let dx = p.x - prev.x
                let dy = p.y - prev.y
                let lenSq = dx * dx + dy * dy
                var proj: CGFloat = 0
                if lenSq > 0 {
                    proj = max(0, min(1, ((screenPos.x - prev.x) * dx + (screenPos.y - prev.y) * dy) / lenSq))
                }
                let px = prev.x + proj * dx
                let py = prev.y + proj * dy
                let ddx = screenPos.x - px
                let ddy = screenPos.y - py
                let dist = sqrt(ddx * ddx + ddy * ddy)
                if dist < minDist { minDist = dist }
                prev = p
            }
            if minDist <= threshold {
                return connection.id
            }
        }
        return nil
    }
    
    override func scrollWheel(with event: NSEvent) {
        if isIsolated { return }
        if event.modifierFlags.contains(.command) {
            // Command + 滚动 = 放大缩小（以视口中心为缩放锚点；delta 累积合并，见 accumulateZoom）
            let delta = event.scrollingDeltaY
            guard delta != 0 else { return }
            accumulateZoom(factor: delta > 0 ? 1.1 : 0.9)
        } else if isMiddleDragging || event.buttonNumber == 2 {
            // 中键拖动 = 移动画布（delta 累积合并，见 accumulatePan）
            let dx = event.scrollingDeltaX
            let dy = event.scrollingDeltaY
            accumulatePan(dx: dx, dy: -dy)
        } else {
            // 普通滚动 = 上下滚动（触控板/鼠标滚轮加速，与创作横向滚动共用全局常量）
            var dy = event.scrollingDeltaY
            if event.hasPreciseScrollingDeltas {
                dy *= wheelScrollAccelerationPrecise
            } else {
                dy *= wheelScrollAccelerationWheel
            }
            accumulatePan(dx: 0, dy: -dy)
        }
    }
    
    /// 累积一次平移 delta；同一 runloop tick 内多次滚动事件合并为一次 offset 写入（首事件排 async 块，后续只累加）
    private func accumulatePan(dx: CGFloat, dy: CGFloat) {
        guard offsetBinding != nil else { return }
        pendingOffsetDelta.x += dx
        pendingOffsetDelta.y += dy
        guard !offsetFlushScheduled else { return }
        offsetFlushScheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.offsetFlushScheduled = false
            let delta = self.pendingOffsetDelta
            self.pendingOffsetDelta = .zero
            self.applyPendingOffset(delta)
        }
    }
    
    /// 将累积的平移 delta 合并写入 offset（同一 runloop tick 内只写一次）
    private func applyPendingOffset(_ delta: CGPoint) {
        guard delta != .zero else { return }
        guard var offset = offsetBinding?.wrappedValue else { return }
        offset.x += delta.x
        offset.y += delta.y
        offsetBinding?.wrappedValue = offset
    }
    
    /// 累积一次缩放 factor；同一 runloop tick 内多次缩放事件合并为一次 zoom+offset 写入
    /// （以视口中心为缩放锚点，缩放前后视口中心屏幕位置不变：offset = 锚点屏幕位置 - 锚点画布坐标 * newZoom，
    /// 避免只改 zoom 导致缩放中心漂移；合并窗口内 offset 不被缩放事件修改，锚点反推与逐事件执行一致，无漂移）
    private func accumulateZoom(factor: CGFloat) {
        pendingZoomFactor *= factor
        guard !zoomFlushScheduled else { return }
        zoomFlushScheduled = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.zoomFlushScheduled = false
            // 先冲刷残留的平移 delta（平移→缩放快速切换时，旧位移不得叠加到缩放后的 offset 上）
            if self.pendingOffsetDelta != .zero {
                let panDelta = self.pendingOffsetDelta
                self.pendingOffsetDelta = .zero
                self.applyPendingOffset(panDelta)
            }
            let factor = self.pendingZoomFactor
            self.pendingZoomFactor = 1.0
            guard factor != 1.0 else { return }
            guard let zoomBinding = self.zoomBinding, let offsetBinding = self.offsetBinding else { return }
            let oldZoom = zoomBinding.wrappedValue
            let oldOffset = offsetBinding.wrappedValue
            let newZoom = min(3.0, max(0.1, oldZoom * factor))
            // 锚点屏幕位置 = 视口中心
            let anchorScreen = CGPoint(x: self.viewSize.width / 2, y: self.viewSize.height / 2)
            // 锚点画布坐标（缩放前）
            let anchorCanvas = viewToCanvas(anchorScreen, offset: oldOffset, zoom: oldZoom)
            zoomBinding.wrappedValue = newZoom
            // 缩放后保持锚点屏幕位置不变
            offsetBinding.wrappedValue = CGPoint(
                x: anchorScreen.x - anchorCanvas.x * newZoom,
                y: anchorScreen.y - anchorCanvas.y * newZoom
            )
        }
    }
    
    // 命中检测：屏幕坐标是否落在空节点占位区按钮区域（按钮组中心 ≈ 占位区中心）
    // 返回 (节点 id, 按钮动作)；仅空节点（无图片）有按钮；左半=上传，右半=资产库
    // 按钮区域与渲染侧共用 emptyNodeButtonRect（节点 ui.swift），乘 zoom + offset 转屏幕坐标，保证与 scaleEffect(zoom) 渲染一致
    private func hitTestEmptyNodeButton(at screenPos: CGPoint) -> (UUID, EmptyNodeButtonAction)? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect   // ⑦ 可见区裁剪
        for node in nodes where node.imageFileName == nil {
            let size = nodeSizes[node.id] ?? .zero
            guard size.height > 0 else { continue }
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            // 复用渲染侧同款按钮区域（相对节点中心，逻辑尺寸），乘 zoom 转屏幕坐标
            let rect = emptyNodeButtonRect(for: size, type: node.type)
            let screenRect = CGRect(x: center.x + rect.minX * zoom,
                                    y: center.y + rect.minY * zoom,
                                    width: rect.width * zoom, height: rect.height * zoom)
            if screenRect.contains(screenPos) {
                let action: EmptyNodeButtonAction
                if node.type.allowedAssetCategories.isEmpty {
                    // 文本节点只有「上传」按钮，整个按钮区都算上传
                    action = .upload
                } else {
                    action = screenPos.x < center.x ? .upload : .library
                }
                return (node.id, action)
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在视频节点信息栏的「尾帧」开关区域
    // 返回命中的节点 id；仅视频且有实际内容（缩略图）的节点显示尾帧开关（渲染侧同条件）
    // 区域与渲染侧共用 tailFrameSwitchHitRect（节点 ui.swift），乘 zoom + offset 转屏幕坐标，保证与渲染一致
    private func hitTestTailFrameSwitch(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect   // ⑦ 可见区裁剪
        for node in nodes {
            guard node.type == .video, node.imageFileName != nil else { continue }
            let size = nodeSizes[node.id] ?? .zero
            guard size.height > 0 else { continue }
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            // 复用渲染侧同款命中区域（相对节点中心，逻辑尺寸），乘 zoom 转屏幕坐标
            let rect = tailFrameSwitchHitRect(for: size, type: node.type, imageFileName: node.imageFileName)
            let screenRect = CGRect(x: center.x + rect.minX * zoom,
                                    y: center.y + rect.minY * zoom,
                                    width: rect.width * zoom,
                                    height: rect.height * zoom)
            if screenRect.contains(screenPos) {
                return node.id
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在音频/视频节点的播放按钮区域
    // 返回命中的节点 id；仅音频/视频且有实际媒体内容的节点显示播放按钮（渲染侧同条件）
    // 按钮中心与渲染侧共用 playButtonCenterOffset（节点 ui.swift），乘 zoom + offset 转屏幕坐标，保证与渲染一致
    private func hitTestPlayButton(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect   // ⑦ 可见区裁剪
        for node in nodes {
            guard node.type == .audio || node.type == .video, mediaFileName(for: node) != nil else { continue }
            let size = nodeSizes[node.id] ?? .zero
            guard size.height > 0 else { continue }
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            // 复用渲染侧同款按钮中心偏移（相对节点中心，逻辑尺寸），乘 zoom 转屏幕坐标
            let offsetPoint = playButtonCenterOffset(for: size, type: node.type, imageFileName: node.imageFileName)
            let buttonCenter = CGPoint(x: center.x + offsetPoint.x * zoom,
                                       y: center.y + offsetPoint.y * zoom)
            let half = playButtonHitSize * zoom / 2
            let rect = CGRect(x: buttonCenter.x - half, y: buttonCenter.y - half,
                              width: playButtonHitSize * zoom, height: playButtonHitSize * zoom)
            if rect.contains(screenPos) {
                return node.id
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在节点的历史徽章区域
    // 返回命中的节点 id；仅节点有内容（图片或媒体）时显示徽章（渲染侧同条件）
    // 徽章中心与渲染侧共用 historyBadgeCenterOffset（节点 ui.swift），乘 zoom + offset 转屏幕坐标，保证与渲染一致
    private func hitTestHistoryBadge(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect   // ⑦ 可见区裁剪
        for node in nodes {
            // 无历史时不显示徽章（渲染侧同条件），命中检测必须同步跳过，否则"看不见却点得到"
            guard let history = node.history, !history.isEmpty else { continue }
            guard node.imageFileName != nil || mediaFileName(for: node) != nil else { continue }
            let size = nodeSizes[node.id] ?? .zero
            guard size.height > 0 else { continue }
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            let offsetPoint = historyBadgeCenterOffset(for: size, type: node.type, imageFileName: node.imageFileName)
            let badgeCenter = CGPoint(x: center.x + offsetPoint.x * zoom,
                                      y: center.y + offsetPoint.y * zoom)
            let half = historyBadgeHitSize * zoom / 2
            let rect = CGRect(x: badgeCenter.x - half, y: badgeCenter.y - half,
                              width: historyBadgeHitSize * zoom, height: historyBadgeHitSize * zoom)
            if rect.contains(screenPos) {
                return node.id
            }
        }
        return nil
    }

    // 命中检测：屏幕坐标是否落在生成状态覆盖层"取消"按钮区域
    // 返回命中的节点 id；仅该节点有活跃生成任务（pending/running）时覆盖层显示（渲染侧同条件）
    // 按钮区域与渲染侧共用 generationCancelButtonRect（节点 ui.swift），乘 zoom + offset 转屏幕坐标，保证与渲染一致
    private func hitTestGenerationCancelButton(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect   // ⑦ 可见区裁剪
        for node in nodes {
            guard GenerationQueue.shared.taskID(for: node.id) != nil else { continue }
            let size = nodeSizes[node.id] ?? .zero
            guard size.height > 0 else { continue }
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            let rect = generationCancelButtonRect(for: size, type: node.type)
            let screenRect = CGRect(x: center.x + rect.minX * zoom,
                                    y: center.y + rect.minY * zoom,
                                    width: rect.width * zoom,
                                    height: rect.height * zoom)
            if screenRect.contains(screenPos) {
                return node.id
            }
        }
        return nil
    }

    // 命中检测：屏幕坐标是否落在播放中节点的进度条区域
    // 返回 (节点 id, 0...1 进度比例)；仅当前正在播放的视频/音频节点参与检测（进度条只在播放中渲染）
    // 进度条区域与渲染侧共用 progressBarRect（媒体播放.swift），乘 zoom + offset 转屏幕坐标，保证与渲染一致
    private func hitTestProgressBar(at screenPos: CGPoint) -> (UUID, Double)? {
        guard let pid = playingNodeID else { return nil }
        guard let node = nodes.first(where: { $0.id == pid }),
              node.type == .audio || node.type == .video else { return nil }
        let zoom = currentZoom
        let offset = currentOffset
        let size = nodeSizes[node.id] ?? .zero
        guard size.height > 0 else { return nil }
        let center = canvasToView(node.position, offset: offset, zoom: zoom)
        let rect = progressBarRect(for: size, type: node.type, imageFileName: node.imageFileName)
        let screenRect = CGRect(x: center.x + rect.minX * zoom,
                                y: center.y + rect.minY * zoom,
                                width: rect.width * zoom,
                                height: rect.height * zoom)
        // 命中放宽：上下各扩 6pt（条高 5pt 太细，方便点击）
        let hitRect = screenRect.insetBy(dx: 0, dy: -6)
        guard hitRect.contains(screenPos) else { return nil }
        let ratio = Double(min(1, max(0, (screenPos.x - screenRect.minX) / max(screenRect.width, 1))))
        return (node.id, ratio)
    }
    
    // 拖动期间按鼠标 x 计算进度比例（不限制高度：鼠标略移出进度条仍持续跟随）
    private func progressRatio(at screenPos: CGPoint, for nodeID: UUID) -> Double? {
        guard let node = nodes.first(where: { $0.id == nodeID }) else { return nil }
        let zoom = currentZoom
        let offset = currentOffset
        let size = nodeSizes[node.id] ?? .zero
        guard size.height > 0 else { return nil }
        let center = canvasToView(node.position, offset: offset, zoom: zoom)
        let rect = progressBarRect(for: size, type: node.type, imageFileName: node.imageFileName)
        let minX = center.x + rect.minX * zoom
        let width = rect.width * zoom
        guard width > 1 else { return nil }
        return Double(min(1, max(0, (screenPos.x - minX) / width)))
    }
}
