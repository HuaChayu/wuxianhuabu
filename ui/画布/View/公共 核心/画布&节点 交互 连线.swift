// ============================================================
//  文件作用：节点交互层（连线相关）。NodeCanvasView 负责连线渲染（已建立连线、
//  选中连线的中心裁断按钮、拖拽中的临时连线 DraggingLine）与加号拖拽连线回调
//  （handleConnectDragChanged / handleConnectDragEnded）。
//  互动文件：引用 画布 状态.swift（CanvasStore、AssetDragData、坐标换算）、
//  节点ui.swift（NodeView、ConnectionLine、DraggingLine、CanvasNode、NodeConnection、
//  ConnectSide、connectionMidPoint、portPosition、hitTestPort）、
//  资产管理ui.swift（AssetStore）、画布 节点操作.swift（NodeFactory）；
//  被 画布 ui.swift 引用（NodeCanvasView）。
//  鼠标控制层（MouseControlView / MouseControlNSView）已移至 画布 交互2鼠标.swift。
// ============================================================

import SwiftUI
import UniformTypeIdentifiers

// ============================================================

// MARK: - 节点交互层（节点渲染 + 连线渲染 + 拖拽落点 + 创建节点 + 连线回调）

/// 画布节点交互层：负责节点与连线的渲染、资产拖拽落点创建节点、加号拖拽连线回调。
/// 位于鼠标控制层之上以支持加号拖拽连线；纯渲染 + dropDestination，不拦截鼠标事件。
struct NodeCanvasView: View {
    @EnvironmentObject var store: CanvasStore
    @ObservedObject var assetStore = AssetStore.shared
    /// 媒体播放管理（观察播放状态：屏幕叠加层按节点坐标补画真实视频画面）
    @ObservedObject private var playerManager = MediaPlayerManager.shared
    
    /// 拖动节点的固定预分配余量（画布坐标）：拖动期间不随 nodeDragOffset 每帧变化，
    /// 避免内容层位图尺寸/锚点反复变化触发 drawingGroup 重采样导致周围静止节点抖动；
    /// 拖动距离超过该余量时节点可能被位图裁剪（视觉裁剪，松手后恢复）。
    private let nodeDragExtentMargin: CGFloat = 4000
    /// 内容层离屏合帧的半边长（屏幕坐标）：以所有节点画布坐标的最大半径（含缩放）为界，再加边距。
    /// drawingGroup 的离屏位图只包含视图 bounds 内的内容，超出部分会被裁剪，必须保证节点、连线端点和裁断按钮都在界内。
    /// 拖动余量固定预分配（nodeDragExtentMargin），拖动节点期间 extent 保持不变，避免位图每帧重采样。
    private func contentExtent(for nodes: [CanvasNode], zoom: Double) -> CGFloat {
        let margin: CGFloat = 600   // 覆盖节点半宽/半高、连线端点偏移与裁断按钮
        guard !nodes.isEmpty else { return margin }
        let maxR = nodes.reduce(CGFloat(0)) { partial, node in
            max(partial, abs(node.position.x), abs(node.position.y))
        }
        return (maxR + nodeDragExtentMargin) * CGFloat(zoom) + margin
    }
    
    var body: some View {
        // 内容层离屏合帧的半边长（屏幕坐标）：覆盖所有节点中心范围 + 边距，防止 drawingGroup 裁剪
        let extent = contentExtent(for: store.nodes, zoom: store.zoom)
        let layerOffset = CGPoint(x: extent, y: extent)   // 内容层中心 = 画布原点，内部坐标 + extent 对齐
        // 预建节点字典，替代连线/裁断按钮/临时连线里对 nodes 的 first(where:) O(n) 查找（⑤）
        let nodesByID = Dictionary(uniqueKeysWithValues: store.nodes.map { ($0.id, $0) })
        // 按 groupID 分组（仅已打组节点），内容层据此画组包围盒
        let groupedNodes: [(key: UUID, nodes: [CanvasNode])] = {
            let dict = Dictionary(grouping: store.nodes.filter { $0.groupID != nil }, by: { $0.groupID! })
            return dict.map { (key: $0.key, nodes: $0.value) }
        }()
        ZStack {
            // ===== 画布内容层：内部使用画布坐标（不含视口 offset），离屏合帧后整体平移 =====
            // 内容层中心用 .position 钉在画布原点（= offset），平移画布时 SwiftUI 只移动整张位图，不再逐节点重算
            ZStack {
            // 已建立的连线（节点下方）
            ForEach(store.connections) { connection in
                if let from = nodesByID[connection.fromID],
                   let to = nodesByID[connection.toID] {
                    // 线禁用态按目标节点输入合规性判定：视频节点选了模型后，连到它身上不符合要求的线禁用灰色虚线
                    let validity = store.inputValidity(for: to.id)
                    let isDisabled = validity.applies && validity.invalidConnectionIDs.contains(connection.id)
                    // 连续视频发光：级联传播即将触发的连线按全局色闪烁（仅发光线进入动画分支）
                    let isGlowing = store.glowingConnectionIDs.contains(connection.id)
                    let isSelected = store.selectedConnectionID == connection.id
                    // ③动画移出合帧层：发光/选中线含 60fps TimelineView，若留在合帧层内会每帧触发整张
                    // 内容层位图重采样；这里跳过，改由屏幕叠加层以屏幕坐标渲染动画版本（视觉无缝衔接）
                    if !isGlowing && !isSelected {
                        ConnectionLine(from: from, to: to, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: layerOffset,
                                       color: store.nodeColor,
                                       isSelected: false,
                                       isDisabled: isDisabled,
                                       isGlowing: false,
                                       dragOffset: store.nodeDragOffset,
                                       draggingNodeIDs: store.draggingNodeIDs)
                            .equatable()   // 平移画布时渲染输入不变 → 剪枝 body，连线不重算、位图不重采样
                    }
                }
            }
            
            // 选中连线的中心裁断按钮（点击移除该连线；点击命中由鼠标控制层处理，此处仅渲染视觉）
            if let connID = store.selectedConnectionID,
               let connection = store.connections.first(where: { $0.id == connID }),
               let from = nodesByID[connection.fromID],
               let to = nodesByID[connection.toID] {
                // 拖动期间端点节点已叠加 nodeDragOffset，裁断按钮需同步补偿位置（否则与节点视觉错位）；
                // 中点是两端位移的均值：只有一端节点被拖动时补偿一半
                let dragSumX = (store.draggingNodeIDs.contains(from.id) ? store.nodeDragOffset.x : 0)
                             + (store.draggingNodeIDs.contains(to.id) ? store.nodeDragOffset.x : 0)
                let dragSumY = (store.draggingNodeIDs.contains(from.id) ? store.nodeDragOffset.y : 0)
                             + (store.draggingNodeIDs.contains(to.id) ? store.nodeDragOffset.y : 0)
                let mid = connectionMidPoint(from: from, to: to, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: layerOffset)
                Label("裁断", systemImage: "scissors")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(store.nodeColor, in: Capsule())
                    .shadow(color: .black.opacity(0.25), radius: 4, y: 2)
                    .position(x: mid.x + dragSumX / 2 * store.zoom, y: mid.y + dragSumY / 2 * store.zoom)
            }
            
            // 打组包围盒（按 groupID 分组，画在连线之上节点之下；拖动中节点按临时位移实时包裹）
            // 盒子样式：主题色半透明底（装着节点的盒子感）+ 加深虚线描边 + 淡阴影
            ForEach(groupedNodes, id: \.key) { group in
                // 拖动期间 rect 已按「起始位置 + 临时位移」计算（nodesBoundingRect 支持拖动参数），
                // 大小/位置实时跟随拖动中的节点，松手合并写回后 rect 与拖动中一致、无跳变
                let rect = nodesBoundingRect(group.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom,
                                             draggingNodeIDs: store.draggingNodeIDs,
                                             dragOffset: store.nodeDragOffset)
                // 整组选中：选中集恰好等于该组全部节点（含组外节点不算），组盒高亮实线
                let groupNodeIDs = Set(group.nodes.map(\.id))
                let isGroupSelected = store.selectedNodeIDs == groupNodeIDs
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(store.nodeColor.opacity(isGroupSelected ? 0.18 : 0.09))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(
                                store.nodeColor.opacity(isGroupSelected ? 0.95 : 0.55),
                                style: StrokeStyle(lineWidth: isGroupSelected ? 2.5 : 1.5,
                                                   dash: isGroupSelected ? [] : [6, 4])
                            )
                    )
                    .shadow(color: store.nodeColor.opacity(isGroupSelected ? 0.45 : 0.15),
                            radius: isGroupSelected ? 10 : 6, y: 2)
                    .frame(width: rect.width + 2 * groupBoxPadding * store.zoom,
                           height: rect.height + 2 * groupBoxPadding * store.zoom)
                    .position(x: layerOffset.x + rect.midX,
                              y: layerOffset.y + rect.midY)
                // 组盒加号（整组选中 / 本组为连线拖出源 / 拖拽连线中本组加号为吸附目标 时显示）：
                // 与组盒同款 .position 绝对定位，位置消费公共函数 groupPortBallCenter（画布坐标）+ layerOffset 转内容层，
                // 吸附目标态高亮缩放 + 跟随鼠标吸附偏移（与节点加号 connectButton 同构）；命中与拖拽由鼠标控制层处理
                let isLeftGroupTarget = store.hoveredGroupPortID == group.key && store.hoveredGroupPortSide == .left
                let isRightGroupTarget = store.hoveredGroupPortID == group.key && store.hoveredGroupPortSide == .right
                if isGroupSelected || store.draggingConnection?.fromID == group.key || isLeftGroupTarget || isRightGroupTarget {
                    groupPortBall(side: .left, rect: rect, zoom: store.zoom, layerOffset: layerOffset,
                                  isTarget: isLeftGroupTarget,
                                  snapOffset: isLeftGroupTarget ? (store.hoveredGroupPortOffset ?? .zero) : .zero)
                    groupPortBall(side: .right, rect: rect, zoom: store.zoom, layerOffset: layerOffset,
                                  isTarget: isRightGroupTarget,
                                  snapOffset: isRightGroupTarget ? (store.hoveredGroupPortOffset ?? .zero) : .zero)
                }
            }
            
            // 画布节点（画布坐标，随画布平移缩放）
            // 播放中的视频/音频节点不在此层渲染：
            // - 视频：drawingGroup 离屏合帧会吞掉 AVPlayerLayer 画面（有声无画），该节点整体移出合帧层、
            //   在内容层外以屏幕坐标渲染（见下方"播放中节点层"），画面与缩略图天然对齐；
            // - 音频：播放中的进度条/频谱动画每 0.1s 高频刷新，留在合帧层内会触发整张内容层位图重绘（卡顿），
            //   同样移出合帧层单独渲染，动效不拖累画布性能。
            ForEach(store.nodes) { node in
                if !((node.type == .video || node.type == .audio) && playerManager.isPlaying(nodeID: node.id)) {
                NodeView(
                    node: node,
                    zoom: store.zoom,
                    isSelected: store.selectedNodeIDs.contains(node.id),
                    isHovered: store.hoveredNodeID == node.id,
                    hoveredEmptyButton: store.hoveredEmptyButtonNodeID == node.id ? store.hoveredEmptyButtonAction : nil,
                    accentColor: store.nodeColor,
                    portOffset: store.hoveredNodeID == node.id ? store.hoveredPortOffset : nil,
                    portSide: store.hoveredNodeID == node.id ? store.hoveredPortSide : nil,
                    showPortSide: {
                        // 拖拽连线中：仅鼠标命中的节点亮出与拖出端相反方向的加号（输出→输入）
                        guard let dragging = store.draggingConnection, dragging.fromID != node.id else { return nil }
                        guard store.hoveredNodeID == node.id else { return nil }
                        return dragging.side.opposite
                    }(),
                    isPortDragging: store.draggingConnection?.fromID == node.id,
                    // 视频画面改由屏幕叠加层渲染（drawingGroup 离屏合帧吞掉 AVPlayerLayer 画面，见屏幕层视频画面补画）
                    videoSurfaceHidden: true,
                    onSizeChange: { size in
                        // 值相同不写回：避免 @Published nodeSizes 在节点尺寸无变化时仍每帧触发，连锁重建内容层（拖动抖动根因之一）
                        if store.nodeSizes[node.id] != size {
                            store.nodeSizes[node.id] = size
                            debugLog("节点尺寸上报: \(node.id) \(size)")
                        }
                    },
                    onConnectDragChanged: { _, localPos, _, side in
                        handleConnectDragChanged(node: node, localPos: localPos, side: side)
                    },
                    onConnectDragEnded: { _, localPos, _, side in
                        handleConnectDragEnded(node: node, localPos: localPos, side: side)
                    }
                )
                .equatable()   // 平移画布时渲染输入不变 → 剪枝 body，节点不重算、位图不重采样
                .position(
                    x: extent + node.position.x * store.zoom,
                    y: extent + node.position.y * store.zoom
                )
                // 拖动中的节点叠加临时位移（仅变换，NodeView 本体参数不变；松手后合并写回 position）
                .offset(
                    x: store.draggingNodeIDs.contains(node.id) ? store.nodeDragOffset.x * store.zoom : 0,
                    y: store.draggingNodeIDs.contains(node.id) ? store.nodeDragOffset.y * store.zoom : 0
                )
                }   // 非播放中视频节点
            }
            }   // 内容层结束
            .frame(width: extent * 2, height: extent * 2)   // bounds 覆盖所有节点，避免 drawingGroup 裁剪
            .drawingGroup()   // 离屏合帧：连线+节点只合成一次（position 在其外层，平移只做位图变换）
            .position(x: store.offset.x, y: store.offset.y)   // 内容层中心 = 画布原点：节点最终位置 = offset + canvas*zoom，与原来一致
            
            // ===== 播放中视频/音频节点层（内容层外、屏幕坐标）=====
            // AVPlayerLayer 不参与 drawingGroup 离屏合帧（内容层内会"有声无画"），播放中的视频节点整体在合帧层外渲染；
            // 音频节点同理移出合帧层，进度条/频谱动画不触发整张内容层位图重绘。
            // NodeView 内占位区 ZStack 与缩略图天然对齐，不依赖任何坐标推算，播放瞬间无跳变。
            // 屏幕位置 = 内容层内节点坐标（extent + position*zoom）经内容层平移（offset - extent）后 = position*zoom + offset。
            ForEach(store.nodes) { node in
                if (node.type == .video || node.type == .audio), playerManager.isPlaying(nodeID: node.id) {
                    NodeView(
                        node: node,
                        zoom: store.zoom,
                        isSelected: store.selectedNodeIDs.contains(node.id),
                        isHovered: store.hoveredNodeID == node.id,
                        hoveredEmptyButton: store.hoveredEmptyButtonNodeID == node.id ? store.hoveredEmptyButtonAction : nil,
                        accentColor: store.nodeColor,
                        portOffset: store.hoveredNodeID == node.id ? store.hoveredPortOffset : nil,
                        portSide: store.hoveredNodeID == node.id ? store.hoveredPortSide : nil,
                        showPortSide: {
                            guard let dragging = store.draggingConnection, dragging.fromID != node.id else { return nil }
                            guard store.hoveredNodeID == node.id else { return nil }
                            return dragging.side.opposite
                        }(),
                        isPortDragging: store.draggingConnection?.fromID == node.id,
                        videoSurfaceHidden: false,
                        onSizeChange: { size in
                            // 值相同不写回：避免 @Published nodeSizes 在节点尺寸无变化时仍每帧触发，连锁重建内容层（拖动抖动根因之一）
                            if store.nodeSizes[node.id] != size {
                                store.nodeSizes[node.id] = size
                                debugLog("节点尺寸上报: \(node.id) \(size)")
                            }
                        },
                        onConnectDragChanged: { _, localPos, _, side in
                            handleConnectDragChanged(node: node, localPos: localPos, side: side)
                        },
                        onConnectDragEnded: { _, localPos, _, side in
                            handleConnectDragEnded(node: node, localPos: localPos, side: side)
                        }
                    )
                    .equatable()   // 平移画布时渲染输入不变 → 剪枝 body（外层 .position 仍随 offset 正常平移）
                    .position(
                        x: node.position.x * store.zoom + store.offset.x,
                        y: node.position.y * store.zoom + store.offset.y
                    )
                    .offset(
                        x: store.draggingNodeIDs.contains(node.id) ? store.nodeDragOffset.x * store.zoom : 0,
                        y: store.draggingNodeIDs.contains(node.id) ? store.nodeDragOffset.y * store.zoom : 0
                    )
                }
            }
            
            // ===== 屏幕叠加层：屏幕坐标，不随画布平移 =====
            // ③动画连线层（合帧层外）：发光/选中连线在屏幕坐标渲染 60fps TimelineView 动画版本，
            // 与内容层静态线视觉无缝衔接；动画在合帧层外 → 不触发内容层位图每帧重采样
            ForEach(store.connections) { connection in
                if let from = nodesByID[connection.fromID],
                   let to = nodesByID[connection.toID] {
                    let isGlowing = store.glowingConnectionIDs.contains(connection.id)
                    let isSelected = store.selectedConnectionID == connection.id
                    if isGlowing || isSelected {
                        let validity = store.inputValidity(for: to.id)
                        let isDisabled = validity.applies && validity.invalidConnectionIDs.contains(connection.id)
                        ConnectionLine(from: from, to: to, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset,
                                       color: store.nodeColor,
                                       isSelected: isSelected,
                                       isDisabled: isDisabled,
                                       isGlowing: isGlowing,
                                       dragOffset: store.nodeDragOffset,
                                       draggingNodeIDs: store.draggingNodeIDs)
                    }
                }
            }
            // 正在拖拽的临时连线（虚线，最上层；端点可能超出内容层，留在屏幕层避免裁剪）
            // 节点拖出：起点 = 节点端口；组盒加号拖出（fromID 是组 ID）：起点 = 组盒对应侧球心
            if let dragging = store.draggingConnection {
                if let from = nodesByID[dragging.fromID] {
                    DraggingLine(from: from, to: dragging.currentPos, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset, side: dragging.side, color: store.nodeColor)
                } else if store.nodes.contains(where: { $0.groupID == dragging.fromID }) {
                    let groupNodes = store.nodes.filter { $0.groupID == dragging.fromID }
                    let rect = nodesBoundingRect(groupNodes, nodeSizes: store.nodeSizes, zoom: store.zoom)
                    let ballCenter = groupPortBallCenter(rect: rect, side: dragging.side, zoom: store.zoom)
                    let start = CGPoint(x: ballCenter.x + store.offset.x, y: ballCenter.y + store.offset.y)
                    DraggingLine(from: nil, startPoint: start, to: dragging.currentPos, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset, side: dragging.side, color: store.nodeColor)
                }
            }
            
            // 框选矩形（屏幕坐标，左上原点）
            if let rect = store.selectionRect {
                Rectangle()
                    .stroke(store.nodeColor.opacity(0.7), lineWidth: 1)
                    .background(store.nodeColor.opacity(0.08))
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
            }
            
            // 对齐参考线（拖动节点吸附时显示，虚线贯穿画布，最上层）
            ForEach(store.alignmentGuides) { guide in
                Path { path in
                    if guide.isVertical {
                        path.move(to: CGPoint(x: guide.position, y: -5000))
                        path.addLine(to: CGPoint(x: guide.position, y: 5000))
                    } else {
                        path.move(to: CGPoint(x: -5000, y: guide.position))
                        path.addLine(to: CGPoint(x: 5000, y: guide.position))
                    }
                }
                .stroke(store.nodeColor.opacity(0.8), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .dropDestination(for: AssetDragData.self) { items, location in
            // 批量拖拽：每个节点在鼠标位置基础上斜向偏移，避免重叠
            store.pushSnapshot()   // 批量创建节点前记录快照（用于撤销，整批算一次）
            var index = 0
            for item in items {
                for id in item.assetIDs {
                    if let asset = assetStore.assets.first(where: { $0.id == id }) {
                        let offset = CGPoint(x: CGFloat(index) * 24, y: CGFloat(index) * 24)
                        addNode(from: asset, at: CGPoint(x: location.x + offset.x, y: location.y + offset.y),
                                offset: store.offset, zoom: store.zoom, store: store)
                        index += 1
                    }
                }
            }
            // 批量拖拽视为一次操作，只保存一次
            store.save()
            return true
        }
        // 接收外部拖入的媒体文件：画布区域导入成节点（落点位置建节点，批量斜向错开）
        // 外部拖放落点是视图坐标：先经 viewToCanvas 换算为画布坐标再传 importAsset/nodePosition，
        // 对齐内部资产拖拽 addNode 的换算链路，避免 canvasToView→viewToCanvas 两次变换抵消导致节点偏离落点（平移/缩放后更明显）
        .mediaDropReceiver(store: store) { location in
            Optional(viewToCanvas(location, offset: store.offset, zoom: store.zoom))
        }
    }
    
    // 加号拖拽中：节点本地坐标 → 屏幕坐标，公共入口更新临时连线 + 命中检测（实现见下方 updateConnectDrag）
    private func handleConnectDragChanged(node: CanvasNode, localPos: CGPoint, side: ConnectSide) {
        let nodeScreen = canvasToView(node.position, offset: store.offset, zoom: store.zoom)
        let screenPos = CGPoint(x: nodeScreen.x + localPos.x, y: nodeScreen.y + localPos.y)
        updateConnectDrag(screenPos: screenPos, fromNodeID: node.id, side: side, store: store)
    }
    
    // 加号拖拽结束：公共入口完成建连（实现见下方 finishConnectDrag）；
    // 该路径（ConnectButton 拖拽）未命中时不弹生成面板，仅打日志
    private func handleConnectDragEnded(node: CanvasNode, localPos: CGPoint, side: ConnectSide) {
        let nodeScreen = canvasToView(node.position, offset: store.offset, zoom: store.zoom)
        let screenPos = CGPoint(x: nodeScreen.x + localPos.x, y: nodeScreen.y + localPos.y)
        finishConnectDrag(screenPos: screenPos, fromNodeID: node.id, side: side, store: store)
    }
    
    // 组盒加号球（选择态/拖出源/吸附目标显示，纯视觉）：与节点加号同款外观，尺寸随 zoom 缩放；
    // 位置消费公共函数 groupPortBallCenter（画布坐标）→ 加 layerOffset 转内容层，
    // 与组盒同款 .position 绝对定位，保证随组同步移动；目标态高亮缩放 + snapOffset 跟随鼠标吸附
    @ViewBuilder
    func groupPortBall(side: ConnectSide, rect: CGRect, zoom: Double, layerOffset: CGPoint,
                       isTarget: Bool = false, snapOffset: CGPoint = .zero) -> some View {
        let center = groupPortBallCenter(rect: rect, side: side, zoom: zoom)
        Image(systemName: "plus")
            .font(.system(size: 14, weight: .bold))
            .foregroundColor(isTarget ? .white : store.nodeColor)
            .frame(width: nodePortSize * zoom, height: nodePortSize * zoom)
            .background(Circle().fill(isTarget ? store.nodeColor : Color.white))
            .overlay(Circle().stroke(isTarget ? store.nodeColor : store.nodeColor.opacity(0.6), lineWidth: 1.5))
            .shadow(color: isTarget ? store.nodeColor.opacity(0.5) : .black.opacity(0.15), radius: isTarget ? 6 : 3)
            .scaleEffect(isTarget ? 1.15 : 1)
            .position(x: layerOffset.x + center.x + snapOffset.x,
                      y: layerOffset.y + center.y + snapOffset.y)
    }
}

// 加号连线拖拽中（公共入口）：屏幕坐标 → 画布坐标更新临时连线，并实时命中检测。
// 供 ConnectButton 拖拽（handleConnectDragChanged）与鼠标层加号拖拽（画布 ui.swift onPortDragChanged）共用，
// 消除两份几乎逐行相同的实现；拖拽中鼠标移动事件不触发，这里手动命中检测更新悬停目标
func updateConnectDrag(screenPos: CGPoint, fromNodeID: UUID, side: ConnectSide, store: CanvasStore) {
    let canvasPos = viewToCanvas(screenPos, offset: store.offset, zoom: store.zoom)
    store.draggingConnection = DraggingConnection(fromID: fromNodeID, currentPos: canvasPos, side: side)
    // 目标节点亮出对侧加号并吸附鼠标
    if let hit = hitTestPort(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset, includeOffset: true),
       hit.nodeID != fromNodeID {
        store.hoveredNodeID = hit.nodeID
        store.hoveredPortSide = hit.side
        store.hoveredPortOffset = hit.portOffset
        store.hoveredGroupPortID = nil
        store.hoveredGroupPortSide = nil
        store.hoveredGroupPortOffset = nil
        debugLog("连线拖拽中: 命中球 node=\(hit.nodeID) side=\(hit.side) screen=\(screenPos)")
    } else if let hit = hitTestGroupPortForDrag(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset) {
        // 组盒加号目标：亮出该组对侧加号并吸附鼠标（与节点加号同构；不再排除拖出组自身，同组也亮提示，松手时由 finishConnectDrag 拦截）
        store.hoveredGroupPortID = hit.0
        store.hoveredGroupPortSide = hit.1
        store.hoveredGroupPortOffset = hit.2
        store.hoveredNodeID = nil
        store.hoveredPortSide = nil
        store.hoveredPortOffset = nil
        debugLog("连线拖拽中: 命中组加号 group=\(hit.0) side=\(hit.1) screen=\(screenPos)")
    } else if let hitID = hitTestNodeBody(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset),
              hitID != fromNodeID {
        store.hoveredNodeID = hitID
        store.hoveredPortSide = nil
        store.hoveredPortOffset = nil
        store.hoveredGroupPortID = nil
        store.hoveredGroupPortSide = nil
        store.hoveredGroupPortOffset = nil
        debugLog("连线拖拽中: 命中本体 node=\(hitID) screen=\(screenPos)")
    } else {
        store.hoveredNodeID = nil
        store.hoveredPortSide = nil
        store.hoveredPortOffset = nil
        store.hoveredGroupPortID = nil
        store.hoveredGroupPortSide = nil
        store.hoveredGroupPortOffset = nil
        debugLog("连线拖拽中: 未命中 screen=\(screenPos)")
    }
}

// 加号连线拖拽结束（公共入口）：命中目标节点加号且方向相反（输出→输入）则建立连线；
// 目标支持 节点加号 / 组盒加号（组内全部节点批量）/ 节点本体（方向自动取与拖出端相反）。
// 返回是否完成本次拖拽处理（命中目标或已存在连线）；未命中返回 false，供调用方决定是否弹「引用该节点生成」面板
@discardableResult
func finishConnectDrag(screenPos: CGPoint, fromNodeID: UUID, side: ConnectSide, store: CanvasStore) -> Bool {
    // 无论命中与否，结束统一清理临时连线与组加号吸附态（含多次 return 分支）
    defer {
        store.draggingConnection = nil
        store.hoveredGroupPortID = nil
        store.hoveredGroupPortSide = nil
        store.hoveredGroupPortOffset = nil
    }
    // 目标与方向：优先命中节点加号（精确方向），其次组盒加号（目标组全部节点），再命中节点本体（自动取相反方向）
    var targetIDs: [UUID] = []
    var targetSide: ConnectSide?
    if let hit = hitTestPort(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset),
       hit.nodeID != fromNodeID,
       hit.side == side.opposite {
        targetIDs = [hit.nodeID]
        targetSide = hit.side
    } else if let hit = hitTestGroupPortForDrag(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset),
              hit.1 == side.opposite {
        // 松手在拖出组自身的加号上：仅提示不建连（避免组内互连），静默取消不弹生成面板
        if hit.0 == fromNodeID {
            debugLog("连线拖拽结束：松手在拖出组自身加号，取消")
            return true
        }
        // 目标组加号：方向匹配时与组内全部节点批量连线
        let members = store.nodes.filter { $0.groupID == hit.0 }.map(\.id)
        guard !members.isEmpty else {
            store.draggingConnection = nil
            debugLog("连线拖拽结束：目标组为空")
            return false
        }
        targetIDs = members
        targetSide = hit.1
    } else if let hitID = hitTestNodeBody(at: screenPos, nodes: store.nodes, nodeSizes: store.nodeSizes, zoom: store.zoom, offset: store.offset),
              hitID != fromNodeID {
        targetIDs = [hitID]
        targetSide = side.opposite
    }
    guard !targetIDs.isEmpty, let targetSide = targetSide else {
        store.draggingConnection = nil
        debugLog("连线拖拽结束：未命中目标节点")
        return false
    }
    // 组盒加号拖出（fromNodeID 是组 ID）：组内每个节点与目标集合逐个建连（批量，省时间设计）
    let groupMembers = store.nodes.filter { $0.groupID == fromNodeID }
    if !groupMembers.isEmpty {
        let newConnections: [NodeConnection] = groupMembers.flatMap { node in
            targetIDs.compactMap { targetID in
                // 目标在组内：该节点跳过（避免自连）；连线方向固定为 输出 → 输入
                if node.id == targetID { return nil }
                let fromID = side.isOutput ? node.id : targetID
                let toID = side.isOutput ? targetID : node.id
                let alreadyConnected = store.connections.contains {
                    ($0.fromID == fromID && $0.toID == toID) ||
                    ($0.fromID == toID && $0.toID == fromID)
                }
                return alreadyConnected ? nil : NodeConnection(fromID: fromID, toID: toID)
            }
        }
        if !newConnections.isEmpty {
            store.pushSnapshot()   // 批量建连前记录快照（用于撤销）
            store.connections.append(contentsOf: newConnections)
            store.save()
            debugLog("组连线拖拽结束：批量建立 \(newConnections.count) 条连线")
        } else {
            debugLog("组连线拖拽结束：均已存在连线，跳过")
        }
        store.draggingConnection = nil
        return true
    }
    // 连线方向固定为 输出 → 输入（fromNodeID 是普通节点；目标可能是组内多个节点）
    let newConnections: [NodeConnection] = targetIDs.compactMap { targetID in
        guard targetID != fromNodeID else { return nil }
        let fromID: UUID
        let toID: UUID
        if side.isOutput {
            // 拖出端是输出（右侧），目标端是输入（左侧）
            fromID = fromNodeID
            toID = targetID
        } else {
            // 拖出端是输入（左侧），目标端是输出（右侧）
            fromID = targetID
            toID = fromNodeID
        }
        let alreadyConnected = store.connections.contains {
            ($0.fromID == fromID && $0.toID == toID) ||
            ($0.fromID == toID && $0.toID == fromID)
        }
        return alreadyConnected ? nil : NodeConnection(fromID: fromID, toID: toID)
    }
    if !newConnections.isEmpty {
        store.pushSnapshot()   // 建立连线前记录快照（用于撤销）
        store.connections.append(contentsOf: newConnections)
        store.save()
        debugLog("连线拖拽结束：建立 \(newConnections.count) 条连线")
    } else {
        debugLog("连线拖拽结束：已存在连线，跳过")
    }
    store.draggingConnection = nil
    return true
}

// 命中检测：屏幕坐标是否落在某个节点本体矩形内（拖拽连线松手时，拖到节点上也算连线）
// 节点本体随全局缩放（scaleEffect zoom），矩形尺寸乘 zoom 与渲染一致
func hitTestNodeBody(at screenPos: CGPoint, nodes: [CanvasNode], nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint) -> UUID? {
    for node in nodes {
        let size = nodeSizes[node.id] ?? .zero
        let center = canvasToView(node.position, offset: offset, zoom: zoom)
        let rect = CGRect(x: center.x - size.width * zoom / 2, y: center.y - size.height * zoom / 2,
                          width: size.width * zoom, height: size.height * zoom)
        if rect.contains(screenPos) {
            return node.id
        }
    }
    return nil
}

// MARK: - 空节点「资产库」选择面板（与大纲资产库同款：来源过滤 + 角色/场景分类 + 资产网格单选 + 右下角确定）

struct NodeAssetPickerView: View {
    @ObservedObject var assetStore: AssetStore
    @ObservedObject var projectStore: ProjectStore
    @Binding var category: AssetCategory
    @Binding var scope: UUID?
    @Binding var selectedAssetID: UUID?
    /// 允许选择的资产分类（按节点类型限定：视频只显示视频、音频只显示音频、图像/角色/场景显示这 3 种）
    var allowedCategories: [AssetCategory]
    var onConfirm: (AssetItem) -> Void
    var onCancel: () -> Void
    
    var body: some View {
        VStack(spacing: 0) {
            // 标题栏
            HStack {
                Text("选择资产")
                    .font(.headline)
                Spacer()
                Button("取消", action: {
                    debugLog("选择资产：取消")
                    onCancel()
                })
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            
            Divider()
            
            // 来源过滤 + 角色/场景分类
            HStack(spacing: 8) {
                Menu {
                    Button("全局") {
                        debugLog("选择资产：来源切换为 全局")
                        scope = nil
                    }
                    ForEach(projectStore.projects) { project in
                        Button(project.title) {
                            debugLog("选择资产：来源切换为「\(project.title)」")
                            scope = project.id
                        }
                    }
                } label: {
                    Text(scopeTitle)
                        .font(.system(size: 11))
                        .foregroundColor(.primary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(Color.black.opacity(0.05))
                        )
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                
                Picker("", selection: $category) {
                    ForEach(allowedCategories) { cat in
                        Text(cat.rawValue).tag(cat)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 220)
                
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            
            // 资产网格（单选）：先按来源过滤出全部被引用资产，再按分类筛
            let assets = assetStore.filteredAssets(in: scope, category: category)
            if assets.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: category.icon)
                        .font(.system(size: 24))
                        .foregroundColor(.secondary)
                    Text("暂无\(category.rawValue)")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 76), spacing: 8)], spacing: 8) {
                        ForEach(assets) { asset in
                            let isSelected = selectedAssetID == asset.id
                            VStack(spacing: 4) {
                                Image(nsImage: asset.image)
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                                    .frame(width: 68, height: 68)
                                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                                            .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
                                    )
                                Text(asset.name)
                                    .font(.system(size: 10))
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                            }
                            .frame(width: 76)
                            .padding(2)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
                            )
                            .contentShape(Rectangle())
                            .onTapGesture {
                                debugLog("选择资产：选中「\(asset.name)」")
                                selectedAssetID = asset.id
                            }
                            .onTapGesture(count: 2) {
                                // 双击 = 选中并直接确定（单击先触发选中，双击再确认，两者天然衔接）
                                debugLog("选择资产：双击确定「\(asset.name)」")
                                selectedAssetID = asset.id
                                onConfirm(asset)
                            }
                        }
                    }
                    .padding(16)
                }
            }
            
            Divider()
            
            // 右下角确定
            HStack {
                Spacer()
                Button("确定") {
                    debugLog("选择资产：确定")
                    if let id = selectedAssetID,
                       let asset = assetStore.assets.first(where: { $0.id == id }) {
                        onConfirm(asset)
                    }
                }
                .disabled(selectedAssetID == nil)
                .buttonStyle(.borderedProminent)
            }
            .padding(12)
        }
        .frame(width: 420, height: 480)
    }
    
    /// 当前来源过滤标题
    var scopeTitle: String {
        if let scope = scope,
           let project = projectStore.projects.first(where: { $0.id == scope }) {
            return project.title
        }
        return "全局"
    }
}
