// ============================================================
//  文件作用：节点操作。NodeFactory 统一创建节点；CanvasView 扩展的 addNode /
//  generateNode / copyNodeText / duplicateNodes / deleteNodes / targetNodeIDs。
//  互动文件：引用 画布 状态.swift（CanvasStore、viewToCanvas）、节点ui.swift
//  （NodeType、CanvasNode、NodeConnection）；NodeFactory 被 交互.swift 引用；
//  操作函数被 画布 ui.swift、画布 菜单.swift、画布 工具栏2.swift 引用。
// ============================================================

import SwiftUI
import AppKit

// ============================================================

// MARK: - 节点工厂（统一创建入口）

enum NodeFactory {
    /// 统一创建节点：默认标题「未命名<类型>」、副标题「出现集数：暂无」、待补充标记
    /// 完整参数版本：新增字段全部带默认值，旧调用点不受影响；粘贴侧按「存在性」传入（nil/默认 = 跳过该参数）。
    static func createNode(type: NodeType, title: String? = nil, subtitle: String? = nil, needsSupplement: Bool = true, position: CGPoint, imageFileName: String? = nil, mediaFileName: String? = nil, prompt: String = "", groupID: UUID? = nil, ratio: CanvasStore.Ratio? = nil, duration: VideoDuration? = nil, tailFrameEnabled: Bool = true, model: VideoModel = .ltx25Distill, quality: VideoQuality = .standard, imageModel: ImageModel = .hidreamO1, imageQuality: ImageQuality = .p720, history: [NodeContentHistory]? = nil) -> CanvasNode {
        CanvasNode(
            type: type,
            title: title ?? "未命名\(type.rawValue)",
            subtitle: subtitle ?? "出现集数：暂无",
            needsSupplement: needsSupplement,
            position: position,
            imageFileName: imageFileName,
            mediaFileName: mediaFileName,
            prompt: prompt,
            groupID: groupID,
            ratio: ratio,
            duration: duration,
            tailFrameEnabled: tailFrameEnabled,
            model: model,
            quality: quality,
            imageModel: imageModel,
            imageQuality: imageQuality,
            history: history
        )
    }
}

// ============================================================

// MARK: - 节点操作（创建 / 复制 / 删除 / 目标组）

extension CanvasView {
    // 在鼠标位置创建节点（屏幕坐标转画布坐标）
    func addNode(_ type: NodeType) {
        let canvasPos = viewToCanvas(contextMenuPos, offset: store.offset, zoom: store.zoom)
        debugLog("添加节点：\(type.rawValue) 位置(\(Int(canvasPos.x)),\(Int(canvasPos.y)))")
        store.pushSnapshot()   // 创建节点前记录快照（用于撤销）
        let node = NodeFactory.createNode(type: type, position: canvasPos)
        store.nodes.append(node)
        store.save()
    }
    
    // 在松手位置创建新节点并建立连线（引用该节点生成）
    func generateNode(of type: NodeType) {
        guard let fromID = generateFromNodeID, let side = generateFromSide else { return }
        let canvasPos = viewToCanvas(generatePanelPos, offset: store.offset, zoom: store.zoom)
        store.pushSnapshot()   // 创建节点+连线前记录快照（用于撤销）
        let node = NodeFactory.createNode(type: type, position: canvasPos)
        store.nodes.append(node)
        // 组盒加号拖出（fromID 是组 ID）：新节点与组内所有节点对应侧批量连线，方向与组拖出侧一致
        let groupMembers = store.nodes.filter { $0.groupID == fromID }
        if !groupMembers.isEmpty {
            let newConnections: [NodeConnection] = groupMembers.map { member in
                // 新节点刚创建必不在组内，无需跳过；连线方向固定为 输出 → 输入
                let from = side.isOutput ? member.id : node.id
                let to = side.isOutput ? node.id : member.id
                return NodeConnection(fromID: from, toID: to)
            }
            store.connections.append(contentsOf: newConnections)
            debugLog("组拖出生成：批量建立 \(newConnections.count) 条连线")
        } else {
            // 连线方向固定为 输出 → 输入
            let from: UUID
            let to: UUID
            if side.isOutput {
                from = fromID
                to = node.id
            } else {
                from = node.id
                to = fromID
            }
            store.connections.append(NodeConnection(fromID: from, toID: to))
        }
        store.save()
    }
    
    // 复制文本节点内容到剪贴板（仅文本节点可用）
    func copyNodeText(_ node: CanvasNode) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(node.title, forType: .string)
    }
    
    // 创建副本：复制节点（新 id，允许多个同名同类型节点），位置向右下偏移避免完全重叠；
    // 若原节点有实际内容（imageFileName 资产库引用），副本一并带上，避免复制出空节点。
    // 只复制用户可见可编辑的参数（提示词、模型、档位、比例、时长、尾帧开关）+ 所属组；
    // 同组节点复制后映射为新组 id，重建组框，避免与原组混在一起。
    func duplicateNodes(_ nodeIDs: Set<UUID>) {
        store.pushSnapshot()   // 复制节点前记录快照（用于撤销）
        let targets = store.nodes.filter { nodeIDs.contains($0.id) }
        // 组映射：同组节点（复制集合内同一 groupID ≥2 个）映射为新组 id，重建组框
        var groupCounts: [UUID: Int] = [:]
        for node in targets {
            if let gid = node.groupID { groupCounts[gid, default: 0] += 1 }
        }
        var groupIDMap: [UUID: UUID] = [:]
        for (gid, count) in groupCounts where count >= 2 {
            groupIDMap[gid] = UUID()
        }
        for node in targets {
            let copy = NodeFactory.createNode(
                type: node.type,
                title: node.title,
                subtitle: node.subtitle,
                needsSupplement: node.needsSupplement,
                position: CGPoint(x: node.position.x + 30, y: node.position.y + 30),
                imageFileName: node.imageFileName,
                prompt: node.prompt,
                groupID: node.groupID.flatMap { groupIDMap[$0] },
                ratio: node.ratio,
                duration: node.duration,
                tailFrameEnabled: node.tailFrameEnabled,
                model: node.model,
                quality: node.quality,
                imageModel: node.imageModel,
                imageQuality: node.imageQuality
            )
            store.nodes.append(copy)
        }
        store.save()
    }
    
    // 删除节点及其相关连线
    func deleteNodes(_ nodeIDs: Set<UUID>) {
        store.pushSnapshot()   // 删除节点前记录快照（用于撤销）
        // 删除节点中含正在播放的节点时，先停止播放（避免节点删除后声音还在响）
        if let playingID = MediaPlayerManager.shared.playingNodeID, nodeIDs.contains(playingID) {
            MediaPlayerManager.shared.stop()
        }
        // 删除前记录受影响组：删除后组内剩余节点不足 2 个时整组解散（保留剩余节点）
        let affectedGroups = Set(store.nodes
            .filter { nodeIDs.contains($0.id) && $0.groupID != nil }
            .compactMap { $0.groupID })
        store.nodes.removeAll { nodeIDs.contains($0.id) }
        store.connections.removeAll { nodeIDs.contains($0.fromID) || nodeIDs.contains($0.toID) }
        store.selectedNodeIDs.subtract(nodeIDs)
        // 若选中的连线随节点被删除，清除选中态
        if let sel = store.selectedConnectionID,
           !store.connections.contains(where: { $0.id == sel }) {
            store.selectedConnectionID = nil
        }
        for id in nodeIDs { store.nodeSizes.removeValue(forKey: id) }
        for gid in affectedGroups { dissolveGroupIfUnderflow(gid) }
        store.save()
    }
    
    // 右键命中的节点若在选中组中，则操作作用于整个选中组；否则只作用于该节点
    func targetNodeIDs(for nodeID: UUID) -> Set<UUID> {
        if store.selectedNodeIDs.contains(nodeID) {
            return store.selectedNodeIDs
        }
        return [nodeID]
    }
}

// ============================================================

// MARK: - 节点悬停命中检测（MouseControlNSView 扩展）

// 悬停节点高亮 + 两侧加号 的命中检测：由画布鼠标层 MouseControlNSView 调用，
// 逻辑归节点交互层。命中高度严格取 nodeSizes 实际尺寸（未来节点尺寸可变时自动适配）。

extension MouseControlNSView {
    // 命中检测公共样板：zoom/offset 统一取绑定量（无绑定回退 1.0 / .zero）；
    // 可见区裁剪统一取最大余量（覆盖节点半宽/半高与缩放误差），避免大画布每帧全量 O(n) 遍历
    // 供本文件 hitTestNode/hitTestBall 与 画布 交互 鼠标.swift hitTestEmptyNodeButton/hitTestPlayButton 共用
    var currentZoom: Double { zoomBinding?.wrappedValue ?? 1.0 }
    var currentOffset: CGPoint { offsetBinding?.wrappedValue ?? .zero }
    var hitVisibleRect: CGRect { bounds.insetBy(dx: -800, dy: -800) }
    
    // 命中检测：屏幕坐标是否落在某个节点的矩形内（节点本体随全局缩放，矩形尺寸乘 zoom）
    // ⑦ 优化：先按可见区裁剪，节点中心不在可见区（含余量）内直接跳过
    func hitTestNode(at screenPos: CGPoint) -> UUID? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect
        for node in nodes {
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            let size = nodeSizes[node.id] ?? .zero
            let rect = CGRect(x: center.x - size.width * zoom / 2, y: center.y - size.height * zoom / 2,
                              width: size.width * zoom, height: size.height * zoom)
            if rect.contains(screenPos) {
                return node.id
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在某个组盒内（组包围盒 + 渲染同款 padding，非节点区域）
    // 与组盒渲染一致：nodesBoundingRect 返回含缩放的画布坐标 → 加 offset 转屏幕坐标 → 每边扩 groupBoxPadding*zoom（渲染侧 frame 加 2*groupBoxPadding*zoom）
    func hitTestGroupBlank(at screenPos: CGPoint) -> [CanvasNode]? {
        let zoom = currentZoom
        let offset = currentOffset
        var groups: [UUID: [CanvasNode]] = [:]
        for node in nodes {
            if let gid = node.groupID {
                groups[gid, default: []].append(node)
            }
        }
        for (_, groupNodes) in groups {
            let rect = nodesBoundingRect(groupNodes, nodeSizes: nodeSizes, zoom: zoom)
            let pad = groupBoxPadding * zoom
            let screenRect = CGRect(x: rect.minX + offset.x - pad, y: rect.minY + offset.y - pad,
                                    width: rect.width + 2 * pad, height: rect.height + 2 * pad)
            if screenRect.contains(screenPos) {
                return groupNodes
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在组盒左右两侧的加号球容器内（仅整组选中时组盒加号存在）
    // 与组盒加号渲染一致：球心 = groupPortBallCenter（组盒 frame 外沿 + ballRadius*zoom），随 zoom 缩放；
    // 返回 (组 id, 方向, 加号相对球心的吸附偏移)；吸附复用唯一入口 portSnapOffset（与节点加号一致）
    func hitTestGroupPort(at screenPos: CGPoint) -> (UUID, ConnectSide, CGPoint)? {
        let zoom = currentZoom
        let offset = currentOffset
        var groups: [UUID: [CanvasNode]] = [:]
        for node in nodes {
            if let gid = node.groupID {
                groups[gid, default: []].append(node)
            }
        }
        for (gid, groupNodes) in groups {
            // 加号渲染条件：整组选中（isGroupSelected），未选中时无加号不可拖
            let groupIDs = Set(groupNodes.map(\.id))
            guard selectedNodeIDs == groupIDs else { continue }
            let rect = nodesBoundingRect(groupNodes, nodeSizes: nodeSizes, zoom: zoom)
            for side in [ConnectSide.left, ConnectSide.right] {
                let ballCenter = groupPortBallCenter(rect: rect, side: side, zoom: zoom)
                let screenCenter = CGPoint(x: ballCenter.x + offset.x, y: ballCenter.y + offset.y)
                let dist = distance(from: screenPos, to: screenCenter)
                if dist <= nodeBallRadius * zoom {
                    return (gid, side, portSnapOffset(screenPos: screenPos, ballCenter: screenCenter, zoom: zoom))
                }
            }
        }
        return nil
    }
    
    // 命中检测：屏幕坐标是否落在节点左右两侧的球容器内
    // 返回 (节点 id, 方向, 加号相对球中心的偏移)；加号偏移限制在球内，避免超出
    // 球容器随全局缩放，ballRadius 与 maxOffset 乘 zoom 与渲染一致
    // ⑦ 优化：可见区裁剪（球心在节点边缘外，余量取大些）
    func hitTestBall(at screenPos: CGPoint) -> (UUID, ConnectSide, CGPoint)? {
        let zoom = currentZoom
        let offset = currentOffset
        let visibleRect = hitVisibleRect
        for node in nodes {
            let center = canvasToView(node.position, offset: offset, zoom: zoom)
            guard visibleRect.contains(center) else { continue }
            for side in [ConnectSide.left, ConnectSide.right] {
                let ballCenter = portCircleCenter(of: node, side: side, nodeSizes: nodeSizes, zoom: zoom, offset: offset, ballRadius: ballRadius)
                let dist = distance(from: screenPos, to: ballCenter)
                if dist <= ballRadius * zoom {
                    // 加号吸附偏移复用唯一入口 portSnapOffset（与拖拽命中 hitTestPort(includeOffset:) 一致）
                    return (node.id, side, portSnapOffset(screenPos: screenPos, ballCenter: ballCenter, zoom: zoom))
                }
            }
        }
        return nil
    }
}

// 命中检测：屏幕坐标是否落在某组盒左右两侧的加号球容器内（连线拖拽目标专用）
// 与整组选中拖出用的 hitTestGroupPort 同构，但**不要求整组选中**：拖拽连线中鼠标靠近任意组加号即可命中，
// 供 updateConnectDrag / finishConnectDrag 做目标吸附与建连；方向由调用方校验（目标端应为拖出端对侧）。
// 球心 = groupPortBallCenter（组盒 frame 外沿 + ballRadius*zoom），吸附复用唯一入口 portSnapOffset
func hitTestGroupPortForDrag(at screenPos: CGPoint, nodes: [CanvasNode], nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint) -> (UUID, ConnectSide, CGPoint)? {
    var groups: [UUID: [CanvasNode]] = [:]
    for node in nodes {
        if let gid = node.groupID {
            groups[gid, default: []].append(node)
        }
    }
    for (gid, groupNodes) in groups {
        let rect = nodesBoundingRect(groupNodes, nodeSizes: nodeSizes, zoom: zoom)
        for side in [ConnectSide.left, ConnectSide.right] {
            let ballCenter = groupPortBallCenter(rect: rect, side: side, zoom: zoom)
            let screenCenter = CGPoint(x: ballCenter.x + offset.x, y: ballCenter.y + offset.y)
            let dist = distance(from: screenPos, to: screenCenter)
            if dist <= nodeBallRadius * zoom {
                return (gid, side, portSnapOffset(screenPos: screenPos, ballCenter: screenCenter, zoom: zoom))
            }
        }
    }
    return nil
}

// ============================================================

// MARK: - 节点端口（加号）坐标与命中检测

// 端口（加号球）圆心：基于 nodeSizes 实际尺寸（未来节点尺寸可变时自动适配），而非硬编码半宽
// 球中心偏移统一走 ballCenterOffset(for:side:)（节点 ui.swift），与渲染侧 connectButton 共用同一来源
// 球心偏移随全局缩放乘 zoom（渲染侧 scaleEffect zoom），与加号视觉位置一致
func portCircleCenter(of node: CanvasNode, side: ConnectSide, nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint, ballRadius: CGFloat) -> CGPoint {
    let size = nodeSizes[node.id] ?? .zero
    let center = canvasToView(node.position, offset: offset, zoom: zoom)
    let ball = ballCenterOffset(for: size, side: side)
    return CGPoint(x: center.x + ball.x * zoom, y: center.y + ball.y * zoom)
}

// 节点端口（加号）的屏幕坐标：节点中心 ± (半宽 + 端口偏移)，半宽取 nodeSizes 实际尺寸
// 半宽随全局缩放乘 zoom（节点本体 scaleEffect zoom），连线端点贴缩放后的节点边缘
func portPosition(of node: CanvasNode, side: ConnectSide, nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint) -> CGPoint {
    let size = nodeSizes[node.id] ?? .zero
    let center = canvasToView(node.position, offset: offset, zoom: zoom)
    let halfWidth = size.width / 2 * zoom   // 节点实际宽度的一半（随缩放）
    let portOffset: CGFloat = 0   // 线头贴节点本体边缘
    let x = side == .left ? center.x - halfWidth - portOffset : center.x + halfWidth + portOffset
    return CGPoint(x: x, y: center.y)
}

// 命中检测：判断屏幕坐标是否落在某个节点的加号端口附近，返回命中的节点 id 与端口方向；
// includeOffset=true 时附带加号相对球心的吸附偏移（复用 portSnapOffset，供拖拽连线中目标加号跟随鼠标）。
// 命中位置对准加号球中心（节点边缘外 ballRadius 处），与连线端点（节点边缘）分离；
// 球半径随全局缩放乘 zoom（加号 scaleEffect zoom），与加号视觉尺寸一致
func hitTestPort(at screenPos: CGPoint, nodes: [CanvasNode], nodeSizes: [UUID: CGSize], zoom: Double, offset: CGPoint, includeOffset: Bool = false) -> (nodeID: UUID, side: ConnectSide, portOffset: CGPoint)? {
    let threshold = nodeBallRadius * zoom   // 加号球半径（与 NodeView 共用常量，随缩放）
    for node in nodes {
        for side in [ConnectSide.left, ConnectSide.right] {
            let port = portCircleCenter(of: node, side: side, nodeSizes: nodeSizes, zoom: zoom, offset: offset, ballRadius: threshold)
            let dist = distance(from: port, to: screenPos)
            // 拖拽吸附命中用 <=（与渲染侧视觉一致）；松手建连用 <（保持原精确判定）
            let hit = includeOffset ? dist <= threshold : dist < threshold
            if hit {
                let snap = includeOffset ? portSnapOffset(screenPos: screenPos, ballCenter: port, zoom: zoom) : .zero
                return (node.id, side, snap)
            }
        }
    }
    return nil
}

// 加号吸附偏移（唯一入口）：鼠标相对球心偏移，限制在球内（加号半径 portSize/2，随缩放乘 zoom）。
// 命中侧 hitTestBall 与拖拽命中 hitTestPort(includeOffset:) 共用，保证吸附运动方式一致。
func portSnapOffset(screenPos: CGPoint, ballCenter: CGPoint, zoom: Double) -> CGPoint {
    let dx = screenPos.x - ballCenter.x
    let dy = screenPos.y - ballCenter.y
    let dist = distance(from: screenPos, to: ballCenter)
    let maxOffset = (nodeBallRadius - nodePortSize / 2) * zoom
    let scale = min(1.0, maxOffset / max(dist, 0.001))
    return CGPoint(x: dx * scale, y: dy * scale)
}

// 命中检测（带吸附偏移）：已合并进 hitTestPort(includeOffset: true)，见上
