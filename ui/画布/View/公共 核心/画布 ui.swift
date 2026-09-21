// ============================================================
//  文件作用：画布主视图。定义 CanvasView（组装网格背景、节点交互层、鼠标控制层、
//  隔离拦截层、工具栏、资产库、浮层面板层）、DraggableGrid 网格背景、
//  CanvasEditorView 画布编辑器（含返回与改名）。
//  浮层面板层 floatingPanelLayer：所有悬浮面板（大纲视图/右键菜单/生成面板/
//  添加节点面板）统一放在此层，与底层画布完全隔离；面板打开时隔离拦截层
//  覆盖画布拦截点击，后续新增面板只需在此层追加。
//  互动文件：引用 画布 状态.swift（CanvasStore、坐标换算）、交互.swift
//  （NodeCanvasView、MouseControlView）、节点ui.swift（NodeType、ConnectSide 等）、
//  资产管理ui.swift（AssetStore）、创作ui.swift（ProjectStore），以及 画布 工具栏.swift /
//  工具栏2.swift / 菜单.swift / 资产面板.swift / 节点操作.swift 的 CanvasView 扩展；
//  被 主ui.swift 引用（CanvasEditorView）。
// ============================================================

import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers
import AVFoundation   // 视频生成：AVURLAsset / AVAssetImageGenerator（提取首帧缩略图）

// ============================================================

// MARK: - 无限网格画布视图

/// 大纲视图面板的 Tab 类型：大纲视图 / 资产库
enum OutlinePanelTab: String, CaseIterable {
    case outline = "大纲视图"
    case asset = "资产库"
}

struct CanvasView: View {
    @EnvironmentObject var store: CanvasStore
    @EnvironmentObject var projectStore: ProjectStore
    @ObservedObject var assetStore = AssetStore.shared
    @ObservedObject var modelDownloadManager = ModelDownloadManager.shared
    @Binding var canvasName: String
    @State var showRatioMenu = false
    @State var showColorMenu = false
    @State var showDurationMenu = false
    @State var hoveredColor: Color?
    @State var hoveredRatio: CanvasStore.Ratio?
    @State var hoveredDuration: VideoDuration?
    @State var showContextMenu = false
    @State var showAddNodePanel = false
    /// 左侧工具栏加号：添加节点面板（与右键添加节点面板独立）
    @State var showToolbarAddPanel = false
    /// 左侧工具栏：大纲视图面板（列出当前画布所有节点）
    @State var showOutlinePanel = false
    /// 大纲视图：按名字搜索
    @State var outlineSearchText = ""
    /// 大纲视图：按类型过滤（nil 表示全部）
    @State var outlineTypeFilter: NodeType? = nil
    /// 大纲视图：悬停的节点 id（用于行悬停高亮）
    @State var hoveredOutlineNodeID: UUID? = nil
    /// 大纲视图：已展开的组 id 集合（展开显示组内节点）
    @State var outlineExpandedGroupIDs: Set<UUID> = []
    /// 大纲视图：悬停的组 id（用于组行悬停高亮）
    @State var hoveredOutlineGroupID: UUID? = nil
    /// 大纲视图：重命名面板对应的节点 id（nil 表示未弹出）
    @State var outlineRenameNodeID: UUID? = nil
    /// 大纲视图：重命名输入框文本
    @State var outlineRenameText = ""
    @State var outlineScrollTarget: String? = nil   // 大纲视图滚动聚焦目标行 id
    /// 左侧工具栏在窗口中的位置（用于从工具栏添加节点时定位）
    @State var leftToolbarFrame: CGRect = .zero
    /// 画布视图尺寸（用于大纲视图聚焦节点时计算居中偏移）
    @State var canvasViewSize: CGSize = .zero
    @State var contextMenuPos: CGPoint = .zero
    /// 右键命中的节点 id（nil 表示点在空白处）
    @State var contextMenuNodeID: UUID?
    /// 右键来源：true=大纲内节点右键，false=画布内节点右键（决定菜单项）
    @State var contextMenuFromOutline = false
    /// 拖拽连线在空白处松手时弹出的「引用该节点生成」面板
    @State var showGeneratePanel = false
    @State var generatePanelPos: CGPoint = .zero
    @State var generateFromNodeID: UUID?
    @State var generateFromSide: ConnectSide?
    /// 节点往期内容面板：正在展示的节点 id（nil = 未打开）
    @State var showHistoryPanelNodeID: UUID?
    /// 节点往期内容面板：锚定位置（节点右侧屏幕坐标）
    @State var historyPanelPos: CGPoint = .zero
    /// 大纲视图面板的 Tab：大纲视图 / 资产库
    @State var outlinePanelTab: OutlinePanelTab = .outline
    /// 资产库过滤范围：nil 表示全局资产，否则为对应项目 id
    @State var assetScope: UUID? = nil
    /// 资产库分类 Tab：角色 / 场景
    @State var assetCategory: AssetCategory = .character
    /// 资产库中已选中的资产 id（支持批量多选后拖拽）
    @State var selectedAssetIDs: Set<UUID> = []
    /// 资产库框选：框选矩形 / 框选临时命中的资产
    @State var assetSelectionRect: CGRect?
    @State var assetRubberSelected: Set<UUID> = []
    /// 资产库中每个资产项在面板坐标系中的 frame（用于框选命中检测）
    @State var assetFrames: [UUID: CGRect] = [:]
    /// 空节点「资产库」选择面板：目标节点 id（nil 表示未弹出）
    @State var pickAssetNodeID: UUID?
    /// 资产库选择面板：当前分类（随节点类型初始化）
    @State var pickAssetCategory: AssetCategory = .character
    /// 资产库选择面板：来源过滤（nil=全局）
    @State var pickAssetScope: UUID? = nil
    /// 资产库选择面板：当前选中资产
    @State var pickSelectedAssetID: UUID?
    
    /// 节点下方输入框：草稿（失焦/发送/展开关闭/离开画布时写回，引用模型下写回资产或节点自身）
    @State var promptDraft = ""
    /// 节点输入校验错误（视频节点发送时检查直接连线输入；key=节点 id，非空表示要求不对）
    @State var inputValidationErrors: [UUID: String] = [:]
    /// 节点输入框展开聚焦模式：当前展开的节点 id（nil = 未展开）
    @State var expandedPromptNodeID: UUID?
    /// 节点下方输入框卡片真实高度（由 background(GeometryReader) 上报，用于精确贴底定位）
    @State var nodePromptBarHeight: CGFloat = 140
    
    // 网格参数
    let gridSize: Double = 40
    let gridColor: Color = Color(red: 0.85, green: 0.85, blue: 0.88)
    let majorGridColor: Color = Color(red: 0.75, green: 0.75, blue: 0.78)

    /// 模型下载面板（画布中间）：下载时显示标题+总进度+活动文件列表，水色跟随全局颜色
    @ViewBuilder
    private var modelDownloadProgressView: some View {
        if modelDownloadManager.isDownloading || modelDownloadManager.errorMessage != nil {
            ModelDownloadPanel(
                progress: modelDownloadManager.progress,
                activeFiles: modelDownloadManager.activeFiles,
                errorMessage: modelDownloadManager.errorMessage,
                accentColor: AppSettings.shared.defaultNodeColor,
                onRetry: { modelDownloadManager.retryDownload() },
                onDismiss: { modelDownloadManager.dismissError() }
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // 下载中纯展示不拦截画布交互；错误态放行以便"重试/关闭"按钮可点击
            .allowsHitTesting(modelDownloadManager.errorMessage != nil)
            .transition(.opacity)
        }
    }

    var body: some View {
        ZStack {
            // 背景网格（可拖拽）
            DraggableGrid(
                offset: $store.offset,
                gridSize: gridSize,
                gridColor: gridColor,
                majorGridColor: majorGridColor,
                showGrid: store.showGrid,
                zoom: store.zoom
            )
            
            // 节点交互层（节点渲染 + 连线渲染 + 资产拖拽落点 + 连线回调）
            NodeCanvasView()
            
            // 鼠标控制层（左键拖拽节点/框选 + 中键拖动移动 / 滚轮滚动 / Command+滚轮缩放 / 右键菜单）
            MouseControlView(
                offset: $store.offset,
                zoom: $store.zoom,
                gridSize: gridSize,
                snapToGrid: store.snapToGrid,
                viewSize: canvasViewSize,
                isIsolated: isAnyPanelOpen,
                nodes: store.nodes,
                nodeSizes: store.nodeSizes,
                selectedNodeIDs: store.selectedNodeIDs,
                connections: store.connections,
                selectedConnectionID: store.selectedConnectionID,
                playingNodeID: MediaPlayerManager.shared.playingNodeID,
                onDragEnded: {
                    store.save()
                },
                onRightClick: { point, nodeID in
                    contextMenuPos = point
                    contextMenuNodeID = nodeID
                    contextMenuFromOutline = false
                    showAddNodePanel = false
                    showToolbarAddPanel = false
                    withAnimation { showContextMenu = true }
                },
                onNodeDragStart: { startPositions in
                    store.pushSnapshot()   // 移动节点前记录快照（用于撤销）
                    store.beginNodeDrag(startPositions: startPositions)   // 记录拖动起始位置，拖动中不写 nodes
                },
                onNodeDrag: { id, newPos in
                    // 拖动期间只更新整体位移（渲染侧对拖动节点叠加显示），松手才合并写回，避免每帧重建内容层
                    if let start = store.draggingStartPositions[id] {
                        let offset = CGPoint(x: newPos.x - start.x, y: newPos.y - start.y)
                        if offset != store.nodeDragOffset {
                            store.nodeDragOffset = offset
                        }
                    }
                },
                onNodeDragEnded: {
                    store.commitNodeDrag()   // 把位移一次性合并进节点 position
                    store.save()
                },
                onSelectionChanged: { rect in
                    store.selectionRect = rect
                    // 开始框选（空白处按下）：取消连线选中
                    store.selectedConnectionID = nil
                },
                onSelectionEnded: { ids in
                    store.selectedNodeIDs = Set(ids)
                    store.selectionRect = nil
                },
                onNodeClick: { id in
                    store.endNodeDrag()   // 单击未拖动：清掉 beginNodeDrag 残留的拖动状态
                    store.selectedNodeIDs = [id]
                    store.selectedConnectionID = nil
                },
                onSelectGroup: { nodeIDs in
                    // 单击组盒空白：选中整组（组内节点全部选中，形成完整组选中态）
                    store.endNodeDrag()
                    store.selectedNodeIDs = Set(nodeIDs)
                    store.selectedConnectionID = nil
                },
                onConnectionClick: { connID in
                    // 点击连线：选中该线（虚线流动 + 中心裁断按钮），取消节点选中
                    store.selectedNodeIDs = []
                    store.selectedConnectionID = connID
                },
                onCutConnection: { connID in
                    // 点击裁断按钮：移除该连线
                    store.pushSnapshot()   // 裁断连线前记录快照（用于撤销）
                    store.connections.removeAll { $0.id == connID }
                    store.selectedConnectionID = nil
                    store.save()
                },
                onAlignmentGuides: { guides in
                    store.alignmentGuides = guides
                },
                onHoverNode: { id, side, offset in
                    store.hoveredNodeID = id
                    store.hoveredPortSide = side
                    store.hoveredPortOffset = offset
                },
                onHoverGroupPort: { gid, side, offset in
                    // 组盒加号悬停态（鼠标控制层 mouseMoved 命中）：与拖拽目标态共用同一状态，
                    // 渲染层 groupPortBall 亮出该组对侧加号并吸附
                    store.hoveredGroupPortID = gid
                    store.hoveredGroupPortSide = side
                    store.hoveredGroupPortOffset = offset
                    // 仅真正命中组盒加号（gid 非 nil）时才与节点悬停互斥清节点态；
                    // nil 清除组态调用（如节点加号命中后的 onHoverGroupPort?(nil,nil,nil)）不得误伤节点悬停
                    if gid != nil {
                        store.hoveredNodeID = nil
                        store.hoveredPortSide = nil
                        store.hoveredPortOffset = nil
                    }
                },
                onPortDragChanged: { nodeID, screenPos, side in
                    // 加号连线拖拽中：公共入口更新临时连线 + 命中检测（与 ConnectButton 拖拽共用）
                    updateConnectDrag(screenPos: screenPos, fromNodeID: nodeID, side: side, store: store)
                },
                onPortDragEnded: { nodeID, screenPos, side in
                    // 加号连线拖拽结束：公共入口完成建连；未命中时在松手位置弹出「引用该节点生成」面板
                    let connected = finishConnectDrag(screenPos: screenPos, fromNodeID: nodeID, side: side, store: store)
                    if !connected {
                        generatePanelPos = screenPos
                        generateFromNodeID = nodeID
                        generateFromSide = side
                        showContextMenu = false
                        showAddNodePanel = false
                        withAnimation { showGeneratePanel = true }
                    }
                },
                onEmptyNodeButtonClick: { nodeID, action in
                    handleEmptyNodeButtonClick(nodeID: nodeID, action: action)
                },
                onEmptyNodeButtonHover: { nodeID, action in
                    store.hoveredEmptyButtonNodeID = nodeID
                    store.hoveredEmptyButtonAction = action
                },
                onExternalDrop: { urls, viewPos in
                    // Finder 拖入媒体文件：落点是视图坐标，先复用公共函数 viewToCanvas 换算为画布坐标再导入成节点，
                    // 避免 importAsset 内部 canvasToView→addNode viewToCanvas 两次变换抵消导致节点偏离落点（平移/缩放后更明显）；
                    // 取帧/解码放后台（receiveDroppedMedia 内部 await thumbnailImage），回主线程仅做建节点/更新 UI
                    let canvasPos = viewToCanvas(viewPos, offset: store.offset, zoom: store.zoom)
                    let targetStore = store
                    Task { @MainActor in
                        await receiveDroppedMedia(urls: urls, store: targetStore, nodePosition: canvasPos)
                    }
                },
                onPlayButtonClick: { nodeID in
                    // 播放按钮点击：切换播放/停止（按钮只在已关联媒体文件的音频/视频节点上显示）
                    if let node = store.nodes.first(where: { $0.id == nodeID }) {
                        MediaPlayerManager.shared.toggle(node: node)
                    }
                },
                onHistoryBadgeClick: { nodeID in
                    // 历史徽章点击：在节点右侧展开「往期内容」面板
                    guard let node = store.nodes.first(where: { $0.id == nodeID }) else { return }
                    openHistoryPanel(for: node)
                },
                onTailFrameSwitchClick: { nodeID in
                    // 尾帧开关点击：切换 on/off（状态存节点模型，数据单一来源；渲染侧按状态显示主题色轨道+手柄在右）
                    if let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) {
                        store.nodes[idx].tailFrameEnabled.toggle()
                    }
                },
                onProgressSeek: { nodeID, ratio in
                    // 播放进度条拖动：跳转播放进度
                    MediaPlayerManager.shared.seek(nodeID: nodeID, ratio: ratio)
                }
            )
            
            // 隔离拦截层：有浮层面板打开时覆盖画布，拦截点击防止穿透到画布。
            // 位于工具栏之下：面板打开时工具栏仍可点击（如再次点击大纲按钮关闭面板）。
            if isAnyPanelOpen {
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture {
                        debugLog("画布：点击空白关闭浮层面板")
                        closeAllPanels()
                    }
            }
            
            // 右上角：比例按钮 + 比例菜单（从按钮下方延伸）
            topRightView
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
            
            // 左下角：工具按钮
            bottomLeftView
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
            
            // 左侧居中：竖直工具面板
            leftSideToolbar
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
            
            // 浮层面板层：所有悬浮面板（大纲视图/右键菜单/生成面板/添加节点面板）
            // 统一放在此层，与底层画布完全隔离；后续新增面板在此层追加即可
            floatingPanelLayer
            
            // 快捷键处理层（透明视图，捕获 F2/空格/Ctrl+Z/Ctrl+Alt+Z，
            // 与工具栏按钮操作等同；文本输入框聚焦时不拦截）
            KeyboardShortcutHandler(
                onRename: {
                    // F2：对当前选中节点进入行内重命名（自动打开大纲面板）
                    if let nodeID = store.selectedNodeIDs.first,
                       let node = store.nodes.first(where: { $0.id == nodeID }) {
                        beginOutlineRename(node)
                    }
                },
                onFitZoom: {
                    // 空格：缩放适配（重置偏移/缩放/比例，与工具栏按钮一致：重置后落盘）
                    store.reset()
                    store.save()
                },
                onUndo: {
                    // Ctrl+Z：撤销
                    store.undo()
                },
                onRedo: {
                    // Ctrl+Alt+Z：重做
                    store.redo()
                },
                onCopy: {
                    // Command+C：复制选中节点（含节点间连线）到剪贴板
                    copySelectedNodesToClipboard()
                },
                onPaste: {
                    // Command+V：从剪贴板粘贴节点
                    pasteNodesFromClipboard()
                },
                onDelete: {
                    // Delete：优先删除选中的连线（等效裁断按钮，记录快照可撤销）
                    if let connID = store.selectedConnectionID {
                        store.pushSnapshot()
                        store.connections.removeAll { $0.id == connID }
                        store.selectedConnectionID = nil
                        store.save()
                        debugLog("Delete：删除选中连线")
                    } else if selectedIsCompleteGroup {
                        // 选中完整组（盒子）时解散组、组内节点不动
                        ungroupSelectedGroup()
                    } else if !store.selectedNodeIDs.isEmpty {
                        deleteNodes(store.selectedNodeIDs)
                    }
                }
            )
            
            // ===== 节点输入框 / 打组 浮层（放最上层：跟随选中节点、多选打组、展开聚焦模式）=====
            nodePromptOverlay
            
            // ===== 历史面板（置于输入框浮层之上：避免被节点下方输入框遮挡；展开聚焦模式时不显示）=====
            if let nodeID = showHistoryPanelNodeID, expandedPromptNodeID == nil {
                historyPanel(nodeID: nodeID)
                    .position(historyPanelPos)
            }
            
            // ===== 模型下载试管进度条（画布中间）：下载时显示，水色跟随全局颜色 =====
            modelDownloadProgressView
        }
        .background(GeometryReader { geo in
            Color.clear
                .onAppear {
                    canvasViewSize = geo.size
                    store.canvasViewSize = geo.size
                }
                .onChange(of: geo.size) { _, newSize in
                    canvasViewSize = newSize
                    store.canvasViewSize = newSize
                }
        })
        // 画布出现（从资产管理 Tab 切回 / 进入项目）时，把资产面板改过的提示词批量同步到节点
        // 与播放逻辑同方案：资产管理 Tab 与画布互斥，离开总资产期间 onChange 收不到事件，
        // 依赖视图生命周期（onDisappear 停止播放 / onAppear 同步提示词）完成跨区域通信
        .onAppear {
            if !assetStore.promptEditedAssetIDs.isEmpty {
                syncNodesFromAssetPrompts(assetStore.promptEditedAssetIDs)
                assetStore.promptEditedAssetIDs.removeAll()
            }
            // 生成队列完成回调：把产物接入对应节点（图片 → attachGeneratedImage，视频 → attachGeneratedVideo）
            GenerationQueue.shared.onTaskResult = { task, outputPath in
                guard let outputPath else { return }
                // 视频接入需后台取首帧（attachGeneratedVideo 内部 await frameImage），主线程挂起让出不阻塞；
                // 完成后继续在主线程更新节点（Task @MainActor），连续视频传播保持在节点更新之后触发
                Task { @MainActor in
                    switch task.kind {
                    case .image:
                        self.attachGeneratedImage(to: task.nodeID, pngPath: outputPath, prompt: task.prompt)
                    case .video:
                        await self.attachGeneratedVideo(to: task.nodeID, mp4Path: outputPath, prompt: task.prompt)
                        // 连续视频模式：视频完成后沿输出连线自动触发下游视频节点（禁用连线跳过，visited 防环）
                        if self.store.continuousVideoMode {
                            self.propagateContinuousVideo(from: task.nodeID, visited: [])
                        }
                    }
                }
            }
            // 任务取消回调：移除取消节点相关的发光线（指向它或从它出发的线）
            GenerationQueue.shared.onTaskCancelled = { nodeID in
                store.glowingConnectionIDs = store.glowingConnectionIDs.filter { connID in
                    guard let conn = store.connections.first(where: { $0.id == connID }) else { return false }
                    return conn.toID != nodeID && conn.fromID != nodeID
                }
            }
        }
        // 离开画布（退出项目 / 切走）时兜底提交输入框草稿：未失焦/未发送的提示词也增量写回节点并同步资产
        .onDisappear {
            commitPromptDraft()
        }
        // 选中节点变化：写回旧节点草稿，载入新节点草稿（多选/取消选中清空草稿）
        .onChange(of: store.selectedNodeIDs) { oldIDs, newIDs in
            if oldIDs.count == 1, let oldID = oldIDs.first,
               let oldNode = store.nodes.first(where: { $0.id == oldID }),
               promptDraft != effectivePrompt(for: oldNode) {
                commitNodePrompt(promptDraft, nodeID: oldID)
            }
            if newIDs.count == 1, let newID = newIDs.first,
               let newNode = store.nodes.first(where: { $0.id == newID }) {
                promptDraft = effectivePrompt(for: newNode)
            } else {
                promptDraft = ""
            }
        }
        // 资产面板提示词改动：同步画布中引用该资产的节点输入框
        .onChange(of: assetStore.promptEditedAssetIDs) { _, ids in
            guard !ids.isEmpty else { return }
            syncNodesFromAssetPrompts(ids)
            assetStore.promptEditedAssetIDs.removeAll()
        }
        // 空节点「资产库」选择面板（sheet 弹出，右下角确定后设为节点实际内容）
        .sheet(isPresented: Binding(
            get: { pickAssetNodeID != nil },
            set: { if !$0 { pickAssetNodeID = nil } }
        )) {
            NodeAssetPickerView(
                assetStore: assetStore,
                projectStore: projectStore,
                category: $pickAssetCategory,
                scope: $pickAssetScope,
                selectedAssetID: $pickSelectedAssetID,
                allowedCategories: pickAssetNodeID.flatMap { id in
                    store.nodes.first(where: { $0.id == id })?.type.allowedAssetCategories
                } ?? [],
                onConfirm: { asset in
                    confirmPickAsset(asset)
                },
                onCancel: {
                    pickAssetNodeID = nil
                }
            )
        }
    }
    
    // MARK: - 面板互斥：关闭所有浮层面板（6 个按钮面板 + 右键菜单 + 生成面板）
    
    func closeAllPanels() {
        // 正在编辑节点名称时，关闭面板等同确认编辑（提交重命名）
        if let renameID = outlineRenameNodeID,
           let node = store.nodes.first(where: { $0.id == renameID }) {
            commitOutlineRename(node)
        }
        showRatioMenu = false
        showColorMenu = false
        showDurationMenu = false
        showContextMenu = false
        showAddNodePanel = false
        showGeneratePanel = false
        showToolbarAddPanel = false
        showOutlinePanel = false
        showHistoryPanelNodeID = nil
    }
    
    // MARK: - 浮层面板层：与底层画布隔离的所有悬浮面板
    
    /// 是否有浮层面板打开（决定是否显示隔离拦截层）
    var isAnyPanelOpen: Bool {
        showRatioMenu || showColorMenu || showContextMenu || showGeneratePanel || showOutlinePanel || showToolbarAddPanel || showHistoryPanelNodeID != nil
    }
    
    // MARK: - 节点往期内容面板
    
    /// 打开节点往期内容面板：锚定在节点右侧（节点中心 + 半宽 + 间距，屏幕坐标）
    func openHistoryPanel(for node: CanvasNode) {
        // 若已在展示该节点则直接关闭（切换开关）
        if showHistoryPanelNodeID == node.id {
            closeAllPanels()
            return
        }
        let zoom = store.zoom
        let offset = store.offset
        let center = CGPoint(
            x: node.position.x * zoom + offset.x,
            y: node.position.y * zoom + offset.y
        )
        let nodeHalfW = nodeWidthForType(node.type) * zoom / 2
        historyPanelPos = CGPoint(x: center.x + nodeHalfW + 12, y: center.y)
        showContextMenu = false
        showGeneratePanel = false
        showOutlinePanel = false
        withAnimation { showHistoryPanelNodeID = node.id }
    }

    /// 节点往期内容面板：展示往期图片，上下排列有间隔；点击图片替换为节点实际内容（含标题与提示词）。
    /// 尺寸适配节点与全局缩放：横向略小于节点宽、竖向高于节点宽；zoom 缩放时随节点同步变化。
    @ViewBuilder
    func historyPanel(nodeID: UUID) -> some View {
        // 面板基准尺寸：横向 = 节点宽 × 0.85，竖向 = 节点宽 × 1.2（随 store.zoom 联动）
        // 宽度必须固定（防止父容器拉伸占满窗口），并设上下限保证可用性
        let base = store.nodes.first(where: { $0.id == nodeID }).map { nodeWidthForType($0.type) * store.zoom } ?? 280
        let panelW = min(max(base * 0.85, 220), 460)
        let panelH = min(max(base * 0.9, 220), 400)

        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("结果")
                    .font(.headline)
                if let node = store.nodes.first(where: { $0.id == nodeID }) {
                    Text("· \(node.history?.count ?? 0)")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button {
                    closeAllPanels()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .frame(width: 20, height: 20)
                        .background(Circle().fill(Color.gray.opacity(0.12)))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 14)
            .padding(.top, 12)
            .padding(.bottom, 8)

            if let node = store.nodes.first(where: { $0.id == nodeID }),
               let history = node.history, !history.isEmpty {
                // 不用 ScrollView：它会拦截滚轮事件导致画布无法滚动/缩放；改用裁剪的 VStack，滚轮穿透到画布
                VStack(spacing: 12) {
                    ForEach(history.reversed()) { entry in
                        historyImageCell(entry: entry, cellWidth: panelW - 24, imageHeight: panelW * 0.62) {
                            restoreNodeHistory(to: nodeID, entry: entry)
                            closeAllPanels()
                        }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 480)
                .clipped()
                Text("点击图片可替换当前内容")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.bottom, 10)
            } else {
                Text("还没有往期内容，重新生成时旧内容会自动保留在这里")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 14)
                    .padding(.bottom, 12)
            }
        }
        .padding(.vertical, 6)
        .frame(width: panelW)
        .frame(minHeight: panelH, maxHeight: 430, alignment: .top)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white.opacity(0.97))
                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        )
    }

    /// 往期图片格：大图铺满 + 右下角时间角标；点击整格替换（无按钮视觉，纯图）
    /// 固定宽度 + clipped 防止 aspectRatio(.fill) 溢出容器
    private func historyImageCell(entry: NodeContentHistory, cellWidth: CGFloat, imageHeight: CGFloat, onTap: @escaping () -> Void) -> some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if let imgName = entry.imageFileName, let img = cachedAssetImage(for: imgName) {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: cellWidth, height: imageHeight)
                        .clipped()
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                } else if entry.mediaFileName != nil {
                    VStack(spacing: 8) {
                        Image(systemName: "music.note")
                            .font(.system(size: 28))
                            .foregroundColor(.secondary)
                        Text("音频内容")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(width: cellWidth, height: imageHeight)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color.gray.opacity(0.1)))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 28))
                            .foregroundColor(.secondary)
                        Text("无图内容")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(width: cellWidth, height: imageHeight)
                    .background(RoundedRectangle(cornerRadius: 10).fill(Color.gray.opacity(0.1)))
                }
            }
            Text(entry.createdAt.formatted(date: .omitted, time: .shortened))
                .font(.system(size: 9))
                .foregroundColor(.white)
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Capsule().fill(Color.black.opacity(0.45)))
                .padding(8)
        }
        .overlay(alignment: .topLeading) {
            Text(entry.title)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(Capsule().fill(Color.black.opacity(0.5)))
                .padding(8)
        }
        .contentShape(RoundedRectangle(cornerRadius: 10))
        .onTapGesture(perform: onTap)
    }
    
    // MARK: - 节点输入框 / 打组 浮层
    
    /// 当前单选选中的节点（输入框只服务单选）
    func selectedPromptNode() -> CanvasNode? {
        guard store.selectedNodeIDs.count == 1, let id = store.selectedNodeIDs.first,
              let node = store.nodes.first(where: { $0.id == id }) else { return nil }
        return node
    }
    
    /// 节点当前生效的提示词（引用模型，单一数据源）：
    /// 节点引用资产时读资产库 prompt（改源头/改节点都改这一份）；
    /// 未引用资产（文本/空节点）用节点自带 prompt。
    func effectivePrompt(for node: CanvasNode) -> String {
        // 节点自身有副本优先用节点自己的（编辑过 / 生成条件已写入）；
        // 副本为空且引用资产时引用资产源头提示词（兼容旧数据与"从资产库选择"场景）
        if !node.prompt.isEmpty {
            return node.prompt
        }
        if let fileName = node.imageFileName,
           let asset = assetStore.assets.first(where: { $0.fileName == fileName }) {
            return asset.prompt
        }
        return node.prompt
    }

    /// 节点输入框草稿统一写回（引用模型：只写单一数据源）。
    /// 节点引用资产 → 写资产库 prompt，所有引用该资产的节点天然同步；
    /// 未引用资产 → 写节点自身 prompt。
    func commitNodePrompt(_ prompt: String, nodeID: UUID) {
        guard let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) else { return }
        // 节点提示词是"引用源头的本地副本"：编辑只写节点自身，不改总资产源头
        store.nodes[idx].prompt = prompt
        store.save()
    }

    /// 资产面板提示词改动（引用模型：资产是单一数据源，节点不存副本）。
    /// 只需刷新当前选中节点的输入框草稿；其他节点下次选中时自然读到资产新值。
    func syncNodesFromAssetPrompts(_ ids: [UUID]) {
        guard let node = selectedPromptNode(),
              let fileName = node.imageFileName,
              let asset = assetStore.assets.first(where: { $0.fileName == fileName }),
              ids.contains(asset.id) else { return }
        promptDraft = asset.prompt
    }

    /// 草稿写回节点（失焦/发送/展开关闭时调用；输入过程中不入撤销栈、不落盘，避免打字每帧存盘）
    func commitPromptDraft() {
        guard let node = selectedPromptNode(), promptDraft != effectivePrompt(for: node) else { return }
        commitNodePrompt(promptDraft, nodeID: node.id)
    }
    
    /// 发送输入框内容：写回 + 视频节点触发生成管线
    func sendPrompt() {
        guard let node = selectedPromptNode() else { return }
        commitPromptDraft()
        let promptText = promptDraft
        debugLog("节点输入框：发送「\(promptText)」，节点 \(node.title)")
        // 发送后保留输入框文本，供用户继续编辑（再次发送视为全新任务，不复用旧提示词/旧文本条件）
        // 落盘由 commitPromptDraft→commitNodePrompt 完成（草稿与生效提示词不同时才写）；此处无需重复 save
        // 视频节点：点发送即开始生成视频（文本节点等只写回提示词，不触发）
        if node.type == .video {
            startVideoGeneration(nodeID: node.id, prompt: promptText)
        }
        // 图像/角色/场景节点：点发送即开始生成图像（HiDream-O1）
        if node.type == .image || node.type == .character || node.type == .scene {
            startImageGeneration(nodeID: node.id, prompt: promptText)
        }
    }
    
    /// 视频生成入口：后台执行 LTX2.5 管线，完成后把成品 mp4 接入节点（替换实际内容）
    /// 生成期间发送按钮转圈禁用（videoGeneratingNodeID 非 nil），防止重复提交。
    /// 发送前校验直接连线输入（只查上一层连线，不递归）：
    /// - 图片节点 + 有内容且开尾帧的视频节点 → 图片条件（最多 2 张）
    /// - 音频节点 → 音频条件（最多 1 个）
    /// - 视频节点（无尾帧）→ 视频条件（最多 1 个）
    /// 超限则取消发送，写 inputValidationErrors（输入框感叹号变红，点击显示报错原因）。
    /// 空节点跳过；校验通过后把图片/音频条件传给管线。
    func startVideoGeneration(nodeID: UUID, prompt: String, visited: Set<UUID> = []) {
        // 连续视频模式级联防环：同一传播链内已触发过的节点不再重复触发
        if visited.contains(nodeID) {
            debugLog("连续视频模式：节点 \(nodeID) 已在本次传播链中，跳过防环")
            return
        }
        // 连续视频模式：视频节点必须开启尾帧，否则取消本次生成（关闭模式时不拦截，行为与现状一致）
        if store.continuousVideoMode,
           let candidate = store.nodes.first(where: { $0.id == nodeID }),
           candidate.type == .video,
           !candidate.tailFrameEnabled {
            // 取消生成：该链上无下一个任务，发光线全部熄灭
            store.glowingConnectionIDs.removeAll()
            debugLog("连续视频模式：节点尾帧未开启，取消生成")
            return
        }
        // 发光提示（手动与级联统一入口）：入队前清空旧线，只亮"下一个即将生成"的那一条——
        // 按顺序找第一条输出连线：目标为视频节点、尾帧开启、不在防环集合中、且该连线对目标有效（非禁用）。
        // 手动触发 1 亮 1→2；级联触发 B 入队时自动从 A→B 换到 B→C；找不到下游则全灭。
        if store.continuousVideoMode {
            store.glowingConnectionIDs.removeAll()
            if let nextConn = store.connections.first(where: { conn in
                guard conn.fromID == nodeID,
                      let target = store.nodes.first(where: { $0.id == conn.toID }),
                      target.type == .video,
                      target.tailFrameEnabled,
                      !visited.contains(target.id) else { return false }
                let validity = store.inputValidity(for: target.id)
                return !(validity.applies && validity.invalidConnectionIDs.contains(conn.id))
            }) {
                store.glowingConnectionIDs.insert(nextConn.id)
            }
        }
        // 收集有效输入：走公共判定（视频节点选了模型后，不符合要求的线禁用跳过，只收集合规线）
        let validity = store.inputValidity(for: nodeID)
        let imagePaths = validity.imagePaths
        let videoPaths = validity.videoPaths
        let audioPaths = validity.audioPaths
        if !validity.invalidConnectionIDs.isEmpty {
            debugLog("视频生成：存在 \(validity.invalidConnectionIDs.count) 条无效连线（已禁用跳过），本次仅用合规输入")
        }
        // 模型专属阻断：H3 无任何有效条件输入时拒绝发送并给出明确原因
        if let reason = validity.errorMessage {
            // H3 链式放行（图空也放行）：有 .h3cc 走 latent 续接；无 .h3cc / 无上游开尾帧视频
            // 统一回退为文本生成（文生视频），仅 toast 提示。
            if let srcID = store.h3ChainSourceID(for: nodeID) {
                let cacheRoot = "\(AppSettings.shared.canvasRootPath)/cache/h3-continuation"
                let cachePath = cacheRoot + "/" + srcID.uuidString + ".h3cc"
                if FileManager.default.fileExists(atPath: cachePath) {
                    debugLog("视频生成：H3 无图条件但存在有效续接源（.h3cc），按 latent 续接继续")
                } else {
                    store.toastMessage = "上一个节点没有latent 且无条件图 本次为文本生成"
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        if store.toastMessage == "上一个节点没有latent 且无条件图 本次为文本生成" {
                            store.toastMessage = nil
                        }
                    }
                    debugLog("视频生成：H3 无图条件，上游 \(srcID) 无 .h3cc 缓存，按文本生成继续")
                }
            } else {
                // 无上游开尾帧视频 + 无条件图：同样放行，管线走纯文本生成（T2V）
                store.toastMessage = "上一个节点没有latent 且无条件图 本次为文本生成"
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    if store.toastMessage == "上一个节点没有latent 且无条件图 本次为文本生成" {
                        store.toastMessage = nil
                    }
                }
                debugLog("视频生成：H3 无图条件且无上游续接源（\(reason)），按文本生成继续")
            }
        }
        // 误触保护：提示词为空且无任何输入条件（图片/视频/音频）时直接取消，不启动管线
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPrompt.isEmpty && imagePaths.isEmpty && videoPaths.isEmpty && audioPaths.isEmpty {
            inputValidationErrors[nodeID] = "提示词为空且未连接图片/视频/音频条件，已取消运行（请填写提示词或接入条件）"
            debugLog("视频生成：误触保护（无提示词、无图片、无视频、无音频），取消运行")
            return
        }
        inputValidationErrors[nodeID] = nil
        debugLog("视频生成：输入收集完成（图片 \(imagePaths.count) 张，视频 \(videoPaths.count) 个，音频 \(audioPaths.count) 个）")
        // 尺寸（动态表）：视频节点不被图片输入比例劫持——有前置视频（尾帧续接链路）时跟随前置实际尺寸
        // 保证续接指纹 width/height 一致；非续接视频节点按节点私有比例（nil 跟随全局）× 档位。
        // 非视频节点保持原逻辑：有图按图最接近比例 × 档位；无图按节点比例。
        let node = store.nodes.first(where: { $0.id == nodeID })
        let quality = node?.quality ?? .standard
        let ratio = node?.ratio ?? store.currentRatio
        let duration = node?.duration ?? store.currentDuration
        let size: (width: Int, height: Int)
        if node?.type == .video {
            // 尾帧续接链路：跟随前置视频实际尺寸（模型无关，读取资产库视频文件像素尺寸）
            if let prevVideoID = store.h3ChainSourceID(for: nodeID),
               let prevNode = store.nodes.first(where: { $0.id == prevVideoID }),
               let prevSize = store.actualVideoPixelSize(for: prevNode) {
                size = prevSize
            } else {
                // 非续接视频节点：按节点自身比例/全局比例，图片输入不参与尺寸计算
                size = videoSize(ratio: ratio, quality: quality)
            }
        } else {
            if let first = imagePaths.first, let img = NSImage(contentsOfFile: first) {
                size = videoSize(imageWidth: Int(img.size.width), imageHeight: Int(img.size.height), quality: quality)
            } else {
                size = videoSize(ratio: ratio, quality: quality)
            }
        }
        let modelName = node?.model.displayName ?? "?"
        // ★ 模型就绪检查：生成前确认所选模型权重已下载；未下载则自动下载（画布中间显示试管进度条），
        //   下载完成后自动续跑本次生成（不同模型走各自下载地址，见 ModelDownloadSpec.all）
        if let node, let dlSpec = ModelDownloadSpec.spec(for: node.model), !dlSpec.isReady {
            debugLog("视频生成：模型 \(node.model.displayName) 权重未下载（\(dlSpec.localDir)），先下载后生成")
            modelDownloadManager.ensureModel(node.model) {
                self.startVideoGeneration(nodeID: nodeID, prompt: prompt, visited: visited)
            }
            return
        }
        debugLog("视频生成：入队（节点 \(nodeID)，模型 \(modelName)，提示词「\(trimmedPrompt)」，尺寸 \(size.width)×\(size.height)，时长 \(duration.displayName)）")
        let task = QueueTask(
            kind: .video,
            nodeID: nodeID,
            prompt: trimmedPrompt,
            imagePaths: imagePaths,
            videoPaths: videoPaths,
            audioPaths: audioPaths,
            videoWidth: size.width,
            videoHeight: size.height,
            duration: duration,
            model: node?.model ?? .ltx25Distill,
            // ★ 尾帧延续（.h3cc）链路（2026-09-20 重装）：透传本节点开关 + 前置延续源 ID
            //   （连接顺序第一条入边 from.type == .video && from.tailFrameEnabled，队列端据此 load 前置缓存）
            h3TailFrameEnabled: node?.tailFrameEnabled ?? true,
            h3ChainSourceID: store.h3ChainSourceID(for: nodeID)
        )
        GenerationQueue.shared.enqueue(task)
    }
    
    /// 把生成的 mp4 接入节点：复制到 资产库/、异步提取首帧做缩略图、更新节点 imageFileName/mediaFileName
    /// （旧内容文件保留在资产库不删除，避免误删资产；节点引用切换后即完成"替换实际内容"）
    func attachGeneratedVideo(to nodeID: UUID, mp4Path: String, prompt: String) async {
        guard let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) else {
            debugLog("视频生成：节点已不存在，产物未接入")
            return
        }
        ensureCanvasDirectories()
        let fm = FileManager.default
        // 管线落盘已是计数名（ltx_N），这里从路径提取；进资产库是第二次传递，同名则追加 _ 避免覆盖
        var baseName = URL(fileURLWithPath: mp4Path).deletingPathExtension().lastPathComponent
        // 产物可能为 .mp4(h264) 或 .mov(ProRes 422)，按源扩展名入库，避免 mov 内容错标 mp4 导致播放/分享工具误判
        let srcExt = URL(fileURLWithPath: mp4Path).pathExtension.isEmpty ? "mp4" : URL(fileURLWithPath: mp4Path).pathExtension
        while fm.fileExists(atPath: assetLibraryURL.appendingPathComponent("\(baseName).\(srcExt)").path) {
            baseName += "_"
        }
        let mp4Name = "\(baseName).\(srcExt)"
        let thumbName = "\(baseName)_thumb.png"
        let mp4URL = assetLibraryURL.appendingPathComponent(mp4Name)
        let thumbURL = assetLibraryURL.appendingPathComponent(thumbName)
        
        // 1) 复制 mp4 到资产库
        do {
            try fm.copyItem(at: URL(fileURLWithPath: mp4Path), to: mp4URL)
        } catch {
            debugLog("视频生成：mp4 复制到资产库失败 \(error.localizedDescription)")
            return
        }
        // 2) 异步提取首帧缩略图（视频节点占位区用图片渲染）：generateCGImageAsynchronously 后台解码，
        //    主线程 await 挂起让出，不产生同步等待低 QoS 解码线程的优先级反转
        let avAsset = AVURLAsset(url: mp4URL)
        let imgGen = AVAssetImageGenerator(asset: avAsset)
        imgGen.appliesPreferredTrackTransform = true
        if let cg = await frameImage(from: imgGen, at: .zero) {
            let rep = NSBitmapImageRep(cgImage: cg)
            if let png = rep.representation(using: .png, properties: [:]) {
                try? png.write(to: thumbURL)
            }
        }
        // 2.5) 登记进资产库（资产库面板可见）：文件已在 资产库/，无需再复制
        if let thumbImage = NSImage(contentsOf: thumbURL) {
            let newAsset = AssetItem(
                id: UUID(),
                name: baseName,
                image: thumbImage,
                prompt: prompt,
                category: .video,
                fileName: thumbName,
                mediaFileName: mp4Name
            )
            AssetStore.shared.assets.append(newAsset)
            saveAssetLibraryJSON()
            debugLog("视频生成：已登记资产库 \(newAsset.name)")
        }
        // 3) 更新节点：缩略图 + 媒体文件，节点从空/旧内容切换到生成结果；标题同步为 ltx_n
        //    旧内容先入历史（右下角徽标可恢复），再替换为新产物
        pushNodeHistoryIfNeeded(store.nodes[idx])
        store.nodes[idx].title = baseName
        store.nodes[idx].imageFileName = thumbName
        store.nodes[idx].mediaFileName = mp4Name
        store.save()
        debugLog("视频生成：完成，节点 \(nodeID) 已替换为 \(mp4Name)")
    }
    
    /// 连续视频模式级联传播：沿完成节点的输出连线触发下游视频节点生成（递归延续）。
    /// 禁用连线（inputValidity 判定无效）跳过；visited 只在本次传播链内有效，防止 A→B→A 循环。
    func propagateContinuousVideo(from nodeID: UUID, visited: Set<UUID>) {
        // 发光提示：先熄灭全部旧发光线（本节点完成后，上一跳的连线不再发光）
        store.glowingConnectionIDs.removeAll()
        var newVisited = visited
        newVisited.insert(nodeID)
        for conn in store.connections where conn.fromID == nodeID {
            guard let target = store.nodes.first(where: { $0.id == conn.toID }),
                  target.type == .video else { continue }
            // 防环：已在本次传播链中的节点不触发也不点亮（避免"即将生成"语义失真）
            if newVisited.contains(target.id) {
                debugLog("连续视频模式：下游节点 \(target.id) 已在传播链中，跳过防环")
                continue
            }
            // 禁用连线不算数：用目标节点的输入合规性判定该连线是否在禁用集合中
            let validity = store.inputValidity(for: target.id)
            if validity.applies && validity.invalidConnectionIDs.contains(conn.id) {
                debugLog("连续视频模式：连线 \(conn.id) 对下游节点无效（禁用），跳过")
                continue
            }
            debugLog("连续视频模式：节点 \(nodeID) 完成 → 触发下游视频节点 \(target.id)")
            // 发光统一由 startVideoGeneration 入队前管理：B 入队瞬间自动从 A→B 换到 B→C
            startVideoGeneration(nodeID: target.id, prompt: target.prompt, visited: newVisited)
        }
    }
    
    /// 节点有旧内容时（替换前）把当前内容快照入历史；无内容 / 内容与最近一条相同则跳过
    /// 提示词不随历史存储：生成必带提示词且已写入源（资产库），恢复时从源读
    func pushNodeHistoryIfNeeded(_ node: CanvasNode) {
        guard let idx = store.nodes.firstIndex(where: { $0.id == node.id }) else { return }
        let hasContent = node.imageFileName != nil || node.mediaFileName != nil
        guard hasContent else { return }
        let entry = NodeContentHistory(
            title: node.title,
            imageFileName: node.imageFileName,
            mediaFileName: node.mediaFileName
        )
        var history = store.nodes[idx].history ?? []
        // 按内容判重（历史只存"不在节点上"的内容，切换过多次后可能已在历史中，避免重复追加）
        let isDuplicate = history.contains { $0.imageFileName == entry.imageFileName && $0.mediaFileName == entry.mediaFileName }
        if !isDuplicate {
            history.append(entry)
            store.nodes[idx].history = history
        }
    }
    
    /// 把节点内容切换为某条历史条目（交换模型）：
    /// 被点击的条目从历史移出进入节点；节点当前内容（若有）移入历史。
    /// 历史永远只保存"不在节点上"的内容，点谁显示谁，来回切换不堆积。
    func restoreNodeHistory(to nodeID: UUID, entry: NodeContentHistory) {
        guard let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) else { return }
        let node = store.nodes[idx]
        var history = store.nodes[idx].history ?? []
        // 1) 被恢复的条目移出历史
        history.removeAll { $0.id == entry.id }
        // 2) 节点当前内容（若有，且未在历史中）移入历史
        if node.imageFileName != nil || node.mediaFileName != nil {
            let currentEntry = NodeContentHistory(
                title: node.title,
                imageFileName: node.imageFileName,
                mediaFileName: node.mediaFileName
            )
            let isDuplicate = history.contains { $0.imageFileName == currentEntry.imageFileName && $0.mediaFileName == currentEntry.mediaFileName }
            if !isDuplicate {
                history.append(currentEntry)
            }
        }
        store.nodes[idx].history = history
        // 3) 节点恢复为该条目的内容（标题、提示词、资产引用一并替换）
        store.nodes[idx].title = entry.title
        store.nodes[idx].imageFileName = entry.imageFileName
        store.nodes[idx].mediaFileName = entry.mediaFileName
        // 提示词恢复：历史条目只存源引用，生成必带提示词且已写入源（资产库），恢复直接从源读
        let resolvedPrompt: String
        if let fileName = entry.imageFileName,
           let asset = assetStore.assets.first(where: { $0.fileName == fileName }) {
            resolvedPrompt = asset.prompt
        } else if let media = entry.mediaFileName,
                  let asset = assetStore.assets.first(where: { $0.mediaFileName == media }) {
            resolvedPrompt = asset.prompt
        } else {
            resolvedPrompt = ""
        }
        commitNodePrompt(resolvedPrompt, nodeID: nodeID)
        store.save()
        // 恢复的是当前选中节点时，同步刷新输入框草稿（否则内容/名字变了但提示词输入框文字不更新）
        if let selected = selectedPromptNode(), selected.id == nodeID {
            promptDraft = effectivePrompt(for: selected)
        }
        debugLog("节点历史切换：\(nodeID) → \(entry.title)")
    }
    
    /// 图像生成入口（图像/角色/场景节点共用）：后台执行 HiDream-O1 管线，完成后把成品 PNG 接入节点。
    /// 生成期间发送按钮转圈禁用（imageGeneratingNodeID 非 nil），防止重复提交。
    /// 提示词为空时误触保护直接取消（不启动管线）。
    func startImageGeneration(nodeID: UUID, prompt: String) {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPrompt.isEmpty {
            inputValidationErrors[nodeID] = "提示词为空，已取消运行（请填写提示词）"
            debugLog("图像生成：误触保护（无提示词），取消运行")
            return
        }
        inputValidationErrors[nodeID] = nil
        // ★ 模型就绪检查：图像模型（HiDream-O1）权重未下载则自动下载（画布中间显示试管进度条），
        //   下载完成后自动续跑本次生成（与视频模型 H3 同款流程，见 ModelDownloadSpec.all）
        if let targetNode = store.nodes.first(where: { $0.id == nodeID }),
           let dlSpec = ModelDownloadSpec.spec(forImage: targetNode.imageModel), !dlSpec.isReady {
            debugLog("图像生成：模型 \(targetNode.imageModel.displayName) 权重未下载（\(dlSpec.localDir)），先下载后生成")
            modelDownloadManager.ensureModel(targetNode.imageModel) {
                self.startImageGeneration(nodeID: nodeID, prompt: prompt)
            }
            return
        }
        // 收集合规图像输入（HiDream-O1 规则：图像/角色/场景节点最多 5 张，音频/视频/文本/空节点禁用跳过）。
        let validity = store.inputValidity(for: nodeID)
        let referencePaths = validity.imagePaths
        // 多参考主体顺序：已在 inputValidity 按源节点画布中心 y 排序（y 小=上方=主体图0，与视频节点 ltx25Distill 同款）。
        // 编号约定：主体 = 图0（不计入提示词编号）；index1 = 提示词中的"图1"、index2 = "图2"，以此类推，
        // 融合图从排除主体后的下一张开始编号（prompt 写"图1融合进去、图2融合到"）。
        // 单参考（1 张）保持现有编辑功能不变，不排序、不做主体切换。
        if !validity.invalidConnectionIDs.isEmpty {
            debugLog("图像生成：存在 \(validity.invalidConnectionIDs.count) 条无效连线（已禁用跳过），合规图片 \(referencePaths.count) 张")
        }
        // 尺寸：按节点档位（默认 720）+ 节点私有比例（nil 跟随全局）计算，64 倍数对齐。
        // 编辑模式（有参考图输入）：比例下拉禁用、比例来源由参考图决定，输出宽高比沿用参考图0的宽高比——
        // 单参考时参考图0即唯一参考图；多参考时参考图0即排序后 index0=主体（最高参考图）。
        // 短边 = 档位目标值，长边 = 短边 × 参考图0 宽高比后 64 倍数对齐（与 imageResolution 同款对齐规则）；
        // 无参考图（纯 T2I）保持原逻辑不变。
        let node = store.nodes.first(where: { $0.id == nodeID })
        let ratio = node?.ratio ?? store.currentRatio
        // 编辑模式（有参考图）档位仅 1080/2K 有效：当前档位无效时自动落到第一个有效档位（1080），保证实际生效值与下拉显示一致
        let quality = (node?.imageQuality ?? .p720).resolved(hasValidImageInput: !referencePaths.isEmpty)
        let size: (width: Int, height: Int)
        let subjectInfo = referencePaths.isEmpty ? nil : NSImage(contentsOfFile: referencePaths[0])?.cgImage(forProposedRect: nil, context: nil, hints: nil)
        if let subjectCG = subjectInfo, subjectCG.width > 0, subjectCG.height > 0 {
            let short = quality.targetShortSide
            let subjectW = subjectCG.width, subjectH = subjectCG.height
            let isLandscape = subjectW >= subjectH
            let longFloat = Float(short) * (isLandscape ? Float(subjectW) / Float(subjectH) : Float(subjectH) / Float(subjectW))
            let long = Int((longFloat / 64).rounded()) * 64
            size = isLandscape ? (max(long, short), short) : (short, max(long, short))
            debugLog("图像生成：编辑模式沿用参考图0（index0=\((referencePaths[0] as NSString).lastPathComponent)）宽高比 \(subjectW)×\(subjectH)，isLandscape=\(isLandscape)，short=\(short) → 输出 \(size.width)×\(size.height)")
        } else {
            size = imageResolution(for: ratio, quality: quality)
            debugLog("图像生成：⚠️ 编辑模式未沿用参考图0（参考图 \(referencePaths.count) 张，subjectCG 读取=\(subjectInfo == nil ? "nil" : "\(subjectInfo!.width)×\(subjectInfo!.height)")），回退 imageResolution(ratio=\(ratio.width):\(ratio.height), quality=\(quality.rawValue)) → \(size.width)×\(size.height)")
        }
        // 编辑模式：合规参考图作为编辑条件输入传给管线（管线侧有参考图走 hidreamGenerateEdit 编辑分支）。
        debugLog("图像生成：入队（节点 \(nodeID)，提示词「\(trimmedPrompt)」，尺寸 \(size.width)×\(size.height)，参考图 \(referencePaths.count) 张）")
        let task = QueueTask(
            kind: .image,
            nodeID: nodeID,
            prompt: trimmedPrompt,
            imageWidth: size.width,
            imageHeight: size.height,
            referencePaths: referencePaths
        )
        GenerationQueue.shared.enqueue(task)
    }
    
    /// 把生成的 PNG 接入节点：复制到 资产库/、生成缩略图、更新节点 imageFileName
    /// （旧内容文件保留在资产库不删除，避免误删资产；节点引用切换后即完成"替换实际内容"）
    func attachGeneratedImage(to nodeID: UUID, pngPath: String, prompt: String) {
        guard let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) else {
            debugLog("图像生成：节点已不存在，产物未接入")
            return
        }
        ensureCanvasDirectories()
        let fm = FileManager.default
        // 管线落盘已是计数名（hidream_N），这里从路径提取；进资产库是第二次传递，同名则追加 _ 避免覆盖
        var baseName = URL(fileURLWithPath: pngPath).deletingPathExtension().lastPathComponent
        while fm.fileExists(atPath: assetLibraryURL.appendingPathComponent("\(baseName).png").path) {
            baseName += "_"
        }
        let pngName = "\(baseName).png"
        let pngURL = assetLibraryURL.appendingPathComponent(pngName)
        // 1) 复制 PNG 到资产库（缩略图直接用成品图本身）
        do {
            try fm.copyItem(at: URL(fileURLWithPath: pngPath), to: pngURL)
        } catch {
            debugLog("图像生成：PNG 复制到资产库失败 \(error.localizedDescription)")
            return
        }
        // 2) 登记进资产库（资产库面板可见）
        if let image = NSImage(contentsOf: pngURL) {
            let newAsset = AssetItem(
                id: UUID(),
                name: baseName,
                image: image,
                prompt: prompt,
                category: .image,
                fileName: pngName,
                mediaFileName: nil
            )
            AssetStore.shared.assets.append(newAsset)
            saveAssetLibraryJSON()
            debugLog("图像生成：已登记资产库 \(newAsset.name)")
        }
        // 3) 更新节点：缩略图指向生成结果（媒体文件置空，纯图片资产）；标题同步为 hidream_n
        //    旧内容先入历史（右下角徽标可恢复），再替换为新产物
        pushNodeHistoryIfNeeded(store.nodes[idx])
        store.nodes[idx].title = baseName
        store.nodes[idx].imageFileName = pngName
        store.nodes[idx].mediaFileName = nil
        store.save()
        debugLog("图像生成：完成，节点 \(nodeID) 已替换为 \(pngName)")
    }
    
    /// 多选打组：全部赋同一个 groupID（内容层据此画组包围盒）
    func groupSelectedNodes() {
        guard store.selectedNodeIDs.count > 1 else { return }
        store.pushSnapshot()   // 打组前记录快照（用于撤销）
        let gid = UUID()
        for id in store.selectedNodeIDs {
            if let idx = store.nodes.firstIndex(where: { $0.id == id }) {
                store.nodes[idx].groupID = gid
            }
        }
        store.save()
    }
    
    /// 目标组剩余节点不足 2 个时整组解散（清空 groupID 保留节点）。
    /// 仅做数据变更，调用方负责 pushSnapshot 与 save。
    func dissolveGroupIfUnderflow(_ gid: UUID) {
        let remaining = store.nodes.filter { $0.groupID == gid }
        guard remaining.count < 2 else { return }
        for i in store.nodes.indices where store.nodes[i].groupID == gid {
            store.nodes[i].groupID = nil
        }
    }
    
    /// 开始大纲重命名：打开大纲面板、进入行内编辑；
    /// 组内节点自动展开所在组，并让大纲滚动栏聚焦到该节点行
    func beginOutlineRename(_ node: CanvasNode) {
        showOutlinePanel = true
        outlineRenameNodeID = node.id
        outlineRenameText = node.title
        if let gid = node.groupID {
            outlineExpandedGroupIDs.insert(gid)
        }
        outlineScrollTarget = "node-\(node.id.uuidString)"
        debugLog("大纲视图：开始重命名节点「\(node.title)」")
    }
    
    /// 右键「移除组」：把该节点移出所在组；若移除后组内剩余节点不足 2 个，整组解散
    func removeNodeFromGroup(_ node: CanvasNode) {
        guard let gid = node.groupID else { return }
        store.pushSnapshot()   // 移除前记录快照（用于撤销）
        if let idx = store.nodes.firstIndex(where: { $0.id == node.id }) {
            store.nodes[idx].groupID = nil
        }
        dissolveGroupIfUnderflow(gid)
        store.save()
    }
    
    /// 选中集是否恰好是某个完整组（组内节点全部被选中）：整组选中视为已打组，不再显示打组按钮
    private var selectedIsCompleteGroup: Bool {
        let selected = store.nodes.filter { store.selectedNodeIDs.contains($0.id) }
        guard selected.count > 1, let gid = selected.first?.groupID else { return false }
        return selected.allSatisfy { $0.groupID == gid }
            && store.nodes.filter { $0.groupID == gid }.count == selected.count
    }
    
    /// 解散整组：保留组内节点，仅清除 groupID（选中集为完整组时由 Delete 触发）
    func ungroupSelectedGroup() {
        guard let gid = store.nodes.first(where: { store.selectedNodeIDs.contains($0.id) })?.groupID else { return }
        store.pushSnapshot()   // 解散前记录快照（用于撤销）
        for i in store.nodes.indices where store.nodes[i].groupID == gid {
            store.nodes[i].groupID = nil
        }
        store.save()
    }

    // MARK: - 节点字段更新回调工厂（单选输入框 / 展开输入框共用）
    // 统一「快照 → keyPath 写入 → 落盘 → 日志」；Optional 字段（ratio/duration）走重载自动包装。
    // 非 Optional 字段：keyPath: \.model 直接写回
    func makeNodeFieldUpdater<Raw>(
        _ nodeID: UUID,
        keyPath: WritableKeyPath<CanvasNode, Raw>,
        log: @escaping (Raw) -> String
    ) -> (Raw) -> Void {
        { value in
            if applyNodeField(nodeID, keyPath: keyPath, value: value) {
                debugLog(log(value))
            }
        }
    }

    // Optional 字段：keyPath: \.ratio 时 log 收到的是解包后的值，写回时包装为 Optional
    func makeNodeFieldUpdater<Wrapped>(
        _ nodeID: UUID,
        keyPath: WritableKeyPath<CanvasNode, Wrapped?>,
        log: @escaping (Wrapped) -> String
    ) -> (Wrapped) -> Void {
        { value in
            if applyNodeField(nodeID, keyPath: keyPath, value: Optional.some(value)) {
                debugLog(log(value))
            }
        }
    }

    // 快照 → 写入 → 落盘 公共步骤（日志由各重载用 UI 值打，避免 Optional 包装导致的类型不匹配）
    private func applyNodeField<Raw>(
        _ nodeID: UUID,
        keyPath: WritableKeyPath<CanvasNode, Raw>,
        value: Raw
    ) -> Bool {
        if let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) {
            store.pushSnapshot()
            store.nodes[idx][keyPath: keyPath] = value
            store.save()
            return true
        }
        return false
    }
    
    /// 节点输入框 / 打组 浮层：单选显示节点下方输入框，多选显示打组按钮，展开时全屏聚焦模式盖住一切
    @ViewBuilder
    var nodePromptOverlay: some View {
        ZStack {
            // 多选：打组按钮（选中节点群包围盒上方居中；整组选中视为已打组，不显示）
            if store.selectedNodeIDs.count > 1, expandedPromptNodeID == nil, !selectedIsCompleteGroup {
                let selectedNodes = store.nodes.filter { store.selectedNodeIDs.contains($0.id) }
                let rect = nodesBoundingRect(selectedNodes, nodeSizes: store.nodeSizes, zoom: store.zoom,
                                             draggingNodeIDs: store.draggingNodeIDs,
                                             dragOffset: store.nodeDragOffset)
                GroupNodesButton(count: store.selectedNodeIDs.count) {
                    groupSelectedNodes()
                }
                .position(x: store.offset.x + rect.midX, y: store.offset.y + rect.minY - 22)
            }
            
            // 单选：节点下方输入框（卡片顶部贴节点完整底边 + 10）
            if let node = selectedPromptNode(), expandedPromptNodeID == nil {
                let nodeCenter = canvasToView(node.position, offset: store.offset, zoom: store.zoom)
                let dragging = store.draggingNodeIDs.contains(node.id)
                let dragX = dragging ? store.nodeDragOffset.x * store.zoom : 0
                let dragY = dragging ? store.nodeDragOffset.y * store.zoom : 0
                let nodeHeight = nodeCardTotalHeight(for: node, nodeSizes: store.nodeSizes)
                let bottomY = nodeCenter.y + nodeHeight * store.zoom / 2
                NodePromptBar(
                    node: node,
                    currentGlobalRatio: store.currentRatio,
                    currentGlobalDuration: store.currentDuration,
                    text: $promptDraft,
                    onExpand: { expandedPromptNodeID = node.id },
                    onSend: { sendPrompt() },
                    chainDisplayInfo: store.chainVideoDisplayInfo(for: node),
                    onRatioChange: makeNodeFieldUpdater(node.id, keyPath: \.ratio) { "节点输入框：写入节点私有比例 \($0.displayName)" },
                    onModelChange: makeNodeFieldUpdater(node.id, keyPath: \.model) { "节点输入框：写入视频节点模型 \($0.displayName)" },
                    onImageModelChange: { model in
                        makeNodeFieldUpdater(node.id, keyPath: \.imageModel) { "节点输入框：写入图像节点模型 \($0.displayName)" }(model)
                        // ★ 选择 HiDream 时检查权重：未下载自动开始下载（下载面板显示），已就绪直接可用
                        if let spec = ModelDownloadSpec.spec(forImage: model), !spec.isReady {
                            debugLog("图像模型 \(model.displayName) 权重未下载（\(spec.localDir)），自动开始下载")
                            modelDownloadManager.ensureModel(model)
                        }
                    },
                    hasValidImageInput: !store.inputValidity(for: node.id).imagePaths.isEmpty,
                    onQualityChange: makeNodeFieldUpdater(node.id, keyPath: \.quality) { "节点输入框：写入视频节点档位 \($0.displayName)" },
                    onDurationChange: makeNodeFieldUpdater(node.id, keyPath: \.duration) { "节点输入框：写入视频节点时长 \($0.displayName)" },
                    onImageQualityChange: makeNodeFieldUpdater(node.id, keyPath: \.imageQuality) { "节点输入框：写入图像节点档位 \($0.displayName)" },
                    isGenerating: GenerationQueue.shared.isGenerating(nodeID: node.id),
                    validationError: inputValidationErrors[node.id]
                )
                // 背景 GeometryReader 只读尺寸不改布局；position 中心 = 底边 + 10 + 卡片高/2，顶部正好贴底边 + 10
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .onAppear { nodePromptBarHeight = geo.size.height }
                            .onChange(of: geo.size.height) { _, h in nodePromptBarHeight = h }
                    }
                )
                .position(x: nodeCenter.x + dragX,
                          y: bottomY + 10 + nodePromptBarHeight / 2 + dragY)
            }
            
            // 展开聚焦模式：全屏深色遮罩 + 大输入框（最上层，盖住画布与所有面板）
            if let nodeID = expandedPromptNodeID, let node = store.nodes.first(where: { $0.id == nodeID }) {
                ExpandedNodePromptView(
                    node: node,
                    currentGlobalRatio: store.currentRatio,
                    currentGlobalDuration: store.currentDuration,
                    text: $promptDraft,
                    onClose: {
                        commitPromptDraft()
                        expandedPromptNodeID = nil
                    },
                    onSend: {
                        sendPrompt()
                        expandedPromptNodeID = nil
                    },
                    chainDisplayInfo: store.chainVideoDisplayInfo(for: node),
                    onRatioChange: makeNodeFieldUpdater(node.id, keyPath: \.ratio) { "展开输入框：写入节点私有比例 \($0.displayName)" },
                    onModelChange: makeNodeFieldUpdater(node.id, keyPath: \.model) { "展开输入框：写入视频节点模型 \($0.displayName)" },
                    onImageModelChange: { model in
                        makeNodeFieldUpdater(node.id, keyPath: \.imageModel) { "展开输入框：写入图像节点模型 \($0.displayName)" }(model)
                        // ★ 选择 HiDream 时检查权重：未下载自动开始下载（下载面板显示），已就绪直接可用
                        if let spec = ModelDownloadSpec.spec(forImage: model), !spec.isReady {
                            debugLog("图像模型 \(model.displayName) 权重未下载（\(spec.localDir)），自动开始下载")
                            modelDownloadManager.ensureModel(model)
                        }
                    },
                    hasValidImageInput: !store.inputValidity(for: node.id).imagePaths.isEmpty,
                    onQualityChange: makeNodeFieldUpdater(node.id, keyPath: \.quality) { "展开输入框：写入视频节点档位 \($0.displayName)" },
                    onDurationChange: makeNodeFieldUpdater(node.id, keyPath: \.duration) { "展开输入框：写入视频节点时长 \($0.displayName)" },
                    onImageQualityChange: makeNodeFieldUpdater(node.id, keyPath: \.imageQuality) { "展开输入框：写入图像节点档位 \($0.displayName)" },
                    isGenerating: GenerationQueue.shared.isGenerating(nodeID: node.id),
                    validationError: inputValidationErrors[node.id]
                )
            }
        }
    }
    
    /// 浮层面板层：所有悬浮面板统一放在此层，与底层画布（网格/节点/鼠标控制）完全隔离。
    /// 面板打开时，画布上方的隔离拦截层会吃掉画布区域的点击，防止穿透到画布；
    /// 点击面板外部（拦截层）关闭所有面板；工具栏位于拦截层之上，仍可正常操作。
    /// 后续新增面板只需在此层追加，天然与画布隔离。
    @ViewBuilder
    var floatingPanelLayer: some View {
        ZStack {
            // 引用该节点生成面板（拖拽连线在空白处松手时弹出，鼠标在哪面板在哪）
            if showGeneratePanel {
                generatePanel
                    .position(generatePanelPos)
            }

            // 大纲视图面板（从左侧工具栏右侧延伸）
            if showOutlinePanel {
                outlinePanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                    .offset(x: leftToolbarFrame.width + 12)
            }
            
            // 添加节点面板（左侧工具栏加号，从工具栏右侧延伸）
            if showToolbarAddPanel {
                addNodePanel {
                    showToolbarAddPanel = false
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                .offset(x: leftToolbarFrame.width + 12)
            }
            
            // 右键菜单（鼠标在哪面板在哪，置于最上层，不被大纲等面板遮挡）
            if showContextMenu {
                contextMenu
                    .position(contextMenuPos)
            }
        }
    }
    
    // MARK: - 大纲视图聚焦节点：画布居中该节点并放大
    
    func focusOnNode(_ node: CanvasNode) {
        store.selectedNodeIDs = [node.id]
        // 放大到目标缩放（当前不足 1.5 时放大到 1.5）
        let targetZoom = max(1.5, store.zoom)
        store.pushSnapshot()
        store.zoom = targetZoom
        // 让节点居中到画布中心
        let center = CGPoint(x: canvasViewSize.width / 2, y: canvasViewSize.height / 2)
        store.offset = CGPoint(
            x: center.x - node.position.x * targetZoom,
            y: center.y - node.position.y * targetZoom
        )
        store.save()
    }
    
    /// 大纲视图聚焦组：选中组内全部节点，画布居中到组包围盒中心并放大
    func focusOnGroup(_ groupNodes: [CanvasNode]) {
        guard !groupNodes.isEmpty else { return }
        store.selectedNodeIDs = Set(groupNodes.map(\.id))
        // 放大到目标缩放（当前不足 1.5 时放大到 1.5）
        let targetZoom = max(1.5, store.zoom)
        store.pushSnapshot()
        store.zoom = targetZoom
        // 组包围盒（画布坐标，含缩放）居中到画布中心
        let rect = nodesBoundingRect(groupNodes, nodeSizes: store.nodeSizes, zoom: targetZoom)
        let center = CGPoint(x: canvasViewSize.width / 2, y: canvasViewSize.height / 2)
        store.offset = CGPoint(
            x: center.x - rect.midX,
            y: center.y - rect.midY
        )
        store.save()
    }
    
    // MARK: - 空节点占位区按钮（上传 / 资产库）
    
    // 空节点占位区按钮点击：上传 / 资产库（由鼠标控制层命中按钮区域后回调）
    private func handleEmptyNodeButtonClick(nodeID: UUID, action: EmptyNodeButtonAction) {
        guard let node = store.nodes.first(where: { $0.id == nodeID }) else { return }
        switch action {
        case .upload:
            handleUploadContent(for: node)
        case .library:
            pickAssetNodeID = node.id
            // 分类默认选中节点类型对应分类（视频→视频、音频→音频、角色/场景/图像→对应分类），
            // 否则 Picker 选中值不在 allowedCategories 内，面板打开时分类显示未选中需手动点一下
            pickAssetCategory = node.type.assetCategory
            pickAssetScope = nil
            pickSelectedAssetID = nil
        }
    }
    
    // 空节点「上传」：文件选择过滤受节点类型约束（单选），选后入库（分类随节点类型）并补图到节点
    private func handleUploadContent(for node: CanvasNode) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = uploadAllowedTypes(for: node.type)
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            // 取帧/解码放后台（thumbnailImage 内部 await frameImage），回主线程补图到节点
            let targetStore = store
            let targetNode = node
            Task { @MainActor in
                guard let image = await thumbnailImage(for: url) else { return }
                let isImage = targetNode.type.assetCategory == .image
                importAsset(
                    image: image,
                    name: url.deletingPathExtension().lastPathComponent,
                    category: targetNode.type.assetCategory,
                    sourceURL: isImage ? url : nil,
                    mediaSourceURL: isImage ? nil : url,
                    targetNodeID: targetNode.id,
                    store: targetStore
                )
            }
        }
    }
    
    // 右键「上传」：多选文件，按文件类型建不同节点进不同分类（图片/视频/音频），位置在右键处错开
    func handleUploadFiles() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = uploadAllowedTypes(for: nil)
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.begin { response in
            guard response == .OK else { return }
            let urls = panel.urls
            // 按文件类型分组
            var imageURLs: [URL] = []
            var videoURLs: [URL] = []
            var audioURLs: [URL] = []
            for url in urls {
                switch fileKind(of: url) {
                case .video: videoURLs.append(url)
                case .audio: audioURLs.append(url)
                case .image: imageURLs.append(url)
                }
            }
            // 右键位置（画布坐标）作为起点，节点依次错开；取帧/解码放后台（await thumbnailImage），
            // 回主线程建节点/更新 UI（Task @MainActor），避免 NSOpenPanel 回调线程被视频解码阻塞
            let baseCanvas = viewToCanvas(contextMenuPos, offset: store.offset, zoom: store.zoom)
            let targetStore = store
            Task { @MainActor in
                var index = 0
                func place(_ url: URL, category: AssetCategory) async {
                    guard let thumb = await thumbnailImage(for: url) else { return }
                    let pos = CGPoint(x: baseCanvas.x + Double(index) * 40,
                                      y: baseCanvas.y + Double(index) * 40)
                    importAsset(
                        image: thumb,
                        name: url.deletingPathExtension().lastPathComponent,
                        category: category,
                        sourceURL: category == .image ? url : nil,
                        mediaSourceURL: category == .image ? nil : url,
                        createNode: true,
                        nodePosition: pos,
                        store: targetStore
                    )
                    index += 1
                }
                for url in imageURLs { await place(url, category: .image) }
                for url in videoURLs { await place(url, category: .video) }
                for url in audioURLs { await place(url, category: .audio) }
            }
        }
    }
    
    // 空节点「资产库」面板确定：将选中资产设为节点实际内容（资产已在库中，仅补图；
    // 空节点获得内容的那一刻，节点标题自动改为内容名字）
    private func confirmPickAsset(_ asset: AssetItem) {
        guard let nodeID = pickAssetNodeID else {
            pickAssetNodeID = nil
            return
        }
        attachAssetToNode(asset, nodeID: nodeID, store: store)
        // 若该节点正是当前选中的输入框节点，立即刷新草稿显示资产提示词
        if store.selectedNodeIDs == [nodeID],
           let node = store.nodes.first(where: { $0.id == nodeID }) {
            promptDraft = effectivePrompt(for: node)
        }
        pickAssetNodeID = nil
    }

    // MARK: - 复制 / 粘贴（Command+C / Command+V）

    /// 复制选中节点（含节点间连线）为 JSON 到系统剪贴板
    private func copySelectedNodesToClipboard() {
        guard !store.selectedNodeIDs.isEmpty else { return }
        let selected = Set(store.selectedNodeIDs)
        let nodes = store.nodes.filter { selected.contains($0.id) }
        // 仅保留两端都在选中集合内的连线
        let connections = store.connections.filter { selected.contains($0.fromID) && selected.contains($0.toID) }
        let data = ClipboardCanvasData(
            nodes: nodes.map {
                ClipboardNode(id: $0.id, type: $0.type, title: $0.title, subtitle: $0.subtitle,
                              needsSupplement: $0.needsSupplement, position: $0.position,
                              imageFileName: $0.imageFileName)
            },
            connections: connections.map { ClipboardConnection(fromID: $0.fromID, toID: $0.toID) }
        )
        guard let jsonData = try? JSONEncoder().encode(data),
              let jsonString = String(data: jsonData, encoding: .utf8) else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(jsonString, forType: .string)
        debugLog("复制节点 \(nodes.count) 个、连线 \(connections.count) 条")
    }

    /// 从剪贴板粘贴节点：整体重建，保留相对位置与节点间连线，粘贴到视图中心
    private func pasteNodesFromClipboard() {
        guard let jsonString = NSPasteboard.general.string(forType: .string),
              let jsonData = jsonString.data(using: .utf8),
              let data = try? JSONDecoder().decode(ClipboardCanvasData.self, from: jsonData),
              !data.nodes.isEmpty else { return }
        store.pushSnapshot()
        // 原节点包围盒中心
        let xs = data.nodes.map { $0.position.x }
        let ys = data.nodes.map { $0.position.y }
        let oldCenter = CGPoint(x: (xs.min()! + xs.max()!) / 2, y: (ys.min()! + ys.max()!) / 2)
        // 视图中心对应的画布坐标
        let viewCenter = viewToCanvas(CGPoint(x: canvasViewSize.width / 2, y: canvasViewSize.height / 2),
                                      offset: store.offset, zoom: store.zoom)
        // 旧 id → 新 id 映射
        var idMap: [UUID: UUID] = [:]
        for oldNode in data.nodes {
            let newPos = CGPoint(x: oldNode.position.x - oldCenter.x + viewCenter.x,
                                 y: oldNode.position.y - oldCenter.y + viewCenter.y)
            let newNode = NodeFactory.createNode(
                type: oldNode.type,
                title: oldNode.title,
                subtitle: oldNode.subtitle,
                needsSupplement: oldNode.needsSupplement,
                position: newPos,
                imageFileName: oldNode.imageFileName
            )
            idMap[oldNode.id] = newNode.id
            store.nodes.append(newNode)
        }
        // 重建节点间连线（id 映射到新节点）
        for conn in data.connections {
            if let from = idMap[conn.fromID], let to = idMap[conn.toID] {
                store.connections.append(NodeConnection(fromID: from, toID: to))
            }
        }
        store.save()
        debugLog("粘贴节点 \(data.nodes.count) 个、连线 \(data.connections.count) 条")
    }
    
}

// ============================================================

// MARK: - 可拖拽网格背景

// MARK: - 可拖拽网格背景（AppKit 图层实现）

/// 网格背景：点阵路径只构建一次（覆盖视口 + 每边一个周期），拖动/缩放时
/// 用 SwiftUI .offset 平移整个 NSView 位图——与内容层 .position(offset) 同坐标系，
/// 方向必然一致；不经过 SwiftUI Canvas 逐帧重绘，也不碰 AppKit isFlipped 坐标系。
struct DraggableGrid: View {
    @Binding var offset: CGPoint
    let gridSize: Double
    let gridColor: Color
    let majorGridColor: Color
    var showGrid: Bool = true
    var zoom: Double = 1.0
    
    var body: some View {
        let g = gridSize * zoom
        let phaseX = posMod(offset.x, g)
        let phaseY = posMod(offset.y, g)
        DotGridLayerView(spacing: g,
                         dotSize: 3,
                         dotColor: NSColor(gridColor).cgColor,
                         showGrid: showGrid)
            .offset(x: phaseX - g, y: phaseY - g)
            .ignoresSafeArea()
    }
    
    /// 取模归一化到 [0, m)，兼容负数
    private func posMod(_ v: Double, _ m: Double) -> Double {
        let r = v.truncatingRemainder(dividingBy: m)
        return r >= 0 ? r : r + m
    }
}

/// NSView 包装：只画静态点阵，update 仅在间距/尺寸/颜色变化时重建路径，不做任何平移
struct DotGridLayerView: NSViewRepresentable {
    var spacing: Double
    var dotSize: Double
    var dotColor: CGColor
    var showGrid: Bool
    
    func makeNSView(context: Context) -> DotGridLayerNSView {
        DotGridLayerNSView(dotSize: dotSize)
    }
    
    func updateNSView(_ nsView: DotGridLayerNSView, context: Context) {
        nsView.update(spacing: spacing,
                      dotColor: dotColor,
                      showGrid: showGrid)
    }
}

/// 静态圆点阵图层：单个 CAShapeLayer 铺满「视口 + 每边一个周期」，路径只构建一次。
/// 平移交给 SwiftUI .offset 完成，本类不做任何相位计算。
/// 首次进入画布时 update 可能早于布局（bounds 仍为 0），故参数先缓存，
/// 在 layout() 尺寸就绪后再补建路径，保证进入即显示、无需先拖动一次。
final class DotGridLayerNSView: NSView {
    private let dotLayer = CAShapeLayer()
    private var lastSpacing: CGFloat = 0
    private var lastBoundsSize: CGSize = .zero
    private var lastColor: CGColor?
    private var pendingSpacing: CGFloat = 0
    private var pendingColor: CGColor?
    private var pendingShowGrid = true
    private let dotSize: CGFloat
    
    init(dotSize: CGFloat) {
        self.dotSize = dotSize
        super.init(frame: .zero)
        wantsLayer = true
        dotLayer.fillColor = nil
        layer?.addSublayer(dotLayer)
    }
    
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    
    func update(spacing: CGFloat, dotColor: CGColor, showGrid: Bool) {
        pendingSpacing = spacing
        pendingColor = dotColor
        pendingShowGrid = showGrid
        dotLayer.isHidden = !showGrid
        applyIfReady()
    }
    
    override func layout() {
        super.layout()
        applyIfReady()
    }
    
    private func applyIfReady() {
        guard pendingShowGrid, pendingSpacing > 0, bounds.width > 0, bounds.height > 0 else { return }
        
        // 间距 / 视口尺寸 / 颜色变化时才重建路径；拖动平移由 SwiftUI .offset 完成
        if lastSpacing != pendingSpacing
            || lastBoundsSize != bounds.size
            || lastColor != pendingColor {
            rebuild(spacing: pendingSpacing, size: bounds.size, dotColor: pendingColor ?? .white)
            lastSpacing = pendingSpacing
            lastBoundsSize = bounds.size
            lastColor = pendingColor
        }
    }
    
    private func rebuild(spacing: CGFloat, size: CGSize, dotColor: CGColor) {
        let layerW = size.width + spacing * 2
        let layerH = size.height + spacing * 2
        let r = dotSize / 2
        
        let path = CGMutablePath()
        var ix = 0
        while CGFloat(ix) * spacing <= layerW {
            var iy = 0
            while CGFloat(iy) * spacing <= layerH {
                let cx = CGFloat(ix) * spacing
                let cy = CGFloat(iy) * spacing
                path.addEllipse(in: CGRect(x: cx - r, y: cy - r,
                                           width: dotSize, height: dotSize))
                iy += 1
            }
            ix += 1
        }
        
        dotLayer.path = path
        dotLayer.fillColor = dotColor
    }
}

// ============================================================

// MARK: - 画布主视图（集成到创作页）

struct CanvasEditorView: View {
    @EnvironmentObject var projectStore: ProjectStore
    @StateObject private var canvasStore = CanvasStore()
    @Binding var selectedProject: ProjectItem?
    @State var canvasName: String = ""
    @State var showNameConflict = false
    
    var body: some View {
        ZStack {
            // 画布内容
            CanvasView(canvasName: $canvasName)
                .environmentObject(canvasStore)
                .environmentObject(projectStore)
            
            // 顶部菜单栏：返回 + 画布名
            VStack {
                HStack(spacing: 10) {
                    // 返回按钮
                    Button(action: {
                        debugLog("导航：返回项目列表")
                        withAnimation {
                            // 落盘由 onDisappear 兜底（任何离开方式都保存），此处无需重复 save
                            selectedProject = nil
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                            Text("返回")
                        }
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    
                    // 画布名
                    TextField("画布名称", text: $canvasName)
                        .textFieldStyle(.plain)
                        .font(.headline)
                        .frame(width: 160)
                        .onChange(of: canvasName) { _, newName in
                            handleCanvasNameChange(newName)
                        }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.white.opacity(0.95))
                        .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
                )
                .padding(12)
                
                Spacer()
            }
            
            // 重名提示（自动消失）
            if showNameConflict {
                VStack {
                    Spacer()
                    Text("项目名重复")
                        .font(.subheadline)
                        .foregroundColor(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            Capsule()
                                .fill(Color.black.opacity(0.75))
                        )
                        .padding(.bottom, 40)
                }
                .transition(.opacity)
            }
        }
        .onAppear {
            if let project = selectedProject {
                canvasName = project.title
                // 注入当前项目，供 save() / 自动保存写入正确目录
                canvasStore.currentProject = project
                // 从磁盘重建画布内容（节点/连线/偏移/缩放/比例）
                loadProjectCanvas(project, into: canvasStore)
            }
        }
        .onDisappear {
            canvasStore.save()
            // 区域互斥保护：退出/离开画布时停止画布节点的播放
            MediaPlayerManager.shared.stopNodePlayback()
        }
    }
    
    // 画布改名：同步到项目管理 + 重名检查
    private func handleCanvasNameChange(_ newName: String) {
        guard let project = selectedProject else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return }
        
        // 检查是否与其他项目重名
        let conflict = projectStore.projects.contains { $0.id != project.id && $0.title == trimmed }
        if conflict {
            showNameConflict = true
            // 回滚为原项目名
            canvasName = project.title
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                withAnimation { showNameConflict = false }
            }
        } else {
            // 同步到项目管理
            projectStore.rename(item: project, to: trimmed)
            // 同步到当前项目（保存时写入新标题）
            canvasStore.currentProject?.title = trimmed
        }
        canvasStore.save()
    }

}

// ============================================================
//  模型下载面板：画布中间显示"正在下载模型"标题 + 总进度百分比 +
//  当前活动下载文件列表（并发 N 个显示 N 项；完成消失、自动补上下一个；
//  每行该文件独立速度 + 文件级进度条；错误态保留重试/关闭按钮）
// ============================================================

struct ModelDownloadPanel: View {
    let progress: Double
    let activeFiles: [ActiveFileDownload]
    let errorMessage: String?
    let accentColor: Color
    var onRetry: (() -> Void)?
    var onDismiss: (() -> Void)?

    var body: some View {
        VStack(spacing: 16) {
            if let errorMessage {
                Text(errorMessage)
                    .font(.callout)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 300)
                HStack(spacing: 12) {
                    Button("重试") { onRetry?() }
                        .buttonStyle(.borderedProminent)
                        .tint(accentColor)
                    Button("关闭") { onDismiss?() }
                        .buttonStyle(.bordered)
                }
            } else {
                // 标题 + 总进度百分比
                HStack(spacing: 8) {
                    Text("正在下载模型")
                        .font(.headline)
                        .foregroundColor(.primary)
                    Text("\(Int(progress * 100))%")
                        .font(.headline)
                        .foregroundColor(accentColor)
                        .monospacedDigit()
                }
                // 活动文件列表：在下载几个就显示几个，完成消失、自动补上下一个
                if activeFiles.isEmpty {
                    Text("准备中…")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(activeFiles, id: \.name) { item in
                            fileRow(item)
                        }
                    }
                    .frame(maxWidth: 340)
                }
            }
        }
        .padding(26)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(accentColor.opacity(0.35), lineWidth: 1.5)
        )
        .shadow(color: .black.opacity(0.18), radius: 24, y: 10)
    }

    /// 单文件行：文件名 + 该文件独立实时速度 + 文件级进度条
    private func fileRow(_ item: ActiveFileDownload) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text(item.name)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 8)
                let speedText = ModelDownloadManager.formatSpeed(item.speedBytesPerSec)
                Text(speedText.isEmpty ? "…" : speedText)
                    .font(.caption.monospacedDigit())
                    .foregroundColor(accentColor)
            }
            // 文件级进度（written/total）
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.black.opacity(0.08))
                    Capsule()
                        .fill(accentColor.opacity(0.75))
                        .frame(width: geo.size.width * fileProgress(item))
                }
            }
            .frame(height: 4)
        }
    }

    private func fileProgress(_ item: ActiveFileDownload) -> CGFloat {
        guard item.total > 0 else { return 0 }
        return CGFloat(min(Double(item.written) / Double(item.total), 1.0))
    }
}
