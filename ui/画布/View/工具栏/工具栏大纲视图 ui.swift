// ============================================================
//  文件作用：CanvasView 扩展。左侧竖直工具面板（加号/用户/大纲/帮助）、大纲视图
//  面板（搜索/类型过滤/重命名/复制/聚焦节点）、LeftToolbarFrameKey 位置上报、
//  OutlineRenameField 重命名输入框。
//  互动文件：引用 画布 状态.swift（CanvasStore）、节点ui.swift（CanvasNode、NodeType）、
//  画布 菜单.swift（addNodePanel）、画布 节点操作.swift（duplicateNodes）、
//  画布 ui.swift（focusOnNode）；被 画布 ui.swift 引用（leftSideToolbar）。
// ============================================================

import SwiftUI

// ============================================================

// MARK: - 左侧竖直工具面板 + 大纲视图（含内部交互逻辑）

extension CanvasView {
    // MARK: - 左侧居中：竖直工具面板
    
    var leftSideToolbar: some View {
        VStack(spacing: 4) {
            // 加号（主操作，黑色圆形背景）→ 添加节点（复用右键添加节点面板，独立弹出）
            Button(action: {
                // 以工具栏右侧为创建位置，复用右键 addNode 逻辑
                contextMenuPos = CGPoint(x: leftToolbarFrame.maxX + 60, y: leftToolbarFrame.midY)
                withAnimation {
                    closeAllPanels()
                    showToolbarAddPanel.toggle()
                }
                debugLog("工具栏：点击加号 打开添加节点面板")
            }) {
                Image(systemName: "plus")
                    .font(.system(size: 17, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Color.black))
            }
            .buttonStyle(.plain)
            .help("添加节点")
            
            // 用户+
            Button(action: {
                withAnimation { closeAllPanels() }
                debugLog("工具栏：点击用户")
            }) {
                Image(systemName: "person.badge.plus")
                    .font(.system(size: 16))
                    .foregroundColor(.primary)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("用户")
            
            // 文件夹 → 大纲视图（列出当前画布所有节点）
            Button(action: {
                withAnimation {
                    closeAllPanels()
                    showOutlinePanel.toggle()
                }
                debugLog("工具栏：点击大纲视图 \(showOutlinePanel ? "打开" : "关闭")")
            }) {
                Image(systemName: "folder")
                    .font(.system(size: 16))
                    .foregroundColor(.primary)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("大纲视图")
            
            // 问号
            Button(action: {
                withAnimation { closeAllPanels() }
                debugLog("工具栏：点击帮助")
            }) {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 16))
                    .foregroundColor(.primary)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("帮助")
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.95))
                .shadow(color: Color.black.opacity(0.1), radius: 3, x: 0, y: 1)
        )
        .padding(8)
        // 上报工具栏在窗口中的位置（供浮层面板层定位面板）
        .background(GeometryReader { geo in
            Color.clear.preference(key: LeftToolbarFrameKey.self, value: geo.frame(in: .global))
        })
        .onPreferenceChange(LeftToolbarFrameKey.self) { frame in
            leftToolbarFrame = frame
        }
    }
    
    // MARK: - 大纲视图面板（列出当前画布所有节点）
    
    /// 大纲视图：按搜索词 + 类型过滤后的节点
    var outlineFilteredNodes: [CanvasNode] {
        store.nodes.filter { node in
            (outlineSearchText.isEmpty || node.title.localizedCaseInsensitiveContains(outlineSearchText))
            && (outlineTypeFilter == nil || node.type == outlineTypeFilter)
        }
    }
    
    /// 大纲视图展示行：组行（可展开/收缩、点击聚焦）或节点行
    enum OutlineRow: Identifiable {
        case group(UUID, [CanvasNode])   // 组 id + 组内节点（已过滤）
        case node(CanvasNode)            // 未分组节点 或 展开的组内节点
        
        var id: String {
            switch self {
            case .group(let gid, _): return "group-\(gid.uuidString)"
            case .node(let node): return "node-\(node.id.uuidString)"
            }
        }
    }
    
    /// 大纲视图展示行序列：按节点顺序，组首次出现时渲染组行，
    /// 展开的组紧跟组内节点；未分组节点原样渲染
    var outlineRows: [OutlineRow] {
        let filtered = outlineFilteredNodes
        var rows: [OutlineRow] = []
        var seenGroupIDs: Set<UUID> = []
        for node in filtered {
            if let gid = node.groupID {
                guard !seenGroupIDs.contains(gid) else { continue }
                seenGroupIDs.insert(gid)
                let groupNodes = filtered.filter { $0.groupID == gid }
                rows.append(.group(gid, groupNodes))
                if outlineExpandedGroupIDs.contains(gid) {
                    rows.append(contentsOf: groupNodes.map { OutlineRow.node($0) })
                }
            } else {
                rows.append(.node(node))
            }
        }
        return rows
    }
    
    /// 大纲视图：切换组展开/收缩
    func toggleOutlineGroup(_ gid: UUID) {
        if outlineExpandedGroupIDs.contains(gid) {
            outlineExpandedGroupIDs.remove(gid)
        } else {
            outlineExpandedGroupIDs.insert(gid)
        }
    }
    
    /// 大纲视图节点行（未分组节点 或 展开的组内节点；组内节点缩进展示）
    func outlineNodeRow(_ node: CanvasNode, indent: Bool) -> some View {
        HStack(spacing: 0) {
            if outlineRenameNodeID == node.id {
                // 重命名模式：行内直接编辑，回车提交
                HStack(spacing: 12) {
                    Image(systemName: node.type.icon)
                        .font(.system(size: 15))
                        .frame(width: 22)
                    OutlineRenameField(text: $outlineRenameText, onSubmit: {
                        commitOutlineRename(node)
                    }, onCancel: {
                        cancelOutlineRename(node)
                    })
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(
                        RoundedRectangle(cornerRadius: 5, style: .continuous)
                            .fill(Color.white)
                            .overlay(
                                RoundedRectangle(cornerRadius: 5, style: .continuous)
                                    .stroke(Color.black.opacity(0.25), lineWidth: 1)
                            )
                    )
                    Spacer()
                }
                .foregroundColor(.primary)
                .padding(.leading, indent ? 40 : 16)
                .padding(.trailing, 16)
                .padding(.vertical, 10)
            } else {
                // 主区域：点击选中并聚焦该节点（面板保持打开）
                // 用 Button 而非 onTapGesture：Button 识别器由 AppKit 管理，
                // 不因 hover 触发的整行重建而销毁，避免偶发点不到
                // padding 放进 Button 内部，让整行（含 padding）都能触发聚焦，
                // 避免点击行内空白穿透到面板 onTapGesture 被当成"点击面板空白"
                Button(action: {
                    focusOnNode(node)
                    debugLog("大纲视图：选中并聚焦节点「\(node.title)」")
                }) {
                    HStack(spacing: 12) {
                        Image(systemName: node.type.icon)
                            .font(.system(size: 15))
                            .frame(width: 22)
                        Text(node.title)
                            .font(.subheadline)
                            .lineLimit(1)
                        Spacer()
                    }
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.leading, indent ? 40 : 16)
                    .padding(.trailing, 16)
                    .padding(.vertical, 10)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .background(
            hoveredOutlineNodeID == node.id ? Color.black.opacity(0.06) : Color.clear
        )
        .contentShape(Rectangle())
        .onHover { hovering in
            // 延迟到 runloop 空闲再更新 hover 状态，避免 hover 触发的整行重建
            // 在重建窗口内销毁 onTapGesture 识别器、吞掉点击（偶发点不到）
            DispatchQueue.main.async {
                hoveredOutlineNodeID = hovering ? node.id : nil
            }
            debugLog("大纲视图：\(hovering ? "悬停" : "离开")节点「\(node.title)」", throttle: true)
        }
        // 右键捕获：大纲内节点右键 → 触发画布右键菜单（改名/复制）
        // 左键穿透到下方 Button（聚焦），仅拦截右键
        .overlay {
            RightClickCatcher { pos in
                contextMenuNodeID = node.id
                contextMenuPos = pos
                contextMenuFromOutline = true
                showAddNodePanel = false
                showToolbarAddPanel = false
                withAnimation { showContextMenu = true }
                debugLog("大纲视图：右键节点「\(node.title)」")
            }
        }
    }
    
    /// 大纲视图组行：展开/收缩箭头 + 文件夹图标 + 分组名，点击行主体聚焦整组
    func outlineGroupRow(_ gid: UUID, _ groupNodes: [CanvasNode]) -> some View {
        let expanded = outlineExpandedGroupIDs.contains(gid)
        return HStack(spacing: 0) {
            // 展开/收缩箭头（独立按钮，不触发聚焦）
            Button(action: {
                toggleOutlineGroup(gid)
                debugLog("大纲视图：\(expanded ? "收缩" : "展开")分组(\(groupNodes.count))")
            }) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                    .frame(width: 20, height: 20)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            // 主区域：点击选中并聚焦整组
            Button(action: {
                focusOnGroup(groupNodes)
                debugLog("大纲视图：聚焦分组(\(groupNodes.count))")
            }) {
                HStack(spacing: 12) {
                    Image(systemName: "folder.fill")
                        .font(.system(size: 15))
                        .foregroundColor(store.nodeColor)
                        .frame(width: 22)
                    Text("分组 (\(groupNodes.count))")
                        .font(.subheadline)
                        .lineLimit(1)
                    Spacer()
                }
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.leading, 4)
                .padding(.trailing, 16)
                .padding(.vertical, 10)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(.leading, 16)
        .background(
            hoveredOutlineGroupID == gid ? Color.black.opacity(0.06) : Color.clear
        )
        .contentShape(Rectangle())
        .onHover { hovering in
            DispatchQueue.main.async {
                hoveredOutlineGroupID = hovering ? gid : nil
            }
        }
    }
    
    var outlinePanel: some View {
        // 三层隔离架构：
        // 第一层 = 底层画布（CanvasView）
        // 第二层 = 大纲视图面板（本视图，含背景拦截层）
        // 第三层 = 大纲内的内容（标题/搜索框/节点列表）
        // 节点操作（改名/复制）已并入画布右键菜单：大纲内节点右键触发
        ZStack {
            // ===== 第二层：面板背景拦截层 =====
            // 点击面板空白：关闭三点菜单 + 确认编辑（提交重命名）
            // 用 simultaneousGesture 而非 onTapGesture：onTapGesture 会与节点行 Button 竞争点击，
            // 把点击节点行的操作抢走误判为"点击面板空白"；simultaneousGesture 与 Button 同时触发，
            // 点击节点行时聚焦由 Button 负责，这里只做面板收尾，互不干扰
            Color.clear
                .contentShape(Rectangle())
                .simultaneousGesture(
                    TapGesture().onEnded {
                        if let renameID = outlineRenameNodeID,
                           let node = store.nodes.first(where: { $0.id == renameID }) {
                            commitOutlineRename(node)
                            debugLog("大纲视图：点击面板空白确认编辑")
                        }
                    }
                )
            
            // ===== 第三层：大纲内容 =====
            VStack(alignment: .leading, spacing: 0) {
                // Tab 切换：大纲视图 / 资产库
                Picker("", selection: $outlinePanelTab) {
                    Text("大纲视图").tag(OutlinePanelTab.outline)
                    Text("资产库").tag(OutlinePanelTab.asset)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 10)
            
            if outlinePanelTab == .outline {
            // 搜索框 + 类型过滤下拉菜单
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                    TextField("搜索节点", text: $outlineSearchText)
                        .textFieldStyle(.plain)
                        .font(.subheadline)
                    if !outlineSearchText.isEmpty {
                        Button(action: {
                            debugLog("大纲视图：清空搜索")
                            outlineSearchText = ""
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.black.opacity(0.05))
                )
                .onChange(of: outlineSearchText) { _, newValue in
                    debugLog("大纲视图：搜索 \(newValue.isEmpty ? "清空" : "「\(newValue)」")")
                }
                
                Menu {
                    Button("全部类型") {
                        outlineTypeFilter = nil
                        debugLog("大纲视图：类型过滤 全部")
                    }
                    ForEach(NodeType.allCases) { type in
                        Button(type.rawValue) {
                            outlineTypeFilter = type
                            debugLog("大纲视图：类型过滤 \(type.rawValue)")
                        }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text(outlineTypeFilter?.rawValue ?? "全部")
                            .font(.subheadline)
                    }
                    .foregroundColor(.primary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.black.opacity(0.05))
                    )
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 8)
            
            if outlineFilteredNodes.isEmpty {
                // 无节点/无匹配：整个区域都是空白，点击可确定编辑/关闭菜单
                ZStack {
                    Color.clear
                        .contentShape(Rectangle())
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .onTapGesture {
                            if let renameID = outlineRenameNodeID,
                               let node = store.nodes.first(where: { $0.id == renameID }) {
                                commitOutlineRename(node)
                                debugLog("大纲视图：点击空白确认编辑")
                            }
                        }
                    Text(store.nodes.isEmpty ? "暂无节点" : "无匹配节点")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 16)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        ZStack(alignment: .top) {
                            // 第二层空白拦截层：覆盖整个 ScrollView 视口
                            // 点击空白：关闭三点菜单 + 确认编辑（提交重命名）
                            Color.clear
                                .contentShape(Rectangle())
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .onTapGesture {
                                    if let renameID = outlineRenameNodeID,
                                       let node = store.nodes.first(where: { $0.id == renameID }) {
                                        commitOutlineRename(node)
                                        debugLog("大纲视图：点击空白确认编辑")
                                    }
                                }
                            
                            // 第三层内容：节点/组行
                            VStack(alignment: .leading, spacing: 0) {
                                ForEach(outlineRows) { row in
                                    switch row {
                                    case .group(let gid, let groupNodes):
                                        outlineGroupRow(gid, groupNodes)
                                            .id(row.id)
                                    case .node(let node):
                                        outlineNodeRow(node, indent: node.groupID != nil)
                                            .id(row.id)
                                    }
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .top)
                        }
                    }
                    .onChange(of: outlineScrollTarget) { _, target in
                        guard let target else { return }
                        DispatchQueue.main.async {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                proxy.scrollTo(target, anchor: .center)
                            }
                        }
                    }
                }
            }
            } else {
                // 资产库 Tab：来源过滤 / 角色场景分类 / 资产网格
                assetPanelContent
            }
            }
        }
        .padding(.vertical, 8)
        .frame(width: 260, height: 500)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.white.opacity(0.97))
                .shadow(color: Color.black.opacity(0.15), radius: 10, x: 0, y: 4)
        )
    }
    
    // MARK: - 大纲视图节点重命名提交（行内编辑，回车完成）
    
    func commitOutlineRename(_ node: CanvasNode) {
        guard outlineRenameNodeID == node.id else { return }
        if let idx = store.nodes.firstIndex(where: { $0.id == node.id }) {
            store.nodes[idx].title = outlineRenameText
            store.save()
            debugLog("大纲视图：重命名节点为「\(outlineRenameText)」")
        }
        outlineRenameNodeID = nil
    }

    // 取消重命名：不提交，恢复原名称
    func cancelOutlineRename(_ node: CanvasNode) {
        guard outlineRenameNodeID == node.id else { return }
        outlineRenameNodeID = nil
        debugLog("大纲视图：取消重命名节点「\(node.title)」")
    }
}

// ============================================================

// MARK: - 左侧工具栏位置上报

struct LeftToolbarFrameKey: PreferenceKey {
    static var defaultValue: CGRect = .zero
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}

// MARK: - 大纲视图重命名输入框（自动全选 + 失焦提交）

struct OutlineRenameField: NSViewRepresentable {
    @Binding var text: String
    var onSubmit: () -> Void
    var onCancel: () -> Void

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField()
        field.isBordered = false
        field.drawsBackground = false
        field.focusRingType = .none
        field.font = .systemFont(ofSize: 13)
        field.delegate = context.coordinator
        field.target = context.coordinator
        field.action = #selector(Coordinator.submit)
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        if nsView.stringValue != text {
            nsView.stringValue = text
        }
        // 首次出现：获得焦点并全选文本，方便直接输入
        if !context.coordinator.didSelect {
            context.coordinator.didSelect = true
            DispatchQueue.main.async {
                nsView.window?.makeFirstResponder(nsView)
                nsView.selectText(nil)
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, NSTextFieldDelegate {
        var parent: OutlineRenameField
        var didSelect = false
        /// 是否真正进入编辑（聚焦后为 true），用于过滤初始化瞬间的误触发
        var isEditing = false
        init(_ parent: OutlineRenameField) { self.parent = parent }

        func controlTextDidChange(_ obj: Notification) {
            if let field = obj.object as? NSTextField {
                parent.text = field.stringValue
            }
        }

        func controlTextDidBeginEditing(_ obj: Notification) {
            isEditing = true
        }

        // 失焦（点击面板其他地方 / 画布）等同回车提交；仅当真正进入编辑后才提交
        func controlTextDidEndEditing(_ obj: Notification) {
            guard isEditing else { return }
            parent.onSubmit()
        }

        // ESC 键：取消编辑（不提交）
        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.cancelOperation(_:)) {
                parent.onCancel()
                return true
            }
            return false
        }

        @objc func submit() {
            parent.onSubmit()
        }
    }
}

// ============================================================

// MARK: - 右键捕获器（大纲节点行右键 → 触发画布右键菜单）

/// 叠加在节点行上的透明视图：仅拦截右键，左键/滚轮等事件穿透到下层 SwiftUI
/// （节点行 Button 聚焦）。右键时把事件位置转成画布视图坐标（左上原点）回调，
/// 供画布右键菜单 contextMenu 定位。
struct RightClickCatcher: NSViewRepresentable {
    var onRightClick: (CGPoint) -> Void

    func makeNSView(context: Context) -> RightClickCatcherNSView {
        let view = RightClickCatcherNSView()
        view.onRightClick = onRightClick
        return view
    }

    func updateNSView(_ nsView: RightClickCatcherNSView, context: Context) {
        nsView.onRightClick = onRightClick
    }
}

final class RightClickCatcherNSView: NSView {
    var onRightClick: ((CGPoint) -> Void)?

    // 仅拦截右键：左键/滚轮等事件 hitTest 返回 nil，穿透到下层 SwiftUI
    override func hitTest(_ point: NSPoint) -> NSView? {
        if NSApp.currentEvent?.type == .rightMouseDown {
            return self
        }
        return nil
    }

    override func rightMouseDown(with event: NSEvent) {
        // 事件位置转成画布视图坐标（左上原点）
        let p = convert(event.locationInWindow, from: nil)
        let originInWindow = convert(NSPoint.zero, to: nil)
        let contentHeight = window?.contentView?.bounds.height ?? 0
        let canvasPos = CGPoint(
            x: originInWindow.x + p.x,
            y: contentHeight - (originInWindow.y + p.y)
        )
        onRightClick?(canvasPos)
    }
}
