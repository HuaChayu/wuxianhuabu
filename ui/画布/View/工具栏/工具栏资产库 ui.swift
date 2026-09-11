// ============================================================
//  文件作用：CanvasView 扩展。资产库内容视图（来源过滤/角色场景分类/
//  资产网格/多选拖拽）。
//  互动文件：引用 资产管理ui.swift（AssetStore、AssetCategory、AssetItem）、
//  创作ui.swift（ProjectStore）、画布 状态.swift（AssetDragData）；
//  被 画布 工具栏2.swift 引用（assetPanelContent，作为大纲视图面板「资产库」Tab）。
// ============================================================

import SwiftUI

// ============================================================

// MARK: - 资产库框选：资产项 frame 上报

/// 收集资产库中每个资产项在面板坐标系中的 frame，用于框选命中检测
struct AssetFrameKey: PreferenceKey {
    static var defaultValue: [UUID: CGRect] = [:]
    static func reduce(value: inout [UUID: CGRect], nextValue: () -> [UUID: CGRect]) {
        value.merge(nextValue()) { $1 }
    }
}

// MARK: - 资产库按钮 + 面板

extension CanvasView {
    // 资产库内容（供大纲视图面板「资产库」Tab 使用）
    // 含来源过滤 / 角色场景分类 / 资产网格，无独立背景与固定高度，随面板撑满
    var assetPanelContent: some View {
        // 当前过滤范围内、当前分类下的资产
        let filteredAssets = assetStore.filteredAssets(in: assetScope, category: assetCategory)
        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                // 来源下拉菜单：全局 / 各项目
                Menu {
                    Button("全局") {
                        debugLog("资产库：来源切换为 全局")
                        assetScope = nil
                    }
                    ForEach(projectStore.projects) { project in
                        Button(project.title) {
                            debugLog("资产库：来源切换为 \(project.title)")
                            assetScope = project.id
                        }
                    }
                } label: {
                    Text(assetScopeTitle)
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
                
                Spacer()
                Text("\(filteredAssets.count) 项")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 16)
            
            // 分类过滤（资产自带分类，按分类筛）
            Picker("", selection: $assetCategory) {
                ForEach(AssetCategory.allCases) { category in
                    Text(category.rawValue).tag(category)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 16)
            .onChange(of: assetCategory) { _, newValue in
                debugLog("资产库：分类切换为 \(newValue.rawValue)")
            }
            
            // 内容区：按来源 + 分类过滤后的资产网格（左右占满面板，滚动条贴边）
            assetPanelGrid(for: assetCategory)
        }
        .padding(.bottom, 12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
    
    /// 资产库面板：按分类渲染资产网格（含空态、选择、拖拽、框选）
    func assetPanelGrid(for category: AssetCategory) -> some View {
        let assets = assetStore.filteredAssets(in: assetScope, category: category)
        if assets.isEmpty {
            return AnyView(
                VStack(spacing: 6) {
                    Image(systemName: category.icon)
                        .font(.system(size: 24))
                        .foregroundColor(.secondary)
                    Text("暂无\(category.rawValue)")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            )
        }
        return AnyView(
            GeometryReader { geo in
                ScrollView {
                    // 面板可用宽度（左右各留 16 首尾留白）
                    let availableWidth = geo.size.width - 32
                    // 每项最小宽度 76，据此算列数（自适应，加分类/缩放不破版）
                    let columns = max(1, Int(availableWidth / 76))
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: columns), spacing: 8) {
                        ForEach(assets) { asset in
                            let isSelected = selectedAssetIDs.contains(asset.id) || assetRubberSelected.contains(asset.id)
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
                            if selectedAssetIDs.contains(asset.id) {
                                selectedAssetIDs.remove(asset.id)
                                debugLog("资产库：取消选择资产「\(asset.name)」")
                            } else {
                                selectedAssetIDs.insert(asset.id)
                                debugLog("资产库：选择资产「\(asset.name)」")
                            }
                        }
                        .draggable(AssetDragData(assetIDs: dragAssetIDs(for: asset)))
                        .background(
                            GeometryReader { geo in
                                Color.clear.preference(
                                    key: AssetFrameKey.self,
                                    value: [asset.id: geo.frame(in: .named("assetPanelViewport"))]
                                )
                            }
                        )
                    }
                }
                .padding(.horizontal, 16)
                .onPreferenceChange(AssetFrameKey.self) { frames in
                    assetFrames = frames
                }
                .overlay(
                    Group {
                        if let rect = assetSelectionRect {
                            Rectangle()
                                .fill(Color.accentColor.opacity(0.15))
                                .overlay(
                                    Rectangle().stroke(Color.accentColor, lineWidth: 1)
                                )
                                .frame(width: rect.width, height: rect.height)
                                .position(x: rect.midX, y: rect.midY)
                                .allowsHitTesting(false)
                        }
                    }
                )
            }
            .coordinateSpace(name: "assetPanelViewport")
            .overlay(
                AssetSelectionView(
                    assetFrames: assetFrames,
                    onSelectionChanged: { rect in
                        handleAssetSelectionChanged(rect)
                    },
                    onSelectionEnded: { ids in
                        handleAssetSelectionEnded(ids)
                    }
                )
            )
        }
        )
    }
    
    /// 资产库框选：框选矩形变化时更新矩形与临时命中集合（视口坐标，与 assetFrames 同基准）
    func handleAssetSelectionChanged(_ rect: CGRect) {
        assetSelectionRect = rect
        assetRubberSelected = Set(
            assetFrames.filter { $0.value.intersects(rect) }.map { $0.key }
        )
    }
    
    /// 资产库框选：松手时提交选择；点击空白（未框到资产）则取消选择
    func handleAssetSelectionEnded(_ ids: [UUID]) {
        assetSelectionRect = nil
        assetRubberSelected = []
        if ids.isEmpty {
            // 点击空白 / 未框到资产：取消选择
            selectedAssetIDs.removeAll()
            debugLog("资产库：框选结束 未命中资产，取消选择")
        } else {
            // 框选：以框选结果替换当前选择
            selectedAssetIDs = Set(ids)
            debugLog("资产库：框选结束 选中\(ids.count)个资产")
        }
    }
    
    /// 当前过滤范围标题
    var assetScopeTitle: String {
        if let scope = assetScope,
           let project = projectStore.projects.first(where: { $0.id == scope }) {
            return project.title
        }
        return "全局"
    }
    
    // 拖拽时携带的资产 id：若当前项已被选中则携带全部选中项，否则只携带当前项
    func dragAssetIDs(for asset: AssetItem) -> [UUID] {
        if selectedAssetIDs.contains(asset.id) {
            return Array(selectedAssetIDs)
        }
        return [asset.id]
    }
}
