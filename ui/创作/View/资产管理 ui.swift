// ============================================================
//  文件作用：资产管理页（纯 View）。AssetManagementView 管理页（资产网格 +
//  属性管理面板 + 单选/多选/空白框选 + Delete 键与右键菜单删除）、
//  AssetThumbnailCell 资产格子、AssetDeleteKeyHandler Delete 键监听。
//  互动文件：AssetManagementView 被 主ui.swift 引用；
//  资产模型与状态已拆分至 ../Domain/资产 状态.swift（AssetCategory / AssetItem /
//  AssetStore）；AssetStore / AssetCategory 被 画布 ui.swift、交互.swift、
//  画布 状态.swift、画布 资产面板.swift 引用。
// ============================================================
// ============================================================
//  无限画布 · 资产管理
// ============================================================
//  【导入资产（图片 + 提示词）】
//  在项目其他 Swift 文件中，直接调用全局统一导入函数 importAsset
//  （定义在 画布/公共 核心/画布状态 公共函数.swift）：
//
//  importAsset(
//      image:    NSImage,                    // 必填：要导入的图片
//      name:     String? = nil,              // 可选：资产名字，默认「资产 N」
//      prompt:   String  = "",               // 可选：该图片的提示词
//      category: AssetCategory = .character  // 可选：.character 角色 / .scene 场景
//  )
//
//  示例：
//      let img = NSImage(contentsOfFile: "/path/to/photo.png")!
//      importAsset(image: img,
//                  name: "主角",
//                  prompt: "赛博朋克风格，霓虹灯光",
//                  category: .character)
//
//  导入后自动出现在「资产管理」页对应分类下，可在右侧「属性管理」面板
//  点击名字改名、编辑提示词。
//  更多目标参数（addToLibrary / createNode / nodePosition / targetNodeID / store）
//  见 画布状态 公共函数.swift 中 importAsset 的完整签名。
// ============================================================

import SwiftUI
import Combine
import AppKit
import AVFoundation
import UniformTypeIdentifiers


// MARK: - 资产管理页面（左侧资产网格 + 右侧属性管理面板）

// MARK: - 资产排序方式

/// 总资产库排序方式：名称（A→Z，自然序）/ 加入时间（新→旧）
enum AssetSortOrder: String, CaseIterable, Identifiable {
    case name = "名称"
    case addedTime = "加入时间"

    var id: String { rawValue }

    /// 按钮上的小图标（名称 = 文字排序，加入时间 = 时钟）
    var icon: String {
        switch self {
        case .name: return "textformat.abc"
        case .addedTime: return "clock"
        }
    }
}

/// 让底层 NSScrollView 使用「常驻」的系统滚动条（NSScroller legacy 样式）。
///
/// 为什么必须动 AppKit 层：系统偏好 AppleShowScrollBars 未设置时等于「自动」，
/// SwiftUI 的 ScrollView 会用 overlay（悬浮）滚动条 —— 静止时淡出、几乎不可见。
/// `.scrollIndicators(.visible)` 只是让它在滚动时浮现，不会改变 NSScroller 的样式，
/// 所以在这类系统上无论怎么改 SwiftUI 修饰符，静态都看不到滚动条。
/// 这里只把底层 NSScrollView 的 scrollerStyle 切到 .legacy 并关闭自动隐藏，
/// 用的仍然完全是系统原生 NSScroller，没有绘制任何自定义控件。
struct PersistentScrollIndicator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView { ProbeView() }

    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? ProbeView)?.apply()
    }

    final class ProbeView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            apply()
        }

        func apply() {
            // 刚上屏时 NSScrollView 的内部装配可能尚未完成，做几次递进重试
            for delay in [0.0, 0.05, 0.2, 0.6] {
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                    guard let scrollView = self?.enclosingScrollView else { return }
                    scrollView.scrollerStyle = .legacy
                    scrollView.autohidesScrollers = false
                    scrollView.hasVerticalScroller = true
                }
            }
        }
    }
}

struct AssetManagementView: View {
    @StateObject private var assetStore = AssetStore.shared
    /// 全局设置（观察默认节点颜色，导入按钮底色跟随全局主题色）
    @ObservedObject private var settings = AppSettings.shared
    /// 媒体播放管理（观察播放状态刷新播放按钮；资产格子/属性预览共用）
    @ObservedObject private var playerManager = MediaPlayerManager.shared
    @State private var assetCategory: AssetCategory = .character
    @State private var showImageImporter = false
    /// 导入按钮悬停态（仅用于动效，不影响逻辑）
    @State private var isImportHovered = false
    /// 排序方式（名称 / 加入时间），默认按加入时间（最新在前）
    @State private var assetSort: AssetSortOrder = .addedTime
    /// 排序方向：false = 升序，true = 降序（默认加入时间降序 = 新→旧）
    @State private var assetSortDescending = true
    /// 选中资产集合（单选=1 项，Command/Shift 点选与空白框选=多项；Delete 键与右键菜单删除均按此集合）
    @State private var selectedAssets: Set<UUID> = []
    @State private var showDeleteAssetsConfirm = false
    /// 待删除的资产 id（右键/Delete 触发时锁定，确认弹窗按此执行）
    @State private var pendingDeleteIDs: Set<UUID> = []
    @State private var selectedAssetID: UUID?
    /// 框选：框选矩形 / 框选临时命中资产 / 每个资产项在网格坐标系中的 frame（与大纲视图资产页同一套基准）
    @State private var assetSelectionRect: CGRect?
    @State private var assetRubberSelected: Set<UUID> = []
    @State private var assetFrames: [UUID: CGRect] = [:]
    /// 方向键切换后需要滚动到可视区的资产 id（网格内 onChange 消费后清空）
    @State private var scrollTargetAssetID: UUID?
    @State private var isRenaming = false
    @State private var renameText = ""
    /// 属性管理面板展示的媒体信息（异步读取，避免阻塞主线程）
    @State private var mediaInfo: MediaInfo?
    /// 当前 mediaInfo 对应的资产 id（切换选中时丢弃过期结果）
    @State private var mediaInfoAssetID: UUID?

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                // 顶部操作行：导入素材按钮（底色跟随全局主题色）+ 排序下拉菜单，整组右对齐
                HStack(spacing: 8) {
                    Spacer()
                    Button {
                        debugLog("资产管理：打开素材导入面板")
                        showImageImporter = true
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "plus")
                                .font(.system(size: 10.5, weight: .bold))
                            Text("导入素材")
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(
                                    LinearGradient(
                                        colors: [
                                            settings.defaultNodeColor,
                                            settings.defaultNodeColor.opacity(0.82)
                                        ],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                        )
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Color.white.opacity(0.28), lineWidth: 0.8)
                        )
                        .shadow(
                            color: settings.defaultNodeColor.opacity(isImportHovered ? 0.45 : 0.22),
                            radius: isImportHovered ? 7 : 3,
                            x: 0,
                            y: isImportHovered ? 2.5 : 1.5
                        )
                        .scaleEffect(isImportHovered ? 1.04 : 1.0)
                        .contentShape(Capsule(style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .onHover { hovering in
                        withAnimation(.easeOut(duration: 0.14)) {
                            isImportHovered = hovering
                        }
                    }
                    .help("导入素材：可多选，图片 / 视频 / 音频自动归类，其它类型自动跳过")

                    // 排序下拉：名称 / 加入时间 + 升序·降序切换，菜单内勾选当前项与当前方向
                    Menu {
                        ForEach(AssetSortOrder.allCases) { order in
                            Button {
                                withAnimation(.easeOut(duration: 0.16)) {
                                    assetSort = order
                                }
                                debugLog("资产管理：排序方式切换为 \(order.rawValue)")
                            } label: {
                                if assetSort == order {
                                    Label(order.rawValue, systemImage: "checkmark")
                                } else {
                                    Text(order.rawValue)
                                }
                            }
                        }
                        Divider()
                        Button {
                            withAnimation(.easeOut(duration: 0.16)) {
                                assetSortDescending = false
                            }
                            debugLog("资产管理：排序方向切换为 升序")
                        } label: {
                            if !assetSortDescending {
                                Label("升序", systemImage: "checkmark")
                            } else {
                                Text("升序")
                            }
                        }
                        Button {
                            withAnimation(.easeOut(duration: 0.16)) {
                                assetSortDescending = true
                            }
                            debugLog("资产管理：排序方向切换为 降序")
                        } label: {
                            if assetSortDescending {
                                Label("降序", systemImage: "checkmark")
                            } else {
                                Text("降序")
                            }
                        }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: assetSort.icon)
                                .font(.system(size: 10, weight: .semibold))
                            Text(assetSort.rawValue)
                                .font(.system(size: 12, weight: .medium))
                            // 方向指示：升序 ↑ / 降序 ↓
                            Image(systemName: assetSortDescending ? "arrow.down" : "arrow.up")
                                .font(.system(size: 9, weight: .bold))
                        }
                        .foregroundColor(.primary.opacity(0.85))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule(style: .continuous)
                                .fill(Color.white.opacity(0.92))
                        )
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Color.black.opacity(0.08), lineWidth: 0.8)
                        )
                        .shadow(color: .black.opacity(0.06), radius: 2.5, x: 0, y: 1)
                        .contentShape(Capsule(style: .continuous))
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("排序：\(assetSort.rawValue)（\(assetSortDescending ? "降序" : "升序")），可在菜单中切换方式与方向")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)

                // TabView 占满内容区
                TabView(selection: $assetCategory) {
                    ForEach(AssetCategory.allCases) { category in
                        assetGrid(for: category)
                            .tabItem { Label(category.rawValue, systemImage: category.icon) }
                            .tag(category)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                // 选中状态提示（无按钮：删除只走 Delete 键 / 右键菜单）
                .overlay(alignment: .bottom) {
                    if selectedAssets.count > 1 {
                        Text("已选 \(selectedAssets.count) 项 · Delete 删除")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                Capsule().fill(Color(red: 0.90, green: 0.90, blue: 0.92).opacity(0.95))
                            )
                            .padding(.bottom, 10)
                            .allowsHitTesting(false)
                    }
                }
                .onChange(of: assetCategory) { _, newValue in
                    // 切分类：清空选中与框选态，避免对不可见资产误删
                    debugLog("资产管理：分类切换为 \(newValue.rawValue)，清空选中")
                    selectedAssets.removeAll()
                    assetSelectionRect = nil
                    assetRubberSelected = []
                    assetFrames.removeAll()
                }
                // 排序变化后旧 frame 与新排列不对应，清掉等重排后重新上报
                .onChange(of: assetSort) { _, _ in assetFrames.removeAll() }
                .onChange(of: assetSortDescending) { _, _ in assetFrames.removeAll() }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.93, green: 0.93, blue: 0.95))

            Divider()

            propertyPanel
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.93, green: 0.93, blue: 0.95))
        .fileImporter(isPresented: $showImageImporter,
                      allowedContentTypes: assetImportContentTypes,
                      allowsMultipleSelection: true) { result in
            handleAssetImport(result)
        }
        .confirmationDialog("删除所选资产？", isPresented: $showDeleteAssetsConfirm, titleVisibility: .visible) {
            Button("删除", role: .destructive) {
                performDelete(ids: pendingDeleteIDs)
            }
            Button("取消", role: .cancel) {
                debugLog("资产管理：取消删除")
                pendingDeleteIDs = []
            }
        } message: {
            Text("将删除 \(pendingDeleteIDs.count) 个资产（含磁盘文件，不可恢复）")
        }
        // Delete 键（退格 51 / 向前删除 117）删除当前选中资产，单选/多选一致
        .background(
            AssetDeleteKeyHandler(
                canDelete: { !selectedAssets.isEmpty },
                onDelete: {
                    debugLog("资产管理：Delete 键删除 \(selectedAssets.count) 项")
                    requestDelete(ids: selectedAssets)
                }
            )
            .frame(width: 0, height: 0)
        )
        // 空格键：选中单个资产时打开 / 关闭快速预览（模仿 macOS 系统预览）
        .background(
            AssetQuickLookKeyHandler(
                selectedAsset: { singleSelectedAsset },
                onSpace: { asset in
                    debugLog("资产管理：空格快速预览「\(asset.name)」")
                    MediaQuickLookController.shared.toggle(asset: asset)
                },
                onArrow: { keyCode in
                    moveAssetSelection(arrowKeyCode: keyCode)
                }
            )
            .frame(width: 0, height: 0)
        )
        // 选中变化：关闭快速预览，并按新选中资产重新读取媒体信息
        .task(id: selectedAssetID) {
            guard let id = selectedAssetID,
                  let asset = assetStore.assets.first(where: { $0.id == id }) else {
                MediaQuickLookController.shared.close()
                mediaInfo = nil
                mediaInfoAssetID = nil
                return
            }
            // 预览窗口开着就原地换成新资产继续播（方向键切换 / 点选都跟随）
            if MediaQuickLookController.shared.isPresented {
                MediaQuickLookController.shared.preview(asset: asset)
            }
            // 不清空旧数据：切换期间沿用上一组字段并降透明度，避免卡片高度塌陷造成闪烁
            mediaInfoAssetID = id
            let info = await MediaInfoLoader.load(asset: asset)
            guard mediaInfoAssetID == id else { return }
            mediaInfo = info
            debugLog("资产管理：媒体信息读取完成「\(asset.name)」")
        }
        // 预览窗内按方向键：通知过来切换选中（选中变化后由 .task 更新预览内容）
        .onReceive(NotificationCenter.default.publisher(for: .assetQuickLookNavigate)) { note in
            guard let code = note.userInfo?["keyCode"] as? Int else { return }
            moveAssetSelection(arrowKeyCode: code)
        }
        // 多选 / 取消选中时预览不再对应单选目标，直接关闭
        .onChange(of: selectedAssets) { _, newValue in
            if newValue.count != 1 {
                MediaQuickLookController.shared.close()
            }
        }
        // 接收外部拖入的媒体文件：总资产库区域直接导入资产库（按文件类型自动分类）
        .mediaDropReceiver()
        // 区域互斥保护：离开总资产库（切到画布/创作等）停止资产库的播放与快速预览
        .onDisappear {
            MediaPlayerManager.shared.stopAssetPlayback()
            MediaQuickLookController.shared.close()
        }
    }

    /// 当前单选资产（空格快速预览只对恰好 1 个选中资产生效）
    private var singleSelectedAsset: AssetItem? {
        guard selectedAssets.count == 1, let id = selectedAssets.first else { return nil }
        return assetStore.assets.first { $0.id == id }
    }

    // MARK: 方向键切换选中

    private enum AssetNavDirection { case left, right, up, down }

    /// 方向键切换选中：左右 = 上一个/下一个；上下 = 网格上一行/下一行（按格子实际位置几何导航）
    private func moveAssetSelection(arrowKeyCode: Int) {
        let items = sortedAssets(assetStore.assets.filter { $0.category == assetCategory })
        guard !items.isEmpty else { return }
        let direction: AssetNavDirection
        switch arrowKeyCode {
        case 123: direction = .left
        case 124: direction = .right
        case 126: direction = .up
        default: direction = .down
        }

        // 没有任何选中：落到第一项
        guard let currentID = selectedAssetID,
              let currentIndex = items.firstIndex(where: { $0.id == currentID }) else {
            applySingleSelection(items[0])
            return
        }

        // 格子位置还没上报（分类刚切 / 首次布局）：按线性索引兜底
        guard !assetFrames.isEmpty else {
            let columns = 4
            let delta: Int
            switch direction {
            case .left: delta = -1
            case .right: delta = 1
            case .up: delta = -columns
            case .down: delta = columns
            }
            let nextIndex = min(max(currentIndex + delta, 0), items.count - 1)
            if nextIndex != currentIndex { applySingleSelection(items[nextIndex]) }
            return
        }

        if let nextID = neighborID(from: currentID, direction: direction),
           let next = items.first(where: { $0.id == nextID }) {
            applySingleSelection(next)
            return
        }
        // 行首按左 / 行尾按右：按「上一个 / 下一个」线性回退（符合直觉）；上下越界则不动
        if direction == .left || direction == .right {
            let delta = direction == .left ? -1 : 1
            let nextIndex = min(max(currentIndex + delta, 0), items.count - 1)
            if nextIndex != currentIndex { applySingleSelection(items[nextIndex]) }
        }
    }

    /// 在已上报的格子位置里找指定方向上最近的邻居（同排优先，跨行也能命中）
    private func neighborID(from currentID: UUID, direction: AssetNavDirection) -> UUID? {
        guard let current = assetFrames[currentID] else { return nil }
        var best: (id: UUID, score: CGFloat)?
        for (id, frame) in assetFrames where id != currentID {
            let dx = frame.midX - current.midX
            let dy = frame.midY - current.midY
            let forward: CGFloat
            let lateral: CGFloat
            switch direction {
            case .left:
                guard dx < -8 else { continue }
                forward = -dx
                lateral = abs(dy)
            case .right:
                guard dx > 8 else { continue }
                forward = dx
                lateral = abs(dy)
            case .up:
                guard dy < -8 else { continue }
                forward = -dy
                lateral = abs(dx)
            case .down:
                guard dy > 8 else { continue }
                forward = dy
                lateral = abs(dx)
            }
            // 主方向距离优先，横向偏移加权：同排邻居永远优于斜对角
            let score = forward + lateral * 2
            if best == nil || score < best!.score { best = (id, score) }
        }
        return best?.id
    }

    /// 单选某项（方向键用）：替换选择、锚定属性面板、滚动到可视区
    private func applySingleSelection(_ asset: AssetItem) {
        selectedAssets = [asset.id]
        selectedAssetID = asset.id
        assetRubberSelected.removeAll()
        isRenaming = false
        scrollTargetAssetID = asset.id
        debugLog("资产管理：方向键切换选中「\(asset.name)」")
    }

    // 右侧属性管理面板
    private var propertyPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("属性管理")
                .font(.headline)

            if let idx = selectedAssetIndex {
                let asset = assetStore.assets[idx]

                // 媒体信息（媒体专业数据：分辨率 / 时长 / 帧率 / 编码 / 码率 / 采样率 / 声道 / 文件大小 / 修改时间）
                mediaInfoCard(for: asset)

                // 内容1：名字（点击改名）
                VStack(alignment: .leading, spacing: 6) {
                    Text("名字")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if isRenaming {
                        TextField("输入新名字", text: $renameText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { commitRename() }
                            .onExitCommand { isRenaming = false }
                    } else {
                        HStack(spacing: 6) {
                            Text(asset.name)
                                .font(.body.weight(.medium))
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Image(systemName: "pencil")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
                        )
                        .contentShape(Rectangle())
                        .onTapGesture {
                            renameText = asset.name
                            isRenaming = true
                        }
                    }
                }

                // 内容2：提示词输入框
                VStack(alignment: .leading, spacing: 6) {
                    Text("提示词")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextEditor(text: Binding(
                        get: { assetStore.assets[idx].prompt },
                        set: { newValue in
                            assetStore.assets[idx].prompt = newValue
                            assetStore.promptEditedAssetIDs.append(assetStore.assets[idx].id)
                            saveAssetLibraryJSON()
                        }
                    ))
                    .font(.body)
                    .frame(minHeight: 140)
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(Color(red: 0.86, green: 0.86, blue: 0.89), lineWidth: 1)
                    )
                }
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "info.circle")
                        .font(.system(size: 36))
                        .foregroundColor(Color(red: 0.75, green: 0.75, blue: 0.78))
                    Text("点击左侧图片查看属性")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 16)
        .frame(width: 280)
        .background(Color(red: 0.90, green: 0.90, blue: 0.92))
    }

    // MARK: 属性面板 · 媒体信息

    /// 媒体信息卡：展示该资产的专业数据（图片 / 视频 / 音频共用一套行式排版）
    /// 高度按资产类别预留，切换选中时只做透明度过渡，不切分支、不塌陷，避免闪烁
    private func mediaInfoCard(for asset: AssetItem) -> some View {
        let isReady = (mediaInfoAssetID == asset.id) && (mediaInfo != nil)
        let rows = mediaInfo.map { infoRows(for: $0) } ?? []

        return VStack(alignment: .leading, spacing: 6) {
            if rows.isEmpty {
                Text("读取媒体信息…")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 2)
            } else {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(row.0)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Spacer(minLength: 8)
                        Text(row.1)
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundColor(.primary.opacity(0.85))
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
        }
        // 预留高度：按该类资产的最大行数固定，数据回来不再改变卡片高度
        .frame(maxWidth: .infinity, minHeight: reservedInfoHeight(for: asset.category), alignment: .topLeading)
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.white.opacity(0.7))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(red: 0.86, green: 0.86, blue: 0.89), lineWidth: 1)
        )
        // 读取中：沿用上一组数据、整体压暗；就绪后回到不透明（不出现空态）
        .opacity(isReady || rows.isEmpty ? 1 : 0.45)
        .animation(.easeOut(duration: 0.12), value: isReady)
    }

    /// 信息卡预留高度（行高 11pt 字 + 6 间距 ≈ 20pt/行，按类别最大行数取）
    private func reservedInfoHeight(for category: AssetCategory) -> CGFloat {
        let rowHeight: CGFloat = 20
        let rowCount: Int
        switch category {
        case .video: rowCount = 11
        case .audio: rowCount = 9
        case .image, .character, .scene: rowCount = 6
        }
        return CGFloat(rowCount) * rowHeight
    }

    /// 按资产类型拼装信息行（缺字段自动省略）
    private func infoRows(for info: MediaInfo) -> [(String, String)] {
        var rows: [(String, String)] = []
        if let v = MediaInfoFormat.dimensions(info.dimensions) { rows.append(("分辨率", v)) }
        if let d = info.duration {
            rows.append(("时长", MediaInfoFormat.duration(d)))
        }
        if let v = MediaInfoFormat.frameRate(info.frameRate) { rows.append(("帧率", v)) }
        if let v = info.videoCodec { rows.append(("视频编码", v)) }
        if let v = MediaInfoFormat.bitrate(info.videoBitrate) { rows.append(("视频码率", v)) }
        if let v = info.audioCodec { rows.append(("音频编码", v)) }
        if let v = MediaInfoFormat.sampleRate(info.sampleRate) { rows.append(("采样率", v)) }
        if let v = MediaInfoFormat.channels(info.channels) { rows.append(("声道", v)) }
        if let b = info.depth { rows.append(("位深", "\(b) bit")) }
        if let c = info.colorSpace { rows.append(("色彩", c)) }
        if let v = MediaInfoFormat.bitrate(info.audioBitrate), info.videoBitrate == nil {
            rows.append(("音频码率", v))
        }
        if let v = MediaInfoFormat.fileSize(info.fileSize) { rows.append(("文件大小", v)) }
        if let v = MediaInfoFormat.date(info.modifiedAt) { rows.append(("修改时间", v)) }
        if let f = info.fileType { rows.append(("格式", f)) }
        return rows
    }

    // 当前选中资产在数组中的索引
    private var selectedAssetIndex: Int? {
        guard let id = selectedAssetID else { return nil }
        return assetStore.assets.firstIndex { $0.id == id }
    }

    // 提交改名
    private func commitRename() {
        guard let idx = selectedAssetIndex, !renameText.isEmpty else {
            isRenaming = false
            return
        }
        assetStore.renameAsset(id: assetStore.assets[idx].id, to: renameText)
        isRenaming = false
    }

    /// 按当前排序方式与方向排序资产：名称用 Finder 习惯自然序，加入时间按文件落盘时间
    private func sortedAssets(_ assets: [AssetItem]) -> [AssetItem] {
        let descending = assetSortDescending
        switch assetSort {
        case .name:
            return assets.sorted {
                let result = $0.name.localizedStandardCompare($1.name)
                return descending ? result == .orderedDescending : result == .orderedAscending
            }
        case .addedTime:
            // 加入时间取资产文件落盘时间（导入/保存即写入），同时间按名称兜底保证顺序稳定
            let dates = addedDates(for: assets)
            return assets.sorted {
                let l = dates[$0.id] ?? .distantPast
                let r = dates[$1.id] ?? .distantPast
                if l != r { return descending ? l > r : l < r }
                return $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
        }
    }

    /// 批量读取资产文件创建时间（缺失时退化为修改时间；一次遍历避免排序比较时反复 stat）
    private func addedDates(for assets: [AssetItem]) -> [UUID: Date] {
        var result: [UUID: Date] = [:]
        let fm = FileManager.default
        for asset in assets {
            let url = assetLibraryURL.appendingPathComponent(asset.fileName)
            let attrs = try? fm.attributesOfItem(atPath: url.path)
            let created = attrs?[.creationDate] as? Date
            let modified = attrs?[.modificationDate] as? Date
            if let date = created ?? modified {
                result[asset.id] = date
            }
        }
        return result
    }

    // 按分类渲染资产网格（空态 / 网格）
    private func assetGrid(for category: AssetCategory) -> some View {
        let items = sortedAssets(assetStore.assets.filter { $0.category == category })
        return Group {
            if items.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: category.icon)
                        .font(.system(size: 40))
                        .foregroundColor(Color(red: 0.75, green: 0.75, blue: 0.78))
                    Text("暂无\(category.rawValue)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                GeometryReader { geo in
                    ScrollViewReader { scrollProxy in
                    ScrollView(.vertical) {
                        ZStack(alignment: .topLeading) {
                            LazyVGrid(columns: assetColumns, spacing: 16) {
                                ForEach(items) { asset in
                                    assetThumbnail(asset)
                                        // 供方向键切换后 scrollTo 定位
                                        .id(asset.id)
                                }
                            }
                            .padding(20)
                        }
                        // 宽度锁死为视口宽度（原来只给 minWidth，等于放开上限）：
                        // 框选矩形原先作为 ZStack 子视图参与布局，拖动时把内容撑宽 → LazyVGrid(.adaptive) 重算列数 → 整片格子拉伸重排。
                        // 常驻滚动条（.legacy）会占用 15pt 宽的系统滚动条槽，
                        // 内容宽度必须扣掉这 15pt，否则最右一列会被滚动条槽压住、右边距变窄。
                        .frame(width: max(geo.size.width - 15, 120), alignment: .topLeading)
                        // 高度仍然只给下限：内容不足一屏时，下方空白区同样可以起框选 / 取消选择
                        .frame(minHeight: geo.size.height, alignment: .topLeading)
                        // 框选矩形（与 assetFrames 同基准：网格内容坐标系）
                        // 挂 overlay 而不是放进 ZStack：overlay 不参与父视图尺寸计算，拖动框选时不再影响网格布局
                        .overlay(alignment: .topLeading) {
                            if let rect = assetSelectionRect {
                                Rectangle()
                                    .fill(Color.accentColor.opacity(0.15))
                                    .overlay(Rectangle().stroke(Color.accentColor, lineWidth: 1))
                                    .frame(width: rect.width, height: rect.height)
                                    .offset(x: rect.minX, y: rect.minY)
                                    .allowsHitTesting(false)
                            }
                        }
                        // 整块内容都可作为框选起点（桌面 Finder 式行为）。
                        // 修复根因：此前框选手势挂在一条固定「一屏高」的 Color.clear 上，
                        // 向下滚动后可视区已不在那一屏范围内，手势收不到事件，表现为框选时灵时不灵；
                        // 且挂在 background 子层上的手势无法被命中格子的点击派发到（只沿命中视图祖先链走）。
                        // 现改为挂在内容容器自身，覆盖整个可滚动区域，滚动到任何位置都能框选。
                        .contentShape(Rectangle())
                        .gesture(rubberBandGesture(items: items))
                        // 空白处单击（且未命中任何资产）才取消选择，避免与单元格点击冲突
                        .simultaneousGesture(
                            SpatialTapGesture(coordinateSpace: .named(assetGridCoordinateSpace))
                                .onEnded { value in
                                    let point = value.location
                                    let hitsAsset = assetFrames.contains { $0.value.contains(point) }
                                    guard !hitsAsset, !selectedAssets.isEmpty else { return }
                                    debugLog("资产管理：点击空白处，取消选择")
                                    selectedAssets.removeAll()
                                    selectedAssetID = nil
                                    isRenaming = false
                                }
                        )
                        .coordinateSpace(name: assetGridCoordinateSpace)
                        // 修正器挂在内容层内部（documentView 子树），才能向上找到该 ScrollView 的 NSScrollView
                        .background(PersistentScrollIndicator())
                        .onPreferenceChange(AssetFrameKey.self) { frames in
                            // 过滤掉已不在当前分类/已被删除资产的历史 frame，避免命中幽灵资产
                            let validIDs = Set(items.map(\.id))
                            assetFrames = frames.filter { validIDs.contains($0.key) }
                        }
                    }
                    // 系统原生滚动条常驻显示：.visible 覆盖系统「滚动时显示」的默认偏好，
                    // 让它像左侧大纲一样始终可见、可点轨道翻页、可拖滑块定位。
                    // 关键：这里不能再挂 .clipped()。.clipped() 会在 ScrollView 外再套一层独立裁剪层，
                    // 而 macOS 上滚动条（system scroller）绘制在该层之外，会被整条裁掉——
                    // 此前自绘 overlay 滚动条「一条都看不到」也是同一个原因。
                    // 内容本身已用 .frame(width:) 锁死为视口宽度，不会画到右侧属性面板，
                    // 因此去掉 .clipped() 不会产生溢出。
                    .scrollIndicators(.visible)
                    // 方向键切到屏外资产时滚到可视区（anchor nil = 最小距离滚动，不做居中跳转）
                    .onChange(of: scrollTargetAssetID) { _, newValue in
                        guard let newValue else { return }
                        withAnimation(.easeOut(duration: 0.12)) {
                            scrollProxy.scrollTo(newValue, anchor: nil)
                        }
                        scrollTargetAssetID = nil
                    }
                    }
                }
                .background(Color(red: 0.93, green: 0.93, blue: 0.95))
            }
        }
    }

    /// 框选命中检测用的坐标系名（网格内容坐标，滚动时随内容移动，与资产项 frame 同基准）
    private var assetGridCoordinateSpace: String { "assetManagementGrid" }

    /// 空白处框选：拖动生成矩形并实时高亮命中资产，松手提交选择
    private func rubberBandGesture(items: [AssetItem]) -> some Gesture {
        let validIDs = Set(items.map(\.id))
        // minimumDistance 取 3：与格子点击手势共存，轻微手抖仍判定为点击（保证单选稳定）
        return DragGesture(minimumDistance: 3, coordinateSpace: .named(assetGridCoordinateSpace))
            .onChanged { value in
                let rect = rubberRect(from: value.startLocation, to: value.location)
                assetSelectionRect = rect
                assetRubberSelected = hitAssetIDs(in: rect, validIDs: validIDs)
            }
            .onEnded { value in
                let rect = rubberRect(from: value.startLocation, to: value.location)
                let ids = hitAssetIDs(in: rect, validIDs: validIDs)
                assetSelectionRect = nil
                assetRubberSelected = []
                if ids.isEmpty {
                    // 点击空白 / 框选未命中资产：取消选择
                    debugLog("资产管理：框选未命中资产，取消选择")
                    selectedAssets.removeAll()
                    selectedAssetID = nil
                    isRenaming = false
                } else {
                    // 框选：以框选结果替换当前选择，属性面板锚定第一个命中资产
                    selectedAssets = Set(ids)
                    selectedAssetID = items.first { ids.contains($0.id) }?.id
                    isRenaming = false
                    debugLog("资产管理：框选结束 选中\(ids.count)个资产")
                }
            }
    }

    /// 由拖拽起止点生成规范化矩形（支持任意方向拖拽）
    private func rubberRect(from start: CGPoint, to end: CGPoint) -> CGRect {
        CGRect(x: min(start.x, end.x),
               y: min(start.y, end.y),
               width: abs(end.x - start.x),
               height: abs(end.y - start.y))
    }

    /// 命中检测：只统计当前列表内资产的 frame，避免历史/幽灵 frame 造成多选或漏选
    private func hitAssetIDs(in rect: CGRect, validIDs: Set<UUID>) -> Set<UUID> {
        guard rect.width > 0, rect.height > 0 else { return [] }
        var result: Set<UUID> = []
        for (id, frame) in assetFrames where validIDs.contains(id) {
            if frame.intersects(rect) { result.insert(id) }
        }
        return result
    }

    private var assetColumns: [GridItem] {
        [GridItem(.adaptive(minimum: 110), spacing: 16)]
    }

    private func assetThumbnail(_ asset: AssetItem) -> some View {
        AssetThumbnailCell(
            asset: asset,
            isSelected: selectedAssets.contains(asset.id) || assetRubberSelected.contains(asset.id),
            onSelect: { handleAssetClick(asset) }
        )
        // 上报资产项 frame（框选命中检测）
        .background(
            GeometryReader { geo in
                Color.clear.preference(
                    key: AssetFrameKey.self,
                    value: [asset.id: geo.frame(in: .named(assetGridCoordinateSpace))]
                )
            }
        )
        // 右键菜单：删除（未选中则先选中该项；已多选时按整个选中集合删除）
        .contextMenu {
            Button(role: .destructive) {
                debugLog("资产管理：右键删除资产「\(asset.name)」")
                if !selectedAssets.contains(asset.id) {
                    selectedAssets = [asset.id]
                    selectedAssetID = asset.id
                    isRenaming = false
                }
                requestDelete(ids: selectedAssets)
            } label: {
                Label(selectedAssets.count > 1 ? "删除所选 \(selectedAssets.count) 项" : "删除",
                      systemImage: "trash")
            }
        }
    }

    /// 点击资产：无修饰键 = 单选（替换选择，属性面板跟随）；Command/Shift = 加选/减选
    private func handleAssetClick(_ asset: AssetItem) {
        let flags = NSEvent.modifierFlags
        if flags.contains(.command) || flags.contains(.shift) {
            if selectedAssets.contains(asset.id) {
                selectedAssets.remove(asset.id)
                debugLog("资产管理：取消选择资产「\(asset.name)」")
                if selectedAssetID == asset.id { selectedAssetID = nil }
            } else {
                selectedAssets.insert(asset.id)
                selectedAssetID = asset.id
                debugLog("资产管理：加选资产「\(asset.name)」")
            }
            isRenaming = false
        } else {
            selectedAssets = [asset.id]
            selectedAssetID = asset.id
            isRenaming = false
            debugLog("资产管理：选中资产「\(asset.name)」")
        }
    }

    /// 请求删除（Delete 键 / 右键菜单共用）：锁定待删集合后走确认弹窗
    private func requestDelete(ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        pendingDeleteIDs = ids
        showDeleteAssetsConfirm = true
    }

    /// 执行删除：内存 + 磁盘文件，并把选中/属性面板/播放状态一并清理
    private func performDelete(ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        debugLog("资产管理：确认删除 \(ids.count) 个资产")
        if let playing = playerManager.playingNodeID, ids.contains(playing) {
            // 正在播放的资产被删除：先停止播放，避免播放已删除文件
            playerManager.stop()
        }
        let toDelete = assetStore.assets.filter { ids.contains($0.id) }
        assetStore.assets.removeAll { ids.contains($0.id) }
        deleteAssetFiles(toDelete)
        saveAssetLibraryJSON()
        selectedAssets.subtract(ids)
        assetRubberSelected.subtract(ids)
        if let sel = selectedAssetID, ids.contains(sel) {
            selectedAssetID = nil
            isRenaming = false
        }
        pendingDeleteIDs = []
    }

    /// 导入面板可选类型：图片 / 视频 / 音频三类全放开（选完再按真实类型自动归类）
    private var assetImportContentTypes: [UTType] {
        [.image, .movie, .video, .audio, .mpeg4Movie, .quickTimeMovie, .mp3, .wav]
    }

    /// 按扩展名判定导入目标分类；不属于图片 / 视频 / 音频三类返回 nil（跳过不导入）
    private func importCategory(for url: URL) -> AssetCategory? {
        let ext = url.pathExtension.lowercased()
        if ["mp4", "mov", "m4v", "avi", "mkv", "webm", "mpg", "mpeg"].contains(ext) {
            return .video
        }
        if ["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma"].contains(ext) {
            return .audio
        }
        if ["png", "jpg", "jpeg", "gif", "bmp", "webp", "heic", "heif", "tiff", "tif", "svg", "icns"].contains(ext) {
            return .image
        }
        return nil
    }

    /// 通用导入：图片走 sourceURL 入库，视频/音频走 mediaSourceURL 入库（与拖入落库同款逻辑）
    /// 非图片 / 视频 / 音频的文件直接跳过；全部导入完成后自动切到本次导入涉及的分类
    private func handleAssetImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            // 取帧/解码放后台（thumbnailImage 内部 await frameImage），完成后回主线程入库；
            // fileImporter 回调即主线程，Task @MainActor 保持 importAsset / UI 更新在主线程
            Task { @MainActor in
                var importedCategory: AssetCategory?
                var skippedCount = 0
                for url in urls {
                    guard let category = importCategory(for: url) else {
                        skippedCount += 1
                        debugLog("资产管理：跳过非素材文件 \(url.lastPathComponent)")
                        continue
                    }
                    let didStart = url.startAccessingSecurityScopedResource()
                    defer { if didStart { url.stopAccessingSecurityScopedResource() } }
                    guard let thumb = await thumbnailImage(for: url) else {
                        skippedCount += 1
                        debugLog("资产管理：跳过无法解析的文件 \(url.lastPathComponent)")
                        continue
                    }
                    let isImage = category == .image
                    importAsset(
                        image: thumb,
                        name: url.deletingPathExtension().lastPathComponent,
                        category: category,
                        sourceURL: isImage ? url : nil,
                        mediaSourceURL: isImage ? nil : url
                    )
                    if importedCategory == nil { importedCategory = category }
                }
                // 有导入则切到对应分类，直接看到新素材
                if let category = importedCategory {
                    assetCategory = category
                }
                debugLog("资产管理：导入完成，跳过 \(skippedCount) 个非素材文件")
            }
        case .failure:
            break
        }
    }
}

// MARK: - 资产格子（缩略图 + 悬停播放按钮）

/// 总资产库单个资产格子：视频/音频资产悬停显示播放/停止按钮（与画布节点相同出现逻辑：
/// 仅悬停显示、离开隐藏、播放中再悬停可停止），视频播放中显示视频画面覆盖缩略图。
/// 选中态由外部的 selectedAssets（单选/多选/框选）决定，格子本身只上报点击。
struct AssetThumbnailCell: View {
    let asset: AssetItem
    let isSelected: Bool
    let onSelect: () -> Void

    @State private var isHovered = false
    @ObservedObject private var playerManager = MediaPlayerManager.shared

    /// 是否可播放（视频/音频且有实际媒体内容）
    private var isPlayable: Bool {
        (asset.category == .audio || asset.category == .video)
            && asset.mediaFileName != nil && !asset.mediaFileName!.isEmpty
    }
    /// 当前资产是否正在播放
    private var isAssetPlaying: Bool {
        playerManager.isPlaying(assetID: asset.id)
    }

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .topTrailing) {
                // 缩略图 + 视频播放画面 + 悬停播放按钮（居中）
                ZStack {
                    Image(nsImage: asset.image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 120, height: 120)
                        .clipped()

                    // 视频播放中：显示视频画面（覆盖缩略图）
                    if isAssetPlaying, asset.category == .video, let player = playerManager.videoPlayer {
                        VideoPlayerView(player: player)
                            .frame(width: 120, height: 120)
                            .clipped()
                    }

                    // 播放按钮（音频/视频悬停显示；播放中再悬停可停止）
                    if isPlayable {
                        MediaPlayButton(isPlaying: isAssetPlaying) {
                            debugLog("资产管理：播放资产「\(asset.name)」")
                            playerManager.toggle(asset: asset)
                        }
                        .opacity(isHovered ? 1 : 0)
                        .allowsHitTesting(isHovered)
                    }
                    // 播放中的底部进度条（公共容器：底部留白/悬停显隐/命中区统一，见 MediaProgressBarOverlay）
                    if isAssetPlaying {
                        MediaProgressBarOverlay(
                            accentColor: AppSettings.shared.defaultNodeColor,
                            isVisible: isHovered,
                            onSeek: { ratio in
                                playerManager.seek(nodeID: asset.id, ratio: ratio)
                            }
                        )
                    }
                }
                .frame(width: 120, height: 120)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
            .frame(width: 120, height: 120)
            .overlay(
                // 选中态描边（单选/多选/框选一致，与大纲视图资产页同款高亮）
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
            )
            .contentShape(Rectangle())
            .onTapGesture(perform: onSelect)
            .onHover { isHovered = $0 }

            Text(asset.name)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(width: 120)
        }
    }
}

// ============================================================

// MARK: - 总资产库 Delete 键监听

/// 总资产库面板的 Delete 键删除监听（与 快捷键.swift 的 KeyboardShortcutHandler 同一套做法：
/// 本地事件监听 + 穿透视图，不参与鼠标命中测试）。
/// 生效条件：事件来自本面板所在窗口；文本输入（重命名 / 提示词）聚焦时不拦截；无 Command/Option/Control/Shift 修饰。
struct AssetDeleteKeyHandler: NSViewRepresentable {
    /// 当前是否有可删除的选中资产
    var canDelete: () -> Bool
    /// 触发删除（由调用方弹确认）
    var onDelete: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSView {
        let view = PassthroughView()
        context.coordinator.hostView = view
        context.coordinator.installMonitor()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator {
        var parent: AssetDeleteKeyHandler
        weak var hostView: NSView?
        private var monitor: Any?

        init(_ parent: AssetDeleteKeyHandler) {
            self.parent = parent
        }

        func installMonitor() {
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                // 仅处理本面板所在窗口（切到画布等其他窗口、其他面板不响应）
                guard let hostWindow = self.hostView?.window, event.window === hostWindow else { return event }
                // 文本输入（重命名输入框 / 提示词编辑器）聚焦时不拦截按键
                if let responder = hostWindow.firstResponder,
                   responder is NSTextView || responder is NSTextField {
                    return event
                }
                let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                // Delete（退格 51 / 向前删除 117），排除带修饰键组合
                guard event.keyCode == 51 || event.keyCode == 117,
                      !flags.contains(.command), !flags.contains(.option),
                      !flags.contains(.control), !flags.contains(.shift) else { return event }
                guard self.parent.canDelete() else { return event }
                self.parent.onDelete()
                return nil
            }
        }

        deinit {
            if let monitor = monitor {
                NSEvent.removeMonitor(monitor)
            }
        }
    }
}
