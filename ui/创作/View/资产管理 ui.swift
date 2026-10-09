// ============================================================
//  文件作用：资产管理页（纯 View）。AssetManagementView 管理页（资产网格 +
//  属性管理面板 + 批量管理）、AssetThumbnailCell 资产格子。
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
import UniformTypeIdentifiers


// MARK: - 资产管理页面（左侧资产网格 + 右侧属性管理面板）

struct AssetManagementView: View {
    @StateObject private var assetStore = AssetStore.shared
    /// 媒体播放管理（观察播放状态刷新播放按钮；资产格子/属性预览共用）
    @ObservedObject private var playerManager = MediaPlayerManager.shared
    @State private var assetCategory: AssetCategory = .character
    @State private var pendingImportCategory: AssetCategory = .character
    @State private var showImageImporter = false
    @State private var isBatchManaging = false
    @State private var selectedAssets: Set<UUID> = []
    @State private var showDeleteAssetsConfirm = false
    @State private var selectedAssetID: UUID?
    @State private var isRenaming = false
    @State private var renameText = ""
    /// 属性管理预览区悬停状态（视频/音频资产悬停显示播放按钮）
    @State private var isPreviewHovered = false

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                // 顶部操作行：加号 + 批量管理 右上角
                HStack {
                    Spacer()
                    Menu {
                        ForEach(AssetCategory.allCases) { category in
                            Button {
                                debugLog("资产管理：添加\(category.rawValue)")
                                pendingImportCategory = category
                                showImageImporter = true
                            } label: {
                                Label("添加\(category.rawValue)", systemImage: category.icon)
                            }
                        }
                    } label: {
                        Image(systemName: "plus")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.primary)
                            .padding(6)
                            .background(Circle().fill(Color(red: 0.90, green: 0.90, blue: 0.92)))
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()

                    Button(action: {
                        debugLog("资产管理：\(isBatchManaging ? "完成" : "批量管理")")
                        withAnimation {
                            isBatchManaging.toggle()
                            if !isBatchManaging { selectedAssets.removeAll() }
                        }
                    }) {
                        Text(isBatchManaging ? "完成" : "批量管理")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(.primary)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(
                                Capsule().fill(Color(red: 0.90, green: 0.90, blue: 0.92))
                            )
                    }
                    .buttonStyle(.plain)
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
                .overlay(alignment: .bottom) {
                    if isBatchManaging && !assetStore.assets.isEmpty {
                        HStack {
                            Button("取消") {
                                debugLog("资产管理：取消批量管理")
                                isBatchManaging = false
                                selectedAssets.removeAll()
                            }
                            Spacer()
                            Text("已选 \(selectedAssets.count) 项")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Spacer()
                            Button("删除", role: .destructive) {
                                debugLog("资产管理：请求删除所选 \(selectedAssets.count) 项")
                                showDeleteAssetsConfirm = true
                            }
                            .disabled(selectedAssets.isEmpty)
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(Color(red: 0.90, green: 0.90, blue: 0.92))
                        )
                        .padding(.horizontal, 20)
                        .padding(.bottom, 12)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.93, green: 0.93, blue: 0.95))

            Divider()

            propertyPanel
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.93, green: 0.93, blue: 0.95))
        .fileImporter(isPresented: $showImageImporter,
                      allowedContentTypes: importContentTypes(for: pendingImportCategory),
                      allowsMultipleSelection: true) { result in
            handleAssetImport(result)
        }
        .confirmationDialog("删除所选资产？", isPresented: $showDeleteAssetsConfirm, titleVisibility: .visible) {
            Button("删除", role: .destructive) {
                debugLog("资产管理：确认删除 \(selectedAssets.count) 个资产")
                let toDelete = assetStore.assets.filter { selectedAssets.contains($0.id) }
                assetStore.assets.removeAll { selectedAssets.contains($0.id) }
                deleteAssetFiles(toDelete)
                saveAssetLibraryJSON()
                selectedAssets.removeAll()
                isBatchManaging = false
            }
            Button("取消", role: .cancel) {
                debugLog("资产管理：取消删除")
            }
        } message: {
            Text("将删除 \(selectedAssets.count) 个资产")
        }
        // 接收外部拖入的媒体文件：总资产库区域直接导入资产库（按文件类型自动分类）
        .mediaDropReceiver()
        // 区域互斥保护：离开总资产库（切到画布/创作等）停止资产库的播放
        .onDisappear {
            MediaPlayerManager.shared.stopAssetPlayback()
        }
    }

    // 右侧属性管理面板
    private var propertyPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("属性管理")
                .font(.headline)

            if let idx = selectedAssetIndex {
                let asset = assetStore.assets[idx]

                // 预览（图片缩略图；视频/音频资产悬停显示播放按钮，播放中显示视频画面）
                let isPreviewPlayable = (asset.category == .audio || asset.category == .video)
                    && asset.mediaFileName != nil && !asset.mediaFileName!.isEmpty
                let isPreviewPlaying = playerManager.isPlaying(assetID: asset.id)
                ZStack {
                    if isPreviewPlaying, asset.category == .video, let player = playerManager.videoPlayer {
                        VideoPlayerView(player: player)
                            .frame(maxWidth: .infinity)
                            .frame(height: 150)
                    } else {
                        Image(nsImage: asset.image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity)
                            .frame(height: 150)
                    }
                    // 播放按钮（视频/音频悬停显示；播放中再悬停可停止）
                    if isPreviewPlayable {
                        MediaPlayButton(isPlaying: isPreviewPlaying) {
                            debugLog("资产管理：属性预览播放「\(asset.name)」")
                            playerManager.toggle(asset: asset)
                        }
                        .opacity(isPreviewHovered ? 1 : 0)
                        .allowsHitTesting(isPreviewHovered)
                    }
                    // 播放中的底部进度条（公共容器：底部留白/悬停显隐统一；预览区进度条更高）
                    if isPreviewPlaying {
                        MediaProgressBarOverlay(
                            accentColor: AppSettings.shared.defaultNodeColor,
                            isVisible: isPreviewHovered,
                            onSeek: { ratio in
                                playerManager.seek(nodeID: asset.id, ratio: ratio)
                            },
                            barHeight: 12
                        )
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 150)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color(red: 0.86, green: 0.86, blue: 0.89), lineWidth: 1)
                )
                .onHover { isPreviewHovered = $0 }

                // 图片尺寸（小号显示）
                Text("\(Int(asset.image.size.width)) × \(Int(asset.image.size.height))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)

                // 内容1：图片名字（点击改名）
                VStack(alignment: .leading, spacing: 6) {
                    Text("图片名字")
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

    // 按分类渲染资产网格（空态 / 网格）
    private func assetGrid(for category: AssetCategory) -> some View {
        let items = assetStore.assets.filter { $0.category == category }
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
                ScrollView {
                    LazyVGrid(columns: assetColumns, spacing: 16) {
                        ForEach(items) { asset in
                            assetThumbnail(asset)
                        }
                    }
                    .padding(20)
                }
                .background(Color(red: 0.93, green: 0.93, blue: 0.95))
            }
        }
    }

    private var assetColumns: [GridItem] {
        [GridItem(.adaptive(minimum: 110), spacing: 16)]
    }

    private func assetThumbnail(_ asset: AssetItem) -> some View {
        AssetThumbnailCell(
            asset: asset,
            isBatchManaging: isBatchManaging,
            isSelected: selectedAssetID == asset.id,
            isChecked: selectedAssets.contains(asset.id),
            onToggleCheck: {
                debugLog("资产管理：勾选资产「\(asset.name)」")
                if selectedAssets.contains(asset.id) {
                    selectedAssets.remove(asset.id)
                } else {
                    selectedAssets.insert(asset.id)
                }
            },
            onSelect: {
                debugLog("资产管理：点击资产「\(asset.name)」")
                if isBatchManaging {
                    if selectedAssets.contains(asset.id) {
                        selectedAssets.remove(asset.id)
                    } else {
                        selectedAssets.insert(asset.id)
                    }
                } else {
                    selectedAssetID = asset.id
                }
            }
        )
    }

    /// 按目标分类返回可选文件类型（角色/场景/图片 → 图片；视频 → 视频；音频 → 音频）
    private func importContentTypes(for category: AssetCategory) -> [UTType] {
        switch category {
        case .character, .scene, .image: return [.image]
        case .video: return [.movie, .mpeg4Movie, .quickTimeMovie]
        case .audio: return [.audio, .mp3, .wav]
        }
    }

    /// 通用导入：图片走 sourceURL 入库，视频/音频走 mediaSourceURL 入库（与拖入落库同款逻辑）
    private func handleAssetImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            // 取帧/解码放后台（thumbnailImage 内部 await frameImage），完成后回主线程入库；
            // fileImporter 回调即主线程，Task @MainActor 保持 importAsset / UI 更新在主线程
            let category = pendingImportCategory
            Task { @MainActor in
                for url in urls {
                    let didStart = url.startAccessingSecurityScopedResource()
                    defer { if didStart { url.stopAccessingSecurityScopedResource() } }
                    guard let thumb = await thumbnailImage(for: url) else { continue }
                    let isImage = fileKind(of: url) == .image
                    importAsset(
                        image: thumb,
                        name: url.deletingPathExtension().lastPathComponent,
                        category: category,
                        sourceURL: isImage ? url : nil,
                        mediaSourceURL: isImage ? nil : url
                    )
                }
            }
        case .failure:
            break
        }
    }
}

// MARK: - 资产格子（缩略图 + 悬停播放按钮）

/// 总资产库单个资产格子：视频/音频资产悬停显示播放/停止按钮（与画布节点相同出现逻辑：
/// 仅悬停显示、离开隐藏、播放中再悬停可停止），视频播放中显示视频画面覆盖缩略图。
struct AssetThumbnailCell: View {
    let asset: AssetItem
    let isBatchManaging: Bool
    let isSelected: Bool
    let isChecked: Bool
    let onToggleCheck: () -> Void
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

                // 批量管理勾选按钮
                if isBatchManaging {
                    Button(action: onToggleCheck) {
                        Image(systemName: isChecked ? "checkmark.circle.fill" : "circle")
                            .font(.system(size: 20))
                            .foregroundColor(isChecked ? Color.black : Color.white)
                            .background(
                                Circle().fill(Color.white.opacity(isChecked ? 1 : 0.65))
                            )
                    }
                    .buttonStyle(.plain)
                    .padding(6)
                }
            }
            .frame(width: 120, height: 120)
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(isSelected ? Color.black : (isBatchManaging && isChecked ? Color.black : Color.clear), lineWidth: 2)
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
