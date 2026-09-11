// ============================================================
//  文件作用：资产管理（数据层逻辑）。AssetCategory / AssetItem 资产模型，AssetStore 资产库单例（按项目过滤/公共过滤/重命名）。
//  互动文件：AssetStore / AssetCategory 被 画布 ui.swift、交互.swift、画布 状态.swift、画布 资产面板.swift 及本文件 AssetManagementView 引用；saveAssetLibraryJSON / loadProjectNodes / deleteAssetFiles 在根目录与画布 Domain。
// ============================================================

import SwiftUI
import AppKit
import Foundation
import Combine

// MARK: - 资产模型（图片资产）

enum AssetCategory: String, CaseIterable, Identifiable, Codable {
    case character = "角色"
    case scene = "场景"
    case image = "图片"
    case video = "视频"
    case audio = "音频"
    var id: String { rawValue }
    
    /// 分类图标（资产库 Tab / 资产管理页 Tab 共用）
    var icon: String {
        switch self {
        case .character: return "person"
        case .scene: return "photo.on.rectangle.angled"
        case .image: return "photo"
        case .video: return "video"
        case .audio: return "music.note"
        }
    }
}

struct AssetItem: Identifiable {
    let id: UUID
    var name: String
    let image: NSImage
    var prompt: String
    let category: AssetCategory
    /// 磁盘上的图片文件名（如 <id>.png），由保存逻辑写入
    var fileName: String
    /// 原始媒体文件名（视频/音频等非图片资产保留原文件；nil 表示纯图片资产）
    var mediaFileName: String? = nil
    /// 所属项目 id（nil 表示全局资产）
    var projectID: UUID? = nil

    init(id: UUID = UUID(), name: String, image: NSImage, prompt: String = "", category: AssetCategory = .character, fileName: String = "", mediaFileName: String? = nil, projectID: UUID? = nil) {
        self.id = id
        self.name = name
        self.image = image
        self.prompt = prompt
        self.category = category
        self.fileName = fileName
        self.mediaFileName = mediaFileName
        self.projectID = projectID
    }
}

// MARK: - 资产库（开放 API 入口）

final class AssetStore: ObservableObject {
    static let shared = AssetStore()
    @Published var assets: [AssetItem] = []
    /// 最近编辑过提示词的资产 id 列表（画布侧监听后同步节点输入框，并清空）
    @Published var promptEditedAssetIDs: [UUID] = []

    private init() {}

    /// 按项目过滤资产（nil 表示全局资产）
    func assets(in projectID: UUID?) -> [AssetItem] {
        assets.filter { $0.projectID == projectID }
    }

    /// 资产库公共过滤（大纲资产面板 / 空节点资产面板共用）。
    /// - scope: nil = 全局（总资产库引用，返回全部资产，不限归属项目）；非 nil = 该项目所有节点实际使用的资产
    /// - category: 分类过滤；nil = 不限分类
    func filteredAssets(in scope: UUID?, category: AssetCategory? = nil) -> [AssetItem] {
        let base: [AssetItem]
        if let scope = scope {
            // 项目范围：收集该项目所有节点引用的资产文件名，再从资产库中匹配
            let usedNames = Set(loadProjectNodes(projectID: scope).flatMap { node -> [String] in
                var names: [String] = []
                if let f = node.imageFileName { names.append(f) }
                if let m = node.mediaFileName { names.append(m) }
                return names
            })
            base = assets.filter { asset in
                usedNames.contains(asset.fileName)
                    || (asset.mediaFileName.map { usedNames.contains($0) } ?? false)
            }
        } else {
            // 全局：总资产库引用 = 全部资产
            base = assets
        }
        guard let category = category else { return base }
        return base.filter { $0.category == category }
    }

    /// 重命名资产（触发 UI 刷新）
    func renameAsset(id: UUID, to newName: String) {
        guard let idx = assets.firstIndex(where: { $0.id == id }), !newName.isEmpty else { return }
        assets[idx].name = newName
        objectWillChange.send()
        saveAssetLibraryJSON()
    }
}
