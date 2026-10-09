//
//  读取&保存.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/16.
//

// ============================================================
//  文件作用：操作历史（撤销 / 重做）。CanvasSnapshot 记录一次操作前
//  画布的完整状态（节点 + 连线 + 视图），CanvasStore 扩展提供
//  pushSnapshot / undo / redo。覆盖操作：移动节点、创建节点、连线、
//  删除节点、复制节点、批量拖拽创建等。
//  互动文件：被 画布状态 公共函数.swift（undoStack/redoStack 存储）、
//  画布 ui.swift、画布 交互 鼠标.swift、画布&节点 交互 连线.swift、
//  节点 交互 鼠标.swift 调用 pushSnapshot 记录操作。
// ============================================================

import SwiftUI
import Combine

// ============================================================

// MARK: - 操作快照

/// 一次操作前的画布完整状态（节点 + 连线 + 视图），值类型深拷贝
struct CanvasSnapshot {
    let nodes: [CanvasNode]
    let connections: [NodeConnection]
    let offset: CGPoint
    let zoom: Double
    let ratio: CanvasStore.Ratio
}

// ============================================================

// MARK: - 操作历史（撤销 / 重做）

extension CanvasStore {
    /// 操作前记录快照（用于撤销）：记录当前节点 + 连线 + 视图状态
    func pushSnapshot() {
        undoStack.append(CanvasSnapshot(
            nodes: nodes,
            connections: connections,
            offset: offset,
            zoom: zoom,
            ratio: currentRatio
        ))
        if undoStack.count > 50 { undoStack.removeFirst() }
        redoStack.removeAll()
        canUndo = !undoStack.isEmpty
        canRedo = false
        // 所有修改操作的统一收口：此处立即落盘，各操作函数无需手动调 save
        save()
    }

    /// 撤销：恢复上一步操作前的画布状态
    func undo() {
        guard let last = undoStack.popLast() else { return }
        redoStack.append(CanvasSnapshot(
            nodes: nodes,
            connections: connections,
            offset: offset,
            zoom: zoom,
            ratio: currentRatio
        ))
        restore(last)
        canUndo = !undoStack.isEmpty
        canRedo = true
        save()
    }

    /// 重做：恢复被撤销的操作
    func redo() {
        guard let last = redoStack.popLast() else { return }
        undoStack.append(CanvasSnapshot(
            nodes: nodes,
            connections: connections,
            offset: offset,
            zoom: zoom,
            ratio: currentRatio
        ))
        restore(last)
        canUndo = true
        canRedo = !redoStack.isEmpty
        save()
    }

    /// 恢复快照并清理失效的选中态 / 尺寸缓存
    private func restore(_ snapshot: CanvasSnapshot) {
        nodes = snapshot.nodes
        connections = snapshot.connections
        offset = snapshot.offset
        zoom = snapshot.zoom
        currentRatio = snapshot.ratio
        // 选中态只保留仍存在的节点 / 连线
        selectedNodeIDs = selectedNodeIDs.intersection(Set(nodes.map { $0.id }))
        if let sel = selectedConnectionID, !connections.contains(where: { $0.id == sel }) {
            selectedConnectionID = nil
        }
        nodeSizes = nodeSizes.filter { (id, _) in nodes.contains(where: { $0.id == id }) }
    }
}

// ============================================================

// MARK: - File Paths & Directory Setup

/// Root directory: ~/Documents/无限画布（偏好设置可自定义，空则用默认）
/// Resolved at runtime via FileManager so it works on any machine.
var canvasRootURL: URL {
    let custom = AppSettings.shared.canvasRootPath
    if !custom.isEmpty {
        return URL(fileURLWithPath: custom, isDirectory: true)
    }
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    return docs.appendingPathComponent("无限画布", isDirectory: true)
}

/// Asset library directory: holds image files + 资产库.json
var assetLibraryURL: URL { canvasRootURL.appendingPathComponent("资产库", isDirectory: true) }

/// Projects directory: each project is a xxx.canvas/ folder containing project.json
var projectsURL: URL { canvasRootURL.appendingPathComponent("项目", isDirectory: true) }

/// Model directory: 本地模型权重存放（与 资产库/ 项目/ 同根）
var modelDirURL: URL { canvasRootURL.appendingPathComponent("model", isDirectory: true) }

/// Cache directory: 缓存文件（与 资产库/ 项目/ 同根）
var cacheDirURL: URL { canvasRootURL.appendingPathComponent("cache", isDirectory: true) }

/// Output directory: 生成产物按类型归档（与 资产库/ 项目/ 同根）
var outputDirURL: URL { canvasRootURL.appendingPathComponent("output", isDirectory: true) }

/// Output 子目录：图像 / 音频 / 视频
var outputImageDirURL: URL { outputDirURL.appendingPathComponent("图像", isDirectory: true) }
var outputAudioDirURL: URL { outputDirURL.appendingPathComponent("音频", isDirectory: true) }
var outputVideoDirURL: URL { outputDirURL.appendingPathComponent("视频", isDirectory: true) }

/// Ensure the folder structure exists on launch.
/// Creates 无限画布 / 资产库 / 项目 / model / cache / output（含图像/音频/视频） if missing;
/// leaves existing content untouched.
func ensureCanvasDirectories() {
    let fm = FileManager.default
    for dir in [canvasRootURL, assetLibraryURL, projectsURL,
                modelDirURL, cacheDirURL, outputDirURL,
                outputImageDirURL, outputAudioDirURL, outputVideoDirURL] {
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }
}

/// 迁移文档库：把旧地址的整个根目录（含 资产库/ 项目/）移动到新地址。
/// 若新地址为空，则初始化目录结构（等同新打开软件，建立新文档库）。
func migrateCanvasRoot(from oldPath: String, to newPath: String) {
    guard oldPath != newPath, !oldPath.isEmpty, !newPath.isEmpty else { return }
    let fm = FileManager.default
    let oldRoot = URL(fileURLWithPath: oldPath, isDirectory: true)
    let newRoot = URL(fileURLWithPath: newPath, isDirectory: true)
    guard oldRoot != newRoot else { return }

    // 确保新根目录的父目录存在
    try? fm.createDirectory(at: newRoot.deletingLastPathComponent(), withIntermediateDirectories: true)

    if fm.fileExists(atPath: newRoot.path) {
        // 新根已存在：合并 资产库 / 项目 / model / cache / output 内容
        for sub in ["资产库", "项目", "model", "cache", "output"] {
            let oldDir = oldRoot.appendingPathComponent(sub, isDirectory: true)
            let newDir = newRoot.appendingPathComponent(sub, isDirectory: true)
            guard fm.fileExists(atPath: oldDir.path) else { continue }
            if fm.fileExists(atPath: newDir.path) {
                // 逐项合并，跳过同名冲突
                if let items = try? fm.contentsOfDirectory(at: oldDir, includingPropertiesForKeys: nil) {
                    for item in items {
                        let dest = newDir.appendingPathComponent(item.lastPathComponent)
                        if !fm.fileExists(atPath: dest.path) {
                            try? fm.moveItem(at: item, to: dest)
                        }
                    }
                }
            } else {
                try? fm.moveItem(at: oldDir, to: newDir)
            }
        }
    } else {
        // 新根不存在：整个根目录（含 资产库/ 项目/）整体移动
        try? fm.moveItem(at: oldRoot, to: newRoot)
    }

    // 完成后初始化目录结构（幂等；新地址为空时等同新打开软件，建立新文档库）
    ensureCanvasDirectories()

    // 资产库路径已变，重新加载到 UI
    loadAssetLibrary()
}

// ============================================================

// MARK: - Asset Library Persistence

/// Codable record for one asset. NSImage is NOT stored here —
/// the image lives as a separate PNG file in 资产库/, referenced by fileName.
struct AssetRecord: Codable, Identifiable {
    var id: UUID
    var name: String
    var fileName: String
    var mediaFileName: String?
    var prompt: String
    var category: AssetCategory
    var projectID: UUID?
}

/// Write a single asset's image to 资产库/<fileName>.
/// If sourceURL is provided, copy the original image file as-is (no re-encode, fastest, lossless).
/// Otherwise encode the NSImage to PNG. Runs on a background thread.
/// If mediaSourceURL is provided (video/audio assets), the original media file is
/// additionally copied to 资产库/<mediaFileName>.
func saveAssetImage(_ asset: AssetItem, sourceURL: URL? = nil, mediaSourceURL: URL? = nil) {
    ensureCanvasDirectories()
    let fileURL = assetLibraryURL.appendingPathComponent(asset.fileName)
    debugLog("保存资产图片：\(asset.fileName)")
    DispatchQueue.global(qos: .userInitiated).async {
        if let src = sourceURL {
            try? FileManager.default.copyItem(at: src, to: fileURL)
        } else if let tiff = asset.image.tiffRepresentation,
                  let rep = NSBitmapImageRep(data: tiff),
                  let png = rep.representation(using: .png, properties: [:]) {
            try? png.write(to: fileURL)
        }
        // 媒体原文件：复制到 资产库/<mediaFileName>
        if let mediaSrc = mediaSourceURL, let mediaName = asset.mediaFileName {
            let mediaURL = assetLibraryURL.appendingPathComponent(mediaName)
            try? FileManager.default.copyItem(at: mediaSrc, to: mediaURL)
        }
    }
}

/// Delete asset image files from disk (called after removing assets from memory).
func deleteAssetFiles(_ assets: [AssetItem]) {
    let fm = FileManager.default
    for asset in assets {
        let fileURL = assetLibraryURL.appendingPathComponent(asset.fileName)
        try? fm.removeItem(at: fileURL)
        if let mediaName = asset.mediaFileName {
            let mediaURL = assetLibraryURL.appendingPathComponent(mediaName)
            try? fm.removeItem(at: mediaURL)
        }
    }
}

// ============================================================

// MARK: - Project Persistence

/// Codable record for one project file (项目/<id>.canvas/project.json).
/// cover / ratio are stored as their case names (Color is not Codable).
struct ProjectFile: Codable {
    var id: UUID
    var title: String
    var timestamp: String
    var episodeInfo: String?
    var content: String?
    var cover: String
    var nodes: [CanvasNode]
    var connections: [NodeConnection]
    var offset: CGPoint
    var zoom: Double
    var ratio: String
    var continuousVideoMode: Bool? = nil   // 连续视频模式开关（Optional：旧档缺字段时 decode 兼容）
}

/// Generic project save: write the project (with current canvas state)
/// into 项目/<id>.canvas/project.json. Creates the folder if missing.
func saveProject(_ project: ProjectItem, canvas: CanvasStore) {
    ensureCanvasDirectories()
    let dir = projectsURL.appendingPathComponent(project.id.uuidString + ".canvas", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    debugLog("保存项目：\(project.title)（节点\(canvas.nodes.count) 连线\(canvas.connections.count)）")
    let file = ProjectFile(
        id: project.id,
        title: project.title,
        timestamp: project.timestamp,
        episodeInfo: project.episodeInfo,
        content: project.content,
        cover: project.cover.caseName,
        nodes: canvas.nodes,
        connections: canvas.connections,
        offset: canvas.offset,
        zoom: canvas.zoom,
        ratio: canvas.currentRatio.rawValue,
        continuousVideoMode: canvas.continuousVideoMode
    )
    if let data = encodeJSONFile(file) {
        try? data.write(to: dir.appendingPathComponent("project.json"))
    }
}

/// Delete a project's folder (项目/<id>.canvas) from disk.
func deleteProjectFiles(_ project: ProjectItem) {
    let dir = projectsURL.appendingPathComponent(project.id.uuidString + ".canvas", isDirectory: true)
    try? FileManager.default.removeItem(at: dir)
}

/// Rebuild the project list from disk (项目/<id>.canvas/project.json) on launch.
/// Color-based covers can't be restored from JSON, so they fall back to placeholder.
func loadProjects() -> [ProjectItem] {
    ensureCanvasDirectories()
    let fm = FileManager.default
    guard let dirs = try? fm.contentsOfDirectory(at: projectsURL, includingPropertiesForKeys: nil) else { return [] }
    var items: [ProjectItem] = []
    for dir in dirs where dir.pathExtension == "canvas" {
        let jsonURL = dir.appendingPathComponent("project.json")
        guard let data = try? Data(contentsOf: jsonURL),
              let file = decodeJSONFile(ProjectFile.self, from: data) else { continue }
        let cover: ProjectCover
        switch file.cover {
        case "gradient": cover = .gradient([.gray])
        case "scene": cover = .scene(colors: [.gray], label: "")
        default: cover = .placeholder
        }
        items.append(ProjectItem(id: file.id,
                                 title: file.title,
                                 timestamp: file.timestamp,
                                 episodeInfo: file.episodeInfo,
                                 content: file.content,
                                 cover: cover))
    }
    // 新项目在前
    return items.sorted { $0.timestamp > $1.timestamp }
}

/// Load a project's canvas state (nodes/connections/offset/zoom/ratio) from disk
/// into the given CanvasStore, rebuilding the canvas content on entry.
func loadProjectCanvas(_ project: ProjectItem, into canvas: CanvasStore) {
    let jsonURL = projectsURL
        .appendingPathComponent(project.id.uuidString + ".canvas", isDirectory: true)
        .appendingPathComponent("project.json")
    guard let data = try? Data(contentsOf: jsonURL),
          let file = decodeJSONFile(ProjectFile.self, from: data) else {
        debugLog("加载项目画布失败：\(project.title)")
        return
    }
    canvas.nodes = file.nodes
    canvas.connections = file.connections
    canvas.offset = file.offset
    canvas.zoom = file.zoom
    if let ratio = CanvasStore.Ratio(rawValue: file.ratio) {
        canvas.currentRatio = ratio
    }
    canvas.continuousVideoMode = file.continuousVideoMode ?? true
    debugLog("加载项目画布：\(project.title)（节点\(file.nodes.count) 连线\(file.connections.count)）")
}

/// 读取某项目的节点列表（从磁盘加载，不改动当前画布状态）。
/// 供资产库「按项目过滤」使用：筛选该项目所有节点实际引用的资产。
func loadProjectNodes(projectID: UUID) -> [CanvasNode] {
    let jsonURL = projectsURL
        .appendingPathComponent(projectID.uuidString + ".canvas", isDirectory: true)
        .appendingPathComponent("project.json")
    guard let data = try? Data(contentsOf: jsonURL),
          let file = decodeJSONFile(ProjectFile.self, from: data) else {
        return []
    }
    return file.nodes
}

/// Write metadata into 资产库/资产库.json (small, fast).
func saveAssetLibraryJSON() {
    ensureCanvasDirectories()
    var records: [AssetRecord] = []
    for asset in AssetStore.shared.assets {
        records.append(AssetRecord(
            id: asset.id,
            name: asset.name,
            fileName: asset.fileName,
            mediaFileName: asset.mediaFileName,
            prompt: asset.prompt,
            category: asset.category,
            projectID: asset.projectID
        ))
    }
    let jsonURL = assetLibraryURL.appendingPathComponent("资产库.json")
    debugLog("保存资产库 json：\(records.count) 条")
    if let data = encodeJSONFile(records) {
        try? data.write(to: jsonURL)
    }
}

/// Load the asset library from 资产库/资产库.json + image files,
/// rebuilding AssetStore.shared.assets. Missing image files are skipped.
func loadAssetLibrary() {
    let jsonURL = assetLibraryURL.appendingPathComponent("资产库.json")
    guard let data = try? Data(contentsOf: jsonURL),
          let records = decodeJSONFile([AssetRecord].self, from: data) else { return }
    var items: [AssetItem] = []
    for rec in records {
        let fileURL = assetLibraryURL.appendingPathComponent(rec.fileName)
        guard let img = NSImage(contentsOf: fileURL) else { continue }
        items.append(AssetItem(
            id: rec.id,
            name: rec.name,
            image: img,
            prompt: rec.prompt,
            category: rec.category,
            fileName: rec.fileName,
            mediaFileName: rec.mediaFileName,
            projectID: rec.projectID
        ))
    }
    AssetStore.shared.assets = items
}

// ============================================================

// MARK: - JSON 编解码公共样板（文件持久化）

/// 文件持久化编码：prettyPrinted + sortedKeys（与既有磁盘格式一致），失败返回 nil
func encodeJSONFile<T: Encodable>(_ value: T) -> Data? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try? encoder.encode(value)
}

/// 文件持久化解码：默认配置，失败返回 nil
func decodeJSONFile<T: Decodable>(_ type: T.Type, from data: Data) -> T? {
    try? JSONDecoder().decode(type, from: data)
}

