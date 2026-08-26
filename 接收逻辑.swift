// ============================================================
//  文件作用：接收逻辑。接收外部拖入的媒体文件（从 Finder 拖到应用窗口），
//  按落点区域导入：总资产库区域直接导入资产库，画布区域导入成节点。
//  复用现成上传函数 importAsset / thumbnailImage / fileKind（画布状态 公共函数.swift）。
//  互动文件：引用 画布状态 公共函数.swift（importAsset、thumbnailImage、fileKind、
//  UploadFileKind）、资产管理ui.swift（AssetCategory）、画布 ui.swift（CanvasStore）；
//  被 资产管理ui.swift（AssetManagementView）、画布 ui.swift（CanvasView）引用。
// ============================================================

import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - 外部媒体文件拖放接收（统一入口）

/// 按文件媒体类型映射资产分类（图片/视频/音频）
func assetCategory(for url: URL) -> AssetCategory {
    switch fileKind(of: url) {
    case .video: return .video
    case .audio: return .audio
    case .image: return .image
    }
}

/// 接收拖入的媒体文件：按目标区域导入。
/// - Parameters:
///   - urls: 拖入的文件 URL 列表
///   - store: 画布 store；nil 表示总资产库区域（直接导入资产库），非 nil 表示画布区域（导入成节点）
///   - nodePosition: 画布区域建节点的落点（视图坐标；nil 用默认位置）
func receiveDroppedMedia(urls: [URL], store: CanvasStore?, nodePosition: CGPoint? = nil) {
    guard !urls.isEmpty else { return }
    var index = 0
    for url in urls {
        guard let thumb = thumbnailImage(for: url) else { continue }
        let category = assetCategory(for: url)
        let isImage = category == .image
        // 画布区域：节点在落点基础上斜向错开，避免重叠
        var pos: CGPoint? = nil
        if let nodePosition = nodePosition {
            pos = CGPoint(x: nodePosition.x + CGFloat(index) * 24,
                          y: nodePosition.y + CGFloat(index) * 24)
        }
        importAsset(
            image: thumb,
            name: url.deletingPathExtension().lastPathComponent,
            category: category,
            sourceURL: isImage ? url : nil,
            mediaSourceURL: isImage ? nil : url,
            createNode: store != nil,
            nodePosition: pos,
            store: store
        )
        index += 1
    }
    debugLog("接收拖入媒体 \(urls.count) 个：\(store == nil ? "导入总资产库" : "导入画布成节点")")
}

// MARK: - 拖放目标修饰器（SwiftUI 挂载用）

/// 让任意视图接收外部媒体文件拖放，按区域导入：
/// - store 传 nil：总资产库区域，直接导入资产库
/// - store 传画布 store：画布区域，导入成节点（落点位置）
struct MediaDropReceiver: ViewModifier {
    var store: CanvasStore?
    var nodePositionProvider: (CGPoint) -> CGPoint? = { $0 }

    func body(content: Content) -> some View {
        content
            .dropDestination(for: URL.self) { urls, location in
                receiveDroppedMedia(urls: urls, store: store, nodePosition: nodePositionProvider(location))
                return true
            }
    }
}

extension View {
    /// 接收外部媒体文件拖放：总资产库区域（store 传 nil）直接导入，画布区域（store 传画布 store）导入成节点
    func mediaDropReceiver(store: CanvasStore? = nil, nodePosition: ((CGPoint) -> CGPoint?)? = nil) -> some View {
        modifier(MediaDropReceiver(store: store, nodePositionProvider: nodePosition ?? { $0 }))
    }
}
