// ============================================================
//  文件作用：画布状态管理。定义 CanvasStore（平移/缩放/节点/连线/选中/撤销重做/
//  自动保存）、资产拖拽数据 AssetDragData、屏幕↔画布坐标换算函数
//  viewToCanvas / canvasToView、AssetCategory→NodeType 映射扩展。
//  互动文件：引用 资产管理ui.swift（AssetCategory）、节点ui.swift（NodeType）；
//  CanvasStore 被 画布 ui.swift、交互.swift、画布 工具栏.swift、工具栏2.swift、
//  菜单.swift、资产面板.swift、节点操作.swift 引用；AssetDragData 被 交互.swift、
//  资产面板.swift 引用；坐标换算被 交互.swift、画布 ui.swift、节点操作.swift 引用。
// ============================================================

import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers
import AVFoundation

// ============================================================

// MARK: - 滚轮滚动加速常量（全局统一，画布滚动 / 创作横向滚动共用，一个值控制）

/// 触控板（精确滚动）：轻度加速倍数（偏好设置可调）
var wheelScrollAccelerationPrecise: Double { AppSettings.shared.wheelAccelerationPrecise }
/// 普通鼠标滚轮（非精确滚动）：行→像素并明显加速倍数（偏好设置可调）
var wheelScrollAccelerationWheel: Double { AppSettings.shared.wheelAccelerationWheel }

// ============================================================

// MARK: - 资产拖拽数据（资产库 → 画布）

/// 拖拽到画布时携带的资产 id 列表（支持批量）
struct AssetDragData: Transferable, Codable {
    let assetIDs: [UUID]
    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .data)
    }
}

// MARK: - 资产分类 → 节点类型映射

extension AssetCategory {
    var nodeType: NodeType {
        switch self {
        case .character: return .character
        case .scene: return .scene
        case .image: return .image
        case .video: return .video
        case .audio: return .audio
        }
    }
}

// MARK: - 节点类型 → 资产分类映射（空节点上传/选资产时决定入库分类）

extension NodeType {
    /// 节点对应的资产分类（空节点上传/选资产时决定入库分类；text 无对应分类，归入图片）
    var assetCategory: AssetCategory {
        switch self {
        case .character: return .character
        case .scene: return .scene
        case .image: return .image
        case .video: return .video
        case .audio: return .audio
        case .text: return .image
        }
    }

    /// 空节点「资产库」面板允许的资产分类：视频只显示视频、音频只显示音频、图像/角色/场景显示这 3 种；文本无资产库按钮
    var allowedAssetCategories: [AssetCategory] {
        switch self {
        case .character, .scene, .image: return [.character, .scene, .image]
        case .video: return [.video]
        case .audio: return [.audio]
        case .text: return []
        }
    }
}

// MARK: - 上传文件类型过滤（公共函数）

/// 按上传入口需求返回允许的文件类型：
/// - nodeType == nil：右键上传，全部媒体类型
/// - nodeType 有值：空节点上传，受节点类型约束
func uploadAllowedTypes(for nodeType: NodeType?) -> [UTType] {
    guard let nodeType = nodeType else {
        return [.image, .movie, .video, .audio, .mpeg4Movie, .quickTimeMovie]
    }
    switch nodeType {
    case .character, .scene, .image, .text:
        return [.image]
    case .video:
        return [.movie, .video, .mpeg4Movie, .quickTimeMovie]
    case .audio:
        return [.audio]
    }
}

// MARK: - 上传文件分类（图片 / 视频 / 音频）

/// 上传文件的媒体类型
enum UploadFileKind {
    case image, video, audio
}

/// 按扩展名判断文件媒体类型（未知扩展名一律按图片处理）
func fileKind(of url: URL) -> UploadFileKind {
    let ext = url.pathExtension.lowercased()
    if ["mp4", "mov", "m4v", "avi", "mkv", "webm", "mpg", "mpeg"].contains(ext) {
        return .video
    }
    if ["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma"].contains(ext) {
        return .audio
    }
    return .image
}

// MARK: - 资产缩略图生成（图片直接加载 / 视频取首帧 / 音频占位图）

/// 同步提取视频指定时刻的帧图（copyCGImage 已弃用，用异步 API + 信号量保持同步语义）
func syncFrameImage(from generator: AVAssetImageGenerator, at time: CMTime) -> CGImage? {
    let semaphore = DispatchSemaphore(value: 0)
    var result: CGImage?
    generator.generateCGImageAsynchronously(for: time) { cgImage, _, _ in
        result = cgImage
        semaphore.signal()
    }
    semaphore.wait()
    return result
}

/// 从文件 URL 生成资产缩略图：图片直接加载；视频取首帧；音频生成占位图
func thumbnailImage(for url: URL) -> NSImage? {
    switch fileKind(of: url) {
    case .image:
        return NSImage(contentsOf: url)
    case .video:
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 800, height: 800)
        let time = CMTime(seconds: 0, preferredTimescale: 600)
        if let cgImage = syncFrameImage(from: generator, at: time) {
            return NSImage(cgImage: cgImage, size: .zero)
        }
        return nil
    case .audio:
        return audioPlaceholderImage()
    }
}

/// 音频资产占位图（白底 + 主题色梦幻浅渐变 + 主题色音符，随全局节点主题色换肤）
func audioPlaceholderImage() -> NSImage {
    let size = NSSize(width: 400, height: 400)
    let image = NSImage(size: size)
    image.lockFocus()
    let accent = AppSettings.shared.defaultNodeNSColor
    let rect = NSRect(origin: .zero, size: size)
    // 白底
    NSColor.white.setFill()
    NSBezierPath(rect: rect).fill()
    // 对角浅渐变（梦幻淡染：主题色 0.16 → 0.04，视觉几乎白底透一点色）
    if let gradient = NSGradient(colors: [accent.withAlphaComponent(0.16), accent.withAlphaComponent(0.04)]) {
        gradient.draw(in: rect, angle: 135)
    }
    // 主题色音符（半透明，与底色同系）
    if let symbol = NSImage(systemSymbolName: "music.note", accessibilityDescription: nil) {
        let symbolSize = NSSize(width: 160, height: 160)
        let symbolImage = NSImage(size: symbolSize)
        symbolImage.lockFocus()
        accent.withAlphaComponent(0.55).set()
        symbol.draw(in: NSRect(origin: .zero, size: symbolSize))
        symbolImage.unlockFocus()
        symbolImage.draw(in: NSRect(x: (size.width - symbolSize.width) / 2,
                                    y: (size.height - symbolSize.height) / 2,
                                    width: symbolSize.width,
                                    height: symbolSize.height))
    }
    image.unlockFocus()
    return image
}

// ============================================================

// MARK: - 坐标换算（屏幕 ↔ 画布）

/// 屏幕坐标 → 画布坐标
func viewToCanvas(_ point: CGPoint, offset: CGPoint, zoom: Double) -> CGPoint {
    CGPoint(x: (point.x - offset.x) / zoom, y: (point.y - offset.y) / zoom)
}

/// 画布坐标 → 屏幕坐标
func canvasToView(_ point: CGPoint, offset: CGPoint, zoom: Double) -> CGPoint {
    CGPoint(x: point.x * zoom + offset.x, y: point.y * zoom + offset.y)
}

// ============================================================

// MARK: - 画布状态管理

final class CanvasStore: ObservableObject {
    // 画布平移（用于拖拽移动）
    @Published var offset: CGPoint = .zero
    
    // 缩放比例
    @Published var zoom: Double = 1.0
    
    // 画布视口尺寸（由 CanvasView GeometryReader 上报；缩放适配/聚焦居中用）
    @Published var canvasViewSize: CGSize = .zero
    
    // 当前比例
    @Published var currentRatio: Ratio = .default
    
    // 当前视频时长（全局秒数；视频节点私有时长 nil 时跟随）
    @Published var currentDuration: VideoDuration = .default
    
    // 连续视频模式：开 = 视频任务完成后沿输出连线自动触发下游视频节点生成（默认开）
    @Published var continuousVideoMode: Bool = true
    
    // 节点主题色（加号 / 连线 / 选中框 / 悬停高亮）——统一走 AppSettings 单一数据源，偏好设置与画布右上角共用
    var nodeColor: Color {
        get { AppSettings.shared.defaultNodeColor }
        set { AppSettings.shared.setNodeColor(newValue) }
    }
    
    // 保存状态
    @Published var saveState: SaveState = .saved
    
    // 当前所属项目（进入画布时由 CanvasEditorView 注入，用于保存到正确目录）
    var currentProject: ProjectItem?
    
    // 网格吸附开关
    @Published var snapToGrid: Bool = true
    
    // 对齐参考线（拖动节点时显示，释放后清除）
    @Published var alignmentGuides: [AlignmentGuide] = []
    
    // 显示网格开关
    @Published var showGrid: Bool = true
    
    // 画布上的节点列表（didSet 清 inputValidity 缓存：节点内容一变，禁用判定需重算）
    @Published var nodes: [CanvasNode] = [] {
        didSet { inputValidityCache = nil }
    }
    
    // 节点拖动期间的整体位移（画布坐标）：拖动中只更新此值，松手才合并写回 nodes，
    // 避免每帧写 @Published nodes 导致内容层子树全量重建
    @Published var nodeDragOffset: CGPoint = .zero
    // 拖动起始时各拖动节点的原始位置（用于合并位移；命中检测仍用 nodes 原位置，鼠标层自算新位置）
    var draggingStartPositions: [UUID: CGPoint] = [:]
    // 拖动中的节点集合（渲染侧仅对这些节点叠加 nodeDragOffset）
    var draggingNodeIDs: Set<UUID> = []
    
    // 节点之间的连线（didSet 清 inputValidity 缓存：连线增删，禁用判定需重算）
    @Published var connections: [NodeConnection] = [] {
        didSet { inputValidityCache = nil }
    }
    
    /// 输入合规性判定缓存（[nodeID: InputValidity]）：滚动/平移画布只改 offset，nodes/connections 不变
    /// → 判定结果不变；nodes/connections 的 didSet 已负责失效。避免内容层 ForEach 每帧对每条连线
    /// 重复重算 O(入度×n) 的线性查找 + 文件路径解析
    private var inputValidityCache: [UUID: InputValidity]?
    
    // 连续视频模式发光线提示（运行期状态，不持久化）：级联传播即将触发的连线 id
    @Published var glowingConnectionIDs: Set<UUID> = []
    
    // 正在拖拽的连线（from 节点 + 当前鼠标画布坐标）
    @Published var draggingConnection: DraggingConnection?
    
    // 当前选中的连线 id（点击连线后高亮虚线 + 中心裁断按钮）
    @Published var selectedConnectionID: UUID? = nil
    
    // 框选选中的节点 id
    @Published var selectedNodeIDs: Set<UUID> = []
    // 框选矩形（屏幕坐标，左上原点）
    @Published var selectionRect: CGRect? = nil
    // 节点实际渲染尺寸（用于命中检测）
    @Published var nodeSizes: [UUID: CGSize] = [:]
    // 当前悬停的节点 id（nil 表示空白处）
    @Published var hoveredNodeID: UUID? = nil
    // 当前悬停的加号方向（nil 表示在节点矩形内，加号居中）
    @Published var hoveredPortSide: ConnectSide? = nil
    // 当前悬停节点加号相对球中心的偏移（nil 表示在球中心）
    @Published var hoveredPortOffset: CGPoint? = nil
    // 当前悬停空节点按钮的节点 id（nil 表示未悬停按钮）
    @Published var hoveredEmptyButtonNodeID: UUID? = nil
    // 当前悬停空节点按钮的动作（上传 / 资产库）
    @Published var hoveredEmptyButtonAction: EmptyNodeButtonAction? = nil
    // 连线拖拽中悬停的组盒加号（组 id / 方向 / 加号相对球心的吸附偏移，与节点加号 hoveredPort* 同构）
    @Published var hoveredGroupPortID: UUID? = nil
    @Published var hoveredGroupPortSide: ConnectSide? = nil
    @Published var hoveredGroupPortOffset: CGPoint? = nil
    
    // 撤销 / 重做可用状态
    @Published var canUndo: Bool = false
    @Published var canRedo: Bool = false
    
    // 退出兜底：应用正常退出（右键关闭 / Cmd+Q）时把最后的修改落盘
    private var terminateCancellable: AnyCancellable?
    
    // 撤销 / 重做快照栈（操作历史，见 读取&保存.swift）
    var undoStack: [CanvasSnapshot] = []
    var redoStack: [CanvasSnapshot] = []
    
    enum SaveState: String {
        case saved = "已保存"
        case saving = "保存中..."
        case unsaved = "未保存"
    }
    
    enum Ratio: String, CaseIterable, Codable {
        case ratio16_9 = "16:9"
        case ratio9_16 = "9:16"
        case ratio21_9 = "21:9"
        case ratio4_3 = "4:3"
        case ratio3_4 = "3:4"
        case ratio1_1 = "1:1"
        
        static let `default` = Ratio.ratio16_9
        
        var width: Double {
            switch self {
            case .ratio16_9: return 16.0
            case .ratio9_16: return 9.0
            case .ratio21_9: return 21.0
            case .ratio4_3: return 4.0
            case .ratio3_4: return 3.0
            case .ratio1_1: return 1.0
            }
        }
        
        var height: Double {
            switch self {
            case .ratio16_9: return 9.0
            case .ratio9_16: return 16.0
            case .ratio21_9: return 9.0
            case .ratio4_3: return 3.0
            case .ratio3_4: return 4.0
            case .ratio1_1: return 1.0
            }
        }
    }
    
    private var settingsCancellable: AnyCancellable?

    init() {
        // 订阅公共设置：偏好设置改色（含恢复默认）时画布自动刷新
        settingsCancellable = AppSettings.shared.objectWillChange
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.objectWillChange.send() }
            }
        // 应用正常退出（右键关闭 / Cmd+Q）时兜底保存最后一次修改
        terminateCancellable = NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)
            .sink { [weak self] _ in
                self?.save()
            }
    }
    
    deinit {
        settingsCancellable?.cancel()
        terminateCancellable?.cancel()
    }
    
    // 操作历史（pushSnapshot / undo / redo）见 读取&保存.swift
    
    // MARK: - 节点拖动（临时位移叠加：拖动中不改 nodes，松手一次性合并）
    
    /// 拖动开始：记录各拖动节点的起始位置，并清零位移
    func beginNodeDrag(startPositions: [UUID: CGPoint]) {
        draggingStartPositions = startPositions
        draggingNodeIDs = Set(startPositions.keys)
        nodeDragOffset = .zero
    }
    
    /// 拖动结束：把 nodeDragOffset 合并写回所有拖动节点的 position（一次 body 重建），并清空拖动状态
    func commitNodeDrag() {
        guard nodeDragOffset != .zero, !draggingStartPositions.isEmpty else {
            endNodeDrag()
            return
        }
        for (id, startPos) in draggingStartPositions {
            if let idx = nodes.firstIndex(where: { $0.id == id }) {
                nodes[idx].position = CGPoint(x: startPos.x + nodeDragOffset.x,
                                              y: startPos.y + nodeDragOffset.y)
            }
        }
        endNodeDrag()
    }
    
    /// 清空拖动状态（未实际拖动时的收尾）
    func endNodeDrag() {
        draggingStartPositions = [:]
        draggingNodeIDs = []
        nodeDragOffset = .zero
    }
    
    func markUnsaved() {
        saveState = .unsaved
    }
    
    func save() {
        guard let project = currentProject else { return }
        saveState = .saving
        saveProject(project, canvas: self)
        saveState = .saved
    }
    
    func reset() {
        pushSnapshot()
        offset = .zero
        zoom = 1.0
        currentRatio = .default
        currentDuration = .default
        // 缩放归位后把节点包围盒居中到视口中心，避免"缩放了但节点在别处找不到"
        if !nodes.isEmpty, canvasViewSize != .zero {
            let rect = nodesBoundingRect(nodes, nodeSizes: nodeSizes, zoom: zoom)
            offset = CGPoint(
                x: canvasViewSize.width / 2 - rect.midX,
                y: canvasViewSize.height / 2 - rect.midY
            )
        }
        // 不在此标记 saveState：reset 改了 offset/zoom/ratio（会持久化），是否落盘由调用方决定
        // （工具栏缩放适配 = reset()+save()；快捷键空格 = reset()+save()）
    }
}

// ============================================================

// MARK: - 统一导入入口（新图片进入系统：入库 / 画布建节点 / 已有节点补图）

// MARK: - 生成产物自动命名（统一入口）

/// 生成产物自动命名：扫描资产库目录下所有以 prefix 开头、带编号的已有文件，
/// 取最大编号 n，返回 n+1 对应的名字（不含扩展名，如 "ltx_3"）；无同名文件时从 1 开始。
/// 编号按前缀独立计数（ltx_ / hidream_ 各自计数，互不影响）；"ltx_3.mp4" 与 "ltx_3_thumb.png"
/// 均计入编号 3，保存前检查同名由编号递增天然避免覆盖。
/// - Parameters:
///   - prefix: 小写前缀（如 "ltx" / "hidream"），文件名形如 "<prefix>_<编号>.<扩展名>"
/// - Returns: 下一个可用名字，如 "ltx_3"（不含扩展名）
func nextAssetName(prefix: String) -> String {
    let dir = assetLibraryURL.path
    let lowerPrefix = prefix.lowercased()
    let marker = lowerPrefix + "_"
    var maxN = 0
    if let items = try? FileManager.default.contentsOfDirectory(atPath: dir) {
        for item in items {
            guard item.hasPrefix(marker) else { continue }
            // 提取 "ltx_" 之后的数字前缀（如 "ltx_3.mp4" / "ltx_3_thumb.png" → 3）
            let tail = item.dropFirst(marker.count)
            var digits = ""
            for ch in tail {
                guard ch.isNumber else { break }
                digits.append(ch)
            }
            if let n = Int(digits), n > maxN {
                maxN = n
            }
        }
    }
    return "\(lowerPrefix)_\(maxN + 1)"
}

/// 统一导入入口：任意来源的图片，按目标参数决定去向（入库 / 画布建节点 / 已有节点补图）。
/// 未来任何「添加」场景（总资产库按钮、右键空节点上传、外部文件拖入画布/资产库）都走这一个函数。
/// - Parameters:
///   - image: 必填，要导入的图片（视频/音频资产传缩略图或占位图）
///   - name: 可选，资产名字，默认「资产 N」
///   - prompt: 可选，该图片的提示词
///   - category: 可选，分类（.character 角色 / .scene 场景 / .image 图片 / .video 视频 / .audio 音频）
///   - sourceURL: 可选，原始图片文件路径；有则直接复制原文件（不重新编码，最快且无损）
///   - mediaSourceURL: 可选，原始媒体文件路径（视频/音频）；有则复制原文件到 资产库/<mediaFileName>
///   - projectID: 可选，所属项目 id（nil 表示全局资产）
///   - addToLibrary: 是否写入总资产库（默认 true）
///   - createNode: 是否在画布创建节点（默认 false）
///   - nodePosition: 新节点位置（画布坐标；nil 用默认位置）
///   - targetNodeID: 给已有节点补图（nil 表示不补）
///   - store: 画布 store（createNode / targetNodeID 时需要）
/// - Returns: 入库后的 AssetItem（未入库返回 nil）
@discardableResult
func importAsset(
    image: NSImage,
    name: String? = nil,
    prompt: String = "",
    category: AssetCategory = .character,
    sourceURL: URL? = nil,
    mediaSourceURL: URL? = nil,
    projectID: UUID? = nil,
    addToLibrary: Bool = true,
    createNode: Bool = false,
    nodePosition: CGPoint? = nil,
    targetNodeID: UUID? = nil,
    store: CanvasStore? = nil
) -> AssetItem? {
    // 1. 入库（可选）
    var asset: AssetItem? = nil
    if addToLibrary {
        let assetName = name?.isEmpty == false ? name! : "资产 \(AssetStore.shared.assets.count + 1)"
        let assetID = UUID()
        // 文件名优先用用户原始文件名；重名时追加序号避免覆盖
        var fileName = sourceURL?.lastPathComponent ?? (assetID.uuidString + ".png")
        if AssetStore.shared.assets.contains(where: { $0.fileName == fileName }) {
            let base = (fileName as NSString).deletingPathExtension
            let ext = (fileName as NSString).pathExtension
            var n = 1
            while AssetStore.shared.assets.contains(where: { $0.fileName == fileName }) {
                fileName = ext.isEmpty ? "\(base) (\(n))" : "\(base) (\(n)).\(ext)"
                n += 1
            }
        }
        // 媒体原文件名同样唯一化，避免覆盖
        var mediaFileName: String? = nil
        if let mediaSrc = mediaSourceURL {
            var mName = mediaSrc.lastPathComponent
            if AssetStore.shared.assets.contains(where: { $0.mediaFileName == mName }) {
                let base = (mName as NSString).deletingPathExtension
                let ext = (mName as NSString).pathExtension
                var n = 1
                while AssetStore.shared.assets.contains(where: { $0.mediaFileName == mName }) {
                    mName = ext.isEmpty ? "\(base) (\(n))" : "\(base) (\(n)).\(ext)"
                    n += 1
                }
            }
            mediaFileName = mName
        }
        let newAsset = AssetItem(id: assetID, name: assetName, image: image, prompt: prompt, category: category, fileName: fileName, mediaFileName: mediaFileName, projectID: projectID)
        AssetStore.shared.assets.append(newAsset)
        saveAssetImage(newAsset, sourceURL: sourceURL, mediaSourceURL: mediaSourceURL)
        saveAssetLibraryJSON()
        asset = newAsset
    }
    // 2. 画布建节点（可选）
    if createNode, let store = store, let asset = asset {
        let viewPos = canvasToView(nodePosition ?? .zero, offset: store.offset, zoom: store.zoom)
        addNode(from: asset, at: viewPos, offset: store.offset, zoom: store.zoom, store: store)
    }
    // 3. 已有节点补内容（可选）：空节点获得内容时标题自动改为内容名
    if let targetNodeID = targetNodeID, let store = store, let asset = asset {
        attachAssetToNode(asset, nodeID: targetNodeID, store: store)
    }
    return asset
}

// MARK: - 给节点补内容（统一入口）

/// 给已有节点设置实际内容。
/// - 空节点（imageFileName == nil）获得内容的那一刻：把内容名字设为节点标题。
/// - 已有内容的节点更换内容：仅换图，不改标题（尊重用户已命名的标题）。
@discardableResult
func attachAssetToNode(_ asset: AssetItem, nodeID: UUID, store: CanvasStore) -> Bool {
    guard let idx = store.nodes.firstIndex(where: { $0.id == nodeID }) else { return false }
    let wasEmpty = store.nodes[idx].imageFileName == nil
    store.nodes[idx].imageFileName = asset.fileName
    // 引用模型：节点不复制资产提示词，输入框通过 imageFileName 引用读资产 prompt
    if wasEmpty {
        store.nodes[idx].title = asset.name
    }
    store.save()
    return true
}

// MARK: - 资产拖拽创建节点（独立自由函数，对外暴露）

/// 从资产创建节点（拖拽释放位置，视图坐标转画布坐标）。
/// 独立于 NodeCanvasView，供资产拖拽落点等外部调用。
func addNode(from asset: AssetItem, at viewPos: CGPoint, offset: CGPoint, zoom: Double, store: CanvasStore) {
    let canvasPos = viewToCanvas(viewPos, offset: offset, zoom: zoom)
    let node = NodeFactory.createNode(
        type: asset.category.nodeType,
        title: asset.name,
        subtitle: asset.prompt.isEmpty ? nil : asset.prompt,
        position: canvasPos,
        imageFileName: asset.fileName,
        mediaFileName: asset.mediaFileName
        // 引用模型：不传 prompt，节点输入框通过 imageFileName 引用读资产 prompt
    )
    store.nodes.append(node)
}

// MARK: - 通用几何工具

/// 两点间欧氏距离（命中判定等重复样板统一入口）
func distance(from a: CGPoint, to b: CGPoint) -> CGFloat {
    let dx = a.x - b.x
    let dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)
}

// MARK: - 视频节点输入合规性（公共函数：线禁用灰 + 发送收集共用同一判定源）

/// 视频节点输入合规性判定结果
struct InputValidity {
    /// 是否适用判定（仅「视频节点且选了具体模型」才判定；其它节点不判不禁）
    var applies: Bool = false
    /// 明确无效的连线 ID 集合（UI：这些线禁用灰色；发送：跳过这些线不收集）
    var invalidConnectionIDs: Set<UUID> = []
    /// 合规图片路径（按连线顺序，最多 2 张）
    var imagePaths: [String] = []
    /// 合规音频路径（最多 1 个）
    var audioPath: String? = nil
}

extension CanvasStore {
    /// 计算节点的输入合规性（公共入口，按节点类型分派规则）。
    /// - 视频节点：选了具体模型才判定（ltx2.5 规则）
    /// - 图像/角色/场景节点：判定（HiDream-O1 规则：最多 5 张图，其他类型禁用）
    /// - 其它类型节点返回 applies=false（线保持主题色，不判不禁）
    func inputValidity(for nodeID: UUID) -> InputValidity {
        // 缓存命中直接返回：滚动/平移画布时节点与连线不变，判定结果不变（nodes/connections didSet 已负责失效）
        if let cache = inputValidityCache, let hit = cache[nodeID] {
            return hit
        }
        let result: InputValidity
        if let node = nodes.first(where: { $0.id == nodeID }) {
            switch node.type {
            case .video:
                switch node.model {
                case .ltx25Distill:
                    result = ltx25DistillInputValidity(for: node)
                }
            case .image, .character, .scene:
                result = hidreamInputValidity(for: node)
            default:
                result = InputValidity()
            }
        } else {
            result = InputValidity()
        }
        if inputValidityCache == nil { inputValidityCache = [:] }
        inputValidityCache?[nodeID] = result
        return result
    }
    
    /// ltx2.5 蒸馏版规则：
    /// - 空节点（无媒体内容）→ 无效线
    /// - 视频节点（未开尾帧）→ 无效线（不允许视频输入）
    /// - 开尾帧的视频节点视作图片 → 图片线，最多 2 条（超出的无效）
    /// - 图片/角色/场景节点 → 图片线，最多 2 条（超出的无效）
    /// - 音频节点 → 音频线，最多 1 条（超出的无效）
    /// - 其它类型连线（文本）不参与判定（不收集、不变灰）
    private func ltx25DistillInputValidity(for videoNode: CanvasNode) -> InputValidity {
        var result = InputValidity()
        result.applies = true
        var imageCount = 0
        var imageCandidates: [(y: CGFloat, path: String)] = []   // 有效图片输入：记录画布 y 用于定首尾帧
        var audioCount = 0
        for conn in connections where conn.toID == videoNode.id {
            guard let fromNode = nodes.first(where: { $0.id == conn.fromID }) else { continue }
            switch fromNode.type {
            case .image, .scene, .character:
                if imageCount < 2 {
                    if let path = inputFilePath(for: fromNode) {
                        imageCount += 1
                        imageCandidates.append((fromNode.position.y, path))
                    } else {
                        result.invalidConnectionIDs.insert(conn.id)   // 空节点
                    }
                } else {
                    result.invalidConnectionIDs.insert(conn.id)       // 图片超限
                }
            case .audio:
                if audioCount < 1 {
                    if let path = inputFilePath(for: fromNode) {
                        audioCount += 1
                        result.audioPath = path
                    } else {
                        result.invalidConnectionIDs.insert(conn.id)   // 空节点
                    }
                } else {
                    result.invalidConnectionIDs.insert(conn.id)       // 音频超限
                }
            case .video:
                if fromNode.tailFrameEnabled {
                    // 开尾帧视频节点视作图片：提取最后一帧为临时 PNG 作为输出图条件
                    if imageCount < 2 {
                        if let path = tailFrameImagePath(for: fromNode) {
                            imageCount += 1
                            imageCandidates.append((fromNode.position.y, path))
                        } else {
                            result.invalidConnectionIDs.insert(conn.id)
                        }
                    } else {
                        result.invalidConnectionIDs.insert(conn.id)   // 图片超限
                    }
                } else {
                    result.invalidConnectionIDs.insert(conn.id)       // 不允许视频输入
                }
            default:
                break   // 文本/角色/场景等：不判不禁
            }
        }
        // 首尾帧规则：有效图片输入按画布上下位置排序，y 小（上方）为首帧，y 大（下方）为尾帧
        result.imagePaths = imageCandidates.sorted { $0.y < $1.y }.map(\.path)
        return result
    }
    
    /// HiDream-O1 规则（图像/角色/场景节点）：
    /// - 图片/角色/场景节点 → 图片条件，最多 5 张（超出的无效）
    /// - 空节点 → 无效线
    /// - 音频/视频/文本节点 → 无效线（图像生成不支持这些输入条件）
    private func hidreamInputValidity(for imageNode: CanvasNode) -> InputValidity {
        var result = InputValidity()
        result.applies = true
        var imageCount = 0
        var imageCandidates: [(y: CGFloat, path: String)] = []   // 有效图片输入：记录画布 y 用于定主体顺序
        for conn in connections where conn.toID == imageNode.id {
            guard let fromNode = nodes.first(where: { $0.id == conn.fromID }) else { continue }
            switch fromNode.type {
            case .image, .character, .scene:
                if imageCount < 5 {
                    if let path = inputFilePath(for: fromNode) {
                        imageCount += 1
                        imageCandidates.append((fromNode.position.y, path))
                    } else {
                        result.invalidConnectionIDs.insert(conn.id)   // 空节点
                    }
                } else {
                    result.invalidConnectionIDs.insert(conn.id)       // 图片超限（最多 5 张）
                }
            default:
                result.invalidConnectionIDs.insert(conn.id)           // 音频/视频/文本：不支持
            }
        }
        // 主体顺序规则（与视频节点 ltx25Distill 同款）：按源节点画布中心 y 排序，y 小（上方）为主体（图0）
        result.imagePaths = imageCandidates.sorted { $0.y < $1.y }.map(\.path)
        return result
    }
    
    /// 开尾帧视频节点 → 输出图条件：提取视频最后一帧为 PNG（写系统临时目录，不占用资产库）。
    /// 缓存键用资产库文件名（唯一），同名覆盖，避免重复发送累积临时文件。
    private func tailFrameImagePath(for videoNode: CanvasNode) -> String? {
        guard let fileName = mediaFileName(for: videoNode) else { return nil }
        let srcURL = assetLibraryURL.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: srcURL.path) else { return nil }
        let asset = AVURLAsset(url: srcURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 2048, height: 2048)
        // 取最后一帧：末尾回退 1/30s，避免请求越界失败
        let tailTime = CMTime(seconds: max(0, asset.duration.seconds - 1.0 / 30.0), preferredTimescale: 600)
        guard let cg = syncFrameImage(from: generator, at: tailTime) else { return nil }
        let rep = NSBitmapImageRep(cgImage: cg)
        guard let png = rep.representation(using: .png, properties: [:]) else { return nil }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("无限画布尾帧", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let stem = (fileName as NSString).deletingPathExtension
        let outURL = dir.appendingPathComponent("\(stem)_tail.png")
        do {
            try png.write(to: outURL)
            return outURL.path
        } catch {
            return nil
        }
    }

    /// 来源节点媒体文件绝对路径（nil = 空节点无内容）
    private func inputFilePath(for node: CanvasNode) -> String? {
        switch node.type {
        case .image, .scene, .character:
            // 图片/角色/场景：纯图片资产 mediaFileName 为 nil，实际文件就是资产库里的 imageFileName
            guard let imgName = node.imageFileName, !imgName.isEmpty else { return nil }
            return assetLibraryURL.appendingPathComponent(imgName).path
        default:
            // 音频/视频等：原始媒体文件在资产库 mediaFileName
            guard let fileName = mediaFileName(for: node) else { return nil }
            return assetLibraryURL.appendingPathComponent(fileName).path
        }
    }
}
