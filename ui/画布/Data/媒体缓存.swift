// ============================================================
//  文件作用：媒体与资产缓存。mediaDurationCache / mediaDuration 媒体文件时长缓存，
//  assetPixelSize 只读头部取像素尺寸（布局专用，不触发像素解码），
//  AssetThumbnailStore / cachedAssetImage 显示用降采样缩略图（磁盘 + 内存两级缓存，后台并发生成），
//  fullAssetImage 原图（仅少数需要全清晰度的场景，如大纲悬停大图）。
//  互动文件：cachedAssetImage 被 节点 ui.swift 渲染侧、节点 几何.swift 布局侧、
//  节点 交互 鼠标.swift 命中侧、工具栏大纲视图 ui.swift 引用；assetLibraryURL / cacheDirURL 来自根目录。
// ============================================================

import AVFoundation
import AppKit
import Foundation
import ImageIO   // CGImageSource：只读头部取尺寸 / 按目标边长降采样解码
import UniformTypeIdentifiers

// MARK: - 媒体时长缓存

/// 媒体文件时长缓存（按媒体文件名缓存秒数，避免节点渲染反复读盘解析）
private let mediaDurationCache = NSCache<NSString, NSNumber>()

/// 媒体文件时长（秒；仅音频/视频且有实际媒体内容时返回，无内容/读失败返回 nil）
func mediaDuration(for node: CanvasNode) -> TimeInterval? {
    guard node.type == .video || node.type == .audio,
          let fileName = mediaFileName(for: node) else { return nil }
    let fp = assetFileFingerprint(fileName)
    let key = assetCacheKey(fileName, fp) as NSString
    // 缓存约定：> 0 = 真实时长；<= 0 = 已知拿不到时长（文件缺失 / 解析失败），负缓存避免反复读盘。
    // 节点 body 每帧都会问时长，失败不缓存就会不停建 AVAsset，让 CoreMedia 往控制台刷
    // "<<<< FigFile >>>> signalled err=-17913"。
    if let cached = mediaDurationCache.object(forKey: key) {
        let v = cached.doubleValue
        return v > 0 ? v : nil
    }
    let url = assetLibraryURL.appendingPathComponent(fileName)
    guard fp != nil else {
        mediaDurationCache.setObject(NSNumber(value: 0), forKey: key)
        return nil
    }
    let duration: TimeInterval?
    if node.type == .video {
        let secs = AVAsset(url: url).duration.seconds
        duration = secs.isFinite && secs > 0 ? secs : nil
    } else {
        duration = (try? AVAudioPlayer(contentsOf: url))?.duration
    }
    let d = (duration ?? 0)
    mediaDurationCache.setObject(NSNumber(value: d.isFinite ? d : 0), forKey: key)
    return duration
}

// MARK: - 资产像素尺寸（只读头部，布局专用）

/// 像素尺寸缓存（按文件名；CGImageSource 读头部即可拿到宽高，单张 ~0.2ms）
private let assetPixelSizeCache = NSCache<NSString, NSValue>()

/// 资产图片像素尺寸：只解析图片头部，**不解码像素**。
/// 布局侧（节点尺寸/比例推导）一律走这里，绝不能为拿宽高去加载整张原图。
func assetPixelSize(for fileName: String) -> CGSize? {
    let fp = assetFileFingerprint(fileName)
    let key = assetCacheKey(fileName, fp) as NSString
    if let hit = assetPixelSizeCache.object(forKey: key) {
        let size = hit.sizeValue
        return size.width > 0 ? size : nil   // .zero = 已知拿不到（解码失败），负缓存
    }
    let url = assetLibraryURL.appendingPathComponent(fileName)
    // 布局侧是每帧热路径：文件缺失时必须先 fileExists，否则 ImageIO 会逐帧打 "can't open"；
    // 且失败要负缓存，不然下次 body 重算还会再打一遍。
    guard fp != nil else {
        assetPixelSizeCache.setObject(NSValue(size: .zero), forKey: key)
        return nil
    }
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let w = props[kCGImagePropertyPixelWidth] as? Int,
          let h = props[kCGImagePropertyPixelHeight] as? Int,
          w > 0, h > 0 else {
        assetPixelSizeCache.setObject(NSValue(size: .zero), forKey: key)
        return nil
    }
    let size = CGSize(width: CGFloat(w), height: CGFloat(h))
    assetPixelSizeCache.setObject(NSValue(size: size), forKey: key)
    return size
}

// MARK: - 资产文件存在性

/// 资产文件存在性缓存（区分「空节点」与「有图但缩略图未就绪」；避免每次 body 都 stat 磁盘）
private let assetExistsCache = NSCache<NSString, NSNumber>()

/// 资产文件是否真实存在（缺失/被清理的历史引用返回 false）
/// 只缓存「存在」：文件名被重新生成时（删除后又加回同名文件）必须能立刻反映出来。
func assetFileExists(_ fileName: String) -> Bool {
    let key = fileName as NSString
    if let hit = assetExistsCache.object(forKey: key) { return hit.boolValue }
    let exists = FileManager.default.fileExists(atPath: assetLibraryURL.appendingPathComponent(fileName).path)
    if exists { assetExistsCache.setObject(NSNumber(value: true), forKey: key) }
    return exists
}

/// 资产文件指纹：`尺寸_修改时间(秒)`；文件不存在返回 nil。
/// 用途：所有以「文件名」为键的缓存都必须带上它，否则「删掉 1.png → 很久后再加一个同名的 1.png」
/// 会命中旧缓存，画布显示上一张图，而 fullAssetImage（每次读盘）显示新图，同节点两副面孔。
func assetFileFingerprint(_ fileName: String) -> String? {
    let url = assetLibraryURL.appendingPathComponent(fileName)
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) else { return nil }
    let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
    let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
    return "\(size)_\(Int(mtime))"
}

/// 缓存键：文件名 + 指纹（缺失则用 missing）。指纹变化 = 新文件，缓存键自然变化 → 自动失效。
func assetCacheKey(_ fileName: String, _ fingerprint: String?) -> String {
    "\(fileName)|\(fingerprint ?? "missing")"
}

// MARK: - 显示用缩略图（磁盘 + 内存两级缓存）

/// 缩略图长边像素。节点长边 240pt，2x 屏 = 480px，留 1.6x 缩放余量 → 768
let assetThumbPixelSide: Int = 768

/// 缩略图就绪通知：后台生成完一批后发出，渲染侧据此重绘（object 为 nil）
extension Notification.Name {
    static let assetThumbnailDidUpdate = Notification.Name("CanvasAssetThumbnailDidUpdate")
}

/// 资产缩略图仓库：
/// - 读路径（cachedAssetImage）：内存缓存 → 磁盘缩略图（JPEG，解码 ~1ms/张）→ 都没有则返回 nil 并触发后台生成
/// - 写路径（后台）：原图 PNG 解码 → 缩放到 768 → 存 JPEG 落盘 + 内存
/// 为什么不在主线程解码原图：2848×1600 的 PNG 单张解码约 30–45ms，60 张就是 2 秒级主线程阻塞（进入画布转风车）。
final class AssetThumbnailStore: @unchecked Sendable {
    static let shared = AssetThumbnailStore()

    private let memory = NSCache<NSString, NSImage>()
    private let lock = NSLock()
    private var inFlight: Set<String> = []
    /// 已判定不可用的文件名（文件缺失 / 解码失败）：永不重试。
    /// 没有它就会死循环：body 未命中 → 排队 → 失败 → 发通知 → body 重算 → 再排队，
    /// 同时每次 CGImageSource 打开都会往控制台打一行 IIOImageSource 错误。
    private var failed: Set<String> = []
    private let queue: OperationQueue = {
        let q = OperationQueue()
        q.name = "canvas.asset.thumbnail"
        q.maxConcurrentOperationCount = 6
        q.qualityOfService = .userInitiated
        return q
    }()

    private init() {
        memory.countLimit = 400
    }

    // MARK: 路径

    private func cacheDir() -> URL {
        let dir = cacheDirURL.appendingPathComponent("画布缩略图", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    /// 缩略图文件名：文件名 FNV-1a 哈希（避免中文/括号/超长名落盘问题）+ 文件指纹 + 目标边长。
    /// 指纹参与命名，同名不同图不会互相覆盖；指纹一变磁盘缓存自动失效（旧 jpg 在生成新图时清掉）。
    private func diskURL(for fileName: String, fingerprint: String) -> URL {
        let prefix = Self.hashPrefix(for: fileName)
        return cacheDir().appendingPathComponent("\(prefix)_\(fingerprint)@\(assetThumbPixelSide).jpg")
    }

    private static func hashPrefix(for fileName: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in fileName.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
        }
        return String(format: "%016llx", hash)
    }

    /// 说明：这里刻意不做「删掉同前缀旧指纹文件」的清理。
    /// 缩略图目录是共享的，一旦清理逻辑有偏差就会误删其它资产的缓存（实测踩到过），
    /// 而旧指纹文件残留只是占几十 KB，代价远小于误删。需要腾空间时整目录删掉即可自动重建。

    // MARK: 读

    /// 显示用图片：命中内存/磁盘缩略图则立刻返回；否则返回 nil 并排队后台生成（不阻塞主线程）
    func image(for fileName: String) -> NSImage? {
        let fp = assetFileFingerprint(fileName)
        let cacheKey = assetCacheKey(fileName, fp) as NSString
        if let hit = memory.object(forKey: cacheKey) { return hit }
        guard let fp else {
            // 文件不存在（历史引用已被清理）：记入失败集，不排队、不重试
            lock.lock(); failed.insert(cacheKey as String); lock.unlock()
            return nil
        }
        if let img = Self.decodeThumbnail(at: diskURL(for: fileName, fingerprint: fp)) {
            memory.setObject(img, forKey: cacheKey)
            return img
        }
        lock.lock()
        let isFailed = failed.contains(cacheKey as String)
        lock.unlock()
        if isFailed { return nil }
        enqueue([fileName])
        return nil
    }

    /// 原图（仅在确定需要全清晰度时用，例如大纲悬停大图；会走磁盘 + 全量解码）
    /// 带小容量内存缓存：同一张图反复悬停不重复解码
    private let fullImageCache = NSCache<NSString, NSImage>()

    func fullImage(for fileName: String) -> NSImage? {
        let fp = assetFileFingerprint(fileName)
        let key = assetCacheKey(fileName, fp) as NSString
        if let hit = fullImageCache.object(forKey: key) { return hit }
        // 调用方常见写法是 `cachedAssetImage(for:) ?? fullAssetImage(for:)`，缩略图未就绪时会兜底到这里；
        // 文件已缺失时若不先 fileExists，每帧都会让 ImageIO 打一行 "can't open" 到控制台。
        guard fp != nil else { return nil }
        let url = assetLibraryURL.appendingPathComponent(fileName)
        guard let img = NSImage(contentsOf: url) else { return nil }
        fullImageCache.countLimit = 16
        fullImageCache.setObject(img, forKey: key)
        return img
    }

    // MARK: 后台生成

    /// 批量预热：进入画布时调用，把该项目用到的图片全部补齐缩略图（已就绪的跳过）
    func prewarm(_ fileNames: [String]) {
        var todo: [String] = []
        for name in fileNames {
            let fp = assetFileFingerprint(name)
            let cacheKey = assetCacheKey(name, fp) as NSString
            if memory.object(forKey: cacheKey) != nil { continue }
            guard let fp else {   // 资产已被清理的历史引用直接跳过
                lock.lock(); failed.insert(cacheKey as String); lock.unlock()
                continue
            }
            if FileManager.default.fileExists(atPath: diskURL(for: name, fingerprint: fp).path) { continue }
            lock.lock()
            let bad = failed.contains(cacheKey as String)
            lock.unlock()
            if bad { continue }
            todo.append(name)
        }
        enqueue(todo)
    }

    private func enqueue(_ fileNames: [String]) {
        guard !fileNames.isEmpty else { return }
        var accepted: [String] = []
        lock.lock()
        for name in fileNames where !inFlight.contains(name) {
            inFlight.insert(name)
            accepted.append(name)
        }
        lock.unlock()
        guard !accepted.isEmpty else { return }

        // 分批：每 8 张完成刷新一次，图片逐批浮现而不是等全部完成才一次性出现
        let batchSize = 8
        let batches = stride(from: 0, to: accepted.count, by: batchSize).map {
            Array(accepted[$0..<min($0 + batchSize, accepted.count)])
        }
        for (batchIndex, batch) in batches.enumerated() {
            queue.addOperation { [weak self] in
                guard let self else { return }
                var produced = 0
                for name in batch {
                    let (image, fp) = self.generateThumbnail(for: name)
                    let cacheKey = assetCacheKey(name, fp) as NSString
                    if let image {
                        self.memory.setObject(image, forKey: cacheKey)   // 键带指纹：同名新图不会命中旧图
                        produced += 1
                    }
                    self.lock.lock()
                    self.inFlight.remove(name)
                    if image == nil { self.failed.insert(cacheKey as String) }   // 解码失败不再重试
                    self.lock.unlock()
                }
                // 本批确实产出新图才通知重绘，否则纯属无意义刷新
                if produced > 0 {
                    DispatchQueue.main.async {
                        NotificationCenter.default.post(name: .assetThumbnailDidUpdate, object: nil)
                    }
                }
                _ = batchIndex
            }
        }
    }

    /// 原图 → 768 长边 JPEG，写入磁盘并返回解码后的 NSImage（后台线程执行）
    /// - Returns: (缩略图, 生成时该文件的指纹)；失败时返回 (nil, 指纹或 nil)
    private func generateThumbnail(for fileName: String) -> (NSImage?, String?) {
        let fp = assetFileFingerprint(fileName)
        guard let fp else { return (nil, nil) }   // 排队期间文件被删掉了
        let srcURL = assetLibraryURL.appendingPathComponent(fileName)
        guard let source = CGImageSourceCreateWithURL(srcURL as CFURL, nil) else { return (nil, fp) }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: assetThumbPixelSide,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return (nil, fp) }
        let image = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
        let outURL = diskURL(for: fileName, fingerprint: fp)
        if let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.jpeg.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
            CGImageDestinationFinalize(dest)
        }
        return (image, fp)
    }

    /// 从磁盘缩略图解码（JPEG 支持按目标尺寸快速解码，单张 ~1ms）
    /// 先 fileExists 再建 ImageSource：对不存在的路径 CGImageSourceCreateWithURL 会往系统日志打
    /// 一行 "IIOImageSource: can't open / fileExists == false"，每帧每次命中失败都打就是控制台刷屏。
    private static func decodeThumbnail(at url: URL) -> NSImage? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: assetThumbPixelSide,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}

/// 显示用资产图片（画布节点 / 列表格 / 引用缩略图）：走降采样缩略图，不阻塞主线程。
/// 返回 nil 表示暂未就绪（后台正在生成，完成后会发 .assetThumbnailDidUpdate）。
/// 注意：调用方若需要"是否有图"的判断，请看 node.imageFileName，而不是本函数返回值。
func cachedAssetImage(for fileName: String) -> NSImage? {
    AssetThumbnailStore.shared.image(for: fileName)
}

/// 资产原图（全清晰度，会同步解码）。仅用于确需原始分辨率的场景（如大纲悬停大图预览）。
func fullAssetImage(for fileName: String) -> NSImage? {
    AssetThumbnailStore.shared.fullImage(for: fileName)
}

/// 预热一批资产的显示缩略图（进入画布时调用；后台并发，不阻塞 UI）
func prewarmAssetThumbnails(_ fileNames: [String]) {
    AssetThumbnailStore.shared.prewarm(fileNames)
}

// MARK: - 列表行小缩略图缓存

/// 列表行缩略图缓存（键 = 文件名 + 目标像素边长）。
/// 与画布缩略图分开：列表一次要画几十行，用 CGImageSource 直接按目标边长解码，
/// 不把整张原图（可能 4K）解码进内存再交给 SwiftUI 压缩，避免列表首开卡顿与内存膨胀。
private let rowThumbnailCache = NSCache<NSString, NSImage>()

/// 列表行缩略图：按目标像素边长解码并缓存；文件缺失/不可解码返回 nil（调用方退回类型图标）
/// - Parameters:
///   - fileName: 资产库内的文件名（节点 imageFileName，视频节点为首帧缩略图）
///   - pixelSide: 目标边长（像素）；按行内显示尺寸 × 屏幕倍数传值即可
/// 已知不可用的资产文件（原图已被清理）：负缓存，避免每次 body 重算都去打开一次不存在
/// 的文件——CGImageSourceCreateWithURL 失败会往系统日志打 "IIOImageSource: can't open"，
/// 列表几十行 × 每帧重算 = 控制台刷屏。
private var missingAssetFiles: Set<String> = []
private let missingAssetLock = NSLock()

func cachedRowThumbnail(for fileName: String, pixelSide: CGFloat = 60) -> NSImage? {
    let fp = assetFileFingerprint(fileName)
    let key = "\(assetCacheKey(fileName, fp))@\(Int(pixelSide))" as NSString
    if let hit = rowThumbnailCache.object(forKey: key) { return hit }
    guard fp != nil else {
        missingAssetLock.lock(); missingAssetFiles.insert(key as String); missingAssetLock.unlock()
        return nil
    }
    missingAssetLock.lock()
    let knownMissing = missingAssetFiles.contains(key as String)
    missingAssetLock.unlock()
    if knownMissing { return nil }
    let url = assetLibraryURL.appendingPathComponent(fileName)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: Int(pixelSide),
        kCGImageSourceShouldCacheImmediately: true
    ]
    guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
    let image = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    rowThumbnailCache.setObject(image, forKey: key)
    return image
}
