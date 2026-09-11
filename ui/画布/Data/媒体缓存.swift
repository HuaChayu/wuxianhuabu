// ============================================================
//  文件作用：媒体与资产缓存。mediaDurationCache / mediaDuration 媒体文件时长缓存，nodeImageCache / cachedAssetImage 资产图片内存缓存。
//  互动文件：mediaDuration / cachedAssetImage 被 节点 ui.swift 渲染侧与 节点 交互 鼠标.swift 命中侧引用；assetLibraryURL 来自根目录。
// ============================================================

import AVFoundation
import AppKit
import Foundation

// MARK: - 媒体时长缓存

/// 媒体文件时长缓存（按媒体文件名缓存秒数，避免节点渲染反复读盘解析）
private let mediaDurationCache = NSCache<NSString, NSNumber>()

/// 媒体文件时长（秒；仅音频/视频且有实际媒体内容时返回，无内容/读失败返回 nil）
func mediaDuration(for node: CanvasNode) -> TimeInterval? {
    guard node.type == .video || node.type == .audio,
          let fileName = mediaFileName(for: node) else { return nil }
    let key = fileName as NSString
    if let cached = mediaDurationCache.object(forKey: key) { return cached.doubleValue }
    let url = assetLibraryURL.appendingPathComponent(fileName)
    let duration: TimeInterval?
    if node.type == .video {
        let secs = AVAsset(url: url).duration.seconds
        duration = secs.isFinite && secs > 0 ? secs : nil
    } else {
        duration = (try? AVAudioPlayer(contentsOf: url))?.duration
    }
    if let d = duration {
        mediaDurationCache.setObject(NSNumber(value: d), forKey: key)
        return d
    }
    return nil
}

// MARK: - 资产图片缓存

/// 资产图片内存缓存（按文件名缓存解码后的 NSImage，避免移动画布等重绘场景反复从磁盘加载解码）
private let nodeImageCache = NSCache<NSString, NSImage>()

/// 从缓存加载资产图片：命中直接返回，未命中读盘解码后写入缓存。
/// 渲染侧 nodeImage 与占位高度计算共用，避免每次 body 重算都走磁盘 I/O。
func cachedAssetImage(for fileName: String) -> NSImage? {
    let key = fileName as NSString
    if let img = nodeImageCache.object(forKey: key) { return img }
    guard let img = NSImage(contentsOf: assetLibraryURL.appendingPathComponent(fileName)) else { return nil }
    nodeImageCache.setObject(img, forKey: key)
    return img
}
