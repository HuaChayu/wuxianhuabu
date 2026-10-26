//
//  媒体信息.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/10/26.
//
// ============================================================
//  文件作用：读取资产文件的媒体专业信息（分辨率、时长、帧率、编码、码率、
//  采样率、声道、色彩空间、位深、文件大小、修改时间），供属性管理面板展示。
//  视频/音频走 AVFoundation（异步 load 属性，避免阻塞主线程）；
//  图片走 ImageIO，直接读取像素尺寸与色彩元数据。
//  互动文件：被 资产管理 ui.swift（属性管理面板）与 媒体快速预览.swift 引用。
// ============================================================

import Foundation
import AVFoundation
import AppKit
import ImageIO
import CoreMedia
import AudioToolbox
import UniformTypeIdentifiers

// ============================================================

// MARK: - 媒体信息数据

/// 一份资产文件的媒体信息快照（字段缺失表示该类型不适用或读取失败）
struct MediaInfo {
    /// 像素尺寸（视频按旋转矫正后的显示尺寸）
    var dimensions: CGSize?
    /// 时长（秒）
    var duration: Double?
    /// 帧率
    var frameRate: Double?
    /// 视频编码（H.264 / HEVC / ProRes ...）
    var videoCodec: String?
    /// 视频码率（bps）
    var videoBitrate: Double?
    /// 色彩空间 / 颜色模型
    var colorSpace: String?
    /// 位深
    var depth: Int?
    /// 音频编码
    var audioCodec: String?
    /// 音频采样率（Hz）
    var sampleRate: Double?
    /// 声道数
    var channels: Int?
    /// 音频码率（bps）
    var audioBitrate: Double?
    /// 文件大小（字节）
    var fileSize: Int64?
    /// 文件修改时间
    var modifiedAt: Date?
    /// 文件类型描述（如 MP4 / PNG）
    var fileType: String?

    /// 是否为空（完全没有读到任何有效字段）
    var isEmpty: Bool {
        dimensions == nil && duration == nil && fileSize == nil
    }
}

// ============================================================

// MARK: - 媒体信息读取

enum MediaInfoLoader {

    /// 资产对应的原始文件 URL（视频/音频优先原始媒体文件，图片用展示图文件）
    static func mediaURL(for asset: AssetItem) -> URL? {
        let name: String?
        switch asset.category {
        case .video, .audio:
            name = (asset.mediaFileName?.isEmpty == false) ? asset.mediaFileName : asset.fileName
        default:
            name = asset.fileName
        }
        guard let name, !name.isEmpty else { return nil }
        return assetLibraryURL.appendingPathComponent(name)
    }

    /// 图片文件像素尺寸（供快速预览算窗口大小）
    static func pixelSize(of url: URL) -> CGSize? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return CGSize(width: w, height: h)
    }

    /// 读取资产媒体信息（异步；任一步失败都只是字段缺失，不抛错）
    static func load(asset: AssetItem) async -> MediaInfo {
        var info = MediaInfo()
        guard let url = mediaURL(for: asset),
              FileManager.default.fileExists(atPath: url.path) else { return info }

        // 通用文件属性
        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path) {
            info.fileSize = (attrs[.size] as? NSNumber)?.int64Value
            info.modifiedAt = attrs[.modificationDate] as? Date
        }
        let ext = url.pathExtension.uppercased()
        info.fileType = ext.isEmpty ? nil : ext

        switch asset.category {
        case .video:
            info = await merge(info, loadVideo(url: url))
        case .audio:
            info = await merge(info, loadAudio(url: url))
        default:
            info = merge(info, loadImage(url: url))
        }
        return info
    }

    /// 用媒体探测结果补全通用字段（媒体结果优先，通用字段缺失时补上）
    private static func merge(_ base: MediaInfo, _ detected: MediaInfo) -> MediaInfo {
        var merged = base
        if let v = detected.dimensions { merged.dimensions = v }
        if let v = detected.duration { merged.duration = v }
        if let v = detected.frameRate { merged.frameRate = v }
        if let v = detected.videoCodec { merged.videoCodec = v }
        if let v = detected.videoBitrate { merged.videoBitrate = v }
        if let v = detected.colorSpace { merged.colorSpace = v }
        if let v = detected.depth { merged.depth = v }
        if let v = detected.audioCodec { merged.audioCodec = v }
        if let v = detected.sampleRate { merged.sampleRate = v }
        if let v = detected.channels { merged.channels = v }
        if let v = detected.audioBitrate { merged.audioBitrate = v }
        if merged.fileSize == nil, let v = detected.fileSize { merged.fileSize = v }
        return merged
    }

    // MARK: 视频

    private static func loadVideo(url: URL) async -> MediaInfo {
        var info = MediaInfo()
        let av = AVURLAsset(url: url)
        if let duration = try? await av.load(.duration), duration.isNumeric {
            info.duration = duration.seconds
        }
        guard let track = (try? await av.loadTracks(withMediaType: .video))?.first else { return info }

        if let natural = try? await track.load(.naturalSize) {
            var size = natural
            if let transform = try? await track.load(.preferredTransform) {
                size = natural.applying(transform)
            }
            info.dimensions = CGSize(width: abs(size.width), height: abs(size.height))
        }
        if let fps = try? await track.load(.nominalFrameRate), fps > 0 {
            info.frameRate = Double(fps)
        }
        if let bitrate = try? await track.load(.estimatedDataRate), bitrate > 0 {
            info.videoBitrate = Double(bitrate)
        }
        if let formats = try? await track.load(.formatDescriptions), let first = formats.first {
            info.videoCodec = videoCodecName(CMFormatDescriptionGetMediaSubType(first))
            let dims = CMVideoFormatDescriptionGetDimensions(first)
            if info.dimensions == nil, dims.width > 0, dims.height > 0 {
                info.dimensions = CGSize(width: Int(dims.width), height: Int(dims.height))
            }
        }
        // 视频常带音轨，一并展示音频参数
        if let audioTrack = (try? await av.loadTracks(withMediaType: .audio))?.first,
           let formats = try? await audioTrack.load(.formatDescriptions),
           let first = formats.first {
            let subType = CMFormatDescriptionGetMediaSubType(first)
            info.audioCodec = audioCodecName(subType)
            if let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(first)?.pointee {
                info.sampleRate = asbd.mSampleRate > 0 ? asbd.mSampleRate : nil
                info.channels = asbd.mChannelsPerFrame > 0 ? Int(asbd.mChannelsPerFrame) : nil
            }
            if let rate = try? await audioTrack.load(.estimatedDataRate), rate > 0 {
                info.audioBitrate = Double(rate)
            }
        }
        return info
    }

    // MARK: 音频

    private static func loadAudio(url: URL) async -> MediaInfo {
        var info = MediaInfo()
        let av = AVURLAsset(url: url)
        if let duration = try? await av.load(.duration), duration.isNumeric {
            info.duration = duration.seconds
        }
        guard let track = (try? await av.loadTracks(withMediaType: .audio))?.first else { return info }
        if let formats = try? await track.load(.formatDescriptions), let first = formats.first {
            info.audioCodec = audioCodecName(CMFormatDescriptionGetMediaSubType(first))
            if let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(first)?.pointee {
                info.sampleRate = asbd.mSampleRate > 0 ? asbd.mSampleRate : nil
                info.channels = asbd.mChannelsPerFrame > 0 ? Int(asbd.mChannelsPerFrame) : nil
                if asbd.mBitsPerChannel > 0 { info.depth = Int(asbd.mBitsPerChannel) }
            }
        }
        if let rate = try? await track.load(.estimatedDataRate), rate > 0 {
            info.audioBitrate = Double(rate)
        }
        return info
    }

    // MARK: 图片

    private static func loadImage(url: URL) -> MediaInfo {
        var info = MediaInfo()
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] else { return info }
        let w = props[kCGImagePropertyPixelWidth] as? Int
        let h = props[kCGImagePropertyPixelHeight] as? Int
        if let w, let h, w > 0, h > 0 {
            info.dimensions = CGSize(width: w, height: h)
        }
        if let depth = props[kCGImagePropertyDepth] as? Int {
            info.depth = depth
        }
        if let model = props[kCGImagePropertyColorModel] as? String {
            info.colorSpace = model
        }
        if let profile = props[kCGImagePropertyProfileName] as? String, !profile.isEmpty {
            info.colorSpace = info.colorSpace.map { "\($0) · \(profile)" } ?? profile
        }
        return info
    }

    // MARK: 编码名映射

    private static func videoCodecName(_ code: FourCharCode) -> String? {
        switch code {
        case kCMVideoCodecType_H264: return "H.264"
        case kCMVideoCodecType_HEVC: return "HEVC"
        case kCMVideoCodecType_AppleProRes422: return "ProRes 422"
        case kCMVideoCodecType_AppleProRes422HQ: return "ProRes 422 HQ"
        case kCMVideoCodecType_AppleProRes422LT: return "ProRes 422 LT"
        case kCMVideoCodecType_AppleProRes422Proxy: return "ProRes 422 Proxy"
        case kCMVideoCodecType_AppleProRes4444: return "ProRes 4444"
        case kCMVideoCodecType_AppleProRes4444XQ: return "ProRes 4444 XQ"
        default:
            let name = fourCC(code)
            return name.isEmpty ? nil : name
        }
    }

    private static func audioCodecName(_ code: FourCharCode) -> String? {
        switch code {
        case kAudioFormatMPEG4AAC: return "AAC"
        case kAudioFormatAppleLossless: return "ALAC"
        case kAudioFormatLinearPCM: return "PCM"
        case kAudioFormatMPEGLayer3: return "MP3"
        case kAudioFormatFLAC: return "FLAC"
        case kAudioFormatOpus: return "Opus"
        default:
            let name = fourCC(code)
            return name.isEmpty ? nil : name
        }
    }

    private static func fourCC(_ code: FourCharCode) -> String {
        let bytes: [UInt8] = [
            UInt8((code >> 24) & 0xFF),
            UInt8((code >> 16) & 0xFF),
            UInt8((code >> 8) & 0xFF),
            UInt8(code & 0xFF)
        ]
        guard let s = String(bytes: bytes, encoding: .ascii) else { return "" }
        return s.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
    }
}

// ============================================================

// MARK: - 展示格式化

enum MediaInfoFormat {

    /// 时长：mm:ss / h:mm:ss
    static func duration(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "—" }
        let total = Int(seconds.rounded())
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        if h > 0 { return String(format: "%d:%02d:%02d", h, m, s) }
        return String(format: "%02d:%02d", m, s)
    }

    /// 像素尺寸
    static func dimensions(_ size: CGSize?) -> String? {
        guard let size, size.width >= 1, size.height >= 1 else { return nil }
        return "\(Int(size.width.rounded())) × \(Int(size.height.rounded()))"
    }

    /// 码率：Mbps / Kbps
    static func bitrate(_ bps: Double?) -> String? {
        guard let bps, bps > 0 else { return nil }
        if bps >= 1_000_000 { return String(format: "%.1f Mbps", bps / 1_000_000) }
        return String(format: "%.0f Kbps", bps / 1_000)
    }

    /// 采样率：44.1 kHz / 48 kHz
    static func sampleRate(_ hz: Double?) -> String? {
        guard let hz, hz > 0 else { return nil }
        return String(format: "%.1f kHz", hz / 1_000)
    }

    /// 声道：单声道 / 立体声 / N 声道
    static func channels(_ count: Int?) -> String? {
        guard let count, count > 0 else { return nil }
        switch count {
        case 1: return "单声道"
        case 2: return "立体声"
        default: return "\(count) 声道"
        }
    }

    /// 文件大小
    static func fileSize(_ bytes: Int64?) -> String? {
        guard let bytes, bytes > 0 else { return nil }
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowedUnits = [.useKB, .useMB, .useGB]
        return f.string(fromByteCount: bytes)
    }

    /// 修改时间
    static func date(_ date: Date?) -> String? {
        guard let date else { return nil }
        let f = DateFormatter()
        f.locale = Locale(identifier: "zh_CN")
        f.dateFormat = "yyyy/MM/dd HH:mm"
        return f.string(from: date)
    }

    /// 帧率：30 fps / 29.97 fps
    static func frameRate(_ fps: Double?) -> String? {
        guard let fps, fps > 0 else { return nil }
        let rounded = fps.rounded()
        if abs(fps - rounded) < 0.01 { return String(format: "%.0f fps", fps) }
        return String(format: "%.2f fps", fps)
    }
}
