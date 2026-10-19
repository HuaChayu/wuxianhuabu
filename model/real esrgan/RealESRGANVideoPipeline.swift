import Foundation
import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox

/// Real-ESRGAN 视频超分调用管线（独立文件）。
///
/// 流程：AVAssetReader 逐帧读出 BGRA -> 转 RGB float [0,1] -> RealESRGANCoreML 超分
///       -> 转回 BGRA8 -> AVAssetWriter(H.264) + PixelBufferAdaptor 按源时间戳写回。
/// 无第三方依赖，纯 AVFoundation + CoreML。
public final class RealESRGANVideoPipeline {

    private let esrgan: RealESRGANCoreML

    public init(esrgan: RealESRGANCoreML) {
        self.esrgan = esrgan
    }

    /// 对视频逐帧做 scale 倍超分，写出 H.264 mov。
    /// - Parameters:
    ///   - inputURL: 源视频（支持 ProRes / H.264 等任意 AVFoundation 可解格式）
    ///   - outputURL: 输出 mov 路径
    ///   - bitrate: 输出码率（bps），nil 时按分辨率估算
    ///   - progress: 进度回调 (已处理帧数, 估算总帧数)；总数按帧率估算，可能略偏
    ///
    /// 注：元数据读取使用同步旧 API（tracks/naturalSize/nominalFrameRate），
    /// 方便命令行与回调场景直接调用；如需完全避免 deprecation 警告可自行改为 async load。
    public func upscaleVideo(inputURL: URL,
                             outputURL: URL,
                             bitrate: Int? = nil,
                             progress: ((Int, Int) -> Void)? = nil) throws {
        let asset = AVURLAsset(url: inputURL)
        // AVAssetWriter 要求目标文件不存在，先清理旧输出
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw NSError(domain: "RealESRGANVideoPipeline", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "源视频没有视频轨"])
        }

        let srcW = Int(track.naturalSize.width)
        let srcH = Int(track.naturalSize.height)
        let s = esrgan.config.scale
        let outW = srcW * s
        let outH = srcH * s
        let fps = track.nominalFrameRate > 0 ? track.nominalFrameRate : 30.0
        let totalFrames = Int(asset.duration.seconds * Double(fps))

        // ---- Reader：强制解码为 BGRA ----
        let reader = try AVAssetReader(asset: asset)
        let readerSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: readerSettings)
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)
        guard reader.startReading() else {
            throw reader.error ?? NSError(domain: "RealESRGANVideoPipeline", code: 11,
                                          userInfo: [NSLocalizedDescriptionKey: "reader 启动失败"])
        }

        // ---- Writer：H.264 mov + PixelBufferAdaptor(BGRA) ----
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let codecSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outW,
            AVVideoHeightKey: outH,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate ?? estimateBitrate(width: outW, height: outH, fps: Double(fps)),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoAllowFrameReorderingKey: false,
            ],
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: codecSettings)
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outW,
                kCVPixelBufferHeightKey as String: outH,
            ])
        writer.add(writerInput)
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "RealESRGANVideoPipeline", code: 12,
                                          userInfo: [NSLocalizedDescriptionKey: "writer 启动失败"])
        }
        writer.startSession(atSourceTime: .zero)

        // ---- 逐帧处理 ----
        var frameIndex = 0
        while true {
            guard let sample = readerOutput.copyNextSampleBuffer() else { break }
            defer { CMSampleBufferInvalidate(sample) }

            guard let srcPixel = CMSampleBufferGetImageBuffer(sample) else { continue }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample)

            // BGRA -> RGB float
            let (rgb, w, h) = bgraToRgbFloat(pixelBuffer: srcPixel)

            // 超分
            let (outRgb, _, _) = try esrgan.upscale(rgb: rgb, width: w, height: h)

            // RGB float -> BGRA8 pixel buffer
            let outPixel = try makePixelBuffer(width: outW, height: outH)
            rgbFloatToBGRA(rgb: outRgb, width: outW, height: outH, pixelBuffer: outPixel)

            guard adaptor.append(outPixel, withPresentationTime: pts) else {
                throw writer.error ?? NSError(domain: "RealESRGANVideoPipeline", code: 13,
                                              userInfo: [NSLocalizedDescriptionKey: "写入帧失败"])
            }

            frameIndex += 1
            progress?(frameIndex, totalFrames)
        }

        reader.cancelReading()
        writerInput.markAsFinished()
        let sema = DispatchSemaphore(value: 0)
        writer.finishWriting { sema.signal() }
        sema.wait()

        if writer.status != .completed {
            throw writer.error ?? NSError(domain: "RealESRGANVideoPipeline", code: 14,
                                          userInfo: [NSLocalizedDescriptionKey: "视频写出失败"])
        }
    }

    // MARK: - 像素转换

    /// CVPixelBuffer(BGRA8) -> RGB float [0,1]，行优先 HWC。正确处理 bytesPerRow 对齐。
    private func bgraToRgbFloat(pixelBuffer: CVPixelBuffer) -> (rgb: [Float], w: Int, h: Int) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return ([], 0, 0)
        }

        var rgb = [Float](repeating: 0, count: w * h * 3)
        rgb.withUnsafeMutableBytes { dstRaw in
            let dst = dstRaw.bindMemory(to: Float.self)
            let src = base.assumingMemoryBound(to: UInt8.self)
            for row in 0..<h {
                let rowBase = row * bpr
                for col in 0..<w {
                    let b = src[rowBase + col * 4]
                    let g = src[rowBase + col * 4 + 1]
                    let r = src[rowBase + col * 4 + 2]
                    let idx = row * w + col
                    dst[idx * 3] = Float(r) / 255.0
                    dst[idx * 3 + 1] = Float(g) / 255.0
                    dst[idx * 3 + 2] = Float(b) / 255.0
                }
            }
        }
        return (rgb, w, h)
    }

    /// RGB float [0,1] -> CVPixelBuffer(BGRA8)。
    private func rgbFloatToBGRA(rgb: [Float], width: Int, height: Int, pixelBuffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

        rgb.withUnsafeBytes { srcRaw in
            let src = srcRaw.bindMemory(to: Float.self)
            let dst = base.assumingMemoryBound(to: UInt8.self)
            for row in 0..<height {
                let rowBase = row * bpr
                for col in 0..<width {
                    let idx = row * width + col
                    let r = UInt8(min(max(src[idx * 3] * 255, 0), 255))
                    let g = UInt8(min(max(src[idx * 3 + 1] * 255, 0), 255))
                    let b = UInt8(min(max(src[idx * 3 + 2] * 255, 0), 255))
                    dst[rowBase + col * 4] = b
                    dst[rowBase + col * 4 + 1] = g
                    dst[rowBase + col * 4 + 2] = r
                    dst[rowBase + col * 4 + 3] = 255
                }
            }
        }
    }

    private func makePixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let pixel = pb else {
            throw NSError(domain: "RealESRGANVideoPipeline", code: 15,
                          userInfo: [NSLocalizedDescriptionKey: "CVPixelBuffer 创建失败: \(status)"])
        }
        return pixel
    }

    private func estimateBitrate(width: Int, height: Int, fps: Double) -> Int {
        let pixels = Double(width) * Double(height)
        // 超分输出是高画质内容，0.15-0.25 bpp 会导致 H.264 块状马赛克（用户看到的"细小网格"）。
        // 码率 = 像素数 × bpp × 帧率：按 0.6 bpp 估算（2688x1536@24fps 约 59Mbps），下限 60Mbps。
        return max(Int(pixels * 0.6 * fps), 60_000_000)
    }
}
