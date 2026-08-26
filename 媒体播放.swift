// ============================================================
//  文件作用：媒体播放管理。音频/视频节点与资产（资产库格子、属性管理预览）的
//  播放/停止统一入口，一次性播放（播完自动回到未播放态），同一时刻只播放一个目标。
//  互动文件：引用 节点ui.swift（CanvasNode、NodeType、playButtonSize）、
//  读取&保存.swift（assetLibraryURL）、资产管理 ui.swift（AssetItem）；
//  被 节点ui.swift（NodeView 播放按钮）、资产管理 ui.swift（资产格子/属性预览）引用。
// ============================================================

import SwiftUI
import AppKit
import Combine
import AVFoundation

// MARK: - 媒体播放管理器（全局单例）

/// 播放目标类型（节点 / 资产库资产），用于区域互斥停止：离开资产库只停资产、离开画布只停节点
enum MediaPlayTargetKind {
    case node
    case asset
}

/// 统一获取节点的媒体文件名（磁盘资产库中的原始媒体文件名）：
/// 优先用节点自带 mediaFileName（已直接关联）；否则通过节点 imageFileName
/// 反查资产库对应资产拿 mediaFileName。这样任何添加入口只要正常写了
/// imageFileName（资产库展示图），播放按钮与播放逻辑都能工作，无需入口单独维护。
func mediaFileName(for node: CanvasNode) -> String? {
    if let m = node.mediaFileName, !m.isEmpty { return m }
    guard let imgName = node.imageFileName, !imgName.isEmpty else { return nil }
    return AssetStore.shared.assets.first { $0.fileName == imgName }?.mediaFileName
}

/// 音频/视频节点与资产播放管理：同一时刻只播放一个目标，播完自动停止（一次性播放）。
/// 节点视图与资产库格子/属性预览通过 @ObservedObject 观察 playingNodeID / isPlaying
/// 刷新播放按钮状态。
final class MediaPlayerManager: NSObject, ObservableObject {
    static let shared = MediaPlayerManager()

    /// 当前播放目标的 id（节点 CanvasNode.id 或资产 AssetItem.id；nil = 无播放）
    @Published var playingNodeID: UUID?
    /// 是否正在播放
    @Published var isPlaying = false
    /// 当前播放目标类型（nil = 无播放）。用于区域互斥保护：离开资产库只停资产、离开画布只停节点
    @Published private(set) var playingTargetKind: MediaPlayTargetKind?

    /// 当前视频播放器（视频节点播放中供画面渲染使用；nil = 无视频播放）
    var videoPlayer: AVPlayer?

    private var audioPlayer: AVAudioPlayer?
    private var videoEndObserver: NSObjectProtocol?
    /// 播放进度/频谱数据（独立 ObservableObject：每 0.1s 高频刷新，仅进度条/频谱视图观察，避免全画布重算）
    let progressModel = MediaProgressModel.shared
    /// 播放进度定时器（0.1s 刷新 currentTime/duration）
    private var progressTimer: Timer?

    private override init() {
        super.init()
    }

    /// 指定节点是否正在播放
    func isPlaying(nodeID: UUID) -> Bool {
        playingNodeID == nodeID && isPlaying
    }

    /// 指定资产是否正在播放（资产库格子 / 属性管理预览共用）
    func isPlaying(assetID: UUID) -> Bool {
        playingNodeID == assetID && isPlaying
    }

    /// 切换播放/停止：正在播放则停止，否则开始播放
    func toggle(node: CanvasNode) {
        togglePlayback(isPlaying: isPlaying(nodeID: node.id)) {
            play(mediaFileName: mediaFileName(for: node), isVideo: node.type == .video, targetID: node.id, kind: .node)
        }
    }

    /// 切换播放/停止：正在播放则停止，否则开始播放（资产库资产）
    func toggle(asset: AssetItem) {
        togglePlayback(isPlaying: isPlaying(assetID: asset.id)) {
            play(mediaFileName: asset.mediaFileName, isVideo: asset.category == .video, targetID: asset.id, kind: .asset)
        }
    }

    /// 统一切换逻辑：正在播放则停止，否则执行开始播放动作（节点与资产共用）
    private func togglePlayback(isPlaying: Bool, playAction: () -> Void) {
        if isPlaying {
            stop()
        } else {
            playAction()
        }
    }

    /// 开始播放媒体内容（音频/视频；无媒体文件则忽略）——节点与资产共用
    private func play(mediaFileName: String?, isVideo: Bool, targetID: UUID, kind: MediaPlayTargetKind) {
        guard let mediaFileName, !mediaFileName.isEmpty else {
            debugLog("媒体播放：无媒体文件")
            return
        }
        startPlayback(mediaFileName: mediaFileName, isVideo: isVideo, targetID: targetID, kind: kind)
    }

    /// 统一播放核心：按媒体文件名启动音频/视频播放（节点与资产共用），
    /// targetID 为播放目标标识（CanvasNode.id 或 AssetItem.id），同一时刻只播放一个目标。
    private func startPlayback(mediaFileName: String, isVideo: Bool, targetID: UUID, kind: MediaPlayTargetKind) {
        stop()
        let url = assetLibraryURL.appendingPathComponent(mediaFileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            debugLog("媒体播放：文件不存在 \(mediaFileName)")
            return
        }
        playingNodeID = targetID
        playingTargetKind = kind
        if isVideo {
            let player = AVPlayer(url: url)
            videoPlayer = player
            player.play()
            isPlaying = true
            // 初始时长：AVPlayerItem.duration 播放前可能为 indefinite，先取 asset.duration，后续由定时器校正
            progressModel.duration = player.currentItem?.asset.duration.seconds ?? 0
            progressModel.currentTime = 0
            // 播放结束自动停止（一次性播放）
            videoEndObserver = NotificationCenter.default.addObserver(
                forName: .AVPlayerItemDidPlayToEndTime,
                object: player.currentItem,
                queue: .main
            ) { [weak self] _ in
                self?.stop()
            }
            debugLog("媒体播放：视频开始 \(mediaFileName)")
        } else {
            audioPlayer = try? AVAudioPlayer(contentsOf: url)
            audioPlayer?.delegate = self
            audioPlayer?.play()
            isPlaying = true
            progressModel.duration = audioPlayer?.duration ?? 0
            progressModel.currentTime = 0
            // 后台读取音频 PCM 样本生成真波形（峰值降采样；异步完成，播放不受阻）
            progressModel.waveform = []
            generateWaveform(url: url) { [weak self] peaks in
                guard let self, self.playingNodeID == targetID, self.isPlaying else { return }
                self.progressModel.waveform = peaks
            }
            debugLog("媒体播放：音频开始 \(mediaFileName)")
        }
        startProgressTimer()
    }

    // MARK: - 播放进度（进度条 / 频谱）

    /// 启动进度定时器：0.1s 刷新一次进度与频谱数据（仅播放期间）
    private func startProgressTimer() {
        progressTimer?.invalidate()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.updateProgress()
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    /// 定时刷新当前进度与音频频谱数据
    private func updateProgress() {
        if let player = videoPlayer {
            let d = player.currentItem?.duration.seconds ?? 0
            if d.isFinite, d > 0 { progressModel.duration = d }
            progressModel.currentTime = player.currentTime().seconds
        } else if let player = audioPlayer {
            progressModel.duration = player.duration
            progressModel.currentTime = player.currentTime
        } else {
            // 播放器已释放（异常路径）：停止并清理
            stop()
        }
    }

    /// 按比例跳转播放进度（进度条拖动；仅对当前播放目标生效）
    func seek(nodeID: UUID, ratio: Double) {
        guard playingNodeID == nodeID, isPlaying else { return }
        let clamped = min(max(ratio, 0), 1)
        if let player = videoPlayer {
            let duration = player.currentItem?.duration.seconds ?? 0
            guard duration.isFinite, duration > 0 else { return }
            let target = duration * clamped
            // 拖动期间允许较大容差，降低高频 seek 成本
            player.seek(to: CMTime(seconds: target, preferredTimescale: 600),
                        toleranceBefore: CMTime(seconds: 0.5, preferredTimescale: 600),
                        toleranceAfter: CMTime(seconds: 0.5, preferredTimescale: 600))
            progressModel.currentTime = target
        } else if let player = audioPlayer {
            player.currentTime = player.duration * clamped
            progressModel.currentTime = player.currentTime
        }
    }

    /// 停止当前播放（音频/视频一并停止，回到未播放态）
    func stop() {
        progressTimer?.invalidate()
        progressTimer = nil
        audioPlayer?.stop()
        audioPlayer = nil
        videoPlayer?.pause()
        videoPlayer = nil
        if let obs = videoEndObserver {
            NotificationCenter.default.removeObserver(obs)
            videoEndObserver = nil
        }
        if playingNodeID != nil || isPlaying {
            debugLog("媒体播放：停止")
        }
        playingNodeID = nil
        playingTargetKind = nil
        isPlaying = false
        // 进度/波形数据归零（进度条隐藏、波形视图停动）
        progressModel.currentTime = 0
        progressModel.duration = 0
        progressModel.waveform = []
    }

    /// 区域互斥保护：停止资产库的播放（仅当当前在播的是资产时生效）。
    /// 调用时机：离开总资产库、进入画布等资产库不可见场景。
    func stopAssetPlayback() {
        guard playingTargetKind == .asset else { return }
        debugLog("媒体播放：资产库区域互斥停止")
        stop()
    }

    /// 区域互斥保护：停止画布节点的播放（仅当当前在播的是节点时生效）。
    /// 调用时机：退出/离开画布等画布不可见场景。
    func stopNodePlayback() {
        guard playingTargetKind == .node else { return }
        debugLog("媒体播放：画布区域互斥停止")
        stop()
    }
}

// MARK: - 音频播放结束回调（一次性播放）

extension MediaPlayerManager: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stop()
    }
}

// MARK: - 视频画面渲染（AVPlayerLayer 包裹，供视频节点播放中显示画面）

/// 视频节点播放中显示视频画面的 NSView 容器
final class PlayerNSView: NSView {
    let playerLayer = AVPlayerLayer()

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        playerLayer.videoGravity = .resizeAspect
        layer?.addSublayer(playerLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        // 禁止隐式动画：CALayer 的 frame 变化默认带约 0.25s 的隐式过渡动画，
        // 画布 Command+滚轮缩放时 zoom 是即时跳变、节点本体由 drawingGroup 位图即时缩放，
        // 若此处让视频画面做 0.25s 动画过渡，会形成画面滞后/先缩小再放大的卡顿感。
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        playerLayer.frame = bounds
        CATransaction.commit()
    }
}

/// SwiftUI 包裹：把 AVPlayer 画面渲染到指定区域
struct VideoPlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> PlayerNSView {
        let view = PlayerNSView()
        view.playerLayer.player = player
        return view
    }

    func updateNSView(_ nsView: PlayerNSView, context: Context) {
        nsView.playerLayer.player = player
    }
}

// MARK: - 通用播放/停止按钮（节点、资产库格子、属性管理预览共用）

/// 玻璃质感播放/停止按钮：悬停显示、点击切换播放/停止。
/// 出现/隐藏由调用方按 isHovered 控制（.opacity / .allowsHitTesting），此处仅渲染按钮本身。
struct MediaPlayButton: View {
    let isPlaying: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                .font(.system(size: 18, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: playButtonSize, height: playButtonSize)
                .background(.ultraThinMaterial, in: Circle())
                .overlay(Circle().stroke(Color.white.opacity(0.55), lineWidth: 1.2))
                .shadow(color: .black.opacity(0.35), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - 播放进度数据（进度条 / 音频频谱）

/// 播放进度与频谱数据：独立 ObservableObject，每 0.1s 由 MediaPlayerManager 定时刷新。
/// 仅进度条/频谱视图观察本对象（高频刷新），画布与节点视图只观察 Manager 的播放状态（低频），互不干扰。
final class MediaProgressModel: ObservableObject {
    static let shared = MediaProgressModel()
    /// 当前播放进度（秒）
    @Published var currentTime: TimeInterval = 0
    /// 当前播放总时长（秒）
    @Published var duration: TimeInterval = 0
    /// 当前音频的波形峰值采样（0...1 归一化，按时间顺序；生成于播放开始时，音频播放中常显波形视图用）
    @Published var waveform: [Float] = []
    private init() {}
}

// MARK: - 播放进度条几何（渲染侧与命中侧共用）

/// 播放进度条区域（相对节点中心，逻辑尺寸；NodeView 渲染底部 overlay 与命中侧屏幕坐标推算共用）。
/// 布局约定：占位区底部水平条，左右 padding 10、底部 padding 10、条高 5（与渲染侧 padding 完全一致）。
func progressBarRect(for size: CGSize, type: NodeType, imageFileName: String?) -> CGRect {
    let topLabelH: CGFloat = type.topLabel.isEmpty ? 0 : nodeTopLabelHeight
    let ph = placeholderHeightForType(type, imageFileName: imageFileName)
    // 宽度与渲染侧同源：视频有图 = 长边统一算法（横屏 432 宽），其余走映射表
    let width: CGFloat = (type == .video && imageFileName != nil)
        ? videoNodeSize(imageFileName: imageFileName).width
        : nodeWidthForType(type)
    let barHeight: CGFloat = 5
    let bottomPadding: CGFloat = 10
    let horizontalPadding: CGFloat = 10
    let barMaxY = topLabelH + ph - bottomPadding - size.height / 2   // 进度条底边相对节点中心
    return CGRect(x: -width / 2 + horizontalPadding,
                  y: barMaxY - barHeight,
                  width: width - horizontalPadding * 2,
                  height: barHeight)
}

// MARK: - 播放进度条（节点占位区底部；拖动由鼠标控制层命中处理）

/// 播放进度条：已播放部分主题色填充。
/// 画布节点：仅渲染外观，拖动由 MouseControlNSView 统一拦截（见 画布 交互 鼠标.swift hitTestProgressBar）；
/// 资产库/属性预览：传 onSeek 后启用 SwiftUI DragGesture 拖动（无鼠标控制层）。
/// 视觉高度即命中高度，不额外扩空白区。
struct MediaProgressBarView: View {
    @ObservedObject private var progress = MediaProgressModel.shared
    let accentColor: Color
    /// 非 nil 时启用拖动，回调参数为 0~1 播放比例（画布节点保持 nil）
    var onSeek: ((Double) -> Void)? = nil
    /// 视觉条高（资产格子默认 7，预览区可传更高）
    var barHeight: CGFloat = 7

    private var ratio: CGFloat {
        guard progress.duration > 0 else { return 0 }
        return CGFloat(min(1, max(0, progress.currentTime / progress.duration)))
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.black.opacity(0.4))
                Capsule()
                    .fill(accentColor)
                    .frame(width: max(0, geo.size.width * ratio))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard let onSeek, geo.size.width > 0 else { return }
                        onSeek(min(1, max(0, Double(value.location.x / geo.size.width))))
                    }
            )
        }
        .frame(height: barHeight)
        .shadow(color: .black.opacity(0.3), radius: 1.5, y: 0.5)
    }
}

/// 播放中底部进度条容器（资产库格子 / 属性预览共用）：
/// 统一"贴底部 + 底部/水平留白 + 悬停显隐 + 悬停才可命中"，
/// 避免各处重复写布局与 padding；进度条高度按场景传入。
struct MediaProgressBarOverlay: View {
    let accentColor: Color
    /// 是否可见（悬停时才显示且可命中）
    let isVisible: Bool
    /// 拖动回调（0~1 播放比例）
    var onSeek: ((Double) -> Void)? = nil
    /// 进度条视觉高（资产格子默认，预览区传更高）
    var barHeight: CGFloat = 7

    private let horizontalPadding: CGFloat = 12
    private let bottomPadding: CGFloat = 12

    var body: some View {
        VStack {
            Spacer()
            MediaProgressBarView(accentColor: accentColor, onSeek: onSeek, barHeight: barHeight)
                .opacity(isVisible ? 1 : 0)
                .allowsHitTesting(isVisible)
                .padding(.horizontal, horizontalPadding)
                .padding(.bottom, bottomPadding)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - 音频波形（真波形，非音量假动效）

/// 音频播放中的真波形视图：显示当前音频文件的实际 PCM 波形（峰值降采样），
/// 波形窗口随播放进度滚动（当前播放位置居中，窗口默认 2 秒），
/// 播放开始时由 MediaPlayerManager 后台读取样本生成 progressModel.waveform。
struct WaveformView: View {
    @ObservedObject private var progress = MediaProgressModel.shared
    let accentColor: Color
    /// 可见波形窗口时长（秒）：当前播放位置居中
    private let windowSeconds: Double = 2.0

    var body: some View {
        Canvas { context, size in
            guard progress.duration > 0, !progress.waveform.isEmpty, size.width > 1 else { return }
            let samples = progress.waveform
            let total = samples.count
            let windowPoints = max(2, Int(Double(total) * windowSeconds / progress.duration))
            guard windowPoints <= total else { return }
            // 窗口中心对齐当前播放位置（开头/结尾时夹紧到边界）
            let centerRatio = CGFloat(progress.currentTime / progress.duration)
            let start = min(max(0, Int(CGFloat(total) * centerRatio) - windowPoints / 2), total - windowPoints)
            let end = start + windowPoints
            let midY = size.height / 2
            let halfH = max(1, size.height / 2 - 1)
            let stepX = size.width / CGFloat(windowPoints)
            var path = Path()
            for i in start..<end {
                let x = CGFloat(i - start) * stepX
                let v = CGFloat(min(max(samples[i], 0), 1)) * halfH
                path.move(to: CGPoint(x: x, y: midY - v))
                path.addLine(to: CGPoint(x: x, y: midY + v))
            }
            context.stroke(path, with: .color(accentColor.opacity(0.9)), lineWidth: 1.2)
        }
    }
}

/// 后台读取音频文件的 PCM 样本，生成 0...1 归一化的峰值波形（降采样到约 1200 点）。
/// 完整遍历一次文件，普通歌曲毫秒级完成；在主线程回调返回采样数组。
private func generateWaveform(url: URL, completion: @escaping ([Float]) -> Void) {
    DispatchQueue.global(qos: .userInitiated).async {
        // 统一以 float32 非交织格式读取（AAC/MP3 等解码后处理格式通常已是 LPCM，此处显式指定更稳）
        guard let file = try? AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false) else {
            DispatchQueue.main.async { completion([]) }
            return
        }
        let format = file.processingFormat
        let totalFrames = file.length
        guard totalFrames > 0, format.channelCount > 0 else {
            DispatchQueue.main.async { completion([]) }
            return
        }
        let targetPoints = 1200
        let framesPerPoint = max(1, Int(Double(totalFrames) / Double(targetPoints)))
        // 每批最多读 framesPerPoint*4096 帧（约 4K 个波形点），避免超大 buffer 占用
        let batchFrames = Int64(framesPerPoint) * 4096
        let batchCapacity = AVAudioFrameCount(max(1, min(totalFrames, batchFrames)))
        var peaks: [Float] = []
        peaks.reserveCapacity(targetPoints)
        var frameOffset: AVAudioFramePosition = 0
        while frameOffset < totalFrames && peaks.count < targetPoints {
            let remaining = totalFrames - frameOffset
            let frameCount = AVAudioFrameCount(min(remaining, Int64(batchCapacity)))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { break }
            do { try file.read(into: buffer, frameCount: frameCount) } catch { break }
            guard let channel = buffer.floatChannelData?[0] else { break }
            let frames = Int(buffer.frameLength)
            var i = 0
            while i < frames && peaks.count < targetPoints {
                let end = min(i + framesPerPoint, frames)
                var peak: Float = 0
                for j in i..<end {
                    let v = abs(channel[j])
                    if v > peak { peak = v }
                }
                peaks.append(peak)
                i = end
            }
            frameOffset += Int64(frames)
        }
        DispatchQueue.main.async { completion(peaks) }
    }
}
