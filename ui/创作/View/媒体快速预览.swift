//
//  媒体快速预览.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/10/26.
//
// ============================================================
//  文件作用：模仿 macOS「快速查看（Quick Look）」的空格预览。
//  选中单个资产（图片 / 视频 / 音频）按空格 → 弹出独立无边框窗口：
//    · 窗口尺寸 = 媒体的原始水平尺寸（超屏时等比缩到屏幕可用区域内）
//    · 图片按原图居中显示；视频自动播放；音频显示封面并播放
//    · 再按空格（或 Esc）→ 停止播放并关闭窗口
//  窗口为独立 NSWindow（无标题栏、圆角、阴影、可拖动），与系统预览一致的交互形态。
//  互动文件：被 资产管理 ui.swift 引用（AssetQuickLookKeyHandler / MediaQuickLookController）。
// ============================================================

import SwiftUI
import AppKit
import AVFoundation
import Combine

// ============================================================

extension Notification.Name {
    /// 快速预览窗口内的方向键导航（userInfo: ["keyCode": Int]，123 左 / 124 右 / 125 下 / 126 上）
    static let assetQuickLookNavigate = Notification.Name("assetQuickLookNavigate")
}

// MARK: - 快速预览控制器

/// 空格快速预览的唯一入口：负责窗口创建 / 媒体播放 / 关闭清场。
@MainActor
final class MediaQuickLookController: ObservableObject {
    static let shared = MediaQuickLookController()

    /// 正在预览的资产 id（nil = 预览未打开）
    @Published private(set) var presentedID: UUID?
    /// 音频预览进度 0...1
    @Published private(set) var audioProgress: Double = 0
    /// 音频预览时长（秒）
    @Published private(set) var audioDuration: Double = 0

    private var window: QuickLookWindow?
    private var videoPlayer: AVPlayer?
    private var audioPlayer: AVAudioPlayer?
    private var audioTimer: Timer?
    /// 本次展示令牌：异步探测尺寸期间若已关闭 / 切换目标，旧任务自动作废
    private var presentationToken: UUID?

    private init() {}

    var isPresented: Bool { window != nil }

    /// 空格：已打开则关闭，未打开则预览该资产
    func toggle(asset: AssetItem) {
        if isPresented {
            close()
        } else {
            present(asset)
        }
    }

    /// 预览中切换目标（方向键 / 点选）：窗口已打开则原位换内容继续播放，未打开则新建
    func preview(asset: AssetItem) {
        present(asset, keepPlacement: isPresented)
    }

    // MARK: 打开

    private func present(_ asset: AssetItem, keepPlacement: Bool = false) {
        let token = UUID()
        presentationToken = token
        // 切换场景记住窗口位置，避免每次换内容都弹回屏幕正中
        let previousOrigin = keepPlacement ? window?.frame.origin : nil
        Task { @MainActor in
            guard let url = MediaInfoLoader.mediaURL(for: asset),
                  FileManager.default.fileExists(atPath: url.path) else {
                debugLog("快速预览：文件不存在，无法预览「\(asset.name)」")
                NSSound.beep()
                if self.presentationToken == token { self.presentationToken = nil }
                return
            }
            let pixelSize = await self.contentPixelSize(asset: asset, url: url)
            // 探测期间用户已关闭 / 又按了别的资产：放弃本次展示
            guard self.presentationToken == token else { return }
            self.makeWindow(asset: asset, url: url, pixelSize: pixelSize, reuseOrigin: previousOrigin)
            self.presentationToken = nil
        }
    }

    /// 媒体原始像素尺寸（视频按 preferredTransform 矫正；音频用固定版式）
    private func contentPixelSize(asset: AssetItem, url: URL) async -> CGSize {
        switch asset.category {
        case .video:
            let av = AVURLAsset(url: url)
            guard let track = (try? await av.loadTracks(withMediaType: .video))?.first,
                  let natural = try? await track.load(.naturalSize),
                  natural.width > 0, natural.height > 0 else {
                return CGSize(width: 960, height: 540)
            }
            var size = natural
            if let transform = try? await track.load(.preferredTransform) {
                size = natural.applying(transform)
            }
            return CGSize(width: abs(size.width), height: abs(size.height))
        case .audio:
            return CGSize(width: 440, height: 340)
        default:
            if let size = MediaInfoLoader.pixelSize(of: url), size.width > 0, size.height > 0 {
                return size
            }
            return asset.image.size.width > 0 ? asset.image.size : CGSize(width: 800, height: 600)
        }
    }

    /// 按屏幕可用区域等比适配（1:1 还原媒体水平尺寸，超屏才缩小）
    private func fittedSize(_ pixelSize: CGSize) -> CGSize {
        let screen = NSApp.keyWindow?.screen ?? NSScreen.main
        let visible = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let maxW = max(320, visible.width - 120)
        let maxH = max(220, visible.height - 120)
        let scale = min(1, maxW / pixelSize.width, maxH / pixelSize.height)
        return CGSize(width: (pixelSize.width * scale).rounded(),
                      height: (pixelSize.height * scale).rounded())
    }

    private func makeWindow(asset: AssetItem, url: URL, pixelSize: CGSize, reuseOrigin: CGPoint? = nil) {
        // 只停旧播放器，不销毁旧窗口：新窗口上屏后再关旧窗，切换过程不出现空档闪烁
        stopPlayback()

        let isVideo = asset.category == .video
        let isAudio = asset.category == .audio
        var previewImage: NSImage?

        if isVideo {
            let player = AVPlayer(url: url)
            videoPlayer = player
            player.play()
        } else if isAudio {
            let player = try? AVAudioPlayer(contentsOf: url)
            audioPlayer = player
            audioDuration = player?.duration ?? 0
            audioProgress = 0
            player?.play()
            if player != nil { startAudioTimer() }
            previewImage = asset.image
        } else {
            previewImage = NSImage(contentsOf: url) ?? asset.image
        }

        let size = fittedSize(pixelSize)
        let hosting = NSHostingView(rootView: MediaQuickLookContentView(
            assetName: asset.name,
            previewImage: previewImage,
            videoPlayer: videoPlayer,
            isAudio: isAudio,
            controller: self
        ))
        hosting.frame = NSRect(origin: .zero, size: size)

        let win = QuickLookWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        win.isOpaque = false
        win.backgroundColor = .clear
        win.hasShadow = true
        win.level = .floating
        win.isMovableByWindowBackground = true
        win.collectionBehavior = [.fullScreenAuxiliary]
        win.contentView = hosting
        win.onDismiss = { [weak self] in self?.close() }
        // 方向键：转发给资产面板做选中切换（面板侧监听后更新预览内容）
        win.onNavigate = { keyCode in
            NotificationCenter.default.post(name: .assetQuickLookNavigate,
                                            object: nil,
                                            userInfo: ["keyCode": keyCode])
        }

        let screenFrame = (NSApp.keyWindow?.screen ?? NSScreen.main)?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let origin: CGPoint
        if let reuseOrigin {
            // 切换目标：沿用上一次窗口位置，并夹在屏幕内防止换尺寸后跑出可见区域
            let maxX = max(screenFrame.minX, screenFrame.maxX - size.width)
            let maxY = max(screenFrame.minY, screenFrame.maxY - size.height)
            origin = CGPoint(x: min(max(reuseOrigin.x, screenFrame.minX), maxX),
                             y: min(max(reuseOrigin.y, screenFrame.minY), maxY))
        } else {
            // 首次打开：相对当前屏幕居中
            origin = CGPoint(x: screenFrame.midX - size.width / 2,
                             y: screenFrame.midY - size.height / 2)
        }
        win.setFrameOrigin(origin)

        // 新窗口就位后再清理旧窗口（preview 切换路径）
        if let old = window, old !== win {
            old.onDismiss = nil
            old.onNavigate = nil
            old.orderOut(nil)
            old.contentView = nil
        }

        window = win
        presentedID = asset.id
        win.makeKeyAndOrderFront(nil)
        debugLog("快速预览：\(reuseOrigin == nil ? "打开" : "切换")「\(asset.name)」 \(Int(size.width))×\(Int(size.height))")
    }

    // MARK: 关闭

    /// 只停播放与计时器，不碰窗口（切换目标 / 关闭窗口共用）
    private func stopPlayback() {
        audioTimer?.invalidate()
        audioTimer = nil
        if let videoPlayer {
            videoPlayer.pause()
            videoPlayer.replaceCurrentItem(with: nil)
        }
        videoPlayer = nil
        audioPlayer?.stop()
        audioPlayer = nil
        audioProgress = 0
        audioDuration = 0
    }

    func close() {
        presentationToken = nil
        stopPlayback()
        if let win = window {
            debugLog("快速预览：关闭")
            win.onDismiss = nil
            win.orderOut(nil)
            win.contentView = nil
            window = nil
        }
        presentedID = nil
    }

    private func startAudioTimer() {
        audioTimer?.invalidate()
        // Timer 回调固定落在主 RunLoop，直接以主 actor 更新进度（避免跨并发上下文捕获）
        audioTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            MainActor.assumeIsolated {
                guard let player = self.audioPlayer, player.duration > 0 else { return }
                self.audioProgress = min(1, player.currentTime / player.duration)
            }
        }
    }
}

// ============================================================

// MARK: - 预览窗口（无边框、可拖动、空格 / Esc 关闭）

final class QuickLookWindow: NSWindow {
    /// 请求关闭（空格 / Esc）。标注主 actor：闭包在主线程被调用，避免跨并发上下文的捕获告警
    var onDismiss: (@MainActor () -> Void)?
    /// 方向键导航回调（123 左 / 124 右 / 125 下 / 126 上）
    var onNavigate: (@MainActor (Int) -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func keyDown(with event: NSEvent) {
        // 按住空格的自动重复事件不再触发（避免刚关闭又被重复键重新打开）
        if event.isARepeat { return }
        // 空格（49）/ Esc（53）：停止播放并关闭，与系统快速查看一致
        if event.keyCode == 49 || event.keyCode == 53 {
            if let handler = onDismiss {
                MainActor.assumeIsolated { handler() }
            }
            return
        }
        // 方向键：切到上一个 / 下一个资产，预览内容原地替换并继续播放
        if event.keyCode == 123 || event.keyCode == 124
            || event.keyCode == 125 || event.keyCode == 126 {
            if let handler = onNavigate {
                MainActor.assumeIsolated { handler(Int(event.keyCode)) }
            }
            return
        }
        super.keyDown(with: event)
    }

    override func cancelOperation(_ sender: Any?) {
        guard let handler = onDismiss else { return }
        MainActor.assumeIsolated { handler() }
    }
}

// ============================================================

// MARK: - 预览内容视图

struct MediaQuickLookContentView: View {
    let assetName: String
    let previewImage: NSImage?
    let videoPlayer: AVPlayer?
    let isAudio: Bool
    @ObservedObject var controller: MediaQuickLookController

    var body: some View {
        ZStack {
            Color.black
            if let videoPlayer {
                VideoPlayerView(player: videoPlayer, videoGravity: .resizeAspect)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let previewImage {
                Image(nsImage: previewImage)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(isAudio ? 28 : 0)
            }
            if isAudio {
                audioPanel
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    /// 音频：封面 + 名称 + 进度（视频/图片走纯画面，保持系统预览的干净观感）
    private var audioPanel: some View {
        VStack(spacing: 14) {
            Spacer(minLength: 0)
            Text(assetName)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white.opacity(0.92))
                .lineLimit(1)
                .truncationMode(.middle)
                .padding(.horizontal, 20)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.18))
                    Capsule()
                        .fill(Color.white.opacity(0.85))
                        .frame(width: max(0, geo.size.width * controller.audioProgress))
                }
            }
            .frame(height: 4)
            .padding(.horizontal, 36)
            .padding(.bottom, 8)
        }
        .padding(.bottom, 18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
    }
}

// ============================================================

// MARK: - 空格键监听（资产库面板）

/// 资产库面板的空格键监听：选中恰好 1 个资产时，空格 = 打开/关闭快速预览。
/// 与 快捷键.swift 的 KeyboardShortcutHandler 同一套做法（本地事件监听 + 穿透视图），
/// 仅响应本面板所在窗口、面板可见、且文本输入未聚焦时的事件。
struct AssetQuickLookKeyHandler: NSViewRepresentable {
    /// 当前单选资产（非单选 / 无选中返回 nil）
    var selectedAsset: () -> AssetItem?
    /// 空格触发（打开或关闭预览）
    var onSpace: (AssetItem) -> Void
    /// 方向键触发（123 左 / 124 右 / 125 下 / 126 上）：切换选中
    var onArrow: (Int) -> Void = { _ in }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSView {
        let view = PassthroughView()
        context.coordinator.hostView = view
        context.coordinator.installMonitor()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator {
        var parent: AssetQuickLookKeyHandler
        weak var hostView: NSView?
        private var monitor: Any?

        init(_ parent: AssetQuickLookKeyHandler) {
            self.parent = parent
        }

        func installMonitor() {
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self = self else { return event }
                // 仅处理本面板所在窗口
                guard let hostWindow = self.hostView?.window, event.window === hostWindow else { return event }
                // 面板不在可见层级（切到其他 Tab）时不响应
                if let hostView = self.hostView, hostView.isHiddenOrHasHiddenAncestor { return event }
                // 文本输入（重命名 / 提示词 / 搜索）聚焦时不拦截
                if let responder = hostWindow.firstResponder,
                   responder is NSTextView || responder is NSTextField {
                    return event
                }
                let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                let blockedModifiers = flags.contains(.command) || flags.contains(.option)
                    || flags.contains(.control) || flags.contains(.shift)

                // 方向键：切换选中（上下左右皆可，按住会自动重复，与 Finder 网格一致）
                let arrowKeys: Set<UInt16> = [123, 124, 125, 126]
                if arrowKeys.contains(event.keyCode), !blockedModifiers {
                    self.parent.onArrow(Int(event.keyCode))
                    return nil
                }

                guard event.keyCode == 49, !blockedModifiers else { return event }
                // 按住空格的自动重复事件忽略（否则会反复开关预览）
                if event.isARepeat { return nil }
                guard let asset = self.parent.selectedAsset() else { return event }
                self.parent.onSpace(asset)
                return nil
            }
        }

        deinit {
            if let monitor = monitor {
                NSEvent.removeMonitor(monitor)
            }
        }
    }
}
