// ============================================================
//  文件作用：应用入口与主框架。定义 @main InfiniteCanvasApp（应用启动）、
//  AppDelegate（窗口标题栏配置）、MainPanelView 主面板（TabView 切换「创作」/
//  「资产管理」，进入项目后切换到 CanvasEditorView 画布编辑器）、
//  AllProjectsView 二级面板（我的项目/全部项目网格）。
//  互动文件：引用 创作ui.swift（ProjectStore、CreationView、ProjectCardView）、
//  画布 ui.swift（CanvasEditorView）、资产管理ui.swift（AssetManagementView）、
//  偏好设置.swift（PreferencesView，Settings 场景）；AllProjectsView 被 创作ui.swift 的 CreationView 引用。
// ============================================================

import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers
import AVFoundation

@main
struct InfiniteCanvasApp: App {
    @StateObject private var store = ProjectStore()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var showLaunchPreview = true

    var body: some Scene {
        WindowGroup {
            if showLaunchPreview {
                LaunchPreviewView(isPresented: $showLaunchPreview)
                    .containerBackground(Color(red: 0.93, green: 0.93, blue: 0.95), for: .window)
            } else {
                MainPanelView()
                    .environmentObject(store)
                    .containerBackground(Color(red: 0.93, green: 0.93, blue: 0.95), for: .window)
            }
        }
        // 偏好设置：菜单栏应用菜单自动出现「偏好设置…」（⌘,），打开 偏好设置.swift 的 PreferencesView
        Settings {
            PreferencesView()
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        if ProcessInfo.processInfo.environment["NA_SELFCHECK"] == "1" {
            let code = NASelfCheck.run()
            exit(code)
        }
        if ProcessInfo.processInfo.environment["NA_BENCH"] == "1" {
            let code = NABench.run()
            exit(code)
        }
        if let h3t = ProcessInfo.processInfo.environment["NA_H3TEST"], let h3v = Int(h3t), h3v >= 1 && h3v <= 30 {
            let code = H3PipelineRun.run()
            exit(code)
        }
        if ProcessInfo.processInfo.environment["NA_VAETEST"] == "1" {
            let code = H3VAETestRun.run()
            exit(code)
        }
        ensureCanvasDirectories()
        loadAssetLibrary()
    }
}

// MARK: - 启动预览（阶段 1：开篇视频全铺满；阶段 2：项目地址选择，右下角箭头进入）

struct LaunchPreviewView: View {
    @Binding var isPresented: Bool
    @ObservedObject private var settings = AppSettings.shared
    @State private var videoPlayer: AVPlayer?
    @State private var videoReady = false
    @State private var loopObserver: NSObjectProtocol?
    @State private var hostWindow: NSWindow?

    var body: some View {
        ZStack {
            // 全窗口开篇视频（铺满整个窗口，穿透标题栏，无控制条）
            Color.black
            if videoReady, let videoPlayer {
                AVPlayerLayerView(player: videoPlayer)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onAppear {
                        videoPlayer.play()
                    }
            } else {
                ProgressView()
                    .controlSize(.small)
            }

            // 视频内靠下：项目地址选择 + 进入箭头（醒目浮层）
            VStack {
                Spacer()
                HStack(alignment: .center, spacing: 20) {
                    if !hasValidProjectPath {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("选择项目地址")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundColor(.white)
                        HStack(spacing: 8) {
                            Text(settings.canvasRootPath)
                                .font(.system(size: 13))
                                .foregroundColor(.white)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(width: 680, alignment: .leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.black.opacity(0.35))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(Color.white.opacity(0.55), lineWidth: 1)
                                )
                            Button("选择…") {
                                debugLog("启动预览：选择项目文件地址")
                                let panel = NSOpenPanel()
                                panel.canChooseDirectories = true
                                panel.canChooseFiles = false
                                panel.allowsMultipleSelection = false
                                panel.directoryURL = URL(fileURLWithPath: settings.canvasRootPath)
                                if panel.runModal() == .OK, let url = panel.url {
                                    // 文档库根 = 选择目录/无限画布（补上根文件夹）
                                    let newRoot = url.appendingPathComponent("无限画布", isDirectory: true).path
                                    settings.changeCanvasRootPath(to: newRoot)
                                }
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 6)
                            .background(Color.black)
                            .cornerRadius(6)
                        }
                    }
                    }
                    Spacer()
                    // 右下角：向右箭头，点击进入软件本体
                    Button {
                        debugLog("启动预览：进入应用")
                        isPresented = false
                    } label: {
                        Image(systemName: "arrow.right.circle.fill")
                            .font(.system(size: 52))
                            .foregroundColor(.black)
                    }
                    .buttonStyle(.plain)
                    .help("进入应用")
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 40)
            }
        }
        .clipped()
        .ignoresSafeArea()
        .frame(width: previewWindowSize(for: NSScreen.main).width,
               height: previewWindowSize(for: NSScreen.main).height)
        .background(WindowAccessor { window in
            if let window {
                hostWindow = window
                // 强制窗口尺寸与内容一致，杜绝四周露出窗口背景（白边真因）
                let fixed = previewWindowSize(for: window.screen ?? NSScreen.main)
                if window.contentView?.frame.size != fixed {
                    window.setContentSize(fixed)
                }
                window.center()
                applyTitleBar(hidden: true)
            }
        })
        .onAppear {
            setupOpeningVideo()
        }
        .onDisappear {
            applyTitleBar(hidden: false)
            if let loopObserver {
                NotificationCenter.default.removeObserver(loopObserver)
            }
            videoPlayer?.pause()
            videoPlayer = nil
        }
    }

    /// 按屏幕可用区域按比例计算预览窗口尺寸：高=可用高度85%（上限800），宽=16:9（上限可用宽度86%）
    /// 自动适配任意分辨率屏幕与外接屏，窗口始终完整落在可视区内
    private func previewWindowSize(for screen: NSScreen?) -> NSSize {
        let visible = screen?.visibleFrame
            ?? NSScreen.main?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1280, height: 720)
        let sw = max(visible.width, 800)
        let sh = max(visible.height, 600)
        let h = floor(min(sh * 0.85, 800))
        let w = floor(min(h * 16.0 / 9.0, sw * 0.86))
        return NSSize(width: w, height: h)
    }

    /// 项目路径是否已有效（目录存在即视为已配置过，启动预览不再显示选择框）
    private var hasValidProjectPath: Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: settings.canvasRootPath, isDirectory: &isDir)
        return exists && isDir.boolValue
    }

    /// 预览期间隐藏标题栏（关闭/缩放按钮悬浮在视频上），进入主面板后恢复
    private func applyTitleBar(hidden: Bool) {
        guard let window = hostWindow ?? NSApp.windows.first else { return }
        if hidden {
            window.title = ""
            window.titleVisibility = .hidden
            window.titlebarAppearsTransparent = true
            window.styleMask.insert(.fullSizeContentView)
            // 钉死预览窗口尺寸：移除可缩放，禁止拖拽边缘放大/缩小，杜绝放大后露白边
            window.styleMask.remove(.resizable)
            window.isMovableByWindowBackground = true
        } else {
            window.title = "无限画布"
            window.titleVisibility = .visible
            window.titlebarAppearsTransparent = false
            window.styleMask.remove(.fullSizeContentView)
            // 恢复主面板窗口可缩放
            window.styleMask.insert(.resizable)
            window.isMovableByWindowBackground = false
        }
    }

    /// 加载「开篇」视频：优先 Bundle 资源，回退到项目源码根目录
    private func setupOpeningVideo() {
        var url = Bundle.main.url(forResource: "开篇", withExtension: "mov")
        if url == nil {
            let fallback = URL(fileURLWithPath: settings.canvasRootPath)
                .deletingLastPathComponent()
                .appendingPathComponent("开篇.mov")
            if FileManager.default.fileExists(atPath: fallback.path) {
                url = fallback
            }
        }
        guard let url else { return }
        let player = AVPlayer(url: url)
        videoPlayer = player
        videoReady = true
        // 播完循环播放（开篇为欢迎视频）
        loopObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player.currentItem,
            queue: .main
        ) { [weak player] _ in
            player?.seek(to: .zero)
            player?.play()
        }
    }
}

// MARK: - AVPlayerLayer 容器（无控制条，视频全铺满）

/// 用 AVPlayerLayer 直接铺满视图，无任何播放控制条（开篇为纯展示视频）
final class PlayerContainerView: NSView {
    private var playerLayer: AVPlayerLayer?

    func attach(player: AVPlayer) {
        wantsLayer = true
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspectFill
        layer.frame = bounds
        layer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        self.layer?.addSublayer(layer)
        playerLayer = layer
    }

    override func layout() {
        super.layout()
        // 每次视图尺寸变化都让视频层跟随，保证全铺满无留白
        playerLayer?.frame = bounds
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        playerLayer?.frame = bounds
    }
}

struct AVPlayerLayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> PlayerContainerView {
        let view = PlayerContainerView()
        view.attach(player: player)
        return view
    }

    func updateNSView(_ nsView: PlayerContainerView, context: Context) {}
}

// MARK: - 窗口访问器（视图真正加入窗口后回调，可靠拿到宿主 NSWindow）

struct WindowAccessor: NSViewRepresentable {
    var callback: (NSWindow?) -> Void

    func makeNSView(context: Context) -> WindowAccessorView {
        let view = WindowAccessorView()
        view.onWindowChange = callback
        return view
    }

    func updateNSView(_ nsView: WindowAccessorView, context: Context) {
        nsView.onWindowChange = callback
        nsView.notify()
    }
}

/// 在 viewDidMoveToWindow 时机回调，保证 window 一定可用（比 async 拿 window 可靠）
final class WindowAccessorView: NSView {
    var onWindowChange: ((NSWindow?) -> Void)?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        notify()
        // 窗口初始化是异步的，延迟重试确保设置不被后续初始化覆盖
        DispatchQueue.main.async { [weak self] in
            self?.notify()
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            self?.notify()
        }
    }

    func notify() {
        onWindowChange?(window)
    }
}

// MARK: - 主面板（TabView 标签栏）

struct MainPanelView: View {
    @State private var selectedModule = 0
    @EnvironmentObject var store: ProjectStore

    var body: some View {
        ZStack {
            // 主画布切换：点击项目进入它自己的无限画布
            if store.selectedProject != nil {
                CanvasEditorView(selectedProject: $store.selectedProject)
                    .transition(.opacity)
            } else {
                TabView(selection: $selectedModule) {
                    CreationView()
                        .tabItem {
                            Label("创作", systemImage: "pencil.line")
                        }
                        .tag(0)

                    AssetManagementView()
                        .tabItem {
                            Label("资产管理", systemImage: "photo.on.rectangle")
                        }
                        .tag(1)
                }
                .background(Color(red: 0.93, green: 0.93, blue: 0.95))
            }

            // 调试开关 + 右下角监控浮层（见 调试中心.swift）
            DebugControlPanel()
        }
        .animation(.easeInOut(duration: 0.25), value: store.selectedProject?.id)
    }
}

// ============================================================

// MARK: - 二级面板（我的项目 · 全部项目网格）

struct AllProjectsView: View {
    @Binding var isPresented: Bool
    @EnvironmentObject var store: ProjectStore

    private let columns = [
        GridItem(.adaptive(minimum: 150), spacing: 16)
    ]

    init(isPresented: Binding<Bool>) {
        self._isPresented = isPresented
    }

    var body: some View {
        VStack(spacing: 0) {
            navBar

            ScrollView {
                if store.projects.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "folder.badge.plus")
                            .font(.system(size: 40))
                            .foregroundColor(Color(red: 0.75, green: 0.75, blue: 0.78))
                        Text("还没有项目")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        Text("返回主面板点击「+」新建项目")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 60)
                } else {
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                        ForEach(store.projects) { item in
                            ProjectCardView(item: item)
                        }
                    }
                    .padding(20)
                }
            }
            .background(Color.white)
        }
        .background(Color.white)
    }

    // 顶部返回箭头 + 标题
    private var navBar: some View {
        HStack {
            Button(action: {
                debugLog("导航：返回「我的项目」")
                withAnimation { isPresented = false }
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left")
                    Text("我的项目")
                }
                .font(.headline)
                .foregroundColor(.primary)
            }
            .buttonStyle(.plain)
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(Color(red: 0.95, green: 0.95, blue: 0.97))
    }
}
