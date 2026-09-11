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

@main
struct InfiniteCanvasApp: App {
    @StateObject private var store = ProjectStore()
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            MainPanelView()
                .environmentObject(store)
                .containerBackground(Color(red: 0.93, green: 0.93, blue: 0.95), for: .window)
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
        if let h3t = ProcessInfo.processInfo.environment["NA_H3TEST"], let h3v = Int(h3t), h3v >= 1 && h3v <= 20 {
            let code = H3PipelineRun.run()
            exit(code)
        }
        if ProcessInfo.processInfo.environment["NA_VAETEST"] == "1" {
            let code = H3VAETestRun.run()
            exit(code)
        }
        ensureCanvasDirectories()
        loadAssetLibrary()
        DispatchQueue.main.async {
            for window in NSApp.windows {
                window.titlebarAppearsTransparent = false
            }
        }
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
