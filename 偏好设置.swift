//
//  偏好设置.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/14.
//
// ============================================================
//  文件作用：偏好设置面板（图2样式：左右分栏）+ 全局设置存储 AppSettings。
//  左侧导航：通用设置 / API / 授权设置 / 关于；
//  右侧内容区：通用设置页含项目文件地址，
//  以及「画布」分组（节点默认颜色、滚轮速度倍率，UserDefaults 持久化）。
//  互动文件：被 主ui.swift 的 Settings 场景引用（PreferencesView）；
//  AppSettings 被 画布状态 公共函数.swift、读取&保存.swift 读取。
// ============================================================

import SwiftUI
import Combine

// ============================================================

// MARK: - 全局设置存储（UserDefaults 持久化）

final class AppSettings: ObservableObject {
    static let shared = AppSettings()
    private let defaults = UserDefaults.standard

    // 节点默认颜色（存颜色名，Color 不可 Codable）
    @Published var defaultNodeColorName: String {
        didSet { defaults.set(defaultNodeColorName, forKey: "defaultNodeColorName") }
    }
    // 触控板滚轮加速倍率
    @Published var wheelAccelerationPrecise: Double {
        didSet { defaults.set(wheelAccelerationPrecise, forKey: "wheelAccelerationPrecise") }
    }
    // 鼠标滚轮加速倍率
    @Published var wheelAccelerationWheel: Double {
        didSet { defaults.set(wheelAccelerationWheel, forKey: "wheelAccelerationWheel") }
    }
    // 项目文件地址（默认 ~/Documents/无限画布，可自定义）
    @Published var canvasRootPath: String {
        didSet { defaults.set(canvasRootPath, forKey: "canvasRootPath") }
    }

    init() {
        let d = UserDefaults.standard
        defaultNodeColorName = d.string(forKey: "defaultNodeColorName") ?? "pink"
        wheelAccelerationPrecise = d.object(forKey: "wheelAccelerationPrecise") as? Double ?? 2.0
        wheelAccelerationWheel = d.object(forKey: "wheelAccelerationWheel") as? Double ?? 30.0
        let savedPath = d.string(forKey: "canvasRootPath") ?? ""
        canvasRootPath = savedPath.isEmpty ? Self.defaultRootPath : savedPath
    }

    // 默认文档地址（~/Documents/无限画布）
    static var defaultRootPath: String {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("无限画布").path
    }

    // 可选节点主题色（有序，与工具栏一致）
    var colorOptions: [(name: String, color: Color)] {
        [
            ("pink", .pink),
            ("标准粉", Color(red: 0.98, green: 0.65, blue: 0.72)),
            ("blue", .blue),
            ("red", .red),
            ("orange", .orange),
            ("green", .green),
            ("purple", .purple),
            ("teal", .teal),
            ("black", .black)
        ]
    }

    // 当前默认节点颜色
    var defaultNodeColor: Color {
        colorOptions.first { $0.name == defaultNodeColorName }?.color ?? .pink
    }

    /// 当前默认节点颜色的 NSColor 版（与 colorOptions 同名一一对应，供位图绘制/占位图生成用）
    var defaultNodeNSColor: NSColor {
        switch defaultNodeColorName {
        case "pink": return .systemPink
        case "标准粉": return NSColor(calibratedRed: 0.98, green: 0.65, blue: 0.72, alpha: 1)
        case "blue": return .systemBlue
        case "red": return .systemRed
        case "orange": return .systemOrange
        case "green": return .systemGreen
        case "purple": return .systemPurple
        case "teal": return .systemTeal
        case "black": return .black
        default: return .systemPink
        }
    }

    /// 公共改色入口：偏好设置与画布右上角共用，改一边另一边同步（唯一颜色数据源）
    func setNodeColor(_ color: Color) {
        guard let name = colorOptions.first(where: { $0.color == color })?.name else { return }
        defaultNodeColorName = name
    }

    // 恢复默认参数
    func resetToDefaults() {
        defaultNodeColorName = "pink"
        wheelAccelerationPrecise = 2.0
        wheelAccelerationWheel = 30.0
        let oldPath = canvasRootPath
        canvasRootPath = Self.defaultRootPath
        if oldPath != canvasRootPath {
            migrateCanvasRoot(from: oldPath, to: canvasRootPath)
        }
    }

    /// 变更项目文件地址并迁移旧地址数据（选择新地址时调用）
    func changeCanvasRootPath(to newPath: String) {
        let oldPath = canvasRootPath
        guard oldPath != newPath, !oldPath.isEmpty, !newPath.isEmpty else { return }
        canvasRootPath = newPath
        migrateCanvasRoot(from: oldPath, to: newPath)
    }
}

// ============================================================

// MARK: - 偏好设置面板（左右分栏）

struct PreferencesView: View {
    @State private var selectedSection = 0
    @State private var showResetMenu = false
    @ObservedObject private var settings = AppSettings.shared

    private let sections = ["通用设置", "API", "授权设置", "快捷键", "关于"]

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            content
        }
        .frame(width: 620, height: 440)
        .background(Color(red: 0.95, green: 0.95, blue: 0.97))
        .background(FloatingWindow())
    }

    // 左侧导航栏
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(sections.indices, id: \.self) { index in
                Button {
                    selectedSection = index
                } label: {
                    HStack {
                        Text(sections[index])
                            .font(.system(size: 13))
                            .foregroundColor(selectedSection == index ? .white : .primary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(selectedSection == index ? Color.accentColor : Color.clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Spacer()
            // 左下角齿轮：恢复默认参数
            Button {
                showResetMenu.toggle()
            } label: {
                HStack {
                    Image(systemName: "gearshape")
                        .font(.system(size: 13))
                    Text("恢复默认")
                        .font(.system(size: 12))
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showResetMenu, arrowEdge: .bottom) {
                VStack(spacing: 10) {
                    Text("恢复默认参数")
                        .font(.system(size: 13, weight: .medium))
                    Text("将把所有设置恢复为默认值")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("恢复默认") {
                        debugLog("偏好设置：恢复默认参数")
                        settings.resetToDefaults()
                        showResetMenu = false
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(14)
            }
        }
        .padding(10)
        .frame(width: 180)
        .background(Color(red: 0.90, green: 0.90, blue: 0.93))
    }

    // 右侧内容区
    private var content: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(sections[selectedSection])
                .font(.title2)
                .bold()
            Divider()
            if selectedSection == 0 {
                generalSettings
            } else if selectedSection == 3 {
                shortcutSettings
            } else {
                Text("这里是「\(sections[selectedSection])」的内容区域，占位文本，后续补充。")
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.white)
    }

    // 通用设置页
    private var generalSettings: some View {
        VStack(alignment: .leading, spacing: 18) {
            // 项目文件地址
            VStack(alignment: .leading, spacing: 6) {
                Text("项目文件地址")
                    .font(.system(size: 13, weight: .medium))
                HStack(spacing: 8) {
                    TextField("", text: $settings.canvasRootPath)
                        .textFieldStyle(.roundedBorder)
                    Button("选择…") {
                        debugLog("偏好设置：选择项目文件地址")
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
                }
            }

            // 画布分组
            VStack(alignment: .leading, spacing: 12) {
                Text("画布")
                    .font(.system(size: 13, weight: .medium))

                VStack(alignment: .leading, spacing: 12) {
                    // 节点默认颜色
                    VStack(alignment: .leading, spacing: 8) {
                        Text("节点默认颜色")
                            .font(.system(size: 13, weight: .medium))
                        HStack(spacing: 10) {
                            ForEach(settings.colorOptions, id: \.name) { option in
                                Button {
                                    settings.setNodeColor(option.color)
                                } label: {
                                    ZStack {
                                        Circle()
                                            .fill(option.color)
                                            .frame(width: 22, height: 22)
                                        if settings.defaultNodeColorName == option.name {
                                            Image(systemName: "checkmark")
                                                .font(.system(size: 11, weight: .bold))
                                                .foregroundColor(.white)
                                        }
                                    }
                                    .overlay(
                                        Circle().stroke(settings.defaultNodeColorName == option.name ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: 2.5)
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    // 滚轮速度倍率
                    HStack(spacing: 8) {
                        Text("触控板滚轮倍率")
                            .font(.system(size: 13))
                        TextField("", value: $settings.wheelAccelerationPrecise, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                    }
                    HStack(spacing: 8) {
                        Text("鼠标滚轮倍率")
                            .font(.system(size: 13))
                        TextField("", value: $settings.wheelAccelerationWheel, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                    }
                }
                .padding(.leading, 12)
            }

            Spacer()
        }
    }

    // 快捷键设置页：展示画布快捷键（与 快捷键.swift 中实现一致）
    private var shortcutSettings: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("画布快捷键")
                .font(.system(size: 13, weight: .medium))

            VStack(spacing: 0) {
                shortcutRow(name: "改名", keys: ["F2"])
                Divider()
                shortcutRow(name: "缩放适配", keys: ["空格"])
                Divider()
                shortcutRow(name: "撤销", keys: ["Command", "Z"])
                Divider()
                shortcutRow(name: "重做", keys: ["Command", "Alt", "Z"])
                Divider()
                shortcutRow(name: "复制节点", keys: ["Command", "C"])
                Divider()
                shortcutRow(name: "粘贴节点", keys: ["Command", "V"])
                Divider()
                shortcutRow(name: "删除", keys: ["Delete"])
            }
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
            )

            Text("提示：在画布中按对应快捷键即可触发，与工具栏按钮操作等同。")
                .font(.system(size: 11))
                .foregroundColor(.secondary)

            Spacer()
        }
    }

    // 快捷键行：功能名 + 键帽
    private func shortcutRow(name: String, keys: [String]) -> some View {
        HStack {
            Text(name)
                .font(.system(size: 13))
            Spacer()
            HStack(spacing: 4) {
                ForEach(keys, id: \.self) { key in
                    Text(key)
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.white)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(Color.gray.opacity(0.4), lineWidth: 1)
                                )
                        )
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }
}

// ============================================================

// MARK: - 偏好设置窗口置顶

/// 让偏好设置窗口保持 floating 层级，点击主体窗口时不会沉到后面。
struct FloatingWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            view.window?.level = .floating
        }
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
