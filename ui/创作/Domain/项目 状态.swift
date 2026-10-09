// ============================================================
//  文件作用：项目状态（数据层逻辑）。ProjectCover / ProjectItem / UploadedFile 项目数据模型，ProjectStore 项目状态（创建/重命名/复制/删除/加载保存），loadTxtContent 文本读取。
//  互动文件：ProjectStore 被 创作 ui.swift 的 CreationView 引用；loadProjects/saveProject/deleteProjectFiles 等全局函数在根目录读取&保存.swift。
// ============================================================

import SwiftUI
import AppKit
import Foundation
import Combine

// MARK: - 项目封面样式

enum ProjectCover {
    case placeholder                          // 灰色占位（人像图标）
    case gradient([Color])                    // 彩色渐变封面
    case scene(colors: [Color], label: String) // 带文字的场景封面

    /// 封面类型名（用于持久化，Color 不可 Codable）
    var caseName: String {
        switch self {
        case .placeholder: return "placeholder"
        case .gradient: return "gradient"
        case .scene: return "scene"
        }
    }
}

// MARK: - 项目数据模型

struct ProjectItem: Identifiable {
    let id: UUID
    var title: String
    let timestamp: String
    let episodeInfo: String?
    let content: String?
    let cover: ProjectCover

    init(id: UUID = UUID(),
         title: String,
         timestamp: String,
         episodeInfo: String? = nil,
         content: String? = nil,
         cover: ProjectCover) {
        self.id = id
        self.title = title
        self.timestamp = timestamp
        self.episodeInfo = episodeInfo
        self.content = content
        self.cover = cover
    }
}

// MARK: - 已上传文件（当前仅支持 txt）

struct UploadedFile: Identifiable {
    let id = UUID()
    let name: String
    let content: String
    let size: Int

    var sizeText: String {
        if size >= 1_048_576 {
            return String(format: "%.1fMB", Double(size) / 1_048_576)
        } else if size >= 1024 {
            return String(format: "%.0fKB", Double(size) / 1024)
        } else {
            return "\(size)B"
        }
    }
}

// MARK: - 项目状态（共享，支持手动创建）

final class ProjectStore: ObservableObject {
    @Published var projects: [ProjectItem] = []
    // 当前打开的画布项目（nil 表示未进入画布）
    @Published var selectedProject: ProjectItem?

    init() {
        // 启动时从磁盘重建项目列表
        projects = loadProjects()
    }

    func createProject(content: String? = nil) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        let item = ProjectItem(title: "未命名",
                               timestamp: formatter.string(from: Date()),
                               content: content,
                               cover: .placeholder)
        projects.insert(item, at: 0)
        // 空项目也立即落盘（项目/<id>.canvas/project.json）
        saveProject(item, canvas: CanvasStore())
    }

    func rename(item: ProjectItem, to newTitle: String) {
        guard !newTitle.isEmpty else { return }
        if let index = projects.firstIndex(where: { $0.id == item.id }) {
            projects[index].title = newTitle
        }
    }

    func duplicate(item: ProjectItem) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        let copy = ProjectItem(title: item.title + " 副本",
                               timestamp: formatter.string(from: Date()),
                               episodeInfo: item.episodeInfo,
                               content: item.content,
                               cover: item.cover)
        projects.insert(copy, at: 0)
    }

    func delete(item: ProjectItem) {
        projects.removeAll { $0.id == item.id }
        // 同步删除磁盘上的 项目/<id>.canvas 目录
        deleteProjectFiles(item)
    }
}

// MARK: - 剧本输入框（回车发送、Alt/Ctrl+回车换行、隐藏滚动条、滚轮滚动）

final class ScriptTextView: NSTextView {
    var onSend: ((String) -> Void)?

    override func keyDown(with event: NSEvent) {
        // 36 = Return，76 = 数字键盘 Enter
        if event.keyCode == 36 || event.keyCode == 76 {
            let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
            if flags.contains(.option) || flags.contains(.control) {
                insertNewline(nil)   // Alt/Ctrl + 回车 = 换行
            } else {
                onSend?(string)       // 纯回车 = 发送
            }
            return
        }
        super.keyDown(with: event)
    }
}

struct ScriptTextEditor: NSViewRepresentable {
    @Binding var text: String
    var onSend: (String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        let textView = ScriptTextView(frame: .zero)

        textView.delegate = context.coordinator
        textView.onSend = { context.coordinator.parent.onSend($0) }
        textView.isRichText = false
        textView.font = NSFont.systemFont(ofSize: 13)
        textView.textColor = .labelColor
        textView.drawsBackground = false
        textView.backgroundColor = .clear
        textView.textContainerInset = NSSize(width: 4, height: 6)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)

        scrollView.documentView = textView
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = false
        scrollView.backgroundColor = .clear

        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? ScriptTextView else { return }
        if textView.string != text {
            textView.string = text
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ScriptTextEditor

        init(_ parent: ScriptTextEditor) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
        }
    }
}

// MARK: - 横向滚轮滚动容器（鼠标滚轮驱动横向滚动）

final class HorizontalWheelScrollView: NSScrollView {
    override func scrollWheel(with event: NSEvent) {
        if abs(event.scrollingDeltaY) > abs(event.scrollingDeltaX) {
            var delta = event.scrollingDeltaY
            if event.hasPreciseScrollingDeltas {
                delta *= wheelScrollAccelerationPrecise   // 触控板：轻度加速
            } else {
                delta *= wheelScrollAccelerationWheel     // 普通鼠标滚轮：行→像素并明显加速
            }
            let current = contentView.bounds.origin.x
            let docWidth = documentView?.frame.width ?? 0
            let maxX = max(0, docWidth - contentView.bounds.width)
            let newX = min(max(0, current - delta), maxX)
            contentView.scroll(to: NSPoint(x: newX, y: contentView.bounds.origin.y))
            reflectScrolledClipView(contentView)
        } else {
            super.scrollWheel(with: event)
        }
    }
}

// MARK: - 文本文件读取（txt 导入）

/// 读取 txt 文件文本内容（优先 UTF-8，失败回退 GB18030 / ISO Latin-1）
func loadTxtContent(from url: URL) -> String? {
    guard let data = try? Data(contentsOf: url) else { return nil }
    if let s = String(data: data, encoding: .utf8) { return s }
    let gb18030 = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
    if let s = String(data: data, encoding: gb18030) { return s }
    return String(data: data, encoding: .isoLatin1)
}
