// ============================================================
//  文件作用：创作页。定义项目数据模型 ProjectStore / ProjectItem / ProjectCover、
//  上传文件模型 UploadedFile、剧本输入框 ScriptTextEditor、横向滚轮容器
//  HorizontalWheelScrollView、项目卡片 ProjectCardView、创作页 CreationView。
//  互动文件：引用 主ui.swift（AllProjectsView）；ProjectStore / ProjectCardView
//  被 主ui.swift 引用；ProjectStore 被 画布 ui.swift、画布 资产面板.swift 引用。
// ============================================================

import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers

// ============================================================

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

struct HorizontalWheelScroll<Content: View>: NSViewRepresentable {
    @ViewBuilder let content: () -> Content

    func makeNSView(context: Context) -> HorizontalWheelScrollView {
        let scrollView = HorizontalWheelScrollView()
        let hosting = NSHostingView(rootView: content())
        scrollView.documentView = hosting
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = false
        scrollView.verticalScrollElasticity = .none
        return scrollView
    }

    func updateNSView(_ scrollView: HorizontalWheelScrollView, context: Context) {
        guard let hosting = scrollView.documentView as? NSHostingView<Content> else { return }
        hosting.rootView = content()
        let ideal = hosting.fittingSize
        let height = max(ideal.height, scrollView.contentView.bounds.height)
        hosting.setFrameSize(NSSize(width: max(ideal.width, 1), height: height))
        hosting.frame.origin = .zero
    }
}

// MARK: - 项目卡片（主面板与二级面板共用）

struct ProjectCardView: View {
    @EnvironmentObject var store: ProjectStore
    let item: ProjectItem

    @State private var showRenameAlert = false
    @State private var newTitle = ""
    @State private var showDeleteConfirm = false

    init(item: ProjectItem) {
        self.item = item
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ZStack(alignment: .topTrailing) {
                coverContent
                    .frame(maxWidth: .infinity)
                    .frame(height: 108)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                Menu {
                    Button("重命名") {
                        debugLog("项目卡片：重命名「\(item.title)」")
                        newTitle = item.title
                        showRenameAlert = true
                    }
                    Button("复制") {
                        debugLog("项目卡片：复制「\(item.title)」")
                        store.duplicate(item: item)
                    }
                    Button("导出") {
                        debugLog("项目卡片：导出「\(item.title)」")
                        // TODO: 实现导出功能
                    }
                    Divider()
                    Button(role: .destructive) {
                        debugLog("项目卡片：请求删除「\(item.title)」")
                        showDeleteConfirm = true
                    } label: {
                        Label("删除", systemImage: "trash")
                    }
                } label: {
                    Image(systemName: "ellipsis")
                        .font(.system(size: 12, weight: .bold))
                        .foregroundColor(.primary)
                        .padding(5)
                        .background(Circle().fill(Color.white.opacity(0.92)))
                }
                .menuIndicator(.hidden)
                .menuStyle(.borderlessButton)
                .padding(6)
            }

            Text(item.title)
                .font(.subheadline.weight(.medium))
                .foregroundColor(.primary)
                .lineLimit(1)

            HStack(spacing: 6) {
                Text(item.timestamp)
                    .font(.caption)
                    .foregroundColor(.secondary)
                if let episode = item.episodeInfo {
                    Text(episode)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white)
                .shadow(color: Color.black.opacity(0.08), radius: 7, x: 0, y: 3)
        )
        .alert("重命名项目", isPresented: $showRenameAlert) {
            TextField("项目名称", text: $newTitle)
            Button("取消", role: .cancel) {}
            Button("确定") {
                debugLog("项目卡片：确认重命名为「\(newTitle)」")
                store.rename(item: item, to: newTitle)
            }
        } message: {
            Text("请输入新的项目名称")
        }
        .confirmationDialog("确定删除该项目吗？", isPresented: $showDeleteConfirm, titleVisibility: .visible) {
            Button("删除", role: .destructive) {
                debugLog("项目卡片：确认删除「\(item.title)」")
                store.delete(item: item)
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("删除后不可恢复")
        }
        .contentShape(Rectangle())
        .onTapGesture {
            // 点击项目进入它自己的无限画布
            debugLog("项目卡片：进入「\(item.title)」")
            store.selectedProject = item
        }
    }

    @ViewBuilder
    private var coverContent: some View {
        switch item.cover {
        case .placeholder:
            ZStack {
                Color(red: 0.90, green: 0.90, blue: 0.92)
                Image(systemName: "person.fill")
                    .font(.system(size: 32))
                    .foregroundColor(Color(red: 0.75, green: 0.75, blue: 0.78))
            }
        case .gradient(let colors):
            LinearGradient(gradient: Gradient(colors: colors),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
        case .scene(let colors, let label):
            ZStack {
                LinearGradient(gradient: Gradient(colors: colors),
                               startPoint: .topLeading,
                               endPoint: .bottomTrailing)
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.9))
                    .padding(4)
            }
        }
    }
}

// MARK: - 创作页

struct CreationView: View {
    @EnvironmentObject var store: ProjectStore
    @State private var showAllProjects = false
    @State private var pastedText = ""
    @State private var uploadedFiles: [UploadedFile] = []
    @State private var showFileImporter = false

    var body: some View {
        ZStack {
            if showAllProjects {
                AllProjectsView(isPresented: $showAllProjects)
                    .transition(.move(edge: .trailing))
            } else {
                mainContent
                    .transition(.move(edge: .leading))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: showAllProjects)
        .fileImporter(isPresented: $showFileImporter,
                      allowedContentTypes: [.plainText],
                      allowsMultipleSelection: true) { result in
            handleFileImport(result)
        }
    }

    private var mainContent: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    uploadArea
                    myProjectsSection
                }
                .padding(20)
            }
            .scrollContentBackground(.hidden)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(red: 0.93, green: 0.93, blue: 0.95))
    }

    // 上传操作区（上方悬浮已选文件列表，输入框内底部：左侧「+ 选择文件」胶囊、右侧发送圆钮）
    private var uploadArea: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !uploadedFiles.isEmpty {
                HorizontalWheelScroll {
                    HStack(spacing: 10) {
                        ForEach(uploadedFiles) { file in
                            uploadedFileChip(file)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .frame(height: 48)
            }

            // 对话记录区已移除（内置编码器不支持聊天）

            ZStack(alignment: .bottom) {
                ScriptTextEditor(text: $pastedText) { content in
                    sendProject(content: content)
                }
                .frame(height: 140)
                .padding(.bottom, 34)

                if pastedText.isEmpty {
                    Text("在此粘贴或输入剧本文本")
                        .font(.body)
                        .foregroundColor(Color(red: 0.60, green: 0.60, blue: 0.62))
                        .padding(.horizontal, 8)
                        .padding(.top, 10)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .allowsHitTesting(false)
                }

                HStack {
                    Button(action: {
                        debugLog("发送项目：选择文件")
                        showFileImporter = true
                    }) {
                        Label("选择文件", systemImage: "plus")
                            .font(.subheadline.weight(.medium))
                            .foregroundColor(.primary)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(
                                Capsule()
                                    .fill(Color.white)
                                    .overlay(
                                        Capsule().stroke(Color(red: 0.86, green: 0.86, blue: 0.89), lineWidth: 1)
                                    )
                            )
                    }
                    .buttonStyle(.plain)

                    Spacer()

                    Button(action: {
                        debugLog("发送项目")
                        sendProject(content: pastedText)
                    }) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 30, height: 30)
                            .background(
                                Circle().fill(Color.black.opacity(canSend ? 1 : 0.35))
                            )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
            }
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
            )
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color(red: 0.86, green: 0.86, blue: 0.89), lineWidth: 1)
                )
        )
    }

    private var canSend: Bool {
        !uploadedFiles.isEmpty
    }

    /// 发送入口：选择 txt 文件 → 创建项目。（聊天已移除：内置编码器不支持对话）

    private func sendProject(content: String) {
        guard !uploadedFiles.isEmpty else { return }
        var merged = content
        for file in uploadedFiles {
            if !merged.isEmpty { merged += "\n\n" }
            merged += "【\(file.name)】\n\(file.content)"
        }
        store.createProject(content: merged)
        pastedText = ""
        uploadedFiles.removeAll()
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                guard url.pathExtension.lowercased() == "txt" else { continue }
                let didStart = url.startAccessingSecurityScopedResource()
                defer { if didStart { url.stopAccessingSecurityScopedResource() } }
                guard let content = Self.loadTxtContent(from: url) else { continue }
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                uploadedFiles.append(UploadedFile(name: url.lastPathComponent, content: content, size: size))
            }
        case .failure:
            break
        }
    }

    private static func loadTxtContent(from url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        if let s = String(data: data, encoding: .utf8) { return s }
        let gb18030 = String.Encoding(rawValue: CFStringConvertEncodingToNSStringEncoding(CFStringEncoding(CFStringEncodings.GB_18030_2000.rawValue)))
        if let s = String(data: data, encoding: gb18030) { return s }
        return String(data: data, encoding: .isoLatin1)
    }

    private func uploadedFileChip(_ file: UploadedFile) -> some View {
        ZStack(alignment: .topTrailing) {
            HStack(spacing: 8) {
                Image(systemName: "doc.text.fill")
                    .font(.system(size: 16))
                    .foregroundColor(Color(red: 0.52, green: 0.52, blue: 0.85))
                VStack(alignment: .leading, spacing: 1) {
                    Text(file.name)
                        .font(.caption.weight(.medium))
                        .foregroundColor(.primary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text("txt · \(file.sizeText)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.trailing, 14)
            }
            .padding(.leading, 10)
            .padding(.trailing, 8)
            .padding(.vertical, 8)

            Button(action: {
                debugLog("发送项目：移除文件「\(file.name)」")
                uploadedFiles.removeAll { $0.id == file.id }
            }) {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 15, height: 15)
                    .background(Circle().fill(Color.black))
            }
            .buttonStyle(.plain)
            .padding(.top, 6)
            .padding(.trailing, 6)
        }
        .frame(width: 180)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.white)
        )
    }

    // 底部「我的项目」横向列表 + 新建按钮 + 全部入口
    private var myProjectsSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("我的项目")
                    .font(.title3.weight(.semibold))
                    .underline()

                Button(action: {
                    debugLog("项目列表：新建项目")
                    withAnimation { store.createProject() }
                }) {
                    Image(systemName: "plus")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(.primary)
                        .padding(6)
                        .background(Circle().fill(Color(red: 0.90, green: 0.90, blue: 0.92)))
                }
                .buttonStyle(.plain)

                Spacer()

                Button(action: {
                    debugLog("项目列表：打开全部项目")
                    withAnimation { showAllProjects = true }
                }) {
                    HStack(spacing: 4) {
                        Text("全部")
                        Image(systemName: "chevron.right")
                    }
                    .font(.subheadline)
                    .foregroundColor(.primary)
                }
                .buttonStyle(.plain)
            }

            if store.projects.isEmpty {
                emptyState
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 16) {
                        ForEach(store.projects) { item in
                            ProjectCardView(item: item)
                                .frame(width: 168)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 40))
                .foregroundColor(Color(red: 0.75, green: 0.75, blue: 0.78))
            Text("还没有项目")
                .font(.subheadline)
                .foregroundColor(.secondary)
            Text("点击「+」新建项目")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 30)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.white.opacity(0.6))
        )
    }
}
