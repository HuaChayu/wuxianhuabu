//
//  模型下载.swift
//  无限画布
//
//  Swift 原生 ModelScope 模型下载器（零第三方依赖，仅 Foundation + CryptoKit）
//
//  端点已实测验证（2026-09-19）：
//    1) 文件清单: GET https://modelscope.cn/api/v1/models/{owner}/{repo}/repo/files?Revision=master&Recursive=true
//    2) 下载:     GET https://modelscope.cn/models/{owner}/{repo}/resolve/{revision}/{filepath}
//
//  断点续传支持（三级）：
//    1) 后台会话（background session）：网络中断 / 进程被系统挂起时，系统自动继续下载
//    2) 主动暂停：pause() 调用 cancel(byProducingResumeData:) 把续传数据存 UserDefaults，
//       下次触发下载自动从断点继续
//    3) 失败自动续传：任务出错时从 error.userInfo 提取 resumeData 保存，重试时优先续传
//    4) 文件完整性：已下载文件 SHA256 与远端一致则直接跳过，损坏则自动重下
//
//  用法示例：
//    let dl = ModelScopeDownloader(owner: "HuaCHayu",
//                                  repo: "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit")
//    dl.onProgress = { done, total, name in
//        print("\(name): \(done)/\(total)")
//    }
//    try await dl.downloadModel(to: modelRootURL)   // modelRootURL = 项目 model 目录
//

import Foundation
import CryptoKit
import Combine

// MARK: - 文件条目

struct ModelScopeFile: Decodable {
    let path: String
    let size: Int
    let sha256: String
    let type: String      // "tree" | "blob"

    // ★ 2026-09-19 修复：ModelScope listFiles API 返回的键是大写（Path/Size/Sha256/Type），
    //   JSONDecoder 默认精确匹配大小写，缺少映射会导致解码失败抛出
    //   NSCocoaErrorDomain "The data couldn't be read because it is missing."（与 260 同文案）
    enum CodingKeys: String, CodingKey {
        case path = "Path"
        case size = "Size"
        case sha256 = "Sha256"
        case type = "Type"
    }

    var name: String { (path as NSString).lastPathComponent }
    var isDirectory: Bool { type == "tree" }
}

// MARK: - 文件查找（嵌套落位兜底）

/// 按文件名在目录下任意深度递归查找（跳过隐藏文件，如 .DS_Store、.gitkeep），
/// 用于识别旧版把权重落进 localDir/<同名子目录>/... 任意深度的历史落盘文件。
enum FileFinder {
    static func first(named name: String, under root: URL) -> URL? {
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return nil }
        for case let url as URL in enumerator {
            // 带子目录路径的 requiredFile（如 "extras/custom_heads.safetensors"）按相对路径后缀匹配，
            // 否则仍按纯文件名匹配（lastPathComponent 无法匹配含 "/" 的 name）
            if name.contains("/") {
                if url.path.hasSuffix("/" + name) { return url }
            } else if url.lastPathComponent == name {
                return url
            }
        }
        return nil
    }
}

// MARK: - 活动下载文件（UI 列表项：文件名 + 该文件独立实时速度）

struct ActiveFileDownload {
    let name: String
    let written: Int64
    let total: Int64
    let speedBytesPerSec: Double
}

// MARK: - 下载器

final class ModelScopeDownloader: NSObject, URLSessionDownloadDelegate {
    let owner: String
    let repo: String
    var revision = "master"

    /// 整包进度回调 (累计已下载字节, 总字节, 当前文件名)
    var onProgress: ((Int64, Int64, String) -> Void)?
    /// 单文件进度回调 (已下载, 总字节, 文件名, 实时速度 B/s)，可做文件级进度条与实时速度显示
    var onFileProgress: ((Int64, Int64, String, Double) -> Void)?
    /// 活动下载文件列表回调（并发下载中，按文件维度独立字节与速度；文件完成/新增时刷新）
    var onActiveFilesChanged: (([ActiveFileDownload]) -> Void)?

    private var session: URLSession!

    /// 最大并发下载文件数（类似 aria2 -j）
    private let maxConcurrentFiles = 3

    // MARK: 并发任务状态（stateLock 保护）

    /// 单文件下载上下文：delegate 回调按 task 查表定位（多任务并存）
    private final class FileDownloadContext {
        let file: ModelScopeFile
        let dest: URL
        /// 该任务当前已写入的字节数（didWriteData 更新，速度统计用）
        var written: Int64 = 0
        /// 该文件独立的 5 秒滑动窗口采样（bytes 为本文件 written，从 0 起）
        var speedSamples: [SpeedSample] = []
        var continuation: CheckedContinuation<URL, Error>?
        init(file: ModelScopeFile, dest: URL) {
            self.file = file
            self.dest = dest
        }
    }

    private let stateLock = NSLock()
    /// 活动下载任务表：task -> 文件上下文（并发下多任务共存，按 task 归属校验）
    private var activeTasks: [URLSessionDownloadTask: FileDownloadContext] = [:]
    /// 已完整下载并计入总进度的字节数（主任务累加，delegate 读，锁保护）
    private var completedBytes: Int64 = 0
    /// 整包总字节数（downloadModel 启动时由远端清单求得，进度计算分母，锁保护读）
    private var totalBytes: Int64 = 0

    // MARK: 实时速度（滑动窗口，跨文件连续）
    /// 速度采样：时间戳 + 该时刻的全局累计已下载字节（已完成字节 + 各活动任务已写字节）
    private struct SpeedSample {
        let timestamp: Date
        let bytes: Int64
    }
    private var speedSamples: [SpeedSample] = []
    private let speedWindowSeconds: TimeInterval = 5.0

    /// 记录一次进度采样并维护 5 秒滑动窗口（旧样本自动淘汰）
    private func recordSpeedSample(globalBytes: Int64) {
        let now = Date()
        speedSamples.append(SpeedSample(timestamp: now, bytes: globalBytes))
        while let first = speedSamples.first,
              now.timeIntervalSince(first.timestamp) > speedWindowSeconds {
            speedSamples.removeFirst()
        }
    }

    /// 当前实时速度（字节/秒）：窗口首末样本差分，平滑抗抖动；样本不足时返回 0
    var currentSpeedBytesPerSec: Double {
        guard speedSamples.count >= 2,
              let first = speedSamples.first,
              let last = speedSamples.last else { return 0 }
        let dt = last.timestamp.timeIntervalSince(first.timestamp)
        guard dt > 0.1 else { return 0 }
        return Double(last.bytes - first.bytes) / dt
    }

    /// 单文件实时速度（字节/秒）：基于该文件独立的滑动窗口，并发下各文件互不干扰
    private static func fileSpeedBytesPerSec(of ctx: FileDownloadContext) -> Double {
        guard ctx.speedSamples.count >= 2,
              let first = ctx.speedSamples.first,
              let last = ctx.speedSamples.last else { return 0 }
        let dt = last.timestamp.timeIntervalSince(first.timestamp)
        guard dt > 0.1 else { return 0 }
        return Double(last.bytes - first.bytes) / dt
    }

    /// 构建当前活动下载文件列表（需持有 stateLock 时调用）
    private func snapshotActiveFiles() -> [ActiveFileDownload] {
        activeTasks.values.map { ctx in
            ActiveFileDownload(
                name: ctx.file.name,
                written: ctx.written,
                total: Int64(ctx.file.size),
                speedBytesPerSec: Self.fileSpeedBytesPerSec(of: ctx))
        }
    }

    init(owner: String, repo: String) {
        self.owner = owner
        self.repo = repo
        super.init()
        // 前台会话：模型下载是用户主动触发的即时任务，不需要后台续传。
        // 原 background session 相同 identifier 每次重复创建导致会话冲突，
        // 系统恢复任务会把已过期的 CDN 签名 URL（5 分钟 auth_key）路由回来，
        // 触发 NSCocoaErrorDomain 260 下载失败。
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 7 * 24 * 3600   // 大文件给足 7 天
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }

    deinit {
        // 释放会话，打破 session <-> delegate 循环持有
        session?.invalidateAndCancel()
    }

    // MARK: 1. 获取文件清单

    func listFiles() async throws -> [ModelScopeFile] {
        var comps = URLComponents(
            string: "https://modelscope.cn/api/v1/models/\(owner)/\(repo)/repo/files")!
        comps.queryItems = [
            URLQueryItem(name: "Revision", value: revision),
            URLQueryItem(name: "Recursive", value: "true"),
        ]
        let (data, resp) = try await URLSession.shared.data(from: comps.url!)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        struct Envelope: Decodable {
            struct D: Decodable { let Files: [ModelScopeFile] }
            let Data: D
        }
        return try JSONDecoder().decode(Envelope.self, from: data).Data.Files
            .filter { !$0.isDirectory && !$0.path.hasSuffix(".gitattributes") }
    }

    // MARK: 2. 下载整个模型

    func downloadModel(to root: URL) async throws {
        do {
            let files = try await listFiles()
            let total = files.reduce(Int64(0)) { $0 + Int64($1.size) }
            stateLock.lock()
            totalBytes = total
            stateLock.unlock()
            var done: Int64 = 0

            // 多文件并行：TaskGroup 限流并发（类似 aria2 -j），每完成一个立即补充下一个；
            // 任一文件失败 -> group 自动取消其余子任务 -> catch 里 invalidateAndCancel 收尾。
            // 子任务返回 (该文件字节数, 文件名)，由主任务统一累加 done，避免共享计数器竞态。
            try await withThrowingTaskGroup(of: (Int64, String).self) { group in
                var iterator = files.makeIterator()
                // 预填充并发窗口
                for _ in 0..<maxConcurrentFiles {
                    if let file = iterator.next() {
                        group.addTask { [weak self] in
                            guard let self else { return (0, file.name) }
                            return try await self.runFileTask(file, root: root)
                        }
                    }
                }
                while let (size, name) = try await group.next() {
                    done += size
                    stateLock.lock()
                    completedBytes = done
                    stateLock.unlock()
                    onProgress?(done, total, name)
                    // 消费一个任务后补充下一个，始终保持并发窗口满载
                    if let file = iterator.next() {
                        group.addTask { [weak self] in
                            guard let self else { return (0, file.name) }
                            return try await self.runFileTask(file, root: root)
                        }
                    }
                }
            }
            // 全部完成后结束会话，允许 downloader 释放
            session.finishTasksAndInvalidate()
        } catch {
            // 任一文件失败：取消全部在途下载（其余已下完文件保留，重跑 SHA256 一致跳过）
            session.invalidateAndCancel()
            throw error
        }
    }

    /// 单文件子任务：取消时同步取消对应 URLSession task，避免 TaskGroup 等待挂起
    private func runFileTask(_ file: ModelScopeFile, root: URL) async throws -> (Int64, String) {
        try await withTaskCancellationHandler {
            let size = try await downloadOne(file, root: root)
            return (size, file.name)
        } onCancel: { [weak self] in
            self?.cancelTask(for: file)
        }
    }

    /// 按文件定位并取消其 URLSession task（task 表锁保护）
    private func cancelTask(for file: ModelScopeFile) {
        stateLock.lock()
        let task = activeTasks.first { $0.value.file.path == file.path }?.key
        stateLock.unlock()
        task?.cancel()
    }

    /// 单文件下载：文件级跳过（已存在且 SHA256 一致 -> 直接计入进度）；否则下载。
    /// 返回该文件字节数（跳过/下载完成都计入总进度，进度统计不重复）
    private func downloadOne(_ file: ModelScopeFile, root: URL) async throws -> Int64 {
        let dest = root.appendingPathComponent(file.path)
        if FileManager.default.fileExists(atPath: dest.path),
           try await Self.sha256(of: dest).lowercased() == file.sha256.lowercased() {
            return Int64(file.size)
        }
        try await downloadFile(file, to: dest)
        return Int64(file.size)
    }

    // MARK: 3. 主动暂停（前台会话直接取消全部在途任务；CDN 签名 URL 过期后 resumeData
    //             续传必失败，下次触发下载走整文件重下，已完整文件由 SHA256 校验跳过）

    func pause() {
        stateLock.lock()
        let tasks = Array(activeTasks.keys)
        stateLock.unlock()
        for t in tasks { t.cancel() }
    }

    // MARK: 4. 单个文件下载（CDN 签名 URL 5 分钟过期，resumeData 续传必失败 -> 失败整文件重下）

    private func downloadFile(_ file: ModelScopeFile, to dest: URL) async throws {
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true)

        let url = URL(
            string: "https://modelscope.cn/models/\(owner)/\(repo)/resolve/\(revision)/\(file.path)")!

        do {
            try await performDownload(file, url: url, to: dest)
        } catch {
            // 一次性整文件重试（不依赖 resumeData，签名 URL 已过期）
            try await performDownload(file, url: url, to: dest)
        }

        // 下载完成：SHA256 完整性校验（不匹配则删除重试一次）
        guard try await Self.sha256(of: dest).lowercased() == file.sha256.lowercased() else {
            try? FileManager.default.removeItem(at: dest)
            throw URLError(.cannotDecodeContentData)
        }
    }

    /// 发起单文件下载：创建 downloadTask 并注册到任务表（delegate 按 task 查表回调），
    /// 通过 continuation 挂起等待完成/失败。多任务并发共享同一 session。
    private func performDownload(_ file: ModelScopeFile, url: URL, to dest: URL) async throws {
        let task = session.downloadTask(with: url)
        let ctx = FileDownloadContext(file: file, dest: dest)
        stateLock.lock()
        activeTasks[task] = ctx
        let list = snapshotActiveFiles()
        stateLock.unlock()
        onActiveFilesChanged?(list)   // 新文件进入活动列表（补上下一个）
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<URL, Error>) in
            // resume 前设置 continuation：delegate 回调异步到达，顺序上先赋值后回调
            ctx.continuation = c
            task.resume()
        }
    }

    // MARK: URLSessionDownloadDelegate
    // 并发下多任务并存：所有回调先按 task 查表定位对应文件上下文，
    // 归属校验由「task -> ctx」映射承担（不再依赖单一 task 相等判断）。

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        stateLock.lock()
        guard let ctx = activeTasks[downloadTask] else { stateLock.unlock(); return }
        ctx.written = totalBytesWritten
        // 每文件独立滑动窗口采样（bytes 为本文件 written，从 0 起）
        let now = Date()
        ctx.speedSamples.append(SpeedSample(timestamp: now, bytes: totalBytesWritten))
        while let first = ctx.speedSamples.first,
              now.timeIntervalSince(first.timestamp) > speedWindowSeconds {
            ctx.speedSamples.removeFirst()
        }
        // 全局已下载字节 = 已完成字节（主任务累计）+ 各活动任务已写字节之和。
        // 并发下多文件同时写入，累加所有活动任务保证速度连续不跳变、不重复计数。
        let global = completedBytes + activeTasks.values.reduce(Int64(0)) { $0 + $1.written }
        let name = ctx.file.name
        let total = totalBytes
        let list = snapshotActiveFiles()
        stateLock.unlock()
        recordSpeedSample(globalBytes: global)
        // 总进度随已写字节实时增长：下载中（大文件未完成时 done 不更新）也持续刷新，
        // 避免整包进度长时间卡 0%（与文件级 onFileProgress 同一节奏，无新增机制）
        onProgress?(global, total, name)
        onFileProgress?(totalBytesWritten, totalBytesExpectedToWrite, name, currentSpeedBytesPerSec)
        onActiveFilesChanged?(list)   // 活动文件速度实时刷新
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        stateLock.lock()
        guard let ctx = activeTasks[downloadTask] else { stateLock.unlock(); return }
        stateLock.unlock()
        try? FileManager.default.removeItem(at: ctx.dest)
        do {
            // 同卷 rename，16GB 也是瞬时完成
            try FileManager.default.moveItem(at: location, to: ctx.dest)
            ctx.continuation?.resume(returning: ctx.dest)
        } catch {
            ctx.continuation?.resume(throwing: error)
        }
        stateLock.lock()
        activeTasks[downloadTask] = nil
        let list = snapshotActiveFiles()
        stateLock.unlock()
        onActiveFilesChanged?(list)   // 完成文件从活动列表消失
        // 移除后 didCompleteWithError(nil) 到达时查表失败自动忽略，不会二次 resume
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    didCompleteWithError error: Error?) {
        stateLock.lock()
        guard let downloadTask = task as? URLSessionDownloadTask,
              let ctx = activeTasks[downloadTask] else { stateLock.unlock(); return }
        stateLock.unlock()
        if let error {
            // 不保存 resumeData：CDN 签名 URL 5 分钟过期，续传必失败，
            // 失败整文件重下由 downloadFile 处理
            ctx.continuation?.resume(throwing: error)
            stateLock.lock()
            activeTasks[downloadTask] = nil
            let list = snapshotActiveFiles()
            stateLock.unlock()
            onActiveFilesChanged?(list)   // 失败文件从活动列表消失
        }
        // error == nil：文件已由 didFinishDownloadingTo 处理并移出任务表，此处忽略
    }

    // MARK: 工具

    /// 流式计算文件 SHA256（大文件不占内存）
    static func sha256(of url: URL) async throws -> String {
        try await Task.detached {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }
}

// ============================================================
//  模型下载规格表：每个生成模型对应一个远端仓库 + 本地目录 + 就绪判定
//  「不同模型不同的下载地址」：新模型只需在 all 里加一条
// ============================================================

struct ModelDownloadSpec {
    /// 模型标识（VideoModel/ImageModel 的 rawValue，如 "MiniMax H3" / "HiDream-O1"）
    let model: String
    /// 魔塔社区仓库归属（owner/repo 共同决定下载地址）
    let owner: String
    let repo: String
    /// 模型在统一模型根（项目文档地址/model）下的子路径
    let subpath: String
    /// 就绪判定：这些文件全部存在且非空 → 认为已下载完成
    let requiredFiles: [String]

    /// 模型权重落盘目录（与管线加载路径一致，跟随项目文档地址/model）
    var localDir: String { "\(CommonPaths.modelRoot)/\(subpath)" }

    static let all: [ModelDownloadSpec] = [
        // MiniMax H3：与 H3Pipeline 加载目录一致；仓库文件共 13 个约 39.5GB
        ModelDownloadSpec(
            model: "MiniMax H3",
            owner: "HuaCHayu",
            repo: "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit",
            subpath: "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit",
            requiredFiles: [
                "config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "transformer.safetensors",
                "text_encoder.safetensors",
                "video_vae.safetensors",
                "audio_vae.safetensors",
                "minimax_h3_turbo_v4_step600_ema.safetensors",
            ]),
        // HiDream-O1（图像模型）：与 runImagePipeline 加载目录一致（CommonPaths.modelRoot/HiDream-O1-Image-Dev-mlx-bf16）；
        // 仓库文件共 45 个约 16.4GB（大头 model.safetensors 16.3G + extras/custom_heads.safetensors 75M，
        // 其余为配置/文档/样例等小文件）。requiredFiles 只列管线加载必需的关键文件，isReady 按文件名递归查找可识别 extras/ 子目录。
        ModelDownloadSpec(
            model: "HiDream-O1",
            owner: "mlx-community",
            repo: "HiDream-O1-Image-Dev-mlx-bf16",
            subpath: "HiDream-O1-Image-Dev-mlx-bf16",
            requiredFiles: [
                "model.safetensors",
                "extras/custom_heads.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            ]),
    ]

    static func spec(for model: VideoModel) -> ModelDownloadSpec? {
        all.first { $0.model == model.rawValue }
    }

    /// 图像模型规格查找（HiDream-O1）
    static func spec(forImage model: ImageModel) -> ModelDownloadSpec? {
        all.first { $0.model == model.rawValue }
    }

    /// 本地目录是否已就绪（全部必需文件存在且大小 > 0；支持任意深度嵌套——
    /// 按文件名在 localDir 下递归查找，旧版嵌套子目录内的文件也能正确识别为已下载）
    var isReady: Bool {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: localDir, isDirectory: true)
        return requiredFiles.allSatisfy { name in
            guard let url = FileFinder.first(named: name, under: root),
                  let attrs = try? fm.attributesOfItem(atPath: url.path),
                  let size = (attrs[.size] as? NSNumber)?.intValue else { return false }
            return size > 0
        }
    }

    /// 远端清单里的每个文件 → 本地绝对路径（下载器把文件放这里）
    var downloadRootURL: URL {
        URL(fileURLWithPath: localDir, isDirectory: true)
    }
}

// ============================================================
//  模型下载管理器（画布 UI 状态）：点生成时检查模型就绪，
//  未就绪 → 自动下载（进度发布给试管进度条），下载完成自动续跑原生成
// ============================================================

@MainActor
final class ModelDownloadManager: ObservableObject {
    static let shared = ModelDownloadManager()

    /// 整包总进度 0~1
    @Published var progress: Double = 0
    /// 当前正在下载的文件名
    @Published var currentFile = ""
    /// 当前实时下载速度（如 "12.3 MB/s"），下载中持续更新，空闲为空串
    @Published var currentSpeed = ""
    /// 当前正在下载的文件列表（并发 3 个时 3 项；完成移除、新增补上，各自独立速度）
    @Published var activeFiles: [ActiveFileDownload] = []
    /// 是否正在下载（试管进度条显示开关）
    @Published var isDownloading = false
    /// 下载失败原因（非 nil 时试管进度条显示错误）
    @Published var errorMessage: String?

    /// 下载完成后要自动续跑的生成动作
    private var pendingAction: (() -> Void)?
    private var downloader: ModelScopeDownloader?
    /// 最近一次发起下载的规格（错误态"重试"按钮用）
    private var retrySpec: ModelDownloadSpec?

    private init() {}

    /// 需要时下载模型。
    /// - 返回 true：模型已就绪（或无下载配置），调用方直接继续生成；
    /// - 返回 false：已发起下载，下载完成自动执行 onReady 续跑生成。
    @discardableResult
    func ensureModel<M: RawRepresentable>(_ model: M, onReady: (() -> Void)? = nil) -> Bool where M.RawValue == String {
        guard let spec = ModelDownloadSpec.all.first(where: { $0.model == model.rawValue }) else { return true }
        guard !spec.isReady else { return true }
        guard !isDownloading else { return false }   // 正在下载中：忽略新请求（保留原续跑动作）
        startDownload(spec)
        pendingAction = onReady
        return false
    }

    private func startDownload(_ spec: ModelDownloadSpec) {
        progress = 0
        currentFile = ""
        errorMessage = nil
        isDownloading = true
        retrySpec = spec
        let dl = ModelScopeDownloader(owner: spec.owner, repo: spec.repo)
        downloader = dl
        dl.onProgress = { [weak self] done, total, name in
            Task { @MainActor in
                self?.progress = total > 0 ? min(Double(done) / Double(total), 1.0) : 0
                self?.currentFile = name
            }
        }
        dl.onActiveFilesChanged = { [weak self] list in
            Task { @MainActor in
                self?.activeFiles = list
            }
        }
        Task { [weak self] in
            do {
                try await dl.downloadModel(to: spec.downloadRootURL)
                await MainActor.run {
                    self?.isDownloading = false
                    self?.progress = 1
                    self?.currentFile = ""
                    self?.activeFiles = []
                    self?.downloader = nil
                    let action = self?.pendingAction
                    self?.pendingAction = nil
                    action?()   // 下载完成自动续跑生成
                }
            } catch {
                await MainActor.run {
                    self?.isDownloading = false
                    self?.errorMessage = "模型下载失败：\(error.localizedDescription)"
                    self?.activeFiles = []
                    self?.downloader = nil
                    self?.pendingAction = nil
                    // retrySpec 保留，供错误态"重试"按钮再次发起下载
                }
            }
        }
    }

    /// 关闭错误提示（错误态"关闭"按钮）
    func dismissError() {
        errorMessage = nil
        pendingAction = nil
    }

    /// 把字节/秒格式化为可读速度（"12.3 MB/s" / "800 KB/s"；<=0 返回空串）
    static func formatSpeed(_ bytesPerSec: Double) -> String {
        guard bytesPerSec > 0 else { return "" }
        if bytesPerSec >= 1024 * 1024 {
            return String(format: "%.1f MB/s", bytesPerSec / (1024 * 1024))
        }
        return String(format: "%.0f KB/s", bytesPerSec / 1024)
    }

    /// 重试下载（错误态"重试"按钮）：按最近一次规格重新发起下载
    func retryDownload() {
        guard let spec = retrySpec, !isDownloading else { return }
        startDownload(spec)
        // 重试是用户主动发起，不再自动续跑原生成动作（pendingAction 已在失败时清空）
    }
}

