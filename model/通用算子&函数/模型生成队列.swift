//
//  生成队列.swift
//  无限画布
//
//  生成队列管理器：多次点击生成入队、同类型聚簇排序（避免异模型来回卸载）、
//  串行执行、每任务可取消（排队中直接标记；生成中通过管线检查点中断采样）。
//

import SwiftUI
import Combine

// MARK: - 取消错误（管线检查点抛出）

enum GenerationCancelError: Error {
    case cancelled
}

// MARK: - 队列任务类型

/// 队列任务类型（与内存策略里的 GenerationTask 枚举区分开）
enum QueueTaskKind: String {
    case image = "图像"
    case video = "视频"
}

/// 队列任务状态
enum QueueTaskStatus: String {
    case pending = "排队中"
    case running = "生成中"
    case done = "已完成"
    case cancelled = "已取消"
    case failed = "失败"
}

/// 线程安全取消标志：管线在后台线程检查，UI 主线程置位
final class CancelToken: @unchecked Sendable {
    private let lock = NSLock()
    private var _cancelled = false

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return _cancelled
    }

    func cancel() {
        lock.lock()
        _cancelled = true
        lock.unlock()
    }
}

/// 单个生成任务（入队时收集好全部参数，轮到执行时直接跑管线）
struct QueueTask: Identifiable, Sendable {
    let id = UUID()
    let kind: QueueTaskKind
    let nodeID: UUID
    let prompt: String
    let createdAt = Date()

    // 图像参数（kind == .image 时有效）
    var imageWidth: Int = 0
    var imageHeight: Int = 0
    // 图像参考条件（编辑/多参考；来自上游图片节点连线，非空时管线走 hidreamGenerateEdit 编辑分支）
    var referencePaths: [String] = []

    // 视频参数（kind == .video 时有效）
    var imagePaths: [String] = []
    var audioPath: String? = nil
    var videoWidth: Int = 0
    var videoHeight: Int = 0
    var duration: VideoDuration = .fiveSeconds

    // 状态
    var status: QueueTaskStatus = .pending
    let cancelToken = CancelToken()

    /// 面板展示用摘要
    var summary: String {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return p.isEmpty ? "（无提示词）" : (p.count > 12 ? String(p.prefix(12)) + "…" : p)
    }
}

// MARK: - 队列管理器

/// 生成队列：串行执行（同一时刻只跑一个），同类型聚簇排序。
/// 设计动机：图像/视频用不同模型，来回切换会反复卸载/加载权重；
/// 把同类型任务排在一起，只切换一次模型，降低内存压力。
@MainActor
final class GenerationQueue: ObservableObject {
    static let shared = GenerationQueue()

    /// 队列全部任务（按聚簇顺序）
    @Published private(set) var tasks: [QueueTask] = []
    /// 当前正在执行的任务 id（nil = 空闲）
    @Published private(set) var currentTaskID: UUID?

    /// 任务执行完毕回调（主线程）：由 CanvasView 注入，把产物接入节点
    var onTaskResult: ((QueueTask, String?) -> Void)?
    /// 任务取消回调（主线程）：由 CanvasView 注入，画布据此清理取消节点的关联发光状态
    var onTaskCancelled: ((UUID) -> Void)?

    private var isRunning = false

    private init() {}

    // MARK: 入队

    /// 入队一个生成任务：追加 → 同类型聚簇排序 → 触发调度
    func enqueue(_ task: QueueTask) {
        tasks.append(task)
        resort()
        pump()
    }

    // MARK: 取消

    /// 取消任务：排队中直接移除；生成中置取消标志并从列表移除，
    /// 管线在采样检查点退出后由 finish 收尾（此时才允许跑下一个任务）。
    func cancel(_ taskID: UUID) {
        guard let idx = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        let t = tasks[idx]
        guard t.status == .pending || t.status == .running else { return }
        t.cancelToken.cancel()
        tasks.remove(at: idx)
        // 取消联动：通知画布清理该节点相关的发光线（指向它或从它出发的线）
        onTaskCancelled?(t.nodeID)
        // 空闲时取消排队任务可立即调度下一个；生成中则等管线在检查点退出后 finish 再泵
        if !isRunning {
            pump()
        }
    }

    /// 节点是否正在生成（发送按钮转圈用；排队中不转圈，允许继续入队）
    func isGenerating(nodeID: UUID) -> Bool {
        tasks.contains { $0.nodeID == nodeID && $0.status == .running }
    }

    /// 节点当前队列状态（节点内容区覆盖显示用）：无任务 / 已结束返回 nil
    func queueStatus(for nodeID: UUID) -> QueueTaskStatus? {
        tasks.first { $0.nodeID == nodeID && ($0.status == .pending || $0.status == .running) }?.status
    }

    /// 节点当前活跃任务 id（供节点覆盖层取消按钮使用）：pending/running 中第一个匹配任务；无则 nil
    func taskID(for nodeID: UUID) -> UUID? {
        tasks.first { $0.nodeID == nodeID && ($0.status == .pending || $0.status == .running) }?.id
    }

    // MARK: 聚簇排序

    /// 同类型聚簇：以当前执行（或队首排队）类型为锚，同类型块在前、异类型块在后；块内按加入顺序。
    /// 例：当前跑图像 → 图像任务全部排前面，视频任务排后面；当前跑视频则相反。
    private func resort() {
        let anchor = currentRunningKind()
            ?? tasks.first(where: { $0.status == .pending })?.kind
            ?? .image
        let sorted = tasks.enumerated()
            .sorted { a, b in
                let aActive = a.element.kind == anchor
                let bActive = b.element.kind == anchor
                if aActive != bActive { return aActive }
                return a.element.createdAt < b.element.createdAt
            }
            .map { $0.element }
        tasks = sorted
    }

    private func currentRunningKind() -> QueueTaskKind? {
        tasks.first(where: { $0.status == .running })?.kind
    }

    // MARK: 串行调度

    /// 泵：取第一个排队任务执行；同一时刻只跑一个（避免并发生成挤爆内存）
    private func pump() {
        guard !isRunning else { return }
        // 顺带清理已结束任务，保持队列只显示活跃 + 排队
        tasks.removeAll { $0.status == .done || $0.status == .cancelled || $0.status == .failed }
        guard let idx = tasks.firstIndex(where: { $0.status == .pending }) else { return }
        isRunning = true
        tasks[idx].status = .running
        currentTaskID = tasks[idx].id
        let task = tasks[idx]
        MonitorCenter.shared.taskStart(task.kind.rawValue)

        Task.detached(priority: .userInitiated) { [weak self] in
            let path = await self?.runPipeline(task)
            await self?.finish(task, outputPath: path)
        }
    }

    /// 后台执行管线（nonisolated：不能占用主线程跑 MLX 重计算）
    nonisolated private func runPipeline(_ task: QueueTask) async -> String? {
        switch task.kind {
        case .image:
            // 内存预算：加载前评估水位（HiDream backbone+扩散头 约 17G；已缓存传 0 不触发卸载）
            guard MemoryPolicy.ensureCapacity(
                for: HiDreamModelCache.shared.hasHiDream ? 0 : 17_000_000_000,
                task: .image, current: .sampling,
                width: task.imageWidth, height: task.imageHeight, frames: 0) else {
                debugLog("图像生成：内存不足，无法加载该模型（HiDream 约 17G），已取消运行")
                return nil
            }
            return await runImagePipeline(
                prompt: task.prompt,
                referencePaths: task.referencePaths,
                width: task.imageWidth,
                height: task.imageHeight,
                isCancelled: { task.cancelToken.isCancelled })
        case .video:
            // 内存预算：加载前评估水位（Gemma+connector 约 12G；均已缓存传 0 不触发卸载）
            guard MemoryPolicy.ensureCapacity(
                for: TextEncoderCache.shared.hasTextEncoder ? 0 : 12_000_000_000,
                task: .video, current: .encoding, next: .sampling,
                width: task.videoWidth, height: task.videoHeight, frames: task.duration.numFrames) else {
                debugLog("视频生成：内存不足，无法加载该模型（Gemma+connector 约 12G），已取消运行")
                return nil
            }
            return await runVideoPipeline(
                prompt: task.prompt, audioPath: task.audioPath, imagePaths: task.imagePaths,
                width: task.videoWidth, height: task.videoHeight, duration: task.duration,
                isCancelled: { task.cancelToken.isCancelled })
        }
    }

    /// 收尾（主线程，@MainActor 类方法）：更新状态、回传结果给 CanvasView、泵下一个
    private func finish(_ task: QueueTask, outputPath: String?) async {
        guard let idx = tasks.firstIndex(where: { $0.id == task.id }) else {
            // 任务已被取消移除（cancel 时从列表删除）：仍要收尾，允许下一个任务执行
            isRunning = false
            currentTaskID = nil
            MonitorCenter.shared.taskEnd(task.kind.rawValue)
            pump()
            return
        }
        tasks[idx].status = task.cancelToken.isCancelled
            ? .cancelled
            : (outputPath != nil ? .done : .failed)
        if tasks[idx].id == currentTaskID { currentTaskID = nil }
        MonitorCenter.shared.taskEnd(task.kind.rawValue)
        isRunning = false
        // 已取消的任务不回传产物（节点不接入）
        if !task.cancelToken.isCancelled {
            onTaskResult?(tasks[idx], outputPath)
        }
        pump()
    }
}
