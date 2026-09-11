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
    /// 视频生成模型（kind == .video 时有效；runPipeline 按此分派管线）
    var model: VideoModel = .ltx25Distill

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
            switch task.model {
            case .ltx25Distill:
                return await runVideoPipeline(
                    prompt: task.prompt, audioPath: task.audioPath, imagePaths: task.imagePaths,
                    width: task.videoWidth, height: task.videoHeight, duration: task.duration,
                    stage2Refine: AppSettings.shared.videoUseStage2,
                    isCancelled: { task.cancelToken.isCancelled })
            case .minimaxH3:
                return await runH3VideoPipeline(task: task)
            }
        }
    }

    /// MiniMax H3 视频分派：QueueTask → H3FL2VAPipeline
    /// - 首/尾帧：imagePaths[0] / imagePaths[1]（H3 恰好 2 张条件图，由 inputValidity 保证）
    /// - 分辨率规则（与 LTX 同口径，读偏好设置「第二阶段」开关 AppSettings.videoUseStage2）：
    ///   开 → 输入目标 ÷2（32 对齐）跑 H3 turbo 直出（stage1 半清，含 H3 自产音轨），
    ///        随后走 LTX 通用升频×2 + Stage2 3 步 refine（像素桥，产物音轨沿用半清视频，首尾帧钉入）；
    ///   关 → 不除2、不升频、不二采：目标分辨率单遍直出。
    /// - 时长：duration.numFrames → alignFrameCount → videoLatentT（latentT）
    /// - 产物：mp4 写入 output/视频，回传路径交给 onTaskResult → attachGeneratedVideo
    nonisolated private func runH3VideoPipeline(task: QueueTask) async -> String? {
        // 改造 B：二采无提示词 —— 不再传保真引导词给 stage2（原 stage2RefinePrompt 常量已删除）；
        // 二采 refine 的文本条件统一为完全静态空条件（loadStaticEmptyTextCond：常量读张量 / 缺失全零兜底，
        // 零模型权重调用，不加载 Gemma/connector）；
        // 画面由 stage1 音轨 IC（frozen_a 锁口型）与画面结构/guide/首尾帧参考控制，不重画内容。
        guard task.imagePaths.count >= 2 else {
            pipelineLog("H3 视频生成：条件图不足 2 张（当前 \(task.imagePaths.count)），取消运行")
            return nil
        }
        let alignedFrames = H3Const.alignFrameCount(UInt32(task.duration.numFrames))
        let latentT = H3Const.videoLatentT(frameCount: alignedFrames)
        let outDir = outputVideoDirURL.path
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let baseName = nextAssetName(prefix: "h3")

        // 第二阶段开：生成尺寸 = 目标 ÷2 并对齐 32（升频 ×2 后还原目标；与 LTX 相同策略）
        let stage2On = AppSettings.shared.videoUseStage2
        let genW = stage2On ? max(32, (task.videoWidth / 2 / 32) * 32) : task.videoWidth
        let genH = stage2On ? max(32, (task.videoHeight / 2 / 32) * 32) : task.videoHeight
        // 方案 C：二采开时 stage1 以 h264 最高质量 mp4 落盘，仅作“给用户看”的预览 + 音轨源
        //（二采像素已走内存直通，不落盘）；二采关：单遍直出维持 h264 (.mp4) 原样
        let genPath = "\(outDir)/\(baseName)\(stage2On ? "_stage1_preview" : "").mp4"
        if stage2On {
            pipelineLog("H3 视频生成：目标÷2=\(genW)×\(genH) 先 turbo 直出，再接像素桥升频×2+Stage2 refine（第二阶段开）@ \(alignedFrames) 帧（latentT=\(latentT)），首帧 \(task.imagePaths[0])，尾帧 \(task.imagePaths[1])")
        } else {
            pipelineLog("H3 视频生成：直出目标 \(genW)×\(genH) 6 步 turbo，不除2/不升频/不二采（第二阶段关）@ \(alignedFrames) 帧（latentT=\(latentT)），首帧 \(task.imagePaths[0])，尾帧 \(task.imagePaths[1])")
        }
        do {
            // 方案 C：stage2 开时创建内存直通桥（generateVideo 内部把 stage1 解码像素交给它，
            // 不落盘；随后由像素桥内存入口消费）；stage1 落盘统一 h264（不再 ProRes 中间件）
            let stage2Bridge = stage2On ? H3Stage2MemoryBridge() : nil
            let stage1Path = try await H3FL2VAPipeline.generateVideo(
                prompt: task.prompt,
                firstImagePath: task.imagePaths[0],
                lastImagePath: task.imagePaths[1],
                outPath: genPath,
                width: genW,
                height: genH,
                steps: 6,
                latentT: latentT,
                scheduleStyle: .official,
                log: { pipelineLog("[H3] \($0)") },
                proResOutput: false,
                stage2MemBridge: stage2Bridge
            )
            pipelineLog("H3 视频生成完成（stage1）：\(stage1Path)")
            guard stage2On else {
                return stage1Path
            }
            // 第二阶段：像素桥（LTX 通用升频×2 + Stage2 3步），音轨沿用半清源视频
            // 改造 B：二采无提示词 —— refine 不再编码/传递任何文本条件（原“传保真型 stage2RefinePrompt
            // 而非 task.prompt”的语义已被空串常量取代：Gemma 空串编码与正常 prompt 同构、不注入创作指令）。
            // 参考条件来源（v1.3）：直接用 H3 生成时传入的用户原图 task.imagePaths[0]/[1] 作 refine
            // 首/尾帧参考（stage1 本来就是拿这两张原图当条件的，内容一致且信息量无损）。
            // 历史沿革：v1.1 直接钉原图，当时出现整体 RGB 色边/内容漂移，判定根因是"H3 实际画面与
            // 创作图有出入时外来参考会把精修往创作图拽"；v1.2 因此改为自抽 stage1 首/尾帧作自引用参考。
            // 但自抽帧是半清（genW×genH）产物，像素桥会把它放大到 fullPixelWidth×fullPixelHeight 再
            // VAE 编码钉入，细节全是插值，首尾帧画质反而被拉低——低清帧作全清钉帧参考不可用。
            // 综合：默认走原图（v1.1 语义）；若需对照色边是否复发，PIX_REF_SOURCE=video 切回自抽路径。
            let refSource = ProcessInfo.processInfo.environment["PIX_REF_SOURCE"] ?? "image"
            var refineRefs: [String] = []
            if refSource == "video" {
                if let refV = readVideoFramesToBCFHW(videoPath: stage1Path), refV.frameCount >= 9 {
                    let refFirst = "\(outDir)/_pb_ref_\(baseName)_first.png"
                    let refLast = "\(outDir)/_pb_ref_\(baseName)_last.png"
                    h3WriteFramePNG(refV.pixels, frame: 0, path: refFirst)
                    h3WriteFramePNG(refV.pixels, frame: refV.frameCount - 1, path: refLast)
                    refineRefs = [refFirst, refLast]
                    pipelineLog("H3 像素桥：自抽 stage1 首/尾帧作 refine 参考（\(refV.width)×\(refV.height)，共 \(refV.frameCount) 帧）→ \(refFirst)")
                } else {
                    pipelineLog("H3 像素桥：stage1 首尾帧自抽失败，refine 将无参考条件（不推荐，会退回自由重画）")
                }
            } else {
                refineRefs = [task.imagePaths[0], task.imagePaths[1]]
                pipelineLog("H3 像素桥：直接用用户传入原图作 refine 首/尾帧参考（\(refineRefs.count) 张）→ \(refineRefs[0]) / \(refineRefs[1])")
            }
            guard let bridge = stage2Bridge,
                  let stage2Pixels = bridge.pixels, bridge.frameCount >= 9 else {
                pipelineLog("H3 像素桥：stage1 内存像素不可用（bridge 为空/帧数不足），放弃二采")
                return nil
            }
            guard let finalPath = await ltxEnhanceExternalVideoWithStage2Pixels(
                pixels: stage2Pixels,
                pixelWidth: bridge.width,
                pixelHeight: bridge.height,
                pixelFps: Double(bridge.fps),
                sourceAudioVideoPath: stage1Path,
                imagePaths: refineRefs,
                isCancelled: { task.cancelToken.isCancelled },
                // 尾帧软引导（对齐官方 keyframe-guide 语义，不再 100% 硬钉外部图）：
                // 默认 0.6，消除末段清晰度被高清参考图硬接管的接缝；复现旧硬钉行为设 PIX_TAIL_M=0。
                tailGuideMask: (Float(ProcessInfo.processInfo.environment["PIX_TAIL_M"] ?? "") ?? 0.6)) else {
                pipelineLog("H3 像素桥升频+二采失败（stage1 预览保留：\(stage1Path)）")
                return nil
            }
            pipelineLog("H3 视频生成完成（像素桥最终产物）：\(finalPath)")
            return finalPath
        } catch {
            pipelineLog("H3 视频生成失败：\(error.localizedDescription)")
            return nil
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

