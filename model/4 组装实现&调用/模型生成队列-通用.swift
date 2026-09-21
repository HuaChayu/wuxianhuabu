//
//  生成队列.swift
//  无限画布
//
//  生成队列管理器：多次点击生成入队、同类型聚簇排序（避免异模型来回卸载）、
//  串行执行、每任务可取消（排队中直接标记；生成中通过管线检查点中断采样）。
//

import SwiftUI
import Combine
// setenv/unsetenv（SelfLift 二阶段解耦环境变量）
import Darwin
// H3→LTX latent 直通通道需读取 MLXArray（bridge.ltxHalfLatent 的形状打印）
import MLX

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
    /// 条件图片（ltx2.5：首尾帧最多 2 张；MiniMax H3：多参考最多 9 张）
    var imagePaths: [String] = []
    /// 参考视频（仅 MiniMax H3 支持，最多 3 个）
    var videoPaths: [String] = []
    /// 参考音频（ltx2.5 只用第 1 个；MiniMax H3 最多 3 个）
    var audioPaths: [String] = []
    /// 单个音频（ltx2.5 管线入参语义）
    var audioPath: String? { audioPaths.first }
    var videoWidth: Int = 0
    var videoHeight: Int = 0
    var duration: VideoDuration = .fiveSeconds
    /// 视频生成模型（kind == .video 时有效；runPipeline 按此分派管线）
    var model: VideoModel = .ltx25Distill

    // ★ 尾帧延续（.h3cc）链路（2026-09-20 重装）：由 UI 构造时透传；默认值保证旧构造点零回归
    /// 本视频节点是否开启尾帧延续落盘（= 节点数据模型.tailFrameEnabled）
    var h3TailFrameEnabled: Bool = true
    /// 前置延续源节点 ID（连接顺序第一条入边 from.type == .video && from.tailFrameEnabled；nil = 无前置）
    var h3ChainSourceID: UUID? = nil
    /// 前置延续源分支 ID（多分支输出时区分，可选）
    var h3ChainBranchID: String? = nil

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
    /// - 条件通路：imagePaths 恰好 2 张且无视频/音频参考 → fl2va 首尾帧；
    ///   其余（1 张、3~9 张，或带视频/音频参考）→ ref2va 多参考图通路（首/尾帧入参被忽略）
    /// - 参考视频/音频：条件层已放开收集，管线侧编码尚未接入 → 明确日志提示后忽略
    /// - 分辨率规则（与 LTX 同口径，读偏好设置「第二阶段」开关 AppSettings.videoUseStage2）：
    ///   开 → 输入目标 ÷2（32 对齐）跑 H3 turbo 直出（stage1 半清，含 H3 自产音轨），
    ///        随后走 LTX 通用升频×2 + Stage2 3 步 refine（像素桥，产物音轨沿用半清视频，首尾帧钉入）；
    ///   关 → 不除2、不升频、不跑像素桥 refine：目标分辨率单遍直出。
    /// - Apple 超分通道（第三套二采）与上述阶段2 开关**解耦**：它只消费 stage1 的内存像素
    ///   （H3Stage2MemoryBridge），不需要除2生成、也不需要像素桥升频/refine。
    ///   故阶段2 关 + Apple 超分开 → 目标分辨率直出后单独做逐帧 Apple 超分（本次改造引入的组合）。
    /// - 时长：duration.numFrames → alignFrameCount → videoLatentT（latentT）
    /// - 产物：mp4 写入 output/视频，回传路径交给 onTaskResult → attachGeneratedVideo
    nonisolated private func runH3VideoPipeline(task: QueueTask) async -> String? {
        // 改造 B：二采无提示词 —— 不再传保真引导词给 stage2（原 stage2RefinePrompt 常量已删除）；
        // 二采 refine 的文本条件统一为完全静态空条件（loadStaticEmptyTextCond：常量读张量 / 缺失全零兜底，
        // 零模型权重调用，不加载 Gemma/connector）；
        // 画面由 stage1 音轨 IC（frozen_a 锁口型）与画面结构/guide/首尾帧参考控制，不重画内容。
        if task.imagePaths.isEmpty {
            pipelineLog("H3 视频生成：无条件图，本次走文本生成（纯文本条件，无 keyframe/参考图；.h3cc 命中与否由下游闸门决定）")
        }
        // 参考视频/音频：UI 条件层已放开收集，但管线侧编码通路未接入 → 明确提示，不静默忽略
        if !task.videoPaths.isEmpty {
            pipelineLog("H3 视频生成：收到 \(task.videoPaths.count) 个参考视频（上限 \(H3Const.maxRefVideos)），当前管线未接入视频参考编码，本次忽略")
        }
        if !task.audioPaths.isEmpty {
            pipelineLog("H3 视频生成：收到 \(task.audioPaths.count) 个参考音频（上限 \(H3Const.maxRefAudios)），当前管线未接入音频参考编码，本次忽略")
        }
        // 多参考（ref2va）通路判定：非「恰好 2 张图片、无视频/音频参考」的纯首尾帧场景 → 走多参考
        // ★ 2026-09-18 fl2va→refs 开关：NA_H3_FL2VA_AS_REFS=1 时「恰好 2 张首尾帧」也走 ref2va 通路
        //   （首尾帧作为 2 个软参考块 + 视觉块，语言固定首尾）。背景重影根因是 keyframes 时间硬锚 +
        //   低清/高清双网格竞争（h3_62/h3_65），refs 软参考无端点锚定（h3_64 已验证无重影），
        //   且高清条件行直接用原生全分辨率参考块，顺带消除低清提升导致的"首尾毛玻璃"（h3_65）。
        // ★ 2026-09-18 fl2va→refs 开关：默认开启（UI 直跑即生效）；设 NA_H3_FL2VA_AS_REFS=0 可回退旧 fl2va
        let fl2vaAsRefs = (ProcessInfo.processInfo.environment["NA_H3_FL2VA_AS_REFS"].flatMap { Int($0) } ?? 1) > 0
        // 开关值显式写入 env：默认开启时用户未设变量，管线层也要能读到（用于首/尾帧文本标签）
        if fl2vaAsRefs { setenv("NA_H3_FL2VA_AS_REFS", "1", 1) } else { unsetenv("NA_H3_FL2VA_AS_REFS") }
        let useRef2VA = !task.videoPaths.isEmpty || !task.audioPaths.isEmpty || task.imagePaths.count != 2 || fl2vaAsRefs
        // ref2va 通路下首/尾帧入参被忽略，仅作占位；图片不足 2 张时用首张补位
        // ★ 2026-09-20 续接空图安全取值：纯续接链本节点无图条件，空串占位（管线侧按续接空图跳过编码）
        let firstPath = task.imagePaths.isEmpty ? "" : task.imagePaths[0]
        let lastPath = task.imagePaths.count >= 2 ? task.imagePaths[1] : firstPath
        let alignedFrames = H3Const.alignFrameCount(UInt32(task.duration.numFrames))
        let latentT = H3Const.videoLatentT(frameCount: alignedFrames)
        let outDir = outputVideoDirURL.path
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        let baseName = nextAssetName(prefix: "h3")

        // ★ 2026-09-17：H3 SelfLift 内部已有 ×0.5 除2（低清段自己半清自己高清），外部不再 ÷2。
        //   阶段2 开启时同样直出全清目标尺寸：H3 全清 latent → H3-to-LTX-Latent-Adapter 转域
        //   → LTX 二采（runLTXStage2RefineOnLatent 收到全清网格，不再升频×2、不再跑解耦低清段）。
        //   Apple 超分通道只要 stage1 的内存像素，同样按目标分辨率直出。
        // ★ 2026-09-21 关闭 H3→LTX 二采（阶段2 / 像素桥 / adapter 直通）：
        //   当前主力链路 = H3 单遍直出 + H3 SelfLift（第三分支），不再使用 LTX 二采。
        //   下方强制 false（无视偏好设置 videoUseStage2），LTX 像素桥 / CQ/IC /
        //   H3→LTX adapter 直通 / lowOnly 分工随之全部短路；Apple 超分通道与阶段2
        //   解耦，不受影响。恢复：改回 AppSettings.shared.videoUseStage2 即可。
        let stage2On = false // 原：AppSettings.shared.videoUseStage2（2026-09-21 强制关闭）
        let appleSROn = appleSRSecondPassEnabled()
        let needMemoryBridge = stage2On || appleSROn   // 两套二采都吃 stage1 内存像素
        let genW = task.videoWidth
        let genH = task.videoHeight
        // 方案 C：只要还有二采（阶段2 或 Apple 超分），stage1 落盘就仅作“给用户看”的预览 + 音轨源
        //（二采像素已走内存直通，不落盘）；两者都关：单遍直出即最终产物。
        // 2026-09-19：H3 一采统一改 .mov + ProRes 422（直出与 SelfLift 分支不再接 LTX 二采时
        // 一采即最终产物，ProRes 10bit 优于 h264 8bit；有二采时同步升级为高质量预览/音轨源）。
        let stage1IsPreview = stage2On || appleSROn
        let genPath = "\(outDir)/\(baseName)\(stage1IsPreview ? "_stage1_preview" : "").mov"
        let condSummary: String
        if useRef2VA {
            if fl2vaAsRefs && task.videoPaths.isEmpty && task.audioPaths.isEmpty && task.imagePaths.count == 2 {
                condSummary = "首/尾帧作参考图 ×2（fl2va→refs 软参考 + 视觉块，语言固定首尾）"
            } else {
                condSummary = "多参考 \(task.imagePaths.count) 张图（ref2va）"
            }
        } else {
            condSummary = "首/尾帧 \(task.imagePaths[0]) / \(task.imagePaths[1])（fl2va）"
        }
        // ★ H3 一采第一阶段总步数 N：单一取自偏好设置「模型管理 → H3 一采设置」滑杆（4–12，默认 6）。
        // 该值经 generateVideo(steps:) → sigmaSchedule(steps:) → N = sigmas.count - 1 进入一采链路；
        // SelfLift 第三分支的 transition_step 不写死，由官方 75% 规则随 N 推导（ts = clamp(floor(0.75N),1,N-1)）。
        let h3Stage1Steps = AppSettings.shared.h3Stage1Steps
        let h3Stage1Ts = SelfLiftScheduleSplit.transitionStep(forTotalSteps: h3Stage1Steps)
        // ★ SelfLift 第三分支开关：同一偏好设置页「H3 一采设置」的开关，透传给管线（不改管线默认行为）
        let h3SelfLiftOn = AppSettings.shared.h3SelfLiftEnabled
        // ★ IC 链路 lowOnly（2026-09-18 架构纠正）：「H3 低分 + LTX 高分」两个模型组成 lift——
        //   阶段2（IC/CQ 像素桥或 adapter 直通）开启且无 Apple 超分竞争时，H3 一采只跑低分半清段，
        //   LTX 二采负责升频×2 + 高清精修（fullResInput=false）；Apple 超分开启时保留 H3 全清直出
        //   （Apple SR 吃全清像素做独立放大），SelfLift 关闭时 lowOnly 无意义（自动走原单遍全清）。
        let h3LowOnly = h3SelfLiftOn && stage2On && !appleSROn
        // ★ lowOnly 分工下 LTX 二采不再跑自己的半清低清段（NA_PIX_SELFLIFT_DECOUPLE=0）：
        //   H3 已出低分半清 latent，LTX 低清段再以 σ0=0.909375 加噪重绘会把参考/主序列源
        //   漂移成 LTX 想象内容（同帧对比实测：人物替换/结构扭曲/色彩偏移）；
        //   关闭后 IC 参考回到 H3 原 latent（factor=2 官方语义），主序列升频直接进高清段。
        if h3LowOnly {
            setenv("NA_PIX_SELFLIFT_DECOUPLE", "0", 1)
            // ★ 音频条件打卡（2026-09-18）：lowOnly 二采开启 frozen_a 音频条件（LTX_IC_AUDIO=1），
            //   用源音轨锁口型（此前诊断：空文本 + 无音频 → 模型先验主导重画，亚洲人变西方面孔；
            //   音频条件给生成提供音素时序锚定）。注意 frozen_a 会扩大 Nv，需确认显存余量。
            setenv("LTX_IC_AUDIO", "1", 1)
        } else {
            unsetenv("NA_PIX_SELFLIFT_DECOUPLE")
            unsetenv("LTX_IC_AUDIO")
        }
        // ★ SelfLift 二阶段解耦采样开关（自研解耦调度，默认开）：开启时 setenv 三个解耦变量，
        //   管线 H3Pipeline 读到后绕开官方 75% 耦合调度，改走「低清独立 L NFE → 放大重加噪 →
        //   高清独立 3 NFE」；关闭时清掉变量走官方耦合调度。仅 h3SelfLiftOn=true 时有意义。
        let h3SelfLiftDecoupleOn = AppSettings.shared.h3SelfLiftDecouple
        if h3SelfLiftDecoupleOn {
            setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
            // 2026-09-18 步数配置 v5：跟随面板步数 N 动态拆分——低清 N-1 + 高清 1（固定最后一步高清）。
            // 输入 7 → 6+1、输入 4 → 3+1；总 NFE = N，与滑杆一致，不再写死 3+3。
            // LOW_STEPS=N-1 → 低清跑到曲线实际终点 σ_k=12/((N-1)+11)；低清终点即「过渡用的值」= σ_next，
            // 高清等距 1 步从 σ_next 一次直达 0（最后一步全分辨率收细节）。
            let slLowSteps = max(h3Stage1Steps - 1, 2)
            let slLowK = 12.0 / Double(slLowSteps + 11)
            setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_STEPS", "\(slLowSteps)", 1)
            setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", String(format: "%.4f", slLowK), 1)
            setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "1", 1)
        } else {
            unsetenv("NA_H3_SELFLIFT_DECOUPLE")
            unsetenv("NA_H3_SELFLIFT_DECOUPLE_LOW_STEPS")
            unsetenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K")
            unsetenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS")
        }
        // ★ CFG 引导（2026-09-18 重影根因修复，对齐官方 SelfLiftH3Sampler cfg=5.0）：
        //   实测 cfg=5.0 对 600 turbo LoRA（蒸馏模型，训练目标 cfg=1）过强：高对比度马赛克/过曝/
        //   网格噪点，画面崩坏；官方 cfg=5.0 面向原版 H3（非 turbo）。故默认注入 0=关闭（恢复
        //   单路正常画面）；机制保留，显式设 NA_H3_SELFLIFT_CFG=1.5~3.0 可实验性开启。
        setenv("NA_H3_SELFLIFT_CFG", "0", 1)
        // 日志描述：解耦开启时 ts 被绕开（固定低清 3 + 高清 3 = NFE 6），否则官方 75% ts
        let h3SlDesc = h3SelfLiftDecoupleOn
            ? "SelfLift 解耦（低清 \(max(h3Stage1Steps - 1, 2)) NFE + 高清 1 NFE = NFE \(max(h3Stage1Steps - 1, 2) + 1)，固定最后一步高清，无过渡）"
            : "SelfLift ts=\(h3Stage1Ts)"
        if stage2On {
            pipelineLog("H3 视频生成：目标直出 \(genW)×\(genH)（SelfLift 内部×0.5 除2，外部不再÷2）\(h3Stage1Steps) 步 turbo（\(h3SlDesc)），再接 Stage2 refine（第二阶段开，\(h3LowOnly ? "lowOnly 分工：H3 只出低分半清，LTX 二采升频×2 + 高清精修" : "全清 latent 直通、不再升频×2")；Apple 超分\(appleSROn ? "开" : "关")）@ \(alignedFrames) 帧（latentT=\(latentT)），条件：\(condSummary)")
        } else if appleSROn {
            pipelineLog("H3 视频生成：目标直出 \(genW)×\(genH) \(h3Stage1Steps) 步 turbo（\(h3SlDesc)；第二阶段关：不除2/不升频/不跑像素桥 refine），随后单独走 Apple 超分通道二采（与阶段2 解耦）@ \(alignedFrames) 帧（latentT=\(latentT)），条件：\(condSummary)")
        } else {
            pipelineLog("H3 视频生成：直出目标 \(genW)×\(genH) \(h3Stage1Steps) 步 turbo（\(h3SlDesc)），不除2/不升频/不二采（第二阶段关，Apple 超分关）@ \(alignedFrames) 帧（latentT=\(latentT)），条件：\(condSummary)")
        }
        do {
            // ★ 尾帧延续（.h3cc）消费侧（2026-09-20 重装）：前置开尾帧视频节点（连接顺序第一条入边
            //   from.type == .video && from.tailFrameEnabled）存在 .h3cc 时 load 读回尾段 latent，
            //   经指纹校验后作为 continuationSource 注入 generateVideo（真正参与采样，非空转）。
            //   未命中 / 校验失败 → nil，管线回退从头生成（零回归）。生成成功后再删被消费缓存。
            let contCacheRoot = "\(AppSettings.shared.canvasRootPath)/cache/h3-continuation"
            var continuationSource: H3ContinuationCache.Loaded? = nil
            if task.h3TailFrameEnabled, let srcID = task.h3ChainSourceID {
                let contFrameCount = UInt32(h3PlanTemporal(Int(latentT)).outputFrames)
                let contExpect = H3ContinuationCache.Expect(
                    modelKey: H3ContinuationCache.currentModelKey,
                    width: genW, height: genH,
                    steps: UInt32(h3Stage1Steps),
                    latentT: latentT,
                    frameCount: contFrameCount,
                    refCount: useRef2VA ? task.imagePaths.count : 0
                )
                continuationSource = H3ContinuationCache.load(
                    nodeID: srcID, branchID: task.h3ChainBranchID,
                    rootDir: contCacheRoot, expect: contExpect)
                if continuationSource != nil {
                    pipelineLog("H3 尾帧延续：命中前置节点 \(srcID.uuidString) 的 .h3cc，尾段 latent 将注入采样")
                } else {
                    pipelineLog("H3 尾帧延续：前置节点 \(srcID.uuidString) 无有效 .h3cc（不存在/指纹不匹配/损坏），本次从头生成")
                }
            }
            // ★ 2026-09-20 空图最终闸门：无图条件时若未命中有效 .h3cc，不取消——
            // 回退为文本生成（文生视频，纯文本条件，管线走零条件行注入）。
            if task.imagePaths.isEmpty, continuationSource == nil {
                pipelineLog("H3 视频生成：图空且 .h3cc 未命中，本次为文本生成（文生视频，无 latent 窗口续接）")
            }
            // 方案 C：只要有二采（阶段2 或 Apple 超分）就建内存直通桥（generateVideo 内部把 stage1
            // 解码像素交给它，不落盘；随后由二采通道按需消费）；stage1 落盘统一 ProRes 422 .mov
            //（2026-09-19：直出/无二采时一采即最终产物，h264 → ProRes 提升最终输出质量）
            let stage2Bridge = needMemoryBridge ? H3Stage2MemoryBridge() : nil
            // ★ H3→LTX latent 直通（H3-to-LTX-Latent-Adapter，可选）：适配器在 H3 VAE 解码之前
            //   拦截 clean latent 并映射为 LTX 归一化 latent，二采由此省去像素往返。
            //   权重缺失/加载失败/未启用 → nil，管线内部自动回退原像素桥（不影响出片）。
            let h3Adapter = loadH3ToLTXAdapterIfEnabled(stage2On: stage2On)
            let stage1Path = try await H3FL2VAPipeline.generateVideo(
                prompt: task.prompt,
                firstImagePath: firstPath,
                lastImagePath: lastPath,
                outPath: genPath,
                width: genW,
                height: genH,
                // ★ N 来自偏好设置「模型管理 → H3 一采设置」滑杆（4–12，默认 6），不再写死 6；
                //   管线内 N = sigmas.count - 1 = stage1SegmentCount，即 SelfLift 第三分支消费的总步数 N。
                steps: UInt32(h3Stage1Steps),
                latentT: latentT,
                scheduleStyle: .official,
                // ★ SelfLift 第三分支开关（见上）：true 时 stage1 走渐进采样
                selfLiftEnabled: h3SelfLiftOn,
                // ★ IC 链路 lowOnly：阶段2 无 Apple 超分竞争时，H3 一采只出低分半清 latent，
                //   高分（升频×2 + 精修）交由 LTX 二采（fullResInput=false 配套）。
                selfLiftLowOnly: h3LowOnly,
                log: { pipelineLog("[H3] \($0)") },
                proResOutput: true,
                stage2MemBridge: stage2Bridge,
                // 多参考（ref2va）：非纯首尾帧场景把全部合规图片作为多参考传入（首/尾帧入参被忽略）
                referenceImagePaths: useRef2VA ? task.imagePaths : [],
                // ★ H3→LTX latent 直通：h3Adapter 非空即启用（仅阶段2 开启时才可能非空）；
                //   是否连 H3 VAE 解码一并跳过，取决于 Apple 超分通道——它需要 stage1 内存像素，
                //   故 Apple 超分开启时保留解码（skip=false），只省 LTX 编码那一步。
                //   参数顺序须与函数声明一致（两者均位于签名末位）。
                h3ToLTXAdapter: h3Adapter,
                adapterSkipH3Decode: !appleSROn,
                // ★ 尾帧延续（.h3cc）：续接源三元组（前置 .h3cc load 结果，nil = 从头）、
                //   本节点落盘开关 / 缓存根 / 被消费前置 ID（生成成功后幂等删除）/ 本节点 ID。
                continuationSource: continuationSource,
                tailFrameEnabled: task.h3TailFrameEnabled,
                continuationCacheRoot: contCacheRoot,
                continuationConsumedSourceID: task.h3ChainSourceID,
                continuationNodeID: task.nodeID,
                continuationBranchID: task.h3ChainBranchID
            )
            pipelineLog("H3 视频生成完成（stage1）：\(stage1Path)")
            guard stage2On || appleSROn else {
                return stage1Path   // 阶段2 与 Apple 超分全关：stage1 直出即最终产物
            }
            // 第二阶段：像素桥（LTX 通用升频×2 + Stage2 3步），音轨沿用半清源视频
            // 改造 B：二采无提示词 —— refine 不再编码/传递任何文本条件（原“传保真型 stage2RefinePrompt
            // 而非 task.prompt”的语义已被空串常量取代：Gemma 空串编码与正常 prompt 同构、不注入创作指令）。
            // 改造 C（IC 官方模式）：二采只吃一采产出的低清视频作 in-context 参考——由 LTX VAE 在
            // 像素桥内把整段半清像素编码为 halfLatent，再逐帧 token 追加进 transformer 输入序列；
            // 不再引入任何外部参考图：既不做首/尾帧图满锁，也不做 image 来源的 guide 软引导。
            // 历史沿革（v1.1~v1.3 参考图路径已废弃）：v1.1 直接钉创作原图 → 整体 RGB 色边/内容漂移
            // （H3 实际画面与创作图有出入时外来参考会把精修往创作图拽）；v1.2 改自抽 stage1 半清首尾帧
            // → 放大到 fullPixel 后细节全是插值，首尾帧画质反被拉低；v1.3 复归原图。三者均不再启用。
            guard let bridge = stage2Bridge else {
                pipelineLog("H3 二采：内存桥为空，放弃二采")
                // 阶段2 关（仅 Apple 超分）时无 CQ/IC 可回退，保留 stage1 直出结果
                return stage2On ? nil : stage1Path
            }
            // 注意：adapter 跳解码模式（adapterSkipH3Decode=true）下 bridge.pixels 为 nil，
            // 但「latent 直通」分支不需要像素，故像素可用性校验下放到 Apple 超分/像素桥各自入口。
            // ★ 第三套二采通道（与下方 CQ/IC 二采互斥，本通道优先）：
            //   偏好设置「启用 Apple 超分二采（优先）」开启时，H3 一采出的内存像素优先走 Apple VideoToolbox
            //   超分（VTSuperResolutionScaler / VTFrameProcessor）。本通道内部完成：运行期能力探测 →
            //   模型资产准备（downloadRequired 时异步下载并带进度）→ 逐帧超分 → 编码落盘（复用 writeMp4 直写通道）
            //   → 复用源视频音轨；不落任何中间文件。
            //   任何不可用/失败（运行期不支持、模型未就绪、源尺寸越界、像素格式无交集、写盘格式不匹配、
            //   startSession 失败、任务取消等）统一返回 nil，随即回退下方原 CQ/IC 二采并打日志，绝不中断生成。
            //   LTX 原生管线、IC/CQ 老路径与内存桥语义均不受影响（本分支不命中时下方代码逐字不变）。
            //   ★ 与「阶段2」开关解耦：本通道只看 appleSRSecondPassEnabled()（videoUseAppleSR / LTX_APPLE_SR）。
            //   阶段2 关但本通道开时：stage1 已按目标分辨率直出，本通道产出即最终产物；失败则回退 stage1 直出
            //   （无 CQ/IC 像素桥可回退，因为像素桥属于阶段2 的升频 refine 链路）。
            //   环境变量 LTX_APPLE_SR=1/0 优先级高于设置项，可强制开/关本通道。
            //   （本通道必须吃 stage1 内存像素；adapter 跳解码模式下 pixels 为空，此处自然跳过。）
            if appleSROn, let stage2Pixels = bridge.pixels {
                pipelineLog("H3 二采分派：Apple 超分通道已启用（优先），先尝试第三套二采（倍率 \(AppSettings.shared.appleSRScaleFactor > 0 ? "×\(AppSettings.shared.appleSRScaleFactor)" : "×4（默认）")）")
                if let appleSRFinalPath = await appleSuperResolutionSecondPass(
                    pixels: stage2Pixels,
                    pixelWidth: bridge.width,
                    pixelHeight: bridge.height,
                    pixelFps: Double(bridge.fps),
                    sourceAudioVideoPath: stage1Path,
                    isCancelled: { task.cancelToken.isCancelled }) {
                    pipelineLog("H3 视频生成完成（Apple 超分最终产物）：\(appleSRFinalPath)")
                    return appleSRFinalPath
                }
                if task.cancelToken.isCancelled {
                    pipelineLog("H3 二采分派：任务已取消，不再回退 CQ/IC 二采")
                    return nil
                }
                guard stage2On else {
                    // 阶段2 关：像素桥（升频 refine）未启用 → 无 CQ/IC 可回退，保留 stage1 目标分辨率直出结果
                    pipelineLog("H3 二采分派：Apple 超分通道未产出（原因见上方 AppleSR 日志），且第二阶段关闭、无 CQ/IC 可回退 → 保留 stage1 直出结果：\(stage1Path)")
                    return stage1Path
                }
                pipelineLog("H3 二采分派：Apple 超分通道未产出（原因见上方 AppleSR 日志），回退原 CQ/IC 二采")
            }
            // ★ H3→LTX latent 直通二采（adapter 分支）：H3 clean latent 已由适配器映射为 LTX 归一化
            //   latent（bridge.ltxHalfLatent），此处直接交给二采的 latent 入口——不读像素、不跑 LTX VAE
            //   编码（H3 VAE 解码亦可能已跳过）；音轨取 adapter 模式单独落盘的 wav（bridge.audioTrackPath）。
            //   本分支优先于下方像素桥；未启用/未产出时原样落到像素桥路径（语义不变）。
            if stage2On, let adapterBridge = stage2Bridge, let adapterLatent = adapterBridge.ltxHalfLatent {
                pipelineLog("H3 二采分派：latent 直通通道（H3-to-LTX Adapter）优先，latent \(adapterLatent.shape)（\(h3LowOnly ? "lowOnly 半清直通：H3 只出低分，LTX 二采升频×2 + 高清 IC 精修（factor=2）" : "全清直通：H3 SelfLift 内部已升频，LTX 二采跳过升频×2/解耦低清段，IC 参考 factor=1 同格同位")）")
                if let finalPath = await ltxEnhanceExternalVideoWithStage2(
                    videoPath: stage1Path,
                    sourceAudioVideoPath: adapterBridge.audioTrackPath ?? stage1Path,
                    imagePaths: [],
                    isCancelled: { task.cancelToken.isCancelled },
                    icLoRAEnable: true,
                    cqEnhancerEnable: AppSettings.shared.videoUseCQEnhancer,
                    precomputedHalfLatent: adapterLatent,
                    precomputedFrameRate: Double(adapterBridge.fps),
                    fullResInput: !h3LowOnly) {
                    pipelineLog("H3 视频生成完成（latent 直通最终产物）：\(finalPath)")
                    return finalPath
                }
                pipelineLog("H3 二采分派：latent 直通未产出，回退原像素桥路径")
            }
            // 像素可用性与帧数下限仅像素桥（CQ/IC 的 LTX VAE 编码）需要，故从这里才开始校验：
            // adapter 跳解码模式下 pixels 为空且无 stage1 mp4 可回退，此时若 latent 直通也没产出，
            // 只能如实返回失败（不能返回并不存在的 stage1 文件）。
            guard let stage2Pixels = bridge.pixels else {
                pipelineLog("H3 二采：adapter 直通未产出且无 stage1 内存像素可回退，放弃二采")
                return nil
            }
            guard bridge.frameCount >= 9 else {
                pipelineLog("H3 像素桥：stage1 帧数不足（\(bridge.frameCount) < 9），跳过 CQ/IC 二采（stage1 预览保留：\(stage1Path)）")
                return stage1Path
            }
            guard let finalPath = await ltxEnhanceExternalVideoWithStage2Pixels(
                pixels: stage2Pixels,
                pixelWidth: bridge.width,
                pixelHeight: bridge.height,
                pixelFps: Double(bridge.fps),
                sourceAudioVideoPath: stage1Path,
                // 二采不引入任何参考图（改造 C：IC 官方模式只用一采视频自身作 in-context 参考），
                // 也不做首/尾帧软引导（官方 stage2 无此机制，故不再传 tailGuideMask）。
                imagePaths: [],
                isCancelled: { task.cancelToken.isCancelled },
                // 默认切 IC 官方模式（in-context 参考 + 官方 4 步 σ 档 + ancestral SDE）；
                // 如需回退旧非 IC refine 路径，设 LTX_IC_LORA=0。
                // 注：参数顺序须与函数声明一致（isCancelled 在 icLoRAEnable 之前）。
                icLoRAEnable: true,
                // ★ 「第二阶段·CQ 清晰度增强」通道（新增分支）：偏好设置「二采改用 CQ 清晰度增强」开启时，
                //   二采改走官方 CQ Video Enhancer LoRA（σ0=1.0 / 9 段 / euler_ancestral，权重 strength=1.0），
                //   只对 H3 半清画面做清晰度增强，不换脸/不重绘构图；关闭时仍走原 IC 像素桥二采（行为不变）。
                //   CQ 与 IC 互斥、CQ 优先；LTX_CQ_ENHANCER=1/0 可强制覆盖本设置项。
                cqEnhancerEnable: AppSettings.shared.videoUseCQEnhancer,
                // ★ 输入几何随 lowOnly 联动：H3 一采只出低分半清 → LTX 二采升频×2 + 高清精修（false）；
                //   H3 全清直出（SelfLift 关闭 / Apple 超分竞争）→ 全清直通跳过升频（true）。
                fullResInput: !h3LowOnly) else {
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


// MARK: - H3→LTX latent 直通适配器（可选通道）

/// 按开关加载 H3→LTX latent 直通适配器；任何不满足/失败都返回 nil，管线自动回退原像素桥路径。
///
/// 开关优先级：环境变量 `LTX_H3_ADAPTER`（"1" 强制开 / "0" 强制关）> 设置项 `videoUseH3LTXAdapter`。
/// 仅在「阶段2 开启」时才有意义：该通道服务的是 LTX 二采（CQ 增强 / IC 像素桥），
/// Apple 超分通道吃的是 stage1 内存像素、不经 LTX，故与其并行时由调用点保留 H3 解码
/// （`adapterSkipH3Decode=false`），此时只省下 LTX VAE 编码一步。
/// 权重路径取设置项 `h3LTXAdapterPath`（默认 ~/Downloads/h3-fused/H3-to-LTX-Latent-Adapter.safetensors）。
private func loadH3ToLTXAdapterIfEnabled(stage2On: Bool) -> H3ToLTXLatentAdapter? {
    guard stage2On else { return nil }
    let env = ProcessInfo.processInfo.environment["LTX_H3_ADAPTER"]
    let enabled = (env == "1") || (env != "0" && AppSettings.shared.videoUseH3LTXAdapter)
    guard enabled else {
        pipelineLog("H3→LTX 适配器：未启用（LTX_H3_ADAPTER=\(env ?? "未设")，设置项 videoUseH3LTXAdapter=\(AppSettings.shared.videoUseH3LTXAdapter)），走原像素桥路径")
        return nil
    }
    let path = AppSettings.shared.h3LTXAdapterPath
    guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else {
        pipelineLog("H3→LTX 适配器：权重不存在，回退像素桥路径（\(path)）")
        return nil
    }
    do {
        let t0 = Date()
        let adapter = try H3ToLTXLatentAdapter.load(weightsURL: URL(fileURLWithPath: path))
        let sec = String(format: "%.1f", Date().timeIntervalSince(t0))
        pipelineLog("H3→LTX 适配器已加载（\(sec)s，dtype=\(adapter.weightDType)，权重 \(path)）")
        return adapter
    } catch {
        pipelineLog("H3→LTX 适配器加载失败，回退像素桥路径：\(error.localizedDescription)")
        return nil
    }
}
