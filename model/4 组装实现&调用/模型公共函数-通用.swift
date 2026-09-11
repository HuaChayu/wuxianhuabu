
//
//  公共函数.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/21.
//

import Foundation
import MLX
import AVFoundation
import CoreVideo
import CoreMedia
import CoreGraphics
import AppKit
import Darwin
import Accelerate

// MARK: - 内存打点收集（公共函数）

// 收集各阶段 MLX active/cache/peak 快照，不实时打印；
// 全部阶段收集完后调用 memProfileReport() 一次性输出全部结果。

struct MemProfilePoint {
    let label: String
    let active: Int
    let cache: Int
    let peak: Int
}

private var memProfilePoints: [MemProfilePoint] = []

/// 收集当前内存快照（不打印，全部完成后由 memProfileReport 统一输出）
func memProfilePoint(_ label: String) {
    let s = MLX.Memory.snapshot()
    memProfilePoints.append(MemProfilePoint(label: label, active: s.activeMemory, cache: s.cacheMemory, peak: s.peakMemory))
}

/// 一次性将所有打点结果 print 到控制台，并清空收集器
func memProfileReport() {
    guard !memProfilePoints.isEmpty else { return }
    let gb: (Int) -> String = { String(format: "%.1fG", Double($0) / Double(1 << 30)) }
    print("内存打点（\(memProfilePoints.count) 点）")
    for p in memProfilePoints {
        print("\(p.label) A\(gb(p.active)) C\(gb(p.cache)) P\(gb(p.peak))")
    }
    memProfilePoints.removeAll()
}

/// 即时内存打点：调用即打印（不依赖 memProfileReport 的批量收集/统一输出，
/// 卡死前也能留下现场）。直接 pipelineLog + MLX.Memory.snapshot() 的 active/peak/cache。
func memPointLog(_ label: String) {
    let s = MLX.Memory.snapshot()
    let gb: (Int) -> String = { String(format: "%.1fG", Double($0) / Double(1 << 30)) }
    pipelineLog("🔍 [mem] \(label)：active=\(gb(s.activeMemory)) peak=\(gb(s.peakMemory)) cache=\(gb(s.cacheMemory))")
}

// MARK: - 大栈线程执行体（跨模型通用）

private final class BigStackBox<T> {
    var result: T? = nil
}

func runOnBigStack<T>(_ body: @escaping () -> T) -> T {
    let box = BigStackBox<T>()
    let sem = DispatchSemaphore(value: 0)
    let thread = Thread {
        box.result = body()
        sem.signal()
    }
    thread.name = "mlx-bigstack"
    thread.stackSize = 64 * 1024 * 1024
    thread.start()
    sem.wait()
    return box.result!
}


// MARK: - 管线日志（print + 模型监控浮层 + 落盘文件 三通道，跨模型通用）

/// 管线日志（三通道）：Xcode 控制台 + 模型监控浮层 + 落盘文件。
/// - 参数 logURL: 落盘日志文件 URL；nil 时默认 output/视频/pipeline.log（保持视频管线原行为）。
/// - 图像管线（HiDream）传入 output/图像/pipeline.log（方案 a-2：imagePipelineLog 并入本函数）。
func pipelineLog(_ message: String, logURL: URL? = nil) {
    print(message)
    monitorLog(message)
    let url = logURL ?? outputVideoDirURL.appendingPathComponent("pipeline.log")
    let line = "[\(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium))] \(message)\n"
    if let h = try? FileHandle(forWritingTo: url) {
        h.seekToEndOfFile()
        h.write(line.data(using: .utf8)!)
        try? h.close()
    } else {
        try? line.data(using: .utf8)?.write(to: url)
    }
}


// MARK: - 内存自适应策略（跨模型统一水位：视频 LTX / 图像 HiDream 共用）

/// 当前生成任务类型（决定阶段→模块映射：视频采样用 DiT，图像采样用 HiDream）
enum GenerationTask {
    case image   // HiDream 图像生成
    case video   // LTX 视频生成
}

/// 生成管线阶段（完整骨架）：
/// 前三个 case（encoding/sampling/vae）为既有内存路由三阶段，
/// ensureCapacity/unloadIfNeededMidway/protectedModules 保护"当前阶段 + 下一阶段"需要的模块，
/// 只把之前阶段/之前任务用过、当前与下一阶段都不需要的模块列为卸载候选；
/// 后三个 case（refine/exporting/done）为完整骨架新增，供 stageEnter 阶段闸门做
/// 打点/报告/阶段级处理路由。新增 case 已同步补全 protectedModules 内 switch，
/// 既有内存卸载语义保持不变（新增 case 中仅 refine 按 sampling 保护 DiT，供 Stage2 复用）。
enum GenerationStage {
    case encoding   // 视频：文本编码（Gemma + connector）
    case sampling   // 视频：DiT 采样；图像：HiDream 采样
    case vae        // 视频：VAE 解码/升频（临时加载模块，无持久缓存候选）
    case refine     // Stage2 升频/二次采样（内存语义同 sampling：保护 DiT）
    case exporting  // 导出/落盘（mp4/wav/png 写出；无持久缓存模块需要保护）
    case done       // 管线收尾/完成报告（无持久缓存模块需要保护）

    /// 中文阶段名（面板/落盘/打点标签用，中文文案风格与项目一致）
    var label: String {
        switch self {
        case .encoding:   return "编码"
        case .sampling:   return "采样"
        case .vae:        return "VAE解码"
        case .refine:     return "Refine升频"
        case .exporting:  return "导出"
        case .done:       return "完成"
        }
    }
}

/// 可卸载模型模块（持久缓存粒度）与体积：
/// HiDream 扩散头(72M) 已随主干灌入同一 backbone 实例，无独立缓存，不单列；
/// TextEncoderCache 内部可拆 connector / gemma，分开排序（体积从小到大先卸小的）。
enum MemoryModule: String, CaseIterable {
    case connector   // LTX connector，5.9G
    case gemma       // Gemma4 文本编码器（tokenizer 体积小随留），6.2G
    case dit         // LTX DiT 权重，11G（与编译图强绑定）
    case hiDream     // HiDream backbone（含扩散头 72M），16G

    /// 所属模型域：true = LTX 视频侧，false = HiDream 图像侧（异模型优先层级依据）
    var isLTX: Bool {
        switch self {
        case .connector, .gemma, .dit: return true
        case .hiDream: return false
        }
    }

    var sizeBytes: Int {
        switch self {
        case .connector: return 5_900_000_000
        case .gemma:     return 6_200_000_000
        case .dit:       return 11_000_000_000
        case .hiDream:   return 16_000_000_000
        }
    }

    /// 该模块权重当前是否真实驻留缓存（幂等卸载依据）。
    /// 已空转模块返回 false，防止 ensureLoose/ensureCapacity/用完即卸对空缓存重复卸载空转。
    var isLoaded: Bool {
        switch self {
        case .connector: return TextEncoderCache.shared.hasConnector
        case .gemma:     return TextEncoderCache.shared.hasGemma
        case .dit:       return DiTModelCache.shared.hasDiT
        case .hiDream:   return HiDreamModelCache.shared.hasHiDream
        }
    }

    /// 卸载动作（DiT 与编译图强绑定：先清 CompiledForwardCache 再清 DiTModelCache）
    /// 注意：CompiledForwardCache.clear() 会深递归析构多层 compiled 大图（闭包强捕获权重），
    /// 必须在 64MB 大栈线程执行，否则普通线程爆栈 → EXC_BAD_ACCESS code=2（地址落在 Stack Guard）。
    /// - Returns: true=本次实际执行了卸载；false=模块本就未缓存（幂等跳过，调用方不应打日志/清缓存）。
    @discardableResult
    func unload() -> Bool {
        guard isLoaded else { return false }
        switch self {
        case .connector: TextEncoderCache.shared.clearConnector()
        case .gemma:     TextEncoderCache.shared.clearGemma()
        case .dit:
            runOnBigStack {
                CompiledForwardCache.shared.clear()
                DiTModelCache.shared.clear()
            }
        case .hiDream:
            // 与 DiT 同规则：编译图闭包强捕获 backbone，必须先清编译图再清模型缓存，
            // 否则 HiDreamModelCache 置 nil 后 16G 权重仍被闭包持有、释放不掉（实测卸载后 active 不降）。
            runOnBigStack {
                HiDreamCompiledForwardCache.shared.clear()
                HiDreamModelCache.shared.clear()
            }
        }
        return true
    }
}

/// 统一内存水位策略：所有机器（不区分内存大小）在 MLX active 内存逼近物理内存 90% 时，
/// 保护"当前阶段 + 下一阶段"所需模块，只卸之前用过的模块，且按权重大小从小到大卸，
/// 尽量保留大权重，减少连续任务的整体重载时间。
enum MemoryPolicy {
    /// 统一触发阈值：active ≥ 物理内存 × 此比例 时开始动态卸载（不区分内存大小）
    static let memoryThresholdRatio: Double = 0.84

    /// 互斥规则：物理内存 ≤64GB 时，视频/图像两套模型互斥驻留（中途用完即卸时异模型一并卸）；
    /// >64GB 不互斥，异模型保留缓存供快速切换。
    static var mutualExclusive: Bool { physicalMemoryGB <= 64 }

    static var physicalMemoryGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1 << 30))
    }

    /// 物理内存总量（字节），兜底阶梯最后防线/拦截判定用
    static var physicalMemoryBytes: Int {
        Int(ProcessInfo.processInfo.physicalMemory)
    }

    /// 触发卸载的内存水位（字节）= 物理内存 × 0.9
    static var activeThresholdBytes: Int {
        Int(Double(ProcessInfo.processInfo.physicalMemory) * memoryThresholdRatio)
    }

    /// 当前系统已用内存（字节）= 物理内存 − 空闲 − 非活动页（含 MLX、系统与其他 app 占用）
    static var activeBytes: Int {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else {
            return Int(ProcessInfo.processInfo.physicalMemory)
        }
        let pageSize = UInt64(vm_kernel_page_size)
        let freeBytes = UInt64(stats.free_count) * pageSize
        let inactiveBytes = UInt64(stats.inactive_count) * pageSize
        return max(0, Int(ProcessInfo.processInfo.physicalMemory) - Int(freeBytes + inactiveBytes))
    }

    /// 内存压力触发阈值：压力分（压缩×0.7 + 交换×0.3）达到此值时，即使 active 水位未超也触发卸载/降档
    static let pressureTrigger: Double = 10.0

    /// 内存压力分（0~∞）：macOS 压缩内存 + 交换内存的加权和。
    /// 可用内存会被压缩/交换虚高（压缩 11G + 交换 15G 时 available 仍显示充足），
    /// 压力分才是真实紧张度的可信信号。compressedGB = compressor_page_count × vm_page_size；
    /// swapGB = vm.swapusage.xsu_used。实测压缩 2.5G/交换 0 → 压力分 1.75 正常；
    /// 长视频+多任务时压缩/交换上升，压力分随之抬升。
    static var pressureScore: Double {
        // 压缩内存：host_statistics64 的 compressor_page_count（vm.compressor_page_count 在本机是 unknown oid，恒读 0）
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        let compressedGB = kr == KERN_SUCCESS
            ? Double(stats.compressor_page_count) * Double(vm_kernel_page_size) / 1e9
            : 0
        // macOS sysctl "vm.swapusage" 返回 struct xsw_usage（32 字节）：
        // xsu_total/xsu_avail/xsu_used(UInt64×3) + xsu_pagesize(UInt32) + xsu_encrypted(UInt32)
        // 元组标签必须与真实布局对齐，否则 xsu_used 实际读到的是空闲交换量
        var su = (xsu_total: UInt64(0), xsu_avail: UInt64(0), xsu_used: UInt64(0), xsu_pagesize: UInt32(0), xsu_encrypted: UInt32(0))
        var ss = MemoryLayout.size(ofValue: su)
        let skr = withUnsafeMutableBytes(of: &su) { raw in
            sysctlbyname("vm.swapusage", raw.baseAddress, &ss, nil, 0)
        }
        let swapGB = skr == 0 ? Double(su.xsu_used) / 1e9 : 0
        return compressedGB * 0.7 + swapGB * 0.3
    }

    /// MLX 空闲 buffer 缓存上限（统一 2GB：动态卸载策略下权重释放后更快归还系统）
    static var bufferCacheLimit: Int {
        2_000_000_000
    }

    /// MLX 总内存硬上限（物理内存 70%）：阻止 NA 阶段计算图膨胀触发交换/压缩
    static var memoryLimit: Int {
        Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.7)
    }

    static func report() {
        pipelineLog("内存策略：统一动态卸载（active ≥ 物理内存 \(Int(memoryThresholdRatio * 100))% 触发），物理内存 \(physicalMemoryGB)GB")
    }

    /// 阶段 → 需要的模块集合（保护集）
    private static func protectedModules(for task: GenerationTask, current: GenerationStage, next: GenerationStage?) -> Set<MemoryModule> {
        var protect: Set<MemoryModule> = []
        func add(_ stage: GenerationStage?) {
            guard let stage else { return }
            switch stage {
            case .encoding:
                protect.insert(.connector)
                protect.insert(.gemma)
            case .sampling:
                switch task {
                case .video: protect.insert(.dit)
                case .image: protect.insert(.hiDream)
                }
            case .refine:
                // Stage2 升频为二次 DiT 采样，保护语义与 sampling 一致
                switch task {
                case .video: protect.insert(.dit)
                case .image: protect.insert(.hiDream)
                }
            case .vae:
                break   // VAE/升频器临时加载，不在持久缓存候选内
            case .exporting:
                break   // 导出阶段无模型驻留需求，不产生保护集
            case .done:
                break   // 管线收尾不加载模型，不产生保护集
            }
        }
        add(current)
        add(next)
        return protect
    }

    /// 动态安全预留：按任务尺寸与时长计算（小任务更激进、大任务更保守）。
    /// 面积 = width × height / 1_000_000（单位：Mp，百万像素）。
    /// 图像任务：预留 = max(2G, 2_000_000_000 + 1_000_000_000 × 面积)
    ///   （2G 基准 + 1G/Mp；512² ≈2.26G，1024² ≈3.05G）
    /// 视频任务：预留 = max(2G, 1_500_000_000 + 1_500_000_000 × 面积 × (frames / 120))
    ///   （1.5G 基准 + 1.5G/Mp × 时长倍率；5s@24fps=120 帧倍率 1.0，10s=240 帧倍率 2.0；
    ///     大分辨率 + 长时长 更保守，小分辨率 + 短视频 更激进）
    static func safetyBuffer(for task: GenerationTask, width: Int, height: Int, frames: Int) -> Int {
        let area = Double(width * height) / 1_000_000
        switch task {
        case .image:
            return max(2_000_000_000, 2_000_000_000 + Int(1_000_000_000 * area))
        case .video:
            return max(2_000_000_000, 1_500_000_000 + Int(1_500_000_000 * area * (Double(frames) / 120.0)))
        }
    }

    /// 加载前评估水位：在各模型加载点【加载前】调用。
    /// 判定/停止条件：active + 本次预估载入体积 + 本次动态预留 ≤ 物理内存×0.9 时不卸载；
    /// 超过才卸载候选，每卸一个 clearCache 并重新 snapshot，直到满足条件才停。
    /// newSizeBytes=0（模型已在缓存、无需重载）时永不触发卸载，直接复用。
    /// 安全预留由 safetyBuffer(for:width:height:frames:) 按任务尺寸/时长动态计算：
    /// 图像按尺寸、视频按尺寸+时长，小任务更激进、大任务更保守。
    /// 卸载候选分两层，两层之间严格先后：
    ///   第 1 层【异模型优先】：先卸与当前任务不同模型的模块
    ///     （视频任务 → 先卸 HiDream 图像侧；图像任务 → 先卸 LTX 视频侧 connector/gemma/dit），
    ///     异模型内部按体积从小到大卸；
    ///   第 2 层【同模型内部】：第 1 层卸完仍不满足才进入，
    ///     卸当前任务模型内、不在保护集的模块（已经用不上的），按体积从小到大卸。
    /// 体积顺序：connector 5.9G → gemma 6.2G → DiT 11G → HiDream 16G（小/重载快的先卸）。
    /// 前两层卸完仍不满足时按阶梯降级：
    ///   阶梯1：让出本次动态预留（active+newSize ≤ 0.9×物理内存）放行，不再卸新模块；
    ///   阶梯2：最后防线（active+newSize ≤ 物理内存+8G）放行，允许逼近硬上限（少量 swap）；
    ///   阶梯3：active+newSize > 物理内存+8G → 模型本身装不下，返回 false 拦截加载，禁止硬加载超上限。
    /// - Parameters:
    ///   - width: 本次生成任务宽度（像素），用于动态预留计算
    ///   - height: 本次生成任务高度（像素），用于动态预留计算
    ///   - frames: 视频任务真实帧数（5s=120、10s=240，传 duration.numFrames）；图像任务不依赖帧数，可传默认 120
    /// - Returns: true=放行加载；false=内存不足拦截（调用侧应向用户输出清晰提示并中止加载）。
    @discardableResult
    static func ensureCapacity(for newSizeBytes: Int, task: GenerationTask, current: GenerationStage, next: GenerationStage? = nil, width: Int, height: Int, frames: Int = 120) -> Bool {
        let safetyBuffer = safetyBuffer(for: task, width: width, height: height, frames: frames)
        let threshold = activeThresholdBytes
        let projected = activeBytes + newSizeBytes + safetyBuffer
        // 压力分高（压缩/交换虚高）时同样进入卸载流程，不只看 active 水位
        let highPressure = MemoryPolicy.pressureScore >= pressureTrigger
        guard newSizeBytes > 0, projected > threshold || highPressure else { return true }

        let protect = protectedModules(for: task, current: current, next: next)
        let candidates = MemoryModule.allCases.filter { !protect.contains($0) }
        // 异模型内部仍按体积升序
        let ltxSide = candidates.filter { $0.isLTX }.sorted { $0.sizeBytes < $1.sizeBytes }
        let hiDreamSide = candidates.filter { !$0.isLTX }.sorted { $0.sizeBytes < $1.sizeBytes }
        // 两层严格先后：视频任务第 1 层卸 HiDream 侧、第 2 层卸 LTX 侧；图像任务相反
        let layer1: [MemoryModule]
        let layer2: [MemoryModule]
        switch task {
        case .video:
            layer1 = hiDreamSide
            layer2 = ltxSide
        case .image:
            layer1 = ltxSide
            layer2 = hiDreamSide
        }

        // 每卸一个后重新评估：active + newSize + 动态预留 ≤ 阈值即满足，停止卸载
        func satisfied() -> Bool {
            activeBytes + newSizeBytes + safetyBuffer <= threshold
        }

        let nextName = next.map { String(describing: $0) } ?? "无"
        pipelineLog("⚠️ 加载前评估：active=\(activeBytes) + 载入=\(newSizeBytes) + 动态预留=\(safetyBuffer) > 阈值=\(threshold)（\(Int(memoryThresholdRatio * 100))% 物理内存），保护 当前=\(current) 下一=\(nextName)，第1层异模型优先（\(task == .video ? "HiDream" : "LTX视频侧")）卸载...")

        // 第 1 层：异模型优先
        for m in layer1 {
            if satisfied() { break }
            guard m.unload() else { continue }   // 幂等：未缓存的候选直接跳过，不打日志不清缓存
            MLX.Memory.clearCache()
            pipelineLog("✅ [第1层·异模型优先] 卸载候选（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }
        // 第 2 层：同模型内部（第 1 层卸完仍不满足才进入）
        for m in layer2 {
            if satisfied() { break }
            guard m.unload() else { continue }   // 幂等：未缓存的候选直接跳过，不打日志不清缓存
            MLX.Memory.clearCache()
            pipelineLog("✅ [第2层·同模型内部] 卸载候选（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }

        // ── 兜底阶梯：前两层卸完仍不满足时按顺序降级 ──
        let noBuffer = activeBytes + newSizeBytes   // 让出本次动态预留后的净需求
        let physical = physicalMemoryBytes
        let hardLimit = physical + 8 * (1 << 30)    // 兜底硬上限：物理内存 + 8 GiB（任何设备固定 +8G）
        if noBuffer <= threshold {
            // 阶梯1：让出本次动态预留（不再卸新模块），直接放行
            pipelineLog("⚠️ 候选已全部卸载仍超 90%+动态预留，让出本次动态预留 \(safetyBuffer) 字节放行（active=\(activeBytes) + 载入=\(newSizeBytes) ≤ 阈值=\(threshold)）")
            return true
        }
        if noBuffer <= hardLimit {
            // 阶梯2：最后防线，允许逼近 物理内存+8G（允许少量 swap）
            pipelineLog("⚠️ 候选已全部卸载且 90% 阈值仍不满足，最后防线放行（active=\(activeBytes) + 载入=\(newSizeBytes) ≤ 物理内存+8G=\(hardLimit)），允许逼近硬上限")
            return true
        }
        // 阶梯3：模型本身装不下 → 拦截加载，禁止硬加载超上限
        pipelineLog("❌ 内存不足，无法加载该模型（当前占用 \(activeBytes) 字节，需 \(newSizeBytes) 字节，合计 \(noBuffer) > 硬上限 \(hardLimit) 字节），已拦截加载")
        return false
    }

    /// 生成结束后按水位策略卸载模型缓存（保留原调用点）。
    /// 顺序重要：先清 CompiledForwardCache——其编译闭包强捕获 DiT 权重，
    /// 不清它，DiTModelCache 里的引用即使置 nil 权重也释放不掉。
    /// 触发条件：active 水位 ≥ 阈值，或内存压力分 ≥ pressureTrigger（压缩/交换虚高时兜底）。
    static func unloadModelsIfNeeded() {
        let threshold = activeThresholdBytes
        let ps = MemoryPolicy.pressureScore
        guard activeBytes >= threshold || ps >= pressureTrigger else { return }
        pipelineLog("⚠️ 内存仍逼近上限（active=\(activeBytes) 阈值=\(threshold)，压力分 \(String(format: "%.1f", ps))/\(pressureTrigger)），动态卸载模型缓存...")
        // CompiledForwardCache.clear() 深递归析构编译大图，必须在 64MB 大栈线程执行（防栈溢出 EXC_BAD_ACCESS）
        runOnBigStack {
            CompiledForwardCache.shared.clear()
            DiTModelCache.shared.clear()
            TextEncoderCache.shared.clear()
            // HiDream 同规则：先清编译图（闭包强捕获 backbone）再清模型缓存，否则权重卸不掉
            HiDreamCompiledForwardCache.shared.clear()
            HiDreamModelCache.shared.clear()
            MLX.Memory.clearCache()
        }
        pipelineLog("✅ 模型缓存已卸载，常驻内存回落")
    }

    /// 生成中途用完即卸：在阶段切换点（条件嵌入完成→采样前、VAE 解码前）调用。
    /// 不看水位、不看队列——只要"当前+下一阶段保护集外"的模块对本次任务已无用，
    /// 立即卸载让位（Gemma/Connector 采样前丢、DiT 解码前丢，交给页缓存兜底重载）。
    /// 互斥规则：内存 ≤64GB 时视频/图像两套模型互斥，异模型一并卸（任务切换不残留）；
    /// >64GB 不互斥，只卸本任务链的模块，异模型保留缓存供快速切换。
    static func unloadIfNeededMidway(task: GenerationTask, current: GenerationStage, next: GenerationStage? = nil) {
        let protect = protectedModules(for: task, current: current, next: next)
        var candidates = MemoryModule.allCases.filter { !protect.contains($0) }
        if !mutualExclusive {
            // 不互斥（>64GB）：只卸本任务链（LTX 视频侧 / HiDream 图像侧）的模块，异模型保留
            candidates = candidates.filter { $0.isLTX == (task == .video) }
        }
        // 幂等：只对真实驻留缓存的模块发起卸载（空缓存卸载只会刷日志、不降压力）
        candidates = candidates.filter { $0.isLoaded }
        let ordered = candidates.sorted { $0.sizeBytes < $1.sizeBytes }
        guard !ordered.isEmpty else { return }
        let nextName = next.map { String(describing: $0) } ?? "无"
        pipelineLog("⚠️ 中途用完即卸（互斥=\(mutualExclusive)）：保护 当前=\(current) 下一=\(nextName)，卸载候选：\(ordered.map { $0.rawValue }.joined(separator: "、"))")
        for m in ordered {
            m.unload()
            MLX.Memory.clearCache()
            pipelineLog("✅ [用完即卸] 卸载（\(m.sizeBytes) 字节）：\(m.rawValue)")
        }
    }

    /// 内存压力策略档位：按 pressureScore 自动分 3 档（低压力速度优先、中压力均衡、高压力内存优先）。
    enum MemoryStrategy: String {
        case speed = "速度优先"
        case balanced = "均衡"
        case memory = "内存优先"
    }

    /// 档位分界阈值（常量，便于调整）：
    /// 压力分 < strategySpeedMax → 速度优先；
    /// strategySpeedMax ≤ 压力分 < strategyBalancedMax → 均衡；
    /// 压力分 ≥ strategyBalancedMax → 内存优先。
    static let strategySpeedMax: Double = 6.0
    static let strategyBalancedMax: Double = pressureTrigger

    /// 当前策略档位：由实时压力分推导。
    static var currentStrategy: MemoryStrategy {
        let ps = pressureScore
        if ps < strategySpeedMax { return .speed }
        else if ps < strategyBalancedMax { return .balanced }
        else { return .memory }
    }

    /// ensureLoose 实际触发阈值（随策略档位联动）：
    /// 速度优先 → 阈值提高（压力分 14 才触发，少打断采样，保速度）；
    /// 均衡 → 维持 pressureTrigger（10）；
    /// 内存优先 → 阈值降低（压力分 7 即触发，早介入防 OOM）。
    static var ensureLooseTrigger: Double {
        switch currentStrategy {
        case .speed:   return pressureTrigger + 4.0
        case .balanced: return pressureTrigger
        case .memory:  return pressureTrigger - 3.0
        }
    }

    /// 采样循环内轻量水位兜底（ensureLoose）：真实重前向入口每步调用。
    /// 压力分（压缩×0.7 + 交换×0.3）≥ ensureLooseTrigger 时立即清理 MLX 缓存；
    /// 触发阈值随策略档位联动（见 ensureLooseTrigger）。
    /// 清理后压力仍超标则卸载"非保护集"模型缓存（采样中保护当前模型，候选为异模型/已用完模块，卸载无副作用）。
    /// 与 ensureCapacity（加载前评估）/ unloadIfNeededMidway（阶段切换）互补，专防采样途中峰值 OOM。
    /// - Parameter protect: 采样中正在使用的模型模块集合，绝不卸载；缺省空集（仅清缓存，不碰模型缓存）。
    static func ensureLoose(protect: Set<MemoryModule> = [], force: Bool = false) {
        // force=true：编译窗口前无条件清场。编译/首步采样瞬时分配远早于压力分阈值，
        // ensureLoose 门槛（guard ps >= trigger）拦不住这类峰值，故编译前调用方须传 force 直清。
        // 语义与 ensureLoose 完全一致：仅清 MLX 空闲 buffer + 幂等卸载非保护模块（connector/gemma/
        // hiDream 等本段已无用权重，用时会经各自缓存 load 重载）；protect 内模块（.dit）与
        // CompiledForwardCache 同形状编译图（调用方已先 evictExcept）一律不动，可复用项不清。
        if force {
            pipelineLog("⚠️ [ensureLoose·force] 编译窗口前强制清场（无门槛）")
            MLX.Memory.clearCache()
            let candidates = MemoryModule.allCases
                .filter { !protect.contains($0) }
                .filter { $0.isLoaded }
                .sorted { $0.sizeBytes < $1.sizeBytes }
            for m in candidates {
                m.unload()
                MLX.Memory.clearCache()
                pipelineLog("✅ [ensureLoose·force] 卸载非保护模块（\(m.sizeBytes) 字节）：\(m.rawValue)")
            }
            return
        }
        let ps = pressureScore
        let trigger = ensureLooseTrigger
        guard ps >= trigger else { return }
        pipelineLog("⚠️ [ensureLoose] 采样中压力分 \(String(format: "%.1f", ps))/\(trigger)（策略 \(currentStrategy.rawValue)），清理 MLX 缓存")
        MLX.Memory.clearCache()
        if pressureScore >= trigger {
            // 幂等：只对真实驻留缓存的非保护模块发起卸载（采样中途多数模块已空转，过滤后无空转日志）
            let candidates = MemoryModule.allCases
                .filter { !protect.contains($0) }
                .filter { $0.isLoaded }
                .sorted { $0.sizeBytes < $1.sizeBytes }
            for m in candidates {
                m.unload()
                MLX.Memory.clearCache()
                pipelineLog("✅ [ensureLoose] 卸载非保护模块（\(m.sizeBytes) 字节）：\(m.rawValue)")
            }
        }
    }
}


// MARK: - 统一阶段闸门（stageEnter：进入生成阶段时统一做 内存清理决策 + 内存打点 + 报告输出）

/// 阶段闸门：在每个生成阶段边界 / 阶段内关键报告点调用一次，函数体内统一做三件事：
///   a) 内存清理决策：按调用方传入的阶段保护集路由到 MemoryPolicy 既有入口
///      （ensureLoose(protect:) 等）。仅做"调用面收敛"，不改任何清理阈值/水位/
///      模块保护语义；试点阶段只允许在阶段边界（step == nil）且显式给出 protect 时
///      发起清理，避免无任务上下文误卸模型。采样循环内每步的真实清理仍由既有调用点
///      （如 ditX0Guided 内 ensureLoose(protect: [.dit])）负责，闸门不重复干预。
///   b) 内存打点：沿用 memProfilePoint 收集机制（不实时打印，memProfileReport 统一输出），
///      不改其实现与收集器。
///   c) 报告输出：唯一出口 pipelineLog（print + monitorLog 面板 + 落盘），不新建平行输出函数。
/// - Parameters:
///   - phase: 生成阶段（GenerationStage.label 提供中文名）
///   - step: 阶段内进度序号（1-based）；nil 表示阶段边界/阶段级报告
///   - total: 阶段总步数，与 step 成对；nil 表示非分步阶段
///   - detail: 报告正文（原样进入 pipelineLog，文案/排版由调用方自带，保持既有输出一致）
///   - logURL: 落盘 URL，nil 走 pipelineLog 默认路径
///   - protect: 本阶段不可卸载的模块集合（阶段边界清理时传给 ensureLoose）；
///     为空集时不发起清理（该阶段清理由组装层既有阶段边界卸载点负责，防止误卸）。
func stageEnter(
    phase: GenerationStage,
    step: Int? = nil,
    total: Int? = nil,
    detail: String,
    logURL: URL? = nil,
    protect: Set<MemoryModule> = []
) {
    // a) 内存清理决策（仅阶段边界 + 有保护集时收敛调用；每步细粒度点由既有清理点兜底）
    if step == nil, !protect.isEmpty {
        MemoryPolicy.ensureLoose(protect: protect)
    }

    // b) 内存打点（收集式，不打印；memProfileReport 批量输出后清空）
    let pointLabel: String
    if let step = step, let total = total {
        pointLabel = String(format: "%@ 步%02d/%d", phase.label, step, total)
    } else {
        pointLabel = phase.label + " 进入"
    }
    memProfilePoint(pointLabel)

    // c) 报告输出（统一出口：pipelineLog 三通道）
    pipelineLog(detail, logURL: logURL)
}


// MARK: - 通用图像/媒体工具


/// 直接把 MLXArray [3,H,W]（[-1,1] f32）经 vDSP 写入 CVPixelBuffer 的 XRGB 区，
/// 跳过 CGImage 中间层，消除一次全幅像素拷贝。RGB 分量逐位不变，保质量。
/// 布局对齐 CVPixelBuffer nxoneSkipFirst：byte0=alpha 占位(0)，byte1/2/3=R/G/B。
func fillPixelBuffer(_ frame: MLXArray, width: Int, height: Int,
                     base: UnsafeMutableRawPointer?, bytesPerRow: Int) -> Bool {
    guard let base = base else { return false }
    let floats = frame.asArray(Float.self)
    guard floats.count == 3 * width * height else { return false }
    let hw = width * height
    var scale: Float = 127.5
    var offset: Float = 127.5
    var lo: Float = 0
    var hi: Float = 255
    var tmp = [Float](repeating: 0, count: width)
    let rowStride = bytesPerRow
    let pb = base.assumingMemoryBound(to: UInt8.self)
    floats.withUnsafeBufferPointer { src in
        tmp.withUnsafeMutableBufferPointer { tbuf in
            for y in 0..<height {
                let dstBase = pb + y * rowStride
                memset(dstBase, 0, rowStride)   // 清 alpha 占位与行尾 padding
                let rowOff = y * width
                for c in 0..<3 {
                    let srcBase = src.baseAddress! + c * hw + rowOff
                    vDSP_vsmsa(srcBase, 1, &scale, &offset, tbuf.baseAddress!, 1, vDSP_Length(width))
                    vDSP_vclip(tbuf.baseAddress!, 1, &lo, &hi, tbuf.baseAddress!, 1, vDSP_Length(width))
                    vDSP_vfixru8(tbuf.baseAddress!, 1, dstBase + 1 + c, 4, vDSP_Length(width))
                }
            }
        }
    }
    return true
}


/// 10bit 量化：Y code = round(Y' * 876 + 64)，C code = round(C * 896 + 64)，clip [0, 1023]
/// 色度输入 C 为 [0,1] 归一（0.5 = 消色点），故映射到 limited 10bit 区间 64..960（512 为中性灰）。
/// ⚠️ 勿写 +512：C 已含 +0.5 中心化，再 +512 会把色度整体推到 512..1408 后 clip 至上限，
/// 中性灰/肤色帧被推成高饱和品红、绿通道丢失（纯色测试卡因色度本就接近极值无法暴露此 bug）。
@inline(__always) private func v210QuantLuma(_ y: Float) -> UInt32 {
    let v = (y * 876 + 64).rounded()
    return UInt32(min(max(v, 0), 1023))
}
@inline(__always) private func v210QuantChroma(_ c: Float) -> UInt32 {
    let v = (c * 896 + 64).rounded()
    return UInt32(min(max(v, 0), 1023))
}

/// v210（kCVPixelFormatType_422YpCbCr10）10bit 填充路径（ProRes 422 最终输出专用）：
/// 输入 f32 [3,H,W] CHW（[-1,1]，与 fillPixelBuffer 同布局）→ 还原到 [0,1] →
/// BT.709 limited 矩阵转 Y'CbCr → 10bit 量化 → v210 打包。
///
/// BT.709 limited 矩阵（与 VideoToolbox ProRes BT.709 附件同语义）：
///   Y' = 0.2126*R + 0.7152*G + 0.0722*B
///   Cb = (B - Y') / 1.8556 + 0.5   （1.8556 = 2 - 2*0.0722，limited 色度分母）
///   Cr = (R - Y') / 1.5748 + 0.5   （1.5748 = 2 - 2*0.2126）
/// 量化：Y code = round(Y'*876 + 64)，C code = round(C*896 + 64)，clip [0,1023]
/// （10bit limited：亮度 64..940 占 876 级；色度 C∈[0,1] 以 0.5 为中心，code 64..960 占 896 级，
///   即 C=0.5（消色点）→ 512）。⚠️ 色度勿再叠加 512 偏移——C 本身已含 +0.5 中心化。
///
/// v210 排布（每 6 像素 = 16 字节 = 8×16bit little-endian halfword；YCbCr 4:2:2，
/// 像素对 (x0,x1)/(x2,x3)/(x4,x5) 共享 1 个 Cb/Cr）：
///   halfword 序列：Cb0 Y0 Cr0 Y1 Cb2 Y2 Cr2 Y3 Cb4 Y4 Cr4 Y5
///   等价 4×32bit little-endian word（各 word 高 2 bit 未用，置 0）：
///     word0 = Cb0 | Y0<<10 | Cr0<<20
///     word1 = Y1 | Cb2<<10 | Y2<<20
///     word2 = Cr2 | Y3<<10 | Cb4<<20
///     word3 = Y4 | Cr4<<10 | Y5<<20
/// 行宽按实际 rowStride（CVPixelBufferGetBytesPerRow）行走；行首不足 6 像素的宏块按 v210
/// 规范补齐（宽度按编码器 width 读，pad 像素不显示；色度取像素对左采样，pad 像素不污染）。
func fillPixelBufferV210(_ frame: MLXArray, width: Int, height: Int,
                         base: UnsafeMutableRawPointer?, bytesPerRow: Int) -> Bool {
    guard let base = base else { return false }
    let floats = frame.asArray(Float.self)
    guard floats.count == 3 * width * height else { return false }
    let hw = width * height
    let macroBlocks = (width + 5) / 6          // v210 以 6 像素宏块（16 字节）为最小单元
    let usedBytes = macroBlocks * 16
    // v210 行字节最小粒度 16：usedBytes = ceil(width/6)*16。CoreVideo 对 422YpCbCr10 分配的行
    // stride 恒 ≥ 理论行字节（按 16B 分组对齐自动满足）；此处显式防御：若返回值小于理论值则
    // 直接失败，避免按假想 layout 写穿缓冲（缓冲实际按 bytesPerRow 分配，max() 静默放大 rowStride
    // 会让末行写入越过缓冲末尾，故不得用 max 兜底）。
    guard bytesPerRow >= usedBytes else {
        pipelineLog("⚠️ v210 bytesPerRow(\(bytesPerRow)) < usedBytes(\(usedBytes))（width=\(width)），放弃填充")
        return false
    }
    let rowStride = bytesPerRow
    let bytes = base.assumingMemoryBound(to: UInt8.self)
    // 每帧仅一次临时分配（按行复用，无每帧高频分配）
    var out = [UInt32](repeating: 0, count: macroBlocks * 4)
    floats.withUnsafeBufferPointer { src in
        let fp = src.baseAddress!
        // 帧级复用宏块临时缓冲：mb 循环内每宏块分配会高频触发堆分配（每帧 ~17 万次），
        // 提为帧级后每帧仅分配一次；ys[0...5]/cbs[0...2]/crs[0...2] 由下方 p 循环全覆盖写，无需清零。
        var ys = [UInt32](repeating: 0, count: 6)
        var cbs = [UInt32](repeating: 0, count: 3)
        var crs = [UInt32](repeating: 0, count: 3)
        for y in 0..<height {
            memset(bytes + y * rowStride, 0, rowStride)
            for mb in 0..<macroBlocks {
                let px = mb * 6
                // 行尾宏块不足 6 像素（width 非 6 倍数）时按 edge-replicate 补齐：
                // 越界样本复制本宏块最后一个有效像素的 Y（偶数位同时复制其像素对共享的 Cb/Cr），
                // 保证每个宏块必写满 6 个样本且值来自本行真实内容——杜绝伪样本灰边（旧实现 pad 全 0
                // 会产出中性灰 Y≈502/C≈512 的右缘条）与帧级数组残留脏值（ys/cbs/crs 现在全覆盖写）。
                let n = min(6, width - px)
                var lastY: UInt32 = 0
                var lastCb: UInt32 = 0
                var lastCr: UInt32 = 0
                for p in 0..<n {
                    let x = px + p
                    let i = y * width + x
                    let r = (fp[0 * hw + i] + 1) * 0.5   // [-1,1] → [0,1]
                    let g = (fp[1 * hw + i] + 1) * 0.5
                    let b = (fp[2 * hw + i] + 1) * 0.5
                    let yy = 0.2126 * r + 0.7152 * g + 0.0722 * b
                    ys[p] = v210QuantLuma(yy)
                    lastY = ys[p]
                    if p % 2 == 0 {          // 色度取像素对左采样（4:2:2 co-sited 左对齐语义）
                        cbs[p / 2] = v210QuantChroma((b - yy) / 1.8556 + 0.5)
                        crs[p / 2] = v210QuantChroma((r - yy) / 1.5748 + 0.5)
                        lastCb = cbs[p / 2]
                        lastCr = crs[p / 2]
                    }
                }
                for p in n..<6 {             // 越界补齐：复制最后有效像素（Y 全复制；偶数位补其像素对 Cb/Cr）
                    ys[p] = lastY
                    if p % 2 == 0 { cbs[p / 2] = lastCb; crs[p / 2] = lastCr }
                }
                // 按 v210 16 字节宏块组装 4×UInt32（macOS 为 little-endian，存小端字节序）
                out[mb * 4 + 0] = (cbs[0] | (ys[0] << 10) | (crs[0] << 20)).littleEndian
                out[mb * 4 + 1] = (ys[1] | (cbs[1] << 10) | (ys[2] << 20)).littleEndian
                out[mb * 4 + 2] = (crs[1] | (ys[3] << 10) | (cbs[2] << 20)).littleEndian
                out[mb * 4 + 3] = (ys[4] | (crs[2] << 10) | (ys[5] << 20)).littleEndian
            }
            out.withUnsafeBufferPointer { ob in
                memcpy(bytes + y * rowStride, ob.baseAddress!, usedBytes)
            }
        }
    }
    return true
}

/// writeMp4 帧闭包统一分流：像素缓冲为 v210（ProRes 10bit）时走 10bit 打包，否则走原 8bit XRGB。
/// 供各 writeMp4 调用点使用，避免调用点自行判断像素格式导致 8/10bit 混填。
func fillFrameForPixelBuffer(_ frame: MLXArray, width: Int, height: Int,
                             base: UnsafeMutableRawPointer?, bytesPerRow: Int, isV210: Bool) -> Bool {
    if isV210 {
        return fillPixelBufferV210(frame, width: width, height: height, base: base, bytesPerRow: bytesPerRow)
    }
    return fillPixelBuffer(frame, width: width, height: height, base: base, bytesPerRow: bytesPerRow)
}

/// - Parameters:
///   - fillFrame: 帧填充闭包 (frameIndex, baseAddress, bytesPerRow, isV210) -> Bool；
///     isV210 = true 表示本次 writer 的像素缓冲为 v210（ProRes 422 10bit），必须走 v210 打包填充。
func writeMp4(frameCount: Int, width: Int, height: Int, fps: Int, to path: String,
              proRes: Bool = false,
              h264MaxQuality: Bool = false,
              fillFrame: (Int, UnsafeMutableRawPointer?, Int, Bool) -> Bool) throws {
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.removeItem(at: url)
    // ProRes 422 只支持 .mov 容器封装（mp4 容器官方不支持 ProRes 轨道），h264 保持原 .mp4；
    // ProRes 编码器无必需的自定义 compression 属性（h264 原有路径也未设），故不填 AVVideoCompressionPropertiesKey。
    let writer = try AVAssetWriter(outputURL: url, fileType: proRes ? .mov : .mp4)
    var settings: [String: Any] = [
        AVVideoCodecKey: proRes ? AVVideoCodecType.proRes422 : AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height,
        // 显式声明 BT.709 色彩/YCbCr 矩阵，杜绝编码器默认矩阵与读端不一致导致的色度(R/B)错乱
        AVVideoColorPropertiesKey: [
            AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
            AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
        ],
    ]
    // “给用户看”的 H3 stage1 预览等 h264 落盘：显式最高质量档（quality=1.0）。
    // 仅调用方传 h264MaxQuality=true 时生效；默认 false 保持既有 h264 行为（LTX 原生直出等）。
    if !proRes && h264MaxQuality {
        settings[AVVideoCompressionPropertiesKey] = [AVVideoQualityKey: 1.0]
    }
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    // 改造 A：像素缓冲格式按输出分流 —— ProRes 422 用 v210（kCVPixelFormatType_422YpCbCr10，10bit
    // YCbCr 4:2:2 packed，10bit 亮度+色度原样喂给编码器）；h264 预览/直出维持 8bit 32ARGB。
    let pixelFormat: OSType = proRes ? kCVPixelFormatType_422YpCbCr10 : kCVPixelFormatType_32ARGB
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
    writer.add(input)
    guard writer.startWriting() else {
        throw NSError(domain: "mp4", code: 1, userInfo: [NSLocalizedDescriptionKey: "startWriting 失败"])
    }
    writer.startSession(atSourceTime: .zero)
    let frameDur = CMTime(value: 1, timescale: CMTimeScale(fps))
    for i in 0..<frameCount {
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.02)
        }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
        guard let pixelBuffer = pb else { continue }
        // 像素缓冲级色彩声明：VideoToolbox 编码按 buffer 附件选取 YCbCr 矩阵，写进 SPS color info
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(pixelBuffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        // isV210 = proRes：ProRes 分支像素缓冲为 v210，闭包内须走 10bit 打包（fillPixelBufferV210），
        // 由 fillFrameForPixelBuffer 按此布尔分流，避免 8/10bit 混填。
        let filled = fillFrame(i, CVPixelBufferGetBaseAddress(pixelBuffer),
                               CVPixelBufferGetBytesPerRow(pixelBuffer), proRes)
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        guard filled else { continue }
        let t = CMTimeMultiply(frameDur, multiplier: Int32(i))
        adaptor.append(pixelBuffer, withPresentationTime: t)
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    sem.wait()
    guard writer.status == .completed else {
        throw NSError(domain: "mp4", code: 2, userInfo: [NSLocalizedDescriptionKey: "finishWriting 失败: \(writer.error?.localizedDescription ?? "?")"])
    }
}


func archiveIfExists(_ path: String) {
    let fm = FileManager.default
    guard fm.fileExists(atPath: path) else { return }
    let df = DateFormatter()
    df.dateFormat = "yyyyMMdd_HHmmss"
    let stamp = df.string(from: Date())
    let ext = (path as NSString).pathExtension
    let base = (path as NSString).deletingPathExtension
    let newPath = "\(base)_\(stamp).\(ext)"
    try? fm.moveItem(atPath: path, toPath: newPath)
    pipelineLog("📦 归档旧产物：\(newPath)")
}


func muxAudio(videoPath: String, wavPath: String, to outPath: String, proRes: Bool = false) throws {
    let videoURL = URL(fileURLWithPath: videoPath)
    let wavURL = URL(fileURLWithPath: wavPath)
    let outURL = URL(fileURLWithPath: outPath)
    try? FileManager.default.removeItem(at: outURL)

    let videoAsset = AVURLAsset(url: videoURL)
    let audioAsset = AVURLAsset(url: wavURL)
    let comp = AVMutableComposition()
    guard let compVideo = comp.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
          let srcVideo = videoAsset.tracks(withMediaType: .video).first else {
        throw NSError(domain: "mux", code: 1, userInfo: [NSLocalizedDescriptionKey: "视频轨缺失"])
    }
    let dur = srcVideo.timeRange
    try compVideo.insertTimeRange(dur, of: srcVideo, at: .zero)

    if let srcAudio = audioAsset.tracks(withMediaType: .audio).first,
       let compAudio = comp.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
        try compAudio.insertTimeRange(dur, of: srcAudio, at: .zero)
    }

    // ProRes 分支：silent 视频已是 ProRes422，用 AppleProRes422 preset 重封装混入 wav 音轨并输出 .mov；
    // h264 分支维持原 HighestQuality → .mp4（LTX 原生等其它调用路径行为不变）
    guard let exporter = AVAssetExportSession(asset: comp, presetName: proRes ? AVAssetExportPresetAppleProRes422LPCM : AVAssetExportPresetHighestQuality) else {
        throw NSError(domain: "mux", code: 2, userInfo: [NSLocalizedDescriptionKey: "export session 创建失败"])
    }
    exporter.outputURL = outURL
    exporter.outputFileType = proRes ? .mov : .mp4
    let sem = DispatchSemaphore(value: 0)
    exporter.exportAsynchronously { sem.signal() }
    sem.wait()
    guard exporter.status == .completed else {
        throw NSError(domain: "mux", code: 3, userInfo: [NSLocalizedDescriptionKey: "导出失败: \(exporter.error?.localizedDescription ?? "?")"])
    }
}



// MARK: - 图像落盘（MLXArray -> PNG，跨图像模型通用，方案 a-2 上移自 hidream.swift saveHiDreamPNG）

/// 保存 [H,W,3] float32（[0,1]）图像为 PNG。
func saveMLXImagePNG(_ img: MLXArray, to path: String) -> Bool {
    let h = img.dim(0), w = img.dim(1)
    guard h > 0, w > 0 else { return false }
    let floats = img.asType(.float32).flattened().asArray(Float.self)
    guard floats.count == 3 * h * w else { return false }
    #if DEBUG
    let nanCount = floats.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
    let infCount = floats.reduce(0) { $0 + ($1.isInfinite ? 1 : 0) }
    let fMin = floats.min() ?? 0, fMax = floats.max() ?? 0
    let fMean = floats.reduce(0, +) / Float(max(floats.count, 1))
    FileHandle.standardError.write(
        "DBG saveHiDreamPNG nan=\(nanCount) inf=\(infCount) min=\(fMin) max=\(fMax) mean=\(fMean)\n".data(using: .utf8)!)
    #endif
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
        bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: w * 3, bitsPerPixel: 24) else { return false }
    guard let ptr = rep.bitmapData else { return false }
    for i in 0 ..< floats.count {
        ptr[i] = UInt8(max(0, min(255, Int((floats[i] * 255).rounded()))))
    }
    guard let png = rep.representation(using: .png, properties: [:]) else { return false }
    do {
        try png.write(to: URL(fileURLWithPath: path))
        return true
    } catch {
        return false
    }
}


// MARK: - 文件修改时间（跨模型通用扩展，方案 a-2 统一 hidream.swift 与 ltx2.5.swift 的 private 副本）

extension FileManager {
    /// 文件修改时间；不存在或读取失败返回 nil。
    func fileModTime(_ path: String) -> Date? {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return attrs?[.modificationDate] as? Date
    }
}
