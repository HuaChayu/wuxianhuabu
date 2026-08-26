
//
//  公共函数.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/21.
//

import Foundation
import MLX

// MARK: - 便捷构造

/// [Float] → MLXArray（自动推断 shape；一维数组便捷写法）。
func floatArray(_ values: [Float]) -> MLXArray {
    MLXArray(values)
}

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

// MARK: - 数值工具

/// 等差数列：[start, end] 均分 count 个点（纯数学，跨模型通用）。
/// 使用方：HiDream-O1-Image（hidreamGenerate 的 noiseScaleSchedule）。
func linspace(start: Float, end: Float, count: Int) -> [Float] {
    guard count > 1 else { return [start] }
    return (0 ..< count).map { i in
        start + (end - start) * Float(i) / Float(count - 1)
    }
}

// MARK: - 全局模型内存评分机制（纯系统级读数）

/// 全局模型内存评分机制：只做系统内存读数（active / pressure / threshold 等纯系统级量），
/// 模型加载/卸载编排留在 MemoryPolicy（各生成管线侧）。LLM / 图像 / 视频 / 未来任何
/// 模型统一用本枚举读数与评分，MemoryPolicy 内部对系统级量一律引用 SystemMemory.xxx。
enum SystemMemory {
    /// 统一触发阈值：active ≥ 物理内存 × 此比例 时开始动态卸载（不区分内存大小）
    static let memoryThresholdRatio: Double = 0.84

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
    /// 压力分才是真实紧张度的可信信号。compressedGB = compressor_page_count × vm_kernel_page_size；
    /// swapGB = vm.swapusage.xsu_used（已用字节）。实测压缩 2.5G/交换 0 → 压力分 1.75 正常；
    /// 长视频+多任务时压缩/交换上升，压力分随之抬升。
    static var pressureScore: Double {
        // 压缩内存：host_statistics64 读 vm_statistics64.compressor_page_count × vm_kernel_page_size。
        // （sysctl "vm.compressor_page_count" 在 macOS 上不存在 / unknown oid，此路径不可用，
        //  必须走 mach 内核统计——与 MonitorCenter.systemCompressedUsage 同源）
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        let compressedGB = kr == KERN_SUCCESS
            ? Double(stats.compressor_page_count) * Double(vm_kernel_page_size) / 1e9 : 0
        // 交换：macOS sysctl "vm.swapusage" 返回 struct xsw_usage，字段内存顺序是
        // xsu_total/xsu_avail/xsu_used（avail=剩余可用、used=已用）→ 必须读 xsu_used
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

    // MARK: - 峰值保底防线（全局公共能力：LTX / HiDream / 未来模型采样循环共用）

    /// 进入高压阈值：压力分 ≥ 此值视为「高压」，触发保底回收。
    /// 选值 8.0：低于动态卸载阈值 pressureTrigger(10.0)——只回收缓存池不卸模型，
    /// 可在模型卸载之前先低成本减压一档；常驻正常任务压力分约 0~2（压缩 2.5G/交换 0 → 1.75）。
    static let looseEnterHysteresis: Double = 8.0

    /// 退出高压阈值：压力分回落 < 此值才解除滞回。与 enter 之间留 5.0 滞回差，
    /// 避免压力在阈值附近小幅振荡时反复触发 GC（抖振保护）。
    static let looseExitHysteresis: Double = 3.0

    /// 防抖时间窗（秒）：距上次保底触发不足该间隔时跳过，防止高频空转/重复清理。
    static let looseMinGapSeconds: TimeInterval = 10.0

    /// 缓存池上限收紧参数：下限 512MB；每次收缩到当前上限的 60%（一次只小幅调整，防振荡）。
    /// 任务开始 runVideoPipeline 已重置 cacheLimit = bufferCacheLimit，收紧仅作用于本任务内，天然不跨任务残留。
    static let looseCacheLimitFloor: Int = 512_000_000
    static let looseCacheShrink: Double = 0.6

    /// 保底状态（锁保护：只保护标志位读写，成本极低；当前采样同一时刻单任务，未来并发生成也安全）
    private static let looseLock = NSLock()
    private static var looseInHighPressure = false          // 是否处于高压滞回带
    private static var looseLastTriggerAt = Date.distantPast // 上次保底触发时间
    private static var looseCacheLimitTightened = false     // 本轮高压期间是否已收紧过 cacheLimit

    /// 为即将到来的 projected 字节大前向做保底准备（峰值防崩）。
    ///
    /// 语义：projected 为本次真实重前向的峰值字节估算（与 safetyBuffer 同量纲）。
    /// - 预算充足（projected ≤ 当前安全余量）→ 不干预，直接返回 true；
    /// - 预算不足 → 先过「滞回带 + 防抖窗」（高压中且压力未回落到退出阈值则跳过，
    ///   短时间不反复触发），再触发「只碰 MLX 缓存池」的回收：
    ///   `MLX.Memory.clearCache()` 归还空闲 buffer 池 + 按需把 `cacheLimit` 每次收缩 60%
    ///   （clamp [512MB, 当前]），本轮高压内 cacheLimit 只收紧一次（防振荡），随后打印低噪声日志。
    /// 红线：绝不触碰 active 数据 / 权重 / 编译图 / latent，不卸载任何模型；
    /// 模型卸载由 ensureCapacity / unloadModelsIfNeeded / unloadIfNeededMidway 负责，本函数只平抑缓存池。
    /// - Parameters:
    ///   - projected: 即将到来的大前向峰值字节估算（>0；示例：视频步骤用 safetyBuffer 估）。
    /// - Returns: true = 可放心进入前向（无论是否需要干预，本函数不承担拦截职责）。
    @discardableResult
    static func ensureLoose(projected: Int64) -> Bool {
        // 预算充足：active 距 90% 阈值的安全余量足够容纳 projected → 不干预
        let headroom = Int64(max(0, activeThresholdBytes - activeBytes))
        if projected <= headroom { return true }

        // 预算不足：进入保底流程（滞回带 + 防抖窗）
        let ps = pressureScore
        looseLock.lock()
        defer { looseLock.unlock() }
        if looseInHighPressure {
            // 已在高压：压力未回落到退出阈值前不再重复触发（防 GC 抖振）
            guard ps < looseExitHysteresis else { return true }
            looseInHighPressure = false
            looseCacheLimitTightened = false   // 压力缓解，解除滞回，允许下一轮按需再收紧
        } else {
            guard ps >= looseEnterHysteresis else { return true }
            guard Date().timeIntervalSince(looseLastTriggerAt) >= looseMinGapSeconds else { return true }
            looseInHighPressure = true
        }

        // 触发：只回收 MLX 缓存池（空闲 buffer），不动 active/权重/编译图/latent，不卸模型
        let activeBefore = MLX.Memory.activeMemory
        let cacheBefore = MLX.Memory.cacheMemory
        MLX.Memory.clearCache()
        if !looseCacheLimitTightened {
            let current = MLX.Memory.cacheLimit
            let tightened = max(looseCacheLimitFloor, Int(Double(current) * looseCacheShrink))
            if tightened < current {
                MLX.Memory.cacheLimit = tightened
            }
            looseCacheLimitTightened = true
        }
        let activeAfter = MLX.Memory.activeMemory
        let cacheAfter = MLX.Memory.cacheMemory
        looseLastTriggerAt = Date()
        pipelineLog("⚠️ [峰值保底] projected=\(projected) headroom=\(headroom) 压力分=\(String(format: "%.1f", ps)) 触发：clearCache cache \(cacheBefore)→\(cacheAfter) active \(activeBefore)→\(activeAfter)，回收 \(activeBefore - activeAfter)B，cacheLimit→\(MLX.Memory.cacheLimit)")
        return true
    }
}


