// ============================================================
//  文件作用：全局调试中心。DebugCenter 单例收集调试日志（开关控制），
//  DebugOverlay 在软件右下角浮层显示最近执行步骤，DebugControlPanel
//  提供右下角调试开关按钮。
//  互动文件：被 主ui.swift（挂载 DebugControlPanel）、交互.swift
//  （各鼠标事件调用 DebugCenter.shared.log）引用。
// ============================================================

import SwiftUI
import Combine
import Darwin

// MARK: - 全局调试中心（单例）

final class DebugCenter: ObservableObject {
    static let shared = DebugCenter()

    /// 调试开关：打开后开始收集并显示日志
    @Published var isEnabled = false
    /// 日志列表（最新在末尾）
    @Published private(set) var logs: [String] = []

    private let maxLogs = 60
    private var lastThrottleTime: [String: Date] = [:]
    private let throttleInterval: TimeInterval = 0.2

    private init() {}

    /// 记录一条调试日志；throttle=true 时对相同消息做 200ms 节流（用于高频的鼠标移动）
    func log(_ message: String, throttle: Bool = false) {
        guard isEnabled else { return }
        if throttle {
            let now = Date()
            if let last = lastThrottleTime[message], now.timeIntervalSince(last) < throttleInterval {
                return
            }
            lastThrottleTime[message] = now
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        let line = "[\(formatter.string(from: Date()))] \(message)"
        DispatchQueue.main.async {
            self.logs.append(line)
            if self.logs.count > self.maxLogs {
                self.logs.removeFirst(self.logs.count - self.maxLogs)
            }
        }
    }

    func clear() {
        logs.removeAll()
    }
}

// MARK: - 全局调试函数（任何函数都可调用）

/// 全局调试函数：打开调试开关后，将操作步骤打印到右下角监控浮层。
/// 任何函数（鼠标事件、按钮动作、数据操作等）都可直接调用本函数嵌入调试，
/// 例如：debugLog("左键点击：命中节点")。
/// throttle=true 时对相同消息做 200ms 节流（用于高频调用，如鼠标移动）。
func debugLog(_ message: String, throttle: Bool = false) {
    DebugCenter.shared.log(message, throttle: throttle)
}

// MARK: - 右下角监控浮层（显示最近执行步骤）

struct DebugOverlay: View {
    @ObservedObject var center = DebugCenter.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("调试监控")
                    .font(.caption.bold())
                    .foregroundColor(.white)
                Spacer()
                Button("清空") { center.clear() }
                    .buttonStyle(.plain)
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.7))
            }
            Divider().overlay(Color.white.opacity(0.2))
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(center.logs.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.white.opacity(0.9))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(line)
                        }
                    }
                }
                .onChange(of: center.logs.count) { _, _ in
                    if let last = center.logs.last {
                        withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                    }
                }
            }
        }
        .padding(8)
        .frame(width: 310, height: 200)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.black.opacity(0.78))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.white.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - 模型监控中心（单例）

final class MonitorCenter: ObservableObject {
    static let shared = MonitorCenter()

    /// 监控面板开关
    @Published var isEnabled = false
    /// 监控日志（未来大模型运行日志）
    @Published private(set) var logs: [String] = []
    /// 系统内存使用率（0...1）
    @Published private(set) var memoryUsage: Double = 0
    /// 已使用的交换空间（字节）
    @Published private(set) var swapUsed: UInt64 = 0
    /// 交换空间总量（字节）
    @Published private(set) var swapTotal: UInt64 = 0
    /// 系统压缩内存（字节）
    @Published private(set) var compressedUsed: UInt64 = 0
    /// 当前进行中的生成任务名（队列页展示；当前无排队机制，生成即点即跑）
    @Published private(set) var activeTasks: [String] = []
    /// 权重页三态行（已加载 / 页缓存 / 未加载，1s 轮询刷新）
    @Published private(set) var weightRows: [WeightRow] = []

    enum WeightState: String {
        case loaded = "已加载"
        case cached = "页缓存"
        case unloaded = "未加载"
    }

    /// 单个常驻权重的监控行；filePath 为 nil（LTX 编译图）时只有 已加载/未加载 两态
    struct WeightRow: Identifiable {
        let id: String
        let name: String
        let filePath: String?
        let fileBytes: UInt64
        var loaded: Bool
        var residentBytes: UInt64

        init(id: String, name: String, filePath: String?, loaded: Bool = false) {
            self.id = id
            self.name = name
            self.filePath = filePath
            self.fileBytes = filePath.flatMap(MonitorCenter.fileBytes) ?? 0
            self.loaded = loaded
            self.residentBytes = 0
        }

        var state: WeightState {
            if loaded { return .loaded }
            guard filePath != nil, fileBytes > 0 else { return .unloaded }
            // 未加载但页缓存驻留 ≥ 一半 → 视为页缓存态；接近 0 → 未加载/已清理
            return residentBytes >= fileBytes / 2 ? .cached : .unloaded
        }
    }

    /// 常驻权重静态清单（路径与各调用文件一致）
    private static let weightDefs: [(id: String, name: String, path: String?)] = [
        ("hhdream", "HiDream 骨架", "/Users/huachayui/Downloads/HiDream-O1-Image-Dev-mlx-bf16/model.safetensors"),
        ("dit", "LTX DiT", "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/transformer-distilled.safetensors"),
        ("gemma", "Gemma 文本", "/Users/huachayui/Downloads/ltx2.5/gemma4-12b-ltx-v1/model.safetensors"),
        ("connector", "Connector", "/Users/huachayui/Downloads/ltx2.5/LTX-2.5-MLX-Serve-4bit/connector.safetensors"),
        ("compiled", "LTX 编译图", nil),
    ]
    /// 上一轮各权重状态（用于页缓存回收播报）
    private var lastWeightStates: [String: WeightState] = [:]
    /// mincore 驻留统计放后台队列，避免 1s 轮询阻塞主线程（大文件逐页扫描很重）
    private let residencyQueue = DispatchQueue(label: "monitor.residency")
    private var residencyInFlight = false
    /// 驻留统计节流计数：每 3 秒跑一次（mincore 大文件耗时 ~0.5s/轮，1s 太频繁）
    private var residencyTick = 0

    private let maxLogs = 60
    private var timer: Timer?

    private init() {
        refreshMetrics()
        let t = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.refreshMetrics()
        }
        timer = t
        RunLoop.main.add(t, forMode: .common)
    }

    private func refreshMetrics() {
        memoryUsage = systemMemoryUsage()
        (swapUsed, swapTotal) = systemSwapUsage()
        compressedUsed = systemCompressedUsage()
        refreshWeightResidency()
    }

    /// 轮询权重驻留：主线程只读缓存布尔（秒回），mincore 逐页统计丢后台队列，
    /// 完成后回主线程更新三态并播报页缓存回收
    private func refreshWeightResidency() {
        residencyTick += 1
        guard residencyTick % 3 == 0, !residencyInFlight else { return }
        let flags: [(id: String, loaded: Bool)] = [
            ("hhdream", HiDreamModelCache.shared.hasHiDream),
            ("dit", DiTModelCache.shared.hasDiT),
            ("gemma", TextEncoderCache.shared.hasGemma),
            ("connector", TextEncoderCache.shared.hasConnector),
            ("compiled", ltxCompiledForward.isCompiled),
        ]
        residencyInFlight = true
        residencyQueue.async { [weak self] in
            guard let self else { return }
            var rows: [WeightRow] = []
            for def in Self.weightDefs {
                var row = WeightRow(
                    id: def.id, name: def.name, filePath: def.path,
                    loaded: flags.first { $0.id == def.id }?.loaded ?? false)
                if let path = def.path {
                    row.residentBytes = Self.residentBytes(ofFile: path)
                }
                rows.append(row)
            }
            DispatchQueue.main.async {
                self.applyWeightRows(rows)
                self.residencyInFlight = false
            }
        }
    }

    /// 主线程应用驻留结果：播报页缓存回收并刷新三态行
    private func applyWeightRows(_ rows: [WeightRow]) {
        for row in rows {
            // 状态从 页缓存 → 未加载：内核已回收该权重页缓存，下次重载走冷读
            if lastWeightStates[row.id] == .cached, row.state == .unloaded {
                log("⚠️ \(row.name) 页缓存已回收，下次重载走冷读")
            }
            lastWeightStates[row.id] = row.state
        }
        weightRows = rows
    }

    /// 文件大小（字节），失败为 0
    static func fileBytes(_ path: String) -> UInt64 {
        var st = stat()
        guard stat(path, &st) == 0 else { return 0 }
        return UInt64(st.st_size)
    }

    /// mincore 逐页统计文件在页缓存中的驻留字节数；mmap 只建映射不读内容，成本极低
    /// 必须用 MAP_SHARED：MAP_PRIVATE 会建 shadow object，mincore 查到的是私有对象驻留
    /// 而非文件全局页缓存，表现为刚加载完能显示页缓存、下一轮就失真为 0。
    static func residentBytes(ofFile path: String) -> UInt64 {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return 0 }
        defer { close(fd) }
        let size = fileBytes(path)
        guard size > 0, size < UInt64(Int.max) else { return 0 }
        let len = Int(size)
        guard let addr = mmap(nil, len, PROT_READ, MAP_SHARED, fd, 0), addr != MAP_FAILED else { return 0 }
        defer { munmap(addr, len) }
        let pageSize = Int(vm_page_size)
        let pageCount = (len + pageSize - 1) / pageSize
        var vec = [UInt8](repeating: 0, count: pageCount)
        guard mincore(addr, len, &vec) == 0 else { return 0 }
        var resident: UInt64 = 0
        for i in 0..<pageCount where vec[i] & 1 != 0 {
            resident += UInt64(pageSize)
        }
        return resident
    }

    /// 记录监控日志（未来大模型调用点可调用 monitorLog("...")）
    func log(_ message: String) {
        guard isEnabled else { return }
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        let line = "[\(formatter.string(from: Date()))] \(message)"
        DispatchQueue.main.async {
            self.logs.append(line)
            if self.logs.count > self.maxLogs {
                self.logs.removeFirst(self.logs.count - self.maxLogs)
            }
        }
    }

    func clear() {
        logs.removeAll()
    }

    /// 登记生成任务开始（生成入口调用，主线程安全）
    func taskStart(_ name: String) {
        DispatchQueue.main.async {
            guard !self.activeTasks.contains(name) else { return }
            self.activeTasks.append(name)
        }
    }

    /// 登记生成任务结束（生成入口 defer 调用，主线程安全）
    func taskEnd(_ name: String) {
        DispatchQueue.main.async {
            self.activeTasks.removeAll { $0 == name }
        }
    }
}

/// 全局监控日志函数：打开监控面板后，将大模型运行信息打印到监控浮层。
func monitorLog(_ message: String) {
    MonitorCenter.shared.log(message)
}

/// 获取系统内存使用率（0...1），基于 mach 内核统计。
/// 口径对齐 macOS 活动监视器：已使用 = 活跃页 + 固定页 + 压缩页（不含可回收缓存）。
func systemMemoryUsage() -> Double {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    let pageSize = vm_kernel_page_size
    let usedPages = stats.active_count + stats.wire_count + stats.compressor_page_count
    let total = ProcessInfo.processInfo.physicalMemory
    guard total > 0 else { return 0 }
    return min(max(Double(usedPages) * Double(pageSize) / Double(total), 0), 1)
}

/// 获取系统交换空间使用情况（已用字节, 总字节），基于 sysctl vm.swapusage
func systemSwapUsage() -> (used: UInt64, total: UInt64) {
    var xsw = xsw_usage()
    var size = MemoryLayout<xsw_usage>.size
    let result = sysctlbyname("vm.swapusage", &xsw, &size, nil, 0)
    guard result == 0 else { return (0, 0) }
    return (xsw.xsu_used, xsw.xsu_total)
}

/// 获取系统压缩内存（字节），基于 mach 内核统计（compressor 页数 × 页大小）
func systemCompressedUsage() -> UInt64 {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return UInt64(stats.compressor_page_count) * UInt64(vm_kernel_page_size)
}

/// 内存等级：绿（<60%）→ 黄（60%~85%）→ 红（>85%）
/// 交换空间作为重要补充：即使物理内存不高，只要 swap 使用量大（说明在拿 SSD 硬扛），同样告警
enum MemoryLevel {
    case green, yellow, red

    init(usage: Double, swapUsed: UInt64, swapTotal: UInt64) {
        let swapRatio = swapTotal > 0 ? Double(swapUsed) / Double(swapTotal) : 0
        // 红色：物理内存 >85% 或 交换 >2GB 或 交换占总量 >50%
        if usage >= 0.85 || swapUsed >= 2 << 30 || swapRatio >= 0.5 {
            self = .red
        // 黄色：物理内存 60%~85% 或 交换 >512MB 或 交换占总量 >20%
        } else if usage >= 0.6 || swapUsed >= 512 << 20 || swapRatio >= 0.2 {
            self = .yellow
        } else {
            self = .green
        }
    }

    var color: Color {
        switch self {
        case .green: return .green
        case .yellow: return .yellow
        case .red: return .red
        }
    }

    /// 呼吸频率（次/秒）：内存越高闪得越快
    var breathSpeed: Double {
        switch self {
        case .green: return 0.6
        case .yellow: return 1.4
        case .red: return 2.6
        }
    }

    var label: String {
        switch self {
        case .green: return "内存充足"
        case .yellow: return "内存偏高"
        case .red: return "内存告急"
        }
    }
}

// MARK: - 模型监控呼吸灯按钮

struct ModelMonitorButton: View {
    @ObservedObject var center = MonitorCenter.shared

    var level: MemoryLevel {
        MemoryLevel(usage: center.memoryUsage, swapUsed: center.swapUsed, swapTotal: center.swapTotal)
    }

    var body: some View {
        TimelineView(.animation) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            let v = (sin(t * level.breathSpeed * 2 * .pi) + 1) / 2 // 0...1 呼吸曲线
            let accent = AppSettings.shared.defaultNodeColor // 跟随公共节点主题色（偏好设置/画布右上角共用），同色系明暗呼吸
            Button(action: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    // 面板互斥：打开监控时关闭调试面板
                    DebugCenter.shared.isEnabled = false
                    center.isEnabled.toggle()
                }
            }) {
                Image(systemName: "sparkles")
                    .font(.system(size: 15))
                    .foregroundColor(.white)
                    .frame(width: 32, height: 32)
                    .background(
                        Circle().fill(accent.opacity(0.25 + 0.55 * v))
                    )
                    .overlay(
                        Circle().stroke(accent.opacity(0.4 + 0.6 * v), lineWidth: 1.5)
                    )
                    .shadow(color: accent.opacity(0.3 + 0.7 * v), radius: 4 + 4 * v)
            }
            .buttonStyle(.plain)
            .help("模型监控：\(level.label)（内存 \(Int(center.memoryUsage * 100))% / 交换 \(byteString(center.swapUsed))）")
        }
    }
}

/// 字节数格式化为可读字符串（GB / MB）
func byteString(_ bytes: UInt64) -> String {
    let gb = Double(bytes) / Double(1 << 30)
    if gb >= 1 { return String(format: "%.1f GB", gb) }
    let mb = Double(bytes) / Double(1 << 20)
    return String(format: "%.0f MB", mb)
}

// MARK: - 模型监控浮层（内存监控 + 大模型运行日志）

struct ModelMonitorOverlay: View {
    @ObservedObject var center = MonitorCenter.shared
    @ObservedObject var queue = GenerationQueue.shared
    @State private var tab: MonitorTab = .report

    enum MonitorTab: String, CaseIterable {
        case report = "报告"
        case weights = "权重"
        case queue = "队列"
    }

    /// 常驻权重加载状态（VAE/升频器/音频为每次生成临时加载，无常驻缓存不列出）
    /// 数据由 MonitorCenter 1s 轮询维护，见 center.weightRows

    var level: MemoryLevel {
        MemoryLevel(usage: center.memoryUsage, swapUsed: center.swapUsed, swapTotal: center.swapTotal)
    }

    /// 动态状态：压力评分 + 当前内存策略（压力分 ≥ 进入高压阈值 → 内存优先，否则速度优先；
    /// 刻度与峰值保底 ensureLoose / VAE 动态分块一致，随 MonitorCenter 1s 轮询实时刷新）
    var memoryStatusText: String {
        let ps = SystemMemory.pressureScore
        let strategy = ps >= SystemMemory.looseEnterHysteresis ? "内存优先" : "速度优先"
        return String(format: "压力评分：%.1f 策略：%@", ps, strategy)
    }

    var body: some View {
        let accent = AppSettings.shared.defaultNodeColor // 跟随公共节点主题色（偏好设置/画布右上角共用），同色系表达
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("模型监控")
                    .font(.caption.bold())
                    .foregroundColor(.white)
                Spacer()
                Text("内存 \(Int(center.memoryUsage * 100))%")
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(accent)
                Button("清空") { center.clear() }
                    .buttonStyle(.plain)
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.7))
            }
            // 内存占用条（同色系深浅表达）
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.12))
                    Capsule()
                        .fill(accent.opacity(0.85))
                        .frame(width: max(4, geo.size.width * center.memoryUsage))
                }
            }
            .frame(height: 6)
            HStack(spacing: 8) {
                Text(memoryStatusText)
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(accent)
                Spacer()
                Text("交换 \(byteString(center.swapUsed))")
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(center.swapUsed > 0 ? .orange : .white.opacity(0.5))
                Text("压缩 \(byteString(center.compressedUsed))")
                    .font(.caption2.monospacedDigit())
                    .foregroundColor(center.compressedUsed > 0 ? .purple : .white.opacity(0.5))
            }
            HStack(spacing: 14) {
                ForEach(MonitorTab.allCases, id: \.self) { t in
                    Button(t.rawValue) { tab = t }
                        .buttonStyle(.plain)
                        .font(.caption.bold())
                        .foregroundColor(tab == t ? accent : .white.opacity(0.5))
                        .underline(tab == t, color: accent)
                        .padding(.vertical, 2)
                }
                Spacer()
            }
            Divider().overlay(Color.white.opacity(0.2))
            switch tab {
            case .report: reportView
            case .weights: weightView
            case .queue: queueView
            }
        }
        .padding(8)
        .frame(width: 440, height: 340)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(Color.black.opacity(0.78))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.white.opacity(0.2), lineWidth: 1)
        )
    }

    /// 报告页：模型运行日志
    private var reportView: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 1) {
                if center.logs.isEmpty {
                    Text("等待模型监控…")
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.4))
                } else {
                    ForEach(Array(center.logs.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.white.opacity(0.9))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }

    /// 权重页：常驻权重加载情况
    /// 权重页：常驻权重三态（已加载 / 页缓存 / 未加载），数据来自 MonitorCenter 1s 轮询
    private var weightView: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("权重")
                    .font(.caption2.bold())
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
                Text("已加载 / 页缓存 / 未加载")
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
            }
            ForEach(center.weightRows) { row in
                HStack(spacing: 6) {
                    Circle()
                        .fill(stateColor(row.state))
                        .frame(width: 7, height: 7)
                    Text(row.name)
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.9))
                    Spacer()
                    Text(sizeLabel(row))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white.opacity(0.45))
                    Text(row.state.rawValue)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(stateColor(row.state))
                        .frame(width: 44, alignment: .trailing)
                }
            }
            Spacer(minLength: 0)
        }
        .frame(maxHeight: .infinity)
    }

    private func stateColor(_ state: MonitorCenter.WeightState) -> Color {
        switch state {
        case .loaded: return .green
        case .cached: return .yellow
        case .unloaded: return .white.opacity(0.25)
        }
    }

    private func sizeLabel(_ row: MonitorCenter.WeightRow) -> String {
        if row.state == .cached, row.residentBytes > 0 {
            return String(format: "驻留 %.1fG", Double(row.residentBytes) / 1_000_000_000)
        }
        return row.fileBytes > 0 ? "\(row.fileBytes / 1_000_000_000)G" : "—"
    }

    /// 队列页：生成任务队列（排队中 / 生成中，每任务可取消）
    private var queueView: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("队列")
                    .font(.caption2.bold())
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
                Text("\(queue.tasks.count) 个任务")
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.4))
            }
            if queue.tasks.isEmpty {
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.white.opacity(0.25))
                        .frame(width: 7, height: 7)
                    Text("空闲（无排队任务）")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.45))
                }
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 3) {
                        ForEach(queue.tasks) { task in
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(statusColor(task.status))
                                    .frame(width: 7, height: 7)
                                Text("[\(task.kind.rawValue)]")
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(task.kind == .image ? .cyan : .orange)
                                Text(task.summary)
                                    .font(.system(size: 10))
                                    .foregroundColor(.white.opacity(0.9))
                                    .lineLimit(1)
                                Spacer()
                                if task.status == .pending || task.status == .running {
                                    Button("取消") {
                                        GenerationQueue.shared.cancel(task.id)
                                    }
                                    .buttonStyle(.plain)
                                    .font(.system(size: 9))
                                    .foregroundColor(.red.opacity(0.9))
                                }
                            }
                        }
                    }
                }
                .frame(maxHeight: .infinity)
            }
            Spacer(minLength: 0)
        }
        .frame(maxHeight: .infinity)
    }

    private func statusColor(_ s: QueueTaskStatus) -> Color {
        switch s {
        case .pending: return .yellow
        case .running: return .green
        case .done: return .green.opacity(0.6)
        case .cancelled: return .white.opacity(0.25)
        case .failed: return .red
        }
    }
}

// MARK: - 右下角调试开关 + 浮层挂载

struct DebugControlPanel: View {
    @ObservedObject var center = DebugCenter.shared
    @ObservedObject var monitor = MonitorCenter.shared

    var body: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 8) {
                    if center.isEnabled {
                        DebugOverlay()
                            .transition(.scale(scale: 0.9).combined(with: .opacity))
                    }
                    if monitor.isEnabled {
                        ModelMonitorOverlay()
                            .transition(.scale(scale: 0.9).combined(with: .opacity))
                    }
                    ModelMonitorButton()
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            // 面板互斥：打开调试时关闭监控面板
                            monitor.isEnabled = false
                            center.isEnabled.toggle()
                        }
                    }) {
                        Image(systemName: center.isEnabled ? "ladybug.fill" : "ladybug")
                            .font(.system(size: 14))
                            .foregroundColor(center.isEnabled ? .white : .secondary)
                            .frame(width: 32, height: 32)
                            .background(Circle().fill(center.isEnabled ? Color.blue : Color.white.opacity(0.92)))
                            .overlay(Circle().stroke(Color.gray.opacity(0.3), lineWidth: 1))
                            .shadow(color: .black.opacity(0.15), radius: 3, y: 1)
                    }
                    .buttonStyle(.plain)
                    .help("调试开关：打开后右下角显示操作步骤")
                }
                .padding(12)
            }
        }
    }
}
