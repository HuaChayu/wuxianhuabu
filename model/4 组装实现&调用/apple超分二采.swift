//
//  apple超分二采.swift
//  无限画布
//
// ============================================================
//  文件作用：第三套二采方案 —— Apple VideoToolbox 超分（VTSuperResolutionScaler / VTFrameProcessor）。
//  与既有两套二采（LTX 像素桥 IC 二采 / CQ 清晰度增强）**并存互斥、本通道优先**：
//  开关开启时 H3 stage1 的内存像素优先走本通道；任何不可用/失败（运行期不支持、
//  模型资产下载失败、源尺寸越界、源像素格式无交集、逐帧处理出错、被取消）都只打日志并返回 nil，
//  由上游回退原 CQ/IC 二采，绝不中断生成。
//
//  完整链路：
//    MLXArray [1,3,T,H,W] f32 值域[-1,1]（H3 stage1 VAE 解码像素，与 readVideoFramesToBCFHW 同域）
//      → 逐帧填 CVPixelBuffer（像素格式 = 运行期 supportedPixelFormats ∩ 本通道可实现填充集合）
//      → VTFrameProcessor + VTSuperResolutionScalerConfiguration（可选预计算光流）
//      → 输出 CVPixelBuffer（destinationPixelBufferAttributes + IOSurface 支撑）
//      → VTPixelTransferSession 转写到 writeMp4 池缓冲（写盘格式运行期预检择优选）
//      → 编码落盘（复用 模型公共函数-通用.swift writeMp4 的 overrideFrames 直写通道）
//      → 复用 muxSourceAudioOnto 沿用 stage1 源音轨
//
//  能力探测全部以运行期为准（无任何硬编码倍率/分辨率上限）：
//    isSupported / supportedScaleFactors / supportedPixelFormats /
//    sourcePixelBufferAttributes / destinationPixelBufferAttributes /
//    maximumDimensions / minimumDimensions / supportedRevisions / defaultRevision
//
//  偏好设置（偏好设置.swift → AppSettings）：
//    videoUseAppleSR / appleSRScaleFactor / appleSRQualityRawValue / appleSRUsePrecomputedFlow
//    质量优先级：系统公开候选仅 QualityPrioritization.normal(rawValue=1)，面板只列该项
//  环境变量（优先级高于设置项）：
//    LTX_APPLE_SR=1/0、LTX_APPLE_SR_SCALE=<倍率>、LTX_APPLE_SR_QUALITY=<质量名>、LTX_APPLE_SR_FLOW=1/0
//
//  未改动：IC 与 CQ 两条老路径、LTX 原生生成管线、H3 内存桥语义（本文件只读 pixels）。
// ============================================================

import Foundation
import Combine
import ObjectiveC
import CoreVideo
import CoreMedia
import AVFoundation
import VideoToolbox
import MLX

// ============================================================
// MARK: - 一、开关与参数解析（设置项 + 环境变量覆盖）
// ============================================================

/// 环境变量布尔解析：1/true/yes/on → true；0/false/no/off → false；未设/其它 → nil（回退设置项）
func appleSREnvFlag(_ key: String) -> Bool? {
    guard let raw = ProcessInfo.processInfo.environment[key]?
        .trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !raw.isEmpty else { return nil }
    switch raw {
    case "1", "true", "yes", "on": return true
    case "0", "false", "no", "off": return false
    default: return nil
    }
}

/// 环境变量整型解析（LTX_APPLE_SR_SCALE 等）
func appleSREnvInt(_ key: String) -> Int? {
    guard let raw = ProcessInfo.processInfo.environment[key]?
        .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
    return Int(raw)
}

/// 第三套二采总开关：环境变量 LTX_APPLE_SR 优先（1/0），未设时取设置项 videoUseAppleSR。
func appleSRSecondPassEnabled() -> Bool {
    if let v = appleSREnvFlag("LTX_APPLE_SR") {
        pipelineLog("ℹ️ [AppleSR] LTX_APPLE_SR=\(v ? "1" : "0") 覆盖设置项（videoUseAppleSR=\(AppSettings.shared.videoUseAppleSR)）")
        return v
    }
    return AppSettings.shared.videoUseAppleSR
}

/// 请求倍率原始值：LTX_APPLE_SR_SCALE 优先，其次设置项 appleSRScaleFactor。
/// 面板已取消倍率下拉：默认固定 ×4（高质量款本机运行期唯一候选）。
/// 仅作"偏好值"，实际倍率必须经运行期 supportedScaleFactors 校验后才使用（不硬编码生效倍率）。
func appleSRRequestedScaleFactorRaw() -> Int {
    if let v = appleSREnvInt("LTX_APPLE_SR_SCALE") {
        pipelineLog("ℹ️ [AppleSR] LTX_APPLE_SR_SCALE=\(v) 覆盖设置项（appleSRScaleFactor=\(AppSettings.shared.appleSRScaleFactor)）")
        return v
    }
    // 0 为此前“自动”的旧偏好残留，一并归一为 4；生效值仍由 supportedScaleFactors 校验后决定。
    let pref = AppSettings.shared.appleSRScaleFactor
    return pref > 0 ? pref : 4
}

/// 质量优先级 token → 原始值（token 来自运行期枚举候选名，如 "normal"）
func appleSRQualityRawValue(fromToken token: String) -> Int? {
    appleSRQualityOptions().first { $0.token == token }?.rawValue
}

/// 请求质量优先级原始值：LTX_APPLE_SR_QUALITY 优先（名字或整数），其次设置项 appleSRQualityRawValue。
func appleSRRequestedQualityRawValue() -> Int {
    let env = ProcessInfo.processInfo.environment["LTX_APPLE_SR_QUALITY"]?
        .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
    if !env.isEmpty {
        if let mapped = appleSRQualityRawValue(fromToken: env) {
            pipelineLog("ℹ️ [AppleSR] LTX_APPLE_SR_QUALITY=\(env)→原始值\(mapped) 覆盖设置项（appleSRQualityRawValue=\(AppSettings.shared.appleSRQualityRawValue)）")
            return mapped
        }
        pipelineLog("⚠️ [AppleSR] LTX_APPLE_SR_QUALITY=\(env) 无法识别，忽略（运行期候选项：\(appleSRQualityOptions().map { $0.token }.joined(separator: "/"))）")
    }
    return AppSettings.shared.appleSRQualityRawValue
}

/// 预计算光流开关：LTX_APPLE_SR_FLOW 优先，其次设置项 appleSRUsePrecomputedFlow。
func appleSRRequestedUsePrecomputedFlow() -> Bool {
    if let v = appleSREnvFlag("LTX_APPLE_SR_FLOW") { return v }
    return AppSettings.shared.appleSRUsePrecomputedFlow
}

// ============================================================
// MARK: - 二、运行期候选项（供偏好设置面板生成选项，全部来自运行期 API）
// ============================================================

/// 倍率候选项（面板 Picker 用）：value = 运行期 supportedScaleFactors 里的真实倍率
struct AppleSRScaleOption: Identifiable, Hashable {
    let value: Int
    var id: Int { value }
    /// 面板展示文本
    var title: String { "×\(value)" }
}

/// 质量优先级候选项（面板 Picker 用）：rawValue = 运行期枚举原始值，token = 枚举候选名
struct AppleSRQualityOption: Identifiable, Hashable {
    let rawValue: Int
    let token: String
    var id: Int { rawValue }
    /// 面板展示文本（系统枚举为可扩展 ObjC 枚举，String(describing:) 只会打印 rawValue 形式，
    /// 故此处按候选名做显式映射；系统新增候选时在此补一行即可）
    var title: String {
        switch token.lowercased() {
        case "normal": return "Normal（系统唯一有效值）"
        default:
            guard let first = token.first else { return token }
            return String(first).uppercased() + token.dropFirst()
        }
    }
}

/// 运行期倍率候选：直接来自 class 属性 supportedScaleFactors（未探测/不支持时为空数组）。
func appleSRScaleOptions() -> [AppleSRScaleOption] {
    guard VTSuperResolutionScalerConfiguration.isSupported else { return [] }
    return VTSuperResolutionScalerConfiguration.supportedScaleFactors.map { AppleSRScaleOption(value: $0) }
}

/// 运行期质量优先级候选：**只列系统公开的枚举候选**。
/// 重要：QualityPrioritization 在 Swift 侧以「可扩展 ObjC 枚举」导入，init?(rawValue:) 对任意整数
/// 都返回非 nil（不校验），早期用 0...8 扫描会列出 0、2…8 共 8 个未定义值，面板才会出现一堆
/// 看不懂的「质量 0/2/3…」。官方文档公开的候选只有 .normal（rawValue = 1），故此处只纳入它；
/// 系统将来新增候选时，在此数组补一项即可。
func appleSRQualityOptions() -> [AppleSRQualityOption] {
    let documented: [VTSuperResolutionScalerConfiguration.QualityPrioritization] = [.normal]
    return documented.map { AppleSRQualityOption(rawValue: $0.rawValue, token: "normal") }
}

/// 倍率候选文本（日志/面板兜底展示）：如 "×2/×3"
func appleSRScaleOptionsSummary() -> String {
    let opts = appleSRScaleOptions()
    return opts.isEmpty ? "无（本机不支持或未探测）" : opts.map { $0.title }.joined(separator: "/")
}

// ============================================================
// MARK: - 三、格式工具与运行期能力探测
// ============================================================

/// FourCC 文本（像素格式日志用）
func appleSRFourCC(_ format: OSType) -> String {
    let bytes: [UInt8] = [
        UInt8((format >> 24) & 0xff), UInt8((format >> 16) & 0xff),
        UInt8((format >> 8) & 0xff), UInt8(format & 0xff),
    ]
    let text = String(bytes: bytes.map { ($0 >= 32 && $0 < 127) ? $0 : UInt8(ascii: ".") }, encoding: .ascii) ?? ""
    return "\(text)('\(format)')"
}

/// 常用像素格式的可读描述（日志便于核对）
func appleSRFormatDescription(_ format: OSType) -> String {
    switch format {
    case kCVPixelFormatType_32BGRA: return "BGRA 8bit \(appleSRFourCC(format))"
    case kCVPixelFormatType_64RGBAHalf: return "RGhA 半精度 RGBA 64bit \(appleSRFourCC(format))"
    case kCVPixelFormatType_32ARGB: return "ARGB 8bit \(appleSRFourCC(format))"
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420v 双平面 8bit VideoRange \(appleSRFourCC(format))"
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420f 双平面 8bit FullRange \(appleSRFourCC(format))"
    case kCVPixelFormatType_422YpCbCr10: return "v210 4:2:2 10bit \(appleSRFourCC(format))"
    default: return appleSRFourCC(format)
    }
}

/// 从像素缓冲属性字典里取像素格式（属性可能给单个 NSNumber 或数组，取第一个）
func appleSRPixelFormat(fromAttributes attributes: [String: Any]) -> OSType? {
    guard let raw = attributes[kCVPixelBufferPixelFormatTypeKey as String] else { return nil }
    if let n = raw as? NSNumber { return OSType(truncating: n) }
    if let list = raw as? [NSNumber], let first = list.first { return OSType(truncating: first) }
    if let list = raw as? [OSType], let first = list.first { return first }
    return nil
}

/// 以 ObjC 运行期反射读取 configuration 类的尺寸约束（`maximumDimensions` / `minimumDimensions`）。
/// 说明：这两个成员声明在 VTFrameProcessorConfiguration 协议上，部分 configuration 类并未在自身接口里
/// 暴露（Swift 侧未必可见），故一律运行期探测：能响应就取真实值，不响应即视为"未声明约束"。
/// 该值只作面板提示与探测候选，真实可用性最终由 init? / startSession 运行期判定（init? 会在尺寸越界时返回 nil）。
func appleSRClassDimensions(_ cls: AnyClass, selectorName: String) -> CMVideoDimensions? {
    let selector = NSSelectorFromString(selectorName)
    guard let metaClass = object_getClass(cls), class_respondsToSelector(metaClass, selector),
          let implementation = class_getMethodImplementation(metaClass, selector) else { return nil }
    // CMVideoDimensions 为 8 字节整型结构体，按寄存器返回，可直接按 C 调用约定取回
    typealias DimensionGetter = @convention(c) (AnyClass, Selector) -> CMVideoDimensions
    let getter = unsafeBitCast(implementation, to: DimensionGetter.self)
    let value = getter(cls, selector)
    // 合理性校验：异常值（判空/其他返回约定导致）一律视为未声明，避免误当约束
    guard value.width > 0, value.height > 0, value.width <= 65536, value.height <= 65536 else { return nil }
    return value
}

/// 运行期能力探测结果
struct AppleSRCapability {
    /// 探测时能成功构造的配置（用于读取 supportedPixelFormats / 属性字典 / 模型状态）
    let config: VTSuperResolutionScalerConfiguration
    /// 运行期倍率候选（class 属性 supportedScaleFactors）
    let scaleFactors: [Int]
    /// 运行期像素格式候选（该 config 的 supportedPixelFormats）
    let pixelFormats: [OSType]
    /// 单行探测摘要（面板与日志共用）
    let summary: String
}

/// 能力探测摘要：倍率 / 像素格式 / revision / 尺寸范围 / 源目标属性字典
func appleSRCapabilitySummary(_ config: VTSuperResolutionScalerConfiguration) -> String {
    let factors = VTSuperResolutionScalerConfiguration.supportedScaleFactors.map(String.init).joined(separator: "/")
    let formats = config.supportedPixelFormats.map { appleSRFormatDescription($0) }.joined(separator: "、")
    let revisions = VTSuperResolutionScalerConfiguration.supportedRevisions.map(String.init).joined(separator: "/")
    let defaultRev = VTSuperResolutionScalerConfiguration.defaultRevision.rawValue
    // 尺寸约束按运行期反射读取：类未实现该协议成员时视为未声明（真实越界判定仍由 init? 完成）
    let maxText: String
    if let d = appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "maximumDimensions") {
        maxText = "\(d.width)×\(d.height)"
    } else {
        maxText = "无上限声明"
    }
    let minText: String
    if let d = appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "minimumDimensions") {
        minText = "\(d.width)×\(d.height)"
    } else {
        minText = "未声明"
    }
    return "支持倍率[\(factors)] 像素格式[\(formats)] revision[\(revisions)]（默认 \(defaultRev)） 输入尺寸范围[\(minText)…\(maxText)] "
        + "源属性=\(config.sourcePixelBufferAttributes) 目标属性=\(config.destinationPixelBufferAttributes)"
}

/// 倍率尝试顺序：请求值（运行期支持时）优先 → 其余候选按"接近 2 优先"排序。
/// 全部来自运行期 supportedScaleFactors，不做硬编码。
func appleSRScaleLadder(preferred: Int, supported: [Int]) -> [Int] {
    var out: [Int] = []
    if supported.contains(preferred) { out.append(preferred) }
    out.append(contentsOf: supported.filter { $0 != preferred }.sorted { abs($0 - 2) < abs($1 - 2) })
    return out
}

/// 探测尺寸候选：优先取运行期尺寸上下界（最小值 / 最大值），再退化到若干常规尺寸。
/// 哪一组真正可用由 init? 运行期决定，不在本文件硬编码"上限"。
func appleSRProbeSizes(maxDim: CMVideoDimensions?, minDim: CMVideoDimensions?) -> [(Int, Int)] {
    var out: [(Int, Int)] = []
    if let minDim, minDim.width > 0, minDim.height > 0 {
        out.append((Int(minDim.width), Int(minDim.height)))
    }
    if let maxDim, maxDim.width > 0, maxDim.height > 0 {
        out.append((Int(maxDim.width), Int(maxDim.height)))
    }
    out.append(contentsOf: [(256, 256), (512, 288), (640, 360), (960, 544), (1280, 720), (1920, 1080)])
    var seen = Set<String>()
    return out.filter { seen.insert("\($0.0)x\($0.1)").inserted }
}

/// 构造超分配置（唯一入口，含运行期尺寸上下界预检 + revision 阶梯回退）。
/// 尺寸越界 / 倍率不支持 / revision 不支持时返回 nil（调用方据此回退原 CQ/IC 二采）。
func appleSRMakeConfiguration(frameWidth: Int, frameHeight: Int, scaleFactor: Int,
                              usePrecomputedFlow: Bool, qualityRawValue: Int) -> VTSuperResolutionScalerConfiguration? {
    guard frameWidth > 0, frameHeight > 0, scaleFactor > 0 else { return nil }
    // 尺寸/倍率合法性交由 init? 运行期判定（官方文档：尺寸越界或 revision 不支持时 init? 返回 nil），
    // 本文件不硬编码任何分辨率上限，仅在反射可读出声明值时提前短路一次明显越界的尝试。
    if let maxDim = appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "maximumDimensions") {
        guard frameWidth <= Int(maxDim.width), frameHeight <= Int(maxDim.height) else { return nil }
    }
    if let minDim = appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "minimumDimensions") {
        guard frameWidth >= Int(minDim.width), frameHeight >= Int(minDim.height) else { return nil }
    }
    // 质量优先级：只接受系统公开候选（见 appleSRQualityOptions）；旧偏好里遗留的未定义值一律回退 .normal
    let allowedQualities = appleSRQualityOptions().compactMap {
        VTSuperResolutionScalerConfiguration.QualityPrioritization(rawValue: $0.rawValue)
    }
    let quality = allowedQualities.first { $0.rawValue == qualityRawValue } ?? .normal
    var revisions: [VTSuperResolutionScalerConfiguration.Revision] = [VTSuperResolutionScalerConfiguration.defaultRevision]
    for raw in VTSuperResolutionScalerConfiguration.supportedRevisions where raw != revisions[0].rawValue {
        if let rev = VTSuperResolutionScalerConfiguration.Revision(rawValue: raw) { revisions.append(rev) }
    }
    for rev in revisions {
        if let cfg = VTSuperResolutionScalerConfiguration(frameWidth: frameWidth, frameHeight: frameHeight,
                                                         scaleFactor: scaleFactor, inputType: .video,
                                                         usePrecomputedFlow: usePrecomputedFlow,
                                                         qualityPrioritization: quality, revision: rev) {
            return cfg
        }
    }
    return nil
}

/// 无真实任务尺寸时的能力探测：用运行期尺寸上下界 + 倍率候选找到第一个能构造成功的配置。
func appleSRProbeCapability(preferredScale: Int, usePrecomputedFlow: Bool, qualityRawValue: Int) -> AppleSRCapability? {
    guard VTSuperResolutionScalerConfiguration.isSupported else { return nil }
    let factors = VTSuperResolutionScalerConfiguration.supportedScaleFactors
    guard !factors.isEmpty else { return nil }
    let ladder = appleSRScaleLadder(preferred: preferredScale, supported: factors)
    let sizes = appleSRProbeSizes(
        maxDim: appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "maximumDimensions"),
        minDim: appleSRClassDimensions(VTSuperResolutionScalerConfiguration.self, selectorName: "minimumDimensions"))
    for (w, h) in sizes {
        for f in ladder {
            guard let cfg = appleSRMakeConfiguration(frameWidth: w, frameHeight: h, scaleFactor: f,
                                                     usePrecomputedFlow: usePrecomputedFlow,
                                                     qualityRawValue: qualityRawValue) else { continue }
            return AppleSRCapability(config: cfg, scaleFactors: factors,
                                     pixelFormats: cfg.supportedPixelFormats,
                                     summary: appleSRCapabilitySummary(cfg))
        }
    }
    return nil
}

// ============================================================
// MARK: - 四、运行期状态（偏好设置面板共享：能力 / 模型状态 / 下载进度）
// ============================================================

/// Apple 超分运行期状态：面板展示能力探测结果与模型资产状态，并提供手动下载入口。
/// 全部字段更新切回主线程（@Published 刷新 UI）。
final class AppleSRRuntimeState: ObservableObject {
    static let shared = AppleSRRuntimeState()

    /// 是否支持 VTSuperResolutionScaler（nil = 尚未探测）
    @Published private(set) var supported: Bool? = nil
    /// 探测摘要（倍率/像素格式/尺寸范围/revision/属性字典）
    @Published private(set) var capabilitySummary: String = "尚未探测"
    /// 运行期倍率候选项
    @Published private(set) var scaleFactors: [Int] = []
    /// 运行期像素格式候选项（可读文本）
    @Published private(set) var pixelFormats: [String] = []
    /// 模型资产是否就绪
    @Published private(set) var modelReady: Bool = false
    /// 模型资产下载进度（0~1）
    @Published private(set) var modelProgress: Double = 0
    /// 模型资产状态文本
    @Published private(set) var modelStatusText: String = "未探测"
    /// 手动下载进行中
    @Published private(set) var downloading: Bool = false
    /// 最近一次失败原因（面板提示）
    @Published private(set) var lastError: String = ""

    private var manualDownloadTask: Task<Void, Never>? = nil

    private init() {}

    private func onMain(_ block: @escaping () -> Void) {
        if Thread.isMainThread { block() } else { DispatchQueue.main.async(execute: block) }
    }

    /// 刷新运行期能力（面板 onAppear / 「重新探测」按钮调用；同步系统查询，开销极小）
    func refreshCapability(scaleFactor: Int, qualityRawValue: Int, usePrecomputedFlow: Bool) {
        let supportedNow = VTSuperResolutionScalerConfiguration.isSupported
        let factors = supportedNow ? VTSuperResolutionScalerConfiguration.supportedScaleFactors : []
        let cap = supportedNow ? appleSRProbeCapability(preferredScale: scaleFactor,
                                                       usePrecomputedFlow: usePrecomputedFlow,
                                                       qualityRawValue: qualityRawValue) : nil
        let formats = cap?.pixelFormats.map { appleSRFormatDescription($0) } ?? []
        let summary: String
        if !supportedNow {
            summary = "本机不支持 VTSuperResolutionScaler（isSupported=false）"
        } else if let cap {
            summary = cap.summary
        } else {
            summary = "已支持，但运行期未找到可构造的探测配置（详见日志）"
        }
        onMain {
            self.supported = supportedNow
            self.scaleFactors = factors
            self.pixelFormats = formats
            self.capabilitySummary = summary
        }
        refreshModelStatus(scaleFactor: scaleFactor, qualityRawValue: qualityRawValue, usePrecomputedFlow: usePrecomputedFlow)
    }

    /// 刷新模型资产状态（不触发下载，仅查询 configurationModelStatus / 进度）
    func refreshModelStatus(scaleFactor: Int, qualityRawValue: Int, usePrecomputedFlow: Bool) {
        guard let cap = appleSRProbeCapability(preferredScale: scaleFactor,
                                               usePrecomputedFlow: usePrecomputedFlow,
                                               qualityRawValue: qualityRawValue) else {
            onMain {
                self.modelReady = false
                self.modelStatusText = self.supported == true ? "无可用配置，无法查询" : "本机不支持"
            }
            return
        }
        let cfg = cap.config
        let status = cfg.configurationModelStatus
        let pct = Double(cfg.configurationModelPercentageAvailable)
        let text: String
        switch status {
        case .ready: text = "模型已就绪"
        case .downloading: text = "模型下载中 \(Int(pct * 100))%"
        case .downloadRequired: text = "模型未就绪（需要下载，当前 \(Int(pct * 100))%）"
        @unknown default: text = "模型状态未知"
        }
        onMain {
            self.modelReady = (status == .ready)
            self.modelProgress = (status == .ready) ? 1 : pct
            self.modelStatusText = text
        }
    }

    /// 手动下载入口（面板按钮）：后台任务执行，进度回主线程刷新；已完成则直接置就绪。
    func startManualDownload(scaleFactor: Int, qualityRawValue: Int, usePrecomputedFlow: Bool) {
        guard !downloading else { return }
        onMain {
            self.downloading = true
            self.lastError = ""
            self.modelStatusText = "准备下载…"
        }
        manualDownloadTask = Task { [weak self] in
            guard let self else { return }
            defer { self.onMain { self.downloading = false } }
            guard let cap = appleSRProbeCapability(preferredScale: scaleFactor,
                                                   usePrecomputedFlow: usePrecomputedFlow,
                                                   qualityRawValue: qualityRawValue) else {
                self.onMain {
                    self.modelReady = false
                    self.modelStatusText = "探测失败：无可用配置"
                    self.lastError = "无可构造的 VTSuperResolutionScalerConfiguration"
                }
                return
            }
            let cfg = cap.config
            if cfg.configurationModelStatus == .ready {
                self.onMain {
                    self.modelReady = true
                    self.modelProgress = 1
                    self.modelStatusText = "模型已就绪"
                    self.lastError = ""
                }
                return
            }
            do {
                try await appleSRDownloadConfigurationModel(config: cfg) { p in
                    self.onMain {
                        self.modelReady = false
                        self.modelProgress = p
                        self.modelStatusText = "模型下载中 \(Int(p * 100))%"
                    }
                }
                let ok = await appleSRWaitForModelReady(config: cfg, isCancelled: { false })
                self.onMain {
                    self.modelReady = ok
                    self.modelProgress = ok ? 1 : Double(cfg.configurationModelPercentageAvailable)
                    self.modelStatusText = ok ? "模型已就绪" : "下载未完成（可重试）"
                    self.lastError = ok ? "" : "下载未完成"
                }
            } catch {
                self.onMain {
                    self.modelReady = false
                    self.modelStatusText = "下载失败"
                    self.lastError = error.localizedDescription
                }
            }
        }
    }

    /// 生成流程内回报能力不支持（面板可见）
    static func reportUnsupported() {
        shared.onMain {
            shared.supported = false
            shared.capabilitySummary = "本机不支持 VTSuperResolutionScaler（isSupported=false）"
            shared.modelStatusText = "本机不支持"
        }
    }

    /// 生成流程内回报模型资产状态/进度（主线程刷新面板）
    static func reportModel(ready: Bool, progress: Double, text: String) {
        shared.onMain {
            shared.modelReady = ready
            shared.modelProgress = progress
            shared.modelStatusText = text
        }
    }
}

// ============================================================
// MARK: - 五、像素缓冲：创建 / 属性解析 / 逐帧填充（MLXArray → CVPixelBuffer）
// ============================================================

/// 按指定像素格式 + 属性字典创建 IOSurface 支撑的 CVPixelBuffer
/// （VTFrameProcessorFrame 要求 IOSurface 支撑；属性字典来自运行期配置，仅补格式/尺寸/IOSurface 三项）
func appleSRNewPixelBuffer(width: Int, height: Int, pixelFormat: OSType,
                           attributes: [String: Any], requireIOSurface: Bool) -> CVPixelBuffer? {
    var attrs = attributes
    attrs[kCVPixelBufferPixelFormatTypeKey as String] = pixelFormat
    attrs[kCVPixelBufferWidthKey as String] = width
    attrs[kCVPixelBufferHeightKey as String] = height
    if requireIOSurface, attrs[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
        attrs[kCVPixelBufferIOSurfacePropertiesKey as String] = [:] as CFDictionary
    }
    var pb: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat,
                                     attrs as CFDictionary, &pb)
    guard status == kCVReturnSuccess, let buffer = pb else {
        pipelineLog("⚠️ [AppleSR] CVPixelBufferCreate 失败（\(width)×\(height) \(appleSRFourCC(pixelFormat))，OSStatus \(status)）")
        return nil
    }
    if requireIOSurface, CVPixelBufferGetIOSurface(buffer) == nil {
        pipelineLog("⚠️ [AppleSR] 创建的像素缓冲无 IOSurface 支撑（\(appleSRFourCC(pixelFormat))），VTFrameProcessorFrame 可能拒绝")
    }
    return buffer
}

/// 本通道可实现"逐帧直填"的源像素格式集合（固定优先级）：
/// 半精度 RGBA（RGhA，本机 VTSuperResolutionScaler 唯一声明格式，直填零量化）
/// → RGB 直填（BGRA → ARGB）→ 双平面 YCbCr 8bit（420v → 420f）。
/// 是否真的可用由运行期 supportedPixelFormats 决定（交集为空则本通道放弃，回退 CQ/IC）。
private let appleSRSrcRGBFormats: [OSType] = [kCVPixelFormatType_64RGBAHalf,
                                              kCVPixelFormatType_32BGRA,
                                              kCVPixelFormatType_32ARGB]
private let appleSRSrcYCbCrFormats: [OSType] = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]

/// 可选源像素格式：运行期 supportedPixelFormats ∩ 本通道可实现填充集合；交集为空返回 nil。
func appleSRPickSourceFormat(config: VTSuperResolutionScalerConfiguration) -> OSType? {
    let supported = config.supportedPixelFormats
    let candidates = appleSRSrcRGBFormats + appleSRSrcYCbCrFormats
    if let hit = candidates.first(where: { supported.contains($0) }) {
        pipelineLog("🧩 [AppleSR] 源像素格式选定：\(appleSRFormatDescription(hit))；运行期支持[\(supported.map { appleSRFourCC($0) }.joined(separator: ","))]")
        return hit
    }
    pipelineLog("⚠️ [AppleSR] 运行期支持源像素格式[\(supported.map { appleSRFourCC($0) }.joined(separator: ","))]"
        + "与本通道可实现填充集合[\(candidates.map { appleSRFourCC($0) }.joined(separator: ","))]无交集，回退原 CQ/IC 二采")
    return nil
}

/// f32 [-1,1] → 8bit（截断取整，与工程既有 f32→8bit 填充口径一致）
@inline(__always) private func appleSRByte(_ v: Float) -> UInt8 {
    let x = (v + 1) * 127.5
    return UInt8(min(max(x, 0), 255))
}

/// 半精度 RGBA（RGhA，每像素 8 字节，通道序 R/G/B/A）逐帧填充：
/// R/G/B 由 f32 [-1,1] 映射到 [0,1] 后转 half（与 BGRA/ARGB/420 三条分支同一值域口径），A 填 1.0。
///
/// 64RGBAHalf 是 VideoToolbox 的全范围线性浮点像素格式，官方约定值域为 [0,1]；
/// 直接写入 f32 [-1,1] 的负值会被超分器当作"超过白点"解释：
/// 实测输入 -1.0 → 输出 1.0908（10bit 转写后为 940 满白），造成暗部反白、整帧过曝闪烁。
private func appleSRFillRGBAHalf(floats: [Float], width: Int, height: Int,
                                 base: UnsafeMutableRawPointer, bytesPerRow: Int) {
    let hw = width * height
    let p = base.assumingMemoryBound(to: UInt16.self)
    let rowStride = bytesPerRow / 2   // half 为 2 字节，按 UInt16 步进
    let alpha = Float16(1).bitPattern
    // f32 [-1,1] → [0,1]，并 clamp 防 stage1 越界值
    @inline(__always) func unit(_ v: Float) -> UInt16 {
        Float16(min(max((v + 1) * 0.5, 0), 1)).bitPattern
    }
    for y in 0..<height {
        let row = p + y * rowStride
        let rowOff = y * width
        for x in 0..<width {
            let i = rowOff + x
            let o = x * 4
            row[o] = unit(floats[i])
            row[o + 1] = unit(floats[hw + i])
            row[o + 2] = unit(floats[2 * hw + i])
            row[o + 3] = alpha
        }
    }
}

/// BGRA / ARGB 逐帧填充（R=ch0、G=ch1、B=ch2，均已还原到 [0,1] 前的 [-1,1]）
private func appleSRFillRGBA(floats: [Float], width: Int, height: Int,
                             base: UnsafeMutableRawPointer, bytesPerRow: Int, isBGRA: Bool) {
    let hw = width * height
    let p = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        let row = p + y * bytesPerRow
        let rowOff = y * width
        for x in 0..<width {
            let i = rowOff + x
            let r = appleSRByte(floats[i])
            let g = appleSRByte(floats[hw + i])
            let b = appleSRByte(floats[2 * hw + i])
            let o = x * 4
            if isBGRA {
                row[o] = b; row[o + 1] = g; row[o + 2] = r; row[o + 3] = 255
            } else {
                row[o] = 255; row[o + 1] = r; row[o + 2] = g; row[o + 3] = b
            }
        }
    }
}

/// 双平面 4:2:0 8bit 逐帧填充（平面0=Y，平面1=CbCr 交错；BT.709 系数 + 按 420v/420f 走 limited/full 量化）
private func appleSRFill420(floats: [Float], width: Int, height: Int,
                            yPlane: UnsafeMutableRawPointer, yStride: Int,
                            cPlane: UnsafeMutableRawPointer, cStride: Int, fullRange: Bool) {
    let hw = width * height
    let yp = yPlane.assumingMemoryBound(to: UInt8.self)
    // 亮度平面
    for y in 0..<height {
        let row = yp + y * yStride
        let rowOff = y * width
        for x in 0..<width {
            let i = rowOff + x
            let r = (floats[i] + 1) * 0.5
            let g = (floats[hw + i] + 1) * 0.5
            let b = (floats[2 * hw + i] + 1) * 0.5
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let v = fullRange ? (luma * 255) : (luma * 219 + 16)
            row[x] = UInt8(min(max(v, 0), 255).rounded())
        }
    }
    // 色度平面（4:2:0：2×2 盒平均后再做 RGB→CbCr）
    let cp = cPlane.assumingMemoryBound(to: UInt8.self)
    let cw = width / 2, ch = height / 2
    let scale: Float = fullRange ? 127.5 : 224
    for y in 0..<ch {
        let row = cp + y * cStride
        for x in 0..<cw {
            var sr: Float = 0, sg: Float = 0, sb: Float = 0
            for dy in 0..<2 {
                for dx in 0..<2 {
                    let i = (y * 2 + dy) * width + (x * 2 + dx)
                    sr += (floats[i] + 1) * 0.5
                    sg += (floats[hw + i] + 1) * 0.5
                    sb += (floats[2 * hw + i] + 1) * 0.5
                }
            }
            let r = sr * 0.25, g = sg * 0.25, b = sb * 0.25
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            let cb = (b - luma) / 1.8556
            let cr = (r - luma) / 1.5748
            row[x * 2] = UInt8(min(max(cb * scale + 128, 0), 255).rounded())
            row[x * 2 + 1] = UInt8(min(max(cr * scale + 128, 0), 255).rounded())
        }
    }
}

/// 把 MLXArray 的第 frameIndex 帧（CHW f32 [-1,1]）写进指定格式的源像素缓冲（调用方负责 lock/unlock）
func appleSRFillSourceBuffer(_ buffer: CVPixelBuffer, pixels: MLXArray, frameIndex: Int,
                             width: Int, height: Int, format: OSType) -> Bool {
    let shape = pixels.shape
    guard shape.count == 5, shape[1] == 3, frameIndex >= 0, frameIndex < shape[2],
          shape[3] == height, shape[4] == width else { return false }
    let frame = pixels[0 ..< 1, 0 ..< 3, frameIndex ..< (frameIndex + 1), 0 ..< height, 0 ..< width]
        .reshaped([3, height, width])
    let floats = frame.asArray(Float.self)
    guard floats.count == 3 * width * height else { return false }

    switch format {
    case kCVPixelFormatType_64RGBAHalf:
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return false }
        appleSRFillRGBAHalf(floats: floats, width: width, height: height, base: base,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer))
        return true
    case kCVPixelFormatType_32BGRA, kCVPixelFormatType_32ARGB:
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return false }
        appleSRFillRGBA(floats: floats, width: width, height: height, base: base,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                        isBGRA: format == kCVPixelFormatType_32BGRA)
        return true
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
        guard width % 2 == 0, height % 2 == 0, CVPixelBufferGetPlaneCount(buffer) >= 2,
              let yPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let cPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return false }
        appleSRFill420(floats: floats, width: width, height: height,
                       yPlane: yPlane, yStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0),
                       cPlane: cPlane, cStride: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1),
                       fullRange: format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        return true
    default:
        return false
    }
}

// ============================================================
// MARK: - 六、逐帧超分流式处理器（会话 / 源目标缓冲 / 光流 / 跨格式转写）
// ============================================================

/// writeMp4 写盘目标（本通道预检结论：像素格式 + 是否 ProRes 分支）
struct AppleSRWriterFormat {
    let pixelFormat: OSType
    let proRes: Bool
    var description: String { proRes ? "ProRes 422 10bit v210 \(appleSRFourCC(pixelFormat))" : "h264 8bit ARGB \(appleSRFourCC(pixelFormat))" }
}

/// 逐帧超分流式处理器：持有 VTFrameProcessor 会话与源/目标缓冲构造信息，
/// 对外只暴露 fill(index:into:) 供 writeMp4 的 overrideFrames 回调同步调用。
private final class AppleSRStreamer {
    let sourceWidth: Int
    let sourceHeight: Int
    let outputWidth: Int
    let outputHeight: Int
    let scaleFactor: Int

    private let pixels: MLXArray
    private let frameCount: Int
    private let fps: Double
    private let config: VTSuperResolutionScalerConfiguration
    private let sourceFormat: OSType
    private let isCancelled: () -> Bool
    private let sourceAttributes: [String: Any]
    private let destinationAttributes: [String: Any]
    private let destinationFormat: OSType
    private let processor = VTFrameProcessor()
    private var previousSourceFrame: VTFrameProcessorFrame?
    private var previousOutputFrame: VTFrameProcessorFrame?
    private var primedFirstOutput: CVPixelBuffer?
    private var transferSession: VTPixelTransferSession?
    private var flowProcessor: VTFrameProcessor?
    private var flowAttributes: [String: Any]?
    private var flowPixelFormat: OSType?
    private var failed = false
    // 缓冲池：源/目标 CVPixelBuffer 按 frameIndex % 3 轮转复用，消除逐帧新建（目标缓冲 3072×1792×8B ≈ 44MB/帧）。
    // 深度 3 的依据：第 i 帧占用 slot i%3 时，同时存活的引用是 slot (i-1)%3（previousFrame / previousOutputFrame）
    // 与紧随的 slot (i+1)%3（下一帧输出 / 预取的源），三者互不相同；更早的帧已处理完毕，可安全覆写。
    private let bufferPoolDepth = 3
    private var sourcePool: [CVPixelBuffer] = []
    private var destinationPool: [CVPixelBuffer] = []
    /// 已预取（填充完成）的下一帧源缓冲：在 NPU 处理当前帧期间完成 MLX→CPU 像素搬运，与推理重叠。
    private var prefetchedSource: (index: Int, buffer: CVPixelBuffer)?
    // 逐帧耗时统计（仅用于收尾输出一行日志，便于确认优化是否生效；不参与控制流）
    private var statFrames = 0
    private var statFillCost = 0.0
    private var statProcessCost = 0.0
    private var statTransferCost = 0.0
    private var statPrefetchHits = 0

    init?(pixels: MLXArray, sourceWidth: Int, sourceHeight: Int, frameCount: Int, fps: Double,
          config: VTSuperResolutionScalerConfiguration, sourceFormat: OSType,
          usePrecomputedFlow: Bool, isCancelled: @escaping () -> Bool) {
        self.pixels = pixels
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.frameCount = frameCount
        self.fps = max(fps, 1)
        self.config = config
        self.sourceFormat = sourceFormat
        self.scaleFactor = config.scaleFactor
        self.outputWidth = sourceWidth * config.scaleFactor
        self.outputHeight = sourceHeight * config.scaleFactor
        self.isCancelled = isCancelled
        self.sourceAttributes = config.sourcePixelBufferAttributes
        self.destinationAttributes = config.destinationPixelBufferAttributes
        // 目标像素格式取运行期目标属性声明；未声明时退回 32BGRA（writeMp4 亦支持的通用格式，预检仍会实测）
        self.destinationFormat = appleSRPixelFormat(fromAttributes: config.destinationPixelBufferAttributes) ?? kCVPixelFormatType_32BGRA
        do {
            try processor.startSession(configuration: config)
        } catch {
            pipelineLog("⚠️ [AppleSR] VTFrameProcessor 会话启动失败：\(error.localizedDescription)")
            return nil
        }
        // 预计算光流（可选）：配置同样由运行期 VTOpticalFlowConfiguration.init? 决定
        if usePrecomputedFlow {
            if let flowConfig = VTOpticalFlowConfiguration(frameWidth: sourceWidth, frameHeight: sourceHeight,
                                                           qualityPrioritization: .normal, revision: .revision1),
               let flowFormat = appleSRPixelFormat(fromAttributes: flowConfig.destinationPixelBufferAttributes) {
                let proc = VTFrameProcessor()
                do {
                    try proc.startSession(configuration: flowConfig)
                    self.flowProcessor = proc
                    self.flowAttributes = flowConfig.destinationPixelBufferAttributes
                    self.flowPixelFormat = flowFormat
                    pipelineLog("🧮 [AppleSR] 预计算光流已启用（源 \(sourceWidth)×\(sourceHeight)，光流缓冲 \(appleSRFourCC(flowFormat))）")
                } catch {
                    pipelineLog("⚠️ [AppleSR] 预计算光流会话启动失败（\(error.localizedDescription)），本次不使用光流")
                }
            } else {
                pipelineLog("⚠️ [AppleSR] 预计算光流不可用（VTOpticalFlowConfiguration 构造失败或光流缓冲格式未声明），本次不使用光流")
            }
        }
    }

    func endSession() {
        flowProcessor?.endSession()
        processor.endSession()
        if let transferSession {
            VTPixelTransferSessionInvalidate(transferSession)
            self.transferSession = nil
        }
    }

    /// 逐帧处理期间是否出现过失败。writeMp4 对 fill 返回 false 的帧只跳过、不抛错，
    /// 故调用方必须在写盘后检查本标志，避免"跳帧截断"被当成成功产物（见 appleSRRun）。
    var didFail: Bool { failed }

    // MARK: 源/目标缓冲（池化复用 + 源预取）

    /// 按 slot 取源缓冲：池未建满时补齐，建不出来（内存不足等）则返回 nil，由调用方走原有失败路径。
    private func sourceBuffer(slot: Int) -> CVPixelBuffer? {
        while sourcePool.count < bufferPoolDepth {
            guard let buffer = appleSRNewPixelBuffer(width: sourceWidth, height: sourceHeight,
                                                     pixelFormat: sourceFormat,
                                                     attributes: sourceAttributes, requireIOSurface: true) else { break }
            sourcePool.append(buffer)
        }
        return slot < sourcePool.count ? sourcePool[slot] : nil
    }

    /// 按 slot 取目标缓冲（同源缓冲策略）。
    private func destinationBuffer(slot: Int) -> CVPixelBuffer? {
        while destinationPool.count < bufferPoolDepth {
            guard let buffer = appleSRNewPixelBuffer(width: outputWidth, height: outputHeight,
                                                     pixelFormat: destinationFormat,
                                                     attributes: destinationAttributes, requireIOSurface: true) else { break }
            destinationPool.append(buffer)
        }
        return slot < destinationPool.count ? destinationPool[slot] : nil
    }

    /// 把第 frameIndex 帧像素填进指定缓冲（MLX 切片求值 + 逐像素写入）。
    private func fillSourceBuffer(_ buffer: CVPixelBuffer, frameIndex: Int) -> Bool {
        let t0 = Date()
        CVPixelBufferLockBaseAddress(buffer, [])
        let ok = appleSRFillSourceBuffer(buffer, pixels: pixels, frameIndex: frameIndex,
                                         width: sourceWidth, height: sourceHeight, format: sourceFormat)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        statFillCost += Date().timeIntervalSince(t0)
        return ok
    }

    /// 预取下一帧源缓冲：在 NPU 处理当前帧的等待窗口内完成该帧的 MLX→CPU 搬运与填充。
    /// 不改提交顺序、不改 previous 链，仅把 CPU 侧准备移入等待窗口；重复调用为幂等。
    @discardableResult
    private func prefetchSource(frameIndex: Int) -> Bool {
        guard frameIndex >= 0, frameIndex < frameCount else { return false }
        if let existing = prefetchedSource, existing.index == frameIndex { return true }
        guard let buffer = sourceBuffer(slot: frameIndex % bufferPoolDepth),
              fillSourceBuffer(buffer, frameIndex: frameIndex) else { return false }
        prefetchedSource = (frameIndex, buffer)
        return true
    }

    // MARK: 逐帧处理

    private func processFrame(index: Int) -> CVPixelBuffer? {
        let slot = index % bufferPoolDepth
        // 源缓冲：优先取上一帧等待窗口内预取好的结果，否则现场填充
        let src: CVPixelBuffer
        if let prefetched = prefetchedSource, prefetched.index == index {
            src = prefetched.buffer
            prefetchedSource = nil
            statPrefetchHits += 1
        } else {
            guard let buffer = sourceBuffer(slot: slot), fillSourceBuffer(buffer, frameIndex: index) else {
                pipelineLog("⚠️ [AppleSR] 第 \(index) 帧源像素缓冲构造/填充失败（\(appleSRFourCC(sourceFormat))）")
                return nil
            }
            src = buffer
        }
        let pts = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(max(fps.rounded(), 1)))
        guard let srcFrame = VTFrameProcessorFrame(buffer: src, presentationTimeStamp: pts),
              let dst = destinationBuffer(slot: slot),
              let dstFrame = VTFrameProcessorFrame(buffer: dst, presentationTimeStamp: pts) else {
            pipelineLog("⚠️ [AppleSR] 第 \(index) 帧 VTFrameProcessorFrame 构造失败")
            return nil
        }
        // 预计算光流：与上一帧之间的前/后向光流（仅开启时计算）
        var opticalFlow: VTFrameProcessorOpticalFlow? = nil
        if let flowProcessor, let previousSourceFrame {
            opticalFlow = computeOpticalFlow(processor: flowProcessor, previous: previousSourceFrame,
                                             current: srcFrame, pts: pts)
        }
        guard let params = VTSuperResolutionScalerParameters(sourceFrame: srcFrame,
                                                             previousFrame: previousSourceFrame,
                                                             previousOutputFrame: previousOutputFrame,
                                                             opticalFlow: opticalFlow,
                                                             submissionMode: .sequential,
                                                             destinationFrame: dstFrame) else {
            pipelineLog("⚠️ [AppleSR] 第 \(index) 帧参数构造失败（源/目标像素格式不一致？）")
            return nil
        }
        let sem = DispatchSemaphore(value: 0)
        var error: Error? = nil
        processor.process(parameters: params) { _, err in
            error = err
            sem.signal()
        }
        // 提交完成后、等待结果前：预取下一帧源缓冲（MLX→CPU 搬运 + 填充），与 NPU 推理重叠。
        // 仅把 CPU 侧准备工作塞进等待窗口，不改变提交顺序、不改变 previous 链。
        prefetchSource(frameIndex: index + 1)
        let tWait = Date()
        sem.wait()
        statProcessCost += Date().timeIntervalSince(tWait)
        statFrames += 1
        if let error {
            pipelineLog("⚠️ [AppleSR] 第 \(index) 帧超分失败：\(error.localizedDescription)")
            return nil
        }
        previousSourceFrame = srcFrame
        previousOutputFrame = dstFrame
        return dst
    }

    private func computeOpticalFlow(processor: VTFrameProcessor, previous: VTFrameProcessorFrame,
                                    current: VTFrameProcessorFrame, pts: CMTime) -> VTFrameProcessorOpticalFlow? {
        guard let flowAttributes, let flowPixelFormat,
              let fwd = appleSRNewPixelBuffer(width: sourceWidth, height: sourceHeight, pixelFormat: flowPixelFormat,
                                              attributes: flowAttributes, requireIOSurface: true),
              let bwd = appleSRNewPixelBuffer(width: sourceWidth, height: sourceHeight, pixelFormat: flowPixelFormat,
                                              attributes: flowAttributes, requireIOSurface: true),
              let flow = VTFrameProcessorOpticalFlow(forwardFlow: fwd, backwardFlow: bwd),
              let params = VTOpticalFlowParameters(sourceFrame: previous, nextFrame: current,
                                                   submissionMode: .sequential, destinationOpticalFlow: flow) else {
            return nil
        }
        let sem = DispatchSemaphore(value: 0)
        var error: Error? = nil
        processor.process(parameters: params) { _, err in
            error = err
            sem.signal()
        }
        sem.wait()
        if let error {
            pipelineLog("⚠️ [AppleSR] 光流计算失败：\(error.localizedDescription)（本帧退回不使用光流）")
            return nil
        }
        return flow
    }

    // MARK: 写盘格式预检（不假设 SR 输出与 writeMp4 支持格式一致，实测转换）

    /// 拿第 0 帧超分结果向候写盘格式各试转一次，只有真正转换成功才决定 writeMp4 分支与像素格式。
    /// 候选即 writeMp4 的两条分支（ProRes 422 10bit v210 / h264 8bit 32ARGB），属写盘能力而非 SR 能力。
    func preflightWriterFormat() -> AppleSRWriterFormat? {
        guard let sample = primeFirstFrame() else { return nil }
        let candidates = [
            AppleSRWriterFormat(pixelFormat: kCVPixelFormatType_422YpCbCr10, proRes: true),
            AppleSRWriterFormat(pixelFormat: kCVPixelFormatType_32ARGB, proRes: false),
        ]
        for candidate in candidates {
            guard let probe = appleSRNewPixelBuffer(width: outputWidth, height: outputHeight,
                                                    pixelFormat: candidate.pixelFormat,
                                                    attributes: destinationAttributes, requireIOSurface: false) else { continue }
            var session: VTPixelTransferSession?
            guard VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault,
                                               pixelTransferSessionOut: &session) == kCVReturnSuccess,
                  let transfer = session else {
                pipelineLog("⚠️ [AppleSR] 写盘格式预检：VTPixelTransferSession 创建失败")
                return nil
            }
            let status = VTPixelTransferSessionTransferImage(transfer, from: sample, to: probe)
            if status == kCVReturnSuccess {
                transferSession = transfer
                let format = AppleSRWriterFormat(pixelFormat: candidate.pixelFormat, proRes: candidate.proRes)
                pipelineLog("🔎 [AppleSR] 写盘格式预检通过：\(format.description)；超分输出 \(outputWidth)×\(outputHeight)（×\(scaleFactor)）")
                return format
            }
            VTPixelTransferSessionInvalidate(transfer)
            pipelineLog("🔎 [AppleSR] 写盘格式预检未通过：\(candidate.description)（VTPixelTransferSessionTransferImage OSStatus \(status)），尝试下一候选")
        }
        return nil
    }

    /// 预热第 0 帧（超分输出同时作为写盘格式预检样本，并建立上一帧引用链）
    private func primeFirstFrame() -> CVPixelBuffer? {
        guard frameCount > 0 else { return nil }
        let out = processFrame(index: 0)
        primedFirstOutput = out
        return out
    }

    // MARK: writeMp4 逐帧回调

    /// 把第 index 帧超分结果写进 writeMp4 池缓冲；跨格式转换由 VTPixelTransferSession 完成。
    func fill(index: Int, into pixelBuffer: CVPixelBuffer) -> Bool {
        guard !failed else { return false }
        if isCancelled() {
            failed = true
            pipelineLog("⏹ [AppleSR] 已取消，停止超分逐帧处理")
            return false
        }
        let srBuffer: CVPixelBuffer?
        if index == 0, let primed = primedFirstOutput {
            srBuffer = primed
        } else {
            srBuffer = processFrame(index: index)
        }
        guard let sr = srBuffer else {
            failed = true
            return false
        }
        guard let transferSession else {
            failed = true
            return false
        }
        let tTransfer = Date()
        let status = VTPixelTransferSessionTransferImage(transferSession, from: sr, to: pixelBuffer)
        statTransferCost += Date().timeIntervalSince(tTransfer)
        guard status == kCVReturnSuccess else {
            pipelineLog("⚠️ [AppleSR] 第 \(index) 帧超分结果转写写盘缓冲失败（OSStatus \(status)）")
            failed = true
            return false
        }
        if index % 24 == 0 || index == frameCount - 1 {
            pipelineLog("🎞️ [AppleSR] 超分进度 \(index + 1)/\(frameCount)")
        }
        if index == frameCount - 1 {
            let n = Double(max(statFrames, 1))
            pipelineLog(String(format: "📊 [AppleSR] 逐帧耗时均值：源填充 %.1f ms（预取命中 %d/%d 帧）、超分等待 %.1f ms、转写 %.1f ms；缓冲池 %d 源 / %d 目标（复用）",
                               statFillCost * 1000 / n, statPrefetchHits, statFrames,
                               statProcessCost * 1000 / n, statTransferCost * 1000 / n,
                               sourcePool.count, destinationPool.count))
        }
        return true
    }
}

// ============================================================
// MARK: - 七、模型资产（状态查询 / 下载进度 / 等待就绪）
// ============================================================

/// 轮询 configurationModelPercentageAvailable 的进度上报器（下载期间后台刷新，不阻塞任何线程）
private final class AppleSRProgressPoller {
    private let timer: DispatchSourceTimer

    init(config: VTSuperResolutionScalerConfiguration, interval: TimeInterval = 0.5,
         callback: @escaping (Double) -> Void) {
        let queue = DispatchQueue(label: "appleSR.modelProgress", qos: .utility)
        timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler {
            callback(Double(config.configurationModelPercentageAvailable))
        }
        timer.resume()
    }

    func cancel() {
        timer.setEventHandler {}
        timer.cancel()
    }
}

/// 下载模型资产：直接使用系统 async 版 downloadConfigurationModel() async throws；
/// progress 由后台轮询 configurationModelPercentageAvailable 回报，下载本身在系统后台执行、不阻塞调用线程。
func appleSRDownloadConfigurationModel(config: VTSuperResolutionScalerConfiguration,
                                       progress: @escaping (Double) -> Void) async throws {
    let poller = AppleSRProgressPoller(config: config, callback: progress)
    defer { poller.cancel() }
    try await config.downloadConfigurationModel()
}

/// 等待模型资产就绪：.downloading 期间轮询（进度回报），.downloadRequired 回落视为失败，
/// .ready 直接通过；等待超时/被取消返回 false（调用方回退原 CQ/IC 二采）。
func appleSRWaitForModelReady(config: VTSuperResolutionScalerConfiguration,
                              isCancelled: () -> Bool,
                              timeout: TimeInterval = 1800) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    var lastPct = -1
    while Date() < deadline {
        if isCancelled() {
            pipelineLog("⏹ [AppleSR] 等待模型资产期间被取消")
            return false
        }
        switch config.configurationModelStatus {
        case .ready:
            AppleSRRuntimeState.reportModel(ready: true, progress: 1, text: "模型已就绪")
            return true
        case .downloadRequired:
            pipelineLog("⚠️ [AppleSR] 模型资产回落 downloadRequired（下载失败或未触发）")
            AppleSRRuntimeState.reportModel(ready: false, progress: Double(config.configurationModelPercentageAvailable), text: "模型未就绪")
            return false
        case .downloading:
            let pct = Int(Double(config.configurationModelPercentageAvailable) * 100)
            if pct != lastPct {
                lastPct = pct
                pipelineLog("⏳ [AppleSR] 模型资产下载中 \(pct)%")
                AppleSRRuntimeState.reportModel(ready: false, progress: Double(pct) / 100.0, text: "模型下载中 \(pct)%")
            }
        @unknown default:
            pipelineLog("⚠️ [AppleSR] 模型状态未知，按未就绪处理并回退")
            return false
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
    }
    pipelineLog("⚠️ [AppleSR] 等待模型资产超时（\(Int(timeout))s），回退原 CQ/IC 二采")
    return false
}

/// 生成流程内的模型资产准备：ready 直接通过；downloadRequired 触发系统下载（带进度）；
/// downloading 等待就绪。任何失败返回 false（调用方回退原 CQ/IC 二采，不中断生成）。
func appleSREnsureModelAssets(config: VTSuperResolutionScalerConfiguration,
                              isCancelled: () -> Bool) async -> Bool {
    let status = config.configurationModelStatus
    let pct = config.configurationModelPercentageAvailable
    switch status {
    case .ready:
        AppleSRRuntimeState.reportModel(ready: true, progress: 1, text: "模型已就绪")
        return true
    case .downloadRequired:
        pipelineLog("⬇️ [AppleSR] 模型资产未就绪（downloadRequired，当前 \(Int(pct * 100))%），触发系统下载"
            + "（下载在系统后台执行；本函数 await 等待，不阻塞 UI/渲染线程）")
        AppleSRRuntimeState.reportModel(ready: false, progress: Double(pct), text: "模型下载中 \(Int(pct * 100))%")
        do {
            try await appleSRDownloadConfigurationModel(config: config) { p in
                AppleSRRuntimeState.reportModel(ready: false, progress: p, text: "模型下载中 \(Int(p * 100))%")
            }
        } catch {
            pipelineLog("⚠️ [AppleSR] 模型资产下载失败：\(error.localizedDescription)（状态回落 downloadRequired），回退原 CQ/IC 二采")
            AppleSRRuntimeState.reportModel(ready: false, progress: Double(config.configurationModelPercentageAvailable),
                                            text: "下载失败：\(error.localizedDescription)")
            return false
        }
        // 下载回调成功后再确认一次状态（configure/文件就位可能存在极短延迟）
        return await appleSRWaitForModelReady(config: config, isCancelled: isCancelled)
    case .downloading:
        pipelineLog("⏳ [AppleSR] 模型资产下载中（\(Int(pct * 100))%），等待完成…")
        return await appleSRWaitForModelReady(config: config, isCancelled: isCancelled)
    @unknown default:
        pipelineLog("⚠️ [AppleSR] 模型状态未知，回退原 CQ/IC 二采")
        return false
    }
}

// ============================================================
// MARK: - 八、主入口：H3 内存像素 → Apple 超分 → 编码落盘 + 复用源音轨
// ============================================================

/// 第三套二采主入口（与 LTX 二采入口 ltxEnhanceExternalVideoWithStage2Pixels 同签名口径）：
/// H3 stage1 内存像素（[1,3,T,H,W] f32 [-1,1]）→ Apple VideoToolbox 超分 → 编码落盘（ProRes 422 10bit 优先）
/// → 复用 muxSourceAudioOnto 沿用 stage1 源音轨。
///
/// - 任何失败/不可用都返回 nil（只打日志），由调用方回退原 CQ/IC 二采，绝不中断生成。
/// - 本函数不做任何分辨率上限/倍率的硬编码假设：全部以运行期 API 探测结果为准。
/// - 返回最终产物绝对路径；nil 表示本通道未产出。
func appleSuperResolutionSecondPass(
    pixels: MLXArray,
    pixelWidth: Int,
    pixelHeight: Int,
    pixelFps: Double,
    sourceAudioVideoPath: String,
    isCancelled: @escaping () -> Bool = { false }
) async -> String? {
    let t0 = Date()
    guard appleSRSecondPassEnabled() else { return nil }

    // 源张量语义：与 LTX 二采入口一致 [1,3,T,H,W] f32 [-1,1]
    let shape = pixels.shape
    guard shape.count == 5, shape[1] == 3, shape[2] >= 1 else {
        pipelineLog("⚠️ [AppleSR] 像素张量形状非法 \(shape)（期望 [1,3,T,H,W]），回退原 CQ/IC 二采")
        return nil
    }
    let frameCount = shape[2]
    let srcH = shape[3]
    let srcW = shape[4]
    if srcW != pixelWidth || srcH != pixelHeight {
        pipelineLog("ℹ️ [AppleSR] 张量尺寸 \(srcW)×\(srcH) 与桥记录 \(pixelWidth)×\(pixelHeight) 不一致，以张量为准")
    }

    // ① 运行期探测：isSupported
    guard VTSuperResolutionScalerConfiguration.isSupported else {
        AppleSRRuntimeState.reportUnsupported()
        pipelineLog("⚠️ [AppleSR] 本机不支持 VTSuperResolutionScaler（isSupported=false），回退原 CQ/IC 二采")
        return nil
    }
    let supportedFactors = VTSuperResolutionScalerConfiguration.supportedScaleFactors
    guard !supportedFactors.isEmpty else {
        pipelineLog("⚠️ [AppleSR] 运行期倍率候选为空（supportedScaleFactors 为空），回退原 CQ/IC 二采")
        return nil
    }
    let qualityRaw = appleSRRequestedQualityRawValue()
    let useFlow = appleSRRequestedUsePrecomputedFlow()
    let requestedScale = appleSRRequestedScaleFactorRaw()
    pipelineLog("=== [AppleSR] 第三套二采（Apple VideoToolbox 超分）启动：源 \(srcW)×\(srcH) 帧数 \(frameCount) "
        + "倍率候选[\(appleSRScaleOptionsSummary())] 请求倍率×\(requestedScale) 质量原始值 \(qualityRaw) 预计算光流 \(useFlow ? "开" : "关") ===")

    // ② 以真实源尺寸 + 运行期倍率候选构造配置（倍率阶梯回退；尺寸越界/不支持即放弃）
    var chosen: VTSuperResolutionScalerConfiguration? = nil
    var chosenScale = 0
    var triedScales: [String] = []
    for scale in appleSRScaleLadder(preferred: requestedScale, supported: supportedFactors) where scale != 1 {
        triedScales.append("×\(scale)")
        if let cfg = appleSRMakeConfiguration(frameWidth: srcW, frameHeight: srcH, scaleFactor: scale,
                                              usePrecomputedFlow: useFlow, qualityRawValue: qualityRaw) {
            chosen = cfg
            chosenScale = scale
            break
        }
    }
    guard let config = chosen else {
        pipelineLog("⚠️ [AppleSR] 源尺寸 \(srcW)×\(srcH) 在运行期倍率候选[\(triedScales.joined(separator: "/"))]下均无法构造配置"
            + "（越界/不支持），回退原 CQ/IC 二采")
        return nil
    }
    pipelineLog("🧭 [AppleSR] 运行期探测：\(appleSRCapabilitySummary(config))")
    pipelineLog("🧭 [AppleSR] 本次采用倍率 ×\(chosenScale)，输出 \(srcW * chosenScale)×\(srcH * chosenScale)")

    // ③ 源像素格式：运行期 supportedPixelFormats ∩ 本通道可实现填充集合
    guard let sourceFormat = appleSRPickSourceFormat(config: config) else { return nil }

    // ④ 模型资产：必要即下载（带进度，不阻塞 UI/渲染线程）
    AppleSRRuntimeState.reportModel(ready: config.configurationModelStatus == .ready,
                                    progress: Double(config.configurationModelPercentageAvailable),
                                    text: config.configurationModelStatus == .ready ? "模型已就绪" : "检查模型资产…")
    guard await appleSREnsureModelAssets(config: config, isCancelled: isCancelled) else {
        pipelineLog("⚠️ [AppleSR] 模型资产未就绪/下载失败，回退原 CQ/IC 二采")
        return nil
    }
    if isCancelled() {
        pipelineLog("⏹ [AppleSR] 已取消，放弃本通道")
        return nil
    }

    // ⑤ 逐帧超分 + 编码落盘 + 复用源音轨
    guard let finalPath = appleSRRun(pixels: pixels, sourceWidth: srcW, sourceHeight: srcH,
                                     frameCount: frameCount, fps: pixelFps,
                                     sourceAudioVideoPath: sourceAudioVideoPath,
                                     config: config, sourceFormat: sourceFormat,
                                     usePrecomputedFlow: useFlow,
                                     isCancelled: isCancelled) else {
        pipelineLog("⚠️ [AppleSR] 本通道未产出（失败/取消），回退原 CQ/IC 二采")
        return nil
    }
    pipelineLog("✅ [AppleSR] 第三套二采完成（耗时 \(String(format: "%.1f", Date().timeIntervalSince(t0)))s）：\(finalPath)")
    return finalPath
}

/// 超分 + 落盘 + 混音同步核心（在生成后台任务中执行，不触碰主线程）：
/// 先预热第 0 帧并预检写盘格式，再复用 writeMp4 落盘（overrideFrames 直写通道），最后沿用源音轨。
private func appleSRRun(pixels: MLXArray, sourceWidth: Int, sourceHeight: Int, frameCount: Int,
                        fps: Double, sourceAudioVideoPath: String,
                        config: VTSuperResolutionScalerConfiguration, sourceFormat: OSType,
                        usePrecomputedFlow: Bool,
                        isCancelled: @escaping () -> Bool) -> String? {
    guard let streamer = AppleSRStreamer(pixels: pixels, sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                                         frameCount: frameCount, fps: fps, config: config,
                                         sourceFormat: sourceFormat, usePrecomputedFlow: usePrecomputedFlow,
                                         isCancelled: isCancelled) else {
        pipelineLog("⚠️ [AppleSR] 超分会话初始化失败，回退原 CQ/IC 二采")
        return nil
    }
    defer { streamer.endSession() }

    guard let writerFormat = streamer.preflightWriterFormat() else {
        pipelineLog("⚠️ [AppleSR] 写盘格式预检未通过（超分输出与 writeMp4 支持格式无法转换），回退原 CQ/IC 二采")
        return nil
    }

    let outDir = outputVideoDirURL.path
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let ext = writerFormat.proRes ? "mov" : "mp4"
    let baseName = nextAssetName(prefix: "uhdsr")
    let silentPath = (outDir as NSString).appendingPathComponent("\(baseName)_silent.\(ext)")
    let finalPath = (outDir as NSString).appendingPathComponent("\(baseName).\(ext)")
    let fpsInt = max(1, Int(fps.rounded()))

    do {
        try writeMp4(frameCount: frameCount, width: streamer.outputWidth, height: streamer.outputHeight,
                     fps: fpsInt, to: silentPath,
                     proRes: writerFormat.proRes,
                     h264MaxQuality: !writerFormat.proRes,
                     // 本通道专用：池缓冲格式已按 writerFormat 选定，闭包内直接接收超分输出
                     //（跨格式转换在 AppleSRStreamer 内用 VTPixelTransferSession 完成）
                     overrideFrames: { idx, pb in streamer.fill(index: idx, into: pb) })
    } catch {
        pipelineLog("⚠️ [AppleSR] 写盘失败：\(error.localizedDescription)，回退原 CQ/IC 二采")
        try? FileManager.default.removeItem(atPath: silentPath)
        return nil
    }
    if isCancelled() {
        pipelineLog("⏹ [AppleSR] 写盘期间被取消，丢弃产物")
        try? FileManager.default.removeItem(atPath: silentPath)
        return nil
    }
    // writeMp4 对填充失败的帧只跳过、不抛错；必须显式检查，避免"跳帧截断"被当成成功产物。
    if streamer.didFail {
        pipelineLog("⚠️ [AppleSR] 逐帧超分/跨格式转写过程中出现失败（writeMp4 已跳过失败帧），丢弃本次产物并回退原 CQ/IC 二采")
        try? FileManager.default.removeItem(atPath: silentPath)
        return nil
    }

    // 复用像素桥的混音函数：沿用 stage1 源音轨（源无音轨时保留无声产物）
    archiveIfExists(finalPath)
    if muxSourceAudioOnto(silentVideo: silentPath, sourceVideo: sourceAudioVideoPath, to: finalPath) {
        try? FileManager.default.removeItem(atPath: silentPath)
        pipelineLog("🎬 [AppleSR] 超分产物落盘（含源音轨）：\(finalPath)")
        return finalPath
    }
    try? FileManager.default.moveItem(atPath: silentPath, toPath: finalPath)
    pipelineLog("ℹ️ [AppleSR] 混音未成功（源视频无音轨或混流失败），产物为无声视频：\(finalPath)")
    return finalPath
}
