//
//  偏好设置.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/14.
//
// ============================================================
//  文件作用：偏好设置面板（图2样式：左右分栏）+ 全局设置存储 AppSettings。
//  左侧导航：通用设置 / API / 授权设置 / 关于；
//  右侧内容区：通用设置页含项目文件地址，
//  以及「画布」分组（节点默认颜色、滚轮速度倍率，UserDefaults 持久化）。
//  互动文件：被 主ui.swift 的 Settings 场景引用（PreferencesView）；
//  AppSettings 被 画布状态 公共函数.swift、读取&保存.swift 读取。
// ============================================================

import SwiftUI
import Combine

// ============================================================

// MARK: - H3 视频后处理方式（下拉菜单单选）

/// H3 一采之后的后处理通道选择。用户在下拉菜单里单选，
/// 选择时自动联动「阶段2 开关 / Apple 超分 / SelfLift 渐进采样」底层开关，
/// 避免手动组合出错。（2026-09-18：IC/CQ 二采路径放弃，菜单不再暴露）
enum H3PostProcessMode: String, CaseIterable, Identifiable {
    /// 无：啥都不走，H3 一采直出（SelfLift 渐进采样一并关闭，等同接入 SelfLift 前的单遍直出）
    case none = "none"
    /// SelfLift：H3 自己 lift —— 只跑 H3 SelfLift 渐进采样（内部低清→全清）直出，不接 LTX 二采
    case selflift = "selflift"
    /// Apple：Apple 超分二采 —— H3 直出目标尺寸后走 Apple VTSuperResolutionScaler 超分
    /// （与阶段2 解耦的独立二采；接口暂未重点调优，预留下拉位）
    case apple = "apple"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .none: return "无（H3 直出 非常慢）"
        case .selflift: return "SelfLift（半清+高清 非常快）"
        case .apple: return "Apple 超分（预留）"
        }
    }

    var footnote: String {
        switch self {
        case .none:
            return "啥都不走：H3 一采直出目标尺寸"
        case .selflift:
            return "恒定最后一步为半尺寸，前面都是半尺寸。（前面跑结构，最后一步补细节。模糊是分辨率不足或运动模糊，脸结构崩溃扭曲是步数不足，5步不行改6步。）"
        case .apple:
            return "H3 直出目标尺寸后走 Apple 超分（预留位，暂未重点调优）"
        }
    }
}

// MARK: - 全局设置存储（UserDefaults 持久化）

final class AppSettings: ObservableObject {
    static let shared = AppSettings()
    private let defaults = UserDefaults.standard

    // 节点默认颜色（存颜色名，Color 不可 Codable）
    @Published var defaultNodeColorName: String {
        didSet { defaults.set(defaultNodeColorName, forKey: "defaultNodeColorName") }
    }
    // 触控板滚轮加速倍率
    @Published var wheelAccelerationPrecise: Double {
        didSet { defaults.set(wheelAccelerationPrecise, forKey: "wheelAccelerationPrecise") }
    }
    // 鼠标滚轮加速倍率
    @Published var wheelAccelerationWheel: Double {
        didSet { defaults.set(wheelAccelerationWheel, forKey: "wheelAccelerationWheel") }
    }
    // 项目文件地址（默认 ~/Documents/无限画布，可自定义）
    @Published var canvasRootPath: String {
        didSet { defaults.set(canvasRootPath, forKey: "canvasRootPath") }
    }
    // ★ H3 一采第一阶段总步数 N（面板滑杆 4–12，默认 6）：
    // 该值就是「H3 一采总步数 N」——队列把它作为 steps 传给 H3FL2VAPipeline.generateVideo，
    // 管线内部 N = sigmas.count - 1（见 H3Pipeline stage1SegmentCount）。
    // SelfLift 第三分支的 transition_step（ts）不写死，由官方 75% 规则随 N 自动推导：
    //     ts = clamp(floor(0.75 · N), 1, N-1)（SelfLiftScheduleSplit.transitionStep(forTotalSteps:)）
    // 各档：4→3（测试档）、5→3、6→4、7→5、8→6、9→6、10→7、11→8、12→9。
    // 默认 6 与 turbo LoRA（6 步版）标定一致；4 步档仅用于 SelfLift 联动测试，画质不作保证。
    @Published var h3Stage1Steps: Int {
        didSet { defaults.set(h3Stage1Steps, forKey: "h3Stage1Steps") }
    }
    /// H3 一采第一阶段总步数滑杆范围（面板 4–12；官方 SelfLift 硬约束 1 ≤ ts ≤ N-1 由 ts 推导侧 clamp 保证）
    static let h3Stage1StepsRange = 4...12

    // ★ SelfLift 第三分支（H3 一采渐进式采样）总开关：
    // true=一采 stage1 走 SelfLift 渐进采样：低分（×0.5）跑 ts 次 NFE → 零 NFE 过渡块
    //      （官方 H3 路径 nearest 直接 latent 提升 + VAE 像素锚点一致修正，rho=0.6）
    //      → 升回全分辨率 → 高分跑 N-ts 次 NFE；低分 NFE + 高分 NFE = N，总 NFE 不变。
    // false=按原单条 stage1 循环单遍直出（行为与接入前逐字一致）。
    // 注：像素锚点需要一采期间常驻 video_vae，峰值内存高于关闭时；对照实验可关本项。
    @Published var h3SelfLiftEnabled: Bool {
        didSet { defaults.set(h3SelfLiftEnabled, forKey: "h3SelfLiftEnabled") }
    }

    // ★ SelfLift 第三分支·二阶段解耦采样总开关（自研解耦调度）：
    // true=一采 stage1 的 SelfLift 改走解耦调度：低清（×0.5）独立跑
    //      L = round(shift/σ_k − shift + 1) 次 NFE（shift=12、σ_k=0.9 → L=2）→ 放大重加噪
    //      （σ_next 缺省=σ_k≈0.9231，浅起点残留 7.7%）→ 高清独立跑 M=3 次 NFE，总 NFE = L+M = 5；
    //      权重与提升/融合原语复用官方 SelfLift，但低清/高清两条 σ 曲线解耦为自研调度。
    //      （2026-09-17 重影修复 v2：σ_k 0.7→0.9 低清浅跑 L=2 避免固化运动双影，高清 3 步等距充分清洗，
    //       总 NFE=5 < 官方 N=6，不浪费算力）
    // false=按官方 75% 耦合调度（ts = clamp(floor(0.75×N),1,N-1)，低分+高分总 NFE=N）。
    // 注：仅 h3SelfLiftEnabled=true 时有意义；开关经生成队列 setenv
    //      NA_H3_SELFLIFT_DECOUPLE / _LOW_K / _HIGH_STEPS 三变量生效。
    @Published var h3SelfLiftDecouple: Bool {
        didSet { defaults.set(h3SelfLiftDecouple, forKey: "h3SelfLiftDecouple") }
    }

    // LTX-2.5 第二阶段总开关：true=启用第二阶段（旧 refine 精修）；false=仅跑第一阶段
    @Published var videoUseStage2: Bool {
        didSet { defaults.set(videoUseStage2, forKey: "videoUseStage2") }
    }
    // ★ 第二阶段·CQ 清晰度增强（新分支）：true=H3 二采改走官方 LTX-2.5 CQ Video Enhancer LoRA
    // （ltx2.5-CQ-enhancer-lora-for-videos-rank128，strength=1.0，官方 σ0=1.0 / 9 段 / euler_ancestral），
    // 只对 H3 半清视频做清晰度提升（生成式增强，不换脸、不重绘构图）；
    // false=保持原 LTX 像素桥 IC 二采路径（行为完全不变）。默认 true（已改用 CQ 方案）。
    // 说明：本项仅在「阶段2 开关」开启时生效；两通道互斥、CQ 优先。
    // 环境变量 LTX_CQ_ENHANCER=1/0 优先级高于本设置项（便于实验对比）。
    @Published var videoUseCQEnhancer: Bool {
        didSet { defaults.set(videoUseCQEnhancer, forKey: "videoUseCQEnhancer") }
    }
    // ★ 后处理方式（下拉菜单单选，2026-09-18）：H3 一采之后的二采/直出通道选择。
    // 本项是「阶段2 开关 / 二采改用 CQ / Apple 超分 / SelfLift 渐进采样」的上层语义选择器：
    // 选择时经 applyPostProcessMode 联动四个底层开关；底层开关被手动改动时，
    // currentPostProcessMode 会实时推导当前模式（菜单随之刷新），二者始终自洽。
    @Published var videoPostProcessMode: H3PostProcessMode {
        didSet { defaults.set(videoPostProcessMode.rawValue, forKey: "videoPostProcessMode") }
    }
    /// 由当前底层开关状态推导所处的后处理模式（优先级：Apple → SelfLift → 无；IC/CQ 二采路径已放弃）
    var currentPostProcessMode: H3PostProcessMode {
        if videoUseAppleSR { return .apple }
        if h3SelfLiftEnabled { return .selflift }
        return .none
    }
    /// 应用下拉菜单选择：联动更新底层开关，保证与所选模式完全一致
    /// （2026-09-18：IC/CQ 二采路径放弃，任何模式都不再开启 videoUseStage2）
    func applyPostProcessMode(_ mode: H3PostProcessMode) {
        videoPostProcessMode = mode
        switch mode {
        case .none:
            h3SelfLiftEnabled = false
            videoUseStage2 = false
            videoUseCQEnhancer = false
            videoUseAppleSR = false
        case .selflift:
            h3SelfLiftEnabled = true
            videoUseStage2 = false
            videoUseCQEnhancer = false
            videoUseAppleSR = false
        case .apple:
            h3SelfLiftEnabled = true
            videoUseStage2 = false
            videoUseCQEnhancer = false
            videoUseAppleSR = true
        }
    }
    // （2026-09-21：扩散视频解码器设置项已移除，videoUseDiffusionDecoder 字段删除，
    //  代码侧 ltx2.5-(ltx专属).swift 中 useDiffDecoder 已硬编码 false，固定走卷积 VAE）

    // ★ 第三套二采·Apple VideoToolbox 超分（VTSuperResolutionScaler / VTFrameProcessor）总开关：
    // true=H3 stage1 出的内存态视频**优先**走 Apple 超分通道（并存的 CQ/IC 二采保留为失败回退）；
    // false=完全不启用本通道（行为与本次接入前一致）。
    // 任何不可用/失败（运行期不支持、模型资产未就绪、源尺寸越界、源像素格式无交集等）都自动回退原 CQ/IC 二采。
    // 环境变量 LTX_APPLE_SR=1/0 优先级高于本设置项。默认 true（全局目标：H3 出的视频统一走 Apple 超分）。
    @Published var videoUseAppleSR: Bool {
        didSet { defaults.set(videoUseAppleSR, forKey: "videoUseAppleSR") }
    }
    // Apple 超分倍率：面板已取消倍率选择，固定默认 ×4（高质量款 VTSuperResolutionScalerConfiguration
    // 本机运行期唯一候选 supportedScaleFactors=[4]，无其他档可选）。
    // 本项保留仅为兼容环境变量 LTX_APPLE_SR_SCALE 覆盖与旧偏好；
    // 生效值仍必须落在运行期 supportedScaleFactors 内，否则按候选阶梯回退（不硬编码生效倍率）。
    @Published var appleSRScaleFactor: Int {
        didSet { defaults.set(appleSRScaleFactor, forKey: "appleSRScaleFactor") }
    }
    // Apple 超分质量优先级：存运行期枚举原始值（系统公开候选仅 normal=1）。
    // 面板已不出下拉（无可选项），本项保留仅供环境变量 LTX_APPLE_SR_QUALITY 覆盖与旧偏好兼容。
    @Published var appleSRQualityRawValue: Int {
        didSet { defaults.set(appleSRQualityRawValue, forKey: "appleSRQualityRawValue") }
    }
    // 是否使用预计算光流（质量优先时开启；需额外计算前后向光流，耗时/显存更高）
    @Published var appleSRUsePrecomputedFlow: Bool {
        didSet { defaults.set(appleSRUsePrecomputedFlow, forKey: "appleSRUsePrecomputedFlow") }
    }

    // ★ H3→LTX latent 直通（H3-to-LTX-Latent-Adapter）总开关：
    // true=H3 一采的 clean latent 在 VAE 解码之前被截获，经适配器直接映射为 LTX 归一化 latent，
    // 二采不再读像素、不再跑 LTX vaeEncodeVideo；默认连 H3 VAE 解码一并跳过（不写 stage1 mp4，
    // 音轨单独落 wav 由二采最终混流），一步替代「H3 decode → 像素 → LTX encode」两步像素往返。
    // false=完全按原像素桥路径（H3 decode → 像素 → LTX encode），行为与接入前逐字一致。
    // 环境变量 LTX_H3_ADAPTER=1/0 优先级高于本设置项；权重缺失/形状不符/推理失败自动回退原路径。
    @Published var videoUseH3LTXAdapter: Bool {
        didSet { defaults.set(videoUseH3LTXAdapter, forKey: "videoUseH3LTXAdapter") }
    }
    // H3→LTX 适配器权重文件路径；空串=用默认路径（见 defaultH3LTXAdapterPath）。
    @Published var h3LTXAdapterPath: String {
        didSet { defaults.set(h3LTXAdapterPath, forKey: "h3LTXAdapterPath") }
    }

    init() {
        let d = UserDefaults.standard
        defaultNodeColorName = d.string(forKey: "defaultNodeColorName") ?? "pink"
        wheelAccelerationPrecise = d.object(forKey: "wheelAccelerationPrecise") as? Double ?? 2.0
        wheelAccelerationWheel = d.object(forKey: "wheelAccelerationWheel") as? Double ?? 30.0
        let savedPath = d.string(forKey: "canvasRootPath") ?? ""
        // ★ 2026-09-19 崩溃修复：根路径存局部变量，供 init 后续阶段拼默认路径使用
        // （两阶段初始化期间禁止访问 self 属性，故不能直接引用 canvasRootPath）。
        let resolvedRootPath = savedPath.isEmpty ? Self.defaultRootPath : savedPath
        canvasRootPath = resolvedRootPath
        // ★ H3 一采第一阶段总步数 N：默认 6（turbo LoRA 标定值）；存量/越界值按 4–12 夹取，
        // 保证传给采样链路的 N 永远落在滑杆量程内（ts 推导侧另有 1 ≤ ts ≤ N-1 的 clamp）。
        let savedH3Steps = d.object(forKey: "h3Stage1Steps") as? Int ?? 6
        h3Stage1Steps = min(max(savedH3Steps, Self.h3Stage1StepsRange.lowerBound),
                            Self.h3Stage1StepsRange.upperBound)
        // ★ init 阶段禁止读取 self 属性（Swift 两阶段初始化：所有存储属性赋完值前不能读 self），
        //   迁移判断统一用局部变量（取值与赋给属性的完全一致）。
        let stage2 = d.object(forKey: "videoUseStage2") as? Bool ?? true
        videoUseStage2 = stage2
        // ★ SelfLift 第三分支默认开启（关掉即回到接入前的单遍 stage1 直出，便于 A/B 对照）。
        let selfLift = d.object(forKey: "h3SelfLiftEnabled") as? Bool ?? true
        h3SelfLiftEnabled = selfLift
        // ★ 二阶段解耦采样默认开启：低清 2 NFE + 高清 3 NFE = 总 NFE 5（比官方 N=6 直出还少 1 步）。
        h3SelfLiftDecouple = d.object(forKey: "h3SelfLiftDecouple") as? Bool ?? true
        // ★ 默认开启 CQ 清晰度增强：原 IC 像素桥二采存在崩坏/换脸问题，改为默认走 CQ 通道；
        // 需要回退原 IC 二采时在本页关闭本项即可（两通道并存、互斥、CQ 优先）。
        let cq = d.object(forKey: "videoUseCQEnhancer") as? Bool ?? true
        videoUseCQEnhancer = cq
        // （2026-09-21：videoUseDiffusionDecoder 设置字段已移除，不再读取）
        // ★ 第三套二采·Apple 超分：默认开启（运行期不支持/失败会自动回退 CQ/IC 二采，故默认开启不影响可用性）；
        // 倍率固定 ×4（面板已无倍率选择），质量默认 1=normal，预计算光流默认关（耗时/显存更高）。
        let appleSR = d.object(forKey: "videoUseAppleSR") as? Bool ?? true
        videoUseAppleSR = appleSR
        // ★ 后处理方式：优先读已存枚举；不存在时按旧开关组合迁移（与 currentPostProcessMode 同优先级），
        //   迁移结果立即落库，之后以下拉菜单为准。
        if let modeRaw = d.object(forKey: "videoPostProcessMode") as? String,
           let mode = H3PostProcessMode(rawValue: modeRaw) {
            videoPostProcessMode = mode
        } else {
            // 2026-09-18：IC/CQ 二采路径放弃。旧偏好存了 ic/cq（或旧开关组合 stage2 开）
            // 一律迁移为 .selflift（H3 自己 lift），并关闭旧二采底层开关，保证队列不再误走 stage2。
            let migrated: H3PostProcessMode
            if appleSR {
                migrated = .apple
            } else if selfLift || stage2 {
                migrated = .selflift
            } else {
                migrated = .none
            }
            if stage2 {
                videoUseStage2 = false
                videoUseCQEnhancer = false
            }
            videoPostProcessMode = migrated
            defaults.set(migrated.rawValue, forKey: "videoPostProcessMode")
        }
        appleSRScaleFactor = d.object(forKey: "appleSRScaleFactor") as? Int ?? 4
        appleSRQualityRawValue = d.object(forKey: "appleSRQualityRawValue") as? Int ?? 1
        appleSRUsePrecomputedFlow = d.object(forKey: "appleSRUsePrecomputedFlow") as? Bool ?? false
        // ★ H3→LTX latent 直通：默认开启；权重不在默认位置或推理失败时管线自动回退原像素桥路径，
        // 故默认开启不影响可用性（回退链路完整保留）。
        videoUseH3LTXAdapter = d.object(forKey: "videoUseH3LTXAdapter") as? Bool ?? true
        let savedAdapterPath = d.string(forKey: "h3LTXAdapterPath") ?? ""
        if savedAdapterPath.isEmpty {
            // ★ 2026-09-19 崩溃修复：init 期间禁止经 CommonPaths.modelRoot / Self.defaultH3LTXAdapterPath
            // 反向访问 AppSettings.shared（static let 初始化重入 → dispatch_once EXC_BREAKPOINT）。
            // 直接用 init 已解析的局部变量 resolvedRootPath 拼默认路径（两阶段初始化禁访问 self 属性）。
            h3LTXAdapterPath = URL(fileURLWithPath: resolvedRootPath)
                .appendingPathComponent("model/H3-to-LTX-Latent-Adapter.safetensors").path
        } else {
            h3LTXAdapterPath = savedAdapterPath
        }
    }

    // 默认文档地址（~/Documents/无限画布）
    static var defaultRootPath: String {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("无限画布").path
    }

    /// H3→LTX latent 直通适配器权重默认路径（统一模型根下，便于随模型一起分发）
    /// ★ 2026-09-19 崩溃修复：不依赖 CommonPaths.modelRoot（其会访问 AppSettings.shared，
    /// 在 AppSettings.init 期间调用会形成 static let 初始化重入 → EXC_BREAKPOINT）。
    /// 直接读 UserDefaults 的 canvasRootPath 拼路径；defaultRootPath 为纯函数，不触 shared，安全。
    static var defaultH3LTXAdapterPath: String {
        let saved = UserDefaults.standard.string(forKey: "canvasRootPath") ?? ""
        let root = saved.isEmpty ? defaultRootPath : saved
        return URL(fileURLWithPath: root)
            .appendingPathComponent("model/H3-to-LTX-Latent-Adapter.safetensors").path
    }

    // 可选节点主题色（有序，与工具栏一致）
    var colorOptions: [(name: String, color: Color)] {
        [
            ("pink", .pink),
            ("标准粉", Color(red: 0.98, green: 0.65, blue: 0.72)),
            ("blue", .blue),
            ("red", .red),
            ("orange", .orange),
            ("green", .green),
            ("purple", .purple),
            ("teal", .teal),
            ("black", .black)
        ]
    }

    // 当前默认节点颜色
    var defaultNodeColor: Color {
        colorOptions.first { $0.name == defaultNodeColorName }?.color ?? .pink
    }

    /// 当前默认节点颜色的 NSColor 版（与 colorOptions 同名一一对应，供位图绘制/占位图生成用）
    var defaultNodeNSColor: NSColor {
        switch defaultNodeColorName {
        case "pink": return .systemPink
        case "标准粉": return NSColor(calibratedRed: 0.98, green: 0.65, blue: 0.72, alpha: 1)
        case "blue": return .systemBlue
        case "red": return .systemRed
        case "orange": return .systemOrange
        case "green": return .systemGreen
        case "purple": return .systemPurple
        case "teal": return .systemTeal
        case "black": return .black
        default: return .systemPink
        }
    }

    /// 公共改色入口：偏好设置与画布右上角共用，改一边另一边同步（唯一颜色数据源）
    func setNodeColor(_ color: Color) {
        guard let name = colorOptions.first(where: { $0.color == color })?.name else { return }
        defaultNodeColorName = name
    }

    // 恢复默认参数
    func resetToDefaults() {
        defaultNodeColorName = "pink"
        wheelAccelerationPrecise = 2.0
        wheelAccelerationWheel = 30.0
        let oldPath = canvasRootPath
        canvasRootPath = Self.defaultRootPath
        if oldPath != canvasRootPath {
            migrateCanvasRoot(from: oldPath, to: canvasRootPath)
        }
    }

    /// 变更项目文件地址并迁移旧地址数据（选择新地址时调用）
    func changeCanvasRootPath(to newPath: String) {
        let oldPath = canvasRootPath
        guard oldPath != newPath, !oldPath.isEmpty, !newPath.isEmpty else { return }
        canvasRootPath = newPath
        migrateCanvasRoot(from: oldPath, to: newPath)
    }
}

// ============================================================

// MARK: - 偏好设置面板（左右分栏）

struct PreferencesView: View {
    @State private var selectedSection = 0
    @State private var showResetMenu = false
    @ObservedObject private var settings = AppSettings.shared
    // 第三套二采（Apple 超分）运行期状态：能力探测 / 模型资产状态 / 下载进度（生成流程也会回写）
    @ObservedObject private var appleSRState = AppleSRRuntimeState.shared

    private let sections = ["通用设置", "模型管理", "API", "授权设置", "快捷键", "关于"]

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            content
        }
        .frame(width: 620, height: 440)
        .background(Color(red: 0.95, green: 0.95, blue: 0.97))
        .background(FloatingWindow())
    }

    // 左侧导航栏
    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(sections.indices, id: \.self) { index in
                Button {
                    selectedSection = index
                } label: {
                    HStack {
                        Text(sections[index])
                            .font(.system(size: 13))
                            .foregroundColor(selectedSection == index ? .white : .primary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(selectedSection == index ? Color.accentColor : Color.clear)
                    )
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Spacer()
            // 左下角齿轮：恢复默认参数
            Button {
                showResetMenu.toggle()
            } label: {
                HStack {
                    Image(systemName: "gearshape")
                        .font(.system(size: 13))
                    Text("恢复默认")
                        .font(.system(size: 12))
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showResetMenu, arrowEdge: .bottom) {
                VStack(spacing: 10) {
                    Text("恢复默认参数")
                        .font(.system(size: 13, weight: .medium))
                    Text("将把所有设置恢复为默认值")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button("恢复默认") {
                        debugLog("偏好设置：恢复默认参数")
                        settings.resetToDefaults()
                        showResetMenu = false
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(14)
            }
        }
        .padding(10)
        .frame(width: 180)
        .background(Color(red: 0.90, green: 0.90, blue: 0.93))
    }

    // 右侧内容区
    private var content: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(sections[selectedSection])
                .font(.title2)
                .bold()
            Divider()
            if selectedSection == 0 {
                generalSettings
            } else if selectedSection == 1 {
                modelManagementSettings
            } else if selectedSection == 4 {
                shortcutSettings
            } else {
                Text("这里是「\(sections[selectedSection])」的内容区域，占位文本，后续补充。")
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.white)
    }

    // 通用设置页
    private var generalSettings: some View {
        VStack(alignment: .leading, spacing: 18) {
            // 项目文件地址（仅展示与打开所在位置；修改地址在启动预览界面选择）
            VStack(alignment: .leading, spacing: 6) {
                Text("项目文件地址")
                    .font(.system(size: 13, weight: .medium))
                HStack(spacing: 8) {
                    Text(settings.canvasRootPath)
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(Color(red: 0.95, green: 0.95, blue: 0.97))
                        .overlay(
                            RoundedRectangle(cornerRadius: 5)
                                .stroke(Color.gray.opacity(0.35), lineWidth: 1)
                        )
                    Button("打开该地址") {
                        debugLog("偏好设置：打开项目文件地址")
                        NSWorkspace.shared.open(URL(fileURLWithPath: settings.canvasRootPath))
                    }
                }
            }

            // 画布分组
            VStack(alignment: .leading, spacing: 12) {
                Text("画布")
                    .font(.system(size: 13, weight: .medium))

                VStack(alignment: .leading, spacing: 12) {
                    // 节点默认颜色
                    VStack(alignment: .leading, spacing: 8) {
                        Text("节点默认颜色")
                            .font(.system(size: 13, weight: .medium))
                        HStack(spacing: 10) {
                            ForEach(settings.colorOptions, id: \.name) { option in
                                Button {
                                    settings.setNodeColor(option.color)
                                } label: {
                                    ZStack {
                                        Circle()
                                            .fill(option.color)
                                            .frame(width: 22, height: 22)
                                        if settings.defaultNodeColorName == option.name {
                                            Image(systemName: "checkmark")
                                                .font(.system(size: 11, weight: .bold))
                                                .foregroundColor(.white)
                                        }
                                    }
                                    .overlay(
                                        Circle().stroke(settings.defaultNodeColorName == option.name ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: 2.5)
                                    )
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    // 滚轮速度倍率
                    HStack(spacing: 8) {
                        Text("触控板滚轮倍率")
                            .font(.system(size: 13))
                        TextField("", value: $settings.wheelAccelerationPrecise, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                    }
                    HStack(spacing: 8) {
                        Text("鼠标滚轮倍率")
                            .font(.system(size: 13))
                        TextField("", value: $settings.wheelAccelerationWheel, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 80)
                    }
                }
                .padding(.leading, 12)
            }

            Spacer()
        }
    }

    // 模型管理页：LTX-2.5 视频模型阶段设置 + 第三套二采（Apple 超分）设置
    private var modelManagementSettings: some View {
        // 内容较长（阶段2 设置 + 第三套二采 Apple 超分设置），改用 ScrollView 承载，避免固定窗口内被裁切
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                // ★ H3 一采设置：第一阶段总步数 N（4–12 滑杆）。
                // N 就是采样链路的 H3 一采总步数（队列 → generateVideo(steps:) → sigmaSchedule →
                // N = sigmas.count - 1），SelfLift 第三分支的 ts 由官方 75% 规则随 N 推导，面板不写死 ts。
                VStack(alignment: .leading, spacing: 12) {
                    Text("H3 设置")
                        .font(.system(size: 13, weight: .medium))

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Text("总步数")
                                .font(.system(size: 13))
                            Text("\(settings.h3Stage1Steps) 步")
                                .font(.system(size: 12, weight: .medium))
                        }

                        Slider(value: Binding(
                            get: { Double(settings.h3Stage1Steps) },
                            set: { settings.h3Stage1Steps = Int($0.rounded()) }
                        ), in: Double(AppSettings.h3Stage1StepsRange.lowerBound)...Double(AppSettings.h3Stage1StepsRange.upperBound), step: 1)
                            .frame(maxWidth: 320)

                        // 档位刻度：4–12；4/5/6 步档下方分别标注「测试/推荐/更佳」
                        HStack(spacing: 0) {
                            ForEach(Array(AppSettings.h3Stage1StepsRange), id: \.self) { n in
                                VStack(spacing: 1) {
                                    Text("\(n)")
                                        .font(.system(size: 10))
                                        .foregroundColor(n == settings.h3Stage1Steps ? .primary : .secondary)
                                    Text(n == 4 ? "崩坏" : n == 5 ? "勉强" : n == 6 ? "推荐" : " ")
                                        .font(.system(size: 9))
                                        .foregroundColor(.secondary)
                                }
                                .frame(maxWidth: .infinity)
                            }
                        }
                        .frame(maxWidth: 320)

                        // ★ 2026-09-18 UI 收敛：SelfLift 渐进采样 / 二阶段解耦采样不再出勾选框——
                        //   是否走 SelfLift 由下方「后处理方式」下拉菜单统一决定（h3SelfLiftEnabled 由
                        //   applyPostProcessMode 联动写回，禁止用户单独勾）；解耦调度固定默认开，
                        //   需要关闭时用环境变量 NA_H3_SELFLIFT_DECOUPLE=0（避免与菜单模式打架）。
                    }
                    .padding(.leading, 12)
                }

                // 阶段2 设置
                VStack(alignment: .leading, spacing: 12) {
                    Text("阶段2 设置")
                        .font(.system(size: 13, weight: .medium))

                    VStack(alignment: .leading, spacing: 12) {
                        // ★ 2026-09-18 UI 收敛：「阶段2 开关」不再出勾选框——阶段2 是否执行由下方
                        //   「后处理方式」下拉菜单统一决定（videoUseStage2 由 applyPostProcessMode
                        //   联动写回，禁止用户单独勾）。菜单为「无」时即 H3 直出、啥都不走。

                        // ★ 后处理方式（下拉菜单单选，2026-09-18）：统一入口，避免手动勾错组合。
                        // 显示值 = currentPostProcessMode（由底层开关实时推导，手动改开关时菜单自动刷新）；
                        // 选择值 = applyPostProcessMode（联动下方「阶段2 开关 / SelfLift / Apple 超分」）。
                        // 2026-09-18 UI 修正：标题与菜单并排顶在前部（与其他设置一致），说明文字移到下方。
                        HStack(spacing: 8) {
                            Text("模式")
                                .font(.system(size: 13))
                            Picker("", selection: Binding(
                                get: { settings.currentPostProcessMode },
                                set: { settings.applyPostProcessMode($0) }
                            )) {
                                ForEach(H3PostProcessMode.allCases) { mode in
                                    Text(mode.displayName).tag(mode)
                                }
                            }
                            .pickerStyle(.menu)
                            .fixedSize()
                        }
                        if settings.currentPostProcessMode.footnote != "" {
                            Text(settings.currentPostProcessMode.footnote)
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                                .padding(.leading, 2)
                        }

                        // ★ H3→LTX latent 直通（H3-to-LTX-Latent-Adapter）：把 H3 clean latent 直接
                        // 映射为 LTX 归一化 latent，省去「H3 VAE 解码 → 像素 → LTX VAE 编码」两步像素往返。
                        // 2026-09-18 UI 收敛：不再出勾选框（避免与「后处理方式」菜单混在一起、影响用户判断）——
                        // 固定默认开启（AppSettings 默认 true），权重缺失/形状不符/推理失败自动回退原像素桥路径；
                        // 需要关闭时用环境变量 LTX_H3_ADAPTER=0（与解耦等底层开关同一处理口径）。

                        // （2026-09-21：「扩散视频解码器」设置项已移除，固定使用卷积 VAE 解码）
                    }
                    .padding(.leading, 12)
                }

                // ★ 二采方案三：Apple VideoToolbox 超分（VTSuperResolutionScaler）
                appleSRSecondPassSettings

                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onAppear {
            // 进入本页即刷新运行期能力与模型资产状态（均为系统同步查询，开销极小）
            appleSRState.refreshCapability(scaleFactor: settings.appleSRScaleFactor,
                                           qualityRawValue: settings.appleSRQualityRawValue,
                                           usePrecomputedFlow: settings.appleSRUsePrecomputedFlow)
        }
    }

    // 二采方案三：Apple VideoToolbox 超分设置块
    //  - 开关 / 倍率（候选项由运行期 supportedScaleFactors 生成）；质量优先级系统仅公开 Normal，不出 UI
    //  - 是否使用预计算光流
    //  - 运行期能力探测结果（isSupported / supportedScaleFactors / supportedPixelFormats / 尺寸范围）
    //  - 模型资产状态、下载进度与手动下载、重新探测
    private var appleSRSecondPassSettings: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("二采方案三：Apple 4x超分（VideoToolbox）")
                .font(.system(size: 13, weight: .medium))

            VStack(alignment: .leading, spacing: 12) {
                // ★ 2026-09-18 UI 收敛：「启用 Apple 超分二采」不再出勾选框——是否走 Apple 二采由
                //   上方「后处理方式」下拉菜单统一决定（videoUseAppleSR 由 applyPostProcessMode
                //   联动写回，禁止用户单独勾）；本区域仅保留 Apple 二采的固定参数配置。

                // 倍率：面板已取消下拉选择，固定使用运行期唯一候选（高质量款本机 supportedScaleFactors=[4]）。
                // 实际生效倍率始终经 supportedScaleFactors 校验，候选变化时按阶梯回退，不硬编码生效值。
                HStack(spacing: 6) {
                    Text("倍率")
                        .font(.system(size: 12))
                    Text("固定 \(appleSRScaleOptionsSummary())")
                        .font(.system(size: 12, weight: .medium))
                    Spacer()
                }
                .disabled(settings.currentPostProcessMode != .apple)

                Toggle(isOn: $settings.appleSRUsePrecomputedFlow) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("使用预计算光流")
                            .font(.system(size: 12))
                        Text("开启：逐帧先算前/后向光流再做超分（质量更稳，耗时/显存更高）；不可用时自动退化为不使用光流")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                    }
                }
                .toggleStyle(.checkbox)
                .disabled(settings.currentPostProcessMode != .apple)

                // 运行期能力探测结果（只读展示）
                VStack(alignment: .leading, spacing: 3) {
                    Text("运行期探测：\(appleSRSupportText)　倍率候选 \(appleSRScaleOptionsSummary())　质量 Normal（系统唯一有效值）")
                        .font(.system(size: 11))
                    Text(appleSRState.capabilitySummary)
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
                )

                // 模型资产：状态 / 下载进度 / 手动下载 / 重新探测
                VStack(alignment: .leading, spacing: 6) {
                    Text("模型资产：\(appleSRState.modelStatusText)")
                        .font(.system(size: 11))
                    if appleSRState.downloading || (!appleSRState.modelReady && appleSRState.modelProgress > 0) {
                        ProgressView(value: min(max(appleSRState.modelProgress, 0), 1))
                            .frame(maxWidth: 240)
                    }
                    if !appleSRState.lastError.isEmpty {
                        Text("最近错误：\(appleSRState.lastError)")
                            .font(.system(size: 10))
                            .foregroundColor(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 10) {
                        Button("下载模型") {
                            debugLog("偏好设置：手动下载 Apple 超分模型资产")
                            appleSRState.startManualDownload(scaleFactor: settings.appleSRScaleFactor,
                                                             qualityRawValue: settings.appleSRQualityRawValue,
                                                             usePrecomputedFlow: settings.appleSRUsePrecomputedFlow)
                        }
                        .buttonStyle(.bordered)
                        .disabled(settings.currentPostProcessMode != .apple || appleSRState.downloading || appleSRState.modelReady)

                        Button("重新探测") {
                            appleSRState.refreshCapability(scaleFactor: settings.appleSRScaleFactor,
                                                           qualityRawValue: settings.appleSRQualityRawValue,
                                                           usePrecomputedFlow: settings.appleSRUsePrecomputedFlow)
                        }
                        .buttonStyle(.bordered)

                        Text("环境变量：LTX_APPLE_SR / LTX_APPLE_SR_SCALE / LTX_APPLE_SR_QUALITY / LTX_APPLE_SR_FLOW")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                }
            }
            .padding(.leading, 12)
        }
    }

    // Apple 超分支持状态文本（面板展示）
    private var appleSRSupportText: String {
        guard let supported = appleSRState.supported else { return "未探测" }
        return supported ? "支持" : "不支持"
    }

    // 快捷键设置页：展示画布快捷键（与 快捷键.swift 中实现一致）
    private var shortcutSettings: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("画布快捷键")
                .font(.system(size: 13, weight: .medium))

            VStack(spacing: 0) {
                shortcutRow(name: "改名", keys: ["F2"])
                Divider()
                shortcutRow(name: "缩放适配", keys: ["空格"])
                Divider()
                shortcutRow(name: "撤销", keys: ["Command", "Z"])
                Divider()
                shortcutRow(name: "重做", keys: ["Command", "Alt", "Z"])
                Divider()
                shortcutRow(name: "复制节点", keys: ["Command", "C"])
                Divider()
                shortcutRow(name: "粘贴节点", keys: ["Command", "V"])
                Divider()
                shortcutRow(name: "删除", keys: ["Delete"])
            }
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(red: 0.95, green: 0.95, blue: 0.97))
            )

            Text("提示：在画布中按对应快捷键即可触发，与工具栏按钮操作等同。")
                .font(.system(size: 11))
                .foregroundColor(.secondary)

            Spacer()
        }
    }

    // 快捷键行：功能名 + 键帽
    private func shortcutRow(name: String, keys: [String]) -> some View {
        HStack {
            Text(name)
                .font(.system(size: 13))
            Spacer()
            HStack(spacing: 4) {
                ForEach(keys, id: \.self) { key in
                    Text(key)
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Color.white)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 4)
                                        .stroke(Color.gray.opacity(0.4), lineWidth: 1)
                                )
                        )
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }
}

// ============================================================

// MARK: - 偏好设置窗口置顶

/// 让偏好设置窗口保持 floating 层级，点击主体窗口时不会沉到后面。
struct FloatingWindow: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            view.window?.level = .floating
        }
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
