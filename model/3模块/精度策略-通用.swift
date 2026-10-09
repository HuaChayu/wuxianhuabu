//
//  精度策略-通用.swift
//  无限画布 — dtype 通用化（第 0 层标量工厂 + 第 1 层精度分支策略）
//
//  来源：2026-09-03 由 model/MLXDType.swift + model/PrecisionPolicy.swift
//  合并安家到 3模块/（用户决策：不并入 模型加载-通用.swift，保持加载与精度分层）。
//
//  ============================================================
//  分层职责（无表设计，2026-09-03 用户拍板）：
//  · 第 0 层（标量工厂）：收编「与张量混算的裸标量」，标量 dtype 永远绑定参考张量。
//  · 第 1 层（精度策略）：不做任何模型/组件表（表会大量增加维护成本）。
//    全工程统一一个入口——全局默认档 = env 覆盖 ?? 芯片默认链首档；
//    已验证过的组件特判（如 H3 audio 需 fp16）在调用点显式锁一次，
//    其余一律走全局默认。任何大模型接入零策略配置即可按硬件自动跑。
//  · 第 2 层（预留）：M5 int8 实机量化计算路径（占位，asDType 返回 nil）。
//    自动回退闸门：未实现（2026-09-03 决策暂不做），回退链仅作文档化顺序，
//    手动兜底用 env H3_PREC 锁档；未来如需自动降级再补 load 时 round-trip 误差判定。
//  ============================================================

import Foundation
import MLX

// MARK: - 第 0 层：公共标量工厂（原 MLXDType.swift）

/// 背景：MLX 沿用 NumPy 风格 dtype 提升（bf16 + fp32 -> fp32）。裸写
/// MLXArray(0.5) / MLXArray(1.0) 会得到 fp32 标量，任何与 bf16 张量的混算
/// 都会把整条链顶成 fp32（曾导致 259 个 LoRA 模块全链污染）。本扩展收编所有
/// “与张量混算的裸标量”，使标量 dtype 永远绑定参考张量，杜绝隐式提升。
///
/// 用法：MLXArray.scalar(1.0, like: x)
/// - 参考张量为 fp32（VAE/upscaler/有意末端岛）时与旧行为逐位一致；
/// - 参考张量为 bf16 时保持主链精度（bf16 主链 + 统计区瞬时 fp32 的策略见
///   第 1 层；未来接入 M5 int8：权重打包 + bf16 激活，标量绑激活张量天然兼容，
///   无需改动调用点）。
extension MLXArray {

    /// Float 标量工厂：dtype 跟随参考张量。
    static func scalar(_ value: Float, like ref: MLXArray) -> MLXArray {
        MLXArray(value, dtype: ref.dtype)
    }

    /// Double 字面量版本（0.5 / 0.044715 / sqrt 等字面量默认是 Double）。
    static func scalar(_ value: Double, like ref: MLXArray) -> MLXArray {
        MLXArray(value, dtype: ref.dtype)
    }

    /// Int 版本（0 / 1 / -1 等边界常数）。
    static func scalar(_ value: Int, like ref: MLXArray) -> MLXArray {
        MLXArray(value, dtype: ref.dtype)
    }
}

// MARK: - 第 1 层：计算精度分支（原 PrecisionPolicy.swift）

/// 计算精度分支：只描述「激活与计算」走哪个精度（dtype 通用化第 1 层）。
///
/// 与权重文件原生 dtype 解耦：不关心权重是 F16/BF16/F32/4bit，主链统一走分支。
/// 16bit matmul 内部累加 fp32 是硬件行为；norm/softmax/finalHead 等瞬时 fp32 岛
/// 由第 0 层判据保留（内部提升、出口回落），与本分支正交。
public enum ComputeTier: String, CaseIterable {
    /// 激活与计算走 int8（M5 Max 硬件加速）。权重可任意/4bit，不归本层管。
    case int8
    /// 激活/权重 fp16（尾数 10bit，存储保真；动态范围 ±65504）。
    case fp16
    /// 激活/权重 bf16（尾数 8bit，动态范围同 fp32，激活安全）。
    case bf16

    /// int8 在 MLX 暂无「int8 激活 dtype」的通用张量表达，M5 实机走量化计算路径，
    /// 此处映射 nil，由上层决策（占位，接入 M5 时再定实现）。
    public var asDType: DType? {
        switch self {
        case .int8: return nil
        case .fp16: return .float16
        case .bf16: return .bfloat16
        }
    }
}

/// 当前芯片（M 系列代数）。
/// 读取 sysctl machdep.cpu.brand_string（如 "Apple M4 Max" → 4）解析。
public enum HWChip {
    public static let brand: String = {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "" }
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return String(cString: buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }()

    /// M 系列代数：Apple M5 → 5。非 Apple Silicon 返回 0。
    public static let generation: Int = {
        // 匹配 "Apple M(\d+)" 或 "Apple M(\d+) Max/Pro/Ultra"
        let pattern = #"Apple M(\d+)"#
        guard let r = brand.range(of: pattern, options: .regularExpression) else { return 0 }
        let num = brand[r].filter(\.isNumber)
        return Int(num) ?? 0
    }()

    public static let isM5OrLater: Bool = generation >= 5

    /// 硬件默认回退链：
    /// - M5 及以后：int8 → bf16 → fp16（2026-09-03 前拍板口径）
    /// - 更早（M1~M4）：bf16 → fp16
    /// 2026-09-03 补充拍板：MLX 未实现 int8 激活计算，M5 链首 int8 目前不可表达，
    /// resolveDType 会沿链自动落到 bf16 —— 实际 M5 与 M1~M4 一致、都从 bf16 起。
    /// int8 与芯片代数分支保留（HWChip 继续解析），待 MLX int8 计算路径落地后
    /// 补 ComputeTier.int8.asDType 即自动生效，无需改链序与调用点。
    public static func defaultTierChain() -> [ComputeTier] {
        isM5OrLater ? [.int8, .bf16, .fp16] : [.bf16, .fp16]
    }
}

/// env 手动覆盖精度分支（端到端发现画面/音频质量问题后锁精度，无需改代码）。
/// 例：H3_PREC=int8 / fp16 / bf16
/// 注：env 名沿用 H3 阶段命名（H3_PREC），对全工程所有模型通用，跨模型均可覆盖。
public func precisionEnvOverride() -> ComputeTier? {
    guard let raw = ProcessInfo.processInfo.environment["H3_PREC"]?.lowercased(),
          let tier = ComputeTier(rawValue: raw) else { return nil }
    return tier
}

/// 全局精度策略入口（无表设计——不维护任何模型/组件策略表）。
///
/// 生效优先级：env 覆盖（H3_PREC）> 组件调用点显式锁（componentLock）> 芯片默认链首档。
/// - 绝大多数组件不传 componentLock，自动走硬件默认（int8 未落地前所有芯片 = bf16）；
/// - 已验证必须特判的组件（如 H3 audio 需 fp16 保真）在 load 调用点显式锁一次并挂验证注释；
/// - env 永远能压过组件锁，作为手动回退/出问题时的逃生门。
public enum PrecisionPolicy {

    /// 解析最终生效的精度分支。
    public static func resolve(componentLock tier: ComputeTier? = nil) -> ComputeTier {
        precisionEnvOverride() ?? tier ?? HWChip.defaultTierChain().first!
    }

    /// 解析最终生效的主链 DType。
    /// MLX 尚未实现 int8 激活计算（ComputeTier.int8.asDType 为 nil 占位），
    /// 因此 M5 链首 int8 不可表达时沿默认链自动落到下一可用档（bf16），
    /// 与 M1~M4 行为一致、都从 16bit 起；未来 int8 计算路径落地后补上
    /// asDType 即自动启用，无需改任何调用点。
    public static func resolveDType(componentLock tier: ComputeTier? = nil) -> DType {
        let chosen = resolve(componentLock: tier)
        if let dt = chosen.asDType { return dt }
        return HWChip.defaultTierChain().compactMap(\.asDType).first ?? .bfloat16
    }

    /// 全局默认主链 dtype（无组件锁，供组件 load 默认参数使用）。
    public static var defaultMainDType: DType {
        resolveDType()
    }
}
