//
//  SelfLift渐进采样.swift
//  无限画布 — model/3模块/SelfLift通用/
//
//  通用 SelfLift 渐进式采样核心（SelfLift-zero，免训练）
//  参考：comfyui-SelfLift（arXiv:2609.02036 Algorithm 1 / mode = zero）
//
//  ── 它做什么 ────────────────────────────────────────────────────────────
//  在「单次采样调度内部」把总步数切成两段：
//
//      σ:   σ0 > σ1 > … > σ_{ts-1} > σ_ts > … > σ_N
//           |<----- 低分辨率段（ts 次 NFE）---->| ← 切换点
//                                                |<-- 高分辨率段（N-ts 次）-->|
//
//  切换点处做一次「零 NFE 过渡块」：
//      Eq.3    x0 = x - σ_k · v                 （复用低分最后一次前向；该步 Euler 更新被丢弃）
//      Eq.4-5  z_lat = interp(x0)               （直接 latent 提升；官方 H3 路径固定 nearest）
//              z_pix = VAE⁻¹( ↑(VAE(x0)) )      （VAE 像素往返，作为伪影度量锚点）
//      Eq.6-9  δ = z_pix - z_lat；按 |δ| 取风险最高的 ρ 比例位置、强度线性映射 w ∈ [w_min, w_max]
//              z0_high = z_lat + w·δ
//      Eq.10   x_k = (1-σ_k)·z0_high + σ_k·ε    （重加噪，官方 noise_scaling 公式）
//              再手工补一次 (σ_k → σ_next) 的 Euler
//
//  不变量：低分 NFE = ts，高分 NFE = N - ts，合计恒等于原总步数 N（过渡块零 NFE）。
//  加速来源：前 ts 次前向跑在 lowResScale 的低空间分辨率上，单步算力按面积下降。
//
//  ── 参数对齐（唯一参考：comfyui-SelfLift 官方实现，禁止自创）─────────────
//  H3 一采 = 6 步（N = 6，本工程 H3Pipeline 默认 steps = 6）。官方 H3 走 SelfLift-zero 本体，
//  其建议起点为「upscaler_model = none + rho > 0：rho = 0.6、w_min = w_max = 1」（README 行 29 / 61）：
//      transitionStep  = 不写死，按官方 75% 比例随总步数 N 联动推导：
//                          ts = clamp(floor(0.75 · N), 1, N-1)
//                        依据＝官方两个论文实例的比例均为 75%（FLUX.2-Klein 3/4、Z-Image-Turbo 6/8）；
//                        本工程 H3 一采 N = 6 → ts = 4，与原写死值一致。N 取自面板第一阶段总步数滑杆
//                        （4–12），ts 因此随滑杆自动变化；clamp 上界同时满足官方硬约束 ts ≤ N-1。
//                        调用方仍可显式覆盖（transitionStep > 0 时优先于推导值）。
//      lowResScale    = 0.5    官方节点默认值，也是官方诊断表的固定值（nodes.py 行 481 / README 行 54）
//      rho            = 0.0    官方 SelfLiftH3Sampler 默认值（learned upscaler 时纯 z_lat 提升，
//                              像素锚点为默认关闭的可选增强；NA_H3TEST=26 实测 rho>0 注入 z_pix
//                              VAE 往返伪影——运动后段背景角色双轮廓，learned+rho=0 前后段全干净）
//                              README 行 29/61 的「rho=0.6」仅针对 upscaler_model=none（无升频器）建议
//      wMin = wMax    = 1.0    官方 H3 建议起点（README 行 29 / 61）
//      提升插值        = nearest 官方 H3 路径硬编码，不作为可选项（nodes.py 行 509 传入 "nearest"）
//      采样器          = 标准 euler 且 s_churn = 0（README 行 79：Only standard Euler with s_churn=0）
//      cfg            = 5.0    官方 H3 节点默认值（nodes.py 行 480）。本工程一采循环无 CFG 分支，
//                              故仅作对齐记录（见 SelfLiftConfig.officialH3Cfg），不新增参数
//  官方硬约束：1 ≤ ts ≤ N-1（nodes.py 行 98 `_validate_schedule`，上界即 sigmas.numel()-2）。
//  官方节点默认值 ts = 6 在 N = 6 下越界，本工程禁止使用。
//
//  ── 与模型解耦 ──────────────────────────────────────────────────────────
//  本文件不做任何模型相关动作（不碰 layout / rope / VAE / packed rows / VAE 归一化）。
//  模型侧只需提供下面这些能力（全部以闭包注入，见 SelfLiftTransition.run）：
//      1. liftLatent        低分 x0 → 高分 x0（直接插值提升）
//      2. liftPixelAnchor   低分 x0 → 高分 x0（VAE 往返），可选
//      3. consistencyLift   (zLat, zPix) → z0_high（Eq.6-9）
//      4. noiseScaling      (x0, σ, ε) → x（Eq.10）
//      6. 低分/高分两段采样循环 + 分辨率切换后的上下文重建（在各模型的 Runner 里）
//  因此 H3 / LTX / 其它 packed-rows 视频模型各写一份 Runner 即可复用本核心。
//

import Foundation
import MLX

// MARK: - 提升方式

// 官方 comfyui-SelfLift 在 H3 节点里把「直接 latent 提升」的插值方式硬编码为 nearest：
//   · selflift.py 行 18 `paired_lifts(..., latent_mode="nearest", ...)` 的默认值
//   · nodes.py 行 509 H3 节点调用 progressive_sample 时实参写死 "nearest"
// 官方 H3 路径不暴露该选项（Image 节点才有 nearest/bilinear 开关），
// 因此本核心不再提供插值开关，统一固定 nearest——由各模型 Runner 照抄官方实现。

// MARK: - 配置

public struct SelfLiftConfig: Sendable {
    /// 总开关
    public var enabled: Bool
    /// ts：低分辨率前向次数（同时是 sigma 数组上的切换下标），有效范围 [1, N-1]
    ///
    /// **显式覆盖值**：
    ///   · > 0 → 直接用该值（需自行满足 1 ≤ ts ≤ N-1，否则 SelfLiftScheduleSplit.make 抛错）；
    ///   · = 0（默认）→ 视为「不传」，由总步数 N 按官方 75% 比例联动推导，
    ///                  取 `SelfLiftScheduleSplit.transitionStep(forTotalSteps: N)`。
    /// 推导依据：官方两个论文实例的比例均为 75%（FLUX.2-Klein 3/4、Z-Image-Turbo 6/8），
    /// 本工程 H3 一采 N = 6 → ts = 4。N 即面板第一阶段总步数滑杆（4–12）。
    /// 注意官方节点默认值 6 在 N = 6 下越界（nodes.py 行 98 硬约束 ts ≤ N-1 = sigmas.numel()-2），禁止使用。
    public var transitionStep: Int
    /// 低分辨率倍率（相对目标分辨率），0.5 = 半分辨率起步
    /// 官方节点默认值 + 官方诊断表固定值（nodes.py 行 481 / README 行 54）
    public var lowResScale: Double
    /// ρ：只修正风险最高的 ρ 比例位置；0 = 纯 latent 提升（省掉一次 VAE 往返）
    /// 默认 0.0 = 官方 SelfLiftH3Sampler 默认（learned upscaler 纯 z_lat 提升，像素锚点默认关闭）；
    /// README 行 29/61 的 0.6 仅针对 upscaler_model=none（无升频器）路径
    public var rho: Double
    /// 修正强度下界 / 上界
    /// 官方 H3 建议起点 w_min = w_max = 1.0（README 行 29 / 61）
    public var wMin: Float
    public var wMax: Float

    /// CFG 引导强度（官方 SelfLiftH3Sampler 默认 cfg=5.0：低清段+高清段全程
    /// positive/negative 双路引导）。0 = 关闭 CFG（工程旧行为，仅 positive 单路）；
    /// >0 时每步 v = v_neg + cfg·(v_pos − v_neg)，negative 条件行由调用方注入
    /// （见 runH3Stage1WithSelfLift 的 textStatesNeg 参数）。
    /// 动机（2026-09-18 重影根因）：低清网格空间量化使背景边缘与 keyframe 原版错位，
    /// 无 CFG 时高清段对「低清错位背景 + keyframe 精确背景」折中 → 背景双像；
    /// CFG 引导给 keyframe 结构裁决力，压制低清残留。
    public var cfgScale: Double

    /// σ_k 覆盖入口（低清终点 / 交接深度）：nil 时用 `makeFixedFinalStep` 默认 0.75（落调度前吸附公式档）；
    /// 环境变量 NA_H3_SELFLIFT_DECOUPLE_LOW_K 可覆盖（命名沿用旧解耦 A/B 时代）。
    public var decoupleSigmaK: Double?

    /// 交接状态类型开关（2026-10-19 用户需求，默认 false = 带噪直通）：
    ///   · false（默认）：低清末步不做 Eq.3 clean 预测，正常 Euler 走到 σ_k，交接的是 σ_k 处的
    ///     **带噪 x**；升频后不重加噪、不补 Euler，高清直接接手这份含噪态继续跑到 0。
    ///     动机：重影根因正是「σ_k 处预测的半成品 clean x0（内容权重仅约 7.7%）升频后被重加噪，
    ///     与高清重画的内容叠加成鬼影」；高清需要的是噪点状态而不是 clean 内容。
    ///   · true：恢复旧路径（Eq.3 clean x0 → 升频 → 按 σ_next 重加噪）。
    ///     **该路径仅关闭、不从代码移除**，供 A/B 对比与 lowOnly 二采链路使用。
    /// 环境变量 NA_H3_SELFLIFT_CLEAN_HANDOFF=1 可显式恢复旧路径。
    public var cleanEndpointHandoff: Bool

    /// 重加噪深度 σ_ren（2026-10-19 用户设计）：低清跑满到 σ=0 后，clean x0 升频到高清，
    /// 再按 σ_ren 重加噪（Eq.10 新噪声）交给高清收尾。nil = 跟随 sigmaK（等深连续，旧语义）；
    /// σ_k=0 时默认取 0.3（环境变量 NA_H3_SELFLIFT_RENOISE_SIGMA 可覆盖）。
    /// 动机：σ_k=0.5 等深交接时高清要从 50% 噪声里一次重建全部细节，单步跨度过大 →
    /// 纹理/色彩先验接管（风格大变）+ 每帧噪声残差不同步（闪烁）；改成「低清跑满出干净结构，
    /// 高清只从 0.3 的浅噪声里做一次细化」。
    public var renoiseSigma: Double?

    /// - Parameter transitionStep: ts 显式覆盖值；传 0（默认）表示不写死，按总步数 N 自动推导
    public init(enabled: Bool = true,
                transitionStep: Int = 0,
                lowResScale: Double = 0.5,
                rho: Double = 0.0,
                wMin: Float = 1.0,
                wMax: Float = 1.0,
                cfgScale: Double = 0.0,
                decoupleSigmaK: Double? = nil,
                cleanEndpointHandoff: Bool = true,
                renoiseSigma: Double? = nil) {
        self.enabled = enabled
        self.transitionStep = transitionStep
        self.lowResScale = lowResScale
        self.rho = rho
        self.wMin = wMin
        self.wMax = wMax
        self.cfgScale = cfgScale
        self.decoupleSigmaK = decoupleSigmaK
        self.cleanEndpointHandoff = cleanEndpointHandoff
        self.renoiseSigma = renoiseSigma
    }

    /// 解析出本次调度实际使用的 ts：
    ///   · transitionStep > 0 → 该显式值；
    ///   · transitionStep = 0 → 官方 75% 规则按总步数推导（`SelfLiftScheduleSplit.transitionStep(forTotalSteps:)`）。
    /// - Parameter totalSteps: 第一阶段总步数 N（面板滑杆 4–12）
    public func resolvedTransitionStep(totalSteps n: Int) -> Int {
        transitionStep > 0 ? transitionStep : SelfLiftScheduleSplit.transitionStep(forTotalSteps: n)
    }

    /// 是否走 VAE 像素锚点（rho>0 且 wMax>0）——为真时过渡块会多一趟 decode→encode
    /// 与官方 `need_pix = rho > 0.0 and w_max > 0.0` 一致（selflift.py 行 26-28）
    public var needsPixelAnchor: Bool { rho > 0 && wMax > 0 }
    /// 是否走直接 latent 提升（rho=w_min=w_max=1 的纯锚点模式下可省）
    /// 与官方 `need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)` 一致
    public var needsLatentLift: Bool { !(rho >= 1 && wMin >= 1 && wMax >= 1) }

    /// 官方 H3 节点默认 cfg = 5.0（nodes.py 行 480）。本工程一采循环没有 CFG 分支，此值仅作对齐记录。
    public static let officialH3Cfg: Float = 5.0
    /// 官方 H3 路径硬编码的「直接 latent 提升」插值方式（nodes.py 行 509 / selflift.py 行 18）
    public static let officialLatentUpsampleMode: String = "nearest"
}

// MARK: - 调度切分

public enum SelfLiftScheduleError: Error, CustomStringConvertible {
    case tooFewSteps(Int)
    case transitionOutOfRange(ts: Int, n: Int)
    case sigmasNotMonotonic(index: Int, prev: Double, cur: Double)
    case sigmasNonPositiveBeforeEnd(index: Int, value: Double)
    case lastSigmaNegative(Double)
    case transitionSigmaNotBelowOne(Double)

    public var description: String {
        switch self {
        case .tooFewSteps(let n):
            return "SelfLift: 总步数不足（N=\(n)），至少需要 2 步"
        case .transitionOutOfRange(let ts, let n):
            return "SelfLift: transitionStep=\(ts) 越界，有效范围 [1, \(max(1, n - 1))]"
        case .sigmasNotMonotonic(let i, let p, let c):
            return "SelfLift: sigma 非单调非增（i=\(i), \(p) → \(c)）"
        case .sigmasNonPositiveBeforeEnd(let i, let v):
            return "SelfLift: 非末项 sigma ≤ 0（i=\(i), σ=\(v)）"
        case .lastSigmaNegative(let v):
            return "SelfLift: 末项 sigma 为负（\(v)）"
        case .transitionSigmaNotBelowOne(let v):
            return "SelfLift: 切换点起采 σ 必须 < 1（σ[ts]=\(v)）"
        }
    }
}

/// 把一条 sigma 调度切成「低分前缀 + 高分收尾」，并给出过渡块需要的两个 σ。
public struct SelfLiftScheduleSplit: Sendable {
    /// 低分段：sigmas[0 ... ts]
    public let lowSigmas: [Double]
    /// 高分段：sigmas[ts ... N]
    public let highSigmas: [Double]
    public let transitionStep: Int
    /// σ_k = sigmas[ts-1]：切换点上一次评估所在的 σ（Eq.3 端点预测处）
    public let sigmaK: Double
    /// σ_next = sigmas[ts]：高分段起采 σ
    public let sigmaNext: Double

    public var lowNFE: Int { transitionStep }
    public var highNFE: Int { highSigmas.count - 1 }
    public var totalNFE: Int { lowNFE + highNFE }

    /// 官方 75% 比例系数：官方两个论文实例 FLUX.2-Klein（3/4）、Z-Image-Turbo（6/8）均为 0.75；
    /// 本工程 H3 一采 N = 6 → ts = 4 与之一致。
    public static let officialTransitionRatio: Double = 0.75

    /// 用户输入 σ 吸附到公式曲线时的采样分辨率（档数）。256 档时 0.75→0.7491、0.5→0.5042，
    /// 既保证「值是公式真实产出的档位」，又不会像按 N 步网格吸附那样把 0.5 吸到 0.7。
    public static let snapResolution: Int = 256
    // ══════════════════════════════════════════════════════════════════════
    // 【踩坑记录 · SelfLift 重影根因与修复】2026-09-17
    //
    // 现象：短场景（39 帧，seed=42，learned upscaler + rho=0）N 扫描中，
    //   N=8 成片出现明显重影（双像/残影），N=6/7/9 干净；
    //   用户 UI 观察确认：短9还行、短8明显重影。
    //
    // 根因（非随机，是 75% 规则自动推导的调度遗漏）：
    //   1) ts = floor(0.75·N) 推导后，σ_next = sigmas[ts] 随 N 落点不同；
    //   2) 过渡块重加噪时，高分起点 = (1-σ_next)·z0_high + σ_next·噪声，
    //      (1-σ_next) 即「低分内容残留进高分的权重」；
    //   3) N=8 时 ts=6、σ_next=0.8000，残留 20% 为全档最高，且高分仅 2 步、
    //      首步 dσ=0.168 最大，修正不足 → 低分时间维双像伪影残留成片。
    //
    // 各档位数值（sigmaShift=12, shift=12.0）：
    //   N=6: ts=4, σ_next=0.8571, 残留14.3%, 高分2步 ✅ 干净
    //   N=7: ts=5, σ_next=0.8276, 残留17.2%, 高分2步 ✅ 干净
    //   N=8: ts=6, σ_next=0.8000, 残留20.0%, 高分2步 ❌ 重影
    //   N=9: ts=6, σ_next=0.8571, 残留14.3%, 高分3步 ✅ 干净
    //
    // 解决方案：自动推导路径加 σ_next 下限约束（下方 minTransitionSigma = 0.86），
    //   σ_next < 0.86 时 ts 前移（低分少跑、高分多跑、过渡起点变浅）。
    //   第一版 0.85 只覆盖 N=8/10；h3_56（124 帧长视频、UI 默认 N=6）仍重影，
    //   故提到 0.86，让 N=6/7/9 也触发（0.8571/0.8276/0.8571 < 0.86）：
    //   修复后 N=6: ts=3, σ_next=0.9231, 残留7.7%,  高分3步
    //   修复后 N=7: ts=4, σ_next=0.9000, 残留10.0%, 高分3步
    //   修复后 N=8: ts=5, σ_next=0.8780, 残留12.2%, 高分3步，用户确认干净
    //   修复后 N=9: ts=5, σ_next=0.9057, 残留9.4%,  高分4步
    //   修复后 N=10: ts=6, σ_next=0.8889, 残留11.1%, 高分4步
    //
    // 大白话：高分起点 = 高清新画面 + 低清旧画面混合；低清旧画面里带着时间维重影，
    //   叠得越重（残留越大）、清洗次数越少（高分步数越少），重影越洗不掉。
    //   N=8 恰是叠最重(20%)又给最少(2步)的一档。
    //
    // 注意：
    //   - 仅约束自动推导路径（config.transitionStep == 0），显式指定 ts 不覆盖；
    //   - 0.86 为经验安全值，方向保守（只会让 ts 前移，不引入新风险）；
    //     长视频（124 帧）对时间维残留更敏感：残留 > 13% 的档位建议一律前移；
    //     N=11/12 自动推导后残留 ≤12.7%（N=11→ts=7 0.8727、N=12→ts=7 0.8955），无需再提阈值；
    //   - 换 sigma 调度风格（如 betaRefined）时 σ_next 分布不同，需重新核对落点；
    //   - 本问题与提示词/内容无关，纯调度层问题，换提示词同样受约束保护。
    //   ※ 该兜底方案已废弃移除（2026-10-08）：固定最后一步设计（低清 N-1 步线性到 σ_k +
    //     高清 1 步 σ_k→0，当前 σ_k=0.75，见 makeFixedFinalStep）取代自动 75% 推导，
    //     不再需要 σ_next 下限/前移兜底，直接跑。
    // ══════════════════════════════════════════════════════════════════════

    /// 按官方 75% 比例由总步数 N 推导切换点：`ts = clamp(floor(0.75 · N), 1, N-1)`。
    ///
    /// 上界取 N-1 即官方硬约束（nodes.py 行 98 `_validate_schedule`，上界 = sigmas.numel()-2）；
    /// 下界取 1 保证低分段至少 1 次 NFE。
    /// 面板滑杆 N ∈ 4–12 的推导值：4→3（测试档）、5→3、6→4（与本工程原写死值一致）、7→5、
    /// 8→6、9→6、10→7、11→8、12→9。
    /// - Parameter n: 第一阶段总步数 N（面板第一阶段总步数滑杆，4–12）
    public static func transitionStep(forTotalSteps n: Int) -> Int {
        guard n >= 2 else { return 1 }
        let derived = Int((officialTransitionRatio * Double(n)).rounded(.down))
        return min(max(derived, 1), n - 1)
    }

    /// 便捷入口：ts 由 `config` 解析（显式覆盖优先），其余校验与 `make(sigmas:transitionStep:)` 一致。
    /// - Parameter config: SelfLift 配置；`transitionStep = 0` 表示不写死
    public static func make(sigmas: [Double], config: SelfLiftConfig) throws -> SelfLiftScheduleSplit {
        try make(sigmas: sigmas, transitionStep: config.resolvedTransitionStep(totalSteps: sigmas.count - 1))
    }

    /// - Parameters:
    ///   - sigmas: 原始调度（严格非增、只有最后一项可为 0、rectified-flow 约定）
    ///   - ts: 低分辨率前向次数，且 1 ≤ ts ≤ N-1
    public static func make(sigmas: [Double], transitionStep ts: Int) throws -> SelfLiftScheduleSplit {
        let n = sigmas.count - 1
        guard n >= 2 else { throw SelfLiftScheduleError.tooFewSteps(n) }
        guard ts >= 1, ts <= n - 1 else {
            throw SelfLiftScheduleError.transitionOutOfRange(ts: ts, n: n)
        }
        for i in 0..<n where sigmas[i + 1] > sigmas[i] + 1e-9 {
            throw SelfLiftScheduleError.sigmasNotMonotonic(index: i + 1, prev: sigmas[i], cur: sigmas[i + 1])
        }
        for i in 0..<n where sigmas[i] <= 0 {
            throw SelfLiftScheduleError.sigmasNonPositiveBeforeEnd(index: i, value: sigmas[i])
        }
        guard sigmas[n] >= 0 else { throw SelfLiftScheduleError.lastSigmaNegative(sigmas[n]) }
        guard sigmas[ts] < 1.0 else {
            throw SelfLiftScheduleError.transitionSigmaNotBelowOne(sigmas[ts])
        }

        return SelfLiftScheduleSplit(
            lowSigmas: Array(sigmas[0...ts]),
            highSigmas: Array(sigmas[ts...n]),
            transitionStep: ts,
            sigmaK: sigmas[ts - 1],
            sigmaNext: sigmas[ts])
    }

    // ──────────────────────────────────────────────────────────────────────
    // 固定最后一步高清（M=1）· 公式网格调度（2026-10-25 用户需求改版）
    //
    //   旧版（已移除）：低清 L 步「线性等距」跑到 σ_k —— 每步 Δσ 恒为 (1-σ_k)/L。
    //   为什么移除：直出路径全程 σ 走 shift 曲线 σ(b)=shift·b/(1+(shift-1)·b)，等距 σ 只是
    //   另一条轨迹（shift=12、5 步时前几步 Δσ 只有公式的约 1/4，末步又远大于公式），
    //   低清段的评估点与直出路径对不上，两段口径不一致。
    //
    //   新口径：
    //     · 低清网格 = 公式曲线（与直出 `sigmaSchedule(steps:shift:)` 逐点同式），b 空间等距；
    //     · 用户输入的 σ_k（低清跑到哪）/ σ_ren（重加噪深度）先**吸附到公式网格上最近的 σ 档**，
    //       把 0.75 这种「线性口径的值」换成公式真实存在的档位；
    //     · 低清终点精确落在吸附后的 σ_k（b 端点反解，不靠反推取整），高清固定 1 步 σ_ren → 0。
    //   N=6、shift=12、σ_k=σ_ren=0.75 → 吸附到公式档 σ≈0.7491；低清 5 步（clean 为 4 个间隔 +
    //   末步 Eq.3 占位）、高清 1 步，总 NFE 与旧版一致仍为 N=6。
    // ──────────────────────────────────────────────────────────────────────
    /// - Parameter totalSteps: 面板第一阶段总步数 N（总 NFE = 低清步数 + 高清固定 1 步）
    /// - Parameter sigmaK: 用户输入的低清终点 / 交接深度（默认 0.75），内部吸附到公式最近档
    ///   （σ_k 历史 0.86→0.95→0.7→0.6→0.8→0.7→0.8→0.75→0.5→0.75；σ_ren 历史 0.8→0.5→0.55→0.75）
    /// - Parameter cleanEndpoint: true=末步 Eq.3 预测 clean x0 并重加噪（默认）；false=末步正常 Euler 落在 σ_k、交出含噪 x（带噪直通）
    /// - Parameter shift: 视频侧 shift（与直出路径同一个 shift，H3 默认 12.0）
    public static func makeFixedFinalStep(totalSteps: Int, sigmaK: Double = 0.75,
                                         renoiseSigma: Double? = nil,
                                         cleanEndpoint: Bool = true,
                                         shift: Double = 12.0) throws -> SelfLiftScheduleSplit {
        let n = totalSteps
        guard n >= 3 else { throw SelfLiftScheduleError.tooFewSteps(n) }
        // 吸附网格：公式曲线按 snapResolution 档采样（0.75 → 0.7491、0.5 → 0.5042 量级误差），
        // 用来把用户输入的 σ 换成公式曲线上真实存在的档位 —— 不再让 0.75 这种按线性口径取的
        // 值直接落进公式调度，也不用「反推 L 再取整」（那会带来终点漂移）。
        let snap = SelfLiftScheduleSplit.snapResolution
        let snapGrid = formulaGrid(intervals: snap, shift: shift)
        // ① 低清终点 σ_k：吸附到最近公式档（σ_k ≈ 0 表示低清跑满，终点取曲线末档 0）。
        let lowToZero = sigmaK <= 1e-9
        let sigmaKSnapped = lowToZero ? 0.0 : nearestGridValue(to: sigmaK, grid: snapGrid, range: 0...snap)
        // ② 重加噪深度 σ_ren：clean 端点默认 0.75（用户定档「与止点等深」），带噪直通跟随 σ_k；
        //    用户显式给了 renoiseSigma 就用它，统一吸附到同一张公式网格的最近档。
        let renTarget = renoiseSigma ?? (cleanEndpoint ? 0.75 : (lowToZero ? 0.75 : sigmaK))
        let sigmaNext = nearestGridValue(to: renTarget, grid: snapGrid, range: 1...snap)
        // ③ 低清网格：b 空间等距 → 公式 σ，末项精确等于吸附后的 σ_k（不按档截断，避免终点漂移）。
        //    步数与旧版完全一致（总 NFE 仍是 N）：
        //    · 带噪直通：N-1 个间隔，末步 Euler 真正落在 σ_k；
        //    · clean 端点：N-2 个间隔 + 末尾 σ_k 占位 —— 末步只做 Eq.3（Euler 丢弃）。
        let intervals = max(1, cleanEndpoint ? n - 2 : n - 1)
        var lowSigmas = formulaLadder(intervals: intervals, endSigma: sigmaKSnapped, shift: shift)
        if cleanEndpoint { lowSigmas.append(sigmaKSnapped) }
        let highSigmas = [sigmaNext, 0.0]          // 高清固定 1 步 σ_ren → 0
        return SelfLiftScheduleSplit(
            lowSigmas: lowSigmas,
            highSigmas: highSigmas,
            transitionStep: lowSigmas.count - 1,
            sigmaK: sigmaKSnapped,
            sigmaNext: sigmaNext)
    }

    // ── 公式网格三件套（与直出路径 sigmaSchedule 同式，供吸附与低清序列生成）──────────────

    /// 公式曲线：`σ_i = shift·b/(1+(shift-1)·b)`，`b = 1 - i/intervals`。
    /// 与 `sigmaSchedule(steps:shift:)` 逐点等价（intervals 即 steps）。
    public static func formulaGrid(intervals: Int, shift: Double) -> [Double] {
        let l = max(1, intervals)
        return (0...l).map { i in
            let b = 1.0 - Double(i) / Double(l)
            return shift * b / (1.0 + (shift - 1.0) * b)
        }
    }

    /// 公式曲线的一段：b 从 1 等距走到 b_end（由 endSigma 经 shiftToBase 反解），
    /// 因此序列每一项都在公式曲线上，且末项精确等于 endSigma（无截断取整漂移）。
    public static func formulaLadder(intervals: Int, endSigma: Double, shift: Double) -> [Double] {
        let l = max(1, intervals)
        let bEnd = endSigma / (shift + endSigma * (1.0 - shift))   // shiftToBase(endSigma, shift)
        return (0...l).map { i in
            let b = 1.0 - (1.0 - bEnd) * Double(i) / Double(l)
            return shift * b / (1.0 + (shift - 1.0) * b)
        }
    }

    /// 在 grid 的 range 档位里找与 target 最近的 σ 下标（等距时取下标小的 = 较大 σ，给低清留步）。
    public static func nearestGridIndex(to target: Double, grid: [Double], range: ClosedRange<Int>) -> Int {
        var best = range.lowerBound
        var bestD = Double.infinity
        for i in range where i < grid.count {
            let d = abs(grid[i] - target)
            if d < bestD - 1e-12 { bestD = d; best = i }
        }
        return best
    }

    public static func nearestGridValue(to target: Double, grid: [Double], range: ClosedRange<Int>) -> Double {
        grid[nearestGridIndex(to: target, grid: grid, range: range)]
    }

}

// MARK: - 临时张量强引用池（GPU 在飞期间保护小 buffer）

/// SelfLift 采样/过渡块里临时标量、索引、权重张量的 **Swift 侧强引用池**。
///
/// 为什么需要它（与 `H3VAEEncoder.encodeTiled` 里「先物化再出作用域」是同一类坑）：
/// MLX 的 Metal 分配器在 `free` 时把 buffer 放进按大小索引的空闲池，一旦池超过
/// `MLX.Memory.cacheLimit`（本项目 H3 分支为 10GB），或 active 逼近 `MLX.Memory.memoryLimit`
/// （LTX 侧设置的物理内存 70% 硬上限），allocator 会在 **没有任何 GPU 同步** 的情况下
/// 真正 `release()` 池内 MTLBuffer；而 MLX 的 Metal 命令缓冲是异步提交的，于是只要那个
/// buffer 还被在飞命令缓冲引用（例如作为 kernel 参数被 `setBuffer` 绑定过），就会触发
/// `notifyExternalReferencesNonZeroOnDealloc` 断言 —— 本次崩溃被销毁的
/// `AGXG16XFamilyBuffer ... length = 4 / MTLStorageModeShared / MTLHazardTrackingModeUntracked`
/// 正是过渡块里最小的标量（1×float32 = 4 字节）buffer。
///
/// 只要 Swift 侧仍持有 `MLXArray`，其 component buffer 的引用计数就 > 0，永远不会进入
/// 空闲池，也就不会被 allocator trim 掉。因此把过渡块里的小标量/索引/权重张量挂在池子上，
/// 等覆盖它们的 `MLX.eval` 返回（GPU 已同步）之后再 `release()`，即可保证
/// 「GPU 在飞期间这些小 buffer 一定有确定的宿主引用」。
///
/// 使用纪律：
///   1. `hold` 只延长生命周期，不改变任何计算语义；
///   2. `release()` **只允许**在一次覆盖了这些张量消费者的 `MLX.eval` 之后调用
///      （否则调用方自己就制造了悬垂）。
/// 非线程安全：只在一次采样/过渡块的调用栈内使用。
public final class SelfLiftTensorKeepAlive {
    private var items: [MLXArray] = []

    public init() {}

    /// 挂住一张张量（nil 忽略，便于直接挂可选值）
    @inline(__always)
    public func hold(_ a: MLXArray?) {
        if let a { items.append(a) }
    }

    /// 挂住一批张量
    @inline(__always)
    public func hold(_ arrays: [MLXArray]) {
        items.append(contentsOf: arrays)
    }

    public var count: Int { items.count }

    /// 释放全部强引用。仅可在 GPU 已同步（`MLX.eval` 已返回）之后调用。
    public func release() {
        items.removeAll(keepingCapacity: false)
    }
}

// MARK: - 过渡块数学（与模型无关）

public enum SelfLiftCore {

    @inline(__always)
    private static func scalarLike(_ v: Float, _ ref: MLXArray) -> MLXArray {
        ref.dtype == .float32 ? MLXArray(v) : MLXArray(v).asType(ref.dtype)
    }

    /// Eq.3：切换点的 clean endpoint 预测。`velocity` 为模型输出的速度项（dx/dσ）。
    ///
    /// - Parameter keepAlive: 可选强引用池。本函数造出的 σ 标量是 4 字节小张量、且是懒图的输入，
    ///   挂进池子可保证它被过渡块（VAE 往返）消费前不会被分配器空闲池 trim 掉。
    public static func cleanEndpoint(x: MLXArray, velocity: MLXArray, sigma: Double,
                                     keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
        let s = scalarLike(Float(sigma), velocity)
        let term = velocity * s
        let out = x - term
        keepAlive?.hold([s, term, out])
        return out
    }

    /// Eq.6-9：伪影感知一致性修正。
    ///
    /// 逐行照抄官方 `selflift.py` 的 `artifact_aware_consistency_lift`（行 102-136，mask = None 分支）：
    ///   · rho ≤ 0 或 w_max ≤ 0        → 返回直接提升 z_lat（官方行 109-110）
    ///   · rho = w_min = w_max = 1      → 返回像素锚点 z_pix（官方行 111-112）
    ///   · thr = torch.quantile(s.flatten(1), 1-ρ, dim=1)（线性插值分位，官方行 127）
    ///   · selected = s >= thr；一个都没选中 → 返回 z_lat（官方行 128-130）
    ///   · s_min / s_max 只在选中集合上取（官方行 131-132）；w 线性映射后非选中处置 0（官方行 133-134）
    ///   · 返回 z_lat + w · δ（官方行 135）
    /// 官方还有 mask（noise-mask 生成区）分支，本工程 H3 一采的过渡块不传 mask，故未移植。
    ///
    /// - Parameters:
    ///   - zLat / zPix: 同为「目标分辨率」域的 latent，形状一致（推荐 NCDHW）
    ///   - rho: 修正比例（0…1）
    ///   - channelAxis: 通道轴（NCDHW = 1；NHWC = 4）
    /// - Returns: `z_lat + w · δ`（w 仅在选中位置上非零）
    public static func artifactAwareConsistencyLift(zLat: MLXArray,
                                                    zPix: MLXArray,
                                                    rho: Double,
                                                    wMin: Float,
                                                    wMax: Float,
                                                    channelAxis: Int = 1,
                                                    keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
        // 官方：rho ≤ 0 或 w_max ≤ 0 → 全部位置都用直接提升
        if rho <= 0 || wMax <= 0 { return zLat }
        // 官方：纯锚点（rho = w_min = w_max = 1）→ 直接返回像素锚点
        if rho >= 1 && wMin >= 1 && wMax >= 1 { return zPix }

        let delta = zPix - zLat
        // s = mean_c |δ|，逐位置不一致度（官方 `delta.abs().mean(dim=1)`，不带 keepdim）
        let s = delta.abs().mean(axis: channelAxis)
        keepAlive?.hold([delta, s])
        let m = s.size
        // mlx-swift 的升序排序为自由函数 `sorted(_:axis:)`（MLXArray 无同名成员方法）
        let flat = sorted(s.reshaped([m]), axis: 0)
        keepAlive?.hold(flat)
        // 官方 `torch.quantile(flat, 1-ρ, dim=1)`：线性插值分位（本项目 H3 一采 batch = 1，等价于逐样本）
        let q = min(max(1.0 - rho, 0.0), 1.0)
        let pos = q * Double(m - 1)
        let lo = min(max(Int(pos.rounded(.down)), 0), m - 1)
        let hi = min(lo + 1, m - 1)
        let vLo: Float = flat[lo].item()
        let vHi: Float = flat[hi].item()
        let thr = vLo + (vHi - vLo) * Float(pos - Double(lo))
        // 官方 `selected = s >= thr`；一个都没选中 → 直接提升
        let sThr = scalarLike(thr, s)
        let selected = s .>= sThr
        keepAlive?.hold([sThr, selected])
        let selectedCount: Float = selected.asType(.float32).sum().item()
        let k = Int(selectedCount)
        if k <= 0 { return zLat }
        // 官方 s_min / s_max 只在选中集合上取：即升序第 k 大（= 选中集合最小值）与全局最大值
        let sMin: Float = flat[m - k].item()
        let sMax: Float = flat[m - 1].item()
        let sMinArr = scalarLike(sMin, s)
        let denom = scalarLike((sMax - sMin) + 1e-8, s)
        let wMinArr = scalarLike(wMin, s)
        let wMaxArr = scalarLike(wMax, s)
        keepAlive?.hold([sMinArr, denom, wMinArr, wMaxArr])
        let wScaled = wMinArr
            + (wMaxArr - wMinArr) * ((s - sMinArr) / denom)
        // 官方 `where(selected, w, 0)` 再 unsqueeze(1)：未选中位置权重为 0，并补回通道轴
        let wSelected = (wScaled * selected.asType(s.dtype)).expandedDimensions(axis: channelAxis)
        let out = zLat + wSelected * delta
        keepAlive?.hold([wScaled, wSelected, out])
        return out
    }

    /// Eq.10：按 σ 重加噪。
    /// 官方公式（ComfyUI 官方 `comfy/model_sampling.py` 的 `CONST.noise_scaling`）：
    ///   `x = σ · ε + (1 - σ) · x0`
    public static func noiseScaling(sigma: Double, noise: MLXArray, x0: MLXArray,
                                    keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
        let s = scalarLike(Float(sigma), x0)
        let oneMinusS = scalarLike(1, x0) - s
        let out = x0 * oneMinusS + noise * s
        keepAlive?.hold([s, oneMinusS, out])
        return out
    }

}

// MARK: - 过渡块编排（泛型，模型侧动作全部闭包注入）

public enum SelfLiftTransition {

    /// 执行一次 SelfLift 过渡块。返回语义由 `cfg.cleanEndpointHandoff` 决定：
    ///   · false（默认，带噪直通）：返回「σ_k 处的带噪 x 被升频到目标分辨率」的状态，**不重加噪、不补 Euler**；
    ///     高清接手的噪声深度就是 σ_k，与低清出口等深连续。
    ///   · true（旧路径，仅关闭不移除）：返回「clean x0 升频后按 σ_next 重加噪」的状态。
    ///
    /// 调用方职责：
    ///   1. 低分第 ts 次前向之后给出交接状态 `lowState`：
    ///      · 带噪直通模式 → 正常 Euler 更新后的 videoX（σ_k 处的带噪 x）；
    ///      · clean 模式 → Eq.3 产出的 `lowCleanX0`，并**丢弃**该步本应执行的 Euler 更新；
    ///   2. 提供 `noise`（目标分辨率、与返回状态同形状的标准正态；带噪直通模式下不会被执行）；
    ///   3. 用返回状态替换当前采样状态，然后按 `split.highSigmas` 继续跑完剩余步。
    ///
    /// - Parameter holdIntermediate: 可选强引用回调。过渡块里每个中间状态（提升结果 / 像素锚点 /
    ///   Eq.6-9 修正结果 / 重加噪结果 / 最终返回状态）都会回调一次，供调用方把它们挂进强引用池，
    ///   直到覆盖整块的 `MLX.eval` 之后释放 —— 避免这些懒图（及其输入小 buffer）在 GPU 在飞期间
    ///   被分配器空闲池回收。这里用回调而非直接持有 `SelfLiftTensorKeepAlive`，是因为 `State` 为泛型。
    public static func run<State>(
        split: SelfLiftScheduleSplit,
        cfg: SelfLiftConfig,
        lowState: State,
        noise: State,
        liftLatent: (State) -> State,
        liftPixelAnchor: ((State) -> State)?,
        consistencyLift: (State, State) -> State,
        noiseScaling: (State, Double, State) -> State,
        holdIntermediate: ((State) -> Void)? = nil
    ) -> State {
        let zLat = cfg.needsLatentLift ? liftLatent(lowState) : lowState
        holdIntermediate?(zLat)
        // ★ 2026-10-19（用户需求）默认走带噪直通：交接状态本身就是 σ_k 的含噪 x，升频后原样交给高清，
        //   不重加噪（否则会把升频后的噪声实现再乘一遍新的 ε，高清看到的噪声深度与低清出口不对等）、
        //   不补 Euler。高清需要噪点状态接着处理，而不是 clean 内容。
        //   clean 路径（含像素锚点 Eq.6-9 修正 + Eq.10 重加噪）整体保留在下面，仅由开关关闭。
        // ★ 2026-10-19 用户设计（低清跑满 + 浅重加噪）：判据从「只看开关」改为「看高清起点是否等于
        //   低清出口」——σ_next ≠ σ_k（如 0 → 0.3）必须重加噪；两者相等（等深连续）且开关关闭时
        //   才是原样的带噪直通。
        let needsRenoise = cfg.cleanEndpointHandoff || abs(split.sigmaNext - split.sigmaK) > 1e-9
        if !needsRenoise {
            return zLat
        }
        var z0High = zLat
        if cfg.needsPixelAnchor, let anchor = liftPixelAnchor {
            let zPix = anchor(lowState)
            holdIntermediate?(zPix)
            z0High = consistencyLift(zLat, zPix)
        }
        holdIntermediate?(z0High)
        // 用户设计（2026-10-09 收敛为等深）：σ_next == σ_k，按 σ_next 重加噪后直接进高清首步，
        // 不补 Euler。旧版官方档补 Euler 条件恒 false（漏补）、decouple 等深不补，两个成功分支
        // 均无补步；新版曾补 (σ_k → σ_next) 把状态推到更浅起点，试跑仍失败，已移除。
        let renSigma = split.sigmaNext
        let xK = noiseScaling(z0High, renSigma, noise)
        holdIntermediate?(xK)
        let out = xK
        holdIntermediate?(out)
        return out
    }
}
