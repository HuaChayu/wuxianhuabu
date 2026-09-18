//
//  模型通用算子-采样.swift
//  无限画布 — 通用采样调度与去噪步进算子（纯数学，可跨模型复用）
//
//  ============================================================
//  作用：通用采样调度与去噪步进算子（纯数学，可跨模型复用）：
//    · dynamicShiftSchedule：DynamicShift 噪声调度（σ shift）—— 使用方：LTX-2.5（Sampler）
//    · eulerStep：Euler 去噪步进—— 使用方：LTX-2.5（Sampler）
//    · guiderCombineMulti：CFG + rescale 引导合成（原 guiderCombine 已并入，走 ptb/stg/mod 默认参数）—— 使用方：LTX-2.5（Sampler）
//    · timestepEmbedding：sinusoidal timestep 嵌入—— 使用方：HiDream-O1-Image（HiDreamTimestepEmbedder）
//    · FlashFlowMatchScheduler：flow matching Euler 调度器—— 使用方：HiDream-O1-Image（hidreamGenerate）
//    · closestResolution：分辨率吸附到最近合法宽高比—— 使用方：HiDream-O1-Image（hidreamGenerate）
//  ============================================================

import Foundation
import MLX
import MLXNN

/// DynamicShift 噪声调度：按 token 数自适应 shift，返回 numSteps+1 个 σ。
func dynamicShiftSchedule(
    numSteps: Int,
    numTokens: Int,
    baseShift: Double = 0.95,
    maxShift: Double = 2.05,
    baseTokens: Double = 1024,
    maxTokens: Double = 4096,
    stretch: Bool = true,
    terminal: Double = 0.1
) -> [Float] {
    let n = numSteps + 1
    let slope = (maxShift - baseShift) / (maxTokens - baseTokens)
    let intercept = baseShift - slope * baseTokens
    let sigmaShift = Double(numTokens) * slope + intercept
    let es = exp(sigmaShift)
    var sig = [Double](repeating: 0, count: n)
    for i in 0..<n {
        let lin = 1.0 - Double(i) / Double(numSteps)   // linspace(1,0)
        if lin == 0.0 {
            sig[i] = 0.0
        } else {
            sig[i] = es / (es + (1.0 / lin - 1.0))
        }
    }
    if stretch {
        let lastNz = sig[numSteps - 1]
        let scaleFactor = (1.0 - lastNz) / (1.0 - terminal)
        if scaleFactor != 0.0 {
            for i in 0..<n where sig[i] != 0.0 {
                sig[i] = 1.0 - (1.0 - sig[i]) / scaleFactor
            }
        }
    }
    return sig.map { Float($0) }
}

// MARK: - LTX-2.5 蒸馏固定 sigma 调度（蒸馏模型专用）

/// LTX-2.5 蒸馏模型官方固定 sigma 表（stage1，8 步）。
/// 蒸馏模型每步噪声档位训练时钉死，严禁用 dynamicShiftSchedule 等连续调度替代。
/// 使用方：LTX-2.5（Sampler.swift stage1）
let ltx25DistilledSigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

/// LTX-2.5 蒸馏 stage2（升频 refine，3 步）。
/// 使用方：LTX-2.5（视频模型调用&回执.swift vaeDecodeTest）
let ltx25Stage2Sigmas: [Float] = [0.909375, 0.725, 0.421875, 0.0]

/// LTX-2.5 IC 二采保守档（2026-09-18，H3 低清画面锚定增强）：
/// σ0=0.85 起步（主序列 init = 升频 latent*(1-0.85)+noise*0.85，保留 15% 原画面），
/// 不再用官方 0.909375 高噪档（该档对模糊半清/空文本输入会让模型先验主导、参考 KV
/// 锁不住人脸 → 换脸）。尾部 0.421875→0 保留官方高清细节步几何，仍 3 步确定性 Euler。
/// 使用方：runLTXStage2RefineOnLatent（IC 通道，decoupledRefLatent == nil 时默认档）。
let ltx25Stage2SigmasConserve: [Float] = [0.85, 0.65, 0.421875, 0.0]

/// SelfLift 解耦（LTX IC 二采版，2026-09-17）：低清段 / 高清段两段式 σ 曲线，总 NFE = 2+2 = 4。
/// 语义与 H3 一采 SelfLift 解耦同源：低清段先在半清网格上浅跑 2 步（σ0=0.909375 保留升频锚点，
/// 末点 σ_k=0.421875 残留），锁定时间-光照一致的结构 → 升频 ×2 后重加噪到 σ_next=0.421875 →
/// 高清段 IC 官方骨架跑 2 步收细节。官方 IC 3 步全清；本档低清 2 步跑 1/4 网格
/// （算力 ≈ 2.5 全清步，更省），且低清段先锁时序能缓解高分辨率短步数下的帧间闪烁。
/// 使用方：runLTXStage2RefineOnLatent（IC 通道，NA_PIX_SELFLIFT_DECOUPLE=1 时启用，默认开）。
let ltx25Stage2SigmasDecoupleLow: [Float] = [0.909375, 0.725, 0.421875]
let ltx25Stage2SigmasDecoupleHigh: [Float] = [0.421875, 0.2109375, 0.0]

/// 【新增】官方 CQ Video Enhancer LoRA 工作流 ManualSigmas（9 段 / 8 步，σ0 = 1.0）。
/// 原文："1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
/// 使用方：第二阶段·CQ 清晰度增强通道（ltx2.5-(ltx专属).swift runLTXStage2RefineOnLatent 的
/// cqRequested 分支）。该档 σ0=1.0，主序列从纯噪声起采，依赖 CQ 工作流的 in-context 参考
/// （低清视频整段 latent 作 KV）+ CQ LoRA 重建细节；官方采样器为 euler_ancestral。
/// 与 lt25Stage2Sigmas（IC 二采 4 步 σ0=0.909375）并存、互不影响。
let cqEnhancerSigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

/// Ancestral (SDE) Euler 步进：对齐官方 EulerAncestralDiffusionStep（eta=1.0, s_noise=1.0）。
/// 先确定性插值到中间噪声档 sigma_down，再注入噪声回升到 sigma_next。
/// LTX-2.5 蒸馏 stage1 官方默认 ancestral 采样，确定性 Euler 会丢细节。
/// 使用方：LTX-2.5（Sampler.swift stage1）
func ancestralEulerStep(_ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float, noise: MLXArray) -> MLXArray {
    if sigmaNext == 0.0 { return x0 }
    let eta: Float = 1.0
    let sNoise: Float = 1.0
    let downstepRatio = 1.0 + (sigmaNext / sigma - 1.0) * eta
    let sigmaDown = sigmaNext * downstepRatio
    let sigmaDownRatio = sigmaDown / sigma
    var xNext = sigmaDownRatio * x + (1.0 - sigmaDownRatio) * x0
    let alphaNext = 1.0 - sigmaNext
    let alphaDown = 1.0 - sigmaDown
    let renoiseCoeff = sqrt(max(sigmaNext * sigmaNext - sigmaDown * sigmaDown * alphaNext * alphaNext / (alphaDown * alphaDown), 0.0))
    xNext = (alphaNext / alphaDown) * xNext + noise * sNoise * renoiseCoeff
    return xNext
}

/// Euler 去噪步进：x_next = x + (x - x0)/σ * (σ_next - σ)。
func eulerStep(_ x: MLXArray, _ x0: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
    if sigma == 0.0 { return x0 }
    let d = (x - x0) / sigma
    return x + d * (sigmaNext - sigma)
}

/// CFG + rescale 引导合成（由 guiderCombine / guiderCombineMulti 合并而来）。
/// 合并说明：原 guiderCombine 是 guiderCombineMulti 在 ptb=nil、stgScale=0、
/// mod=nil、modalityScale=1 时的特例（STG/Modality 项零贡献），
/// 合并后统一走 guiderCombineMulti；原 guiderCombine 调用方已改为
/// guiderCombineMulti(cond:neg:cfg:ptb:nil:stgScale:0:mod:nil:modalityScale:1:rescale:)。
func guiderCombineMulti(
    cond: MLXArray,
    neg: MLXArray?,
    cfg: Float,
    ptb: MLXArray?,
    stgScale: Float,
    mod: MLXArray?,
    modalityScale: Float,
    rescale: Float
) -> MLXArray {
    var pred = cond
    // CFG 项
    if let neg, abs(cfg - 1.0) > 1e-4 {
        pred = pred + (cfg - 1.0) * (cond - neg)
    }
    // STG 项：stg_scale*(cond - ptb)
    if let ptb, abs(stgScale) > 1e-4 {
        pred = pred + stgScale * (cond - ptb)
    }
    // Modality 项：(modality_scale-1)*(cond - mod)
    if let mod, abs(modalityScale - 1.0) > 1e-4 {
        pred = pred + (modalityScale - 1.0) * (cond - mod)
    }
    if abs(rescale) > 1e-4 {
        let condStd = sqrt(cond.variance() + 1e-8)
        let predStd = sqrt(pred.variance() + 1e-8)
        let factor = condStd / predStd * rescale + (1.0 - rescale)
        pred = pred * factor
    }
    return pred
}

// MARK: - Timestep 嵌入（扩散通用）

/// Sinusoidal timestep embedding（cos/sin 频率表，与 DDPM/DiT 一致）。
/// 输入 t 已归一化到 0~1，使用方内部自行 ×1000；dim 需为偶数（奇数自动补零）。
/// 使用方：HiDream-O1-Image（HiDreamTimestepEmbedder）。
/// Sinusoidal timestep embedding 的 freq 表静态缓存（key = "half_maxPeriod"）。
/// freqs 只依赖 dim/maxPeriod，28 步生成里 timestepEmbedding 会被反复调用，避免每步重算 256 次 exp/log。
private let timestepFreqLock = NSLock()
private var timestepFreqCache: [String: [Float]] = [:]

func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000.0) -> MLXArray {
    let half = dim / 2
    let halfF = Float(half)
    let cacheKey = "\(half)_\(maxPeriod)"
    var freqs: [Float]
    timestepFreqLock.lock()
    if let cached = timestepFreqCache[cacheKey] {
        freqs = cached
    } else {
        var computed = [Float](repeating: 0, count: half)
        for i in 0 ..< half {
            computed[i] = exp(-log(maxPeriod) * Float(i) / halfF)
        }
        timestepFreqCache[cacheKey] = computed
        freqs = computed
    }
    timestepFreqLock.unlock()
    let freqsArr = MLXArray(freqs)
    let tF = t.asType(.float32).reshaped([-1, 1])
    let args = tF * freqsArr[.newAxis, 0...]
    var emb = concatenated([cos(args), sin(args)], axis: -1)
    if dim % 2 == 1 {
        let zeroPad = MLXArray.zeros([t.dim(0), 1], dtype: .float32)
        emb = concatenated([emb, zeroPad], axis: -1)
    }
    return emb
}

// MARK: - Flow Matching 调度器（扩散通用）

/// FlashFlowMatchScheduler：flow matching Euler 调度器，带可选噪声注入。
/// 对应 diffusers FlowMatchEulerDiscreteScheduler：timestep → sigma 归一化，
/// step 按 sigma/sigma_next 做 Euler 更新；支持自定义 timestep 表与 noise clip。
/// 使用方：HiDream-O1-Image（hidreamGenerate 28 步，defaultTimesteps 999→8）。
final class FlashFlowMatchScheduler {

    let numTrainTimesteps: Int
    let shift: Float
    var timestepsNP: [Float] = []
    var sigmasNP: [Float] = []
    var numInferenceSteps: Int = 0
    var stepIndex: Int?

    init(numTrainTimesteps: Int = 1000, shift: Float = 1.0) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }

    func setTimesteps(numInferenceSteps: Int, customTimesteps: [Float]? = nil) {
        var timesteps: [Float]
        var sigmas: [Float]
        if let customTimesteps {
            timesteps = customTimesteps
            sigmas = customTimesteps.map { $0 / Float(numTrainTimesteps) }
        } else {
            timesteps = Array(stride(from: Float(numTrainTimesteps), through: 1.0, by: -Float(numTrainTimesteps - 1) / Float(max(numInferenceSteps - 1, 1))))
            if timesteps.count > numInferenceSteps { timesteps = Array(timesteps.prefix(numInferenceSteps)) }
            var s = timesteps.map { $0 / Float(numTrainTimesteps) }
            s = s.map { shift * $0 / (1.0 + (shift - 1.0) * $0) }
            sigmas = s
        }
        sigmas.append(0.0)
        self.numInferenceSteps = timesteps.count
        self.timestepsNP = timesteps
        self.sigmasNP = sigmas
        self.stepIndex = nil
    }

    var timesteps: MLXArray { MLXArray(timestepsNP) }
    var sigmas: MLXArray { MLXArray(sigmasNP) }

    private func initStepIndex(timestepValue: Float) {
        guard let idx = timestepsNP.firstIndex(where: { abs($0 - timestepValue) < 1e-3 }) else {
            fatalError("timestep \(timestepValue) not in scheduler.timesteps")
        }
        stepIndex = idx
    }

    func step(
        modelOutput: MLXArray,
        timestep: Float,
        sample: MLXArray,
        sNoise: Float = 1.0,
        noiseClipStd: Float = 0.0,
        seed: UInt64? = nil
    ) -> MLXArray {
        if stepIndex == nil { initStepIndex(timestepValue: timestep) }
        guard let idx = stepIndex else { fatalError("stepIndex nil") }

        let sigma = sigmasNP[idx]
        let sigmaNext = sigmasNP[idx + 1]

        let sampleF = sample.asType(.float32)
        let modelOutputF = modelOutput.asType(.float32)

        let denoised = sampleF - modelOutputF * sigma

        var newSample: MLXArray
        if idx < numInferenceSteps {
            var noise: MLXArray
            if let seed {
                let key = MLXRandom.key(UInt64(seed) + UInt64(idx))
                noise = MLXRandom.normal(modelOutputF.shape, key: key)
            } else {
                noise = MLXRandom.normal(modelOutputF.shape)
            }

            if noiseClipStd > 0 {
                let stdVal = Float(std(noise.asType(.float32)).item(Float.self))
                let clipVal = noiseClipStd * stdVal
                noise = clip(noise, min: -clipVal, max: clipVal)
            }

            newSample = sigmaNext * noise * sNoise + (1.0 - sigmaNext) * denoised
        } else {
            newSample = denoised
        }

        stepIndex = idx + 1
        return newSample.asType(sample.dtype)
    }
}

// MARK: - 分辨率对齐（扩散通用）

/// 分辨率对齐：宽高已是 divisible 倍数（含 32 倍数动态表 / 参考图尺寸）时直接放行，
/// 否则吸附到训练表中最接近的宽高比（宽高比匹配优先）。
/// predefined 表按模型训练分辨率传入。
/// 使用方：HiDream-O1-Image（hidreamGenerate，表 = HiDreamDiffusionConstants.predefinedResolutions）。
/// 注意：T2I 与编辑共用本函数。T2I 已实测 32 倍数 off-spec 出图正常（动态表可直接放行）；
/// 编辑 off-spec 曾出纹理，但用户要求先沿用参考图尺寸手动验证，故统一恢复“32 倍数即放行”，
/// 吸附表保留作非 32 倍数兜底。
func closestResolution(
    height: Int, width: Int,
    predefined: [(w: Int, h: Int)],
    divisible: Int = 32
) -> (h: Int, w: Int) {
    if height % divisible == 0 && width % divisible == 0 {
        return (height, width)
    }
    let imgRatio = Float(width) / Float(height)
    var best = predefined[0]
    var bestDiff = Float.greatestFiniteMagnitude
    for r in predefined {
        let diff = abs(Float(r.w) / Float(r.h) - imgRatio)
        if diff < bestDiff {
            bestDiff = diff
            best = r
        }
    }
    return (best.h, best.w)
}

// MARK: - Sinusoidal 时间步嵌入（LTX 兼容别名）

/// LTX-2.5 兼容别名：sinusoidal timestep 嵌入（max_period=10000, scale=1），输出 [cos(args), sin(args)]。
/// 与 timestepEmbedding 数学等价（freqs = maxPeriod^(-i/half)），仅输入值域约定不同
///（本函数保持原调用点语义，不强制 t 归一化到 0~1）。
/// 使用方：LTX-2.5（LTXDiT LTXTimestepEmbedder）。
func sinusoidalEmbedding(_ t: MLXArray, dim: Int) -> MLXArray {
    timestepEmbedding(t, dim: dim, maxPeriod: 10000.0)
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

// ============================================================================
// MARK: - Beta(0.6,0.6) 噪声分布 + 余弦尾段精修 公共调度（纯 Foundation，可复用）
// ============================================================================
import Foundation

// MARK: - ln(Gamma) via Lanczos (g=7, n=9)

func lnGamma(_ x: Double) -> Double {
    let cof: [Double] = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ]
    if x < 0.5 {
        return log(.pi / sin(.pi * x)) - lnGamma(1.0 - x)
    }
    let z = x - 1.0
    var a = cof[0]
    let t = z + 7.0 + 0.5
    for i in 1..<9 { a += cof[i] / (z + Double(i)) }
    return 0.5 * log(2 * .pi) + (z + 0.5) * log(t) - t + log(a)
}

// MARK: - Regularized incomplete beta I_x(a,b)

private func betaContinuedFraction(_ a: Double, _ b: Double, _ x: Double) -> Double {
    let maxIter = 300
    let eps = 3e-13
    let fpmin = 1e-300
    let qab = a + b
    let qap = a + 1.0
    let qam = a - 1.0
    var c = 1.0
    var d = 1.0 - qab * x / qap
    if abs(d) < fpmin { d = fpmin }
    d = 1.0 / d
    var h = d
    for m in 1...maxIter {
        let m2 = 2 * m
        var aa = Double(m) * (b - Double(m)) * x / ((qam + Double(m2)) * (a + Double(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin { d = fpmin }
        c = 1.0 + aa / c
        if abs(c) < fpmin { c = fpmin }
        d = 1.0 / d
        h *= d * c
        aa = -(a + Double(m)) * (qab + Double(m)) * x / ((a + Double(m2)) * (qap + Double(m2)))
        d = 1.0 + aa * d
        if abs(d) < fpmin { d = fpmin }
        c = 1.0 + aa / c
        if abs(c) < fpmin { c = fpmin }
        d = 1.0 / d
        let del = d * c
        h *= del
        if abs(del - 1.0) < eps { break }
    }
    return h
}

func regularizedIncompleteBeta(_ x: Double, _ a: Double, _ b: Double) -> Double {
    if x <= 0 { return 0 }
    if x >= 1 { return 1 }
    let lnbt = lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * log(x) + b * log(1 - x)
    let bt = exp(lnbt)
    if x < (a + 1) / (a + b + 2) {
        return bt * betaContinuedFraction(a, b, x) / a
    } else {
        return 1.0 - bt * betaContinuedFraction(b, a, 1 - x) / b
    }
}

// MARK: - Beta quantile (ppf)

func betaQuantile(_ q: Double, alpha: Double = 0.6, beta: Double = 0.6) -> Double {
    if q <= 0 { return 0 }
    if q >= 1 { return 1 }
    var lo = 0.0, hi = 1.0
    for _ in 0..<80 {
        let mid = (lo + hi) / 2
        if regularizedIncompleteBeta(mid, alpha, beta) < q {
            lo = mid
        } else {
            hi = mid
        }
    }
    return (lo + hi) / 2
}

// MARK: - Schedule builder (continuous, no 1000-step table lookup)

/// Shifted sigma from base grid point b: sigma = shift*b/(1+(shift-1)*b).
func shiftedSigma(_ base: Double, shift: Double) -> Double {
    shift * base / (1.0 + (shift - 1.0) * base)
}

/// Beta-distributed noise points (count descending nonzero values, NO terminal 0).
/// Mirrors ComfyUI BasicScheduler(beta): quantiles q_i = 1 - i/count sampled at
/// symmetric Beta(alpha,beta) CDF. alpha=beta=0.6 → U-shaped density.
func betaNoisePoints(count: Int, shift: Double, alpha: Double = 0.6, beta: Double = 0.6) -> [Double] {
    guard count > 0 else { return [] }
    return (0..<count).map { i in
        let q = 1.0 - Double(i) / Double(count)
        let t = betaQuantile(q, alpha: alpha, beta: beta)
        return shiftedSigma(t, shift: shift)
    }
}

public enum SigmaSpacing {
    case cosine, linear, exponential
}

/// Tail refinement, ported from ComfyUI YCNodes H3SigmaRefiner.
/// - Keeps head before first sigma <= startAtSigma untouched.
/// - Rebuilds the whole tail A->end with (oldTailLen + extraSteps) points
///   distributed by `spacing`; appends terminal 0.0 when the input ended at 0.
/// - Nonzero count increases by exactly `extraSteps` whenever it acts.
func refineSigmaTail(
    _ sigmas: [Double],
    extraSteps: Int = 1,
    startAtSigma: Double = 0.7,
    endAtSigma: Double = 0.0,
    spacing: SigmaSpacing = .cosine
) -> [Double] {
    guard extraSteps > 0, !sigmas.isEmpty else { return sigmas }
    guard let idx = sigmas.firstIndex(where: { $0 <= startAtSigma }),
          idx < sigmas.count - 1 else { return sigmas }

    let head = Array(sigmas[..<idx])
    let a = sigmas[idx]
    let b = max(endAtSigma, sigmas[sigmas.count - 1])
    let oldTailLen = sigmas.count - idx
    let newTailLen = oldTailLen + extraSteps

    var tail = [Double](repeating: 0, count: newTailLen)
    for j in 0..<newTailLen {
        let t = Double(j) / Double(newTailLen - 1)
        let f: Double
        switch spacing {
        case .cosine: f = (1.0 - cos(t * .pi)) / 2.0
        case .linear: f = t
        case .exponential: f = (exp(t * 3.0) - 1.0) / (exp(3.0) - 1.0)
        }
        tail[j] = a + (b - a) * f
    }
    var out = head + tail
    if sigmas[sigmas.count - 1] == 0.0 && b > 0.0 {
        out.append(0.0)
    }
    return out
}

// MARK: - 步数守恒公共入口

/// Beta + 余弦尾段精修调度（步数守恒公共入口）
///
/// 语义：调用方给"最终采样步数" totalSteps；函数内部先按
/// `totalSteps - extraSteps` 生成 Beta 分布非零档，再对低 σ 尾段做
/// cosine/linear/exponential 重排补回 extraSteps 档 —— 返回恒为
/// `totalSteps + 1` 个 σ（含末位 0），采样轮次恰为 totalSteps，末段不再大步跳 0。
///
/// 阈值规则（移植 ComfyUI YCNodes H3SigmaRefiner）：
/// 1. 找到首个 ≤ startAtSigma 的非零档 idx，idx 之前原样保留；
/// 2. 从该档到 endAtSigma 的整段尾巴按 spacing 曲线重排（档数 + extraSteps）；
/// 3. 若序列里没有任何非零档 ≤ startAtSigma（小步数时可能发生，如 4 步末档仍 > 0.7），
///    自动兜底：锁定最后一个非零档 A，把 A→end 按同一 spacing 拆成 extraSteps+1 段，
///    保证总步数守恒且必然生效。
///
/// 返回序列单调降：[1.0 ... >0, 0.0]
public func betaRefinedSchedule(
    totalSteps: Int,
    shift: Double = 12.0,
    alpha: Double = 0.6,
    beta: Double = 0.6,
    extraSteps: Int = 1,
    startAtSigma: Double = 0.7,
    endAtSigma: Double = 0.0,
    spacing: SigmaSpacing = .cosine
) -> [Double] {
    precondition(totalSteps > 0, "totalSteps 必须 > 0")
    let e = max(extraSteps, 0)
    let baseCount = max(totalSteps - e, 1)

    var sig = betaNoisePoints(count: baseCount, shift: shift, alpha: alpha, beta: beta)
    sig.append(0.0)

    var refined = refineSigmaTail(sig, extraSteps: e, startAtSigma: startAtSigma,
                                  endAtSigma: endAtSigma, spacing: spacing)

    // 兜底：阈值未命中（refiner noop，点数没增加）→ 锁最后一个非零档手动拆分
    // 新尾 = [A, ..., 0] 共 e+2 个点（非零 e+1 个 = A + e 个中间档），总非零档数守恒
    if refined.count == sig.count, e > 0, refined.count >= 2 {
        let lastNZIdx = refined.count - 2
        let a = refined[lastNZIdx]
        var out = Array(refined[..<lastNZIdx])
        for j in 0...(e + 1) {
            let t = Double(j) / Double(e + 1)
            let f: Double
            switch spacing {
            case .cosine: f = (1.0 - cos(t * .pi)) / 2.0
            case .linear: f = t
            case .exponential: f = (exp(t * 3.0) - 1.0) / (exp(3.0) - 1.0)
            }
            out.append(a + (endAtSigma - a) * f)
        }
        refined = out
    }

    // 长度保护（浮点/极端参数下仍保证 totalSteps+1）
    while refined.count < totalSteps + 1 { refined.append(0.0) }
    if refined.count > totalSteps + 1 {
        refined = Array(refined[..<(totalSteps + 1)])
    }
    return refined
}

