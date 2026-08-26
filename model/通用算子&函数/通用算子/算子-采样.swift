//
//  模型通用算子-采样.swift
//  无限画布 — 通用采样调度与去噪步进算子（纯数学，可跨模型复用）
//
//  ============================================================
//  作用：通用采样调度与去噪步进算子（纯数学，可跨模型复用）：
//    · dynamicShiftSchedule：DynamicShift 噪声调度（σ shift）—— 使用方：LTX-2.5（Sampler）
//    · eulerStep：Euler 去噪步进—— 使用方：LTX-2.5（Sampler）
//    · guiderCombine：CFG + rescale 引导合成—— 使用方：LTX-2.5（Sampler）
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

/// CFG + rescale 引导合成：pred = cond + (cfg-1)*(cond-neg)；
/// rescale 按全张量标准差恢复范数。蒸馏单阶段无引导（cfg=1.0）时直接返回 cond。
func guiderCombine(
    cond: MLXArray,
    neg: MLXArray?,
    cfg: Float,
    rescale: Float
) -> MLXArray {
    var pred = cond
    if let neg {
        pred = pred + (cfg - 1.0) * (cond - neg)
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
