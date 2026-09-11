//
//  模型通用算子-RoPE.swift
//  无限画布 — 通用 RoPE 频率/应用算子（纯数学，可跨模型复用）
//
//  ============================================================
//  作用：通用 RoPE 频率/应用算子（纯数学，可跨模型复用）：
//    · RopeCS：一组 cos/sin 频率表（[1, numHeads, N, headDim/2]）
//    · ditRope：3D 位置 RoPE 频率生成（视频/音频通用，maxPos 参数化）—— 使用方：LTX-2.5（LTXDiT）
//    · connectorRopeFreqs：连接器 1D RoPE 频率（maxPos 4096）—— 使用方：LTX-2.5（Connector）
//    · applyRopeSplit：旋转一半维度（split-half rope）—— 使用方：LTX-2.5（Connector、LTXDiT）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - RoPE 频率表

/// 一组 cos/sin 频率表，形状 [1, numHeads, N, headDim/2]（已 transpose + contiguous）。
struct RopeCS {
    let cos: MLXArray
    let sin: MLXArray
}

/// 通用 3D 位置 RoPE：pos 为 N×A 位置（A=轴数），逐轴按 maxPos 归一化，
/// 生成 [1, numHeads, N, headDim/2] 的 cos/sin 表（SPLIT 风格）。
func ditRope(pos: [Float], N: Int, A: Int, numHeads: Int, headDim: Int, maxPos: [Float]) -> RopeCS {
    let theta: Float = 10000.0
    let innerDim = numHeads * headDim
    let numFreqs = innerDim / (2 * A)          // 整数除法
    let expected = innerDim / 2
    let pad = expected - numFreqs * A
    let hdHalf = headDim / 2

    var fi = [Float](repeating: 0, count: numFreqs)
    let denom = Float(numFreqs - 1)
    for j in 0..<numFreqs {
        fi[j] = pow(theta, Float(j) / denom) * (Float.pi / 2.0)
    }

    var cosBuf = [Float](repeating: 0, count: N * expected)
    var sinBuf = [Float](repeating: 0, count: N * expected)
    for n in 0..<N {
        let base = n * expected
        for c in 0..<pad {
            cosBuf[base + c] = 1.0
            sinBuf[base + c] = 0.0
        }
        for j in 0..<numFreqs {
            for i in 0..<A {
                let frac = pos[n * A + i] / maxPos[i]
                let sc = frac * 2.0 - 1.0
                let ang = fi[j] * sc
                let idx = base + pad + j * A + i
                cosBuf[idx] = cos(ang)
                sinBuf[idx] = sin(ang)
            }
        }
    }
    // [1, N, numHeads, hdHalf] → transpose(0,2,1,3) → [1, numHeads, N, hdHalf]
    let cos = floatArray(cosBuf).reshaped([1, N, numHeads, hdHalf]).transposed(0, 2, 1, 3)
    let sin = floatArray(sinBuf).reshaped([1, N, numHeads, hdHalf]).transposed(0, 2, 1, 3)
    eval(cos, sin)  // 物化为 contiguous，避免惰性 strided 视图
    return RopeCS(cos: cos, sin: sin)
}

/// ditRopeV2：与 ditRope 数学等价（同一公式），但把 N×numFreqs×A 次
/// CPU 三角函数（sin/cos）移到 GPU 向量化——CPU 只做乘加构造角度矩阵，
/// cos/sin 交给 MLX 内核。N（帧数）较大时 RoPE 表构造延迟显著下降。
/// 旧版 ditRope 保留不动。
func ditRopeV2(pos: [Float], N: Int, A: Int, numHeads: Int, headDim: Int, maxPos: [Float]) -> RopeCS {
    let theta: Float = 10000.0
    let innerDim = numHeads * headDim
    let numFreqs = innerDim / (2 * A)          // 整数除法
    let expected = innerDim / 2
    let pad = expected - numFreqs * A
    let hdHalf = headDim / 2

    var fi = [Float](repeating: 0, count: numFreqs)
    let denom = Float(numFreqs - 1)
    for j in 0..<numFreqs {
        fi[j] = pow(theta, Float(j) / denom) * (Float.pi / 2.0)
    }

    // CPU 只构造角度（乘加，无三角函数）
    var angBuf = [Float](repeating: 0, count: N * numFreqs * A)
    for n in 0..<N {
        let nb = n * numFreqs * A
        for j in 0..<numFreqs {
            let fj = fi[j]
            let jb = j * A
            for i in 0..<A {
                let frac = pos[n * A + i] / maxPos[i]
                angBuf[nb + jb + i] = fj * (frac * 2.0 - 1.0)
            }
        }
    }
    let ang = MLXArray(angBuf, [N, numFreqs * A])   // [N, numFreqs*A]
    let c = cos(ang)                                 // GPU 向量化
    let s = sin(ang)
    // pad 前缀（cos=1, sin=0）拼回 [N, expected] → [1,N,numHeads,hdHalf] → transpose
    let padC = MLXArray.ones([N, pad])
    let padS = MLXArray.zeros([N, pad])
    let cosAll = MLX.concatenated([padC, c], axis: -1).reshaped([1, N, numHeads, hdHalf])
    let sinAll = MLX.concatenated([padS, s], axis: -1).reshaped([1, N, numHeads, hdHalf])
    let cosF = cosAll.transposed(0, 2, 1, 3)
    let sinF = sinAll.transposed(0, 2, 1, 3)
    eval(cosF, sinF)  // 物化为 contiguous，与 ditRope 一致
    return RopeCS(cos: cosF, sin: sinF)
}

/// 连接器 1D RoPE 频率：位置 t 用 maxPos=4096 归一化（split 风格）。
func connectorRopeFreqs(T: Int, dim: Int, headDim: Int) -> (cos: MLXArray, sin: MLXArray) {
    let theta: Float = 10000.0
    let maxPos: Float = 4096.0
    let numFreqs = dim / 2
    let hdHalf = headDim / 2
    let heads = dim / headDim
    let denom = Float(numFreqs - 1)

    // fi[j] = theta^(j/(numFreqs-1)) * π/2
    var fi = [Float](repeating: 0, count: numFreqs)
    for j in 0..<numFreqs {
        let e = Float(j) / denom
        fi[j] = pow(theta, e) * (Float.pi / 2.0)
    }
    // freqs[t,j] = fi[j] * (t/maxPos*2 - 1)
    var freqs = [Float](repeating: 0, count: T * numFreqs)
    for t in 0..<T {
        let frac = Float(t) / maxPos
        let sc = frac * 2.0 - 1.0
        for j in 0..<numFreqs {
            freqs[t * numFreqs + j] = fi[j] * sc
        }
    }
    let freqArr = MLXArray(freqs, [1, T, numFreqs])
    let cosRaw = cos(freqArr)
    let sinRaw = sin(freqArr)
    let cosR = cosRaw.reshaped([1, T, heads, hdHalf])
    let sinR = sinRaw.reshaped([1, T, heads, hdHalf])
    return (cosR.transposed(0, 2, 1, 3), sinR.transposed(0, 2, 1, 3))
}

// connectorRopeFreqsV2 的 fi 频率表静态缓存（只依赖 dim/theta，全任务恒定）
private let connectorRopeFreqLock = NSLock()
private var connectorRopeFreqFiCache: [Int: [Float]] = [:]

/// connectorRopeFreqsV2：与 connectorRopeFreqs 数学等价，两点改良：
/// ① fi 频率表静态缓存，避免每次调用重算 numFreqs 次 pow；
/// ② freqs = outer(frac, fi) 一次 GPU/向量化构造，替代原 CPU 双循环逐元素乘加。
/// T 大（长序列）时构造延迟显著下降。旧版 connectorRopeFreqs 保留不动。
func connectorRopeFreqsV2(T: Int, dim: Int, headDim: Int) -> (cos: MLXArray, sin: MLXArray) {
    let theta: Float = 10000.0
    let maxPos: Float = 4096.0
    let numFreqs = dim / 2
    let hdHalf = headDim / 2
    let heads = dim / headDim
    let denom = Float(numFreqs - 1)

    let fi: [Float]
    connectorRopeFreqLock.lock()
    if let cached = connectorRopeFreqFiCache[numFreqs] {
        fi = cached
    } else {
        var computed = [Float](repeating: 0, count: numFreqs)
        for j in 0..<numFreqs {
            let e = Float(j) / denom
            computed[j] = pow(theta, e) * (Float.pi / 2.0)
        }
        connectorRopeFreqFiCache[numFreqs] = computed
        fi = computed
    }
    connectorRopeFreqLock.unlock()

    // frac[t] = t/maxPos*2 - 1；freqs = outer(frac, fi) → [T, numFreqs]（向量化）
    let frac = MLXArray((0..<T).map { Float($0) / maxPos * 2.0 - 1.0 })  // [T]
    let fiArr = MLXArray(fi)                                             // [numFreqs]
    let freqs = outer(frac, fiArr).expandedDimensions(axis: 0)           // [1, T, numFreqs]
    let cosRaw = cos(freqs)
    let sinRaw = sin(freqs)
    let cosR = cosRaw.reshaped([1, T, heads, hdHalf])
    let sinR = sinRaw.reshaped([1, T, heads, hdHalf])
    return (cosR.transposed(0, 2, 1, 3), sinR.transposed(0, 2, 1, 3))
}

/// 旋转一半维度（split-half rope）：x [..., headDim] 对半切，rot 应用 cos/sin。
func applyRopeSplit(
    _ x: MLXArray, cosF: MLXArray, sinF: MLXArray
) -> MLXArray {
    let hd = x.shape[3]
    let half = hd / 2
    let x1 = x[0..., 0..., 0..., 0..<half]
    let x2 = x[0..., 0..., 0..., half...]
    let lo = x1 * cosF - x2 * sinF
    let hi = x1 * sinF + x2 * cosF
    return concatenated([lo, hi], axis: 3)
}

// MARK: - RoPE 应用（向量旋转）

/// 旋转后半维并取负拼接（rotate_half 变体）。
/// 使用方：HiDream-O1-Image（HiDreamVision / HiDreamLanguage）。
func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let first = x[.ellipsis, 0 ..< half]
    let second = x[.ellipsis, half...]
    return concatenated([-second, first], axis: -1)
}

/// 预展开 rotary cos/sin（只依赖 freqs，视觉塔各层复用，避免每层重复 expand/tile）。
/// 使用方：HiDream-O1-Image（HiDreamVision）。
func precomputeRotary(freqs: MLXArray) -> (MLXArray, MLXArray) {
    var cosVals = cos(freqs)
    var sinVals = sin(freqs)

    cosVals = expandedDimensions(cosVals, axis: 1)
    cosVals = tiled(cosVals, repetitions: [1, 1, 2])
    cosVals = expandedDimensions(cosVals, axis: 0)

    sinVals = expandedDimensions(sinVals, axis: 1)
    sinVals = tiled(sinVals, repetitions: [1, 1, 2])
    sinVals = expandedDimensions(sinVals, axis: 0)

    return (cosVals, sinVals)
}

/// 标准旋转应用：x' = x·cos + rotateHalf(x)·sin。
/// 使用方：HiDream-O1-Image（HiDreamVision）。
func applyRotary(_ tensor: MLXArray, cosVals: MLXArray, sinVals: MLXArray) -> MLXArray {
    let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
    return rotated.asType(tensor.dtype)
}

/// 多模态位置（3 轴 positionIds）RoPE 旋转：cos/sin 扩维后对 q/k 应用。
/// 使用方：HiDream-O1-Image（HiDreamLanguage）。
func applyMultimodalRotary(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    var cos = cos
    var sin = sin
    cos = expandedDimensions(cos, axis: 1)
    sin = expandedDimensions(sin, axis: 1)
    let qEmbedded = (q * cos) + (rotateHalf(q) * sin)
    let kEmbedded = (k * cos) + (rotateHalf(k) * sin)
    return (qEmbedded, kEmbedded)
}

// MARK: - RoPE 频率（视觉/文本）

/// 视觉塔 rotary 频率：theta=10k，invFreq = theta^(-i/dimension)，outer(seq, invFreq)。
/// 使用方：HiDream-O1-Image（HiDreamVision）。
final class VisionRotaryEmbedding {
    let dimension: Int
    let theta: Float

    init(dimension: Int, theta: Float = 10_000) {
        self.dimension = dimension
        self.theta = theta
    }

    func callAsFunction(sequenceLength: Int) -> MLXArray {
        let invFreq =
            1.0
            / pow(
                MLXArray(theta),
                MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                    / Float(dimension)
            )
        let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
        return outer(seq, invFreq)
    }
}

/// 多模态语言侧 rotary（MRoPE）：3 轴位置 interleave 频率，cos/sin 拼接输出。
/// mropeSection 参数化（原模型侧从 ropeScaling 配置取，默认 [24, 20, 20]）。
/// 使用方：HiDream-O1-Image（HiDreamLanguage）。
final class RotaryEmbedding {

    private let invFreq: MLXArray
    private let mropeSection: [Int]

    init(headDim: Int, base: Double, mropeSection: [Int]) {
        var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        freq = freq / Float(headDim)
        let baseArray = MLXArray(Float(base))
        self.invFreq = 1.0 / pow(baseArray, freq)
        self.mropeSection = mropeSection
    }

    private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
        let freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)

        let dims = freqs_t.dim(-1)
        var slices: [MLXArray] = []

        for idx in 0 ..< dims {
            var slice = freqs_t[0..., 0..., idx]

            for (dimIndex, offset) in [(1, 1), (2, 2)] {
                let end = min(mropeSection[dimIndex] * 3, dims)
                if idx >= offset && idx < end && (idx - offset) % 3 == 0 {
                    slice = freqs[dimIndex, 0..., 0..., idx]
                    break
                }
            }

            slices.append(slice)
        }

        return stacked(slices, axis: -1)
    }

    /// 向量化 interleave：当 mropeSection[1]/[2] 的 3 倍覆盖全部 dims 时，
    /// 3 轴逐位交织等价于 stack+transpose+reshape 一次完成（flat[k] 第 k 位
    /// 天然来自轴 k%3），避免原版逐 dim 切片 + stacked 的 CPU 循环（headDim/2 次）。
    /// 不满足全覆盖条件时回退 applyInterleavedMRope 原实现（逐维选择，语义一致）。
    /// 旧版 applyInterleavedMRope 保留不动。
    private func applyInterleavedMRopeFast(_ freqs: MLXArray) -> MLXArray {
        let dims = freqs.dim(-1)
        let sec1 = mropeSection.count > 1 ? mropeSection[1] * 3 : 0
        let sec2 = mropeSection.count > 2 ? mropeSection[2] * 3 : 0
        guard sec1 >= dims, sec2 >= dims else {
            return applyInterleavedMRope(freqs)
        }
        let f0 = freqs[0, 0..., 0..., 0...]   // [bs, seq, dims]
        let f1 = freqs[1, 0..., 0..., 0...]
        let f2 = freqs[2, 0..., 0..., 0...]
        let s = stacked([f0, f1, f2], axis: -1)          // [bs, seq, dims, 3]
        let t = s.transposed(0, 1, 3, 2)                 // [bs, seq, 3, dims]
        let flat = t.reshaped([f0.dim(0), f0.dim(1), dims * 3])  // 3 轴交织
        return flat[0..., 0..., 0..<dims]
    }

    func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
        var positionIds = positionIds
        if positionIds.ndim == 2 {
            positionIds = positionIds[.newAxis, 0..., 0...]
            positionIds = tiled(positionIds, repetitions: [3, 1, 1])
        }

        let pos = positionIds.asType(.float32)
        var invFreq = self.invFreq.asType(.float32)
        invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
        var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
        freqs = applyInterleavedMRopeFast(freqs)

        let emb = concatenated([freqs, freqs], axis: -1)
        let cosValues = cos(emb).asType(dtype)
        let sinValues = sin(emb).asType(dtype)
        return (cosValues, sinValues)
    }
}

// MARK: - Proportional RoPE 频率（Gemma global 层）

/// Gemma global 层 proportional RoPE freqs：[nRot real + nPad inf]，nRot = headDim·factor/2。
/// 纯数学，参数化自 Gemma4Config（globalHeadDim=512, factor=0.25, theta=1e6）。
/// 使用方：LTX-2.5（Gemma4TextEncoder）。
func proportionalRopeFreqs(headDim: Int, partialRotaryFactor: Float, theta: Float) -> MLXArray {
    let ghd = Float(headDim)
    let nRot = Int(Float(headDim) * partialRotaryFactor / 2)
    let nPad = headDim / 2 - nRot
    var freqs = [Float](repeating: 0, count: nRot + nPad)
    for i in 0..<nRot {
        let exponent = Float(2 * i) / ghd
        freqs[i] = pow(theta, exponent)
    }
    for i in nRot..<(nRot + nPad) {
        freqs[i] = .infinity
    }
    return MLXArray(freqs)
}
