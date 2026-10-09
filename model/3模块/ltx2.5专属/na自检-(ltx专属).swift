// 临时自检：稀疏 NA（新版）vs 逐偏移 NA（legacy 参考实现）数值等价性验证。验证后删除本文件。
import Foundation
import MLX
import MLXRandom

enum NASelfCheck {
    static func run() -> Int32 {
        MLXRandom.seed(42)
        let C = 128
        let B = 1, T = 6, H = 8, W = 8
        let kernel: (Int, Int, Int) = (3, 7, 7)
        func randW(_ shape: [Int]) -> MLXArray {
            MLXRandom.normal(shape, dtype: PrecisionPolicy.defaultMainDType) * MLXArray(0.02, dtype: PrecisionPolicy.defaultMainDType)
        }
        let toQW = randW([C, C]); let toQB = randW([C])
        let toKW = randW([C, C]); let toKB = randW([C])
        let toVW = randW([C, C]); let toVB = randW([C])
        let projW = randW([C, C]); let projB = randW([C])
        let qNormW = MLXArray.ones([C], dtype: PrecisionPolicy.defaultMainDType)
        let kNormW = MLXArray.ones([C], dtype: PrecisionPolicy.defaultMainDType)
        func runCase(_ B: Int, _ T: Int, _ H: Int, _ W: Int, _ kernel: (Int, Int, Int), _ label: String, _ budget: Int = 1_000_000_000, qNW: MLXArray? = nil, kNW: MLXArray? = nil) -> Bool {
            let qN = qNW ?? qNormW
            let kN = kNW ?? kNormW
            let x = MLXRandom.normal([B, T, H, W, C], dtype: PrecisionPolicy.defaultMainDType) * MLXArray(0.5, dtype: PrecisionPolicy.defaultMainDType)
            let a = DDVNA.neighborhoodAttention(
                x, kernel: kernel,
                qNormW: qN, kNormW: kN,
                toQW: toQW, toQB: toQB, toKW: toKW, toKB: toKB,
                toVW: toVW, toVB: toVB, projW: projW, projB: projB,
                scoreBudget: budget
            )
            let b = DDVNA.neighborhoodAttentionLegacy(
                x, kernel: kernel,
                qNormW: qN, kNormW: kN,
                toQW: toQW, toQB: toQB, toKW: toKW, toKB: toKB,
                toVW: toVW, toVB: toVB, projW: projW, projB: projB
            )
            let amax: Float = a.abs().max().item()
            let bmax: Float = b.abs().max().item()
            let diff: Float = (a - b).abs().max().item()
            let rel = diff / (bmax + 1e-6)
            print("NA_SELFCHECK[\(label)] nbr=\(kernel.0 * kernel.1 * kernel.2) a_max=\(amax) b_max=\(bmax) max_abs_diff=\(diff) rel=\(rel)")
            // bf16 累加顺序差异允许 ~2e-3 绝对误差；rel 应 <2%
            return rel < 0.02
        }
        var ok = true
        ok = runCase(1, 6, 8, 8, (3, 7, 7), "small-boundary") && ok
        ok = runCase(1, 16, 16, 16, (3, 7, 7), "multi-tile", 10_000_000) && ok
        ok = runCase(1, 32, 32, 32, (3, 7, 7), "large-tiled") && ok
        // 真实权重形状 [headDim]=64（每 head 共享，非 ones 随机权重）：验证 per-head 归一分支
        let qNormH = randW([64]); let kNormH = randW([64])
        ok = runCase(1, 6, 8, 8, (3, 7, 7), "real-headdim-w", qNW: qNormH, kNW: kNormH) && ok
        return ok ? 0 : 1
    }
}

// MARK: - 临时性能实测：稀疏 NA（当前正式实现）vs mask 全矩阵参考（旧版核心计算路径）
// 对比对象：DDVNA.neighborhoodAttention（稀疏 gather+einsum） vs maskRef（nq×nk 全矩阵 additive mask+SDPA）
// 语义一致（窗口铺满、边界内移、3D RoPE、q 缩放），测得端到端 NA 函数级加速比与数值差异。
enum NABench {
    static func run() -> Int32 {
        MLXRandom.seed(42)
        let C = 128
        let kernel: (Int, Int, Int) = (11, 11, 11)   // stage5 真实 kernel
        func randW(_ shape: [Int]) -> MLXArray {
            MLXRandom.normal(shape, dtype: PrecisionPolicy.defaultMainDType) * MLXArray(0.02, dtype: PrecisionPolicy.defaultMainDType)
        }
        let toQW = randW([C, C]); let toQB = randW([C])
        let toKW = randW([C, C]); let toKB = randW([C])
        let toVW = randW([C, C]); let toVB = randW([C])
        let projW = randW([C, C]); let projB = randW([C])
        let qNormW = MLXArray.ones([C], dtype: PrecisionPolicy.defaultMainDType)
        let kNormW = MLXArray.ones([C], dtype: PrecisionPolicy.defaultMainDType)

        func timeIt(_ f: () -> MLXArray) -> (Double, MLXArray) {
            _ = f()   // warmup（含编译/调度）
            var best = Double.infinity
            var last = f()
            last.eval()
            for _ in 0..<3 {
                let t0 = ContinuousClock.now
                let o = f()
                o.eval()
                let dt = ContinuousClock.now - t0
                let secs = Double(dt.components.seconds) + Double(dt.components.attoseconds) * 1e-18
                if secs < best { best = secs; last = o }
            }
            return (best, last)
        }

        func runCase(_ T: Int, _ H: Int, _ W: Int, _ label: String) {
            let B = 1
            let x = MLXRandom.normal([B, T, H, W, C], dtype: PrecisionPolicy.defaultMainDType) * MLXArray(0.5, dtype: PrecisionPolicy.defaultMainDType)
            let (tS, oS) = timeIt {
                DDVNA.neighborhoodAttention(
                    x, kernel: kernel,
                    qNormW: qNormW, kNormW: kNormW,
                    toQW: toQW, toQB: toQB, toKW: toKW, toKB: toKB,
                    toVW: toVW, toVB: toVB, projW: projW, projB: projB,
                    scoreBudget: 1_000_000_000
                )
            }
            let (tM, oM) = timeIt {
                maskRef(
                    x, kernel: kernel,
                    qNormW: qNormW, kNormW: kNormW,
                    toQW: toQW, toQB: toQB, toKW: toKW, toKB: toKB,
                    toVW: toVW, toVB: toVB, projW: projW, projB: projB
                )
            }
            let amax: Float = oS.abs().max().item()
            let bmax: Float = oM.abs().max().item()
            let diff: Float = (oS - oM).abs().max().item()
            let rel = diff / (bmax + 1e-6)
            print("NA_BENCH[\(label)] grid=\(T)x\(H)x\(W) nbr=\(kernel.0 * kernel.1 * kernel.2)")
            print("NA_BENCH[\(label)] sparse=\(String(format: "%.4f", tS))s mask=\(String(format: "%.4f", tM))s speedup=\(String(format: "%.1f", tM / tS))x")
            print("NA_BENCH[\(label)] max_abs_diff=\(diff) rel=\(rel)  (等价判据 rel<2%)")
        }

        runCase(16, 16, 16, "grid16")     // 4096 tokens，mask 全矩阵可控
        runCase(24, 24, 24, "grid24")     // 13824 tokens，更接近真实规模
        func runMLPCase(_ n: Int, _ C: Int, _ mid: Int, _ label: String) {
            let x = MLXRandom.normal([n, C], dtype: PrecisionPolicy.defaultMainDType) * MLXArray(0.5, dtype: PrecisionPolicy.defaultMainDType)
            let upW = randW([mid, C]); let gateW = randW([mid, C]); let downW = randW([C, mid])
            let (tC, oC) = timeIt {
                swigluMLP(x, upW: upW, upB: nil, gateW: gateW, gateB: nil, downW: downW)
            }
            let (tL, oL) = timeIt {
                swigluMLPLegacy(x, upW: upW, upB: nil, gateW: gateW, gateB: nil, downW: downW)
            }
            let diff: Float = (oC - oL).abs().max().item()
            let rel = diff / (oL.abs().max().item() + 1e-6)
            print("NA_BENCH[MLP-\(label)] tokens=\(n) C=\(C) mid=\(mid) tiles=\((n + 16383) / 16384)")
            print("NA_BENCH[MLP-\(label)] compiled=\(String(format: "%.4f", tC))s legacy=\(String(format: "%.4f", tL))s speedup=\(String(format: "%.2f", tL / tC))x")
            print("NA_BENCH[MLP-\(label)] max_abs_diff=\(diff) rel=\(rel)")
        }

        // SwiGLU MLP 编译图对比：真实维度 C=256/mid=1024（diff_blocks），规模取 stage5 每层 token 量
        runMLPCase(65536, 256, 1024, "small-4tiles")          // 65536/16384=4 块
        runMLPCase(1_852_416, 256, 1024, "stage5-113tiles")   // 185 万 token ≈113 块
        return 0
    }

    /// mask 全矩阵参考：整段投影 + nq×nk additive mask + softmax + @v（旧 mask 版核心计算路径的忠实还原）
    static func maskRef(
        _ x: MLXArray,
        kernel: (Int, Int, Int),
        qNormW: MLXArray, kNormW: MLXArray,
        toQW: MLXArray, toQB: MLXArray,
        toKW: MLXArray, toKB: MLXArray,
        toVW: MLXArray, toVB: MLXArray,
        projW: MLXArray, projB: MLXArray
    ) -> MLXArray {
        let shape = x.shape
        let B = shape[0], T = shape[1], H = shape[2], W = shape[3], C = shape[4]
        let headDim = 64, heads = C / headDim
        let N = B * T * H * W
        let xFlat = x.reshaped([N, C])
        let tQW = toQW.transposed(1, 0), tKW = toKW.transposed(1, 0)
        let tVW = toVW.transposed(1, 0), tPW = projW.transposed(1, 0)
        let q = (xFlat.matmul(tQW) + toQB).reshaped([N, heads, headDim])
        let k = (xFlat.matmul(tKW) + toKB).reshaped([N, heads, headDim])
        let v = (xFlat.matmul(tVW) + toVB).reshaped([N, heads, headDim])
        let qn = rmsNormWeighted(q.reshaped([N, C]), weight: qNormW).reshaped([N, heads, headDim]) * MLXArray(1.0 / sqrt(Float(headDim)), dtype: q.dtype)
        let kn = rmsNormWeighted(k.reshaped([N, C]), weight: kNormW).reshaped([N, heads, headDim])
        let qr = DDVNA.rotaryPosEmbed3D(qn.reshaped([B, T, H, W, heads, headDim])).reshaped([N, heads, headDim])
        let kr = DDVNA.rotaryPosEmbed3D(kn.reshaped([B, T, H, W, heads, headDim])).reshaped([N, heads, headDim])
        let scale = powf(Float(headDim), -0.5)
        let kt = Swift.min(kernel.0, T), kh = Swift.min(kernel.1, H), kw = Swift.min(kernel.2, W)
        func windowStarts(_ n: Int, _ k: Int) -> [Int] {
            let half = k / 2
            return (0..<n).map { Swift.max(0, Swift.min($0 - half, n - k)) }
        }
        let st = windowStarts(T, kt), sh = windowStarts(H, kh), sw = windowStarts(W, kw)
        // 全矩阵 scores (N,heads,N)，窗口外 -inf
        let scores = einsum("qhd,nhd->qhn", qr, kr).asType(.float32) * MLXArray(scale)
        let row = MLXArray(Array(0..<N)).asType(.int32)
        let tR = row.floorDivide(H * W) % T
        let hR = row.floorDivide(W) % H
        let wR = row % W
        let stArr = MLXArray(st).asType(.int32).take(tR, axis: 0).reshaped([N, 1])
        let shArr = MLXArray(sh).asType(.int32).take(hR, axis: 0).reshaped([N, 1])
        let swArr = MLXArray(sw).asType(.int32).take(wR, axis: 0).reshaped([N, 1])
        let tC = tR.reshaped([1, N]), hC = hR.reshaped([1, N]), wC = wR.reshaped([1, N])
        // 窗口有效掩码：拆开逐项计算避免 type-check 超时
        let ktArr = MLXArray(Int32(kt)), khArr = MLXArray(Int32(kh)), kwArr = MLXArray(Int32(kw))
        let inT = (tC .>= stArr) & (tC .< stArr + ktArr)
        let inH = (hC .>= shArr) & (hC .< shArr + khArr)
        let inW = (wC .>= swArr) & (wC .< swArr + kwArr)
        let valid = inT & inH & inW
        let mask = MLX.where(valid, MLXArray(0.0), MLXArray(-Float.infinity)).reshaped([N, 1, N])
        let s32 = scores + mask
        let mx = s32.max(axis: -1, keepDims: true)
        let es = (s32 - mx).exp()
        let sm = es / es.sum(axis: -1, keepDims: true)
        let smB = sm.asType(x.dtype)
        let o = einsum("qhn,nhd->qhd", smB, v).reshaped([B, T, H, W, C])
        return o.matmul(tPW) + projB
    }

    /// 旧版（未 compile 图）SwiGLU MLP：NABench 对比编译图收益的参考实现（与改动前逐行一致）
    static func swigluMLPLegacy(
        _ x: MLXArray,
        upW: MLXArray, upB: MLXArray?,
        gateW: MLXArray, gateB: MLXArray?,
        downW: MLXArray
    ) -> MLXArray {
        let shape = x.shape
        let C = shape.last!
        let n = shape.dropLast().reduce(1, *)
        let flat = x.reshaped([n, C])
        let mid = upW.shape[0]
        let ugWt = concatenated([upW, gateW], axis: 0).transposed(1, 0)
        let upBv = upB ?? MLXArray.zeros([mid], dtype: x.dtype)
        let gateBv = gateB ?? MLXArray.zeros([mid], dtype: x.dtype)
        let ugB = concatenated([upBv, gateBv], axis: 0)
        let downWt = downW.transposed(1, 0)
        let tileSize = 16384
        var outs: [MLXArray] = []
        var start = 0
        while start < n {
            let end = Swift.min(start + tileSize, n)
            let ts = end - start
            let xb = flat[start..<end]
            let ug = xb.matmul(ugWt) + ugB
            let up = ug[0..<ts, 0..<mid]
            let gate = ug[0..<ts, mid..<(2 * mid)]
            let down = up * siluActivation(gate)
            outs.append(down.matmul(downWt))
            start = end
        }
        return concatenated(outs, axis: 0).reshaped(shape)
    }
}
