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
//      Eq.4-5  z_lat = upscale(x0)              （learned upscaler 直接提升低分 x0）
//              z_pix = VAE⁻¹( ↑(VAE(x0)) )      （VAE 像素往返，作为伪影度量锚点）
//      Eq.6-9  δ = z_pix - z_lat；按 |δ| 取风险最高的 ρ 比例位置、强度线性映射 w ∈ [w_min, w_max]
//              z0_high = z_lat + w·δ
//      Eq.10   x_k = (1-σ_k)·z0_high + σ_k·ε    （重加噪，官方 noise_scaling 公式）
//              （不补 Euler：重加噪后直接进高清首步）
//
//  不变量：低分 NFE = ts，高分 NFE = N - ts，合计恒等于原总步数 N（过渡块零 NFE）。
//  加速来源：前 ts 次前向跑在 lowResScale 的低空间分辨率上，单步算力按面积下降。
//
//  ── 参数对齐（唯一参考：comfyui-SelfLift 官方实现，禁止自创）─────────────
//  lowResScale  = 0.5   官方节点默认值（nodes.py 行 481 / README 行 54）
//  rho          = 0.0   官方 SelfLiftH3Sampler 默认值：learned upscaler 下纯 z_lat 提升，
//                       像素锚点（rho>0）为可选增强，本工程默认关闭
//  wMin = wMax  = 1.0   官方 H3 建议起点（README 行 29 / 61）
//  提升          = 官方 learned upscaler（时间维感知 ×2）；不可用即显式报错，不降级
//  采样器        = 标准 euler 且 s_churn = 0（README 行 79）
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

// MARK: - 配置

public struct SelfLiftConfig: Sendable {
    /// 总开关
    public var enabled: Bool
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

    /// CFG 引导强度：0 = 关闭（仅 positive 单路，torbo LoRA 适用）；
    /// >0 时每步 v = v_neg + cfg·(v_pos − v_neg)，negative 条件行由调用方注入。
    public var cfgScale: Double

    /// σ_k 覆盖入口（低清终点 / 交接深度）：nil 时用 `makeFixedFinalStep` 默认 0.75（落调度前吸附公式档）；
    /// 环境变量 NA_H3_SELFLIFT_DECOUPLE_LOW_K 可覆盖（命名沿用旧解耦 A/B 时代）。
    public var decoupleSigmaK: Double?

    /// 交接状态：true（默认）= 末步 Eq.3 预测 clean x0 → 升频 → 按 σ_ren 重加噪交高清；
    /// false = 末步正常 Euler 落到 σ_k，交出含噪 x（升频后不重加噪，高清直接接手）。
    /// 环境变量 NA_H3_SELFLIFT_CLEAN_HANDOFF 可覆盖。
    public var cleanEndpointHandoff: Bool

    /// 重加噪深度 σ_ren：clean x0 升频到高清后按该 σ 重加噪（Eq.10）再交高清收尾。
    /// nil = 跟随 makeFixedFinalStep 默认（clean 端点 0.75，与 σ_k 等深）。
    /// 环境变量 NA_H3_SELFLIFT_RENOISE_SIGMA 可覆盖。
    public var renoiseSigma: Double?

    public init(enabled: Bool = true,
                lowResScale: Double = 0.5,
                rho: Double = 0.0,
                wMin: Float = 1.0,
                wMax: Float = 1.0,
                cfgScale: Double = 0.0,
                decoupleSigmaK: Double? = nil,
                cleanEndpointHandoff: Bool = true,
                renoiseSigma: Double? = nil) {
        self.enabled = enabled
        self.lowResScale = lowResScale
        self.rho = rho
        self.wMin = wMin
        self.wMax = wMax
        self.cfgScale = cfgScale
        self.decoupleSigmaK = decoupleSigmaK
        self.cleanEndpointHandoff = cleanEndpointHandoff
        self.renoiseSigma = renoiseSigma
    }


    /// 是否走 VAE 像素锚点（rho>0 且 wMax>0）——为真时过渡块会多一趟 decode→encode
    /// 与官方 `need_pix = rho > 0.0 and w_max > 0.0` 一致（selflift.py 行 26-28）
    public var needsPixelAnchor: Bool { rho > 0 && wMax > 0 }
    /// 是否走直接 latent 提升（rho=w_min=w_max=1 的纯锚点模式下可省）
    /// 与官方 `need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)` 一致
    public var needsLatentLift: Bool { !(rho >= 1 && wMin >= 1 && wMax >= 1) }

}

// MARK: - 调度切分

public enum SelfLiftScheduleError: Error, CustomStringConvertible {
    case tooFewSteps(Int)

    public var description: String {
        switch self {
        case .tooFewSteps(let n):
            return "SelfLift: 总步数不足（N=\(n)），至少需要 2 步"
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

    /// 用户输入 σ 吸附到公式曲线时的采样分辨率（档数）。256 档时 0.75→0.7491、0.5→0.5042，
    /// 既保证「值是公式真实产出的档位」，又不会像按 N 步网格吸附那样把 0.5 吸到 0.7。
    public static let snapResolution: Int = 256
    // 生产调度走 makeFixedFinalStep（固定高清最后 1 步），不用 75% 推导。
    // ══════════════════════════════════════════════════════════════════════

    // ──────────────────────────────────────────────────────────────────────
    // 固定最后一步高清（M=1）· 公式网格调度（2026-10-25 用户需求改版）
    //
    //   新口径：
    //     · 低清网格 = 公式曲线（与直出 `sigmaSchedule(steps:shift:)` 逐点同式），b 空间等距；
    //     · 用户输入的 σ_k（低清跑到哪）/ σ_ren（重加噪深度）先**吸附到公式网格上最近的 σ 档**，
    //       把 0.5 这种「线性口径的值」换成公式真实存在的档位；
    //     · 低清终点精确落在吸附后的 σ_k（b 端点反解，不靠反推取整），高清固定 1 步 σ_ren → 0。
    //   N=6、shift=12、σ_k=σ_ren=0.75 → 吸附到公式档 σ≈0.7491；低清 5 步（clean 为 4 个间隔 +
    //   末步 Eq.3 占位）、高清 1 步，总 NFE 仍为 N=6。
    // ──────────────────────────────────────────────────────────────────────
    /// - Parameter totalSteps: 面板第一阶段总步数 N（总 NFE = 低清步数 + 高清固定 1 步）
    /// - Parameter sigmaK: 用户输入的低清终点 / 交接深度（默认 0.75），内部吸附到公式最近档
    /// - Parameter renoiseSigma: 用户输入的重加噪深度（默认与 sigmaK 等深 0.75）
    /// - Note: 0.75/0.75 为当前定档；若低清残影过重，可把 sigmaK 与 renoiseSigma 一起降回 0.5。
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
        let renTarget = renoiseSigma ?? (cleanEndpoint ? 0.75 : (lowToZero ? 0.5 : sigmaK))
        let sigmaNext = nearestGridValue(to: renTarget, grid: snapGrid, range: 1...snap)
        // ③ 低清网格：b 空间等距 → 公式 σ，末项精确等于吸附后的 σ_k（不按档截断，避免终点漂移）。
        //    总 NFE 恒为 N：
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
        // 不补 Euler（该步并入高清首步）。
        let renSigma = split.sigmaNext
        let xK = noiseScaling(z0High, renSigma, noise)
        holdIntermediate?(xK)
        let out = xK
        holdIntermediate?(out)
        return out
    }
}
