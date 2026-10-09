
//
//  优化方法.swift
//  无限画布 — 跨模型可复用的推理加速技术库
//
//  ============================================================
//  作用：集中存放"训练无关 / 跨模型可复用"的推理加速方法，
//   未来新模型（HiDream 图像、其他视频模型）直接复用。
//  已收录：
//    · TeaCache 特征缓存：相邻去噪步 sigma 差小于阈值时复用上一步 x0，
//      跳过整次 DiT 前向（LTX-2.5 蒸馏表前 4 步间隔 0.00625 → 约一半步数免费）。
//  待评估：
//    · OmniCache / BWCache 分层缓存（按空间分块相似度动态缓存）
//    · 融合 QKV（一次 GEMM 出 Q/K/V，减少 GEMM 调用）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - TeaCache（官方 rel-L1 特征缓存，LTX-2.5 用）

/// TeaCache：官方 ComfyUI-TeaCache（welltop-cn）语义实现，训练无关。
///
/// 官方机制（teacache_ltxvmodel_forward）：
/// 1. 每步以"第一个 transformer block 的视频调制输入（img_attn_norm 调制结果）"为特征；
/// 2. 判据 = 相邻步特征 rel-L1 距离经模型专属多项式（ltxv 系数）映射后累加，
///    累加值 < rel_l1_thresh 才跳步；超阈值则真算并清零累加值；
/// 3. 真算步缓存"输出调制残差"（norm_out+modulate 后 − blocks 输入）；
/// 4. 跳步时 x += previous_residual（残差已含 norm+modulate），再走输出投影，
///    只省掉 blocks（含 norm+modulate）的大头计算，embedding/残差加法/输出头仍重算；
/// 5. CFG 的 cond/uncond 分支各自维护独立状态（官方 teacache_state key 0/1），互不污染。
///
/// 与旧 TeaCacheLegacy（σ 间隔判据 + 缓存整步 x0）的本质区别：真 TeaCache 每步都重算
/// embedding、残差加法与输出头，只有 blocks 被跳过，因此对 dev DynamicShift 表同样合法。
final class TeaCache {
    /// CFG 分支标识（对齐官方 teacache_state key：0=cond，1=uncond）
    enum BranchKind {
        case cond
        case uncond
    }

    /// 官方 LTX-Video 推荐 rel_l1_thresh = 0.06（README 参数表）；蒸馏表可按需调参。
    var threshold: Float = 0.06
    /// 官方 LTX-Video 推荐 max_skip_steps = 3：连续跳步上限，防止残差累积误差与周期性空间伪影（网格感）。
    var maxSkipSteps: Int = 3
    /// ltxv 专属多项式系数（高次→常数，官方 SUPPORTED_MODELS_COEFFICIENTS["ltxv"]）
    var coefficients: [Float] = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]

    /// 单 CFG 分支缓存状态（官方 teacache_state[k]）
    struct BranchState {
        var accumulated: Float = 0          // 多项式映射累加距离
        var skipSteps: Int = 0              // 连续跳步计数（官方 skip_steps，达 maxSkipSteps 强制真算）
        var previousModulated: MLXArray?    // 上一步特征（第一个 block 调制输入）
        var previousResidualV: MLXArray?    // 上一步视频输出调制残差（norm+modulate 后 − blocks 输入）
        var previousResidualA: MLXArray?    // 上一步音频输出调制残差
        var shouldCalc: Bool = true
    }
    var cond = BranchState()     // CFG cond 分支（key 0，condV+condA）
    var uncondV = BranchState()  // 视频 CFG 负向分支（negV+condA；官方 key 1 拆分为 video/audio 两个独立槽，
                                 // 避免本地三次独立 forward 交替 store 覆盖对方残差导致跳步错位）
    var uncondA = BranchState()  // 音频 CFG 负向分支（condV+negA）

    /// 官方 update_cache_state：先判连续跳步上限，再累加 poly1d 映射后的 rel-L1；
    /// 累加值 < 阈值 → 跳步并计数；超阈值 → 真算并清零。
    /// probe 为当前步第一个 block 的视频调制输入（各分支共用同一 latent/sigma，probe 相同，
    /// 但状态独立累积、残差独立缓存，避免跨分支污染）。
    /// 残差缺失时强制真算（首步 / preWarm 复用首步的场景）。
    func update(_ state: inout BranchState, probe: MLXArray) {
        if state.skipSteps >= maxSkipSteps {
            // 官方：连续跳步达上限强制真算（防残差累积误差 → 网格/结构伪影）
            state.shouldCalc = true
            state.accumulated = 0
            state.skipSteps = 0
        } else if let prev = state.previousModulated {
            let rel = abs(probe - prev).mean() / abs(prev).mean()   // 官方 rel_l1（GPU 归约后取标量）
            state.accumulated += teaCachePoly1d(coefficients, rel.item())
            if state.accumulated < threshold {
                state.shouldCalc = false
                state.skipSteps += 1
            } else {
                state.shouldCalc = true
                state.accumulated = 0
                state.skipSteps = 0
            }
        } else {
            state.shouldCalc = true   // 首步必算
        }
        state.previousModulated = probe
        // 残差缺失（preWarm 复用首步等）时无法跳步，强制真算
        if state.previousResidualV == nil || state.previousResidualA == nil {
            state.shouldCalc = true
        }
    }

    /// 真算步后存储输出调制残差，供后续跳步复用。
    func store(_ state: inout BranchState, residualV: MLXArray, residualA: MLXArray) {
        state.previousResidualV = residualV
        state.previousResidualA = residualA
    }

    /// 清空状态（新采样任务/换 sigma 表时调用）。
    func reset() {
        cond = BranchState()
        uncondV = BranchState()
        uncondA = BranchState()
    }
}

/// 官方 poly1d：result = Σ coeff[i] * x^(n−1−i)（高次在前，对齐 nodes.py）
func teaCachePoly1d(_ coefficients: [Float], _ x: Float) -> Float {
    var result: Float = 0
    let n = coefficients.count
    for (i, c) in coefficients.enumerated() {
        result += c * powf(x, Float(n - 1 - i))
    }
    return result
}

// MARK: - TeaCacheLegacy（σ 间隔启发式，仅供 HiDream 等图像模型）

/// 旧版 TeaCache：用 sigma 绝对差做启发式（|σ_i − σ_{i−1}| < threshold 复用上一步 x0）。
/// 仅对蒸馏固定 sigma 表"碰巧"成立（前几步间隔极小），对 dev DynamicShift 表会误命中
/// 且缓存整步 x0 属错误语义。LTX-2.5 已改用官方语义的 TeaCache，本类仅保留给
/// HiDream 等图像管线继续使用（避免大面积回归）。
final class TeaCacheLegacy {
    /// 触发缓存的 sigma 绝对差阈值。
    /// LTX-2.5 蒸馏表：1.0→0.99375→… 前 4 步间隔 0.00625；0.975→0.909375 间隔 0.0656。
    /// 默认 0.02 保守（只覆盖前 4 步）；调大可覆盖更多步但可能引入伪影。
    var threshold: Float = 0.02
    /// 前 N 步不缓存（sigma 高位变化最敏感，保质量；默认 1 = 第 1 步必算）。
    var warmupSteps: Int = 1
    private var stepIndex = 0
    private var prevSigma: Float?
    private var cachedV: MLXArray?
    private var cachedA: MLXArray?

    /// 判断当前步是否命中缓存。
    /// - 命中：outV/outA 被赋值为上一步的 x0，返回 true（调用方跳过模型前向）。
    /// - 未命中：返回 false（调用方正常前向后必须调 store()）。
    func tryReuse(sigma: Float, outV: inout MLXArray?, outA: inout MLXArray?) -> Bool {
        defer { prevSigma = sigma }
        stepIndex += 1
        guard stepIndex > warmupSteps,
              let p = prevSigma,
              abs(sigma - p) < threshold,
              let cv = cachedV,
              let ca = cachedA else { return false }
        outV = cv
        outA = ca
        return true
    }

    /// 正常前向后存储 x0（含 I2V mask 钉入后的结果），供后续步复用。
    func store(v: MLXArray, a: MLXArray) {
        cachedV = v
        cachedA = a
    }

    /// 清空状态（新采样任务/换 sigma 表时调用）。
    func reset() {
        stepIndex = 0
        prevSigma = nil
        cachedV = nil
        cachedA = nil
    }
}

// MARK: - 多槽编译图缓存（通用）

/// 多槽编译图缓存：按 key（shape/配置签名）同时驻留多份 MLX.compile 产物。
/// 解决单槽缓存致命缺陷——stage 切换（如 LTX stage1→stage2）时 shape 不同会
/// 逐出旧图，导致下次生成又要重编译 30-180s，stage 交替时每轮都白编译一次。
///
/// 设计要点：
/// - 泛型 Owner 弱引用：不阻止 DiT 等宿主释放，避免缓存成为长生命周期强引用环。
/// - 编译闭包对权重是 MLXArray 引用语义，多槽存多份图只增加图结构开销，权重不翻倍。
/// - key 建议由 shape + 标志位拼接（如 shapeKey），同一宿主相同 key 复用同一图。
/// - NSLock 保护：采样循环与监控面板可能跨线程访问。
final class CompiledGraphCache<Owner: AnyObject, F> {
    private struct Entry {
        let graph: F
        weak var owner: Owner?
        var lastUsed: Int   // 单调递增时钟，LRU 淘汰依据
    }
    private var entries: [String: Entry] = [:]
    private var tick: Int = 0
    private let lock = NSLock()

    /// 槽位上限：stage2 等形状多变时防止编译图无限累积（图结构也是真内存）。
    /// 常用形状自然留下，冷门形状超限后被 LRU 淘汰。
    var maxEntries: Int = 8

    /// 命中条件：key 存在 且 owner 仍存活且为同一实例。命中刷新 lastUsed。
    func get(owner: Owner, key: String) -> F? {
        lock.lock(); defer { lock.unlock() }
        tick &+= 1
        guard var e = entries[key], e.owner === owner else { return nil }
        e.lastUsed = tick
        entries[key] = e
        return e.graph
    }

    /// 存入编译图；同 key 覆盖（一般同 key 即同宿主同 shape，直接复用）。
    /// 超上限时淘汰最久未使用的槽。
    func store(_ graph: F, owner: Owner, key: String) {
        lock.lock(); defer { lock.unlock() }
        tick &+= 1
        entries[key] = Entry(graph: graph, owner: owner, lastUsed: tick)
        while entries.count > maxEntries, let victim = entries.min(by: { $0.value.lastUsed < $1.value.lastUsed }) {
            entries.removeValue(forKey: victim.key)
        }
    }

    /// 卸载指定槽（如宿主释放、显式动态卸载）。
    func remove(key: String) {
        lock.lock(); defer { lock.unlock() }
        entries.removeValue(forKey: key)
    }

    /// 卸载全部编译图（释放闭包对权重的强捕获）。
    func clear() {
        lock.lock(); defer { lock.unlock() }
        entries.removeAll()
    }

    /// 前缀瘦身：淘汰 key 不以指定前缀开头的槽，保留将复用形状的编译图。
    /// 用于 stage 切换/refine 前：保留同形状图（命中即复用、免重编译——编译窗口正是内存峰值），
    /// 淘汰其余大图，避免多张超大图同时驻留（实测：保最近N张LRU会让上轮全清大图在下一轮
    /// stage2 编译峰值期继续驻留 → 压缩 30G 死机）。原无条件 clear() 会把上轮同形状图一并清掉。
    func evictExcept(keyPrefix: String) {
        lock.lock(); defer { lock.unlock() }
        let victims = entries.keys.filter { !$0.hasPrefix(keyPrefix) }
        for k in victims { entries.removeValue(forKey: k) }
    }

    /// 当前驻留槽数（监控面板用）。
    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.count
    }

    var isEmpty: Bool { count == 0 }
}


