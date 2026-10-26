import MLX
import Foundation

// 使用方：H3（H3Transformer 等具体调用点如下；本文件为通用底层能力，后续其它模型可直接复用）
//   - model/minimax h3/H3Pipeline.swift  采样前创建缓存实例：H3AttnBroadcast(count: dit.blocks.count)（pabK > 1 时）
//   - model/minimax h3/SelfLiftH3-(h3专属).swift  低分/高分段各建一份
//   - model/minimax h3/H3Transformer.swift  H3DiT 层内以形参 attnBcast: H3AttnBroadcast? 接入跨步复用

// MARK: - Attention broadcast cache (PAB-style, 独立文件)
//
// 从 H3Transformer.swift / H3Layout.swift 抽出集中维护的「跨去噪步 attention 缓存」。
//
// 语义澄清（勿与 sparse 折叠混淆）：
// - 这里复用的是**每层 attention 分支的输出张量 at**（计算复用，非画面帧复用），
//   MLP 分支（norm2+fc1+fc2）**每步重算**，对齐官方 zig 实现——官方只缓存 at。
//   旧版额外缓存了 MLP 输出 mo，等于把「上一步 σ 的 MLP 非线性输出」硬套到当前步，
//   是画面鬼影/运动不连贯的主要来源，已于 2026-10-26 移除。
// - 缓存发生在**去噪步之间**（step i → step i+1），不是视频帧之间；
// - 官方 sparseAttend 折叠稀疏（花屏证伪过）与 PAB 是两件独立的事，两者互不替代：
//   非刷新步直接复用 at，跳过 norm1+qkv+rope+attn；
//   刷新步仍走当前 sparsePolicy，并写回 blocks[bi]。
// - gate 加权（modGate）不受缓存影响：无论 refresh/reuse，两处 gate 均用当步新 gate 照常执行，
//   保持既有 PAB 语义（旧 attention 内容 + 新门控）。
// - 复用判据 = 固定 k 调度（attnBroadcastRefresh）**且** σ 过期量未超容差
//   （attnBroadcastMustRefresh）；后者保证「复用的内容来自并非太远的噪声深度」。
//
// ⚠️ MLX.compile 纯函数约束：本类持有跨步可变状态（blocks 为 Swift 引用），
// 若把 H3DiT.forward 整体包进 MLX.compile 闭包会破坏纯函数性导致失效/重编译；
// 需要编译图时应把缓存移出被编译闭包（或改由数组返回值传递），参考 makeCompiledDitForward。

// MARK: - 缓存容器

/// PAB 跨步 block 缓存：blocks[bi] 持有第 bi 层 attention 分支输出 at。
/// 接入方在采样循环外创建（count 取 DiT 层数，如 dit.blocks.count），
/// 循环内按 attnBroadcastRefresh + attnBroadcastMustRefresh 决定每步 refresh/reuse 后传入 forward。
public final class H3AttnBroadcast {
    public var blocks: [MLXArray?]
    /// 最近一次真正重算（写缓存）时对应的当步 σ；+∞ 表示尚未写入任何有效缓存。
    /// 用于按真实过期量强制刷新，与固定 k 调度并行生效。
    public var cachedSigma: Double = .infinity

    public init(count: Int) {
        self.blocks = Array(repeating: nil, count: count)
    }

    public func reset() {
        for i in blocks.indices {
            blocks[i] = nil
        }
        cachedSigma = .infinity
    }
}

// MARK: - 刷新调度

/// 按 30 步等比缩放的门控辅助：步数少时缩到最小 1，步数多时封顶 at30。
func scaledGate(steps: UInt32, at30: UInt32) -> Int {
    max(1, min(Int(at30), Int(steps) * Int(at30) / 30))
}

/// 起始段强制重算的步数（首帧不缓存，保证采样初段按真实调制走）。
public func attnBroadcastWarmup(_ steps: UInt32) -> Int { scaledGate(steps: steps, at30: 4) }

/// 收尾段强制重算的步数（末帧 sigma 变化大，复用收益低且风险高）。
public func attnBroadcastTail(_ steps: UInt32) -> Int { scaledGate(steps: steps, at30: 2) }

/// First-Block Cache 的 warmup 调度（Cache-DiT 风格，尚未接线时保留备用）。
public func stepCacheWarmup(_ steps: UInt32) -> Int { scaledGate(steps: steps, at30: 2) }

/// 全程刷新调度：k<=1 等价禁用（每步都 refresh）；否则 warmup 段 + tail 段
/// 强制重算，中间步按 `(i - warm) % k == 0` 的间隔刷新，其余步复用上一步 attention 输出。
/// 例：6 步 + k=2 → warm=1/tail=1，refresh 步 {0,1,3,5}，复用步 {2,4}。
public func attnBroadcastRefresh(_ i: Int, steps: UInt32, k: UInt32) -> Bool {
    if k <= 1 { return true }
    let warm = attnBroadcastWarmup(steps)
    if i < warm { return true }
    if i + attnBroadcastTail(steps) >= Int(steps) { return true }
    return (i - warm) % Int(k) == 0
}

/// 按真实过期量判定是否必须重算：|σ_now − 缓存写入时 σ| > maxStale ⇒ 强制刷新。
/// 未写入过缓存时 cachedSigma = +∞，比较结果为假 ⇒ 必刷新（首步安全）。
/// 这是「相邻步变化量」的近似判据（与 PAB / TeaCache / Cache-DiT 同一思路），
/// 与固定 k 调度叠加：k 决定「能不能复用」，本函数兜住「复用跨度内 σ 到底变了多少」。
public func attnBroadcastMustRefresh(cachedSigma: Double, sigma: Double, maxStale: Double) -> Bool {
    !(abs(sigma - cachedSigma) <= maxStale)
}

/// 干跑整段调度，返回每步 refresh 标记（与运行时同一判定逻辑）。用途：
/// ① 判定「整段是否根本不具备复用机会」→ 不建缓存实例，省下 count 层 at 常驻（数值等价每步重算）；
/// ② 日志/复盘时对照每个复用步的实际跨度。
public func attnBroadcastPlan(sigmas: [Double], k: UInt32, maxStale: Double) -> [Bool] {
    let nfe = max(0, sigmas.count - 1)
    guard nfe > 0 else { return [] }
    var out: [Bool] = []
    out.reserveCapacity(nfe)
    var cached: Double = .infinity
    for i in 0..<nfe {
        let base = attnBroadcastRefresh(i, steps: UInt32(nfe), k: k)
        let refresh = base || attnBroadcastMustRefresh(cachedSigma: cached, sigma: sigmas[i], maxStale: maxStale)
        if refresh { cached = sigmas[i] }
        out.append(refresh)
    }
    return out
}

/// 默认 σ 过期容差（单次复用的实际跨度 |Δσ| 上限）：
/// 旧「安全」基线是 σ_k=0.95 / 5 步 → Δσ≈0.0125；取 0.04（≈3 倍）作为起点。
/// 按 shift=12 / σ_k=0 曲线实测干跑：N=6 只剩 1 个复用步（跨度 0.024，落在低 σ 细步长段），
/// N=12 剩 4 个、N=30 剩 10 个（≈1/3），即「大步长段自动每步重算，细步长段照常赚复用」。
/// NA_H3_PAB_MAX_DSIGMA / NA_H3_SELFLIFT_PAB_MAX_DSIGMA 可覆盖（放很大 = 退回纯 k 调度）。
public func attnBroadcastDefaultMaxStale() -> Double { 0.04 }
