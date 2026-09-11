import MLX
import Foundation

// 使用方：H3（H3Transformer 等具体调用点如下；本文件为通用底层能力，后续其它模型可直接复用）
//   - model/minimax h3/H3Pipeline.swift:330 / :471  采样前创建缓存实例：H3AttnBroadcast(count: dit.blocks.count)（pabK > 1 时）
//   - model/minimax h3/H3Transformer.swift:882  H3DiT 层内以形参 attnBcast: H3AttnBroadcast? 接入跨步复用

// MARK: - Attention broadcast cache (PAB-style, 独立文件)
//
// 从 H3Transformer.swift / H3Layout.swift 抽出集中维护的「跨去噪步 attention 缓存」。
//
// 语义澄清（勿与 sparse 折叠混淆）：
// - 这里复用的是**每层 attention 分支的输出张量 at**（计算复用，非画面帧复用）；
// - 缓存发生在**去噪步之间**（step i → step i+1），不是视频帧之间；
// - 官方 sparseAttend 折叠稀疏（花屏证伪过）与 PAB 是两件独立的事，两者互不替代：
//   非刷新步直接复用 at，跳过 norm1+qkv+rope+attn 整段；刷新步仍走当前 sparsePolicy。
//
// ⚠️ MLX.compile 纯函数约束：本类持有跨步可变状态（blocks 为 Swift 引用），
// 若把 H3DiT.forward 整体包进 MLX.compile 闭包会破坏纯函数性导致失效/重编译；
// 需要编译图时应把缓存移出被编译闭包（或改由数组返回值传递），参考 makeCompiledDitForward。

// MARK: - 缓存容器

/// PAB 跨步 attention 缓存：blocks[bi] 持有第 bi 层的 attention 分支输出。
/// 接入方在采样循环外创建（count 取 DiT 层数，如 dit.blocks.count），
/// 循环内按 attnBroadcastRefresh 决定每步 refresh/reuse 后传入 forward。
public final class H3AttnBroadcast {
    public var blocks: [MLXArray?]

    public init(count: Int) {
        self.blocks = Array(repeating: nil, count: count)
    }

    public func reset() {
        for i in blocks.indices { blocks[i] = nil }
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
