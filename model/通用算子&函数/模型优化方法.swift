
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

/// 全局 MLX 编译总开关：默认开启（true）。
/// 背景：编译图把单步前向做成一个 C 函数统一调度，中间张量按图管理，采样瞬时峰值远低于 eager；
/// 实测关闭编译后 1080p 点击生成瞬间瞬时活集超过物理内存，触发大量交换/解压。故恢复编译保证峰值稳定。
/// 代价：首次编译 30-90s + 编译图常驻内存；TeaCache 跳步对命中步无编译收益，但真实重前向仍显著省内存。
/// 编译代码与 CompiledForwardCache / MLX.compile 分支全部保留，未删除任何能力：
/// 把本开关改为 false 即全局切 eager（LTX / HiDream 所有编译前向一次全量关闭）。
var mlxCompileEnabled: Bool = true

import MLXNN

// MARK: - TeaCache（训练无关特征缓存）

/// TeaCache：扩散去噪相邻步的输入几乎相同（sigma 差极小）时，
/// 直接复用上一步的 x0 预测，跳过整次模型前向，训练无关（无需任何训练）。
///
/// 原理：去噪过程中相邻两步的画面/timestep 条件只差一点，
/// 但每步仍会把全部 DiT 层重算一遍。TeaCache 用 sigma 差做启发式：
/// |σ_i − σ_{i−1}| < threshold → 直接沿用上一步算好的 x0，本步不调用模型。
/// 命中时 euler 步照常推进 latent（用旧 x0 推进一步），语义与官方 TeaCache 一致。
///
/// 适用：LTX-2.5（蒸馏固定 sigma 表前几步间隔 0.00625，可跳过约一半步数）、
/// HiDream 等图像模型（可只用单侧 latent，传 a 侧为 nil 的占位）。
final class TeaCache {
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

    /// 当前驻留槽数（监控面板用）。
    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.count
    }

    var isEmpty: Bool { count == 0 }
}

// MARK: - 编译专用大栈（仅包 MLX.compile 图构建）

/// 把一段同步体放到 64MB 大栈线程上执行，返回结果。
/// 唯一用途：MLX.compile 构建/析构大计算图时 C++ 深递归，
/// 在 Swift cooperative 线程（512KB 栈）上会爆栈（EXC_BAD_ACCESS code=2）。
/// 只包"编译+预热"那一下；构造完成后编译对象本身线程安全，
/// 采样等运行时回调仍在普通栈执行，不再碰大栈。
private final class BigStackBox<T> {
    var result: T? = nil
}

@discardableResult
func withBigStack<T>(_ body: @escaping () -> T) -> T {
    let box = BigStackBox<T>()
    let sem = DispatchSemaphore(value: 0)
    let thread = Thread {
        box.result = body()
        sem.signal()
    }
    thread.name = "mlx-compile-bigstack"
    thread.stackSize = 64 * 1024 * 1024
    thread.start()
    sem.wait()
    return box.result!
}

// MARK: - 通用编译前向缓存 + 流程骨架

/// 通用编译前向缓存：统一「命中打印 / 编译计时 / 预热 / LRU 存取」骨架。
/// 模型无关：Owner 为宿主（DiT/Backbone 等 Module），F 为编译闭包类型。
/// 预热由模型侧决定（预热需真实输入，无法在此层通用模拟），因此 build 闭包
/// 内部自行完成「构造编译闭包 + 跑一次预热 eval」，本类只负责 取/存/打点。
final class CompiledForwardCache<Owner: AnyObject, F> {
    private let graphs = CompiledGraphCache<Owner, F>()

    /// 取编译前向。未命中时调用 build() 构造并预热，再存入后返回。
    /// - Parameters:
    ///   - owner: 编译图宿主的强持有者（也是 LRU owner 弱引用对象）
    ///   - key: shapeKey（调用方保证前缀区分 T2I/Edit/stage）
    ///   - build: 模型侧提供的构造闭包（含 MLX.compile + 一次预热 eval），仅在未命中时调用
    func resolve(owner: Owner, key: String, _ build: @escaping () -> F) -> F {
        if let f = graphs.get(owner: owner, key: key) {
            print("复用已编译前向（shape 不变，跳过编译）")
            monitorLog("复用已编译前向（shape 不变，跳过编译）")
            return f
        }
        print("MLX.compile 编译前向中（首次约 30-90s）...")
        monitorLog("MLX.compile 编译前向中（首次约 30-90s）...")
        let tC = Date()
        // 编译构建（MLX.compile + 预热 eval）是唯一深递归段，放 64MB 大栈执行防止爆栈；
        // 编译对象线程安全，构建完成后运行时回调仍在普通栈，不再碰大栈。
        let f = withBigStack { build() }         // build 内部含预热：构造后立刻用同 shape 真输入 eval
        let warm = Date()
        let doneMsg = "✅ 编译完成（\(Int(warm.timeIntervalSince(tC)))s，预热 \(Int(Date().timeIntervalSince(warm)))s）"
        print(doneMsg)
        monitorLog(doneMsg)
        graphs.store(f, owner: owner, key: key)
        return f
    }

    /// 卸载全部编译图（释放闭包对权重的强捕获，供廉价设备动态卸载用）。
    func clear() {
        graphs.clear()
    }

    /// 当前是否已驻留编译图（监控面板权重加载状态用）。
    var isCompiled: Bool { !graphs.isEmpty }
}

// MARK: - TeaCache 单输出模型入口扩展

extension TeaCache {
    /// 单输出模型（HiDream 等图像模型，x0 只有单侧 latent）的每步入口：
    /// 返回命中的 x0；nil 表示未命中，调用方须走前向并调 storeSingle 回存。
    func next(sigma: Float) -> MLXArray? {
        var v: MLXArray? = nil
        var a: MLXArray? = nil
        guard tryReuse(sigma: sigma, outV: &v, outA: &a) else { return nil }
        return v
    }

    /// 单输出模型的回存（v/a 同位，a 侧自占位）。
    func storeSingle(_ x: MLXArray) {
        store(v: x, a: x)
    }
}


