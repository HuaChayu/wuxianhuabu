//
//  SelfLiftH3-(h3专属).swift
//  无限画布 — model/minimax h3/
//
//  H3 一采的 SelfLift Runner：把 model/3模块/SelfLift通用/SelfLift渐进采样.swift
//  的通用核心接到 H3 FL2VA 的 packed-rows 一采上。
//
//  ── 语义 ──────────────────────────────────────────────────────────────
//  取代原 stage1 的单条循环，改为：
//      低分（gridH·lowResScale）跑 ts 次 NFE
//        → 过渡块（零 NFE：Eq.3 端点预测 → 提升 → 一致修正 → 按 σ_next=σ_k 重加噪）
//        → 高分（原 gridH）跑 N-ts 次 NFE
//  对外分辨率与调用前完全一致，因此 **不需要** 改动 stage2 / 直出路径的 layout 假设。
//
//  ── 零 NFE 的落点 ─────────────────────────────────────────────────────
//  低分最后一次前向照常执行（它提供 Eq.3 需要的 v），但**不执行其 Euler 更新**；
//  过渡块用提升后的 x0 重建切换点状态：按 σ_next=σ_k 重加噪（等深，不补 Euler），直接进高清首步。
//  低分 NFE = ts，高分 NFE = N-ts，总和不变。
//
//  ── 调度 ─────────────────────────────────────────────────────────────
//  唯一入口 makeFixedFinalStep(totalSteps:sigmaK:renoiseSigma:cleanEndpoint:shift:)：
//  低清按公式网格跑到 σ_k，高清固定最后 1 步 σ_ren → 0，总 NFE 恒为 N。
//  其余参数对齐官方 SelfLiftH3Sampler 默认：rho=0.0、w_min=w_max=1.0、lowres_scale=0.5。
//
//  ── 插值口径（照抄官方，禁止自创）──────────────────────────────────────
//  低分条件行降采样：官方 `_resize_keyframes` = 逐帧 bilinear（align_corners=False）
//      + 逐 (帧, 通道) 加性均值对齐。
//  直接 latent 提升：learned upscaler（时间维感知 ×2），不可用即报错终止。
//  像素锚点重建：官方 `_pixel_anchor_video_single` = decode → bicubic(antialias) → encode。
//  以上三处均由本文件内 SelfLiftResample 表驱动实现（构造表已与 PyTorch 逐点对拍）。
//
//

import Foundation
import MLX
import MLXRandom

/// 只读诊断：进程 phys_footprint（MB）。取法与 H3Pipeline.memFootprintKB 同源
/// （task_info / TASK_VM_INFO），不 eval、不改池、不动 cacheLimit，纯读取。
private func slPhysFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { p in
        p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ip, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1024.0 / 1024.0 : 0
}

/// 低分 / 高分段的只读内存打点（过渡块以外）：与 Runner 内的 slDiag 同源，但标签自带阶段前缀，
/// 并额外附进程 RSS（phys_footprint）—— 只有把「MLX 池内(active+cache)」与「池外/进程整体」
/// 并列，才能判定 swap 里哪部分是 MLX 张量、哪部分是池外常驻。纯读取：不 eval、不改池、不动 cacheLimit。
private func slMem(_ label: String) {
    pipelineLog("[H3] [MEM] SelfLift \(label)：RSS \(Int(slPhysFootprintMB()))MB，MLX active \(MLX.Memory.activeMemory / 1_000_000)MB，cache \(MLX.Memory.cacheMemory / 1_000_000)MB")
}

// MARK: - 宿主钩子（H3 侧注入，避免 Runner 依赖 H3VAE 内部结构）

public struct SelfLiftH3Hooks {
    /// 归一化 latent NCDHW → 像素 [1,3,frames,H*16,W*16]（= H3VAEDecoder.decode）
    public var decodeToPixels: (MLXArray) -> MLXArray
    /// 像素 [1,3,frames,H,W] → 归一化 latent NCDHW（= H3VAEEncoder.encodeVideo）
    public var encodeToLatent: (MLXArray) -> MLXArray
    /// 过渡块出口屏障（已 `Stream.gpu.synchronize()`）之后回调一次，让调用方在此把
    /// **只服务过渡块**的像素锚点 VAE（decoder + encoder，bf16 实测 ≈5.3GB）置 nil。
    /// 高分段的 NFE 一次都不用它的权重，却要按常驻权重一路挂到 Runner 返回（≈高分全段），
    /// 早放可把这 5.3GB 从高分段基线里摘掉；不传 = 保持原行为（Runner 返回后才由调用方释放）。
    public var releasePixelVAE: (() -> Void)?

    public init(decodeToPixels: @escaping (MLXArray) -> MLXArray,
                encodeToLatent: @escaping (MLXArray) -> MLXArray,
                releasePixelVAE: (() -> Void)? = nil) {
        self.decodeToPixels = decodeToPixels
        self.encodeToLatent = encodeToLatent
        self.releasePixelVAE = releasePixelVAE
    }
}

// 官方 learned latent upscaler（时间维感知）的按需加载：以 NA_H3_UPSCALER 环境变量当前值为键缓存。
// 默认 → H3LatentUpscaler（官方 learned upscaler，1344×768 实测显著消除面部重影）；
// 权重不可用时直接抛错终止，不降级。
private var h3LearnedUpscalerCache: (key: String, up: H3LatentUpscaler)? = nil
private func h3LearnedUpscalerForCurrentEnv() throws -> H3LatentUpscaler {
    let key = ProcessInfo.processInfo.environment["NA_H3_UPSCALER"] ?? ""
    if let c = h3LearnedUpscalerCache, c.key == key { return c.up }
    let modelDir = "\(CommonPaths.modelRoot)/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit"
    let modelRootURL = URL(fileURLWithPath: modelDir, isDirectory: true)
    let wURL = { (name: String) -> URL in
        let direct = modelRootURL.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: direct.path) { return direct }
        return FileFinder.first(named: name, under: modelRootURL) ?? direct
    }
    let path = ProcessInfo.processInfo.environment["NA_H3_UPSCALER_PATH"]
        ?? wURL("minimax_h3_latent_upscaler_3d_bf16.safetensors").path
    guard let w = try? H3Weights(url: URL(fileURLWithPath: path)) else {
        throw NSError(domain: "H3SelfLift", code: 41, userInfo: [NSLocalizedDescriptionKey:
            "learned upscaler 不可用（权重加载失败：\(path)），SelfLift 终止"])
    }
    let up: H3LatentUpscaler
    do {
        up = try H3LatentUpscaler(weights: w)
    } catch {
        throw NSError(domain: "H3SelfLift", code: 42, userInfo: [NSLocalizedDescriptionKey:
            "learned upscaler 权重解析失败（\(path)）：\(error.localizedDescription)，SelfLift 终止"])
    }
    h3LearnedUpscalerCache = (key, up)
    return up
}

// MARK: - 局部工具

/// [T,H,W,C] → [1,C,T,H,W]
@inline(__always)
private func toNCDHW(_ nhwc: MLXArray, c: Int) -> MLXArray {
    nhwc.transposed(3, 0, 1, 2).expandedDimensions(axis: 0)
}

/// [1,C,T,H,W] → [T,H,W,C]
@inline(__always)
private func fromNCDHW(_ ncdhw: MLXArray, c: Int) -> MLXArray {
    // 【崩溃根因修复】旧实现 `ncdhw[0]`（整数下标 0，axis=0）在 src 非 row_contiguous
    // （learned upscaler 输出是 transposed 视图）时，mlx 的 take 走 gather 通用 kernel 编码
    // 路径（Gather::eval_gpu，indexing.cpp:39）：对 0 维标量 indices 的 idx_shapes/idx_strides
    // 空 vector 无条件调 compute_encoder.set_vector_bytes → setBytes(vec.data(), 0)，
    // 空 vector 的 data() 为 nullptr → Metal 调试层断言「setBytes: bytes argument cannot be nil」，
    // 经 MLX errorHandler 升级为 Swift assertionFailure（SIGTRAP），崩溃点正是 4.4 eval(videoX)
    // 结算该 take 的时刻（截图堆栈 runH3Stage1WithSelfLift → Gather::eval_gpu 完全吻合）。
    // 修复：改为范围切片 [0 ..< 1] + reshape 剥离 batch 维 —— 语义与 `ncdhw[0]` 完全等价
    // （batch=1），但 Slice/reshape 编码路径不产生空 vector setBytes，图中不再出现 Gather 节点。
    let noBatch = ncdhw[0 ..< 1]
        .reshaped(ncdhw.shape[1], ncdhw.shape[2], ncdhw.shape[3], ncdhw.shape[4])
    return noBatch.transposed(1, 2, 3, 0)
}

/// 3 抽头 [1,2,1]/4 一维模糊（边界复制），供低频锚定的频带分离使用。
private func slBlur1D(_ x: MLXArray, axis: Int) -> MLXArray {
    let n = x.shape[axis]
    let prev = MLXArray((0..<n).map { Int32(max($0 - 1, 0)) }, [n])
    let next = MLXArray((0..<n).map { Int32(min($0 + 1, n - 1)) }, [n])
    let a = x.take(prev, axis: axis)
    let b = x.take(next, axis: axis)
    return (a + x * 2 + b) * 0.25
}

/// 只沿 H/W（axis 3/4）跑两趟 3 抽头模糊 ≈ 高斯 σ≈1.1 latent px（= 8.8 输出像素）的低通。
/// 时间轴刻意不动：时间维低通会给运动加拖影，低频锚定要的是「逐帧的空间低频」。
private func slLowBandBlur(_ x: MLXArray) -> MLXArray {
    var y = x
    for _ in 0..<2 {
        y = slBlur1D(y, axis: 3)
        y = slBlur1D(y, axis: 4)
    }
    return y
}


// MARK: - 官方重采样内核（照抄 selflift.py / nodes.py 的插值调用）

/// PyTorch `F.interpolate` 的表驱动等价实现。
///
/// 官方四处调用里，本工程 H3 一采用到其中三处，逐条照抄、不换自创插值：
///   · 直接 latent 提升        learned upscaler（时间维感知 ×2）      — 见 slLearnedLiftLatent
///   · 像素锚点重建            `mode="bicubic", antialias=True`       — selflift.py 行 95
///   · 条件行（keyframe）降采样 `mode="bilinear", align_corners=False` — nodes.py 行 191-198
///
/// 表构造已与 PyTorch 逐点对拍（bicubic antialias 与 bilinear 非 antialias 均 max|diff| ≈ 1e-7）。
enum SelfLiftResample {

    /// 一维重采样表：`idx[t][i]` 为第 t 个抽头的输入下标，`wts[t][i]` 为其权重；输出长度 = `wts[0].count`
    struct Table {
        let taps: Int
        let idx: [[Int]]
        let wts: [[Float]]
    }

    /// `mode="nearest"`：`srcIdx = floor(i · in/out)`（PyTorch nearest 的下标规则）
    static func nearestTable(inSize: Int, outSize: Int) -> Table {
        let scale = Double(inSize) / Double(outSize)
        var idx = [Int](repeating: 0, count: outSize)
        for i in 0..<outSize {
            idx[i] = min(inSize - 1, max(0, Int((Double(i) * scale).rounded(.down))))
        }
        return Table(taps: 1, idx: [idx], wts: [[Float](repeating: 1, count: outSize)])
    }

    /// `mode="bilinear"` / `"trilinear"` 的一维分解：align_corners=False、antialias=False（2 抽头）
    static func linearTable(inSize: Int, outSize: Int) -> Table {
        let scale = Double(inSize) / Double(outSize)
        var i0s = [Int](repeating: 0, count: outSize)
        var i1s = [Int](repeating: 0, count: outSize)
        var w0s = [Float](repeating: 0, count: outSize)
        var w1s = [Float](repeating: 0, count: outSize)
        for i in 0..<outSize {
            let real = scale * (Double(i) + 0.5) - 0.5
            let base = Int(real.rounded(.down))
            let lam = Float(real - Double(base))
            i0s[i] = min(max(base, 0), inSize - 1)
            i1s[i] = min(base + 1, inSize - 1)
            w0s[i] = 1 - lam
            w1s[i] = lam
        }
        return Table(taps: 2, idx: [i0s, i1s], wts: [w0s, w1s])
    }

    /// `mode="bicubic", antialias=True` 的一维分解
    /// （PIL 式 filterscale + 逐输出归一，与 PyTorch 的抗锯齿实现一致）
    static func bicubicAATable(inSize: Int, outSize: Int) -> Table {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let taps = Int(support.rounded(.up)) * 2 + 1
        let a = -0.5                                        // PyTorch bicubic 的 a
        var idx = [[Int]](repeating: [Int](repeating: 0, count: outSize), count: taps)
        var wts = [[Float]](repeating: [Float](repeating: 0, count: outSize), count: taps)
        for i in 0..<outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize)
            let n = max(0, xmax - xmin)
            var w = [Double](repeating: 0, count: n)
            var sum = 0.0
            for j in 0..<n {
                let x = (Double(j + xmin) + 0.5 - center) / filterscale
                let ax = abs(x)
                let v: Double
                if ax < 1.0 {
                    v = (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1
                } else if ax < 2.0 {
                    v = a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a
                } else {
                    v = 0
                }
                w[j] = v
                sum += v
            }
            for j in 0..<taps {
                if j < n, sum != 0 {
                    idx[j][i] = xmin + j
                    wts[j][i] = Float(w[j] / sum)
                } else {
                    // 越界抽头：下标夹回范围内、权重置 0（官方实现同此处理）
                    idx[j][i] = n > 0 ? xmin + n - 1 : 0
                    wts[j][i] = 0
                }
            }
        }
        return Table(taps: taps, idx: idx, wts: wts)
    }

    /// 沿单个轴按表重采样（taps 个抽头加权求和，等价于对其它所有轴做广播）
    ///
    /// - Parameter keepAlive: 可选强引用池，**只挂「下标表 / 权重表」这类 KB 级小张量**：
    ///   它们是按「表」而非按「帧」分配的小尺寸对象，且被懒图的 `take`/乘加引用，
    ///   必须在 GPU 消费完之前保持强引用（否则可能触发
    ///   `object destroyed while still required by command buffer`）。
    ///   **不要再挂 `term`（`x.take(...) * wT` 的乘加中间量）**：它按输出形状分配，
    ///   全帧 × 抽头合起来是 GB 级；挂住会让「上采样整段的中间量」一直存活到过渡块末尾，
    ///   把 active 峰值推过 MLX 分配器的回收阈值（实测 B 步 active 一次 +13.4GB 的主要来源）。
    ///   中间量在求值后自然进入空闲池，跨块复用由分配器决定，无需（也不应）由上层挂引用。
    static func resizeAxis(_ x: MLXArray, axis: Int, table: Table,
                           keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
        var outShape = [Int](repeating: 1, count: x.ndim)
        outShape[axis] = table.wts[0].count
        var acc: MLXArray? = nil
        for t in 0..<table.taps {
            let w = table.wts[t]
            if w.allSatisfy({ $0 == 0 }) { continue }
            let idxT = MLXArray(table.idx[t])
            let wT = MLXArray(w).reshaped(outShape).asType(x.dtype)
            keepAlive?.hold([idxT, wT])
            let term = x.take(idxT, axis: axis) * wT
            acc = acc.map { $0 + term } ?? term
        }
        return acc ?? x
    }
}

/// 官方低分空间尺寸：`h = max(2, round(H · lowres_scale / 2) · 2)`（nodes.py 行 259-260）
@inline(__always)
private func slLowResSize(_ target: Int, scale: Double) -> Int {
    max(2, Int((Double(target) * scale / 2).rounded()) * 2)
}

/// [T,H,W,C] → [T,outH,outW,C]：官方 learned upscaler 直接 latent 提升
/// （时间维感知 ×2，已在 1344×768 实测显著消除面部重影；目标半尺寸为奇数时 ×2 后与
/// 目标不一致，补一次最近邻 resize 对齐，与官方 old 路径同口径）。
private func slLearnedLiftLatent(_ x: MLXArray, outH: Int, outW: Int,
                                 keepAlive: SelfLiftTensorKeepAlive? = nil) throws -> MLXArray {
    let us = try h3LearnedUpscalerForCurrentEnv()
    let zIn = toNCDHW(x, c: x.shape[3])                                     // [1,C,T,h,w]
    let zUp = us.apply(zIn)                                                 // [1,C,T,H,W]（×2）
    var up = fromNCDHW(zUp, c: x.shape[3])                                  // [T,H,W,C]
    if up.shape[1] != outH || up.shape[2] != outW {
        let h = SelfLiftResample.resizeAxis(
            up, axis: 1, table: SelfLiftResample.nearestTable(inSize: up.shape[1], outSize: outH),
            keepAlive: keepAlive)
        up = SelfLiftResample.resizeAxis(
            h, axis: 2, table: SelfLiftResample.nearestTable(inSize: up.shape[2], outSize: outW),
            keepAlive: keepAlive)
        keepAlive?.hold([h, up])
    }
    keepAlive?.hold([up])
    return up
}

/// 条件行（keyframe）降采样：官方 `nodes.py` 的 `_resize_keyframes`（行 174-208）——
/// 逐帧 bilinear（align_corners=False）＋ 逐 (帧, 通道) 的**加性**均值对齐；尺寸已一致时原样返回。
private func slResizeKeyframeLatent(_ x: MLXArray, outH: Int, outW: Int,
                                    keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
    let s = x.shape                                          // [T,H,W,C]
    if s[1] == outH && s[2] == outW { return x }
    let h = SelfLiftResample.resizeAxis(
        x, axis: 1, table: SelfLiftResample.linearTable(inSize: s[1], outSize: outH),
        keepAlive: keepAlive)
    let resized = SelfLiftResample.resizeAxis(
        h, axis: 2, table: SelfLiftResample.linearTable(inSize: s[2], outSize: outW),
        keepAlive: keepAlive)
    // 官方：`kf.latent = resized + (source_mean - resized_mean)`，均值沿 (H, W) 逐帧逐通道
    let srcMean = x.mean(axis: 2, keepDims: true).mean(axis: 1, keepDims: true)
    let dstMean = resized.mean(axis: 2, keepDims: true).mean(axis: 1, keepDims: true)
    let out = resized + (srcMean - dstMean)
    keepAlive?.hold([h, resized, srcMean, dstMean, out])
    return out
}

/// 像素锚点重建的上采样：官方 `selflift.py` 的 `_pixel_anchor_video_single`（行 87-98）——
/// `F.interpolate(size=(H·ratio, W·ratio), mode="bicubic", antialias=True)`，逐帧独立。
/// 官方按 32 帧分块（GPU 显存考虑）；这里分块更小以控制统一内存峰值，逐帧独立故结果完全一致。
///
/// ★ 分块**逐块立即求值**：本函数是过渡块 active 峰值（B 步）的唯一来源，两条规矩别破：
///   ① 分块要小（chunk=2：每块在飞的抽头中间量 ≈100MB 级，能被空闲池当场复用）；
///   ② 每块 `MLX.eval(piece)` 后**不要**把 `part/a/piece` 挂进 keepAlive —— 那等于把
///      全段上采样的中间量一路钉到过渡块末尾（本轮实测：B 步 active 一次 +13.4GB）。
private func slResizePixelsBicubicAA(_ px: MLXArray, outH: Int, outW: Int,
                                     keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
    let s = px.shape                                         // [1,3,F,H,W]
    if s[3] == outH && s[4] == outW { return px }
    let tableH = SelfLiftResample.bicubicAATable(inSize: s[3], outSize: outH)
    let tableW = SelfLiftResample.bicubicAATable(inSize: s[4], outSize: outW)
    let chunk = 2
    var pieces: [MLXArray] = []
    var start = 0
    while start < s[2] {
        let end = min(start + chunk, s[2])
        let rangeT = MLXArray(Array(start..<end))
        let part = px.take(rangeT, axis: 2)                                // [1,3,chunk,H,W]
        let a = SelfLiftResample.resizeAxis(part, axis: 3, table: tableH, keepAlive: keepAlive)
        let piece = SelfLiftResample.resizeAxis(a, axis: 4, table: tableW, keepAlive: keepAlive)
        // ★ 分块物化点：本块的懒图到此被消费掉（MLX 求值后 primitive 清空），
        //   抽头中间量当即回到空闲池可复用；分块结果只由 pieces 数组引用，
        //   整段 eval 完（B 点）后由拼接结果取代、随即释放。
        MLX.eval(piece)
        pieces.append(piece)
        start = end
    }
    return pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 2)
}

// MARK: - Runner

/// H3 一采 + SelfLift 渐进式分辨率切换。
/// 返回切换后（= 原分辨率）跑完的 `videoX` / `audioX`，可直接接回既有 stage2 / 直出路径。
///
/// - Parameters:
///   - latT/latH/latW/latC: 目标（高分）latent 尺寸与通道数；`gridH = latH/2`、`gridW = latW/2`
///   - condRowsFull: 全分辨率条件行（首/尾帧 keyframe，已 patchify）；Runner 内部按 lowResScale 降采样出低分版本
///   - refBlocks: ref2va 参考块（= 原生 layout 同一份，request 顺序；fl2va 传空 `[]`）
///   - textTags: ref2va 文本行 AdaLN 模态标签（= H3TextEncoder 编码结果的 tags；fl2va 传空 `[]`）
///   - sigmas: 原始一采调度（非增、末项为 0）
///   - attentionBroadcastK: PAB 的 k（≤1 表示不用）
public func runH3Stage1WithSelfLift(
    dit: H3DiT,
    textStates: MLXArray,
    condRowsFull: MLXArray,
    // ★ 2026-10-19 原生半清条件行（源图阶段多编一份 0.5 像素 → 直喂低清段）：
    //   非 nil 时低分条件行不再由 full-res latent 在 latent 域 bilinear 降采样
    //   （官方 `_resize_keyframes` 旧路径保留为回退：含 history video 块 / 行数不匹配时自动走旧路）。
    condRowsLowOverride: MLXArray? = nil,
    refBlocks: [RefBlock] = [],
    textTags: [UInt8] = [],
    latT: Int,
    latH: Int,
    latW: Int,
    latC: Int,
    audioT: UInt32,
    textLen: UInt32,
    frameCount: UInt32,
    sigmas: [Double],
    shiftV: Double,
    shiftA: Double,
    seed: UInt64,
    sparsePolicy: SparsePolicy,
    attentionBroadcastK: UInt32,
    cfg: SelfLiftConfig,
    hooks: SelfLiftH3Hooks,
    log: (String) -> Void,
    // ★ lowOnly 模式（IC 链路用）：true = 只跑低分（半清）段，低清段结束时直接返回
    //   低清 NCDHW latent [1,C,T,Hl,Wl]（跳过过渡块 + 高分循环）；高分（升频×2 + 精修）
    //   交由 LTX 二采完成——「H3 低分 + LTX 高分」两个模型组成 lift，而非 H3 自己 lift。
    lowOnly: Bool = false,
    // ★ CFG 引导（2026-09-18 重影根因修复）：cfg.cfgScale > 0 时启用双路引导。
    //   textStatesNeg 为「negative 条件行」，调用方必须已 pad 到与 positive 相同行数
    //   （= textLen），使 negative forward 可复用同一 layout/rope/plan —— 文本行数的任何
    //   差异都会平移 video/audio 行的时间坐标（PackedLayout cursor 从 textLen 起算），
    //   CFG 两路将不在同一坐标系，合成会失真。
    textStatesNeg: MLXArray? = nil,
    // ★ 续接激活标志：continuationWinLatentT=0 表示非续接，保持原纯噪声初值行为。
    //   2026-10-03 起续接初值路线已删（前置尾段走 contKeyAnchors 条件行锚/参考块，
    //   videoX 恒为本节点新段纯噪声起步）。
    continuationWinLatentT: UInt32 = 0,
    // ★ 续接 keyframe 锚点（2026-09-21 ComfyUI 路线）：前置尾段 latent 作为 never-denoised
    //   条件行锚进新时间轴开头。fl2va 时低清/高清布局按锚点数注入 keyframe 段（切分随
    //   fRows 自动包含，每锚点一段）；ref2va 时为空（参考图约束）。空数组 = 无前置锚点。
    contKeyAnchors: [KeyframeAnchor] = [],
    // 续接前置条件行偏移：condRowsFull 段序 = [前置cond] + [本节点keyframe(fl2va)] + [前置refImg] + [本节点refImg]。
    //   preCondRows = 前置 cond 段行数（仅用于跳过该段，本 Runner 低清/高清条件行只注入参考块与
    //   本节点 keyframe，前置内容由 videoX 前缀承载）；preRefBlocks = 前置 refImg 块，排在 refBlocks
    //   前与 condRows 段序一致（ref2va 续接时合并切行）。0 / 空数组 = 无该段。
    preCondRows: UInt32 = 0,
    preRefBlocks: [RefBlock] = [],
    // ★ 续接音频（2026-09-21，与视频 keyframe 锚点对称）：前置尾段音频 latent 行
    //   [winAudioT*2, 32]（clean，float32/16 均可）。非 nil 时作为 refAudio 段（audioUpdate=false
    //   → never-denoised）锚进 audio stream 开头，模型注意力能看到真实前置音频；
    //   采样更新只作用于本节点行（out.audio 只切目标 audioSegment），前置行不被加噪/改写。
    //   nil = 旧缓存或无音频，audioX 回退纯噪声占位（原路线）。
    audioContinuationRows: MLXArray? = nil,
    // ★ 2026-10-05 fl 锚几何显式同口径（方案A2 几何对齐）：fl 场景调用方（H3Pipeline）把
    //   高分锚几何（keyframeSpans=每锚 fRows、keyframeGrids=目标网格坐标）显式传下；
    //   SelfLift 高分 layout 直接用，低分 layout 按目标网格折算低清等价（span=gHl*gWl/锚、
    //   grid=低清网格=目标网格 0.5 降采样），两阶段「同一个注入」走同一套几何参数机制。
    //   ref+续接（contKeyAnchors=[.first]）不传 → 两布局均 nil 默认，行为完全不变。
    keyframeSpansHigh: [UInt32]? = nil,
    keyframeGridsHigh: [(h: [Double], w: [Double])?]? = nil
) throws -> (videoX: MLXArray, audioX: MLXArray) {

    // 唯一调度：低清按公式网格跑到 σ_k → 高清固定最后 1 步 σ_ren → 0（总 NFE 恒为 N）。
    //   σ_k 默认 0.75（NA_H3_SELFLIFT_DECOUPLE_LOW_K 可覆盖）；σ_ren 缺省 clean 端点 0.75（与止点等深）。
    //   cleanEndpoint：默认交 clean x0（升频后按 σ_ren 重加噪）；带噪直通档跳过 Eq.3、交 σ_k 处含噪 x。
    //   lowOnly（一采只出半清 latent 给 LTX 二采）需要 clean 端点作为下游输入，强制保留。
    let useCleanEndpoint = cfg.cleanEndpointHandoff || lowOnly
    let split: SelfLiftScheduleSplit
    split = try SelfLiftScheduleSplit.makeFixedFinalStep(
        totalSteps: sigmas.count - 1,
        // ★ 用户定档：低清止点 σ_k = 重加噪 σ_ren = 0.75（两者等深）。
        sigmaK: cfg.decoupleSigmaK ?? 0.75,
        renoiseSigma: cfg.renoiseSigma,
        cleanEndpoint: useCleanEndpoint,
        // ★ 2026-10-25：低清网格改为公式生成，shift 必须与直出路径同一档（视频 shiftV）。
        shift: shiftV)
    // ★ 音频调度解耦（2026-10-16）：音频行改用 H3 原生音频调度 sigmaSchedule(总NFE, shiftA)，
    //   与视频侧提升点 sigma_k 解耦。低清段取前 lowNFE+1 项、高分段取自 lowNFE 起的 highNFE+1 项，
    //   逐位喂给 buildTimestepPlan(audioSigma:) 与 da 的 delta_sigma_a 计算。
    //   修复动机：sigma_k→1 时旧口径（音频随视频 sigma_v 经 timeShiftSigma 换算）低清单步仅覆盖
    //   音频行程约 5%，其余约 78% 压进高分单步 Euler → 跨度过大导致削波与高频崩坏。
    let audioSigmas = sigmaSchedule(steps: UInt32(split.lowNFE + split.highNFE), shift: shiftA)
    let audioSigmasLow = Array(audioSigmas[0...split.lowNFE])
    let audioSigmasHigh = Array(audioSigmas[split.lowNFE...])
    log("SelfLift 音频调度解耦：总 NFE=\(split.lowNFE + split.highNFE) shiftA=\(String(format: "%.4f", shiftA))，"
        + "低清 sigma_a \(String(format: "%.4f", audioSigmasLow[0]))→\(String(format: "%.4f", audioSigmasLow[split.lowNFE]))，"
        + "高分 \(String(format: "%.4f", audioSigmasHigh[0]))→\(String(format: "%.4f", audioSigmasHigh[split.highNFE]))")
    let lowRunsToZero = split.sigmaK <= 1e-9
    let sigmaKInput = cfg.decoupleSigmaK ?? 0.75
    let sigmaRenInput = cfg.renoiseSigma ?? (useCleanEndpoint ? 0.75 : (lowRunsToZero ? 0.5 : sigmaKInput))
    log("SelfLift 公式网格（shift=\(String(format: "%.1f", shiftV))）："
        + "σ_k 输入 \(String(format: "%.4f", sigmaKInput)) → 公式最近档 \(String(format: "%.4f", split.sigmaK))；"
        + "σ_ren 输入 \(String(format: "%.4f", sigmaRenInput)) → 公式最近档 \(String(format: "%.4f", split.sigmaNext))")
    log("SelfLift 低清公式网格：L=\(split.lowNFE) 步 \(split.lowSigmas.map { String(format: "%.4f", $0) }.joined(separator: " → "))")
    log("SelfLift 用户设计调度：低清 L=\(split.lowNFE)（终点 σ_k=\(String(format: "%.4f", split.sigmaK))\(lowRunsToZero ? "，跑满完整流程" : "")）"
        + "→ 提升 → \(abs(split.sigmaNext - split.sigmaK) > 1e-9 ? "按 σ_ren=\(String(format: "%.4f", split.sigmaNext)) 重加噪" : "等深交接（σ_next=σ_k，不重加噪）")"
        + " → 高清固定 1 步 → 0；原调度 N=\(sigmas.count - 1) 已绕开")

    // CFG 引导状态（官方 SelfLiftH3Sampler 默认 cfg=5.0 全程双路；工程旧行为无 CFG 单路）。
    let cfgScale = cfg.cfgScale
    if cfgScale > 0 {
        if textStatesNeg == nil {
            log("SelfLift CFG：cfg=\(cfgScale) 但未注入 negative 条件行（textStatesNeg=nil），回退单路（与旧行为一致）")
        } else {
            log("SelfLift CFG：开启 cfg=\(cfgScale)（每步 positive+negative 双 forward；audio 不参与 CFG）")
        }
    } else {
        log("SelfLift CFG：关闭（cfg=0，仅 positive 单路）")
    }

    // 临时小张量强引用池（见 `SelfLiftTensorKeepAlive`）：低分/过渡块/高分各阶段里
    // 那些 4~几十字节的标量、索引、权重张量都挂在这里，按「阶段边界 = 一次覆盖全图的
    // MLX.eval」为界释放，保证 GPU 在飞期间它们一定有确定的宿主持有者。
    let keepAlive = SelfLiftTensorKeepAlive()

    // ── 0. 尺寸与常量 ──────────────────────────────────────────────────
    let gridH = latH / 2, gridW = latW / 2            // patch 2×2
    let vpatch = latC * 4
    let latHl = slLowResSize(latH, scale: cfg.lowResScale)
    let latWl = slLowResSize(latW, scale: cfg.lowResScale)
    let gHl = latHl / 2, gWl = latWl / 2
    // ★ 2026-10-03 fresh-noise 路线：续接初值恒 nil（前置尾段只走 contKeyAnchors 条件行锚/
    //   参考块，不拼进采样初值），videoX 恒为本节点新段纯噪声起步（latT 行）。
    let layoutLatT = UInt32(latT)
    log("SelfLift 低清段启动：低分 NFE=\(split.lowNFE) / 高分 NFE=\(split.highNFE)，"
        + "σ_k=\(String(format: "%.4f", split.sigmaK)) → σ_next=\(String(format: "%.4f", split.sigmaNext))，"
        + "latent \(latH)×\(latW) → 低分 \(latHl)×\(latWl)")

    // ── 1. 低分条件行（官方 `_resize_keyframes`：逐帧 bilinear + 加性均值对齐）──
    // 分段方式与 layout 的 cond 行预算**逐块**对齐（这是本分支的关键不变量）：
    //   · ref2va（refBlocks 非空）：每段 = 一个参考块，用**该块自己的 latent 网格**还原，
    //     低分尺寸按 cfg.lowResScale 经 slLowResSize 同口径缩放；layout 侧传同序同尺寸的 refs。
    //   · fl2va（2026-10-07 恢复双端硬锚）：fl 首尾帧由 H3Pipeline 走 keyframe 硬锚
    //     （[.first,.last]，cond 段），不占 refBlocks。
    let fRows = gridH * gridW
    // ★ 2026-10-02 段序修正（与 H3Pipeline 注入 pieces 严格一致）：注入行序 =
    //   [preCond][boundary][本节点 refImg][前置 refImg][history]，故 allRefBlocks =
    //   refBlocks + preRefBlocks（history video 块经 preRefBlocks 传递、位于最末）；
    //   此前 preRef 先切 → 与注入行序错位（参考块 latent 内容读错行区，ref2va 续接
    //   画面污染的潜在来源，本次一并修复）。
    let allRefBlocks = refBlocks + preRefBlocks

    // ★ 2026-10-19 原生半清条件行启用判定：history video 参考块无法在源图阶段预编
    //   （其 latent 来自前置 .h3cc），出现即回退旧路径；keyframe / refImg 段可整段直通。
    let nativeLowCond: MLXArray? = (condRowsLowOverride != nil
        && !allRefBlocks.contains { $0.kind == .video }) ? condRowsLowOverride : nil
    if condRowsLowOverride != nil, nativeLowCond == nil {
        log("SelfLift 原生半清条件行：检测到 history video 参考块，回退 latent 域降采样旧路径")
    }
    let condOffset = Int(preCondRows)
    var condSegs: [MLXArray] = []
    // ★ 2026-10-02/2026-10-04 语义：高分段条件行 = 原生 full-res 行直通。
    //   fl 双端硬锚已随 2026-10-07 恢复（keyframe cond 段低分重采样/高分直通见下方
    //   ★ 2026-10-07 fl 硬锚 注释）；
    //   ref2va/fl 的 boundary/history/参考图均为去噪完成的干净 full-res latent，
    //   高分段原生直通、不降画面再提升（低分段内部自行重采样处理，互不干扰）。
    var condSegsHigh: [MLXArray] = []
    var lowRefBlocks: [RefBlock] = []
    // ★ 2026-10-07 fl 硬锚+视觉块：fl 首尾帧由 H3Pipeline 改为双端 keyframe 硬锚
    //   （[.first,.last]，cond 段，不占 refBlocks），refBlocks 恒空 → 下方 for 循环零次；
    //   ref/续接仍走 allRefBlocks（ref 多参考 + 前置参考块）；锚点集合差异全由
    //   contKeyAnchors 驱动（fl=2 锚、ref 续接=[.first]）。
    //   纯文生/续接空图零条件行时下方两个 for 循环自然零次（由预算断言兜底）。
    var offset = condOffset
        // ★ 2026-09-21 修复：ref2va 前置尾段 keyframe 锚点先按 keyframe 口径切行（每段 fRows，
        //   低清重采样同 fl2va），再切参考块；段序与 H3Pipeline 注入的 [preCond][contKey][preRef][ownRef]
        //   严格对齐（此前 ref2va 只切参考块，contKey 段未计入 → 尾帧锚点丢失）。
        for (ki, _) in contKeyAnchors.enumerated() {
            guard offset + fRows <= condRowsFull.shape[0] else {
                throw NSError(domain: "H3SelfLift", code: 7, userInfo: [NSLocalizedDescriptionKey:
                    "ref2va contKey 段行数超出 condRowsFull：\(offset + fRows) > \(condRowsFull.shape[0])"])
            }
            let seg = condRowsFull[offset..<(offset + fRows)]
            offset += fRows
            if nativeLowCond == nil {
                let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gridH, gridW: gridW)   // [1,latH,latW,C]
                let low = slResizeKeyframeLatent(lat, outH: latHl, outW: latWl,
                                                 keepAlive: keepAlive)                   // [1,latHl,latWl,C]
                let rows = H3TensorOps.patchifyVideo(low, gridH: gHl, gridW: gWl)
                keepAlive.hold([lat, low, rows])
                condSegs.append(rows)
            }
            // ★ 2026-10-02 回退（用户确认）：前置尾段 latent 是去噪完成的高清 full-res 结果，
            //   高分段直接原生行直通（与低分内部处理解耦）——不再低清重采样→learned upscaler
            //   提升（多此一举且降画面，还引入 2×2 提升网格）。
            condSegsHigh.append(seg)
        }
        for b in allRefBlocks {
            // 本工程 ref2va 支持 image + video（history）参考块；audio 块无视觉行，走到这里报错。
            switch b.kind {
            case .audio:
                throw NSError(domain: "H3SelfLift", code: 3, userInfo: [NSLocalizedDescriptionKey:
                    "SelfLift ref2va 收到 audio 参考块（视觉行切分路径不应出现）"])
            case .image:
                let gh = Int(b.latentH) / Int(H3Const.patchH)
                let gw = Int(b.latentW) / Int(H3Const.patchW)
                let n = gh * gw                                   // = H3Layout.refFrameRows(b)
                guard offset + n <= condRowsFull.shape[0] else {
                    throw NSError(domain: "H3SelfLift", code: 4, userInfo: [NSLocalizedDescriptionKey:
                        "参考块行数超出 condRowsFull：\(offset + n) > \(condRowsFull.shape[0])"])
                }
                let seg = condRowsFull[offset..<(offset + n)]
                offset += n
                let lh = slLowResSize(Int(b.latentH), scale: cfg.lowResScale)
                let lw = slLowResSize(Int(b.latentW), scale: cfg.lowResScale)
                if nativeLowCond == nil {
                    let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gh, gridW: gw)         // [1,latH,latW,C]
                    let low = slResizeKeyframeLatent(lat, outH: lh, outW: lw,
                                                     keepAlive: keepAlive)                   // [1,lh,lw,C]
                    let rows = H3TensorOps.patchifyVideo(low, gridH: lh / Int(H3Const.patchH),
                                                         gridW: lw / Int(H3Const.patchW))
                    keepAlive.hold([lat, low, rows])
                    condSegs.append(rows)
                }
                // ★ 参考图行（含前置 refImg）为 VAE 编码的干净 latent，无低清 patch 网格残留，
                //   保持原生全分辨率注入高清 forward（非端点锚定，低清/高清两套对静态参考无碍）。
                condSegsHigh.append(seg)
                lowRefBlocks.append(RefBlock(kind: .image, latentH: UInt32(lh), latentW: UInt32(lw),
                                             latentT: b.latentT, audioT: b.audioT))
            case .video:
                // ★ 2026-10-02 history 视频参考块（mere-run encodeContinuation 语义）：
                //   整段 latentT 帧逐帧重采样进低清（每帧独立 bilinear，不跨帧混叠），
                //   行序拼接为一段（与布局 refs .video 段一致）；低清块 latentT 保持。
                let gh = Int(b.latentH) / Int(H3Const.patchH)
                let gw = Int(b.latentW) / Int(H3Const.patchW)
                let n = Int(b.latentT) * gh * gw
                guard offset + n <= condRowsFull.shape[0] else {
                    throw NSError(domain: "H3SelfLift", code: 4, userInfo: [NSLocalizedDescriptionKey:
                        "video 参考块行数超出 condRowsFull：\(offset + n) > \(condRowsFull.shape[0])"])
                }
                let lh = slLowResSize(Int(b.latentH), scale: cfg.lowResScale)
                let lw = slLowResSize(Int(b.latentW), scale: cfg.lowResScale)
                var segs: [MLXArray] = []
                var segsHigh: [MLXArray] = []   // ★ 2026-10-02 回退：history 高分段 = 原生 full-res 行直通
                for f in 0..<Int(b.latentT) {
                    let seg = condRowsFull[(offset + f * gh * gw)..<(offset + (f + 1) * gh * gw)]
                    let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gh, gridW: gw)     // [1,latH,latW,C]
                    let low = slResizeKeyframeLatent(lat, outH: lh, outW: lw,
                                                     keepAlive: keepAlive)               // [1,lh,lw,C]
                    let rows = H3TensorOps.patchifyVideo(low, gridH: lh / Int(H3Const.patchH),
                                                         gridW: lw / Int(H3Const.patchW))
                    keepAlive.hold([lat, low, rows])
                    segs.append(rows)
                    // ★ 2026-10-02 回退（用户确认）：history 整段为干净 full-res latent，
                    //   高分段原生直通，不逐帧低清重采样→提升（与 boundary 语义一致）。
                    segsHigh.append(seg)
                }
                let rowsConcat = segs.count == 1 ? segs[0] : concatenated(segs, axis: 0)
                condSegs.append(rowsConcat)
                let rowsHighConcat = segsHigh.count == 1 ? segsHigh[0] : concatenated(segsHigh, axis: 0)
                condSegsHigh.append(rowsHighConcat)
                lowRefBlocks.append(RefBlock(kind: .video, latentH: UInt32(lh), latentW: UInt32(lw),
                                             latentT: b.latentT, audioT: b.audioT))
                offset += n
            }
        }
        guard offset == condRowsFull.shape[0] else {
            throw NSError(domain: "H3SelfLift", code: 5, userInfo: [NSLocalizedDescriptionKey:
                "参考块行数与 condRowsFull 不一致：\(offset) vs \(condRowsFull.shape[0])"])
        }
    // ★ 2026-10-19 原生半清条件行直通：调用方在源图 encode 阶段已按 0.5 像素编好整份条件行，
    //   低清段直接消费 —— 不再走「full-res latent → 逐帧 bilinear + 加性均值对齐」的
    //   重复条件处理（那条路径画质不受益，且每段多算一遍）。
    let condRowsLow: MLXArray
    if let nativeLowCond {
        condRowsLow = nativeLowCond
        log("SelfLift 低清条件行：原生半清直通（rows \(nativeLowCond.shape[0])，跳过 latent 域降采样）")
    } else if condSegs.isEmpty {
        // ★ 2026-10-25 空条件行兜底（修 `[concatenate] No arrays provided` fatal）：
        //   纯文生 / 无锚点 keyframe / 无参考块时 condSegs 为空，原来直接进 concatenated([])，
        //   MLX 在 ops.cpp:677 处 fatalError（ErrorHandler.swift:345）整段进程崩掉。
        //   此处改为构造 0 行条件张量：layout 侧 budgetLow 此时同样为 0，两者仍严格一致。
        condRowsLow = condRowsFull[0..<0]
        log("SelfLift 低清条件行：无锚点/参考块（纯文生），0 行（已跳过 empty concatenate）")
    } else {
        condRowsLow = condSegs.count == 1 ? condSegs[0] : concatenated(condSegs, axis: 0)
    }
    // ★ 2026-10-02 回退后语义：原始 full-res 条件行同时供低分/高分两路消费。
    //   高分段（condRowsHigh）= 原生 full-res 行直通（fl2va 无参考块用低清→提升版除外，
    //   那是官方 keyframe encodeImage 语义；ref2va 的 boundary/history 均为去噪完成的
    //   干净 full-res latent，原生直通不降画面再提升）；低分段内部自行重采样处理。
    //   （learned 不可用即显式终止。）
    let condRowsHigh: MLXArray
    if condSegsHigh.isEmpty {
        // ★ 2026-10-25：纯文生（无锚点、无参考块）时高清条件行同为 0 行，原本次处
        //   preconditionFailure 会直接终止；现在与低清侧同口径兜底 0 行，
        //   budgetHigh 此时也为 0，断言仍成立。非纯文生仍视为结构错误、照旧终止。
        guard contKeyAnchors.isEmpty, allRefBlocks.isEmpty else {
            preconditionFailure("SelfLift: 高清条件行缺失（condSegsHigh 为空，低清/高清同构注入未构建）")
        }
        condRowsHigh = condRowsFull[0..<0]
        log("SelfLift 高清条件行：无锚点/参考块（纯文生），0 行")
    } else {
        condRowsHigh = condSegsHigh.count == 1 ? condSegsHigh[0] : concatenated(condSegsHigh, axis: 0)
    }
    keepAlive.hold([condRowsHigh])
    MLX.eval(condRowsHigh)

    // 修复不变量：layout 侧 cond 行预算必须**严格等于**实际传入的条件行数，
    // 否则多余行会被 H3Transformer 按 layout.segments 静默丢弃、视频段起点整体前移。
    // ★ 2026-10-02：video（history）参考块行数 = latentT × refFrameRows（每帧一行网格）。
    let lowBlockRows: (RefBlock) -> Int = { b in
        b.kind == .video ? Int(b.latentT) * Int(refFrameRows(b)) : Int(refFrameRows(b))
    }
    // ★ 2026-10-07 预算口径：低分 = 锚点 keyframe 段（含 fl 双端硬锚 [.first,.last]）+ 各参考块低清行；
    //   高分 = 锚点 keyframe 段(fRows/锚) + 各参考块原生 full-res 行。
    let anchorLowRows = contKeyAnchors.count * (gHl * gWl)
    let budgetLow = anchorLowRows + lowRefBlocks.reduce(0) { $0 + lowBlockRows($1) }
    let budgetHigh = contKeyAnchors.count * fRows + allRefBlocks.reduce(0) { $0 + lowBlockRows($1) }
    guard budgetLow == condRowsLow.shape[0], budgetHigh == condRowsHigh.shape[0] else {
        throw NSError(domain: "H3SelfLift", code: 6, userInfo: [NSLocalizedDescriptionKey:
            "layout cond 预算与实际条件行不一致：低分 \(budgetLow) vs \(condRowsLow.shape[0])，"
            + "高分 \(budgetHigh) vs \(condRowsHigh.shape[0])"])
    }
    log("SelfLift 条件行：\(contKeyAnchors.count) 锚点 keyframe 段 + \(allRefBlocks.count) 参考块（含前置 \(preRefBlocks.count)；fl 双端硬锚 [.first,.last] 已含于锚点）"
        + " → 低分 \(condRowsLow.shape[0]) 行（layout 预算 \(budgetLow)）/ 高分 \(condRowsHigh.shape[0]) 行（layout 预算 \(budgetHigh)）")
    // 条件行低分全程每步都要 concat 进模型输入：挂进池子并**先物化**，
    // 让这段重采样（含 bilinear 的下标/权重小张量）在低分循环开始前就结算干净。
    keepAlive.hold([condRowsLow])
    MLX.eval(condRowsLow)

    // ★ 音频续接（2026-09-21）：前置尾段音频行数（2×winAudioT）；0 = 无前置音频。
    //   contAudioRef 传给 layout 生成 refAudio 段（audioUpdate=false，never-denoised）；
    //   audioX 全量 = [前置真实行 | 本节点噪声行]，采样更新只作用于后段（out.audio 只切
    //   目标 audioSegment），返回前裁掉前置行。
    let contAudioRows = audioContinuationRows?.shape[0] ?? 0
    let contAudioRef: [RefBlock] = contAudioRows > 0
        ? [RefBlock(kind: .audio, latentH: 0, latentW: 0, latentT: 0,
                    audioT: UInt32(contAudioRows / 2))]
        : []

    // 音频 Euler 更新只作用于本节点行（audioX 后段）；前置行保持真实值不动（never-denoised）。
    // out.audio 形状 = 目标 audioSegment（audioT*2 行），与全量 audioX 行数不同，必须 slice 拼接。
    func updateAudioX(_ ax: MLXArray, delta: MLXArray) -> MLXArray {
        if contAudioRows > 0 {
            let prefix = ax[0..<contAudioRows, 0..<32]
            let tail = ax[contAudioRows..<ax.shape[0], 0..<32] + delta
            return concatenated([prefix, tail], axis: 0)
        }
        return ax + delta
    }

    // ── 2. 低分上下文 ──────────────────────────────────────────────────
    // 布局构造与原生路径（H3Pipeline.swift:383-391）同款：续接前置尾段 keyframe 锚点无条件进
    // keyframes（ref2va 也注入，H3Pipeline 已把 contKeyRows 排在 preRef 之前）。
    // ★ 2026-10-05 方案A2（fl 双端硬锚）：fl 首尾帧统一走 keyframe 硬锚（.first+.last），
    //   fl/ref 的 keyframes 统一 = contKeyAnchors（fl=[.first,.last]；ref 续接=[.first]）。
    // ★ 2026-10-05 fl 锚几何对齐：高分几何（目标网格）折算到低清的等价几何——
    //   span=gHl*gWl（= 低清网格行数）、grid=低清网格（axisFromSqrtArea，目标网格 0.5 降采样）。
    //   数值与 nil 默认一致，但两阶段「同一个注入」参数机制同口径显式化；ref+续接不传参仍走 nil 默认。
    let flKeyframeSpansLow: [UInt32]? = keyframeSpansHigh.map { spans in
        [UInt32](repeating: UInt32(gHl * gWl), count: spans.count)
    }
    let flKeyframeGridsLow: [(h: [Double], w: [Double])?]? = keyframeGridsHigh.map { grids in
        let lArea = sqrt(Double(latHl) * Double(latWl))
        let lH = axisFromSqrtArea(dim: UInt32(latHl), patch: H3Const.patchH, sqrtArea: lArea)
        let lW = axisFromSqrtArea(dim: UInt32(latWl), patch: H3Const.patchW, sqrtArea: lArea)
        return grids.map { _ in (h: lH, w: lW) }
    }
    let layoutLow = PackedLayout(textLen: textLen, latentT: layoutLatT,
                                 latentH: UInt32(latHl), latentW: UInt32(latWl),
                                 audioT: audioT,
                                 keyframes: contKeyAnchors,
                                 frameCount: frameCount,
                                 refs: lowRefBlocks,
                                 contRefs: contAudioRef,
                                 keyframeSpans: flKeyframeSpansLow,
                                 keyframeGrids: flKeyframeGridsLow)
    if let spansHigh = keyframeSpansHigh {
        log("fl 锚几何对齐：高分 span=\(spansHigh)/grid=目标网格 \(gridH)×\(gridW)；"
            + "低分折算 span=\(flKeyframeSpansLow ?? [])/grid=低清网格 \(gHl)×\(gWl)（=目标网格 0.5 降采样）")
    }
    // ref2va 的文本行含 4 个视觉块，AdaLN 模态标签必须与原生 layout 一样下发（H3Pipeline.swift:390），
    // 否则视觉块的 mod 行会落到 text 模态（H3Layout.swift:637 起），条件参考被打错调制。
    layoutLow.textTags = textTags
    let tsLow = collectScheduleTs(layout: layoutLow, sigmas: split.lowSigmas,
                                  shiftV: shiftV, shiftA: shiftA, aug: CondNoiseAug(),
                                  audioSigmas: audioSigmasLow)
    dit.precomputeAdaln(ts: tsLow)
    let ropeLow = buildRope(layout: layoutLow, invFreq: dit.invFreq)

    // ★ 2026-10-03 fresh-noise 路线：续接初值恒 nil（前置尾段只走 contKeyAnchors 条件行锚/
    //   参考块，不拼进采样初值），videoX 恒为本节点新段纯噪声起步（latT 行）。
    // ★ 2026-10-22 噪声口径统一（对齐官方 H3 的单 seed 顺序抽取）：
    //   官方 diffusers 集成里一个请求的所有噪声都从同一个 generator 依序抽（条件噪声 → video → audio），
    //   顺序本身参与可复现性，不存在 seed+1 / seed+7 这种多路独立派生。本 Runner 用同一 seed
    //   split 出的有序 key 槽位复刻该结构：
    //     槽位 0 = 生成视频初值 · 槽位 1 = 音频初值 · 槽位 2 = lift 过渡重加噪
    //   三处共用一条序列，抽取顺序即下标顺序；以后新增噪声一律往后追加槽位，
    //   禁止再写 seed &+ N（旧口径 +0 / +1 / +7 已废弃）。
    let noiseKeys = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
    MLX.eval(noiseKeys)
    var videoX = MLXRandom.normal([Int(latT) * gHl * gWl, vpatch], key: noiseKeys[0])
    var audioX: MLXArray
    if contAudioRows > 0, let preAudio = audioContinuationRows {
        // ★ 音频续接：audioX = [前置真实尾段 | 本节点噪声]，前置行经 refAudio 段注入注意力，
        //   永远不被更新（never-denoised），本节点行从纯噪声采样。
        let preF32 = preAudio.asType(.float32)
        let newNoise = MLXRandom.normal([Int(audioT) * 2, 32], key: noiseKeys[1])
        audioX = concatenated([preF32, newNoise], axis: 0)
        keepAlive.hold([preF32])
        log("SelfLift 音频续接：前置尾段 \(contAudioRows) 行（refAudio 锚点，never-denoised）+ 本节点 \(Int(audioT) * 2) 行纯噪声起步")
    } else {
        audioX = MLXRandom.normal([Int(audioT) * 2, 32], key: noiseKeys[1])
    }
    MLX.eval(videoX, audioX)

    let pabK = attentionBroadcastK
    // ★ 2026-10-26 复用守卫重写（旧守卫恒不触发：lowSigmas 为降序，`dropFirst() - self`
    //   得到的是负值，max() 取到最接近 0 的负最小步长，永远 < 阈值）。
    //   新判定不看「全局平均跨度」，而是**逐步按缓存里实际的 σ 过期量**：
    //   只有 |σ_now − 缓存写入时 σ| ≤ 容差才允许复用，超出立即强制重算。
    //   低清 shift=12 曲线在低 σ 端步长可小到 0.02（可复用）、高 σ 端可大到 0.7（必重算），
    //   逐级判定后低清只保留极少数安全复用步；高分细步长段照常赚复用 —— 规则统一，不再依赖魔法阈值。
    //   NA_H3_SELFLIFT_PAB_MAX_DSIGMA 可覆盖容差（默认与主链同一个 attnBroadcastDefaultMaxStale()，当前 0.04）。
    let pabMaxStale = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_PAB_MAX_DSIGMA"].flatMap(Double.init)
        ?? attnBroadcastDefaultMaxStale()
    // 干跑一遍真实调度：整段一次复用机会都没有时就不建缓存实例
    //（省 count 层 at 常驻，forward 走 cacheable=false，与「每步 refresh」逐位等价）。
    let lowPlan = attnBroadcastPlan(sigmas: split.lowSigmas, k: pabK, maxStale: pabMaxStale)
    let lowPabUsable = pabK > 1 && lowPlan.contains(false)
    if pabK > 1 && !lowPabUsable {
        let dMax = zip(split.lowSigmas.dropFirst(), split.lowSigmas).map { abs($0 - $1) }.max() ?? 0
        log("SelfLift 低分 PAB 判定：逐步过期量下无可复用步（Δσ_max=\(String(format: "%.4f", dMax)) > 容差 \(pabMaxStale)）→ 不建跨步缓存（每步重算）")
    }
    let makePab: () -> H3AttnBroadcast? = { pabK > 1 ? H3AttnBroadcast(count: dit.blocks.count) : nil }
    var pab = lowPabUsable ? makePab() : nil

    // ── 3. 低分循环（默认逐段 Euler 到 σ_k；clean 端点模式末步改取 Eq.3 预测、丢弃 Euler）──
    slMem("低分段 · 起步（DiT 全权重 + 低分 PAB \(pab == nil ? "关闭" : "开启")）")
    var lowCleanX0: MLXArray? = nil
    for i in 0..<split.lowNFE {
        let stepT = Date()
        let sigma = split.lowSigmas[i]
        let sigmaNext = split.lowSigmas[i + 1]
        dit.sparsePolicy = sparsePolicy

        let plan = buildTimestepPlanGlobal(layout: layoutLow, sigmaV: sigma,
                                           shiftV: shiftV, shiftA: shiftA,
                                           aug: CondNoiseAug(), globalTs: dit.adalnTables!.ts,
                                           audioSigma: audioSigmasLow[i])
        let videoIn = concatenated([condRowsLow, videoX], axis: 0)
        let refresh: Bool
        if pab == nil {
            refresh = false
        } else {
            let base = attnBroadcastRefresh(i, steps: UInt32(split.lowNFE), k: pabK)
            let stale = pab!.cachedSigma
            let mustRefresh = attnBroadcastMustRefresh(cachedSigma: stale, sigma: sigma, maxStale: pabMaxStale)
            refresh = base || mustRefresh
            if !base, mustRefresh {
                log("低分 PAB 过期重算 i=\(i)：|Δσ|=\(String(format: "%.4f", abs(sigma - stale))) > \(pabMaxStale)（缓存来自 σ=\(String(format: "%.4f", stale))）")
            }
        }
        let outPos = dit.forward(layout: layoutLow, plan: plan, textStates: textStates,
                                 videoRows: videoIn, audioRows: audioX, rope: ropeLow,
                                 sigmaV: sigma, shiftV: shiftV, shiftA: shiftA,
                                 attnBcast: pab, attnRefresh: refresh)
        // ★ CFG 分支（官方 SelfLiftH3Sampler 默认 cfg=5.0：positive/negative 双路全程引导）：
        //   v = v_neg + cfg·(v_pos − v_neg)。negative 条件行已由调用方 pad 到与 positive 相同
        //   textLen，故复用同一 layout/rope/plan（时间坐标严格一致）；PAB 缓存只服务 positive
        //   （negative 每步现算，不污染跨步注意力缓存）；audio 不参与 CFG，取 positive 一路。
        let out: H3DiTOutput
        if cfgScale > 0, let textNeg = textStatesNeg {
            let outNeg = dit.forward(layout: layoutLow, plan: plan, textStates: textNeg,
                                     videoRows: videoIn, audioRows: audioX, rope: ropeLow,
                                     sigmaV: sigma, shiftV: shiftV, shiftA: shiftA,
                                     attnBcast: nil, attnRefresh: false)
            out = H3DiTOutput(video: outNeg.video + (outPos.video - outNeg.video) * cfgScale,
                              audio: outPos.audio)
        } else {
            out = H3DiTOutput(video: outPos.video, audio: outPos.audio)
        }

        // ★ 音频调度解耦（2026-10-16）：delta_sigma_a 取音频自身调度的相邻差，不再由视频 sigma 换算；
        //   slopeAF 仍用于抵消 out.audio 携带的 d_sigma_a/d_sigma_v 缩放（与原生 H3Pipeline 同款）。
        let slopeAF = timeShiftSlope(sigma, fromShift: shiftV, toShift: shiftA)
        let da = (audioSigmasLow[i + 1] - audioSigmasLow[i]) / slopeAF

        // 步内缩放因子（4 字节标量张量）先取出并挂进强引用池：它们是本步懒图（加噪/音频更新）
        // 的输入 buffer，若在 eval 之前就失去 Swift 侧引用，小 buffer 会被分配器空闲池回收。
        let vScale = H3TensorOps.scalarLike(Float(sigmaNext - sigma), out.video)
        let aScale = H3TensorOps.scalarLike(Float(da), out.audio)
        keepAlive.hold([vScale, aScale])
        if i == split.lowNFE - 1, useCleanEndpoint {
            // 旧路径（clean 端点）：Eq.3 预测 clean x0；丢弃本步 video 的 Euler 更新。
            // ★ 2026-10-19 起默认**不走**这里（仅关闭不移除）：clean x0 内容权重在 σ_k 处过低，
            //   升频后与高清重画内容不重合 → 鬼影。开关见 cfg.cleanEndpointHandoff。
            lowCleanX0 = SelfLiftCore.cleanEndpoint(x: videoX, velocity: out.video, sigma: sigma,
                                                    keepAlive: keepAlive)
            // 音频不参与提升，按正常 Euler 边界步跟到 σ_next（只更新本节点行）
            audioX = updateAudioX(audioX, delta: out.audio * aScale)
        } else {
            // 带噪直通（默认）/ 中间步：正常 Euler 更新。末步在 σ_k 处停下来，
            // videoX 即 σ_k 的带噪 x，直接作为交接状态（不预测 clean、不丢 Euler）。
            videoX = videoX + out.video * vScale
            audioX = updateAudioX(audioX, delta: out.audio * aScale)
        }
        // ★ 物化点：低分最后一步的 video 分支**不在** videoX 的图里（该步按 Eq.3 丢掉了 Euler 更新），
        //   必须把 lowCleanX0 一并带进这次 eval；否则 Eq.3 的整条 video 尾图会被拖进过渡块的
        //   VAE decode/encode 阶段才物化，与分配器的空闲池回收撞在同一个时间窗（GPU 在飞 →
        //   小 buffer 被真 release）→ 正是本次崩溃的那条路径。
        if let x0Pending = lowCleanX0 {
            MLX.eval(videoX, audioX, x0Pending)
        } else {
            MLX.eval(videoX, audioX)
        }
        // 本步 GPU 已同步：池内张量全部已物化，放掉强引用（下一步重建），
        // 内存纪律与原 stage1 循环一致（不额外长期驻留）。
        // 随后把「低分全程复用的条件行」重新挂回池子（它跨步存活，不能被本次 release 带走）。
        keepAlive.release()
        keepAlive.hold([condRowsLow])
        slMem("低分 \(i + 1)/\(split.lowNFE) 步后")
        // 日志格式与原生 stage1 逐字同款（H3Pipeline.swift 行 553）：
        // sigma/dsigma 均 %.4f；方括号槽位放本阶段标记 + PAB 复用/刷新；耗时 = Int(-Date().timeIntervalSince(stepT))。
        let lowBcastMark = pab == nil ? "" : (refresh ? " bcastRefresh" : " bcastReuse")
        log("SelfLift 低分 \(i + 1)/\(split.lowNFE) sigma \(String(format: "%.4f", sigma)) dsigma \(String(format: "%.4f", sigmaNext - sigma)) [low\(lowBcastMark)]（\(Int(-Date().timeIntervalSince(stepT)))s）")
    }
    // 交接状态：clean 模式下必须是 Eq.3 x0；带噪直通模式下就是 σ_k 处的 videoX。
    if cfg.cleanEndpointHandoff, lowCleanX0 == nil {
        throw SelfLiftScheduleError.tooFewSteps(0)
    }
    let handoffState: MLXArray = lowCleanX0 ?? videoX
    if !useCleanEndpoint {
        log("SelfLift 交接状态：低清出口取 σ_k=\(String(format: "%.4f", split.sigmaK)) 的 x（末步正常 Euler，跑满时即 clean 成品）"
            + "；过渡块按 σ_ren=\(String(format: "%.4f", split.sigmaNext)) \(abs(split.sigmaNext - split.sigmaK) > 1e-9 ? "重加噪" : "等深直通")")
    } else {
        log("SelfLift 交接状态：clean 端点 —— 低清出口 Eq.3/z0 升频后按 σ_ren=\(String(format: "%.4f", split.sigmaNext)) 重加噪")
    }

    // ── 3.5 阶段边界屏障（低分出口 → 过渡块入口）───────────────────────────
    // 过渡块是整个 SelfLift 分支里唯一一段「大分配浪 + 大回收浪」窗口：VAE decode（36 层
    // ViT 到目标分辨率）→ bicubic 上采样 → VAE encode（再下采样回 latent），每一步都产生
    // 大量临时 buffer，其中包含大量 4 字节 fp32 标量（MLXArray.scalar / MLXArray(_)）。
    //
    // 【第五轮结论：恢复「入口只清池、不抬 cacheLimit」，依据与边界如下】
    //  · 实证：12:28 日志（只清池版）过渡块 E 结算成功（清池后 cache 0MB）；第四轮撤清池后
    //    11:24 复测崩溃（入口 cache 10109MB 已超 limit 10000MB，随后 Gather setBytes nil）。
    //    同参数（48×84 latent / T=37 / learned lift）一成一崩，差异只在入口池状态 ⇒ 清池有效。
    //  · 边界：第四轮注释 ② 对「抬 cacheLimit」的否定仍然成立，本恢复**只清池、绝不抬 cacheLimit**；
    //    第四轮注释 ①（崩溃点不在 clearCache 内部）不构成对清池本身的证伪 —— 清池只是把低分
    //    累积的空闲缓存归零，让过渡块大分配浪从干净池起步，不改变任何算子与图结构。
    //
    // 【第四轮结论（保留存档）：第三轮的 clearCache + 抬 cacheLimit 方案已撤销，两条依据如下】
    // ① 崩溃点不可能在 clearCache 内部：pipelineLog 的第一句就是同步 print
    //    （模型公共函数-通用.swift:84），而入口日志（旧代码 :515）打印在 clearCache 返回之后。
    //    该日志能在复测中打出来，就说明 clearCache 已完整执行完毕 —— 把「首恶」判给它属误判。
    // ② 「抬 cacheLimit ⇒ 池不会被回收」的前提不成立：MLX 的 Metal allocator 有两条互不相干的
    //    回收路径（backend/metal/allocator.cpp 的 malloc / free）：
    //      a) free()：仅当 cacheMemory < max_pool_size_（即 cacheLimit）才回收进空闲池，
    //         否则立刻 buf->release()；
    //      b) malloc()：mem_required = activeMemory + cacheMemory + size 一旦 >= gc_limit_，
    //         就对空闲池做 release_cached_buffers(...)，**不与 GPU 同步**地真 release MTLBuffer。
    //    其中 gc_limit_ = min(memoryLimit, 0.95 × recommendedMaxWorkingSetSize)
    //    （allocator.cpp 的 set_memory_limit），**与 cacheLimit 完全无关**：抬 cacheLimit 只关掉
    //    a)，关不掉 b)；而 b) 恰在过渡块（VAE decode/encode 全分辨率大分配浪）里最容易命中。
    //    这正是第三轮改完仍崩的原因；而把 cacheLimit 抬到 memoryLimit 反而让空闲池上限逼近
    //    0.95×rws，使 b) 更易被顶到，属负向改动。
    // ③ 第五轮起只补一道「入口清池」（clearCache），不抬 cacheLimit、不加窗口内逻辑；
    //    窗口内各步内存水位由只读诊断 slDiag 打点，供复测把崩溃位置收敛到具体步骤。
    MLX.eval(handoffState)
    Stream.gpu.synchronize()

    // ── 3.6 lowOnly 出口（IC 链路）：低清段完成即返回，跳过过渡块 + 高分循环 ──
    // H3 一采只出低分半清 latent，高分由 LTX 二采负责（升频×2 + 高清 IC 精修，factor=2）。
    // 返回形态：NCDHW [1,C,T,latHl,latWl]（半清网格），供 H3Pipeline 的 adapter 直转 LTX 域。
    if lowOnly {
        let z0LowLat = toNCDHW(H3TensorOps.unpatchifyVideo(handoffState, gridH: gHl, gridW: gWl), c: latC)
        MLX.eval(z0LowLat)
        log("SelfLift lowOnly：低清段完成（NFE=\(split.lowNFE)，latent \(latHl)×\(latWl)），"
            + "直接返回低清 latent \(z0LowLat.shape)；高分交由 LTX 二采升频×2 + IC 精修")
        let outAudio = contAudioRows > 0 ? audioX[contAudioRows..<audioX.shape[0], 0..<32] : audioX
        return (z0LowLat, outAudio)
    }

    // ★ 本轮修复①（压低基线的第一半）：低分 NFE 已全部跑完并同步，PAB 跨步注意力缓存（blocks[]）
    //   在过渡块里不会再被任何一次 forward 读取，但它按「50 层 × 低分全量 token」常驻
    //   （bf16 实测 ≈5.3GB），会一路挂到 E。此处入口屏障后立即释放 —— 读侧（低分 forward）已
    //   全部同步完成，不存在"还有命令缓冲要读它"的情况，故安全。
    //   上游语义不受影响：高分段的 pab 是 627 行另建的 makePab()，与本实例无关。
    let pabWasHeld = pab?.blocks.contains(where: { $0 != nil }) ?? false
    pab?.reset()
    // ★ 修复③（第五轮）：入口清池（只清池、不抬 cacheLimit）——低分阶段累积的空闲缓存
    //   归零，过渡块大分配浪从干净池起步（12:28 实证「清池后 cache 0MB」该次过渡块完整成功；
    //   11:24 未清池入口 cache 10109MB 超限 → Gather setBytes nil 崩溃）。
    let cacheBefore = MLX.Memory.cacheMemory / 1_000_000
    let activeBefore = MLX.Memory.activeMemory / 1_000_000
    MLX.Memory.clearCache()
    log("SelfLift 过渡块窗口：入口屏障完成（Stream.gpu.synchronize + 入口清池，未改 cacheLimit；PAB 跨步缓存已释放=\(pabWasHeld)；清池前 active \(activeBefore)MB / cache \(cacheBefore)MB（limit \(MLX.Memory.cacheLimit / 1_000_000)MB）→ 清池后 cache \(MLX.Memory.cacheMemory / 1_000_000)MB）")
    // 只读诊断（不改任何行为，定位完可随时删）：原 515→598 之间没有任何日志，崩溃点无法收敛到
    // 具体步骤；这里在既有物化点各打一条 MLX 内存水位，复测时最后一条成功的打点即崩溃前一步。
    func slDiag(_ step: String) {
        // 用全局 pipelineLog（而非外层 log 闭包）：本函数会被 liftPixelRows 等 @escaping 闭包调用，
        // 捕获非逃逸的 log 参数会编译不过（escaping closure captures non-escaping parameter 'log'）。
        pipelineLog("[H3] [MEM] SelfLift 过渡块 · \(step)：MLX active \(MLX.Memory.activeMemory / 1_000_000)MB，cache \(MLX.Memory.cacheMemory / 1_000_000)MB")
    }
    // 低分 / 高分段的只读打点见文件级 `slMem(_:)`（必须能在 Runner 任意位置调用，
    // 故不放在函数体内 —— 局部 func 的使用点不能早于其声明点）。
    slDiag("入口屏障后起步")

    // ── 3.9 latent 落盘（低清出口 / 升频器出口，高清之前）──────────────────
    // ★ 2026-10-19（用户需求）：lift 模式默认把「高清之前」的关键阶段 latent 落盘，
    //   NCDHW float32，文件头 5 个 Int32 = shape，供离线解码成视频逐阶段对比。
    //   NA_DUMP_SELFLIFT=0 关闭；NA_DUMP_SELFLIFT_DIR 指定根目录
    //   （默认 /tmp/h3_selflift_dump/run_<时间戳>/，每次生成独立子目录，不互相覆盖）。
    let selfLiftDumpEnabled = (ProcessInfo.processInfo.environment["NA_DUMP_SELFLIFT"] ?? "1") != "0"
    let selfLiftDumpDir: String = {
        let base = ProcessInfo.processInfo.environment["NA_DUMP_SELFLIFT_DIR"] ?? "/tmp/h3_selflift_dump"
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd_HHmmss"
        return base + "/run_" + df.string(from: Date())
    }()
    let dumpSelfLift: (String, MLXArray) -> Void = { name, arr in
        guard selfLiftDumpEnabled else { return }
        let f = arr.asType(.float32)
        MLX.eval(f)
        let floats = f.asArray(Float.self)
        var header = arr.shape.map { Int32($0) }
        var data = Data(bytes: &header, count: header.count * MemoryLayout<Int32>.size)
        floats.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        try? FileManager.default.createDirectory(atPath: selfLiftDumpDir, withIntermediateDirectories: true)
        try? data.write(to: URL(fileURLWithPath: selfLiftDumpDir + "/" + name))
        pipelineLog("SELFLIFT-DUMP \(name) shape=\(arr.shape) floats=\(floats.count) → \(selfLiftDumpDir)/\(name)")
    }

    // ── 4. 过渡块（零 NFE）──────────────────────────────────────────────
    // 4.1 直接 latent 提升：官方 learned upscaler（H3LatentUpscaler，时间维感知，
    //     实测 1344×768 高分辨率下显著消除帧间双像/面部重影，见 h3_scene_ab 自检结论）。
    //     learned 不可用即显式终止，不降级。
    // ★ 2026-10-20 低频锚定（方案 A）：升频器出口 latent 的 NCDHW 视图，零拷贝引用缓存下来，
    //   只在高分收尾后用于「低频带替换」，不参与过渡块噪声（噪声仍是纯白噪）。
    //   ★ 默认关闭（α_low = 1.0）；该引用仅剩 dump 用途，设 NA_H3_SELFLIFT_LOWANCHOR<1 重新开启。
    var liftedLatNCDHW: MLXArray? = nil
    let liftLatentRows: (MLXArray) -> MLXArray = { rows in
        let lat = H3TensorOps.unpatchifyVideo(rows, gridH: gHl, gridW: gWl)      // [latT,latHl,latWl,C]
        let up: MLXArray
        do {
            up = try slLearnedLiftLatent(lat, outH: latH, outW: latW,
                                         keepAlive: keepAlive)                   // [latT,latH,latW,C]
        } catch {
            preconditionFailure("learned upscaler 不可用，SelfLift 终止：\(error.localizedDescription)")
        }
        // ★ 2026-10-19 升频器出口 latent（高清之前，rho=0 时这是唯一能拿到 z_lat 的点：
        //   旧实现只在 consistencyRows 里 dump，rho=0 该闭包不执行 → z_lat 永远落不了盘）
        let upNCDHW = toNCDHW(up, c: latC)
        dumpSelfLift("z_lat.bin", upNCDHW)
        liftedLatNCDHW = upNCDHW
        let rowsOut = H3TensorOps.patchifyVideo(up, gridH: gridH, gridW: gridW)
        keepAlive.hold([lat, up, rowsOut])
        return rowsOut
    }
    // 4.2 像素锚点提升：官方 `_pixel_anchor_video_single`（selflift.py 行 87-98）——
    //     低分 latent → VAE decode → `mode="bicubic", antialias=True` 上采样到 目标尺寸 → VAE encode
    let liftPixelRows: (MLXArray) -> MLXArray = { rows in
        slDiag("4.2 VAE decode 进入（像素锚点最重一段）")
        let lat = H3TensorOps.unpatchifyVideo(rows, gridH: gHl, gridW: gWl)      // [latT,latHl,latWl,C]
        let px = hooks.decodeToPixels(toNCDHW(lat, c: latC))                     // [1,3,frames,latHl*16,latWl*16]
        // ★ 物化点 A：解码（低分 latent → 像素）先结算完，再进 bicubic 上采样。这里是全流程
        //   统一内存峰值最高的位置，若解码的整条懒图与上采样同时在飞，分配器一旦触发空闲池
        //   回收，就会把仍被在飞命令缓冲引用的小 buffer 真 release。
        MLX.eval(px)
        slDiag("A VAE decode 完成")
        // ★ 修复②：这里不再 hold([lat, px])。px 是全分辨率像素（bf16 ≈192MB），它的最后消费者
        //   是下面的分块上采样；闭包内 `let` 本身已保证它在被消费前存活，额外挂进池子只会让它
        //   （以及后续 up）一直留到过渡块 E 之后。中间量用完即弃，交给分配器复用。
        // 官方按实际像素尺寸推目标尺寸（ratio 由像素/低分 latent 相除得到，本工程为 16）
        let ratioH = Double(px.dim(3)) / Double(latHl)
        let ratioW = Double(px.dim(4)) / Double(latWl)
        let up = slResizePixelsBicubicAA(px,
                                        outH: Int((Double(latH) * ratioH).rounded()),
                                        outW: Int((Double(latW) * ratioW).rounded()),
                                        keepAlive: keepAlive)
        // ★ 物化点 B：上采样结果结算完再进 VAE encode；否则上采样的分块惰性张量会一直挂到
        //   编码的分配浪里才被消费（老代码即如此），在飞资源数量峰值叠加。
        MLX.eval(up)
        slDiag("B bicubic 上采样完成")
        let z = hooks.encodeToLatent(up)                                         // [1,C,latT,latH,latW]
        // ★ 物化点 C：编码结果立即物化，别留给整块末尾那次 eval —— 让编码器内部临时 buffer
        //   在编码结束的瞬间就"消费者已通过"，而不是与后面的 Eq.6-9/重加噪一起在飞。
        MLX.eval(z)
        slDiag("C VAE encode 完成")
        let rowsOut = H3TensorOps.patchifyVideo(fromNCDHW(z, c: latC), gridH: gridH, gridW: gridW)
        keepAlive.hold([z, rowsOut])
        return rowsOut
    }
    // 4.3 Eq.6-9：在 NCDHW 域按通道求不一致度
    // latent 落盘 helper 已上移到 4.1 之前（z0_low / z_lat 在过渡块入口即写出）。
    if selfLiftDumpEnabled {
        // 低清出口状态是 rows，先还原成 NCDHW latent 再落盘（与 z_lat/z_pix 同域可比）。
        // ★ 两种模式文件语义不同，分开命名，避免离线解码时误判：
        //   · clean 模式 → z0_low.bin（Eq.3 clean x0）
        //   · 带噪直通（默认）→ x_low_sigmak.bin（σ_k 处的含噪 x，解码出来必然是带噪/半清画面，属正常）
        let lowOutLat = toNCDHW(H3TensorOps.unpatchifyVideo(handoffState, gridH: gHl, gridW: gWl), c: latC)
        dumpSelfLift(useCleanEndpoint ? "z0_low.bin" : "x_low_sigmak.bin", lowOutLat)
    }
    let consistencyRows: (MLXArray, MLXArray) -> MLXArray = { zLat, zPix in
        let a = toNCDHW(H3TensorOps.unpatchifyVideo(zLat, gridH: gridH, gridW: gridW), c: latC)
        let b = toNCDHW(H3TensorOps.unpatchifyVideo(zPix, gridH: gridH, gridW: gridW), c: latC)
        let fixed = SelfLiftCore.artifactAwareConsistencyLift(
            zLat: a, zPix: b, rho: cfg.rho, wMin: cfg.wMin, wMax: cfg.wMax, channelAxis: 1,
            keepAlive: keepAlive)
        if selfLiftDumpEnabled {
            // z_lat 已在 4.1 升频器出口落盘（唯一真源），此处只补 z_pix / z0_high
            dumpSelfLift("z_pix.bin", b)
            dumpSelfLift("z0_high.bin", fixed)
        }
        let rowsOut = H3TensorOps.patchifyVideo(fromNCDHW(fixed, c: latC), gridH: gridH, gridW: gridW)
        keepAlive.hold([a, b, fixed, rowsOut])
        return rowsOut
    }

    // ★ 过渡噪声行数与提升后 latent 行数一致（2026-09-20 修复）：
    //   2026-10-03 起 fresh-noise 路线 layoutLatT=latT（视频段只含本节点新段），
    //   过渡噪声按 layoutLatT 走（与布局/rope/videoX 同一口径）。
    let transitionNoise = MLXRandom.normal([Int(layoutLatT) * gridH * gridW, vpatch],
                                           key: noiseKeys[2])
    // ★ 物化点 D：过渡噪声先物化，别把随机数 kernel 也挤进过渡块的分配浪里。
    MLX.eval(transitionNoise)
    slDiag("D 过渡噪声完成")
    keepAlive.hold(transitionNoise)
    let liftT = Date()
    slDiag("4.4 进入零 NFE 过渡块（lift → 按 σ_next=σ_k 重加噪）")
    // ★ 防御断言（2026-09-20）：过渡噪声行数必须等于提升后 latent 行数（layoutLatT×gridH×gridW），
    //   防止续接/尺寸口径再漂移导致 Eq.10 重加噪 broadcast 崩溃。
    precondition(transitionNoise.shape[0] == Int(layoutLatT) * gridH * gridW,
                 "过渡噪声行数与提升后 latent 行数不符：\(transitionNoise.shape[0]) != \(Int(layoutLatT) * gridH * gridW)")
    videoX = SelfLiftTransition.run(
        split: split,
        cfg: cfg,
        lowState: handoffState,
        noise: transitionNoise,
        liftLatent: liftLatentRows,
        liftPixelAnchor: cfg.needsPixelAnchor ? liftPixelRows : nil,
        consistencyLift: consistencyRows,
        noiseScaling: { x0, s, n in
            SelfLiftCore.noiseScaling(sigma: s, noise: n, x0: x0, keepAlive: keepAlive) },
        holdIntermediate: { keepAlive.hold([$0]) })
    // ★ 物化点 E：过渡块整图在此结算。eval 返回即 GPU 已同步，此时才允许放掉块内所有
    //   临时张量的强引用（早于这次 eval 释放，调用方自己就制造了悬垂）。
    MLX.eval(videoX)
    slDiag("E 过渡块结算后（eval 返回）")
    keepAlive.release()
    // ── 3.5 阶段边界屏障 · 出口 ──
    // 只做 synchronize：等过渡块在飞命令缓冲全部结束。本轮已撤销此处的 clearCache 与
    // cacheLimit 恢复（撤销依据见入口屏障处注释 ①②）。
    Stream.gpu.synchronize()
    log("SelfLift 过渡块窗口：出口屏障完成（仅 Stream.gpu.synchronize，未清池、未改 cacheLimit；cache \(MLX.Memory.cacheMemory / 1_000_000)MB / limit \(MLX.Memory.cacheLimit / 1_000_000)MB）")
    // ★ 修复②（本轮）：像素锚点 VAE 只服务过渡块（decode → bicubic → encode）。出口屏障刚做完
    //   Stream.gpu.synchronize()，VAE 侧已无在飞命令缓冲；而高分段的 NFE 一次都不用它的权重，
    //   却要按 bf16 ≈5.3GB 一路常驻到 Runner 返回（≈高分全段）。此处提前回调置 nil：
    //   只放手、**不 clearCache、不动 cacheLimit**（撤销依据见入口屏障注释 ①②，
    //   与"不得把 cacheLimit 调到会让过渡块再整池收缩的量级"的硬约束一致 —— 这里根本没碰它）；
    //   放掉的权重 buffer 落进空闲池，正好被高分段每步的中间量命中复用，净省一次全新分配。
    hooks.releasePixelVAE?()
    slMem("过渡块后 · 像素锚点 VAE 提前释放")
    // 过渡块（零 NFE）：日志格式与原生 stage1 逐字同款（H3Pipeline.swift 行 553），
    // dsigma = σ_next − σ_k（等深调度下 = 0）；方括号槽位放提升前后尺寸。
    let liftModeMark = useCleanEndpoint ? "clean→重加噪" : "带噪直通·不重加噪"
    log("SelfLift lift sigma \(String(format: "%.4f", split.sigmaK)) dsigma \(String(format: "%.4f", split.sigmaNext - split.sigmaK)) [lift \(latWl)×\(latHl)→\(latW)×\(latH) \(liftModeMark)]（\(Int(-Date().timeIntervalSince(liftT)))s）")

    // ── 5. 高分上下文重建（分辨率/序列长度变了，必须整表重算）──────────────
    // 同上：续接前置尾段 keyframe 锚点无条件进 keyframes（ref2va 也注入）。
    // ★ 2026-10-05 方案A2（fl 双端硬锚）：fl/ref 的 keyframes 统一 = contKeyAnchors；
    //   fl 场景高分锚几何显式同口径（span=fRows/锚、grid=目标网格，与 H3Pipeline 方案A2
    //   构造一致）；ref+续接不传参 → keyframeSpans/Grids nil 默认，行为不变。
    let layoutHigh = PackedLayout(textLen: textLen, latentT: layoutLatT,
                                  latentH: UInt32(latH), latentW: UInt32(latW),
                                  audioT: audioT,
                                  keyframes: contKeyAnchors,
                                  frameCount: frameCount,
                                  refs: allRefBlocks,
                                  contRefs: contAudioRef,
                                  keyframeSpans: keyframeSpansHigh,
                                  keyframeGrids: keyframeGridsHigh)
    layoutHigh.textTags = textTags
    let tsHigh = collectScheduleTs(layout: layoutHigh, sigmas: split.highSigmas,
                                   shiftV: shiftV, shiftA: shiftA, aug: CondNoiseAug(),
                                   audioSigmas: audioSigmasHigh)
    dit.precomputeAdaln(ts: tsHigh)
    let ropeHigh = buildRope(layout: layoutHigh, invFreq: dit.invFreq)
    // ★ 修复①（本轮，主体）：高分段的 PAB 广播缓存只在「确实存在复用步」时才建。
    //   刷新调度 attnBroadcastRefresh 在短高分里是全覆盖的：高清固定 1 步（highNFE=1），而
    //   warmup = attnBroadcastWarmup(2) = max(1, min(4, 2·4/30)) = 1 覆盖第 0 步、
    //   tail = attnBroadcastTail(2) = max(1, min(2, 2·2/30)) = 1 覆盖最后一步 ⇒ 2 步全刷。
    //   实测日志（pipeline.log 行 4221/4222）：「高分 1/2」「高分 2/2」均标 bcastRefresh ——
    //   缓存写进去立刻被下一步覆盖、一次都读不到（零收益），却要按「50 层 × 高分全量 token」常驻 bf16：
    //       高分全量行 ≈3.9 万（低分 1.05 万 × ~4）× 5376 hidden × 2B × 50 层 ≈ 21GB
    //   （低分同口径实测 ≈5.3GB，与行数比基本吻合）
    //   —— 这 21GB 贯穿整个高分段（等于该段真实基线 16.3GB 的 1.3 倍），是唯一一处「零收益纯驻留」，
    //   也是高分段把进程 footprint 顶到物理内存之上、触发 ~40GB swap 的最大可摘项。
    //   关闭后 forward 走 cacheable=false 分支，每步照旧现算，与「每步都 refresh」逐位等价：
    //   数值不变、步数不变、单步耗时不变（本来两步就在现算），只省内存。
    //   可用环境变量 NA_H3_SELFLIFT_HIGH_PAB=1/0 强制开关做 A/B 复测。
    //   注意：旧实例里的 at 缓存是最近一步 forward 的产物，整批放开必须落在 GPU 已同步的点上；
    //   上面的 MLX.eval(videoX)（物化点 E）+ 出口屏障已把过渡块全图结算完，此处再显式 reset 后重建。
    //   ★ 2026-10-26：判定改用与运行时同一套「逐步过期量」干跑（attnBroadcastPlan），
    //   含 σ 容差，故高分若步长偏大同样会被判成无可复用步、直接不建实例。
    let highHasReuse = attnBroadcastPlan(sigmas: split.highSigmas, k: pabK, maxStale: pabMaxStale).contains(false)
    var highPabEnabled = highHasReuse
    if let f = ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_HIGH_PAB"] { highPabEnabled = (f == "1") }
    pab?.reset()
    pab = highPabEnabled ? makePab() : nil
    log("SelfLift 高分段上下文：PAB 广播缓存\(highPabEnabled ? "开启" : "关闭")"
        + "（highNFE=\(split.highNFE)，存在复用步=\(highHasReuse)；关闭时省 50 层 × 高分全量 token 常驻 ≈21GB）")
    slMem("高分段 · 起步")

    // ── 6. 高分循环 ────────────────────────────────────────────────────
    for i in 0..<split.highNFE {
        let stepT = Date()
        let sigma = split.highSigmas[i]
        let sigmaNext = split.highSigmas[i + 1]
        dit.sparsePolicy = sparsePolicy

        let plan = buildTimestepPlanGlobal(layout: layoutHigh, sigmaV: sigma,
                                           shiftV: shiftV, shiftA: shiftA,
                                           aug: CondNoiseAug(), globalTs: dit.adalnTables!.ts,
                                           audioSigma: audioSigmasHigh[i])
        let videoIn = concatenated([condRowsHigh, videoX], axis: 0)
        let refresh: Bool
        if pab == nil {
            refresh = false
        } else {
            let base = attnBroadcastRefresh(i, steps: UInt32(split.highNFE), k: pabK)
            let stale = pab!.cachedSigma
            let mustRefresh = attnBroadcastMustRefresh(cachedSigma: stale, sigma: sigma, maxStale: pabMaxStale)
            refresh = base || mustRefresh
            if !base, mustRefresh {
                log("高分 PAB 过期重算 i=\(i)：|Δσ|=\(String(format: "%.4f", abs(sigma - stale))) > \(pabMaxStale)")
            }
        }
        slMem("高分 \(i + 1)/\(split.highNFE) forward 前（PAB \(highPabEnabled ? "开" : "关")，refresh=\(refresh)）")
        let outPos = dit.forward(layout: layoutHigh, plan: plan, textStates: textStates,
                                 videoRows: videoIn, audioRows: audioX, rope: ropeHigh,
                                 sigmaV: sigma, shiftV: shiftV, shiftA: shiftA,
                                 attnBcast: pab, attnRefresh: refresh)
        // ★ CFG 分支：与低分循环同款（v = v_neg + cfg·(v_pos − v_neg)，audio 走 positive）。
        let out: H3DiTOutput
        if cfgScale > 0, let textNeg = textStatesNeg {
            let outNeg = dit.forward(layout: layoutHigh, plan: plan, textStates: textNeg,
                                     videoRows: videoIn, audioRows: audioX, rope: ropeHigh,
                                     sigmaV: sigma, shiftV: shiftV, shiftA: shiftA,
                                     attnBcast: nil, attnRefresh: false)
            out = H3DiTOutput(video: outNeg.video + (outPos.video - outNeg.video) * cfgScale,
                              audio: outPos.audio)
        } else {
            out = H3DiTOutput(video: outPos.video, audio: outPos.audio)
        }

        // ★ 音频调度解耦（2026-10-16）：delta_sigma_a 取音频自身调度的相邻差（同低清段说明）。
        let slopeAF = timeShiftSlope(sigma, fromShift: shiftV, toShift: shiftA)
        let da = (audioSigmasHigh[i + 1] - audioSigmasHigh[i]) / slopeAF
        // 与低分循环同一纪律：步内标量张量（4 字节小 buffer）先挂池，eval 之后再放。
        let vScale = H3TensorOps.scalarLike(Float(sigmaNext - sigma), out.video)
        let aScale = H3TensorOps.scalarLike(Float(da), out.audio)
        keepAlive.hold([vScale, aScale])
        videoX = videoX + out.video * vScale
        audioX = updateAudioX(audioX, delta: out.audio * aScale)
        MLX.eval(videoX, audioX)
        // 与低分循环对称：放掉本步临时量，再把全过程复用的条件行挂回去（它跨步存活）。
        keepAlive.release()
        keepAlive.hold([condRowsHigh])
        slMem("高分 \(i + 1)/\(split.highNFE) eval 后")
        // 日志格式与原生 stage1 逐字同款（H3Pipeline.swift 行 553）：sigma/dsigma 均 %.4f，
        // 方括号槽位放本阶段标记 + PAB 复用/刷新（本分段 PAB 关闭时为 pabOff），
        // 耗时 = Int(-Date().timeIntervalSince(stepT))。
        let highBcastMark = pab == nil ? " pabOff" : (refresh ? " bcastRefresh" : " bcastReuse")
        log("SelfLift 高分 \(i + 1)/\(split.highNFE) sigma \(String(format: "%.4f", sigma)) dsigma \(String(format: "%.4f", sigmaNext - sigma)) [high\(highBcastMark)]（\(Int(-Date().timeIntervalSince(stepT)))s）")
    }

    log("SelfLift 一采完成（总 NFE=\(split.totalNFE)，与原调度一致）")

    // ── 6.5 低频锚定（方案 A，用户定档：默认关闭 α_low = 1.0）──────────
    // 默认关闭：α_low ≥ 0.999 时整段不执行，高清出口 latent 原样返回，不做频带替换。
    //   开启后高分收尾做一次频带分离（不动噪声 / schedule / 步数）：
    //       x_out = x_high + (1−α)·( LPF(anchor) − LPF(x_high) )
    //   即低频带按 α 混向升频后的低清 latent，高频带保留高清；LPF 只在 H/W 上做，时间轴不动。
    //   α：低频带里保留高清的比例。1.0=关闭（默认）、0.25=低频 75% 取低清锚、0=低频全取低清。
    //   覆盖：NA_H3_SELFLIFT_LOWANCHOR；<1 即重新开启。
    let lowAnchorAlpha = Float(ProcessInfo.processInfo.environment["NA_H3_SELFLIFT_LOWANCHOR"]
                               .flatMap(Double.init) ?? 1.0)
    if lowAnchorAlpha < 0.999, let anchorNCDHW = liftedLatNCDHW {
        let highNCDHW = toNCDHW(H3TensorOps.unpatchifyVideo(videoX, gridH: gridH, gridW: gridW), c: latC)
        if highNCDHW.shape == anchorNCDHW.shape {
            let hf = highNCDHW.asType(.float32)
            let af = anchorNCDHW.asType(.float32)
            let lowDiff = slLowBandBlur(af) - slLowBandBlur(hf)
            let merged = (hf + lowDiff * (1.0 - lowAnchorAlpha)).asType(videoX.dtype)
            MLX.eval(merged)
            let rmsDiff = (lowDiff * lowDiff).mean().sqrt().item(Float.self)
            let rmsHigh = (hf * hf).mean().sqrt().item(Float.self)
            videoX = H3TensorOps.patchifyVideo(fromNCDHW(merged, c: latC), gridH: gridH, gridW: gridW)
            MLX.eval(videoX)
            keepAlive.hold([merged, videoX])
            if selfLiftDumpEnabled { dumpSelfLift("z_lowanchor.bin", merged) }
            log("SelfLift 低频锚定：开启 α_low=\(String(format: "%.2f", lowAnchorAlpha))"
                + "（低频带 \(Int((1 - lowAnchorAlpha) * 100))% 混向升频低清，高频带不动；"
                + "低频修正量 rms=\(String(format: "%.4f", rmsDiff))/画面 rms=\(String(format: "%.4f", rmsHigh))"
                + "，噪声与 schedule 未改）")
        } else {
            log("SelfLift 低频锚定：跳过（形状不符 high=\(highNCDHW.shape) anchor=\(anchorNCDHW.shape)）")
        }
    } else if lowAnchorAlpha >= 0.999 {
        log("SelfLift 低频锚定：关闭（α_low=\(String(format: "%.2f", lowAnchorAlpha))；"
            + "高清出口 latent 原样返回，未做频带替换）")
    } else {
        log("SelfLift 低频锚定：跳过（未取到升频器出口 latent；rho>0 走像素锚点路径时为预期）")
    }
    // 返回前裁掉前置 refAudio 行：audioX 契约 = 本节点音频行 [audioT*2, 32]（与无续接行为一致）
    let outAudioX = contAudioRows > 0 ? audioX[contAudioRows..<audioX.shape[0], 0..<32] : audioX
    return (videoX, outAudioX)
}

// MARK: - 接线方式
//
//  接线已完成（H3Pipeline stage1 段调用 runH3Stage1WithSelfLift）。注意：
//  · `refined` 必须传 stage1 实际使用的 textStates（不是原始 textHidden）。
//  · `sigmas` 若走 .betaRefined 分支，需确认其非增且仅末项为 0（本 Runner 会校验并抛错）。
//  · 结束后 `dit.adalnTables` 已重算回原分辨率的 ts 表，后续 stage2/直出无需处理。
