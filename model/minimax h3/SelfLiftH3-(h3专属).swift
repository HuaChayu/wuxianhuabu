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
//        → 过渡块（零 NFE：Eq.3 端点预测 → 提升 → 一致修正 → 重加噪 → 补 Euler）
//        → 高分（原 gridH）跑 N-ts 次 NFE
//  对外分辨率与调用前完全一致，因此 **不需要** 改动 stage2 / 直出路径的 layout 假设。
//
//  ── 零 NFE 的落点 ─────────────────────────────────────────────────────
//  低分最后一次前向照常执行（它提供 Eq.3 需要的 v），但**不执行其 Euler 更新**；
//  过渡块用提升后的 x0 重建切换点状态并补一次 (σ_k → σ_next) 的 Euler。
//  低分 NFE = ts，高分 NFE = N-ts，总和不变。
//
//  ── ts 不写死：随面板第一阶段总步数 N 按官方 75% 规则联动 ────────────────
//  N 取自面板「第一阶段总步数」滑杆（范围 4–12），ts 不写死、由官方 75% 比例推导：
//      ts = clamp(floor(0.75 · N), 1, N-1)
//  依据＝官方两个论文实例的比例均为 75%（FLUX.2-Klein 3/4、Z-Image-Turbo 6/8）；
//  本工程 H3 一采 N = 6 → ts = 4（低分 4 次 NFE + 高分 2 次），与原写死值一致。
//  滑杆各档：4→3（测试档）、5→3、6→4、7→5、8→6、9→6、10→7、11→8、12→9
//  （实现见 SelfLiftScheduleSplit.transitionStep(forTotalSteps:)，SelfLiftConfig.transitionStep > 0 可显式覆盖）。
//  官方 SelfLift 硬约束 1 ≤ ts ≤ N-1（nodes.py 行 98 `_validate_schedule`，上界 = sigmas.numel()-2），
//  由 clamp 的上界自动满足；官方节点默认值 ts = 6 在 N = 6 下越界，禁止使用。
//  其余参数对齐官方 SelfLiftH3Sampler 默认：rho = 0.0（learned upscaler 纯 z_lat 提升，
//  像素锚点默认关闭；NA_H3TEST=26 实测 rho>0 注入 VAE 往返伪影）、w_min = w_max = 1.0、
//  lowres_scale = 0.5（官方默认值）。README 行 29/61 的 rho=0.6 仅针对无升频器（none）路径。
//
//  ── 插值口径（照抄官方，禁止自创）──────────────────────────────────────
//  低分条件行降采样：官方 `_resize_keyframes`（nodes.py 行 174-208）= 逐帧 bilinear、
//      align_corners=False（3D keyframe latent 按 (B,T,C,h,w) 折成批次逐帧插值），
//      再做逐 (帧, 通道) 的**加性**均值对齐 `resized + (source_mean - resized_mean)`。
//  直接 latent 提升：官方 H3 路径硬编码 nearest（selflift.py 行 18 默认值 + nodes.py 行 509 实参）。
//  像素锚点重建：官方 `_pixel_anchor_video_single`（selflift.py 行 87-98）= decode →
//      `F.interpolate(size=(H·ratio, W·ratio), mode="bicubic", antialias=True)` → encode，
//      ratio = frames.shape[1] // z0_low.shape[-2]（本工程为 16）。
//  以上三处都由本文件内的 `SelfLiftResample` 表驱动实现（构造表已与 PyTorch 逐点对拍：
//  bicubic antialias 与 bilinear 非 antialias 均 max|diff| ≈ 1e-7）。
//
//  ── 落地前请按你工程当前签名复核（本文件按盘点到的签名书写）─────────────
//  1. SparsePolicy（定义于 H3Layout.swift）的实际类型名（本文件按此名引用）。
//  2. H3AttnBroadcast 的重建/复位方式（本文件用「重建新实例」，若已有 reset() 可换）。
//  3. 本文件只依赖 mlx-swift 的 `MLXArray.take(_:axis:)` / `sorted` / `sum` / `item` /
//     `expandedDimensions` / `concatenated`（均已在工程依赖的 mlx-swift 版本中确认存在）。
//  4. hooks 的两个闭包由 H3Pipeline 侧注入 —— 这样 Runner 不依赖 H3VAE 内部字段。
//     · decodeToPixels / encodeToLatent 直接包 H3VAEDecoder.decode / H3VAEEncoder.encodeVideo
//     · 不再需要「latent ×2 上采样」注入：官方 H3 路径固定 nearest，已在文件内实现
//
//  ⚠️ 本文件未改动 H3Pipeline.swift；接线方式见文件末尾注释。
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

// 官方 learned latent upscaler（时间维感知）的按需加载：以 NA_H3_UPSCALER 环境变量当前值为键缓存，
// 支持进程内 A/B 切换（如 h3管线自检 NA_H3TEST=21 里先 nearest 后 learned 顺序跑）。
// 默认（不设 / 空）→ H3LatentUpscaler（官方 learned upscaler，已在 1344×768 实测显著消除面部重影）；
// NA_H3_UPSCALER=nearest → 回退 slNearestLiftLatent（与 h3_53 等历史产物一致）。
private var h3LearnedUpscalerCache: (key: String, up: H3LatentUpscaler?)? = nil
private func h3LearnedUpscalerForCurrentEnv() -> H3LatentUpscaler? {
    let key = ProcessInfo.processInfo.environment["NA_H3_UPSCALER"] ?? ""
    if let c = h3LearnedUpscalerCache, c.key == key { return c.up }
    var up: H3LatentUpscaler? = nil
    if key != "nearest" {
        let modelDir = "\(CommonPaths.modelRoot)/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-mlx-6bit"
        let modelRootURL = URL(fileURLWithPath: modelDir, isDirectory: true)
        let wURL = { (name: String) -> URL in
            let direct = modelRootURL.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: direct.path) { return direct }
            return FileFinder.first(named: name, under: modelRootURL) ?? direct
        }
        let path = ProcessInfo.processInfo.environment["NA_H3_UPSCALER_PATH"]
            ?? wURL("minimax_h3_latent_upscaler_3d_bf16.safetensors").path
        if let w = try? H3Weights(url: URL(fileURLWithPath: path)) {
            up = try? H3LatentUpscaler(weights: w)
        } else {
            NSLog("NA_H3_UPSCALER!=nearest 但 learned 权重加载失败：%@", path)
        }
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

// MARK: - 官方重采样内核（照抄 selflift.py / nodes.py 的插值调用）

/// PyTorch `F.interpolate` 的表驱动等价实现。
///
/// 官方四处调用里，本工程 H3 一采用到其中三处，逐条照抄、不换自创插值：
///   · 直接 latent 提升        `mode="nearest"`                      — selflift.py 行 18 / 54，H3 硬编码
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

/// [T,H,W,C] → [T,outH,outW,C]：**最近邻**直接 latent 提升
/// （官方 H3 路径硬编码 `mode="nearest"`：selflift.py 行 18 / 54、nodes.py 行 509）
private func slNearestLiftLatent(_ x: MLXArray, outH: Int, outW: Int,
                                 keepAlive: SelfLiftTensorKeepAlive? = nil) -> MLXArray {
    let h = SelfLiftResample.resizeAxis(
        x, axis: 1, table: SelfLiftResample.nearestTable(inSize: x.shape[1], outSize: outH),
        keepAlive: keepAlive)
    let out = SelfLiftResample.resizeAxis(
        h, axis: 2, table: SelfLiftResample.nearestTable(inSize: x.shape[2], outSize: outW),
        keepAlive: keepAlive)
    keepAlive?.hold([h, out])
    return out
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
    // ★ 续接初值（保留接口兼容）：2026-09-21 起 H3Pipeline 传 nil——前置尾段不再作采样
    //   初值加噪重画（warmStart 路线已废），改走 contKeyAnchors 条件行锚点。
    //   continuationWinLatentT=0 表示非续接，保持原纯噪声初值行为。
    continuationLatent: MLXArray? = nil,
    continuationWinLatentT: UInt32 = 0,
    // warmStart：2026-09-21 起不再消费（续接走 keyframe 锚点、纯噪声起步）；保留接口兼容。
    continuationStartSigma: Double? = nil,
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
    audioContinuationRows: MLXArray? = nil
) throws -> (videoX: MLXArray, audioX: MLXArray) {

    // 解耦模式（本工程调试扩展，A/B 用；默认关闭，不触碰任何默认值）：
    //   开关在调用方（H3Pipeline）按环境变量注入 cfg.decoupleSigmaK / decoupleSigmaNext /
    //   decoupleHighSteps；本 Runner 只消费，不读取环境（保持只读参数、便于单测）。
    //   动机（2026-09-17 重影根因）：低清 σ 大时运动位置量化歧义 → x0 预测为多层均值（双层影）；
    //   原路径被迫提 σ_next 阈值让低清变浅（残留少），但源头歧义还在。解耦 = 低清浅跑 + 浅残留
    //   重加噪 + 高清等距收细节：UI 默认 σ_k=0.9（L=2、σ_k≈0.9231）、σ_next=σ_k、高清 3 步
    //   （总 NFE 5 < 官方 N=6，不浪费算力）。

    let split: SelfLiftScheduleSplit
    // ★ 续接激活（2026-09-20）：前置窗口 > 0 即续接段；低清曲线从 warmStart 等距起步，
    //   与单次生成同构（「续接 = 无数个单次」）。
    let continuationActive = continuationWinLatentT > 0
    // 续接激活时强制独立曲线（官方 75% 规则会被外层压缩后的 N=2 调度吃掉）
    let decoupleK = cfg.decoupleSigmaK ?? (continuationActive ? 0.9 : nil)
    if let decoupleK = decoupleK {
        // 低清独立曲线：官方 shift 公式（同 shiftV）跑到 σ_k。
        // σ_{L-1} = shift/(L+shift-1)（由 b=1/L 代入 timeShiftSigma 化简），
        // 反推 L = round(shift/σ_k − shift + 1)（shift=12、σ_k=0.9 → L=2、σ_k≈0.9231）
        let shiftD = max(shiftV, 1.0)
        var L: Int
        if cfg.decoupleLowSteps > 0 {
            // 固定低清步数：不再由 σ_k 反推；低清终点直接取 L 步曲线实际终点 σ_{L-1}，
            // 即「过渡用的值」——低清跑完即处于高清起点水平，无需过渡块补 Euler。
            L = min(max(Int(cfg.decoupleLowSteps), 2), 64)
        } else {
            L = max(Int((shiftD / decoupleK - shiftD + 1).rounded()), 2)
            L = min(L, 64)
        }
        let lowBase = sigmaSchedule(steps: UInt32(L), shift: shiftD)   // [σ0…σ_{L-1}, 0]
        // 高清起点：缺省 = 低清终点（等深，固定步数下即过渡用的值）；显式给则独立选择
        let lowEnd = lowBase[L - 1]
        let sigmaNext = cfg.decoupleSigmaNext ?? lowEnd
        // 低清曲线 = 前 L 项 + 末项占位 σ_next：低清最后一步前向的 Euler（videoX 会被过渡块
        // 丢弃）仍需把 audioX 推到 σ_next 水平（与高清起点一致），不能推到 0。
        var lowSigmas = Array(lowBase[0..<L])
        lowSigmas.append(sigmaNext)
        // ★ 续接：低清曲线从 warmStart 等距 L 步到 σ_next（与单次同构，前置尾段不被深噪声淹没）
        if continuationActive, let ws = continuationStartSigma {
            let span = sigmaNext - ws
            lowSigmas = (0...L).map { ws + span * Double($0) / Double(L) }
            log("SelfLift 续接调度：低清从 warmStart \(String(format: "%.4f", ws)) 等距 \(L) 步 → σ_next \(String(format: "%.4f", sigmaNext))")
        }
        // 高清独立曲线：在 [0, σ_next] 区间等距离散化（修复 2026-09-17）
        // 旧实现 = 标准 M 步曲线整体缩放 ×σ_next，shift=12 下 σ 被压扁在 [0.6, σ_next]
        // 区间（M=3 → [0.706, 0.678, 0.605, 0]）：浅区完全无采样，最后一步 0.605→0
        // 一次跳 60.5% 噪声，最终帧 = σ≈0.6 处 x0 预测，低清残留占内容 62.8%
        // （耦合干净档仅 15.5%）→ 时间维重影必然洗不掉。
        // 等距后：每步去噪均匀（dσ = σ_next/M），终点 x0 在 σ≈σ_next/3 处预测，
        // 残留占比降至 ~12.8%，与耦合安全档同水平。
        var M = cfg.decoupleHighSteps > 0 ? cfg.decoupleHighSteps : 6
        // v5：允许 M=1（固定最后一步高清：σ_next → 0 一次直达）；0 缺省仍 6。
        M = max(M, 1)
        let highSigmas = (0...M).map { sigmaNext * (1.0 - Double($0) / Double(M)) }
        split = try SelfLiftScheduleSplit.makeDecoupled(lowSigmas: lowSigmas, highSigmas: highSigmas)
        log("SelfLift 解耦调度：低清 L=\(L)（终点 σ_k=\(String(format: "%.4f", split.sigmaK))）"
            + "→ 提升 → 高清 M=\(M)（σ_next=\(String(format: "%.4f", split.sigmaNext))）；"
            + "低清终点即过渡值（无需过渡补 Euler）；"
            + "高清曲线 [\(highSigmas.map { String(format: "%.3f", $0) }.joined(separator: ", "))]；"
            + "原调度 N=\(sigmas.count - 1) 已绕开")
    } else {
        // ts 不写死：cfg.transitionStep = 0（默认）时按官方 75% 规则由 N 推导，
        // 即 ts = clamp(floor(0.75 · N), 1, N-1)，N = sigmas.count - 1（面板第一阶段总步数滑杆 4–12）。
        split = try SelfLiftScheduleSplit.make(sigmas: sigmas, config: cfg)
    }

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
    // ★ 续接：总 latentT = 前置窗口 + 本节点（布局/注意力全段对齐）；非续接退化为本节点
    let totalLatT = Int(latT) + Int(continuationWinLatentT)
    // ★ 2026-09-21 keyframe 锚点路线：H3Pipeline 已传 continuationLatent=nil，前置尾段改走
    //   contKeyAnchors/preRefBlocks 条件行锚点、不进采样状态，视频段只含新段 latT；
    //   仅旧初值路线（continuationLatent != nil）才把前置窗口拼进采样初值（totalLatT）。
    let layoutLatT = (continuationActive && continuationLatent != nil) ? UInt32(totalLatT) : UInt32(latT)
    log("SelfLift：σ 切分 ts=\(split.transitionStep)（低分 NFE=\(split.lowNFE) / 高分 NFE=\(split.highNFE)），"
        + "σ_k=\(String(format: "%.4f", split.sigmaK)) → σ_next=\(String(format: "%.4f", split.sigmaNext))，"
        + "latent \(latH)×\(latW) → 低分 \(latHl)×\(latWl)")

    // ── 1. 低分条件行（官方 `_resize_keyframes`：逐帧 bilinear + 加性均值对齐）──
    // 分段方式与 layout 的 cond 行预算**逐块**对齐（这是本分支的关键不变量）：
    //   · ref2va（refBlocks 非空）：每段 = 一个参考块，用**该块自己的 latent 网格**还原，
    //     低分尺寸按 cfg.lowResScale 经 slLowResSize 同口径缩放；layout 侧传同序同尺寸的 refs。
    //   · fl2va（refBlocks 为空）：每段 = 一个目标帧（fRows 行）；layout 侧仍用 keyframes [.first,.last]。
    let fRows = gridH * gridW
    // ★ 续接：ref2va 时前置参考块并入块列表（段序 [前置refImg][本节点refImg]，与 condRows 一致）；
    //   condOffset = 前置 cond 段行数（fl2va/ref2va 切行都要跳过该段）
    let allRefBlocks = preRefBlocks + refBlocks
    let condOffset = Int(preCondRows)
    var condSegs: [MLXArray] = []
    var condSegsHigh: [MLXArray] = []   // fl2va：低清 keyframe 提升回全分辨率（替代全分辨率 keyframe 注入）
    var lowRefBlocks: [RefBlock] = []
    if allRefBlocks.isEmpty {
        let nCond = (condRowsFull.shape[0] - condOffset) / fRows
        for c in 0..<nCond {
            let seg = condRowsFull[(condOffset + c * fRows)..<(condOffset + (c + 1) * fRows)]
            let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gridH, gridW: gridW)   // [1,latH,latW,C]
            let low = slResizeKeyframeLatent(lat, outH: latHl, outW: latWl,
                                             keepAlive: keepAlive)                   // [1,latHl,latWl,C]
            let rows = H3TensorOps.patchifyVideo(low, gridH: gHl, gridW: gWl)
            keepAlive.hold([lat, low, rows])
            condSegs.append(rows)
            // ★ 高清条件 = 同一份低清 keyframe 直接提升回全分辨率（nearest，与官方 latent
            //   提升同口径）：背景结构坐标与低清循环固化的一致 → 消除"低清残留 vs 全分辨率
            //   keyframe 重注入"的双套网格错位（h3_62 背景重影根因，h3_64 ref2va 复盘）。
            let up = slNearestLiftLatent(low, outH: latH, outW: latW,
                                         keepAlive: keepAlive)                       // [1,latH,latW,C]
            let rowsHigh = H3TensorOps.patchifyVideo(up, gridH: gridH, gridW: gridW)
            keepAlive.hold([up, rowsHigh])
            condSegsHigh.append(rowsHigh)
        }
    } else {
        var offset = condOffset
        // ★ 2026-09-21 修复：ref2va 前置尾段 keyframe 锚点先按 keyframe 口径切行（每段 fRows，
        //   低清重采样同 fl2va），再切参考块；段序与 H3Pipeline 注入的 [preCond][contKey][preRef][ownRef]
        //   严格对齐（此前 ref2va 只切参考块，contKey 段未计入 → 尾帧锚点丢失）。
        for _ in 0..<contKeyAnchors.count {
            guard offset + fRows <= condRowsFull.shape[0] else {
                throw NSError(domain: "H3SelfLift", code: 7, userInfo: [NSLocalizedDescriptionKey:
                    "ref2va contKey 段行数超出 condRowsFull：\(offset + fRows) > \(condRowsFull.shape[0])"])
            }
            let seg = condRowsFull[offset..<(offset + fRows)]
            offset += fRows
            let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gridH, gridW: gridW)   // [1,latH,latW,C]
            let low = slResizeKeyframeLatent(lat, outH: latHl, outW: latWl,
                                             keepAlive: keepAlive)                   // [1,latHl,latWl,C]
            let rows = H3TensorOps.patchifyVideo(low, gridH: gHl, gridW: gWl)
            keepAlive.hold([lat, low, rows])
            condSegs.append(rows)
        }
        for b in allRefBlocks {
            // 本工程 ref2va 只产出 image 参考块（H3Pipeline.swift:234）；其它类型没有条件行构造口径，
            // 直接报错而不是悄悄错位。
            guard b.kind == .image else {
                throw NSError(domain: "H3SelfLift", code: 3, userInfo: [NSLocalizedDescriptionKey:
                    "SelfLift ref2va 目前只支持 image 参考块，收到 \(b.kind)"])
            }
            let gh = Int(b.latentH) / Int(H3Const.patchH)
            let gw = Int(b.latentW) / Int(H3Const.patchW)
            let n = gh * gw                                   // = H3Layout.refFrameRows(b)
            guard offset + n <= condRowsFull.shape[0] else {
                throw NSError(domain: "H3SelfLift", code: 4, userInfo: [NSLocalizedDescriptionKey:
                    "参考块行数超出 condRowsFull：\(offset + n) > \(condRowsFull.shape[0])"])
            }
            let seg = condRowsFull[offset..<(offset + n)]
            offset += n
            let lat = H3TensorOps.unpatchifyVideo(seg, gridH: gh, gridW: gw)         // [1,latH,latW,C]
            let lh = slLowResSize(Int(b.latentH), scale: cfg.lowResScale)
            let lw = slLowResSize(Int(b.latentW), scale: cfg.lowResScale)
            let low = slResizeKeyframeLatent(lat, outH: lh, outW: lw,
                                             keepAlive: keepAlive)                   // [1,lh,lw,C]
            let rows = H3TensorOps.patchifyVideo(low, gridH: lh / Int(H3Const.patchH),
                                                 gridW: lw / Int(H3Const.patchW))
            keepAlive.hold([lat, low, rows])
            condSegs.append(rows)
            lowRefBlocks.append(RefBlock(kind: .image, latentH: UInt32(lh), latentW: UInt32(lw),
                                         latentT: b.latentT, audioT: b.audioT))
        }
        guard offset == condRowsFull.shape[0] else {
            throw NSError(domain: "H3SelfLift", code: 5, userInfo: [NSLocalizedDescriptionKey:
                "参考块行数与 condRowsFull 不一致：\(offset) vs \(condRowsFull.shape[0])"])
        }
    }
    let condRowsLow = condSegs.count == 1 ? condSegs[0] : concatenated(condSegs, axis: 0)
    // ★ fl2va 高清条件行 = 低清 keyframe 提升回全分辨率（与低分注入同一份结构，避免双套网格错位）；
    //   ref2va 保持原行为（refs 非端点锚定，无错位问题，直接用原生全分辨率参考块）。
    let condRowsHigh: MLXArray
    if allRefBlocks.isEmpty {
        condRowsHigh = condSegsHigh.count == 1 ? condSegsHigh[0] : concatenated(condSegsHigh, axis: 0)
        keepAlive.hold([condRowsHigh])
        MLX.eval(condRowsHigh)
    } else {
        // ★ 续接：高分段截掉前置 cond 段（ref2va 注入段序 [contKey][preRef][ownRef]，contKey 走原生高清行）
        condRowsHigh = condOffset > 0 ? condRowsFull[condOffset..<condRowsFull.shape[0]] : condRowsFull
    }

    // 修复不变量：layout 侧 cond 行预算必须**严格等于**实际传入的条件行数，
    // 否则多余行会被 H3Transformer 按 layout.segments 静默丢弃、视频段起点整体前移。
    let budgetLow = allRefBlocks.isEmpty ? (2 + contKeyAnchors.count) * gHl * gWl : contKeyAnchors.count * gHl * gWl + lowRefBlocks.reduce(0) { $0 + Int(refFrameRows($1)) }
    let budgetHigh = allRefBlocks.isEmpty ? (2 + contKeyAnchors.count) * fRows : contKeyAnchors.count * fRows + allRefBlocks.reduce(0) { $0 + Int(refFrameRows($1)) }
    guard budgetLow == condRowsLow.shape[0], budgetHigh == condRowsHigh.shape[0] else {
        throw NSError(domain: "H3SelfLift", code: 6, userInfo: [NSLocalizedDescriptionKey:
            "layout cond 预算与实际条件行不一致：低分 \(budgetLow) vs \(condRowsLow.shape[0])，"
            + "高分 \(budgetHigh) vs \(condRowsHigh.shape[0])"])
    }
    log("SelfLift 条件行：\(allRefBlocks.isEmpty ? "fl2va keyframes ×\(2 + contKeyAnchors.count)（含续接锚点 ×\(contKeyAnchors.count)；高清=低清提升，无重注入）" : "ref2va 锚点 ×\(contKeyAnchors.count) + \(allRefBlocks.count) 参考块（含前置 \(preRefBlocks.count)）")"
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
    // keyframes（ref2va 也注入，H3Pipeline 已把 contKeyRows 排在 preRef 之前）；fl2va 追加 [.first,.last]。
    let layoutLow = PackedLayout(textLen: textLen, latentT: layoutLatT,
                                 latentH: UInt32(latHl), latentW: UInt32(latWl),
                                 audioT: audioT,
                                 keyframes: contKeyAnchors + (allRefBlocks.isEmpty ? [.first, .last] : []),
                                 frameCount: frameCount,
                                 refs: lowRefBlocks,
                                 contRefs: contAudioRef)
    // ref2va 的文本行含 4 个视觉块，AdaLN 模态标签必须与原生 layout 一样下发（H3Pipeline.swift:390），
    // 否则视觉块的 mod 行会落到 text 模态（H3Layout.swift:637 起），条件参考被打错调制。
    layoutLow.textTags = textTags
    let tsLow = collectScheduleTs(layout: layoutLow, sigmas: split.lowSigmas,
                                  shiftV: shiftV, shiftA: shiftA, aug: CondNoiseAug())
    dit.precomputeAdaln(ts: tsLow)
    let ropeLow = buildRope(layout: layoutLow, invFreq: dit.invFreq)

    // ★ 续接初值（2026-09-20）：前置尾段 clean latent [winT,latH,latW,latC] 整段降采样到
    //   低清网格 → patchify 成低清段前缀行（与低清段同网格同构），新段仍从噪声起步，
    //   最后整体加噪到 warmStart（保留前置结构，后续由低清曲线等距收敛）。
    var videoX: MLXArray
    if continuationActive, let contLat = continuationLatent {
        let contLow = slResizeKeyframeLatent(contLat, outH: latHl, outW: latWl,
                                             keepAlive: keepAlive)   // [winT,latHl,latWl,C]
        let contRows = H3TensorOps.patchifyVideo(contLow, gridH: gHl, gridW: gWl)
        let newRows = MLXRandom.normal([Int(latT) * gHl * gWl, vpatch], key: MLXRandom.key(seed))
        videoX = concatenated([contRows, newRows], axis: 0)
        let s0 = continuationStartSigma ?? split.lowSigmas[0]
        let epsV = MLXRandom.normal(videoX.shape, key: MLXRandom.key(seed &+ 2)).asType(videoX.dtype)
        videoX = videoX * H3TensorOps.scalarLike(Float(1.0 - s0), videoX)
            + epsV * H3TensorOps.scalarLike(Float(s0), videoX)
        keepAlive.hold([contLow, contRows, epsV])
        log("SelfLift 续接初值：前置尾段 \(contLat.shape[0]) 帧降采样 → \(contRows.shape) + 新段 \(newRows.shape)，整体加噪 σ0=\(String(format: "%.4f", s0))")
    } else {
        videoX = MLXRandom.normal([Int(latT) * gHl * gWl, vpatch], key: MLXRandom.key(seed))
    }
    var audioX: MLXArray
    if contAudioRows > 0, let preAudio = audioContinuationRows {
        // ★ 音频续接：audioX = [前置真实尾段 | 本节点噪声]，前置行经 refAudio 段注入注意力，
        //   永远不被更新（never-denoised），本节点行从纯噪声采样。
        let preF32 = preAudio.asType(.float32)
        let newNoise = MLXRandom.normal([Int(audioT) * 2, 32], key: MLXRandom.key(seed &+ 1))
        audioX = concatenated([preF32, newNoise], axis: 0)
        keepAlive.hold([preF32])
        log("SelfLift 音频续接：前置尾段 \(contAudioRows) 行（refAudio 锚点，never-denoised）+ 本节点 \(Int(audioT) * 2) 行纯噪声起步")
    } else {
        audioX = MLXRandom.normal([Int(audioT) * 2, 32], key: MLXRandom.key(seed &+ 1))
    }
    MLX.eval(videoX, audioX)

    let pabK = attentionBroadcastK
    let makePab: () -> H3AttnBroadcast? = { pabK > 1 ? H3AttnBroadcast(count: dit.blocks.count) : nil }
    var pab = makePab()

    // ── 3. 低分循环（第 ts 步只取端点预测，不做 Euler）────────────────────
    slMem("低分段 · 起步（DiT 全权重 + 低分 PAB \(pab == nil ? "关闭" : "开启")）")
    var lowCleanX0: MLXArray? = nil
    for i in 0..<split.lowNFE {
        let stepT = Date()
        let sigma = split.lowSigmas[i]
        let sigmaNext = split.lowSigmas[i + 1]
        dit.sparsePolicy = sparsePolicy

        let plan = buildTimestepPlanGlobal(layout: layoutLow, sigmaV: sigma,
                                           shiftV: shiftV, shiftA: shiftA,
                                           aug: CondNoiseAug(), globalTs: dit.adalnTables!.ts)
        let videoIn = concatenated([condRowsLow, videoX], axis: 0)
        let refresh = pab == nil ? false : attnBroadcastRefresh(i, steps: UInt32(split.lowNFE), k: pabK)
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

        let slopeAF = timeShiftSlope(sigma, fromShift: shiftV, toShift: shiftA)
        let da = (timeShiftSigma(sigmaNext, fromShift: shiftV, toShift: shiftA)
                - timeShiftSigma(sigma, fromShift: shiftV, toShift: shiftA)) / slopeAF

        // 步内缩放因子（4 字节标量张量）先取出并挂进强引用池：它们是本步懒图（加噪/音频更新）
        // 的输入 buffer，若在 eval 之前就失去 Swift 侧引用，小 buffer 会被分配器空闲池回收。
        let vScale = H3TensorOps.scalarLike(Float(sigmaNext - sigma), out.video)
        let aScale = H3TensorOps.scalarLike(Float(da), out.audio)
        keepAlive.hold([vScale, aScale])
        if i == split.lowNFE - 1 {
            // Eq.3：clean endpoint；丢弃本步 video 的 Euler 更新
            lowCleanX0 = SelfLiftCore.cleanEndpoint(x: videoX, velocity: out.video, sigma: sigma,
                                                    keepAlive: keepAlive)
            // 音频不参与提升，按正常 Euler 边界步跟到 σ_next（只更新本节点行）
            audioX = updateAudioX(audioX, delta: out.audio * aScale)
        } else {
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
    guard let x0Low = lowCleanX0 else { throw SelfLiftScheduleError.tooFewSteps(0) }

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
    MLX.eval(x0Low)
    Stream.gpu.synchronize()

    // ── 3.6 lowOnly 出口（IC 链路）：低清段完成即返回，跳过过渡块 + 高分循环 ──
    // H3 一采只出低分半清 latent，高分由 LTX 二采负责（升频×2 + 高清 IC 精修，factor=2）。
    // 返回形态：NCDHW [1,C,T,latHl,latWl]（半清网格），供 H3Pipeline 的 adapter 直转 LTX 域。
    if lowOnly {
        let z0LowLat = toNCDHW(H3TensorOps.unpatchifyVideo(x0Low, gridH: gHl, gridW: gWl), c: latC)
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

    // ── 4. 过渡块（零 NFE）──────────────────────────────────────────────
    // 4.1 直接 latent 提升：默认官方 learned upscaler（H3LatentUpscaler，时间维感知，
    //     实测 1344×768 高分辨率下显著消除帧间双像/面部重影，见 h3_scene_ab 自检结论）；
    //     NA_H3_UPSCALER=nearest 时可回退官方 old 路径 fixed nearest（selflift.py 行 18 / 54）。
    let liftLatentRows: (MLXArray) -> MLXArray = { rows in
        let lat = H3TensorOps.unpatchifyVideo(rows, gridH: gHl, gridW: gWl)      // [latT,latHl,latWl,C]
        let up: MLXArray
        let useNearest = ProcessInfo.processInfo.environment["NA_H3_UPSCALER"] == "nearest"
        if !useNearest, let us = h3LearnedUpscalerForCurrentEnv() {
            let zIn = toNCDHW(lat, c: latC)                                     // [1,C,T,h,w]
            let zUp = us.apply(zIn)                                             // [1,C,T,H,W]（×2）
            var upLat = fromNCDHW(zUp, c: latC)                                 // [latT,latH,latW,C]
            // ★ 2026-09-18：learned upscaler 固定 ×2，当目标 latent 半尺寸为奇数时（如 864×480
            //   → 30×54，低分 16×28 取偶），×2 后 32×56 ≠ 目标 30×54，过渡块 patchify 直接
            //   reshape 崩溃。×2 与目标不一致时补一次 nearest resize 对齐（与官方 old 路径同口径）。
            if upLat.shape[1] != latH || upLat.shape[2] != latW {
                upLat = slNearestLiftLatent(upLat, outH: latH, outW: latW,
                                            keepAlive: keepAlive)
            }
            up = upLat
        } else {
            up = slNearestLiftLatent(lat, outH: latH, outW: latW,
                                     keepAlive: keepAlive)                       // [latT,latH,latW,C]
        }
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
    // NA_DUMP_SELFLIFT=1：导出 z0_low / z_lat / z_pix / z0_high 四份 latent（NCDHW float32，
    // 文件头部 5 个 Int32 为 shape），供离线验证两路提升差异与重影根因（h3_53 复盘）。
    // 注意：debug 模式会在过渡块内提前物化三份高清 latent，内存峰值略升，仅诊断时开启。
    let dumpSelfLift: (String, MLXArray) -> Void = { name, arr in
        let f = arr.asType(.float32)
        MLX.eval(f)
        let floats = f.asArray(Float.self)
        let dims = arr.shape.map { Int32($0) }
        var header = dims
        var data = Data(bytes: &header, count: header.count * MemoryLayout<Int32>.size)
        floats.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
        let dir = ProcessInfo.processInfo.environment["NA_DUMP_SELFLIFT_DIR"] ?? "/tmp/h3_selflift_dump"
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try? data.write(to: URL(fileURLWithPath: dir + "/" + name))
        pipelineLog("SELFLIFT-DUMP \(name) shape=\(arr.shape) floats=\(floats.count) → \(dir)/\(name)")
    }
    if ProcessInfo.processInfo.environment["NA_DUMP_SELFLIFT"] == "1" {
        // z0_low 是低清网格 rows，先还原成 NCDHW latent 再落盘（与 z_lat/z_pix 同域可比）
        let z0LowLat = toNCDHW(H3TensorOps.unpatchifyVideo(x0Low, gridH: gHl, gridW: gWl), c: latC)
        dumpSelfLift("z0_low.bin", z0LowLat)
    }
    let consistencyRows: (MLXArray, MLXArray) -> MLXArray = { zLat, zPix in
        let a = toNCDHW(H3TensorOps.unpatchifyVideo(zLat, gridH: gridH, gridW: gridW), c: latC)
        let b = toNCDHW(H3TensorOps.unpatchifyVideo(zPix, gridH: gridH, gridW: gridW), c: latC)
        let fixed = SelfLiftCore.artifactAwareConsistencyLift(
            zLat: a, zPix: b, rho: cfg.rho, wMin: cfg.wMin, wMax: cfg.wMax, channelAxis: 1,
            keepAlive: keepAlive)
        if ProcessInfo.processInfo.environment["NA_DUMP_SELFLIFT"] == "1" {
            dumpSelfLift("z_lat.bin", a)
            dumpSelfLift("z_pix.bin", b)
            dumpSelfLift("z0_high.bin", fixed)
        }
        let rowsOut = H3TensorOps.patchifyVideo(fromNCDHW(fixed, c: latC), gridH: gridH, gridW: gridW)
        keepAlive.hold([a, b, fixed, rowsOut])
        return rowsOut
    }

    // ★ 修复（2026-09-20）：过渡噪声行数必须与提升后 latent 行数一致 —— 旧初值续接场景下
    //   latent 含前置窗口（totalLatT = latT + continuationWinLatentT），旧代码用 latT 导致
    //   Eq.10 重加噪时 noise 与 z0High broadcast 崩溃（(59472,96) vs (37296,96)）。
    //   ★ 2026-09-21：keyframe 锚点路线（continuationLatent=nil）下视频段只含新段 latT，
    //   过渡噪声按 layoutLatT 走（与布局/rope/videoX 同一口径）。
    let transitionNoise = MLXRandom.normal([Int(layoutLatT) * gridH * gridW, vpatch],
                                           key: MLXRandom.key(seed &+ 7))
    // ★ 物化点 D：过渡噪声先物化，别把随机数 kernel 也挤进过渡块的分配浪里。
    MLX.eval(transitionNoise)
    slDiag("D 过渡噪声完成")
    keepAlive.hold(transitionNoise)
    let liftT = Date()
    slDiag("4.4 进入零 NFE 过渡块（lift → 噪声重放 → Euler）")
    // ★ 防御断言（2026-09-20）：过渡噪声行数必须等于提升后 latent 行数（layoutLatT×gridH×gridW），
    //   防止续接/尺寸口径再漂移导致 Eq.10 重加噪 broadcast 崩溃。
    precondition(transitionNoise.shape[0] == Int(layoutLatT) * gridH * gridW,
                 "过渡噪声行数与提升后 latent 行数不符：\(transitionNoise.shape[0]) != \(Int(layoutLatT) * gridH * gridW)")
    videoX = SelfLiftTransition.run(
        split: split,
        cfg: cfg,
        lowCleanX0: x0Low,
        noise: transitionNoise,
        liftLatent: liftLatentRows,
        liftPixelAnchor: cfg.needsPixelAnchor ? liftPixelRows : nil,
        consistencyLift: consistencyRows,
        noiseScaling: { x0, s, n in
            SelfLiftCore.noiseScaling(sigma: s, noise: n, x0: x0, keepAlive: keepAlive) },
        eulerStep: { x, x0, s, sn in
            SelfLiftCore.eulerStep(x: x, x0: x0, sigma: s, sigmaNext: sn, keepAlive: keepAlive) },
        decoupled: cfg.decoupleSigmaK != nil,
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
    // dsigma = σ_next − σ_k 即本块补完的那段 Euler；方括号槽位放提升前后尺寸。
    log("SelfLift lift sigma \(String(format: "%.4f", split.sigmaK)) dsigma \(String(format: "%.4f", split.sigmaNext - split.sigmaK)) [lift \(latWl)×\(latHl)→\(latW)×\(latH)]（\(Int(-Date().timeIntervalSince(liftT)))s）")

    // ── 5. 高分上下文重建（分辨率/序列长度变了，必须整表重算）──────────────
    // 同上：续接前置尾段 keyframe 锚点无条件进 keyframes（ref2va 也注入）；fl2va 追加 [.first,.last]。
    let layoutHigh = PackedLayout(textLen: textLen, latentT: layoutLatT,
                                  latentH: UInt32(latH), latentW: UInt32(latW),
                                  audioT: audioT,
                                  keyframes: contKeyAnchors + (allRefBlocks.isEmpty ? [.first, .last] : []),
                                  frameCount: frameCount,
                                  refs: allRefBlocks,
                                  contRefs: contAudioRef)
    layoutHigh.textTags = textTags
    let tsHigh = collectScheduleTs(layout: layoutHigh, sigmas: split.highSigmas,
                                   shiftV: shiftV, shiftA: shiftA, aug: CondNoiseAug())
    dit.precomputeAdaln(ts: tsHigh)
    let ropeHigh = buildRope(layout: layoutHigh, invFreq: dit.invFreq)
    // ★ 修复①（本轮，主体）：高分段的 PAB 广播缓存只在「确实存在复用步」时才建。
    //   刷新调度 attnBroadcastRefresh 在短高分里是全覆盖的：N=7 → highNFE=2，而
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
    let highHasReuse = (0..<split.highNFE).contains {
        !attnBroadcastRefresh($0, steps: UInt32(split.highNFE), k: pabK)
    }
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
                                           aug: CondNoiseAug(), globalTs: dit.adalnTables!.ts)
        let videoIn = concatenated([condRowsHigh, videoX], axis: 0)
        let refresh = pab == nil ? false : attnBroadcastRefresh(i, steps: UInt32(split.highNFE), k: pabK)
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

        let slopeAF = timeShiftSlope(sigma, fromShift: shiftV, toShift: shiftA)
        let da = (timeShiftSigma(sigmaNext, fromShift: shiftV, toShift: shiftA)
                - timeShiftSigma(sigma, fromShift: shiftV, toShift: shiftA)) / slopeAF
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
    // 返回前裁掉前置 refAudio 行：audioX 契约 = 本节点音频行 [audioT*2, 32]（与无续接行为一致）
    let outAudioX = contAudioRows > 0 ? audioX[contAudioRows..<audioX.shape[0], 0..<32] : audioX
    return (videoX, outAudioX)
}

// MARK: - 接线方式（不改 H3Pipeline.swift 的其它部分）
//
//  在 H3Pipeline.generateVideo 的 stage1 段（当前为 `for i in 0..<stage1SegmentCount { ... }`，约行 456–496）：
//
//      // ts 不写死：transitionStep 传 0 → 由面板「第一阶段总步数」滑杆 N 按官方 75% 规则自动推导
//      //   ts = clamp(floor(0.75 · N), 1, N-1)   滑杆各档：N=4→3（测试档）、6→4、8→6、12→9
//      //   N = stage1SegmentCount；官方硬约束 1 ≤ ts ≤ N-1 由 clamp 自动满足
//      let selfLift = SelfLiftConfig(enabled: <开关>,
//                                    transitionStep: 0,        // 0 / 不传 = 按 N 自动推导；> 0 = 显式覆盖
//                                    lowResScale: 0.5,
//                                    rho: 0.0, wMin: 1.0, wMax: 1.0)   // rho=0 官方默认：纯 z_lat 提升，跳过 VAE 往返
//      if selfLift.enabled, stage1SegmentCount >= 4 {          // 滑杆 4–12 均可用；ts 随 N 自动变化，不再判 N == 6
//          let hooks = SelfLiftH3Hooks(
//              decodeToPixels:   { decoder!.decode($0) },
//              encodeToLatent:   { vaeEncoder!.encodeVideo($0) },
//              // 过渡块出口屏障后放掉像素锚点 VAE 权重（≈5.3GB），高分段不必背它
//              releasePixelVAE:  { slVae = nil; slVaeWeights = nil })
//          let r = try runH3Stage1WithSelfLift(
//              dit: dit!, textStates: refined, condRowsFull: condRows,
//              // ref2va 必须把原生 layout 的同一份参考块与文本模态标签透传进来（fl2va 传空即可）：
//              //   refBlocks 决定 layout 的 cond 行预算（= condRows 的行数），漏传会把视频段起点前移
//              //   textTags 决定文本行里各视觉块的 AdaLN 模态，漏传视觉块会按 text 模态调制
//              refBlocks: refBlocks, textTags: textTagsForLayout,
//              latT: Int(latentT), latH: latH, latW: latW, latC: latC,
//              audioT: audioT, textLen: UInt32(textHidden.shape[0]), frameCount: frameCount,
//              sigmas: sigmas, shiftV: sv, shiftA: sa, seed: seed,
//              sparsePolicy: sparsePolicy, attentionBroadcastK: attentionBroadcastK,
//              cfg: selfLift, hooks: hooks, log: log)
//          videoX = r.videoX
//          audioX = r.audioX
//      } else {
//          <原 stage1 循环原样保留>
//      }
//
//  注意：
//  · `refined` 必须传 stage1 实际使用的 textStates（不是原始 textHidden）。
//  · `sigmas` 若走 .betaRefined 分支，需确认其非增且仅末项为 0（本 Runner 会校验并抛错）。
//  · 低分条件行不用平均池化：走官方 `_resize_keyframes`（逐帧 bilinear + 加性均值对齐），
//    尺寸一致时（lowResScale = 1）原样返回，与官方 `if kf.latent.shape == ...` 的早退一致。
//  · 过渡块里的三种插值（nearest / bicubic-antialias / bilinear）都在文件内实现，hooks 不再需要注入放大函数。
//  · 结束后 `dit.adalnTables` 已被重算回高分（原分辨率）的 ts 表，后续 stage2/直出无需额外处理。
//  · 过渡块会多跑一次 VAE decode+encode（rho>0 时）；如需最省算力可设 rho=0 走纯 latent 提升。
//  · ts 不写死：Runner 内部按 `cfg.transitionStep`（0 = 自动）解析，N = sigmas.count - 1
//    （= 面板第一阶段总步数滑杆），故滑杆 4–12 各档自动得到 3/3/4/5/6/6/7/8/9；
//    日志首行会打印实际的 ts 与低分/高分 NFE 数，便于核对当前档位。
