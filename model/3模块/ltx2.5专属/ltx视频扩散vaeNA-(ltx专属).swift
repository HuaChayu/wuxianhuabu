import Darwin
import MLX
import MLXFast

// MARK: - LTX-2.5 扩散视频解码器：3D 邻域注意力（Neighborhood Attention）降级实现
//
// 官方 diffusers（LTX2VideoVaeNeighborhoodAttention）依赖 NATTEN 的 na3d 融合内核；
// MLX 生态无 NATTEN（SGLang 官方在无 NATTEN 时回退 FlexAttention block-mask，慢 ~5x）。
// 本文件提供纯 MLX 手动窗口注意力：按 kernel=(kt,kh,kw) 逐偏移 gather 邻域 key/value 计算注意力，
// 语义与 NATTEN 一致：窗口始终铺满、边界内移（不截断）、head 间不交互、q 缩放 head_dim^-0.5、
// 3D RoPE 位置嵌入（head_dim=64 拆 T16/H24/W24，base=10000）。
//
// 布局统一 channels-last (B,T,H,W,C)。权重布局 [out,in] + [out]（safetensors 原样）。

enum DDVNA {

    /// NA 的 q/k RMSNorm，按 weight 形状自适应（x 恒为 (…, heads, headDim)）：
    /// - weight=[headDim]（真实权重 [64]，每 head 共享）→ 直接沿最后维 headDim 归一，不合并 heads；
    /// - weight=[C=heads*headDim]（NASelfCheck 的 [128] 自检权重）→ 先合并 heads 成 (…,C) 再归一、再拆回。
    /// 官方 diffusers 的 q_norm/k_norm 权重为 [head_dim]，per-head 共享归一；旧代码固定合并 heads
    /// 导致真实权重 [64] 与 (…,C) 广播崩溃，此处按 weight 形状正确分派。
    static func rmsNormNA(_ x: MLXArray, weight: MLXArray, heads: Int, headDim: Int) -> MLXArray {
        let C = heads * headDim
        if weight.shape == [C] {
            let pre = x.shape
            let n = pre.dropLast(2).reduce(1, *)
            return rmsNormWeighted(x.reshaped([n, C]), weight: weight).reshaped(pre)
        }
        // weight [headDim]：保持 (…, heads, headDim)，weight 广播到最后维
        return rmsNormWeighted(x, weight: weight)
    }

    // MARK: - 3D RoPE（官方 RotaryPosEmbed3D，base=10000）

    /// 单轴角度表：maxPos × half × 1（half=dim/2，第三维 1 便于广播到 6D）。
    /// 值由 MLX 算子构建（pos*freq → cos/sin → bf16，与旧现算路径同算子同顺序），
    /// gather/reshape 不改变数值，因此与逐 token 现算数学完全等价（NA_SELFCHECK 可保 rel<2%）。
    struct RopeAxisTable {
        let maxPos: Int
        let cos: MLXArray   // (maxPos,half,1) bf16
        let sin: MLXArray
    }

    /// 角度表缓存：按 (T,H,W) 网格尺寸惰性构建，diff decoder 推理中尺寸逐次固定，构建一次全流程复用。
    /// 三轴独立：T 轴 dim=16（half=8），H/W 轴 dim=24（half=12）。MLX pipeline 串行构建计算图，
    /// 采用"局部构建 → 整表替换"避免读侧看到半初始化状态。
    private static var ropeCacheT: RopeAxisTable?
    private static var ropeCacheH: RopeAxisTable?
    private static var ropeCacheW: RopeAxisTable?

    /// 构建单轴角度表（pos=0..<maxPos）：与旧 rotateAxis/rotateAxisBlock 的 invFreq→ang→cos/sin 完全同算子
    private static func buildRopeAxisTable(maxPos: Int, dim: Int) -> RopeAxisTable {
        let half = dim / 2
        let invFreq = (0..<half).map { i -> Float in
            expf(-Float(i) * logf(10000.0) / Float(half))
        }
        let pos = MLXArray(0..<maxPos).asType(.float32).reshaped([maxPos, 1])
        let ang = pos * MLXArray(invFreq).reshaped([1, half])              // (maxPos,half)
        let cosA = cos(ang).reshaped([maxPos, half, 1]).asType(PrecisionPolicy.defaultMainDType)
        let sinA = sin(ang).reshaped([maxPos, half, 1]).asType(PrecisionPolicy.defaultMainDType)
        return RopeAxisTable(maxPos: maxPos, cos: cosA, sin: sinA)
    }

    /// 按网格尺寸确保三轴角度表就绪（尺寸不变则复用；NA 块循环/各 stage 尺寸一致时仅首块构建一次）
    private static func ensureRopeTables(T: Int, H: Int, W: Int) {
        if let t = ropeCacheT, t.maxPos == T { /* 复用 */ } else {
            ropeCacheT = buildRopeAxisTable(maxPos: T, dim: 16)
        }
        if let h = ropeCacheH, h.maxPos == H { /* 复用 */ } else {
            ropeCacheH = buildRopeAxisTable(maxPos: H, dim: 24)
        }
        if let w = ropeCacheW, w.maxPos == W { /* 复用 */ } else {
            ropeCacheW = buildRopeAxisTable(maxPos: W, dim: 24)
        }
    }

    /// x: (B,T,H,W,heads,head_dim)，head_dim 必须 64；按 (16,24,24) 拆 T/H/W 三轴旋转
    static func rotaryPosEmbed3D(_ x: MLXArray) -> MLXArray {
        let dim = x.shape[x.ndim - 1]
        precondition(dim == 64, "DDVNA 当前仅支持 head_dim=64，实际 \(dim)")
        let dimT = 16, dimHW = 24
        let T = x.shape[1], H = x.shape[2], W = x.shape[3]
        ensureRopeTables(T: T, H: H, W: W)
        let xT = x.take(MLXArray(0..<dimT), axis: -1)
        let xH = x.take(MLXArray(dimT..<(dimT + dimHW)), axis: -1)
        let xW = x.take(MLXArray((dimT + dimHW)..<dim), axis: -1)
        let oT = rotateAxis(xT, positions: T, dim: dimT, axisRank: 1)
        let oH = rotateAxis(xH, positions: H, dim: dimHW, axisRank: 2)
        let oW = rotateAxis(xW, positions: W, dim: dimHW, axisRank: 3)
        return concatenated([oT, oH, oW], axis: -1)
    }

    /// 单轴旋转：x 形状 (B,T,H,W,heads,dim)，axisRank 1(T)/2(H)/3(W)。
    /// 角度直接查预计算表（maxPos==positions，整段表 reshape 即完整角度），无现算三角函数
    private static func rotateAxis(_ x: MLXArray, positions: Int, dim: Int, axisRank: Int) -> MLXArray {
        let half = dim / 2
        // 从缓存取该轴角度表（rotaryPosEmbed3D 已 ensure）
        let table: RopeAxisTable
        switch axisRank {
        case 1: table = ropeCacheT!
        case 2: table = ropeCacheH!
        default: table = ropeCacheW!
        }
        precondition(table.maxPos == positions, "DDVNA: 角度表 maxPos \(table.maxPos) != positions \(positions)")
        // 广播到 (B,T,H,W,heads,half)：仅在 axisRank 轴放 positions
        var angShape = [1, 1, 1, 1, 1, half]
        angShape[axisRank] = positions
        // cosA/sinA 保持 bf16：与 bf16 的 even/odd 相乘不再提升 f32，切断 RoPE f32 传播链
        let cosA = table.cos.reshaped(angShape + [1])
        let sinA = table.sin.reshaped(angShape + [1])

        // 最后一维拆成 (half,2)：even/odd 对旋转（split 语义用 take 显式表达）
        let pairs = x.reshaped(Array(x.shape.dropLast()) + [half, 2])
        let even = pairs.take(MLXArray([0]), axis: -1)  // (...,half,1)
        let odd = pairs.take(MLXArray([1]), axis: -1)
        let r1 = (even * cosA - odd * sinA)
        let r2 = (even * sinA + odd * cosA)
        // 输出显式回 bf16：即使 cosA/sinA 已 bf16，仍保证传播链终点 dtype 一致
        return concatenated([r1, r2], axis: -1).reshaped(x.shape).asType(x.dtype)
    }

    // MARK: - 块级 3D RoPE（供 NA 块内投影后使用）

    /// 块级 3D RoPE：x (blk,heads,head_dim)，tPos/hPos/wPos 为块内每个 token 的 (t,h,w) 坐标 (blk,)。
    /// 与整段 rotaryPosEmbed3D 逐 token 数学完全一致（dimT=16/dimHW=24/base=10000），
    /// 但角度按坐标数组生成，不依赖 x.shape 的 T/H/W，因此可在块循环内对块内 token 单独旋转。
    /// grid=(T,H,W) 用于定位/确保角度表覆盖（坐标值域 0..<T/H/W）。
    static func rotaryPosEmbed3DBlock(_ x: MLXArray, tPos: MLXArray, hPos: MLXArray, wPos: MLXArray, grid: (Int, Int, Int)) -> MLXArray {
        let dim = x.shape[x.ndim - 1]
        precondition(dim == 64, "DDVNA 当前仅支持 head_dim=64，实际 \(dim)")
        let dimT = 16, dimHW = 24
        ensureRopeTables(T: grid.0, H: grid.1, W: grid.2)
        let xT = x.take(MLXArray(0..<dimT), axis: -1)
        let xH = x.take(MLXArray(dimT..<(dimT + dimHW)), axis: -1)
        let xW = x.take(MLXArray((dimT + dimHW)..<dim), axis: -1)
        let oT = rotateAxisBlock(xT, positions: tPos, dim: dimT)
        let oH = rotateAxisBlock(xH, positions: hPos, dim: dimHW)
        let oW = rotateAxisBlock(xW, positions: wPos, dim: dimHW)
        return concatenated([oT, oH, oW], axis: -1).asType(x.dtype)
    }

    /// 单轴块级旋转：x (blk,heads,dim)，positions (blk,) 为该轴坐标（对应整段版 axisRank 轴的位置）。
    /// 角度查预计算表按坐标 gather 行（表值 bf16，与现算路径同算子同值），无现算三角函数
    private static func rotateAxisBlock(_ x: MLXArray, positions: MLXArray, dim: Int) -> MLXArray {
        let half = dim / 2
        // dim=16 → T 轴表；dim=24 → H/W 两轴表数学内容相同（同 dim 同 base），
        // 块级坐标只需行数覆盖，统一取行数更多的一张（maxH/maxW 较大者）
        let table: RopeAxisTable
        if dim == 16 {
            table = ropeCacheT!
        } else if let h = ropeCacheH, let w = ropeCacheW, w.maxPos >= h.maxPos {
            table = w
        } else {
            table = ropeCacheH!
        }
        let posIdx = positions.asType(.int32)                     // (blk,)
        // 表 [maxPos,half,1] → gather 行 (blk,half,1) → (blk,1,half,1)
        let cosA = table.cos.take(posIdx, axis: 0).reshaped([-1, 1, half, 1])
        let sinA = table.sin.take(posIdx, axis: 0).reshaped([-1, 1, half, 1])
        // 最后一维拆成 (half,2)：even/odd 对旋转
        let pairs = x.reshaped([-1, x.shape[1], half, 2])
        let even = pairs.take(MLXArray([0]), axis: -1)   // (blk,heads,half,1)
        let odd = pairs.take(MLXArray([1]), axis: -1)
        let r1 = (even * cosA - odd * sinA)
        let r2 = (even * sinA + odd * cosA)
        return concatenated([r1, r2], axis: -1).reshaped(x.shape).asType(x.dtype)
    }

    // MARK: - 手动 3D 邻域注意力（NATTEN 降级）

    /// 输入 x (B,T,H,W,C)。kernel=(kt,kh,kw)。权重均为 [out,in]/[out]。
    /// 备用实现（正确性优先）：整段 q/k/v 投影、rmsNorm、3D RoPE 物化后按全局索引逐偏移 gather，
    /// 27 邻居窗口、边界内移、head 间不交互、q 缩放、3D RoPE。
    /// 内存 O(N*heads*head_dim*4 + N*heads*nbr)，仅用于数值自检/回退，不用于正式推理路径。
    static func neighborhoodAttentionLegacy(
        _ x: MLXArray,
        kernel: (Int, Int, Int),
        qNormW: MLXArray, kNormW: MLXArray,
        toQW: MLXArray, toQB: MLXArray,
        toKW: MLXArray, toKB: MLXArray,
        toVW: MLXArray, toVB: MLXArray,
        projW: MLXArray, projB: MLXArray
    ) -> MLXArray {
        let shape = x.shape
        precondition(shape.count == 5, "NA 输入应为 (B,T,H,W,C)")
        let B = shape[0], T = shape[1], H = shape[2], W = shape[3], C = shape[4]
        let headDim = 64
        let heads = C / headDim
        precondition(heads * headDim == C, "C=\(C) 不能被 headDim=64 整除")
        let kt: Int = kernel.0
        let kh: Int = kernel.1
        let kw: Int = kernel.2
        let nbr = kt * kh * kw

        // 备用实现：正确性优先——整段 q/k/v 投影 + rmsNorm + RoPE 物化后按全局索引逐偏移 gather。
        // 邻居可能跨块（32³ 分块时边界 token 需要相邻块特征），整段特征保证 gather 永不出界。
        // 内存峰值 O(N*heads*headDim*4 + N*heads*nbr)：自检规模（≤32³）下完全可承受。
        let N = B * T * H * W
        let xFlat = x.reshaped([N, C])
        let coefB = T * H * W
        let coefT = H * W
        let coefH = W
        // 权重转置只做一次（块循环外复用）
        let tQW = toQW.transposed(1, 0)
        let tKW = toKW.transposed(1, 0)
        let tVW = toVW.transposed(1, 0)
        let tPW = projW.transposed(1, 0)

        // 每轴“窗口基准位置”：start = clamp(q - k/2, 0, n-k)，邻居 = start + offset（边界内移，窗口满）
        func clampWindow(_ idx: MLXArray, _ n: Int, _ k: Int) -> MLXArray {
            let half = k / 2
            let lo = MLXArray(Int32(0))
            let hi = MLXArray(Int32(n - k))
            return MLX.minimum(MLX.maximum(idx - half, lo), hi)
        }

        // 整段坐标与特征（注意：MLX Swift 整型数组的 / 是浮点除法，必须用 floorDivide 保持 int32）
        let idxAll = MLXArray(Array(0..<N)).asType(.int32)                // (N,)
        let wA = idxAll % W
        let hA = idxAll.floorDivide(W) % H
        let tA = idxAll.floorDivide(W * H) % T
        let bA = idxAll.floorDivide(W * H * T)
        let qAll = xFlat.matmul(tQW) + toQB          // (N,C)
        let kAll = xFlat.matmul(tKW) + toKB
        let vAll = xFlat.matmul(tVW) + toVB
        let qh = qAll.reshaped([N, heads, headDim])
        let kHead = kAll.reshaped([N, heads, headDim])
        let vh = vAll.reshaped([N, heads, headDim])
        // q/k 归一化 + q 缩放（官方 NA 前向：q = rms(q)*head_dim^-0.5，k = rms(k)）
        let qn = rmsNormNA(qh, weight: qNormW, heads: heads, headDim: headDim) * MLXArray(1.0 / sqrt(Float(headDim)), dtype: qh.dtype)
        let kn = rmsNormNA(kHead, weight: kNormW, heads: heads, headDim: headDim)
        // 整段 3D RoPE：按每个 token 的绝对 (t,h,w) 坐标生成角度
        let qr = rotaryPosEmbed3DBlock(qn, tPos: tA, hPos: hA, wPos: wA, grid: (T, H, W))
        let kr = rotaryPosEmbed3DBlock(kn, tPos: tA, hPos: hA, wPos: wA, grid: (T, H, W))
        var scList: [MLXArray] = []
        var kvList: [MLXArray] = []
        for ot: Int in 0..<kt {
            let nT = clampWindow(tA, T, kt) + MLXArray(ot)
            for oh: Int in 0..<kh {
                let nH = clampWindow(hA, H, kh) + MLXArray(oh)
                for ow: Int in 0..<kw {
                    let nW = clampWindow(wA, W, kw) + MLXArray(ow)
                    // 全局展平索引（整段特征，直接取，无越界）
                    let nbrIdx = (bA * coefB + nT * coefT + nH * coefH + nW).asType(.int32)   // (N,)
                    scList.append((qr * kr.take(nbrIdx, axis: 0)).sum(axis: -1))   // (N,heads)
                    kvList.append(vh.take(nbrIdx, axis: 0))                      // (N,heads,headDim)
                }
            }
        }
        // scores 栈保持 bf16；softmax 仅用单份 f32 临时保数值稳定
        let scores = stacked(scList, axis: -1)                          // (N,heads,nbr)
        let s32 = scores.asType(.float32)
        let mx = s32.max(axis: -1, keepDims: true)
        let es = (s32 - mx).exp()
        let sm = es / es.sum(axis: -1, keepDims: true)
        let smB = sm.asType(scores.dtype)
        // 加权 v：逐邻居累加整段 acc
        var acc = MLXArray.zeros([N, heads, headDim], dtype: scores.dtype)
        for i in 0..<nbr {
            // MLXArray([i]) 实测默认 int32，但显式 Int32(i) 防重载解析漂移
            let w = smB.take(MLXArray([Int32(i)]), axis: -1)           // (N,heads)
            acc = acc + kvList[i] * w.reshaped([N, heads, 1])
        }
        let out = acc.reshaped([B, T, H, W, C])
        return out.matmul(tPW) + projB
    }

    // MARK: - mask 版 3D 邻域注意力（对齐参考实现 NeighborhoodAttention3D）

    /// 输入 x (B,T,H,W,C)。kernel=(kt,kh,kw)。权重均为 [out,in]/[out]。
    /// 块级投影版 NA：q/k/v 投影、rmsNorm、3D RoPE 全部移进 query tile 循环，
    /// 每 tile 只对 query 块与 key span（窗口并集）内的 token 做投影，
    /// 用 additive 窗口 mask + MLXFast.scaledDotProductAttention（fused kernel）。
    /// 语义与官方一致：窗口铺满、边界内移、head 间不交互、q 缩放 head_dim^-0.5、
    /// kernel floor（kt=min(kt,T) 等）、3D RoPE（dimT=16/dimHW=24/base=10000）。
    /// 相比旧版整段 q/k/v/qr/kr 5 份特征同时驻留（stage5 单份 ~4.5GB，5 份 22GB+），
    /// 峰值降为单 tile 的 q/k/v + tile 级 scores（scoreBudget 控制）。
    /// 代价：相邻 tile 的 key span 重叠导致 k/v 投影少量重复（tile 越大重复越少）。
    static func neighborhoodAttention(
        _ x: MLXArray,
        kernel: (Int, Int, Int),
        qNormW: MLXArray, kNormW: MLXArray,
        toQW: MLXArray, toQB: MLXArray,
        toKW: MLXArray, toKB: MLXArray,
        toVW: MLXArray, toVB: MLXArray,
        projW: MLXArray, projB: MLXArray,
        scoreBudget: Int = 1_000_000_000
    ) -> MLXArray {
        let shape = x.shape
        precondition(shape.count == 5, "NA 输入应为 (B,T,H,W,C)")
        let B = shape[0], T = shape[1], H = shape[2], W = shape[3], C = shape[4]
        let headDim = 64
        let heads = C / headDim
        precondition(heads * headDim == C, "C=\(C) 不能被 headDim=64 整除")

        let N = B * T * H * W
        let coefT = H * W
        let coefH = W
        let xFlat = x.reshaped([N, C])
        let tQW = toQW.transposed(1, 0)
        let tKW = toKW.transposed(1, 0)
        let tVW = toVW.transposed(1, 0)
        let tPW = projW.transposed(1, 0)
        // k/v 合并投影权重：tKW/tVW 均为 [C,C]（行=输入、列=输出），须按列拼接成 [C,2C]，
        // 每 sub-tile 由 2 次 matmul 并为 1 次（concat axis 0 会拼成 [2C,C] 导致 matmul 崩）
        let kvW = concatenated([tKW, tVW], axis: 1)
        let toKVB = concatenated([toKB, toVB], axis: 0)
        let coefTArr = MLXArray(Int32(coefT))
        let coefHArr = MLXArray(Int32(coefH))

        // 窗口语义：kernel floor + 边界内移（窗口铺满，start 恒为长度 k）
        let kt = Swift.min(kernel.0, T)
        let kh = Swift.min(kernel.1, H)
        let kw = Swift.min(kernel.2, W)
        func windowStarts(_ n: Int, _ k: Int) -> [Int] {
            let half = k / 2
            return (0..<n).map { Swift.max(0, Swift.min($0 - half, n - k)) }
        }
        let st = windowStarts(T, kt)
        let sh = windowStarts(H, kh)
        let sw = windowStarts(W, kw)

        // tile 预算：qn × key span 体积 ≤ scoreBudget。
        // 32M 时 stage5（96³ 网格）会切成 ~12³ 小 tile，key span 重叠使 k/v 投影重复 ~7×；
        // 抬到 1G 后 stage5（128³ 网格）sub-tile ≈ 22³（196 个），scores 峰值 ~3GB（48G 内可控），
        // 投影重复 ~3.4×，mask 构建总元素量比 256M 少 ~3.6×。det 阶段通常单 tile，不受影响。
        var tile = (T, H, W)
        func tileCost(_ t: (Int, Int, Int)) -> Int {
            let (dt, dh, dw) = t
            let qn = dt * dh * dw
            let kSpanT = st[Swift.min(dt, T) - 1] + kt - st[0]
            let kSpanH = sh[Swift.min(dh, H) - 1] + kh - sh[0]
            let kSpanW = sw[Swift.min(dw, W) - 1] + kw - sw[0]
            return qn * kSpanT * kSpanH * kSpanW
        }
        var guardCount = 0
        while tileCost(tile) > scoreBudget && guardCount < 20 {
            let (dt, dh, dw) = tile
            if dt >= dh && dt >= dw && dt > 1 {
                tile.0 = (dt + 1) / 2
            } else if dh >= dw && dh > 1 {
                tile.1 = (dh + 1) / 2
            } else if dw > 1 {
                tile.2 = (dw + 1) / 2
            } else {
                break
            }
            guardCount += 1
        }

        let scale = powf(Float(headDim), -0.5)
        let out = MLXArray.zeros([B, T, H, W, C], dtype: x.dtype)
        // 稀疏邻居注意力：不再构建 nq×nk 全矩阵 additive mask。
        // 每个 query 只保留 kernel 窗口内 kt×kh×kw 个邻居（窗口外 softmax 权重天然为 0，数值等价），
        // 预生成邻居相对偏移（C 序展开 (dt,dh,dw)），sub-tile 内广播成 (nq,nbr) 展平索引，
        // 按索引 gather 邻居 k/v 后仅对 nbr 维做小规模 softmax，计算量 ÷(nk/nbr)≈27×。
        let nbr = kt * kh * kw
        var dtList: [Int32] = [], dhList: [Int32] = [], dwList: [Int32] = []
        dtList.reserveCapacity(nbr); dhList.reserveCapacity(nbr); dwList.reserveCapacity(nbr)
        for dt in 0..<kt { for dh in 0..<kh { for dw in 0..<kw {
            dtList.append(Int32(dt)); dhList.append(Int32(dh)); dwList.append(Int32(dw))
        }}}
        let dtArr = MLXArray(dtList).reshaped([1, nbr])
        let dhArr = MLXArray(dhList).reshaped([1, nbr])
        let dwArr = MLXArray(dwList).reshaped([1, nbr])

        for b in 0..<B {
            let bOff = MLXArray(Int32(b * T * H * W))
            var t0 = 0
            while t0 < T {
                let t1 = Swift.min(t0 + tile.0, T)
                var h0 = 0
                while h0 < H {
                    let h1 = Swift.min(h0 + tile.1, H)
                    var w0 = 0
                    while w0 < W {
                        let w1 = Swift.min(w0 + tile.2, W)
                        // query tile 与 key span（窗口并集，连续 3D 块）
                        let kt0 = st[t0], kt1 = st[t1 - 1] + kt
                        let kh0 = sh[h0], kh1 = sh[h1 - 1] + kh
                        let kw0 = sw[w0], kw1 = sw[w1 - 1] + kw
                        let nt = t1 - t0, nh = h1 - h0, nw = w1 - w0
                        let rt = kt1 - kt0, rh = kh1 - kh0, rw = kw1 - kw0
                        let nq = nt * nh * nw, nk = rt * rh * rw
                        // 坐标/偏移用轴数组广播构建（每块仅 3 个 ≤tile 轴长的小数组，代替整块 Swift 循环）：
                        let tRaw = MLXArray(Array(t0..<t1)).asType(.int32)             // (nt,)
                        let hRaw = MLXArray(Array(h0..<h1)).asType(.int32)             // (nh,)
                        let wRaw = MLXArray(Array(w0..<w1)).asType(.int32)             // (nw,)
                        let ktRaw = MLXArray(Array(kt0..<kt1)).asType(.int32)          // (rt,)
                        let khRaw = MLXArray(Array(kh0..<kh1)).asType(.int32)          // (rh,)
                        let kwRaw = MLXArray(Array(kw0..<kw1)).asType(.int32)          // (rw,)
                        let tC = tRaw * coefTArr, hC = hRaw * coefHArr, wC = wRaw
                        let ktC = ktRaw * coefTArr, khC = khRaw * coefHArr, kwC = kwRaw
                        // C 序展平偏移：(nt,1,1)+(1,nh,1)+(1,1,nw) 广播 → (nq,)
                        let qOff = (tC.reshaped([nt, 1, 1]) + hC.reshaped([1, nh, 1]) + wC.reshaped([1, 1, nw])).reshaped([nq])
                        let kOff = (ktC.reshaped([rt, 1, 1]) + khC.reshaped([1, rh, 1]) + kwC.reshaped([1, 1, rw])).reshaped([nk])
                        // RoPE 坐标三套同形状数组：加零广播展开
                        let qtPos = (tRaw.reshaped([nt, 1, 1]) + MLXArray.zeros([1, nh, nw], dtype: .int32)).reshaped([nq])
                        let qhPos = (hRaw.reshaped([1, nh, 1]) + MLXArray.zeros([nt, 1, nw], dtype: .int32)).reshaped([nq])
                        let qwPos = (wRaw.reshaped([1, 1, nw]) + MLXArray.zeros([nt, nh, 1], dtype: .int32)).reshaped([nq])
                        let ktPos = (ktRaw.reshaped([rt, 1, 1]) + MLXArray.zeros([1, rh, rw], dtype: .int32)).reshaped([nk])
                        let khPos = (khRaw.reshaped([1, rh, 1]) + MLXArray.zeros([rt, 1, rw], dtype: .int32)).reshaped([nk])
                        let kwPos = (kwRaw.reshaped([1, 1, rw]) + MLXArray.zeros([rt, rh, 1], dtype: .int32)).reshaped([nk])
                        let qIdx = (bOff + qOff).asType(.int32)
                        let kIdx = (bOff + kOff).asType(.int32)
                        let xq = xFlat.take(qIdx, axis: 0)   // (nq,C)
                        let xk = xFlat.take(kIdx, axis: 0)   // (nk,C)
                        // 块内投影 + 归一化 + RoPE（v 不归一不旋转）；q/k 只占单 tile 内存
                        let qb = (xq.matmul(tQW) + toQB).reshaped([nq, heads, headDim])
                        // k/v 由合并权重一次 matmul 产出 (nk,2C)，列序 k|v，再按列切回
                        // （mlx-swift 单范围下标沿 axis 0，kvb[0..<C] 会切行而非列，须显式双维切片）
                        let kvb = (xk.matmul(kvW) + toKVB).reshaped([nk, 2 * C])
                        let kb = kvb[0..<nk, 0..<C].reshaped([nk, heads, headDim])
                        let vb = kvb[0..<nk, C..<(2 * C)].reshaped([nk, heads, headDim])
                        // key 先按连续块绝对坐标 RoPE（gather 出的邻居 k 自动携带邻居绝对位置）
                        // rmsNorm 按 weight 形状自适应：真实权重 [headDim] 保持 (…,heads,headDim) per-head 归一；
                        // 自检权重 [C] 合并 heads 再归一（rmsNormNA 内部分派）
                        let kbR = rotaryPosEmbed3DBlock(
                            rmsNormNA(kb, weight: kNormW, heads: heads, headDim: headDim),
                            tPos: ktPos, hPos: khPos, wPos: kwPos, grid: (T, H, W)
                        )
                        // 邻居窗口起点 = query 坐标 - half 后夹取（窗口铺满，与 key span 定义一致）
                        let zero = MLXArray(Int32(0))
                        let tStart = MLX.minimum(MLX.maximum(qtPos - MLXArray(Int32(kt / 2)), zero), MLXArray(Int32(T - kt)))
                        let hStart = MLX.minimum(MLX.maximum(qhPos - MLXArray(Int32(kh / 2)), zero), MLXArray(Int32(H - kh)))
                        let wStart = MLX.minimum(MLX.maximum(qwPos - MLXArray(Int32(kw / 2)), zero), MLXArray(Int32(W - kw)))
                        // (nq,1)+(1,nbr) 广播 → (nq,nbr) 邻居绝对坐标
                        let nT = tStart.reshaped([nq, 1]) + dtArr
                        let nH = hStart.reshaped([nq, 1]) + dhArr
                        let nW = wStart.reshaped([nq, 1]) + dwArr
                        // 转 key span 内局部坐标（kbR 只含 span 内 nk 个 token，须用 span 局部索引 gather）
                        let lT = nT - MLXArray(Int32(kt0))
                        let lH = nH - MLXArray(Int32(kh0))
                        let lW = nW - MLXArray(Int32(kw0))
                        let coefTS = MLXArray(Int32(rh * rw))
                        let coefHS = MLXArray(Int32(rw))
                        let nbrIdx = (lT * coefTS + lH * coefHS + lW).asType(.int32)   // (nq,nbr)
                        // query 分块：逐块 RoPE + gather 邻居 k/v + 小规模 softmax，控制中间峰值
                        // （整块 gather 的 (nq,nbr,heads,headDim) 达 ~7GB，分 512 后 ~350MB/块）
                        let qBlock = 512
                        var oParts: [MLXArray] = []
                        var qs = 0
                        while qs < nq {
                            let qe = Swift.min(qs + qBlock, nq)
                            let qBlk = qe - qs
                            let qr = rotaryPosEmbed3DBlock(
                                rmsNormNA(qb[qs..<qe], weight: qNormW, heads: heads, headDim: headDim),
                                tPos: qtPos[qs..<qe], hPos: qhPos[qs..<qe], wPos: qwPos[qs..<qe], grid: (T, H, W)
                            )                                   // (qBlk,heads,headDim)
                            let idxB = nbrIdx[qs..<qe]          // (qBlk,nbr)
                            let kG = kbR.take(idxB, axis: 0)    // (qBlk,nbr,heads,headDim)，k 已按邻居绝对坐标 RoPE
                            let s32 = einsum("qhd,qnhd->qhn", qr, kG).asType(.float32) * scale
                            let mx = s32.max(axis: -1, keepDims: true)
                            let es = (s32 - mx).exp()
                            let smB = (es / es.sum(axis: -1, keepDims: true)).asType(qr.dtype)
                            let vG = vb.take(idxB, axis: 0)     // (qBlk,nbr,heads,headDim)
                            let oB = einsum("qhn,qnhd->qhd", smB, vG)   // (qBlk,heads,headDim)
                            oParts.append(oB)
                            qs = qe
                        }
                        let oAll = concatenated(oParts, axis: 0)   // (nq,heads,headDim)
                        // (nq,heads,headDim) → (nt,nh,nw,C) → 写回 b 切片
                        out[b..<(b + 1), t0..<t1, h0..<h1, w0..<w1, 0..<C] =
                            oAll.reshaped([1, t1 - t0, h1 - h0, w1 - w0, C])
                        w0 = w1
                    }
                    h0 = h1
                }
                t0 = t1
            }
        }
        return out.matmul(tPW) + projB
    }


    // MARK: - det 阶段 NA block（官方 LTX2VideoVaeNABlock）

    /// norm1 → NA attn → norm2 → SwiGLU MLP，残差相加
    /// （SwiGLU MLP 已下沉为通用算子 swigluMLP，见 2网络算子/网络算子-FFN-通用.swift）
    static func naBlock(
        _ x: MLXArray,
        norm1W: MLXArray, norm2W: MLXArray,
        kernel: (Int, Int, Int),
        attnQW: MLXArray, attnQB: MLXArray, attnKW: MLXArray, attnKB: MLXArray,
        attnVW: MLXArray, attnVB: MLXArray, attnPW: MLXArray, attnPB: MLXArray,
        qNormW: MLXArray, kNormW: MLXArray,
        gateW: MLXArray, upW: MLXArray, downW: MLXArray
    ) -> MLXArray {
        var h = x
        let n1 = rmsNormWeighted(h, weight: norm1W)
        let a = neighborhoodAttention(
            n1, kernel: kernel,
            qNormW: qNormW, kNormW: kNormW,
            toQW: attnQW, toQB: attnQB, toKW: attnKW, toKB: attnKB,
            toVW: attnVW, toVB: attnVB, projW: attnPW, projB: attnPB
        )
        h = h + a
        let n2 = rmsNormWeighted(h, weight: norm2W)
        h = h + swigluMLP(n2, upW: upW, upB: nil, gateW: gateW, gateB: nil, downW: downW)
        return h
    }
}
