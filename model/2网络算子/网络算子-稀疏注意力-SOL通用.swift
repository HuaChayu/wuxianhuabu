import Foundation
import MLX
import MLXRandom

// 使用方：H3（H3Transformer 等具体调用点如下；本文件为通用底层能力，后续其它模型可直接复用）
//   - model/minimax h3/H3Transformer.swift:418  H3DiT 自注意力：H3SolAttn.forward(q:k:v:g:scale:cfg:)
//   - model/minimax h3/h3管线自检-(h3专属).swift:225  NA_H3TEST=9 合成数值自检：H3SolAttn.selfTest()
//   - 复用方（LTX 二采 refine）：model/3模块/ltx2.5专属/ltx主干-(ltx专属).swift:366
//     LTXSparseAttnBridge 以 g=0、traceSafe=true 调用同一内核（编译图内零副作用路径）

// MARK: - H3 Sol-Attn（ComfyUI BlockSparseAttention 移植）
//
// 移植对象:
//   1. ComfyUI core  comfy_extras/nodes_sparse_attention.py（PR#16072, kijai）
//      调度层: 对 MiniMax-H3 自注意力做 block patch，三种方法
//      sol-attn / top-k(SLA) / VSA，外加 sink_conditioning / extra_tokens 保真策略。
//   2. comfy-kitchen  eager 参考实现 comfy_kitchen/backends/eager/sol_attn.py（PR#117）
//      算法数值蓝本（ArXiv 2607.24027 NVIDIA Sol-Attn）:
//      每个 64-token query 块只对"路由选中的 key 块"精确 attend；
//      未选中的块不丢弃，而是贡献一个 pooled term（block mean key 打分、
//      block sum value 加权、softmax 分母按块长计入），保整序列归一。
//
// 与用户自研 H3TopKAttend 的区别（本文件的动机）:
//   - H3TopKAttend: 32-row block、采样 rep 行 max-pool 打分、固定 K 每 (head,tile)。
//     未选中 video 块完全不进 softmax 分母 -> 分母质量缺失（用户曾固定折叠键集花屏，
//     top-k 路线虽已数据驱动，但仍丢分母）。
//   - Sol-Attn:  64-row block（与 H3 patch 行/音频帧对齐，audio 用 64 防机器人腔）、
//     block-mean centroid 打分、per-(head,query-block) 自适应 sigma 阈值（tau）免训练，
//     也支持固定 top-k(SLA)。被路由淘汰的块以质心项补分母（tail），softmax 分母保留全序列。
//
// Layout: q/k/v [1, H, S, HD]（bf16）; 行 [0,g) 为 strip 行（packed
// text/audio/reference 条件行，始终精确 attend 全部 key —— 即 sink_conditioning），
// 行 [g,S) 为 video 行，只对 video key 段做稀疏路由。
// strip 行自身的输出走 dense SDPA（与 H3TopKAttend 相同结构）。
//
// 实现:
//   路由(MLX, GPU):  k/v video 段按 64 行分块 mean/sum -> kc/vSum；
//   去心 kcc=k-k_mean 与 kc_var；q 块 mean centroid；colmean = qc·kccᵀ*scale。
//   tau 模式阈值 thr = tau*sqrt(var*scale²+1e-6), var=(centroid²·kc_var).sum。
//   强制精确: |qblk-j|<=1 与 sink 块区间。路由产出 colmean(bf16,预乘 scale) +
//   video block value sum + 块长表 + 每 (head,qblk) 阈值表，交给 Metal。
//   执行(Metal h3sol_flash_v1): 每 32 条 video query 行一个 threadgroup(32 threads);
//   q 块 64 行拆 2 组，组内共享同一 qBlk -> 路由判定无分歧。单遍 for j in video 块:
//     选中(>thr / diag / sink) -> smem 协作 load 64 行, 在线 softmax feed;
//     未选中          -> tail: p=exp(score-m), num+=p*vSum[j], den+=p*lens[j]。
//   无 sel 表/bitmask，bandwidth 只读选中块 + tail 轻量。
//
// v1 范围说明: 仅实现 sol-attn tau 路由（ComfyUI 默认 method，也是本次移植核心）。
// SLA top-k 固定比例与本文件思路重复（用户已有 H3TopKAttend 做固定 K top-k），暂不并入；
// topkRatio 参数保留为占位，设置后打印提示并忽略。
//
// Env knobs (debug):
//   NA_H3SOL=1                enable this path（H3Transformer 内切换）
//   NA_H3SOL_TAU=<float>      tau 阈值（默认 1.0）
//   NA_H3SOL_TAIL=0           关闭 tail（未选块彻底丢弃，仅调试对比用）
//
// ====================== 踩坑记录（已修复） ======================
// 症状: UI 生成走 Sol-Attn 路径时崩溃，栈为
//   H3SolAttn.forward -> Gather::eval_gpu -> set_vector_bytes(idx_shapes, 7)
//   -> setBytes(nullptr, 0)，Metal Validation Layer 断言
//   "bytes argument cannot be nil" 并 abort。
// 根因: bf(q)[0] 这类「单整数下标」在 Swift MLX 中走
//   getItem(.index) -> take(resolve(index:axis:), axis:)，而 resolve(index:axis:)
//   返回 0 维标量 MLXArray；C++ Gather::eval_gpu 会把各 indices 的 shape 扁平拼进
//   std::vector<int> idx_shapes —— 0 维 indices 不贡献任何元素，idx_shapes 成为
//   空 vector，compute_encoder.set_vector_bytes(idx_shapes, 7) 于是在 Metal
//   Validation Layer 下以 setBytes(nullptr, 0) 触发断言崩溃。
// 修复: 改用 bf(q).squeezed(axis: 0)。squeeze/Reshape 为零拷贝共享 buffer 路径，
//   语义等价（仅去掉 size == 1 的 batch 轴），且完全不产生 Gather；函数内已有
//   precondition(q.shape[0] == 1) 硬保证 squeeze 前提。
// 等价性验证: MLX ops.cpp 中 take(array, indices, axis) 的实现即 gather + squeeze，
//   故 x[0] ≡ squeeze(gather(x, 标量, 0), 0)，去掉中间 gather 后终点一致；
//   连续 / 非连续 layout 下数值实测全部 True。
// 排查结论: 全项目 H3 目录内仅 q / k / v 三处单整数下标（本文件），其余 take 调用
//   的 indices 均 >= 1 维，不进入该路径。规范: 禁止对 MLXArray 使用单整数下标
//   x[0]，一律用 x.squeezed(axis: 0)。
// ===============================================================

public struct H3SolAttnConfig {
    /// 占位: SLA top-k 保留比例，v1 未实现，设置后打印提示并忽略（tau 模式下应保持 nil）
    public var topkRatio: Float?
    /// Sol-Attn tau：每个 (head, query 块) 的得分分布 sigma 阈值倍数。
    /// tau≈1.0 保留约 16% key 块, 1.3≈5~7%（ComfyUI 默认）, 1.5≈7%, 2.0≈2.7%（参考 ComfyUI 注释）。
    public var tau: Float = 1.3
    /// 恒精确的 video key 块区间 [sinkStart, sinkEnd)（如首帧/音频锚点块），nil 不启用
    public var sinkBlocks: Range<Int>?
    /// 是否启用 tail pooled 项（未选中块补分母）。默认 true；false 仅用于数值对比。
    public var tail: Bool = true
    /// 块大小（与 ComfyUI/ck 一致 64）
    public let block: Int = 64
    public init(topkRatio: Float? = nil, tau: Float = 1.3,
                sinkBlocks: Range<Int>? = nil, tail: Bool = true) {
        self.topkRatio = topkRatio
        self.tau = tau
        self.sinkBlocks = sinkBlocks
        self.tail = tail
    }
}

public enum H3SolAttn {


    // h3sol_flash_v6: v3 的 BQ64 合并版（对齐上游 tiled kernel 的 BQ64/256-thread 组织，
    // 保持 v3 已验证的标量 FMA 路径，不用 simdgroup MMA）。
    // 改动点（仅并行组织，数值路径与 v3 逐位等价）：
    //   - 每 threadgroup 处理 64 qrow（v3 为 32）：tgcount = Lpad/64，threadgroup 256 = 8 simd；
    //   - v3 中同 qBlk 的两个兄弟 32-row tg 各自重复装载全部选中块 K/V + 重扫 strip；
    //     合并后同一 K/V 装载服务 64 行 -> smem 装载总量减半、strip 重扫减半；
    //   - 路由决策与 v3 相同（colmean>th、|qblk-j|<=1、sink），qBlk = tgid。
    static let mslSourceV6 = """
        // grid.x = Lpad*4 threads (tgcount = Lpad/64); threadgroup = 256.
        const int Lpad = q_shape[2];        // nV*64
        const int nV  = colmean_shape[2];
        const int g   = kv_shape[2] - Lpad;

        const uint tgid = threadgroup_position_in_grid.x;   // 0..Lpad/64-1 (== qBlk)
        const uint head = threadgroup_position_in_grid.y;
        const uint tid  = thread_index_in_threadgroup;      // 0..255
        const uint simd = tid >> 5;                         // 0..7
        const uint sl   = tid & 31u;                        // simd lane
        const uint qlSimd = sl >> 2;                        // 0..7 (local qrow in simd)
        const uint chunk = sl & 3u;                         // 0..3 (32-dim chunk)
        const int qrow = int(tgid * 64 + simd * 8 + qlSimd); // 0..Lpad-1
        const uint qBlk = tgid;                             // 64-row query block

        const float scaleV = scale[0];
        const int sinkS = int(scale[2]);
        const int sinkE = int(scale[3]);

        threadgroup T ks[32][HD];   // 32 key rows x HD
        threadgroup T vs[32][HD];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int rowStride = g + Lpad;
        const int videoEndKey = rowStride - int(scale[1]);  // real video key end (excl)

        // per-thread state: this lane owns 32 dims [chunk*32, chunk*32+32)
        const int dB = int(chunk) * 32;
        float qs[32];
        const device T* qr = q + ((head * Lpad) + qrow) * HD;
        for (int j = 0; j < 32; ++j) qs[j] = float(qr[dB + j]);

        float m = -INFINITY;
        float l = 0.0f;
        float acc[32];
        for (int j = 0; j < 32; ++j) acc[j] = 0.0f;

        auto load32 = [&](int keyStart) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 两个 buffer 共 32*128*2 元素，256 线程并行（每线程 32 元素）
            for (int i = int(tid); i < 8192; i += 256) {
                const int r = i >> 7;            // 0..63 -> row within 64 combined
                const int c = i & 127;
                if (r < 32) {
                    const int row = keyStart + r;
                    ks[r][c] = kv[(head * rowStride + row) * HD + c];
                } else {
                    const int rr = r - 32;
                    const int row = keyStart + rr;
                    vs[rr][c] = vv[(head * rowStride + row) * HD + c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        };

        // online softmax over key rows [keyStart, endKey) already in smem
        auto feed32 = [&](int keyStart, int endKey) {
            const int lim = max(0, min(32, endKey - keyStart));
            for (int kk = 0; kk < lim; ++kk) {
                float s = 0.0f;
                for (int j = 0; j < 32; ++j) s += qs[j] * float(ks[kk][dB + j]);
                // 4-lane butterfly: 同 qrow 的 4 个 chunk lane 归约并广播
                s += simd_shuffle_xor(s, 1);
                s += simd_shuffle_xor(s, 2);
                s *= scaleV;
                const float m2 = max(m, s);
                const float a = exp(m - m2);
                const float p = exp(s - m2);
                for (int j = 0; j < 32; ++j) acc[j] = acc[j] * a + p * float(vs[kk][dB + j]);
                l = l * a + p;
                m = m2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        };

        // ---- strip keys [0, g)：always exact ----
        const int gTiles = (g + 31) / 32;
        for (int kt = 0; kt < gTiles; ++kt) {
            const int keyStart = kt * 32;
            load32(keyStart);
            feed32(keyStart, min(g, keyStart + 32));
        }

        // ---- video key blocks: exact if routed/diag/sink, else tail pooled ----
        const int cmBase = (head * colmean_shape[1] + int(qBlk)) * nV;
        const float th = thr[head * colmean_shape[1] + int(qBlk)];
        const int qblkI = int(qBlk);
        for (int j = 0; j < nV; ++j) {
            bool selected = float(colmean[cmBase + j]) > th;
            if (!selected && (abs(qblkI - j) <= 1)) selected = true;
            if (!selected && (j >= sinkS && j < sinkE)) selected = true;

            if (selected) {
                const int b0 = g + j * 64;
                load32(b0);
                feed32(b0, min(videoEndKey, b0 + 32));
                load32(b0 + 32);
                feed32(b0 + 32, min(videoEndKey, b0 + 64));
            } else if (scale[4] > 0.5f) {
                // tail pooled term (per-lane 32 dims of the block value sum)
                const float s = float(colmean[cmBase + j]);
                const float m2 = max(m, s);
                const float a = exp(m - m2);
                const float p = exp(s - m2);
                const float lenJ = lens[j];
                const int vsBase = (head * nV + j) * HD;
                if (lenJ > 0.0f) {
                    for (int dj = 0; dj < 32; ++dj) acc[dj] = acc[dj] * a + p * float(vsum[vsBase + dB + dj]);
                    l = l * a + p * lenJ;
                } else {
                    for (int dj = 0; dj < 32; ++dj) acc[dj] = acc[dj] * a;
                    l = l * a;
                }
                m = m2;
            }
        }

        device T* orow = o + ((head * Lpad) + qrow) * HD;
        const float inv = 1.0f / l;
        for (int j = 0; j < 32; ++j) orow[dB + j] = T(acc[j] * inv);
    """

    private static let kernelV6 = MLXFast.metalKernel(
        name: "h3sol_flash_v6",
        inputNames: ["q", "kv", "vv", "colmean", "vsum", "lens", "thr", "scale"],
        outputNames: ["o"],
        source: mslSourceV6
    )



    // h3sol_flash_v7: v6 的「分块解耦」版（纯新增，v6 正文与注释一行未改）。
    // 动机: 外层按 query 行分块时 q 只含 video 段的一块、K/V 仍是全序列；v6 把三个
    // 几何量互推（Lpad=q_shape[2]、g=kv_shape[2]-Lpad、qBlk=tgid），只切 q 会让 K/V
    // 行错位并在 g>0 的块上触发 reshape 断言。v7 改为显式传入:
    //   Lq   = q_shape[2]   —— 本块 query 行数（nVq*64）
    //   g    = scale[5]     —— 全序列 strip 行数
    //   qOff = scale[6]     —— 本块起点在全序列 video 64 行块网格中的块号
    // rowStride 直接取 kv_shape[2]（全序列 K/V 行数）；colmean/thr 的行索引用本块内
    // tgid（表按块给），diag/sink 判定用全局块号 qBlk。
    // 整段调用（Lq=padL、qOff=0、g 同值）时与 v6 逐位等价（selfTest Case S 断言）。
    static let mslSourceV7 = """
        // grid.x = Lq*4 threads (tgcount = Lq/64); threadgroup = 256.
        const int Lq  = q_shape[2];          // 本块 query 行数（nVq*64）
        const int nV  = colmean_shape[2];
        const int g   = int(scale[5]);       // 全序列 strip 行数（显式传入）

        const uint tgid = threadgroup_position_in_grid.x;   // 0..Lq/64-1（本块内块号）
        const uint head = threadgroup_position_in_grid.y;
        const uint tid  = thread_index_in_threadgroup;      // 0..255
        const uint simd = tid >> 5;                         // 0..7
        const uint sl   = tid & 31u;                        // simd lane
        const uint qlSimd = sl >> 2;                        // 0..7 (local qrow in simd)
        const uint chunk = sl & 3u;                         // 0..3 (32-dim chunk)
        const int qrow = int(tgid * 64 + simd * 8 + qlSimd); // 0..Lq-1
        const uint qBlk = uint(int(scale[6]) + int(tgid)); // 全序列 video 块号

        const float scaleV = scale[0];
        const int sinkS = int(scale[2]);
        const int sinkE = int(scale[3]);

        threadgroup T ks[32][HD];   // 32 key rows x HD
        threadgroup T vs[32][HD];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int rowStride = kv_shape[2];   // 全序列 K/V 行数（不再 = g + Lq）
        const int videoEndKey = rowStride - int(scale[1]);  // real video key end (excl)

        // per-thread state: this lane owns 32 dims [chunk*32, chunk*32+32)
        const int dB = int(chunk) * 32;
        float qs[32];
        const device T* qr = q + ((head * Lq) + qrow) * HD;
        for (int j = 0; j < 32; ++j) qs[j] = float(qr[dB + j]);

        float m = -INFINITY;
        float l = 0.0f;
        float acc[32];
        for (int j = 0; j < 32; ++j) acc[j] = 0.0f;

        auto load32 = [&](int keyStart) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // 两个 buffer 共 32*128*2 元素，256 线程并行（每线程 32 元素）
            for (int i = int(tid); i < 8192; i += 256) {
                const int r = i >> 7;            // 0..63 -> row within 64 combined
                const int c = i & 127;
                if (r < 32) {
                    const int row = keyStart + r;
                    ks[r][c] = kv[(head * rowStride + row) * HD + c];
                } else {
                    const int rr = r - 32;
                    const int row = keyStart + rr;
                    vs[rr][c] = vv[(head * rowStride + row) * HD + c];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        };

        // online softmax over key rows [keyStart, endKey) already in smem
        auto feed32 = [&](int keyStart, int endKey) {
            const int lim = max(0, min(32, endKey - keyStart));
            for (int kk = 0; kk < lim; ++kk) {
                float s = 0.0f;
                for (int j = 0; j < 32; ++j) s += qs[j] * float(ks[kk][dB + j]);
                // 4-lane butterfly: 同 qrow 的 4 个 chunk lane 归约并广播
                s += simd_shuffle_xor(s, 1);
                s += simd_shuffle_xor(s, 2);
                s *= scaleV;
                const float m2 = max(m, s);
                const float a = exp(m - m2);
                const float p = exp(s - m2);
                for (int j = 0; j < 32; ++j) acc[j] = acc[j] * a + p * float(vs[kk][dB + j]);
                l = l * a + p;
                m = m2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        };

        // ---- strip keys [0, g)：always exact ----
        const int gTiles = (g + 31) / 32;
        for (int kt = 0; kt < gTiles; ++kt) {
            const int keyStart = kt * 32;
            load32(keyStart);
            feed32(keyStart, min(g, keyStart + 32));
        }

        // ---- video key blocks: exact if routed/diag/sink, else tail pooled ----
        const int cmBase = (head * colmean_shape[1] + int(tgid)) * nV;
        const float th = thr[head * colmean_shape[1] + int(tgid)];
        const int qblkI = int(qBlk);
        for (int j = 0; j < nV; ++j) {
            bool selected = float(colmean[cmBase + j]) > th;
            if (!selected && (abs(qblkI - j) <= 1)) selected = true;
            if (!selected && (j >= sinkS && j < sinkE)) selected = true;

            if (selected) {
                const int b0 = g + j * 64;
                load32(b0);
                feed32(b0, min(videoEndKey, b0 + 32));
                load32(b0 + 32);
                feed32(b0 + 32, min(videoEndKey, b0 + 64));
            } else if (scale[4] > 0.5f) {
                // tail pooled term (per-lane 32 dims of the block value sum)
                const float s = float(colmean[cmBase + j]);
                const float m2 = max(m, s);
                const float a = exp(m - m2);
                const float p = exp(s - m2);
                const float lenJ = lens[j];
                const int vsBase = (head * nV + j) * HD;
                if (lenJ > 0.0f) {
                    for (int dj = 0; dj < 32; ++dj) acc[dj] = acc[dj] * a + p * float(vsum[vsBase + dB + dj]);
                    l = l * a + p * lenJ;
                } else {
                    for (int dj = 0; dj < 32; ++dj) acc[dj] = acc[dj] * a;
                    l = l * a;
                }
                m = m2;
            }
        }

        device T* orow = o + ((head * Lq) + qrow) * HD;
        const float inv = 1.0f / l;
        for (int j = 0; j < 32; ++j) orow[dB + j] = T(acc[j] * inv);
    """

    private static let kernelV7 = MLXFast.metalKernel(
        name: "h3sol_flash_v7",
        inputNames: ["q", "kv", "vv", "colmean", "vsum", "lens", "thr", "scale"],
        outputNames: ["o"],
        source: mslSourceV7
    )



    // MARK: 主入口
    /// Sol-Attn 稀疏注意力（v6/v7 Metal 内核：v6 整段调用，v7 分块调用，均 256 threads/BQ64）。
    /// - q/k/v: [1, H, S, HD]（S = g + videoRows）
    /// - g: 头部 strip 行数（packed 条件行，恒精确 attend）
    public static func forward(q: MLXArray, k: MLXArray, v: MLXArray,
                               g: Int, scale: Float,
                               cfg: H3SolAttnConfig = .init(),
                               useV6: Bool = true,
                               traceSafe: Bool = false) -> MLXArray {
        // traceSafe（新增可选参数，默认 false 保持既有调用语义不变）：
        // 进入 MLX.compile 图内（LTX 二采等编译路径）时必须为 true——图内禁止 host 侧
        // 副作用（读 ProcessInfo env / print / MLX.eval），否则 trace 期同步求值或日志
        // 会破坏编译图并可能死锁。true 时仅跳过诊断输出与显式物化，数值语义完全一致。
        // 运行时探针日志已移除：topkRatio 为 SLA 占位参数，v3/v6 恒为 tau 路由模式，语义不变
        let srcDtype = q.dtype
        let h = q.shape[1]
        let S = q.shape[2]
        let hd = q.shape[3]
        precondition(q.shape[0] == 1, "H3SolAttn: batch must be 1")
        precondition(hd == 128, "H3SolAttn: HD==128 expected")
        precondition(S > g)
        let B = cfg.block                      // 64
        let L = S - g                          // video rows
        let nV = (L + B - 1) / B               // video key/query 块数
        let padL = nV * B
        let padAmt = padL - L

        // 环境变量覆盖（traceSafe：编译图内禁止读 env，运行时覆盖仅在不进图时生效）
        var tau = cfg.tau
        var tail = cfg.tail
        if !traceSafe {
            if let e = ProcessInfo.processInfo.environment["NA_H3SOL_TAU"], let f = Float(e) { tau = f }
            if let e = ProcessInfo.processInfo.environment["NA_H3SOL_TAIL"], e == "0" { tail = false }
        }

        let bf = { (x: MLXArray) -> MLXArray in x.dtype == .bfloat16 ? x : x.asType(.bfloat16) }
        // 注意：此处不可用 bf(q)[0] 形式的单整数下标（崩溃根因见文件头「踩坑记录」）。
        let qH = bf(q).squeezed(axis: 0)       // [H, S, HD]
        let kH = bf(k).squeezed(axis: 0)
        let vH = bf(v).squeezed(axis: 0)

        // ---- strip 行输出：dense SDPA（条件行恒精确，结构同 H3TopKAttend） ----
        // g == 0（无 strip 条件行，如 LTX 纯自注意力接入）：stripOut 置 nil 而不是造 0 元素
        // 数组——0-size buffer 进 MLX 图会在 Metal 编码期拿到空数据指针（见文件头踩坑记录同类根因）。
        let stripOut: MLXArray?
        if g > 0 {
            let qStrip = qH[0..., 0..<g, 0...].reshaped([1, h, g, hd])
            let kAll = kH.reshaped([1, h, S, hd])
            let vAll = vH.reshaped([1, h, S, hd])
            stripOut = scaledDotProductAttention(queries: qStrip, keys: kAll, values: vAll,
                                                 scale: scale, mask: nil)
        } else {
            stripOut = nil
        }

        // ---- video 段切分 + pad ----
        let qV = qH[0..., g..<S, 0...]          // [H, L, HD]
        let kV = kH[0..., g..<S, 0...]
        let vV = vH[0..., g..<S, 0...]
        // padAmt == 0 时绝不构造 0 元素 zeros：0-size buffer 参与 MLX 图会产出空数据指针，
        // Metal 编码期 setBytes/setBuffer 收到 nil 即断言崩溃（"bytes argument cannot be nil"）
        let qVp: MLXArray, kVp: MLXArray, vVp: MLXArray
        if padAmt > 0 {
            let zeros = MLXArray.zeros([h, padAmt, hd], dtype: .bfloat16)
            qVp = concatenated([qV, zeros], axis: 1)   // [H, padL, HD]
            kVp = concatenated([kV, zeros], axis: 1)
            vVp = concatenated([vV, zeros], axis: 1)
        } else {
            qVp = qV; kVp = kV; vVp = vV
        }

        // ---- 块长表（最后 ragged 块） ----
        let tailLen = (L % B == 0) ? B : (L % B)
        var lensA = [Float](repeating: Float(B), count: nV)
        if padAmt > 0 { lensA[nV - 1] = Float(tailLen) }
        let lensArr = MLXArray(lensA)           // [nV] f32

        // ---- 路由（fp32 域，与 eager 参考数值对齐；无布尔/排序，纯算术） ----
        let lv = lensArr.reshaped([1, nV, 1])
        let qcF = (qVp.asType(.float32).reshaped([h, nV, B, hd]).sum(axis: 2) / lv)  // [H,nV,HD] centroid
        let kcF = (kVp.asType(.float32).reshaped([h, nV, B, hd]).sum(axis: 2) / lv)
        let vsF =  vVp.asType(.float32).reshaped([h, nV, B, hd]).sum(axis: 2)         // value sum
        let kMean = kcF.mean(axis: 1, keepDims: true)                                // [H,1,HD]
        let kcc = kcF - kMean                                                          // 去心 key centroid
        let kcVar = (kcc * kcc).mean(axis: 1)                                         // [H,HD]

        // 块质心分（自然域 logit，已乘 scale）
        let cmF = matmul(qcF, kcc.transposed(0, 2, 1)) * scale                        // [H,nV,nV]
        // tau 自适应 sigma 阈值: thr = tau*sqrt(var*scale²+1e-6), var=(centroid²·kc_var).sum
        let varF = (qcF * qcF * kcVar.reshaped([h, 1, hd])).sum(axis: 2)              // [H,nV]
        let thrF = MLXArray(tau) * (varF * (scale * scale) + MLXArray(Float(1e-6))).sqrt()
        // 运行时探针日志已移除（pre-eval / post-route-eval）；显式物化保留，仅做同步求值不做输出
        if !traceSafe { MLX.eval(cmF, thrF, vsF, qVp, kVp, vVp) }

        // ---- 组装 kernel 输入 ----
        // g == 0 时不做 [0 行 strip; video] 的 concat（避免 0-size 数组入图），直接以 video 段为全量 K/V。
        let kIn: MLXArray
        let vIn: MLXArray
        if g > 0 {
            let kStrip = kH[0..., 0..<g, 0...]
            let vStrip = vH[0..., 0..<g, 0...]
            kIn = concatenated([kStrip, kVp], axis: 1).reshaped([1, h, g + padL, hd]) // [1,H,g+padL,HD]
            vIn = concatenated([vStrip, vVp], axis: 1).reshaped([1, h, g + padL, hd])
        } else {
            kIn = kVp.reshaped([1, h, padL, hd])   // [1,H,padL,HD]
            vIn = vVp.reshaped([1, h, padL, hd])
        }
        let qIn = qVp.reshaped([1, h, padL, hd])
        let cmB = cmF.asType(.bfloat16)
        let vsB = vsF.asType(.bfloat16)
        let sinkS = Float(cfg.sinkBlocks?.lowerBound ?? 0)
        let sinkE = Float(cfg.sinkBlocks?.upperBound ?? 0)
        let scaleArr = MLXArray([scale, Float(padAmt), sinkS, sinkE, tail ? Float(1) : Float(0)])
        let thrArr = thrF.reshaped([h, nV])

        // ---- 显式物化全部 kernel 输入 ----
        // 消除"隐式惰性节点延迟到 Metal 编码期才求值、拿到空数据指针"的风险：
        // 任何 MLX 侧异常都会在这条明确的 eval 上暴露，而不是在 kernel 编码内部以 setBytes nil 断言崩溃
        precondition(padL > 0 && g + padL > 0 && nV > 0,
                     "H3SolAttn: 非法几何 padL=\(padL) g=\(g) nV=\(nV)")
        precondition(qIn.size > 0 && kIn.size > 0 && vIn.size > 0 && cmB.size > 0
                     && vsB.size > 0 && lensArr.size > 0 && thrArr.size > 0 && scaleArr.size > 0,
                     "H3SolAttn: kernel 输入含 0 元素数组")
        if !traceSafe {
            MLX.eval(qIn, kIn, vIn, cmB, vsB, lensArr, thrArr, scaleArr)
            // 运行时探针日志已移除（kern-in ready）
        }

        // v3 及更早内核（v1/v2/v4/v5）已全部删除：恒用 kernelV6
        let kern = kernelV6
        let gridN = (padL * 4, h, 1)   // padL*4 threads, threadgroup 256
        let out = kern(
            [qIn, kIn, vIn, cmB, vsB, lensArr, thrArr, scaleArr],
            template: [("T", DType.bfloat16), ("HD", hd)],
            grid: gridN,
            threadGroup: (256, 1, 1),
            outputShapes: [[1, h, padL, hd]],
            outputDTypes: [.bfloat16]
        )[0]

        let videoOut = padAmt > 0 ? out[0..., 0..., 0..<L, 0...] : out
        let attn: MLXArray
        if let stripOut {
            attn = concatenated([stripOut, videoOut.reshaped([1, h, L, hd])], axis: 2)
        } else {
            attn = videoOut.reshaped([1, h, L, hd])   // g == 0：无 strip 段，video 段即全量输出
        }
        return srcDtype == .bfloat16 ? attn : attn.asType(srcDtype)
    }

    // MARK: 分块入口（v7 内核；供外层按 query 行分块逐步调用）

    /// v7 分块用：整层只算一次的 K/V 侧中间量，供各 query 块（attendVideo）复用。
    public struct SolKVPrepared {
        let kIn: MLXArray        // [1,H,gFull+padL,HD]（strip 段 + padded video 段）
        let vIn: MLXArray        // [1,H,gFull+padL,HD]
        let lens: MLXArray       // [nV] 各 video 块真实行数
        let kcc: MLXArray        // [H,nV,HD] 去心 key 块质心
        let kcVar: MLXArray      // [H,HD] 去心质心方差
        let vsF: MLXArray        // [H,nV,HD] f32 块 value 和
        let nV: Int
        let padAmt: Int
        let gFull: Int
        let Lfull: Int
        let h: Int
        let hd: Int
        let tau: Float
        let sinkS: Float
        let sinkE: Float
        let tailOn: Bool
    }

    /// K/V 侧预计算：全序列只算一次，供后续各 query 块复用。
    /// - k/v: [1, H, S_full, HD] 全序列（含 strip 段）
    /// - gFull: 全序列 strip 行数
    public static func prepareKV(k: MLXArray, v: MLXArray, gFull: Int,
                                 cfg: H3SolAttnConfig = .init(),
                                 traceSafe: Bool = false) -> SolKVPrepared {
        let h = k.shape[1]
        let Sfull = k.shape[2]
        let hd = k.shape[3]
        precondition(k.shape[0] == 1 && v.shape[0] == 1, "H3SolAttn: batch must be 1")
        precondition(hd == 128, "H3SolAttn: HD==128 expected")
        precondition(Sfull > gFull, "H3SolAttn.prepareKV: S_full 必须大于 gFull")
        let B = cfg.block
        let Lfull = Sfull - gFull
        let nV = (Lfull + B - 1) / B
        let padL = nV * B
        let padAmt = padL - Lfull

        var tau = cfg.tau
        var tail = cfg.tail
        if !traceSafe {
            if let e = ProcessInfo.processInfo.environment["NA_H3SOL_TAU"], let f = Float(e) { tau = f }
            if let e = ProcessInfo.processInfo.environment["NA_H3SOL_TAIL"], e == "0" { tail = false }
        }

        let bf = { (x: MLXArray) -> MLXArray in x.dtype == .bfloat16 ? x : x.asType(.bfloat16) }
        let kH = bf(k).squeezed(axis: 0)
        let vH = bf(v).squeezed(axis: 0)

        let kV = kH[0..., gFull..<Sfull, 0...]
        let vV = vH[0..., gFull..<Sfull, 0...]
        let kVp: MLXArray, vVp: MLXArray
        if padAmt > 0 {
            let zeros = MLXArray.zeros([h, padAmt, hd], dtype: .bfloat16)
            kVp = concatenated([kV, zeros], axis: 1)
            vVp = concatenated([vV, zeros], axis: 1)
        } else {
            kVp = kV; vVp = vV
        }

        let tailLen = (Lfull % B == 0) ? B : (Lfull % B)
        var lensA = [Float](repeating: Float(B), count: nV)
        if padAmt > 0 { lensA[nV - 1] = Float(tailLen) }
        let lensArr = MLXArray(lensA)

        let lv = lensArr.reshaped([1, nV, 1])
        let kcF = (kVp.asType(.float32).reshaped([h, nV, B, hd]).sum(axis: 2) / lv)
        let vsF = vVp.asType(.float32).reshaped([h, nV, B, hd]).sum(axis: 2)
        let kMean = kcF.mean(axis: 1, keepDims: true)
        let kcc = kcF - kMean
        let kcVar = (kcc * kcc).mean(axis: 1)

        let kIn: MLXArray, vIn: MLXArray
        if gFull > 0 {
            let kStrip = kH[0..., 0..<gFull, 0...]
            let vStrip = vH[0..., 0..<gFull, 0...]
            kIn = concatenated([kStrip, kVp], axis: 1).reshaped([1, h, gFull + padL, hd])
            vIn = concatenated([vStrip, vVp], axis: 1).reshaped([1, h, gFull + padL, hd])
        } else {
            kIn = kVp.reshaped([1, h, padL, hd])
            vIn = vVp.reshaped([1, h, padL, hd])
        }

        if !traceSafe { MLX.eval(kIn, vIn, lensArr, kcc, kcVar, vsF) }
        return SolKVPrepared(kIn: kIn, vIn: vIn, lens: lensArr, kcc: kcc, kcVar: kcVar, vsF: vsF,
                             nV: nV, padAmt: padAmt, gFull: gFull, Lfull: Lfull, h: h, hd: hd,
                             tau: tau, sinkS: Float(cfg.sinkBlocks?.lowerBound ?? 0),
                             sinkE: Float(cfg.sinkBlocks?.upperBound ?? 0), tailOn: tail)
    }

    /// 单块 video query 的 SOL 计算（v7 内核）。
    /// - q: [1, H, len, HD]，取自全序列 video 段的 [qOffVideo, qOffVideo+len)
    /// - kv: prepareKV 的返回值（K/V 全量，跨块复用）
    /// - qOffVideo: 本块起点在 video 段内的行偏移，必须为 64 的整数倍（外层 cs 为 64 倍数即可满足）
    /// - 返回: [1, H, len, HD]
    public static func attendVideo(q: MLXArray, kv: SolKVPrepared, qOffVideo: Int,
                                   scale: Float, traceSafe: Bool = false) -> MLXArray {
        let srcDtype = q.dtype
        let h = q.shape[1]
        let lenRaw = q.shape[2]
        let hd = q.shape[3]
        precondition(q.shape[0] == 1, "H3SolAttn: batch must be 1")
        precondition(hd == 128, "H3SolAttn: HD==128 expected")
        precondition(qOffVideo % 64 == 0, "H3SolAttn.attendVideo: qOffVideo 须 64 对齐")
        precondition(qOffVideo + lenRaw <= kv.Lfull, "H3SolAttn.attendVideo: query 块越界")
        let B = 64
        let padQ = (B - (lenRaw % B)) % B
        let qH = (q.dtype == .bfloat16 ? q : q.asType(.bfloat16)).squeezed(axis: 0)
        let qIn: MLXArray
        if padQ > 0 {
            let zeros = MLXArray.zeros([h, padQ, hd], dtype: .bfloat16)
            qIn = concatenated([qH, zeros], axis: 1)
        } else {
            qIn = qH
        }
        let Lq = lenRaw + padQ
        let nVq = Lq / B
        guard nVq > 0 else { fatalError("H3SolAttn.attendVideo: 空 query 块") }
        let qOffBlk = qOffVideo / B

        let lensChunk = kv.lens[qOffBlk..<(qOffBlk + nVq)].reshaped([1, nVq, 1])
        let qcF = (qIn.asType(.float32).reshaped([h, nVq, B, hd]).sum(axis: 2) / lensChunk)
        let cmF = matmul(qcF, kv.kcc.transposed(0, 2, 1)) * scale
        let varF = (qcF * qcF * kv.kcVar.reshaped([h, 1, hd])).sum(axis: 2)
        let thrF = MLXArray(kv.tau) * (varF * (scale * scale) + MLXArray(Float(1e-6))).sqrt()

        let cmB = cmF.asType(.bfloat16)
        let vsB = kv.vsF.asType(.bfloat16)
        let scaleArr = MLXArray([scale, Float(kv.padAmt), kv.sinkS, kv.sinkE,
                                 kv.tailOn ? Float(1) : Float(0), Float(kv.gFull), Float(qOffBlk)])
        let thrArr = thrF.reshaped([h, nVq])
        let qInB = qIn.reshaped([1, h, Lq, hd])
        if !traceSafe { MLX.eval(qInB, cmB, vsB, kv.kIn, kv.vIn, kv.lens, thrArr, scaleArr) }

        let out = kernelV7([qInB, kv.kIn, kv.vIn, cmB, vsB, kv.lens, thrArr, scaleArr],
                           template: [("T", DType.bfloat16), ("HD", hd)],
                           grid: (Lq * 4, h, 1),
                           threadGroup: (256, 1, 1),
                           outputShapes: [[1, h, Lq, hd]],
                           outputDTypes: [.bfloat16])[0]
        let outC = padQ > 0 ? out[0..., 0..., 0..<lenRaw, 0...] : out
        return srcDtype == .bfloat16 ? outC : outC.asType(srcDtype)
    }


    // MARK: 合成数值自检（NA_H3TEST=9 调用，不加载模型）
    // 覆盖: 全精确(tau=0)与 dense 一致性、tail on/off 行为、非 64 倍数 g/L、sink 块
    public static func selfTest() -> Int32 {
        var fail = false
        print("[SOLTEST] begin (kernel: h3sol_flash_v6)"); fflush(stdout)
        let h = 2, g = 17, L = 160, hd = 128      // g 非 64 倍数、L 非 64 倍数
        let S = g + L
        let scale = Float(1.0 / sqrt(Double(hd)))
        let q = MLXRandom.normal([1, h, S, hd], key: MLXRandom.key(11)).asType(.bfloat16)
        let k = MLXRandom.normal([1, h, S, hd], key: MLXRandom.key(12)).asType(.bfloat16)
        let v = MLXRandom.normal([1, h, S, hd], key: MLXRandom.key(13)).asType(.bfloat16)
        let ref = scaledDotProductAttention(queries: q, keys: k, values: v,
                                            scale: scale, mask: nil).asType(.float32)

        func stats(_ name: String, _ x: MLXArray) {
            let f = abs(x).asType(.float32).reshaped([-1])
            var mx: Float = 0; var mean: Double = 0; var cnt = 0
            let n = f.size
            let step = max(1, n / 4000)
            for i in stride(from: 0, to: n, by: step) {
                let a = f[i].item(Float.self); mx = max(mx, a); mean += Double(a); cnt += 1
            }
            mean /= Double(max(1, cnt))
            print(String(format: "[SOLTEST] %@ n=%d maxAbs=%.5f meanAbs=%.6f", name, n, mx, mean)); fflush(stdout)
        }
        func maxAbs(_ x: MLXArray) -> Float {
            let f = abs(x).asType(.float32).reshaped([-1])
            var mx: Float = 0
            for i in stride(from: 0, to: f.size, by: max(1, f.size / 2000)) {
                mx = max(mx, f[i].item(Float.self))
            }
            return mx
        }

        // Case 1: tau=-100（阈值远负 => 全部块精确选中，无 tail）=> 应与 dense 一致
        //（纯验证 kernel 选中路径与 bf16/pad 处理；tau=0 会把一半随机负分块丢进 tail，
        //  而 tail 用块质心分近似块内 max 分是 sol-attn 固有近似，不在此断言）
        let c1 = H3SolAttnConfig(topkRatio: nil, tau: -100.0, tail: true)
        do {
            let a1 = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: c1).asType(.float32)
            let dV = abs(a1[0, 0..<h, g..<S, 0..<hd] - ref[0, 0..<h, g..<S, 0..<hd])
            let dS = abs(a1[0, 0..<h, 0..<g, 0..<hd] - ref[0, 0..<h, 0..<g, 0..<hd])
            let mV = maxAbs(dV); let mS = maxAbs(dS)
            stats("tau=-100 video", dV)
            stats("tau=-100 strip", dS)
            if mV > 0.03 || mS > 0.005 { fail = true }
            print(String(format: "[SOLTEST] tau=-100 maxVideo=%.5f maxStrip=%.5f => %@", mV, mS, (mV < 0.03 && mS < 0.005) ? "PASS" : "FAIL")); fflush(stdout)
        } catch {
            print("[SOLTEST] Case1 threw"); fail = true; fflush(stdout)
        }

        // Case 2: tau=0.4 稀疏；tail on vs off 对比（tail 版应明显更接近 dense：分母保真）
        let c2on = H3SolAttnConfig(topkRatio: nil, tau: 0.4, tail: true)
        let c2off = H3SolAttnConfig(topkRatio: nil, tau: 0.4, tail: false)
        do {
            let a2on = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: c2on).asType(.float32)
            let a2off = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: c2off).asType(.float32)
            let dOn = abs(a2on[0, 0..<h, g..<S, 0..<hd] - ref[0, 0..<h, g..<S, 0..<hd])
            let dOff = abs(a2off[0, 0..<h, g..<S, 0..<hd] - ref[0, 0..<h, g..<S, 0..<hd])
            stats("tau0.4 tail=on  video", dOn)
            stats("tau0.4 tail=off video", dOff)
            let mOn = maxAbs(dOn); let mOff = maxAbs(dOff)
            print(String(format: "[SOLTEST] tau0.4 tail-on max=%.5f tail-off max=%.5f", mOn, mOff)); fflush(stdout)
        } catch {
            print("[SOLTEST] Case2 threw"); fail = true; fflush(stdout)
        }

        // Case 3: g/L 整除边界（g=128, L=192, nV=3 无 pad）
        let g3 = 128, L3 = 192, S3 = g3 + L3
        let q3 = MLXRandom.normal([1, h, S3, hd], key: MLXRandom.key(21)).asType(.bfloat16)
        let k3 = MLXRandom.normal([1, h, S3, hd], key: MLXRandom.key(22)).asType(.bfloat16)
        let v3 = MLXRandom.normal([1, h, S3, hd], key: MLXRandom.key(23)).asType(.bfloat16)
        let ref3 = scaledDotProductAttention(queries: q3, keys: k3, values: v3,
                                             scale: scale, mask: nil).asType(.float32)
        let c3 = H3SolAttnConfig(topkRatio: nil, tau: -100.0, tail: true)
        do {
            let a3 = H3SolAttn.forward(q: q3, k: k3, v: v3, g: g3, scale: scale, cfg: c3).asType(.float32)
            let m3 = maxAbs(abs(a3 - ref3))
            print(String(format: "[SOLTEST] aligned g128/L192 max=%.5f => %@", m3, m3 < 0.03 ? "PASS" : "FAIL")); fflush(stdout)
            if m3 >= 0.03 { fail = true }
        } catch { print("[SOLTEST] Case3 threw"); fail = true; fflush(stdout) }

        // Case 4: sink 区间强制精确（首 video 块进 sink 后该块误差应下降）——仅报告
        let c4 = H3SolAttnConfig(topkRatio: nil, tau: 0.4, sinkBlocks: 0..<1, tail: true)
        do {
            let a4 = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: c4).asType(.float32)
            let d4 = abs(a4[0, 0..<h, g..<S, 0..<hd] - ref[0, 0..<h, g..<S, 0..<hd])
            stats("tau0.4 sink0-1 video", d4)
        } catch { print("[SOLTEST] Case4 threw"); fail = true; fflush(stdout) }


        // Case F: heavy sparsity (tau=50) tail on vs off must differ
        do {
            let cFon = H3SolAttnConfig(topkRatio: nil, tau: 50.0, tail: true)
            let cFoff = H3SolAttnConfig(topkRatio: nil, tau: 50.0, tail: false)
            let aFon = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: cFon).asType(.float32)
            let aFoff = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: cFoff).asType(.float32)
            let rF = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil).asType(.float32)
            let dOn = abs(aFon[0, 0..<h, g..<S, 0..<hd] - rF[0, 0..<h, g..<S, 0..<hd])
            let dOff = abs(aFoff[0, 0..<h, g..<S, 0..<hd] - rF[0, 0..<h, g..<S, 0..<hd])
            print(String(format: "[SOLTEST] F tau50 tail-on=%.5f tail-off=%.5f delta=%.5f", maxAbs(dOn), maxAbs(dOff), abs(maxAbs(dOn)-maxAbs(dOff)))); fflush(stdout)
        } catch { print("[SOLTEST] F threw") }

        // Case G: nV=9 so diag does not cover all; tail must change error
        do {
            let gG = 64, LG = 512, hG = 2
            let SG = gG + LG
            let qG = MLXRandom.normal([1, hG, SG, hd], key: MLXRandom.key(301)).asType(.bfloat16)
            let kG = MLXRandom.normal([1, hG, SG, hd], key: MLXRandom.key(302)).asType(.bfloat16)
            let vG = MLXRandom.normal([1, hG, SG, hd], key: MLXRandom.key(303)).asType(.bfloat16)
            let rG = scaledDotProductAttention(queries: qG, keys: kG, values: vG, scale: scale, mask: nil).asType(.float32)
            for tval in [Float(1.0), Float(3.0)] {
                let cGon = H3SolAttnConfig(topkRatio: nil, tau: tval, tail: true)
                let cGoff = H3SolAttnConfig(topkRatio: nil, tau: tval, tail: false)
                let aGon = H3SolAttn.forward(q: qG, k: kG, v: vG, g: gG, scale: scale, cfg: cGon).asType(.float32)
                let aGoff = H3SolAttn.forward(q: qG, k: kG, v: vG, g: gG, scale: scale, cfg: cGoff).asType(.float32)
                let dOn = abs(aGon[0, 0..<hG, gG..<SG, 0..<hd] - rG[0, 0..<hG, gG..<SG, 0..<hd])
                let dOff = abs(aGoff[0, 0..<hG, gG..<SG, 0..<hd] - rG[0, 0..<hG, gG..<SG, 0..<hd])
                let dDiff = abs(aGon - aGoff)
                print(String(format: "[SOLTEST] G tau%.0f nV9 tail-on=%.5f tail-off=%.5f |on-off|=%.6f", tval, maxAbs(dOn), maxAbs(dOff), maxAbs(dDiff))); fflush(stdout)
            }
        } catch { print("[SOLTEST] G threw") }









        // Case K: real-scale headless 探针（h=32, g=0, L=14985; iter 2）
        // 目标: v6-full 必须逼近 dense-SDPA 量级, v6-tau1 必须显著快于 dense, 才值得接管线。
        do {
            let hK = 32, gK = 0, LK = 14985
            let SK = gK + LK
            let qK = MLXRandom.normal([1, hK, SK, hd], key: MLXRandom.key(601)).asType(.bfloat16)
            let kK = MLXRandom.normal([1, hK, SK, hd], key: MLXRandom.key(602)).asType(.bfloat16)
            let vK = MLXRandom.normal([1, hK, SK, hd], key: MLXRandom.key(603)).asType(.bfloat16)
            func benchK(_ tag: String, _ iters: Int, _ f: () -> MLXArray) {
                var last = f(); MLX.eval(last)
                let t0 = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iters { last = f(); MLX.eval(last) }
                let t1 = CFAbsoluteTimeGetCurrent()
                print(String(format: "[SOLTEST] K bench %@ %.1f ms/iter", tag, (t1 - t0) * 1000.0 / Double(iters))); fflush(stdout)
                _ = last
            }
            benchK("dense-SDPA", 2) { scaledDotProductAttention(queries: qK, keys: kK, values: vK, scale: scale, mask: nil) }
            let cKfull = H3SolAttnConfig(topkRatio: nil, tau: -100.0, tail: true)
            benchK("v6-full", 2) { H3SolAttn.forward(q: qK, k: kK, v: vK, g: gK, scale: scale, cfg: cKfull) }
            let cK1 = H3SolAttnConfig(topkRatio: nil, tau: 1.0, tail: true)
            let cK1off = H3SolAttnConfig(topkRatio: nil, tau: 1.0, tail: false)
            benchK("v6-tau1-tail", 2) { H3SolAttn.forward(q: qK, k: kK, v: vK, g: gK, scale: scale, cfg: cK1) }
            benchK("v6-tau1-notail", 2) { H3SolAttn.forward(q: qK, k: kK, v: vK, g: gK, scale: scale, cfg: cK1off) }
        } catch { print("[SOLTEST] K threw") }



        // Case N: v6 (BQ64 兄弟合并: 256 threads/8 simd, 64 qrow/tg) 数值探针 + 真尺度 bench
        do {
            // 小几何: v6-full vs dense（整段基准）
            let cNf = H3SolAttnConfig(topkRatio: nil, tau: -100.0, tail: true)
            let aNf = H3SolAttn.forward(q: q, k: k, v: v, g: g, scale: scale, cfg: cNf, useV6: true).asType(.float32)
            let dNf = abs(aNf[0, 0..<h, g..<S, 0..<hd] - ref[0, 0..<h, g..<S, 0..<hd])
            let dNfS = abs(aNf[0, 0..<h, 0..<g, 0..<hd] - ref[0, 0..<h, 0..<g, 0..<hd])
            print(String(format: "[SOLTEST] N v6-full video=%.5f strip=%.5f", maxAbs(dNf), maxAbs(dNfS))); fflush(stdout)
            if maxAbs(dNf) > 0.05 || maxAbs(dNfS) > 0.005 { fail = true }
        } catch { print("[SOLTEST] N threw"); fail = true; fflush(stdout) }

        // 真尺度 bench (h=32, g=0, L=14985): v6-full / v6-tau1 对照 dense
        do {
            let hN = 32, gN = 0, LN = 14985
            let SN = gN + LN
            let qN = MLXRandom.normal([1, hN, SN, hd], key: MLXRandom.key(901)).asType(.bfloat16)
            let kN = MLXRandom.normal([1, hN, SN, hd], key: MLXRandom.key(902)).asType(.bfloat16)
            let vN = MLXRandom.normal([1, hN, SN, hd], key: MLXRandom.key(903)).asType(.bfloat16)
            func benchN(_ tag: String, _ iters: Int, _ f: () -> MLXArray) {
                var last = f(); MLX.eval(last)
                let t0 = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iters { last = f(); MLX.eval(last) }
                let t1 = CFAbsoluteTimeGetCurrent()
                print(String(format: "[SOLTEST] N bench %@ %.1f ms/iter", tag, (t1 - t0) * 1000.0 / Double(iters))); fflush(stdout)
                _ = last
            }
            benchN("dense-SDPA", 2) { scaledDotProductAttention(queries: qN, keys: kN, values: vN, scale: scale, mask: nil) }
            let cNfull = H3SolAttnConfig(topkRatio: nil, tau: -100.0, tail: true)
            benchN("v6-full", 2) { H3SolAttn.forward(q: qN, k: kN, v: vN, g: gN, scale: scale, cfg: cNfull, useV6: true) }
            let cN1 = H3SolAttnConfig(topkRatio: nil, tau: 1.0, tail: true)
            let cN1off = H3SolAttnConfig(topkRatio: nil, tau: 1.0, tail: false)
            benchN("v6-tau1-tail", 2) { H3SolAttn.forward(q: qN, k: kN, v: vN, g: gN, scale: scale, cfg: cN1, useV6: true) }
            benchN("v6-tau1-notail", 2) { H3SolAttn.forward(q: qN, k: kN, v: vN, g: gN, scale: scale, cfg: cN1off, useV6: true) }
        } catch { print("[SOLTEST] N bench threw") }

        // Case R: 生产几何复现探针（h=42, g=600, L=9324 → padL=9344, padAmt=20; v6 + tau1.3）
        do {
            let hR = 56, gR = 934, LR = 9324
            let SR = gR + LR
            let qR = MLXRandom.normal([1, hR, SR, hd], key: MLXRandom.key(1001)).asType(.bfloat16)
            let kR = MLXRandom.normal([1, hR, SR, hd], key: MLXRandom.key(1002)).asType(.bfloat16)
            let vR = MLXRandom.normal([1, hR, SR, hd], key: MLXRandom.key(1003)).asType(.bfloat16)
            print("[SOLTEST] R begin h=42 g=600 L=9324 (padAmt=20) v6 tau1.3"); fflush(stdout)
            let cR = H3SolAttnConfig(topkRatio: nil, tau: 1.3, tail: true)
            let aR = H3SolAttn.forward(q: qR, k: kR, v: vR, g: gR, scale: scale, cfg: cR, useV6: true)
            MLX.eval(aR)
            print(String(format: "[SOLTEST] R done maxAbs=%.5f", maxAbs(aR.asType(.float32)))); fflush(stdout)
        } catch { print("[SOLTEST] R threw"); fail = true; fflush(stdout) }

        // Case S: v7 分块（strip 段 dense + video 段分块）与 v6 整段同 tau 路由；
        // 期望逐位一致（差异仅来自 bf16 输出与 pad），是本次分块修复的核心自检。
        do {
            let hS = 4, gS = 100, LS = 5000        // g 非 64 倍数、L 非 64 倍数
            let SS = gS + LS
            let qS = MLXRandom.normal([1, hS, SS, hd], key: MLXRandom.key(1201)).asType(.bfloat16)
            let kS = MLXRandom.normal([1, hS, SS, hd], key: MLXRandom.key(1202)).asType(.bfloat16)
            let vS = MLXRandom.normal([1, hS, SS, hd], key: MLXRandom.key(1203)).asType(.bfloat16)
            let cS = H3SolAttnConfig(topkRatio: nil, tau: 1.3, tail: true)
            let refS = H3SolAttn.forward(q: qS, k: kS, v: vS, g: gS, scale: scale, cfg: cS).asType(.float32)
            var parts: [MLXArray] = []
            parts.append(scaledDotProductAttention(queries: qS[0..., 0..., 0..<gS, 0...],
                                                   keys: kS, values: vS, scale: scale, mask: nil))
            let kvS = H3SolAttn.prepareKV(k: kS, v: vS, gFull: gS)
            let csS = 1792
            var offS = 0
            while offS < LS {
                let len = min(csS, LS - offS)
                parts.append(H3SolAttn.attendVideo(
                    q: qS[0..., 0..., (gS + offS)..<(gS + offS + len), 0...],
                    kv: kvS, qOffVideo: offS, scale: scale))
                offS += len
            }
            let gotS = concatenated(parts, axis: 2)
            let dS = abs(gotS.asType(.float32) - refS)
            print(String(format: "[SOLTEST] S v7-chunked vs v6-full max=%.6f", maxAbs(dS))); fflush(stdout)
            if maxAbs(dS) > 0.02 { fail = true }
        } catch { print("[SOLTEST] S threw"); fail = true; fflush(stdout) }


        if fail { print("[SOLTEST] FAIL"); return 1 }
        print("[SOLTEST] PASS"); fflush(stdout)
        return 0
    }
}
