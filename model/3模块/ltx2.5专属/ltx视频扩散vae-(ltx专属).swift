import Darwin
import MLX
import MLXRandom

// MARK: - LTX-2.5 扩散视频解码器（Diffusion Video Decoder）Swift/MLX 实现
//
// 依据 diffusers `LTX2VideoDiffusionDecoderModel`（ltx2_diffusion_decoder.py）移植：
//   - det_stages.0~3：确定性上采样阶段（NA block + PixelShuffleUpsampler），从 latent 构建 context volume；
//   - diff_blocks.0~7：像素空间扩散去噪（NA + context_proj 注入 + AdaLN-Zero scale/shift），
//     官方默认 1 步 x0 预测（timesteps=[1.0]，单次前向直接出像素）。
//
// 权重来源（已核实）：
//   - 当前首选：官方 PyTorch 原版 `ltx-2.5-video-vae-bf16.safetensors`（1.4GB，396 张量，bf16，
//     完整 VAE；decoder 部分即扩散解码器结构 det_stages+diff_blocks，键前缀 `decoder.`、
//     注意力为合并 `attn.qkv`，本文件已适配；per_channel_statistics 键为
//     mean-of-means/std-of-means，由 vaeDecodeTest 直接读取）；
//   - 兼容旧 mlx-community 转换包：`vae_diffusion_decoder.safetensors`（约 834MB，408 张量，bf16，
//     前缀 `vae_diffusion_decoder.`，注意力为 to_q/to_k/to_v 拆分键）；
//   - 若需从 PyTorch 自转：对每个 Linear/权重执行 `w = w.transpose(0,1)` 改为 [out,in] 布局，
//     全量转 bf16，键名去掉前缀改为 `vae_diffusion_decoder.*`（或保留原键由本文件统一剥前缀）。
//
// 布局约定：
//   - latent 入：NDHWC [B,T,H,W,128]（与卷积 VAE 完全相同，SGLang 文档证实两者消费同一 latent）
//   - 像素出：BCFHW [B,3,F,H*4,W*4]，值域 [-1,1]（官方 model_output_type=x0，1 步时即预测像素；
//     与现有 VAE 解码输出一致，下游 fillPixelBuffer 内部归一化到 [0,1]）。
//
// 官方关键参数（已与本地权重键逐一核对）：
//   stage_channels=(2048,1024,512,512,256)、stage_depths=(4,6,4,2,8)、
//   kernels=((3,7,7),(3,7,7),(3,5,5),(3,5,5))、stage5_kernel=(11,11,11)、
//   upsample_strides=((1,2,2),(2,1,1),(2,2,2),(2,2,2))、upsample_reductions=(2,2,1,2)、
//   patch_size=4、head_dim=64、t_emb_dim=384、timestep_scale_multiplier=1000、temporal_compression_ratio=8、
//   trailing_pad_latent_frames=2、default_num_inference_steps=1。

enum DDVDecoder {

    // MARK: - 结构配置（官方 config，已由权重 shape 复核；如需调整须与权重匹配）

    struct Config {
        static let stageChannels: [Int] = [2048, 1024, 512, 512, 256]
        static let stageDepths: [Int] = [4, 6, 4, 2, 8]
        static let stageKernels: [(Int, Int, Int)] = [(3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5)]
        static let stage5Kernel: (Int, Int, Int) = (11, 11, 11)
        static let upsampleStrides: [(Int, Int, Int)] = [(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)]
        static let upsampleReductions: [Int] = [2, 2, 1, 2]
        static let patchSize = 4
        static let headDim = 64
        static let stage5Channels = stageChannels[4]  // AdaLN-Zero 调制维度（官方 dim=stage_channels[-1]=256）
        static let tEmbDim = 384
        static let timestepScaleMultiplier: Float = 1000.0
        static let temporalCompressionRatio = 8
        static let trailingPadLatentFrames = 2
        static let defaultNumInferenceSteps = 1
        static let modelOutputType = "x0"  // 官方固定 "x0"
    }

    // MARK: - 权重键

    // 兼容两种权重格式：官方 PyTorch 原版（ltx-2.5-video-vae-bf16.safetensors，前缀 decoder.）
    // 与旧 mlx-community 转换版（vae_diffusion_decoder.safetensors，前缀 vae_diffusion_decoder.）。
    private static func key(_ w: [String: MLXArray], _ k: String) -> MLXArray {
        if let v = w["decoder.\(k)"] { return v }
        if let v = w["vae_diffusion_decoder.\(k)"] { return v }
        fatalError("DDVDecoder: 缺少权重键 decoder.\(k) / vae_diffusion_decoder.\(k)")
    }

    /// 注意力 q/k/v 投影权重。官方 PyTorch 原版为合并键 attn.qkv.weight/bias（[3C,C]/[3C]，行序 q|k|v），
    /// 按行切分为 to_q/to_k/to_v（布局 [out,in]/[out]，与 NA 消费一致）；旧 mlx 版已是拆分键直接取用。
    private static func attnQKV(_ w: [String: MLXArray], _ k: String) -> (
        qW: MLXArray, qB: MLXArray, kW: MLXArray, kB: MLXArray, vW: MLXArray, vB: MLXArray
    ) {
        if let qkvW = w["decoder.\(k).attn.qkv.weight"], let qkvB = w["decoder.\(k).attn.qkv.bias"] {
            let dim = qkvW.shape[0] / 3
            let rows0 = MLXArray(0..<dim)
            let rows1 = MLXArray(dim..<(2 * dim))
            let rows2 = MLXArray((2 * dim)..<(3 * dim))
            return (
                qkvW.take(rows0, axis: 0), qkvB.take(rows0, axis: 0),
                qkvW.take(rows1, axis: 0), qkvB.take(rows1, axis: 0),
                qkvW.take(rows2, axis: 0), qkvB.take(rows2, axis: 0)
            )
        }
        return (
            key(w, "\(k).attn.to_q.weight"), key(w, "\(k).attn.to_q.bias"),
            key(w, "\(k).attn.to_k.weight"), key(w, "\(k).attn.to_k.bias"),
            key(w, "\(k).attn.to_v.weight"), key(w, "\(k).attn.to_v.bias")
        )
    }

    private static func linear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
        x.matmul(w.transposed(1, 0)) + b
    }

    // MARK: - timestep embedding（官方 Timesteps + TimestepEmbedding：正弦 256 → Linear 256→384 → silu → Linear 384→384）

    static func timestepEmbedding(_ t: Float, _ w: [String: MLXArray]) -> MLXArray {
        let half = 128
        let invFreq = (0..<half).map { i -> Float in
            expf(-Float(i) * logf(10000.0) / Float(half))
        }
        let ang = MLXArray([t * Config.timestepScaleMultiplier]) * MLXArray(invFreq)  // (1,128)
        let emb0 = concatenated([sin(ang), cos(ang)], axis: -1)             // (1,256)
        // flip_sin_to_cos=True：cos 在前 sin 在后
        let emb = concatenated([
            emb0.take(MLXArray(128..<256), axis: -1),
            emb0.take(MLXArray(0..<128), axis: -1),
        ], axis: -1)
        // 官方键名：t_embedder.mlp.0.{weight,bias} / t_embedder.mlp.2.{weight,bias}
        let h = linear(emb, key(w, "t_embedder.mlp.0.weight"), key(w, "t_embedder.mlp.0.bias"))
        let a = siluActivation(h)
        return linear(a, key(w, "t_embedder.mlp.2.weight"), key(w, "t_embedder.mlp.2.bias"))  // (1,384)
    }

    // MARK: - PixelShuffleUpsampler（官方 LTX2VideoVaePixelShuffleUpsampler）

    /// x (B,T,H,W,inC) → Linear 扩展 → pixel shuffle → (B,T*st,H*sh,W*sw,outC)
    /// stride_t==2 时 dropLeadingFrame=true 去掉重复首帧（官方）
    static func pixelShuffleUpsample(
        _ x: MLXArray, projW: MLXArray, projB: MLXArray,
        stride: (Int, Int, Int), reduction: Int, dropLeadingFrame: Bool = true
    ) -> MLXArray {
        let shape = x.shape
        let inC = shape[4]
        let (st, sh, sw) = stride
        let prod = st * sh * sw
        let projOut = prod * inC / reduction
        let outC = projOut / prod
        var h = linear(x, projW, projB)
        h = h.reshaped([shape[0], shape[1], shape[2], shape[3], outC, st, sh, sw])
        h = h.transposed(0, 1, 5, 2, 6, 3, 7, 4)          // (b,t,st,h,sh,w,sw,c)
        h = h.reshaped([shape[0], shape[1] * st, shape[2] * sh, shape[3] * sw, outC])
        if st == 2 && dropLeadingFrame {
            h = h.take(MLXArray(1..<h.shape[1]), axis: 1)
        }
        return h
    }

    // MARK: - patchify / unpatchify（官方 _patchify / _unpatchify，patch_size=4）

    /// (B,C,T,H,W) → (B,C*16,T,H/4,W/4)
    static func patchify(_ x: MLXArray, patchSize: Int) -> MLXArray {
        let B = x.shape[0], C = x.shape[1], T = x.shape[2], H = x.shape[3], W = x.shape[4]
        let p = patchSize
        var h = x.reshaped([B, C, T, H / p, p, W / p, p])
        h = h.transposed(0, 1, 6, 4, 2, 3, 5)             // (B,C,p,p,T,H/p,W/p)
        return h.reshaped([B, C * p * p, T, H / p, W / p])
    }

    /// (B,C*16,T,H/4,W/4) → (B,C,T,H,W)
    static func unpatchify(_ x: MLXArray, patchSize: Int) -> MLXArray {
        let B = x.shape[0], Cp = x.shape[1], T = x.shape[2], H = x.shape[3], W = x.shape[4]
        let p = patchSize
        let C = Cp / (p * p)
        var h = x.reshaped([B, C, p, p, T, H, W])
        h = h.transposed(0, 1, 4, 5, 2, 6, 3)             // (B,C,T,H,p,W,p)
        return h.reshaped([B, C, T, H * p, W * p])
    }

    // MARK: - 确定性阶段：stages 1~3（latent → features，NDHWC，含尾部 ghost 帧）

    static func forwardStages1to3(_ latentBCFHW: MLXArray, _ w: [String: MLXArray]) -> MLXArray {
        var x = latentBCFHW
        // 尾部补 2 个 ghost 帧（复制最后一帧，官方 trailing_pad_latent_frames=2）
        let numPad = Config.trailingPadLatentFrames
        if numPad > 0 {
            let last = x.take(MLXArray([x.shape[2] - 1]), axis: 2)   // (B,C,1,H,W)
            var pads: [MLXArray] = [x]
            for _ in 0..<numPad { pads.append(last) }
            x = concatenated(pads, axis: 2)
        }
        x = x.transposed(0, 2, 3, 4, 1)                              // (B,T,H,W,128) NDHWC
        x = linear(x, key(w, "conv_in.weight"), key(w, "conv_in.bias"))  // → 2048
        for si in 0..<3 {
            let dep = Config.stageDepths[si]
            for bi in 0..<dep {
                let qkv = attnQKV(w, "det_stages.\(si).\(bi)")
                x = DDVNA.naBlock(
                    x,
                    norm1W: key(w, "det_stages.\(si).\(bi).norm1.weight"),
                    norm2W: key(w, "det_stages.\(si).\(bi).norm2.weight"),
                    kernel: Config.stageKernels[si],
                    attnQW: qkv.qW,
                    attnQB: qkv.qB,
                    attnKW: qkv.kW,
                    attnKB: qkv.kB,
                    attnVW: qkv.vW,
                    attnVB: qkv.vB,
                    attnPW: key(w, "det_stages.\(si).\(bi).attn.proj.weight"),
                    attnPB: key(w, "det_stages.\(si).\(bi).attn.proj.bias"),
                    qNormW: key(w, "det_stages.\(si).\(bi).attn.q_norm.weight"),
                    kNormW: key(w, "det_stages.\(si).\(bi).attn.k_norm.weight"),
                    gateW: key(w, "det_stages.\(si).\(bi).mlp.w_gate.weight"),
                    upW: key(w, "det_stages.\(si).\(bi).mlp.w_up.weight"),
                    downW: key(w, "det_stages.\(si).\(bi).mlp.w_down.weight")
                )
                eval(x)  // 每层 NA block 后立即落地，释放该层计算图，防 det 阶段内存爆炸
            }
            x = pixelShuffleUpsample(
                x,
                projW: key(w, "upsamples.\(si).proj.weight"),
                projB: key(w, "upsamples.\(si).proj.bias"),
                stride: Config.upsampleStrides[si],
                reduction: Config.upsampleReductions[si]
            )
            memPointLog("det stage\(si + 1) 完成")
        }
        return x  // (B,T',H',W',512)
    }

    // MARK: - 确定性阶段：stage 4（features → context volume，NDHWC (B,T5,H5,W5,256)）

    static func forwardStage4(
        _ features: MLXArray, _ w: [String: MLXArray],
        dropLeadingFrame: Bool = true, cropTrailingGhost: Bool = true
    ) -> MLXArray {
        memPointLog("stage4 开始")
        var x = features
        let si = 3
        for bi in 0..<Config.stageDepths[si] {
            let qkv = attnQKV(w, "det_stages.\(si).\(bi)")
            x = DDVNA.naBlock(
                x,
                norm1W: key(w, "det_stages.\(si).\(bi).norm1.weight"),
                norm2W: key(w, "det_stages.\(si).\(bi).norm2.weight"),
                kernel: Config.stageKernels[si],
                attnQW: qkv.qW,
                attnQB: qkv.qB,
                attnKW: qkv.kW,
                attnKB: qkv.kB,
                attnVW: qkv.vW,
                attnVB: qkv.vB,
                attnPW: key(w, "det_stages.\(si).\(bi).attn.proj.weight"),
                attnPB: key(w, "det_stages.\(si).\(bi).attn.proj.bias"),
                qNormW: key(w, "det_stages.\(si).\(bi).attn.q_norm.weight"),
                kNormW: key(w, "det_stages.\(si).\(bi).attn.k_norm.weight"),
                gateW: key(w, "det_stages.\(si).\(bi).mlp.w_gate.weight"),
                upW: key(w, "det_stages.\(si).\(bi).mlp.w_up.weight"),
                downW: key(w, "det_stages.\(si).\(bi).mlp.w_down.weight")
            )
            eval(x)  // 每层 NA block 后立即落地，释放该层计算图，防 det 阶段内存爆炸
        }
        x = pixelShuffleUpsample(
            x,
            projW: key(w, "upsamples.\(si).proj.weight"),
            projB: key(w, "upsamples.\(si).proj.bias"),
            stride: Config.upsampleStrides[si],
            reduction: Config.upsampleReductions[si],
            dropLeadingFrame: dropLeadingFrame
        )
        if cropTrailingGhost {
            let numPad = Config.trailingPadLatentFrames * Config.temporalCompressionRatio  // 2*8=16 帧
            x = x.take(MLXArray(0..<(x.shape[1] - numPad)), axis: 1)
        }
        memPointLog("stage4 完成")
        return x
    }

    // MARK: - 扩散阶段：单个去噪步（timestep-conditioned，官方 forward_diffusion_step）

    /// latentContext (B,T5,H5,W5,256)；x_t (B,3,T5,H5*4,W5*4)；timestep 标量（官方 1 步=1.0）
    static func forwardDiffusionStep(
        latentContext: MLXArray, x_t: MLXArray, timestep: Float, _ w: [String: MLXArray]
    ) -> MLXArray {
        let tEmb = timestepEmbedding(timestep, w)                    // (1,384)
        let mod = linear(siluActivation(tEmb), key(w, "shared_adaln.proj.weight"), key(w, "shared_adaln.proj.bias"))
        let chunks = mod.reshaped([1, 1, 1, 1, 7, Config.stage5Channels])   // 官方 chunk(7, dim=-1)：每块 stage5 通道 256
        func chunk(_ i: Int) -> MLXArray {
            chunks.take(MLXArray([Int32(i)]), axis: 4).reshaped([1, 1, 1, 1, Config.stage5Channels])
        }

        var hidden = patchify(x_t, patchSize: Config.patchSize)
        hidden = hidden.transposed(0, 2, 3, 4, 1)                    // (B,T5,H5,W5,48) NDHWC
        hidden = linear(hidden, key(w, "conv_in_x_t.weight"), key(w, "conv_in_x_t.bias"))  // →256

        for bi in 0..<Config.stageDepths[4] {
            let ss = key(w, "diff_blocks.\(bi).scale_shift_table")   // (7,256)
            // AdaLN-Zero 调制 + 每 block 的 scale_shift_table 残差（官方取 chunks 0..6，用到 0/1/3/4）
            let s1 = chunk(0) + ss.take(MLXArray([0]), axis: 0).reshaped([1, 1, 1, 1, Config.stage5Channels])
            let sh1 = chunk(1) + ss.take(MLXArray([1]), axis: 0).reshaped([1, 1, 1, 1, Config.stage5Channels])
            let s2 = chunk(3) + ss.take(MLXArray([3]), axis: 0).reshaped([1, 1, 1, 1, Config.stage5Channels])
            let sh2 = chunk(4) + ss.take(MLXArray([4]), axis: 0).reshaped([1, 1, 1, 1, Config.stage5Channels])

            var h = hidden
            // context volume 注入（官方：hidden = hidden + context_proj(latent_context)）
            let ctx = linear(latentContext, key(w, "diff_blocks.\(bi).context_proj.weight"), key(w, "diff_blocks.\(bi).context_proj.bias"))
            h = h + ctx
            // NA attn（AdaLN 调制后）
            let n1 = rmsNormWeighted(h, weight: key(w, "diff_blocks.\(bi).norm1.weight"))
            let qkv = attnQKV(w, "diff_blocks.\(bi)")
            let a = DDVNA.neighborhoodAttention(
                n1 * (1 + s1) + sh1, kernel: Config.stage5Kernel,
                qNormW: key(w, "diff_blocks.\(bi).attn.q_norm.weight"),
                kNormW: key(w, "diff_blocks.\(bi).attn.k_norm.weight"),
                toQW: qkv.qW,
                toQB: qkv.qB,
                toKW: qkv.kW,
                toKB: qkv.kB,
                toVW: qkv.vW,
                toVB: qkv.vB,
                projW: key(w, "diff_blocks.\(bi).attn.proj.weight"),
                projB: key(w, "diff_blocks.\(bi).attn.proj.bias")
            )
            h = h + a
            // SwiGLU MLP（AdaLN 调制后；对齐官方 _SWIGLU_TILE_SIZE=16384 按 token 分块，消除 3×hidden 整段中间张量）
            let n2 = rmsNormWeighted(h, weight: key(w, "diff_blocks.\(bi).norm2.weight"))
            let modIn = n2 * (1 + s2) + sh2
            h = h + swigluMLP(
                modIn,
                upW: key(w, "diff_blocks.\(bi).mlp.w_up.weight"), upB: nil,
                gateW: key(w, "diff_blocks.\(bi).mlp.w_gate.weight"), gateB: nil,
                downW: key(w, "diff_blocks.\(bi).mlp.w_down.weight")
            )
            hidden = h
            // 每层 diff_block 后立即落地：stage5 在 context 网格（tile 96³×256ch 单份 ~4.5GB），
            // 不 eval 会累积 8 层完整计算图（hidden 新旧两份 + ctx/attn/mlp 中间链），峰值直逼上限；
            // 逐层 eval 后峰值只留单层图 + hidden 结果，同 det 阶段治理手法
            eval(hidden)
        }

        hidden = rmsNormWeighted(hidden, weight: key(w, "norm_out.weight"))
        hidden = linear(hidden, key(w, "conv_out.weight"), key(w, "conv_out.bias"))  // →48
        hidden = hidden.transposed(0, 4, 1, 2, 3)                   // (B,48,T5,H5,W5)
        return unpatchify(hidden, patchSize: Config.patchSize)      // (B,3,T5,H5*4,W5*4)
    }

    // MARK: - 去噪循环（官方 denoise：默认 1 步 x0 短路；多步 reverse Euler）

    static func denoise(latentContext: MLXArray, x_t: MLXArray, steps: Int, _ w: [String: MLXArray]) -> MLXArray {
        if steps == 1 && Config.modelOutputType == "x0" {
            return forwardDiffusionStep(latentContext: latentContext, x_t: x_t, timestep: 1.0, w)
        }
        // 多步：timesteps = linspace(1.0, 1/steps, steps)；x0 输出转 eps = (x_t - model_out)/sigma
        var x = x_t
        for si in 0..<steps {
            let tNow = 1.0 - Float(si) * (1.0 - 1.0 / Float(steps)) / Float(max(steps - 1, 1))
            let tNext = si + 1 < steps
                ? 1.0 - Float(si + 1) * (1.0 - 1.0 / Float(steps)) / Float(max(steps - 1, 1))
                : 0.0
            let modelOut = forwardDiffusionStep(latentContext: latentContext, x_t: x, timestep: tNow, w).asType(.float32)
            let xf = x.asType(.float32)
            var eps = modelOut
            if Config.modelOutputType == "x0" {
                eps = (xf - modelOut) / tNow
            }
            let dt = tNow - tNext
            x = (xf - dt * eps).asType(x.dtype)
        }
        return x
    }

    // MARK: - 分块解码（tiling）：stage5 扩散阶段按重叠 tile 独立去噪后线性融合

    /// 生成覆盖 [0,length) 的重叠 tile [start,end)，start 间距 stride，尾部残块并入前块（官方 _tile_intervals 语义）
    static func tileIntervals(length: Int, tileSize: Int, stride: Int, minSize: Int) -> [(Int, Int)] {
        if length <= tileSize { return [(0, length)] }
        var starts: [Int] = []
        var s = 0
        while s < length { starts.append(s); s += stride }
        if starts.isEmpty { starts.append(0) }
        while starts.count > 1 && length - starts.last! < minSize {
            starts.removeLast()
        }
        var out: [(Int, Int)] = []
        for s in starts.dropLast() {
            out.append((s, min(s + tileSize, length)))
        }
        out.append((starts.last!, length))
        return out
    }

    /// 入口：latent NDHWC [B,T,H,W,128] → 像素 BCFHW [B,3,8T-7,32H,32W]
    /// - Parameters:
    ///   - weights: safetensors 权重（键含前缀 vae_diffusion_decoder.）
    ///   - latentNDHWC: 已归一化 latent（与卷积 VAE 输入相同，直接使用）
    ///   - numInferenceSteps: 扩散去噪步数，官方默认 1
    ///   - useTiling: 是否分块解码（默认 true；整段 stage5 峰值超 48G 可用 Metal 内存会
    ///     [metal::malloc] Resource limit 崩溃，官方语义 tiling=显存降级）
    ///   - tileFrames/tileSpatial: context 网格上的 tile 尺寸（默认 128×128 → features 网格 64×64，
    ///     满足官方 tile ≥ 2×overlap 硬约束，保证左右 ramp 互补；
    ///     96→128 后重叠浪费 42%→31%，512×512 空间轴单 tile、时间轴 1~2 块，投影重复也更少）
    ///   - overlapFrames/overlapSpatial: tile 重叠（context 网格，默认 40/40 → features 网格 20/20，
    ///     即官方 compute_tile_halos 推导的 halo 质量下限；低于此值拼接处注意力视野截断会出画面问题）
    static func decode(
        weights: [String: MLXArray],
        latentNDHWC: MLXArray,
        numInferenceSteps: Int = Config.defaultNumInferenceSteps,
        useTiling: Bool = true,
        tileFrames: Int = 128,
        tileSpatial: Int = 128,
        overlapFrames: Int = 40,
        overlapSpatial: Int = 40
    ) -> MLXArray {
        let latentBCFHW = latentNDHWC.asType(latentNDHWC.dtype).transposed(0, 4, 1, 2, 3)  // (B,128,T,H,W)，整链 bf16 内存减半
        let features = forwardStages1to3(latentBCFHW, weights)       // (B,T3,H3,W3,512)
        eval(features)
        memPointLog("stage1-3 完成，stage3 前 active")

        // features（stage3 输出）网格含尾部 ghost 帧（官方 trailing_pad 经 stages1-3 时间上采样后复制帧数：
        // 2 × stage1.stride_t(1) × stage2.stride_t(2) × stage3.stride_t(2) = 8）
        let ghostFrames = Config.trailingPadLatentFrames
            * Config.upsampleStrides[0].0 * Config.upsampleStrides[1].0 * Config.upsampleStrides[2].0
        let numFrames = features.shape[1] - ghostFrames              // 有效视频帧数（features 网格）

        let B = features.shape[0]
        let H3 = features.shape[2], W3 = features.shape[3]
        let p = Config.patchSize
        // stage4 (2,2,2) upsample：context 网格；origin tile 丢 leading 帧（官方 drop_first_frame）
        let T5 = 2 * numFrames - 1
        let H5 = 2 * H3, W5 = 2 * W3
        let pxT = T5, pxH = H5 * p, pxW = W5 * p

        if !useTiling {
            let context = forwardStage4(features, weights)           // 整段（drop leading + crop ghost）
            eval(context)
            let xT = MLXRandom.normal([B, 3, pxT, pxH, pxW]).asType(PrecisionPolicy.defaultMainDType)
            return denoise(latentContext: context, x_t: xT, steps: numInferenceSteps, weights)
        }

        // —— tiled（对齐官方 tiled_decode）：stage4 + stage5 逐 tile 执行，tile 网格取在 features
        // （stage3 输出）上：context 网格 tile 40×24×24 → features 网格 20×12×12；
        // overlap 折算 = context overlap / 2 = 帧 3、空间 8。非 origin tile 保留重复 leading 帧
        // （stage4 upsample 不 drop），融合时其全局起点 = 2*ts - 1；trailing tile 把 ghost 帧一起
        // 带入 stage4 且不 crop，融合时画布边缘外（ghost 延伸）直接丢弃，输出帧数与整段一致。 ——
        let tileF3 = max(tileFrames / 2, 6)
        let tileS3 = max(tileSpatial / 2, 12)
        let ovF3 = max(overlapFrames / 2, 2)
        let ovS3 = max(overlapSpatial / 2, 4)
        let tTiles = tileIntervals(length: numFrames, tileSize: tileF3, stride: tileF3 - ovF3, minSize: 6)
        let hTiles = tileIntervals(length: H3, tileSize: tileS3, stride: tileS3 - ovS3, minSize: 5)
        let wTiles = tileIntervals(length: W3, tileSize: tileS3, stride: tileS3 - ovS3, minSize: 5)
        let totalTiles = tTiles.count * hTiles.count * wTiles.count

        var acc = MLXArray.zeros([B, 3, pxT, pxH, pxW], dtype: PrecisionPolicy.defaultMainDType)
        var wacc = MLXArray.zeros([B, 1, pxT, pxH, pxW], dtype: PrecisionPolicy.defaultMainDType)

        var tileIndex = 0
        for (ts, te) in tTiles {
            let isOrigin = ts == 0
            let isTrailing = te == numFrames
            // trailing tile 把 ghost 帧一起带入 stage4 且不 crop（官方 feature_t1 = features.shape[1]）
            let featTe = isTrailing ? features.shape[1] : te
            for (hs, he) in hTiles {
                for (ws, we) in wTiles {
                    let featuresTile = features[0..<B, ts..<featTe, hs..<he, ws..<we, 0..<512]
                    let ctxTile = forwardStage4(
                        featuresTile, weights,
                        dropLeadingFrame: isOrigin,
                        cropTrailingGhost: isTrailing
                    )
                    eval(ctxTile)
                    memPointLog("stage4 tile [\(ts),\(featTe)]/[\(hs),\(he)]/[\(ws),\(we)] 完成")
                    let tCount = ctxTile.shape[1], hCount = ctxTile.shape[2], wCount = ctxTile.shape[3]
                    // context tile 全局起点：非 origin tile 的首帧是重复 leading 帧 → 起点 = 2*ts - 1
                    let gt0 = isOrigin ? 0 : 2 * ts - 1
                    let gh0 = 2 * hs, gw0 = 2 * ws
                    let xT = MLXRandom.normal([B, 3, tCount, hCount * p, wCount * p]).asType(PrecisionPolicy.defaultMainDType)
                    let outTile: MLXArray
                    if numInferenceSteps == 1 {
                        outTile = forwardDiffusionStep(latentContext: ctxTile, x_t: xT, timestep: 1.0, weights)
                    } else {
                        outTile = denoise(latentContext: ctxTile, x_t: xT, steps: numInferenceSteps, weights)
                    }
                    eval(outTile)
                    // 融合权重：t/h/w 三轴线性 ramp（context 网格单位），再扩到像素格
                    let wt = ramp(count: tCount, startIndex: gt0, total: T5, rampLen: overlapFrames, axis: 1)   // (1,tCount,1,1,1)
                    let wh = ramp(count: hCount, startIndex: gh0, total: H5, rampLen: overlapSpatial, axis: 2)  // (1,1,hCount,1,1)
                    let ww = ramp(count: wCount, startIndex: gw0, total: W5, rampLen: overlapSpatial, axis: 3)  // (1,1,1,wCount,1)
                    let bwGrid = wt * wh * ww                                                          // (1,tCount,hCount,wCount,1)
                    // 空间格重复 p 次 → (1,tCount,hCount*p,wCount*p)
                    let bwPix = bwGrid
                        .reshaped([1, tCount, hCount, 1, wCount, 1])
                        * MLXArray.ones([1, 1, 1, p, 1, p])
                    let bwFull = bwPix.reshaped([1, tCount, hCount * p, wCount * p]).reshaped([1, 1, tCount, hCount * p, wCount * p])
                    // 画布边缘裁剪（trailing tile 的 ghost 延伸帧/像素超出画布，直接丢弃，输出帧数与整段一致）
                    let tMax = Swift.min(tCount, pxT - gt0)
                    let hMax = Swift.min(hCount * p, pxH - gh0 * p)
                    let wMax = Swift.min(wCount * p, pxW - gw0 * p)
                    let outClip = outTile[0..<B, 0..<3, 0..<tMax, 0..<hMax, 0..<wMax]
                    let maskClip = bwFull[0..<B, 0..<1, 0..<tMax, 0..<hMax, 0..<wMax]
                    // 扩到全画布（零填充）
                    let widths: [IntOrPair] = [
                        IntOrPair(0),
                        IntOrPair(0),
                        IntOrPair([gt0, pxT - gt0 - tMax]),
                        IntOrPair([gh0 * p, pxH - gh0 * p - hMax]),
                        IntOrPair([gw0 * p, pxW - gw0 * p - wMax]),
                    ]
                    acc = acc + padded(maskClip * outClip, widths: widths)
                    wacc = wacc + padded(maskClip, widths: widths)
                    eval(acc, wacc)
                    tileIndex += 1
                    if tileIndex % 4 == 0 {
                        memPointLog("tile \(tileIndex)/\(totalTiles) 融合")
                    }
                }
            }
        }
        let out = acc / wacc
        eval(out)
        memPointLog("eval(pixels) 后")
        return out
    }

    /// 单轴线性 ramp：tile 内第 i 格（对应全局 startIndex+i）的融合权重。
    /// 仅当该侧位于全局内部（有其他 tile 重叠）时才降权；贴画布边缘的一侧保持权重 1，
    /// 避免边缘像素只被单块覆盖时权重为 0 导致 0/0=NaN。
    /// - Parameter axis: 权重落在 5 维 shape 的哪个轴（1=t, 2=h, 3=w）
    private static func ramp(count: Int, startIndex: Int, total: Int, rampLen: Int, axis: Int) -> MLXArray {
        var vals: [Float] = []
        for i in 0..<count {
            var v: Float = 1
            if rampLen > 0 {
                let l = min(rampLen, count)
                if startIndex > 0 && i < l { v = min(v, Float(i) / Float(rampLen)) }
                if startIndex + count < total && i >= count - l {
                    v = min(v, Float(count - 1 - i) / Float(rampLen))
                }
            }
            vals.append(v)
        }
        var shape = [1, 1, 1, 1, 1]
        shape[axis] = count
        return MLXArray(vals).reshaped(shape)
    }
}

// MARK: - 公开入口（供 视频模型调用&回执.swift 替换 vaeDecodeTiled 调用）

/// LTX-2.5 扩散视频解码器入口。
/// - Parameters:
///   - weights: MLX.loadArrays 加载的 safetensors 权重字典（键含 `vae_diffusion_decoder.` 前缀）
///   - latentNDHWC: 已归一化 latent [B,T,H,W,128]（与卷积 VAE 输入相同）
///   - numInferenceSteps: 去噪步数，官方默认 1（x0 直接预测）
///   - useTiling: 是否分块解码（默认 true；整段 stage5 峰值超 48G 可用 Metal 内存会崩溃，
///     tiling 参数已对齐官方 halo 下限保证拼接质量）
/// - Returns: 像素 [B,3,8T-7,32H,32W]，值域 [-1,1]
func ltx2VideoDiffusionDecode(
    weights: [String: MLXArray],
    latentNDHWC: MLXArray,
    numInferenceSteps: Int = DDVDecoder.Config.defaultNumInferenceSteps,
    useTiling: Bool = true
) -> MLXArray {
    DDVDecoder.decode(
        weights: weights,
        latentNDHWC: latentNDHWC,
        numInferenceSteps: numInferenceSteps,
        useTiling: useTiling
    )
}
