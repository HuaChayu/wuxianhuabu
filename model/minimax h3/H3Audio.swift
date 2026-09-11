//
//  H3Audio.swift
//  无限画布
//
//  MiniMax-H3 音频 VAE 解码（audio_vae.safetensors，HiFiGAN/BigVGAN vocoder）。
//  依据官方 Zig 实现 mlx-serve-main/src/minimax_h3_audio.zig 的 decode() 保真移植：
//
//    z_norm [1, 32, 2, T]（32 通道、2 立体声、T 帧 @40Hz）
//      → [S, T, C]（S=2 立体声并入 BATCH 轴，vocoder 本体是 mono）
//      → 反归一化 z = z_norm × latents_std + latents_mean
//      → dec_in_proj（32→2048, k1）→ conv_pre（2048→1024, k7 pad3）
//      → 7 级 ConvTranspose1d 上采样（rates 5,5,2,2,2,2,2，kernels 9,9,4,4,4,4,4，
//        pad=(k-rate)/2），每级之后 3 个 AMPBlock 输出取平均（sum ÷ 3，非求和）
//      → activation_post（antiAliasSnakeBeta）→ conv_post（k7 pad3）→ clamp[-1,1]
//      → 波形 [S, L]（32kHz；每 latent 帧 800 采样，总 L = T × 800）
//
//  移植易错点（与 Zig 注释/长期记忆一致）：
//   · audio_vae.safetensors 为 torch 布局：Conv1d 权重 [O, I, k]、
//     ConvTranspose1d 权重 [I, O, k]；MLX 需要 [C_out, k, C_in]，读取后须 permute
//     （[O,I,k] → transposed(0,2,1)；[I,O,k] → transposed(1,2,0)）。
//   · decoder 用 antiAliasSnakeBeta（stored α/β 由 snakeBeta 内部指数化）；
//     与 encoder 的 Snake1d（α 原值、无反走样）是两个原语，不可混用。
//   · resblocks.{idx}.activations.{j}（6 个）为交错布局：
//     [::2]（j*2）喂 conv1、[1::2]（j*2+1）喂 conv2。
//   · 每 stage 的 3 个 AMPBlock 输出取平均；求和会把信号放大 3 倍并裁剪。
//

import Foundation
import MLX

public enum H3AudioConst {
    /// 音频 latent 通道数。
    public static let latentChannels: Int = 32
    /// 立体声声道数（stereo 独立解码，ride batch axis）。
    public static let stereoChannels: Int = 2
    /// 每帧采样数（40 fps × 800 samples）。
    public static let samplesPerFrame: Int = 800
    /// 采样率。
    public static let sampleRate: Int = 32000
}

/// MiniMax-H3 音频 VAE（解码端：BigVGAN vocoder）。
public final class H3AudioVAE {
    /// vocoder 上采样参数（同 minimax_h3_audio.zig：UP_RATES / UP_KERNELS /
    /// RES_KERNELS / RES_DILATIONS）。
    private static let upRates = [5, 5, 2, 2, 2, 2, 2]
    private static let upKernels = [9, 9, 4, 4, 4, 4, 4]
    private static let resKernels = [3, 7, 11]
    private static let resDilations = [1, 3, 5]

    /// 权重句柄（audio_vae.safetensors）。cacheEnabled=false，解码过程按需 mmap
    /// 读取，避免 ~600MB 权重全部驻留内存。
    private let weights: H3Weights
    /// 本实例的计算 dtype（主链 conv/激活统一走该精度；默认显式锁 fp16——2026-09-03
    /// e2e A/B 验证：fp16 vs fp32 同 latent 解码 SNR 44.1dB、maxAbs 0.006 无 NaN；
    /// env H3_PREC 仍可整体覆盖回退。自动回退闸门落地后可摘除显式锁改全局默认）。
    public let dtype: DType
    /// 反归一化统计量（f32 [32]）。
    public let latentsMean: MLXArray
    public let latentsStd: MLXArray

    /// 按 URL 打开权重并构建（解码全程持有句柄，不整包载入）。
    /// - Parameter dtype: 主链计算精度；nil 时走 audio 已验证特判 fp16
    ///   （resolve(componentLock: .fp16)：env H3_PREC 覆盖 > fp16 锁 > 芯片默认）。
    public static func load(url: URL, dtype: DType? = nil) throws -> H3AudioVAE {
        let w = try H3Weights(url: url)
        return try H3AudioVAE(weights: w, dtype: dtype)
    }

    /// 复用调用方已打开的权重句柄。
    public static func load(_ w: H3Weights, dtype: DType? = nil) throws -> H3AudioVAE {
        try H3AudioVAE(weights: w, dtype: dtype)
    }

    private init(weights: H3Weights, dtype: DType?) throws {
        self.weights = weights
        self.dtype = dtype ?? PrecisionPolicy.resolveDType(componentLock: .fp16)
        weights.cacheEnabled = false
        latentsMean = try Self.cast(weights, "latents_mean", to: .float32)
        latentsStd = try Self.cast(weights, "latents_std", to: .float32)
    }

    // MARK: - 权重读取

    private static func cast(_ w: H3Weights, _ key: String, to dt: DType) throws -> MLXArray {
        guard let raw = w.get(key) else { throw H3Error.missingWeight(key) }
        return raw.asType(dt)
    }
    private func tensor(_ key: String) throws -> MLXArray {
        try Self.cast(weights, key, to: dtype)
    }
    private func bias(_ key: String) throws -> MLXArray? {
        guard weights.contains(key) else { return nil }
        return try tensor(key)
    }
    /// Conv1d 权重 torch [O, I, k] → MLX [O, k, I]。
    private func conv1W(_ key: String) throws -> MLXArray {
        try tensor(key).transposed(0, 2, 1)
    }
    /// ConvTranspose1d 权重 torch [I, O, k] → MLX [O, k, I]。
    private func convTW(_ key: String) throws -> MLXArray {
        try tensor(key).transposed(1, 2, 0)
    }
    /// decoder 反走样 SnakeBeta：act.alpha / act.beta + 上/下采样反走样核。
    private func actAt(_ base: String, _ x: MLXArray) throws -> MLXArray {
        let a = try tensor(base + ".act.alpha")
        let b = try tensor(base + ".act.beta")
        let uf = try tensor(base + ".upsample.filter")
        let df = try tensor(base + ".downsample.lowpass.filter")
        return antiAliasSnakeBeta(x, alpha: a, beta: b, upFilt: uf, downFilt: df)
    }

    /// 单个 AMPBlock（kernel 为 3/7/11 之一）：3 子层，层内
    /// act(antiAliasSnakeBeta) → conv1(dilation=resDilations[j]) → act → conv2(dil=1)
    /// → 残差。activations 为交错布局：j*2 → conv1、j*2+1 → conv2。
    private func ampBlock(_ idx: Int, kernel: Int, _ xIn: MLXArray) throws -> MLXArray {
        var x = xIn
        for j in 0..<3 {
            let dil = Self.resDilations[j]
            let base = "decoder.resblocks.\(idx)"
            let a1 = try actAt(base + ".activations.\(j * 2)", x)
            let c1 = conv1dNLC(a1,
                               weight: try conv1W(base + ".convs1.\(j).weight"),
                               bias: try bias(base + ".convs1.\(j).bias"),
                               stride: 1, padding: dil * (kernel - 1) / 2,
                               dilation: dil, groups: 1)
            let a2 = try actAt(base + ".activations.\(j * 2 + 1)", c1)
            let c2 = conv1dNLC(a2,
                               weight: try conv1W(base + ".convs2.\(j).weight"),
                               bias: try bias(base + ".convs2.\(j).bias"),
                               stride: 1, padding: (kernel - 1) / 2,
                               dilation: 1, groups: 1)
            x = x + c2
        }
        return x
    }

    // MARK: - 解码

    /// 反归一化：音频 latent [1, 32, 2, T] → [S, T, 32] f32（立体声进 batch，
    /// 通道后置，与 Zig decode 的 NLC 布局对齐）。
    public func denormalize(_ z: MLXArray) throws -> MLXArray {
        let zf = z.asType(.float32)
        let stereo = z.shape[2]
        // [1, C, S, T] -> [S, T, C]
        let perm = zf.transposed(0, 2, 3, 1)
            .reshaped([stereo, z.shape[3], H3AudioConst.latentChannels])
            .contiguous()
        let lm = latentsMean.reshaped(1, 1, H3AudioConst.latentChannels)
        let ls = latentsStd.reshaped(1, 1, H3AudioConst.latentChannels)
        return perm * ls + lm
    }

    /// vocoder 前向：归一化 latent [1, 32, 2, T]（f32/bf16/fp16 均可）→ 波形 [S, L] f32，
    /// 取值 [-1, 1]，L = T × 800 @32kHz。
    /// 主链按 self.dtype 计算：反归一化在 f32（统计量小张量、精度不损），随后
    /// cast 到 dtype 进 conv/激活链（权重 mmap 时已 cast 到同 dtype，无跨 dtype 提升），
    /// 出口统一回落 f32（波形消费端逐元素，成本可忽略）。
    public func decode(_ zNorm: MLXArray) throws -> MLXArray {
        let stereo = zNorm.shape[2]
        guard zNorm.shape.count == 4,
              zNorm.shape[1] == H3AudioConst.latentChannels,
              stereo == H3AudioConst.stereoChannels else {
            throw H3Error.badFile("audio latent 形状不符：\(zNorm.shape)，期望 [1,32,2,T]")
        }

        // 反归一化 [S,T,32]（f32 岛）→ cast 到 dtype 进主链
        var x = try denormalize(zNorm).asType(dtype)

        // dec_in_proj（32→2048, k1）→ conv_pre（2048→1024, k7 pad3）
        x = conv1dNLC(x, weight: try conv1W("dec_in_proj.weight"),
                      bias: try bias("dec_in_proj.bias"),
                      stride: 1, padding: 0, dilation: 1, groups: 1)
        x = conv1dNLC(x, weight: try conv1W("decoder.conv_pre.weight"),
                      bias: try bias("decoder.conv_pre.bias"),
                      stride: 1, padding: 3, dilation: 1, groups: 1)
        MLX.eval(x)

        // 7 级上采样：ConvTranspose1d（pad=(k-rate)/2）→ 3×AMPBlock 平均
        for i in 0..<Self.upRates.count {
            let rate = Self.upRates[i]
            let k = Self.upKernels[i]
            x = MLX.convTransposed1d(x, try convTW("decoder.ups.\(i).0.weight"),
                                     stride: rate, padding: (k - rate) / 2,
                                     outputPadding: 0, groups: 1)
            if let b = try bias("decoder.ups.\(i).0.bias") { x = x + b }
            var acc: MLXArray? = nil
            for (kidx, rk) in Self.resKernels.enumerated() {
                let rb = try ampBlock(i * 3 + kidx, kernel: rk, x)
                acc = (acc == nil) ? rb : (acc! + rb)
            }
            // 平均：sum ÷ RES_KERNELS.len（求和会放大 3 倍并裁剪）
            x = acc! / 3.0
            MLX.eval(x)
        }

        // activation_post（antiAliasSnakeBeta）→ conv_post（k7 pad3）→ clamp[-1,1]
        x = try actAt("decoder.activation_post", x)
        x = conv1dNLC(x, weight: try conv1W("decoder.conv_post.weight"),
                      bias: try bias("decoder.conv_post.bias"),
                      stride: 1, padding: 3, dilation: 1, groups: 1)
        x = MLX.maximum(x, MLXArray.scalar(-1.0, like: x))
        x = MLX.minimum(x, MLXArray.scalar(1.0, like: x))
        // [S, L, 1] → [S, L]，出口回落 f32
        x = x.reshaped([stereo, x.shape[1]]).asType(.float32)
        MLX.eval(x)
        return x
    }

    /// 采样输出 audio rows [2T, 32]（ch-major：前 T 行 ch0、后 T 行 ch1）
    /// → latent [1, 32, 2, T]。对齐 Zig audioRowsToLatent：
    /// reshape [2, T, 32] → transpose(2,0,1) → reshape [1, 32, 2, T]。
    public static func audioRowsToLatent(_ rows: MLXArray, audioT: Int) throws -> MLXArray {
        let ch = H3AudioConst.latentChannels
        guard rows.dim(0) == audioT * 2, rows.dim(1) == ch else {
            throw H3Error.badFile("audio rows 形状不符：\(rows.shape)，期望 [\(audioT * 2), \(ch)]")
        }
        let rf = rows.asType(.float32).contiguous()
        let a3 = rf.reshaped([H3AudioConst.stereoChannels, audioT, ch])
        let perm = a3.transposed(2, 0, 1).contiguous()   // [ch, 2, audioT]
        return perm.reshaped([1, ch, H3AudioConst.stereoChannels, audioT])
    }
}
