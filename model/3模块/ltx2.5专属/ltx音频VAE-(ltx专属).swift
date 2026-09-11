//
//  AudioVaeDecoder.swift
//  ltx-test — LTX-2.5 音频解码（audio VAE + BigVGAN vocoder）
//
//  ============================================================
//  作用：把去噪后的 audio latent [1, Na, 128] 解码为 16kHz 立体声
//  PCM 波形（interleaved f32，[-1,1]），供 mp4 混流加音轨。
//  参考：mlx-serve src/ltx_audio.zig ——
//        audioVaeDecode(443)：2D causal conv VAE，latent → mel [1,2,T,64] NCHW
//        vocode(854)：BigVGAN v2（conv_pre + 6×ConvTranspose1d + 18 AMPBlocks
//                     平均 + act_post + conv_post），mel → waveform [1,L,2] NLC
//  权重：audio_vae.safetensors（audio_vae.*，102 张量）+ vocoder.safetensors
//        （vocoder.*），MLX 布局 [C_out, k, C_in] / [C_out, kh, kw, C_in]。
//  ============================================================

import Foundation
import MLX
import MLXNN
/// 预激活 ResnetBlock：h = conv2(silu(pn(conv1(silu(pn(x))))))；shortcut = nin_shortcut(x) 或恒等。
func resnetBlock2d(_ weights: [String: MLXArray], base: String, _ x: MLXArray) -> MLXArray {
    let pn1 = pixelNormFast(x, eps: 1e-6)
    let c1 = causalConv2d(weights, base: base + ".conv1.conv", silu(pn1))
    let pn2 = pixelNormFast(c1, eps: 1e-6)
    let c2 = causalConv2d(weights, base: base + ".conv2.conv", silu(pn2))
    let shortcut: MLXArray
    if weights[base + ".nin_shortcut.conv.weight"] != nil {
        shortcut = causalConv2d(weights, base: base + ".nin_shortcut.conv", x)
    } else {
        shortcut = x
    }
    return shortcut + c2
}

/// Nearest ×2 upsample（time & freq）→ causal 3×3 conv → drop first time frame。
func vaeUpsample(_ weights: [String: MLXArray], base: String, _ x: MLXArray) -> MLXArray {
    var up = MLX.repeated(x, count: 2, axis: 1)
    up = MLX.repeated(up, count: 2, axis: 2)
    let upc = up.contiguous()
    let conv = causalConv2d(weights, base: base + ".conv.conv", upc)
    let t = conv.shape[1]
    return conv[1 ..< t, axis: 1].contiguous()
}

// MARK: - Audio VAE 解码器

/// 解码 DiT audio latent [1, Na, 128] → 立体声 mel [1, 2, T, 64]（NCHW）。
func audioVaeDecode(weights: [String: MLXArray], latent: MLXArray) -> MLXArray {
    let na = latent.shape[1]

    // per_channel_statistics：q4 仓库用 _mean_of_means/_std_of_means，distilled 用连字符
    guard let meanRaw = weights["audio_vae.per_channel_statistics._mean_of_means"]
            ?? weights["audio_vae.per_channel_statistics.mean-of-means"],
          let stdRaw = weights["audio_vae.per_channel_statistics._std_of_means"]
            ?? weights["audio_vae.per_channel_statistics.std-of-means"] else {
        fatalError("缺 audio_vae.per_channel_statistics")
    }
    let mean = meanRaw.reshaped([1, 1, 128])
    let std = stdRaw.reshaped([1, 1, 128])

    // denormalize：x = latent * std + mean
    var x = latent * std + mean
    eval(x)

    // unpatchify [1,Na,128] → NHWC [1, Na(time), 16(freq), 8(ch)]（c outer × f inner）
    x = x.reshaped([1, na, 8, 16])          // b t c f
    x = x.transposed(axes: [0, 1, 3, 2])    // b t f c
    x = x.contiguous()
    eval(x)

    // conv_in 8→512
    x = causalConv2d(weights, base: "audio_vae.decoder.conv_in.conv", x)
    eval(x)

    // mid：block_1 / block_2（无 attention）
    for blk in ["block_1", "block_2"] {
        x = resnetBlock2d(weights, base: "audio_vae.decoder.mid." + blk, x)
        eval(x)
    }

    // up path：level 2 → 1 → 0；每级 3 个 resblock；level != 0 时 upsample
    for level in stride(from: 2, through: 0, by: -1) {
        for blk in 0..<3 {
            x = resnetBlock2d(weights, base: "audio_vae.decoder.up.\(level).block.\(blk)", x)
        }
        eval(x)
        if level != 0 {
            x = vaeUpsample(weights, base: "audio_vae.decoder.up.\(level).upsample", x)
            eval(x)
        }
    }

    // norm_out + silu + conv_out（→2）
    x = pixelNormFast(x, eps: 1e-6)
    x = silu(x)
    x = causalConv2d(weights, base: "audio_vae.decoder.conv_out.conv", x)

    // NHWC [1, T, 64, 2] → NCHW [1, 2, T, 64]
    x = x.transposed(axes: [0, 3, 1, 2]).contiguous()
    eval(x)
    return x
}
/// 按 act 基键取反走样 snakeBeta（act.alpha / act.beta / upsample.filter / downsample.lowpass.filter）。
func actByKey(_ weights: [String: MLXArray], base: String, _ x: MLXArray) -> MLXArray {
    guard let alpha = weights[base + ".act.alpha"],
          let beta = weights[base + ".act.beta"],
          let uf = weights[base + ".upsample.filter"],
          let df = weights[base + ".downsample.lowpass.filter"] else {
        fatalError("缺激活权重：\(base)")
    }
    return antiAliasSnakeBeta(x, alpha: alpha, beta: beta, upFilt: uf, downFilt: df)
}

let vocUpRates = [5, 2, 2, 2, 2, 2]
let vocKernels = [3, 7, 11]
let vocDils = [1, 3, 5]

/// 一个 AMPBlock1：3 子层（dil 1,3,5），每层 act→conv1(dil)→act→conv2(dil=1)→残差。
func ampBlock(_ weights: [String: MLXArray], idx: Int, kernel: Int, _ xIn: MLXArray) -> MLXArray {
    var x = xIn
    for j in 0..<3 {
        let base = "vocoder.resblocks.\(idx)"
        let dil = vocDils[j]
        let a1 = actByKey(weights, base: base + ".acts1.\(j)", x)
        let pad1 = dil * (kernel - 1) / 2
        let c1 = conv1dNLC(a1, weight: weights[base + ".convs1.\(j).weight"]!,
                           bias: weights[base + ".convs1.\(j).bias"],
                           stride: 1, padding: pad1, dilation: dil, groups: 1)
        let a2 = actByKey(weights, base: base + ".acts2.\(j)", c1)
        let pad2 = (kernel - 1) / 2
        let c2 = conv1dNLC(a2, weight: weights[base + ".convs2.\(j).weight"]!,
                           bias: weights[base + ".convs2.\(j).bias"],
                           stride: 1, padding: pad2, dilation: 1, groups: 1)
        x = x + c2
    }
    return x
}

/// 立体声 mel [1, 2, T, 64]（NCHW）→ 波形 NLC [1, L, 2]。
func vocode(weights: [String: MLXArray], melNCHW: MLXArray) -> MLXArray {
    let T = melNCHW.shape[2]

    // NCHW [1,2,T,64] → NLC [1, T, 128]（channel = stereo*64 + freq）
    var x = melNCHW.transposed(axes: [0, 2, 1, 3]).contiguous()
    x = x.reshaped([1, T, 128])

    // conv_pre（128→1536，k7 pad3）
    x = conv1dNLC(x, weight: weights["vocoder.conv_pre.weight"]!,
                  bias: weights["vocoder.conv_pre.bias"],
                  stride: 1, padding: 3, dilation: 1, groups: 1)

    // upsample stages：6×ConvTranspose1d（rates 5,2,2,2,2,2 = ×160）+ 每级 3 AMPBlocks 平均
    for i in 0..<vocUpRates.count {
        let base = "vocoder.ups.\(i)"
        guard let w = weights[base + ".weight"] else { fatalError("缺键：\(base).weight") }
        let b = weights[base + ".bias"]
        let kc = w.shape[1]
        let stride = vocUpRates[i]
        let pad = (kc - stride) / 2
        x = MLX.convTransposed1d(x, w, stride: stride, padding: pad, outputPadding: 0, groups: 1)
        if let b { x = x + b }

        var acc: MLXArray? = nil
        for kidx in 0..<3 {
            let rb = ampBlock(weights, idx: i * 3 + kidx, kernel: vocKernels[kidx], x)
            acc = (acc == nil) ? rb : (acc! + rb)
        }
        x = acc! / 3.0
        eval(x)
    }

    // act_post + conv_post + clamp
    x = actByKey(weights, base: "vocoder.act_post", x)
    x = conv1dNLC(x, weight: weights["vocoder.conv_post.weight"]!,
                  bias: weights["vocoder.conv_post.bias"],
                  stride: 1, padding: 3, dilation: 1, groups: 1)
    x = MLX.maximum(x, MLXArray(-1.0))
    x = MLX.minimum(x, MLXArray(1.0))
    eval(x)
    return x
}

// MARK: - 完整音频解码

/// 完整解码：DiT latent [1, Na, 128] → interleaved 立体声 PCM f32 [-1,1] @16kHz。
/// 返回 (pcm, frames, channels=2, sampleRate=16000)。
func decodeAudio(weights: [String: MLXArray], latent: MLXArray) -> (pcm: [Float], frames: Int) {
    let mel = audioVaeDecode(weights: weights, latent: latent)
    let wav = vocode(weights: weights, melNCHW: mel)   // [1, L, 2]
    let frames = wav.shape[1]
    let pcm = wav.asArray(Float.self)
    return (pcm, frames)
}

// MARK: - WAV 写出（16kHz stereo f32 interleaved）

func writeWav(_ pcm: [Float], sampleRate: Int = 16000, channels: Int = 2, to path: String) {
    let dataSize = pcm.count * 4
    var header = Data()
    func appendStr(_ s: String) { header.append(Data(s.utf8)) }
    func appendU32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { header.append(Data($0)) } }
    func appendU16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { header.append(Data($0)) } }
    appendStr("RIFF")
    appendU32(UInt32(36 + dataSize))
    appendStr("WAVE")
    appendStr("fmt ")
    appendU32(16)
    appendU16(3)                       // IEEE float
    appendU16(UInt16(channels))
    appendU32(UInt32(sampleRate))
    appendU32(UInt32(sampleRate * channels * 4))
    appendU16(UInt16(channels * 4))
    appendU16(32)
    appendStr("data")
    appendU32(UInt32(dataSize))
    var body = Data(capacity: dataSize)
    pcm.withUnsafeBufferPointer { buf in
        buf.withMemoryRebound(to: UInt8.self) { bytes in
            body.append(contentsOf: bytes)
        }
    }
    try! (header + body).write(to: URL(fileURLWithPath: path))
}

// MARK: - 音频条件编码（mel + Audio VAE Encoder，对齐 ltx_audio.zig:530-810）

/// 16kHz 输入；mel 参数同官方（ENC_N_FFT=1024 / HOP=160 / MELS=64 / FREQS=513，
/// 常量定义见 通用算子&函数/模型通用算子-音频DSP.swift）。
/// Encoder downsample：causal pad（time lo=2 hi=0，freq lo=0 hi=1）+ stride-2 conv。
/// 键为直接 Conv2d（`down.{l}.downsample.conv.{weight,bias}`，非 .conv.conv 嵌套）。
func encDownsample(_ weights: [String: MLXArray], base: String, _ x: MLXArray) -> MLXArray {
    guard let w = weights[base + ".weight"] else { fatalError("缺键：\(base).weight") }
    let b = weights[base + ".bias"]
    let padded = MLX.padded(x, widths: [
        IntOrPair((0, 0)),
        IntOrPair((2, 0)),
        IntOrPair((0, 1)),
        IntOrPair((0, 0)),
    ], mode: .constant)
    return conv2dNHWC(padded.contiguous(), weight: w, bias: b, stride: (2, 2), padding: (0, 0))
}

/// 立体声 log-mel [1,2,T',64]（NCHW）→ 归一化 DiT 音频 token [1, T, 128]。
func audioVaeEncode(weights: [String: MLXArray], melNCHW: MLXArray) -> MLXArray {
    // NCHW → NHWC [1, T', 64, 2]
    var x = melNCHW.transposed(axes: [0, 2, 3, 1]).contiguous()

    // conv_in 2→128
    x = causalConv2d(weights, base: "audio_vae.encoder.conv_in.conv", x)
    eval(x)

    // down levels（128→256→512）：每级 2 个 resblock；level 0/1 有 downsample
    for level in 0..<3 {
        for blk in 0..<2 {
            x = resnetBlock2d(weights, base: "audio_vae.encoder.down.\(level).block.\(blk)", x)
        }
        if level != 2 {
            x = encDownsample(weights, base: "audio_vae.encoder.down.\(level).downsample.conv", x)
        }
        eval(x)
    }

    // mid：block_1 / block_2
    for blk in ["block_1", "block_2"] {
        x = resnetBlock2d(weights, base: "audio_vae.encoder.mid." + blk, x)
        eval(x)
    }

    // norm_out + silu + conv_out（→16 double_z）
    x = pixelNormFast(x, eps: 1e-6)
    x = silu(x)
    x = causalConv2d(weights, base: "audio_vae.encoder.conv_out.conv", x)

    // 取前 8 个 MEAN 通道 → (b,t,f,c)→(b,t,c,f) → [1,T,128]
    let c8 = x.shape[3] / 2
    x = x[0..., 0..., 0..., 0 ..< c8].contiguous()
    x = x.transposed(axes: [0, 1, 3, 2]).contiguous()
    let tDim = x.shape[1]
    x = x.reshaped([1, tDim, 128])

    // per_channel_statistics 归一化：(x - mean) / (std + 1e-8)
    guard let meanRaw = weights["audio_vae.per_channel_statistics._mean_of_means"]
            ?? weights["audio_vae.per_channel_statistics.mean-of-means"],
          let stdRaw = weights["audio_vae.per_channel_statistics._std_of_means"]
            ?? weights["audio_vae.per_channel_statistics.std-of-means"] else {
        fatalError("缺 audio_vae.per_channel_statistics")
    }
    let mean = meanRaw.reshaped([1, 1, 128])
    let std = stdRaw.reshaped([1, 1, 128])
    x = (x - mean) / (std + MLXArray(1e-8))
    eval(x)
    return x
}

/// 一次性条件编码：interleaved 立体声 16kHz PCM → DiT 音频 token [1, min(T, maxTokens), 128]（bf16）。
/// maxTokens 为视频时长 token 预算（computeAudioTokenCount）；官方只截断、不补零。
func encodeAudioCond(weights: [String: MLXArray], pcm: [Float], maxTokens: Int) -> MLXArray {
    let mel = melSpectrogramStereoV2(pcm: pcm)
    var tokens = audioVaeEncode(weights: weights, melNCHW: mel)
    let tDim = tokens.shape[1]
    if tDim > maxTokens {
        tokens = tokens[0..., 0 ..< maxTokens, 0...].contiguous()
    }
    let bf = tokens.asType(PrecisionPolicy.defaultMainDType)
    eval(bf)
    return bf
}
