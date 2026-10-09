//
//  模型通用算子-音频DSP.swift
//  无限画布 — 通用音频 DSP 算子（纯数学，无模型结构依赖，可跨模型复用）
//
//  ============================================================
//  作用：通用音频 DSP 算子（纯数学，无模型结构依赖，可跨模型复用）：
//    · hzToMelSlaney / melToHzSlaney：Hz ↔ Mel（Slaney 刻度）—— 使用方：LTX-2.5（AudioVaeDecoder 内部）
//    · buildEncMelFilterbank：Mel filterbank（面积归一，[64, 513]）—— 使用方：LTX-2.5（AudioVaeDecoder 内部）
//    · reflectPadCenterF32：中心反射 padding（[Float] 级）—— 使用方：LTX-2.5（AudioVaeDecoder 内部）
//    · melSpectrogramStereoV2：立体声 PCM → log-mel NCHW [1,2,T,64]（常量缓存版，原 V1 已合并去重）—— 使用方：LTX-2.5（AudioVaeDecoder）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 常量（编码器）

let COND_SAMPLE_RATE: Int = 16000
let ENC_N_FFT: Int = 1024
let ENC_HOP: Int = 160
let ENC_N_MELS: Int = 64
let ENC_N_FREQS: Int = ENC_N_FFT / 2 + 1

// MARK: - Mel 刻度换算（Slaney）

func hzToMelSlaney(_ f: Float) -> Float {
    if f < 1000.0 { return 3.0 * f / 200.0 }
    return 15.0 + 27.0 * logf(f / 1000.0) / logf(6.4)
}

func melToHzSlaney(_ m: Float) -> Float {
    if m < 15.0 { return 200.0 * m / 3.0 }
    return 1000.0 * expf((m - 15.0) * logf(6.4) / 27.0)
}

/// Slaney 尺度、面积归一 mel 滤波库 [64, 513]（f32）。
func buildEncMelFilterbank() -> MLXArray {
    let nyq = Float(COND_SAMPLE_RATE) / 2.0
    var allFreqs = [Float](repeating: 0, count: ENC_N_FREQS)
    for i in 0..<ENC_N_FREQS {
        allFreqs[i] = nyq * Float(i) / Float(ENC_N_FREQS - 1)
    }
    let mMin = hzToMelSlaney(0.0)
    let mMax = hzToMelSlaney(nyq)
    var fPts = [Float](repeating: 0, count: ENC_N_MELS + 2)
    for i in 0..<(ENC_N_MELS + 2) {
        let m = mMin + (mMax - mMin) * Float(i) / Float(ENC_N_MELS + 1)
        fPts[i] = melToHzSlaney(m)
    }
    var fb = [Float](repeating: 0, count: ENC_N_MELS * ENC_N_FREQS)
    for m in 0..<ENC_N_MELS {
        let lo = fPts[m]
        let ctr = fPts[m + 1]
        let hi = fPts[m + 2]
        let enorm = 2.0 / (hi - lo)
        for f in 0..<ENC_N_FREQS {
            let fr = allFreqs[f]
            let up = (fr - lo) / (ctr - lo)
            let down = (hi - fr) / (hi - ctr)
            var v = min(up, down)
            if v < 0 { v = 0 }
            fb[m * ENC_N_FREQS + f] = v * enorm
        }
    }
    return MLXArray(fb, [ENC_N_MELS, ENC_N_FREQS])
}

/// Reflect 填充（torchaudio center=True reflect，不重复边界）。
func reflectPadCenterF32(_ x: [Float], pad: Int) -> [Float] {
    var out = [Float](repeating: 0, count: x.count + 2 * pad)
    for i in 0..<pad { out[i] = x[pad - i] }
    for i in 0..<x.count { out[pad + i] = x[i] }
    for i in 0..<pad { out[pad + x.count + i] = x[x.count - 2 - i] }
    return out
}

// MARK: - V2：常量缓存

private let melFilterbankLock = NSLock()
private var melFilterbankCache: MLXArray?
private let melHannLock = NSLock()
private var melHannCache: MLXArray?

/// buildEncMelFilterbank 的常量缓存版本：滤波库只依赖采样率/FFT/Mel 数，
/// 全任务恒定，避免每个音频块重复 CPU 构造（64×513 双重循环）。
/// 旧版 buildEncMelFilterbank 保留不动。
func buildEncMelFilterbankCached() -> MLXArray {
    melFilterbankLock.lock()
    defer { melFilterbankLock.unlock() }
    if let c = melFilterbankCache { return c }
    let fb = buildEncMelFilterbank()
    fb.eval()  // 物化，避免首次使用把 CPU 构造链拖进主图
    melFilterbankCache = fb
    return fb
}

/// 16kHz periodic Hann 窗（[ENC_N_FFT]）常量缓存。
func encHannWindowCached() -> MLXArray {
    melHannLock.lock()
    defer { melHannLock.unlock() }
    if let c = melHannCache { return c }
    var wbuf = [Float](repeating: 0, count: ENC_N_FFT)
    for i in 0..<ENC_N_FFT {
        wbuf[i] = 0.5 * (1.0 - cosf(2.0 * Float.pi * Float(i) / Float(ENC_N_FFT)))
    }
    let w = MLXArray(wbuf, [ENC_N_FFT])
    w.eval()
    melHannCache = w
    return w
}

/// melSpectrogramStereoV2：与 melSpectrogramStereo 数学/输出完全等价，
/// 仅把 mel 滤波库与 Hann 窗改为常量缓存（见 buildEncMelFilterbankCached /
/// encHannWindowCached），避免每个音频块重复 CPU 构造。旧版保留不动。
func melSpectrogramStereoV2(pcm: [Float]) -> MLXArray {
    let nsm = pcm.count / 2          // 每声道样本数
    let pad = ENC_N_FFT / 2
    precondition(nsm > pad, "音频过短")
    let frames = (nsm + 2 * pad - ENC_N_FFT) / ENC_HOP + 1

    // 分声道 + reflect pad + 分帧 → [2*frames, n_fft]
    var fbuf = [Float](repeating: 0, count: 2 * frames * ENC_N_FFT)
    for c in 0..<2 {
        var chan = [Float](repeating: 0, count: nsm)
        for i in 0..<nsm { chan[i] = pcm[i * 2 + c] }
        let padded = reflectPadCenterF32(chan, pad: pad)
        for i in 0..<frames {
            let base = i * ENC_HOP
            let dst = (c * frames + i) * ENC_N_FFT
            for j in 0..<ENC_N_FFT {
                fbuf[dst + j] = padded[base + j]
            }
        }
    }
    var x = MLXArray(fbuf, [2 * frames, ENC_N_FFT])

    // periodic Hann（缓存常量）
    x = x * encHannWindowCached()

    // rfft → |spec|
    let spec = MLX.rfft(x, n: ENC_N_FFT, axis: 1)   // [2*frames, 513] complex
    let mag = MLX.abs(spec)

    // mel = mag @ basis.T → [2*frames, 64]；log(max(, 1e-5))
    let basis = buildEncMelFilterbankCached()        // [64, 513]
    var mel = MLX.matmul(mag, basis.transposed(axes: [1, 0]))
    mel = MLX.log(MLX.maximum(mel, MLXArray(1e-5)))

    // [2*frames, 64] → [1, 2, frames, 64]（声道 0 行在前）
    return mel.reshaped([1, 2, frames, ENC_N_MELS])
}
