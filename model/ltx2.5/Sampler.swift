//
//  Sampler.swift
//  ltx-test — LTX-2.5 流匹配采样器（对齐 mlx-serve src/ltx_video.zig）
//
//  ============================================================
//  作用：纯 t2v 的 guided Euler 采样循环：
//    dynamicShiftSchedule → 噪声 latent → 逐 sigma 步：
//      cond 前向（+ neg 前向做 CFG）→ x0 = x - vel*sigma
//      → CFG + norm-preserving rescale → Euler 步 → 下一 sigma
//  参考：ltx_video.zig 的 dynamicShiftSchedule / eulerStep /
//        guiderCalculate / ditSampleCfg（仅 CFG 子集，无 STG/modality/I2V）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 采样参数

struct SamplerConfig {
    var numSteps: Int = 50
    var cfgV: Float = 3.0         // 视频 CFG（对齐参考 vp.cfg=3.0；蒸馏单阶段传 1.0=无引导）
    var cfgA: Float = 7.0         // 音频 CFG（对齐参考 ap.cfg=7.0，音频引导远强于视频）
    var cfgRescaleV: Float = 0.7  // 视频 norm-preserving rescale（参考 vp.rescale）
    var cfgRescaleA: Float = 0.7  // 音频 norm-preserving rescale（参考 ap.rescale）
    var baseShift: Double = 0.95
    var maxShift: Double = 2.05
    var baseTokens: Double = 1024
    var maxTokens: Double = 4096
    var stretch: Bool = true
    var terminal: Double = 0.1
    var sigmas: [Float]? = nil     // 非 nil → 固定 sigma 表（蒸馏模型专用，跳过 dynamicShiftSchedule）
    var ancestral: Bool = false    // true → ancestral Euler 步进（蒸馏 stage1 SDE 噪声注入）
    var seed: UInt64 = 42
    var useCompile: Bool = true    // MLX.compile 编译 DiT 单步前向（大幅减少 host 调度开销）
    var teaCache: Bool = true              // TeaCache 特征缓存（相邻步 sigma 差小时跳过 DiT 前向）
    var teaCacheThreshold: Float = 0.02    // TeaCache 触发阈值（蒸馏表前 4 步间隔 0.00625，默认保守）
}
// MARK: - 编译前向（MLX.compile）

/// 把 DiT 单步前向编译为融合内核（shape 每步固定，完美契合）。
/// 捕获 dit 权重 + 预构造 rope + 位置数组（常量），运行时只换 latent/timestep/text 输入。
typealias CompiledDitForward = @Sendable (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray)

func makeCompiledDitForward(
    dit: LTXVideoDiT, rope: BlockRope, videoPos: [Float], audioPos: [Float],
    audioTimesteps: MLXArray? = nil,
    condMask: [Float]? = nil,
    maskMLX: MLXArray? = nil,
    maskMLXInv: MLXArray? = nil
) -> CompiledDitForward {
    // I2V：per-token timestep = mask*sigma 与 keyframes gate 均为纯张量运算，
    // 常量（condMask 标志 + 预构造 mask/maskMLXInv）由闭包捕获，可整体编译。
    MLX.compile { (vx: MLXArray, ax: MLXArray, ts: MLXArray, vText: MLXArray, aText: MLXArray) -> (MLXArray, MLXArray) in
        let out = dit(
            videoLatent: vx, audioLatent: ax, timesteps: ts,
            audioTimesteps: audioTimesteps,
            videoText: vText, audioText: aText,
            videoPos: videoPos, audioPos: audioPos,
            rope: rope,
            condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv)
        return (out.video, out.audio)
    }
}

// MARK: - 编译图持久缓存（通用组件实例）

/// 编译图常驻复用：同一 DiT 实例 + 同一 shape 时直接复用上次编译产物，
/// 采样结束后不释放编译图，根治 "Compiled 大图一次性递归析构" 造成的 app 假死。
/// 析构只发生在：换 shape（重新编译时旧图释放）或 app 退出。
/// 通用 CompiledForwardCache<LTXVideoDiT, CompiledDitForward> 见 优化方法.swift，
/// 本实例由 Sampler 采样与 调试中心 / 视频模型调用&回执 的卸载动作共用。
let ltxCompiledForward = CompiledForwardCache<LTXVideoDiT, CompiledDitForward>()

// MARK: - 单步 x0 预测（含可选 CFG 负向）

/// 一次 sigma 步：cond 前向；若提供 neg 条件再做 neg 前向做 CFG。
/// 返回 x0（bf16，与参考 ditX0 一致）。forward 非 nil 时 cond/neg 走编译内核。
/// - frozenAudio: 非 nil 时启用音频条件（frozen_a 语义）：音频流以 timestep 0 运行、
///   音频 x0 pin 为 frozen latent（自身不重建，仅通过注意力条件化视频），
///   视频流仍正常去噪。
/// - condMask/cleanV: 非 nil 时启用 I2V 首帧条件（对齐 ditX0Guided 的 cond_mask/clean_v）：
///   视频前向走 per-token timestep（mask*sigma），采样后 applyDenoiseMask 把干净首帧
///   latent 钉入（x0 = x0*mask + clean*(1-mask)）。
func ditX0Guided(
    dit: LTXVideoDiT,
    forward: CompiledDitForward?,
    vx: MLXArray,
    ax: MLXArray,
    sigma: Float,
    condV: MLXArray,
    condA: MLXArray,
    negV: MLXArray?,
    negA: MLXArray?,
    frozenAudio: MLXArray?,
    condMask: [Float]?,
    cleanV: MLXArray?,
    maskTensor: MLXArray? = nil,   // I2V：预构造 mask [1,T,H,W,1] bf16（采样循环外构造一次，避免每步 CPU→GPU 拷贝）
    maskMLX: MLXArray? = nil,      // I2V：预构造 mask [Nv] f32（DiT per-token timestep 复用）
    maskMLXInv: MLXArray? = nil,   // I2V：预构造 1-mask [Nv] f32（keyframes_abs_pos gate 复用）
    videoPos: [Float],
    audioPos: [Float],
    rope: BlockRope,
    cfg: SamplerConfig
) -> (v: MLXArray, a: MLXArray) {
    // 峰值保底（全局公共能力）：本函数即「真实重前向」（cond+neg 双前向）入口，
    // 进前向计算前做缓存池平抑。projected 由 latent 网格还原全量像素尺寸/帧数，
    // 复用 safetyBuffer 语义（越大越保守）；TeaCache 命中步直出自缓存不经本函数，天然不接入。
    let qf = vx.shape[1]
    let qh = vx.shape[2]
    let qw = vx.shape[3]
    _ = SystemMemory.ensureLoose(projected: Int64(MemoryPolicy.safetyBuffer(
        for: .video, width: qw * 32, height: qh * 32, frames: qf * 8)))
    let ts = MLXArray([sigma])
    // 音频条件模式下音频流走独立时间步 0（audio_sigma=0.0，clean/frozen）
    let audioTs = frozenAudio == nil ? nil : MLXArray([Float(0.0)])

    // ── cond 前向（不单独 eval：与 euler 步合并为一次物化）──
    var vx0: MLXArray
    var ax0: MLXArray
    if let forward {
        let (cv, ca) = forward(vx, ax, ts, condV, condA)
        vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(.bfloat16)   // cv [B,Nv,C] → 5D 广播
        ax0 = (ax - ca * sigma).asType(.bfloat16)
    } else {
        let (cv, ca) = dit(
            videoLatent: vx, audioLatent: ax, timesteps: ts,
            audioTimesteps: audioTs,
            videoText: condV, audioText: condA,
            videoPos: videoPos, audioPos: audioPos, rope: rope,
            condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv)
        vx0 = (vx - cv.reshaped(vx.shape) * sigma).asType(.bfloat16)
        ax0 = (ax - ca * sigma).asType(.bfloat16)
    }

    // ── neg 前向（CFG，视频/音频分模态独立）──
    // 纯 t2v 无音频条件时传 negA=nil：音频分支不做负向前向，无条件直出（省一次前向）
    if let nv = negV {
        var uvx0: MLXArray
        if let forward {
            let (uv, _) = forward(vx, ax, ts, nv, condA)
            uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(.bfloat16)
        } else {
            let (uv, _) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: nv, audioText: condA,
                videoPos: videoPos, audioPos: audioPos, rope: rope,
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv)
            uvx0 = (vx - uv.reshaped(vx.shape) * sigma).asType(.bfloat16)
        }
        vx0 = guiderCombine(cond: vx0, neg: uvx0, cfg: cfg.cfgV, rescale: cfg.cfgRescaleV)
    } else {
        vx0 = guiderCombine(cond: vx0, neg: nil, cfg: cfg.cfgV, rescale: cfg.cfgRescaleV)
    }
    // 音频 CFG：frozen 条件时音频不重建，x0 直接 pin 为 clean latent，跳过 CFG 与 rescale
    if let frozen = frozenAudio {
        ax0 = frozen
    } else if let na = negA {
        var uax0: MLXArray
        if let forward {
            let (_, ua) = forward(vx, ax, ts, condV, na)
            uax0 = (ax - ua * sigma).asType(.bfloat16)
        } else {
            let (_, ua) = dit(
                videoLatent: vx, audioLatent: ax, timesteps: ts,
                audioTimesteps: audioTs,
                videoText: condV, audioText: na,
                videoPos: videoPos, audioPos: audioPos, rope: rope)
            uax0 = (ax - ua * sigma).asType(.bfloat16)
        }
        ax0 = guiderCombine(cond: ax0, neg: uax0, cfg: cfg.cfgA, rescale: cfg.cfgRescaleA)
    } else {
        ax0 = guiderCombine(cond: ax0, neg: nil, cfg: cfg.cfgA, rescale: cfg.cfgRescaleA)
    }

    // ── I2V：把干净首帧 latent 钉入（applyDenoiseMask：x0 = x0*mask + clean*(1-mask)）──
    if let mask = condMask, let clean = cleanV {
        // vx0 [1,T,H,W,128]；mask [Nv] → [1,T,H,W,1] 广播（maskTensor 采样循环外已预构造）
        let m = maskTensor ?? MLXArray(mask).reshaped([1, vx.shape[1], vx.shape[2], vx.shape[3], 1]).asType(.bfloat16)
        vx0 = vx0 * m + clean * (1 - m)
    }
    return (vx0, ax0)
}

// MARK: - 完整采样循环（对齐 ditSampleCfg，纯 t2v / 音频条件）

/// 输入噪声 latent → 输出去噪后的 x0 latent（bf16）。
/// - frozenAudio: 非 nil 时启用音频条件（frozen_a 语义）：噪声音频直接用 clean
///   frozen token，每步音频 x0 pin 回 frozen（只 euler 收敛不重建），视频受音频条件化。
/// - initVideo/cleanV/condMask: 非 nil 时启用 I2V 首帧条件：initVideo 为已替换首帧的
///   初始 latent（ref_tokens + noise[HW:]），cleanV 为干净首帧 latent（ref_tokens + zeros），
///   condMask 前 HW 个 0（干净）其余 1（生成）；每步 applyDenoiseMask 钉入首帧。
func sampleLatents(
    dit: LTXVideoDiT,
    noiseV: MLXArray,       // [1, T, H, W, 128]
    noiseA: MLXArray,       // [1, Na, 128]（frozenAudio 非 nil 时即 frozen token 本身）
    condV: MLXArray,        // [1, 256, 4096]
    condA: MLXArray,        // [1, 256, 2048]
    negV: MLXArray?,        // nil → 跳过 CFG（纯 cond 前向）
    negA: MLXArray?,
    frozenAudio: MLXArray?, // nil → 纯 t2v；非 nil → 音频条件（frozen_a）
    initVideo: MLXArray? = nil, // I2V：初始 latent（含首帧替换）；nil → noiseV
    cleanV: MLXArray? = nil,    // I2V：干净首帧 latent；nil → 无
    condMask: [Float]? = nil,   // I2V：前 HW 个 0（干净）其余 1；nil → 纯 t2v
    videoPos: [Float],      // [Nv*3]
    audioPos: [Float],      // [Na]
    config: SamplerConfig,
    isCancelled: @escaping () -> Bool = { false }   // 每步前检查，true → 提前结束采样（队列取消用）
) -> (video: MLXArray, audio: MLXArray) {
    let Nv = videoPos.count / 3
    let isI2V = condMask != nil
    let sigmas = config.sigmas ?? dynamicShiftSchedule(
        numSteps: config.numSteps, numTokens: Nv,
        baseShift: config.baseShift, maxShift: config.maxShift,
        baseTokens: config.baseTokens, maxTokens: config.maxTokens,
        stretch: config.stretch, terminal: config.terminal)
    print("采样计划：\(sigmas.count - 1) 步，Nv=\(Nv)，sigma[0]=\(sigmas[0])，调度=\(config.sigmas == nil ? "DynamicShift" : "蒸馏固定表")，步进=\(config.ancestral ? "ancestral-SDE" : "确定性Euler")，音频条件=\(frozenAudio != nil)，I2V=\(isI2V)")

    // RoPE 预构造一次（只依赖位置），48 层复用
    let rope = buildBlockRope(config: dit.config, videoPos: videoPos, audioPos: audioPos)

    // I2V mask 预构造一次（仅依赖 Nv/shape）：编译图捕获 + 采样循环每步复用，省重复 CPU→GPU 拷贝
    let maskMLX: MLXArray? = condMask.map { MLXArray($0).asType(.float32) }                     // [Nv] f32
    let maskMLXInv: MLXArray? = condMask.map { MLXArray($0.map { 1.0 - $0 }).asType(.float32) } // [Nv] f32
    let maskTensor: MLXArray? = condMask.map { mask in
        MLXArray(mask).reshaped([1, noiseV.shape[1], noiseV.shape[2], noiseV.shape[3], 1]).asType(.bfloat16)
    }

    // 编译 DiT 单步前向（可选，默认开；编译产物持久缓存复用）
    // 音频条件模式需捕获 audioTimesteps=[0]（frozen），与纯 t2v 编译图分离
    // I2V：per-token timestep = mask*sigma 与 keyframes gate 均为纯张量运算，
    // 常量由闭包捕获进编译图；shapeKey 带 isI2V 标志防止与 T2V 图复用。
    var forward: CompiledDitForward? = nil
    if config.useCompile && mlxCompileEnabled {
        // 编译图依赖 latent/text 的 shape（rope/位置常量由 shape 隐含决定）+ 是否音频条件 + 是否 I2V
        let shapeKey = "\(noiseV.shape)-\(noiseA.shape)-\(condV.shape)-\(condA.shape)-frozen\(frozenAudio != nil)-i2v\(isI2V)"
        forward = ltxCompiledForward.resolve(owner: dit, key: shapeKey) {
            let cf = makeCompiledDitForward(
                dit: dit, rope: rope, videoPos: videoPos, audioPos: audioPos,
                audioTimesteps: frozenAudio == nil ? nil : MLXArray([Float(0.0)]),
                condMask: condMask, maskMLX: maskMLX, maskMLXInv: maskMLXInv)
            // 预热：跑一次编译（触发图构建 + 内核编译），用噪声 latent 同 shape
            let (wv, wa) = cf(noiseV, noiseA, MLXArray([sigmas[0]]), condV, condA)
            eval(wv, wa)
            return cf
        }
    }

    var vx = initVideo ?? noiseV
    var ax = noiseA
    // TeaCache：相邻步 sigma 差小于阈值时复用上一步 x0，跳过整次 DiT 前向（训练无关）。
    let teaCache = TeaCache()
    teaCache.threshold = config.teaCacheThreshold
    let t0 = Date()
    for i in 0..<(sigmas.count - 1) {
        if isCancelled() {
            print("⏹ 采样已取消（第 \(i + 1) 步前），提前结束")
            MLX.Memory.clearCache()
            return (vx, ax)
        }
        let sigma = sigmas[i]
        let sigmaNext = sigmas[i + 1]
        let stepT0 = Date()
        memProfilePoint(String(format: "采样 步%02d前", i + 1))
        // 每步独立作用域：x0/中间张量在本轮结束时立即释放，
        // 避免 48 层前向的中间结果攒到采样结束一次性析构（12GB 级大爆炸）。
        autoreleasepool {
            let x0: (v: MLXArray, a: MLXArray)
            if config.teaCache {
                var cv: MLXArray?
                var ca: MLXArray?
                if teaCache.tryReuse(sigma: sigma, outV: &cv, outA: &ca), let rv = cv, let ra = ca {
                    x0 = (rv, ra)
                    print("    [TeaCache] 步 \(i + 1) σ=\(sigma) 命中缓存，跳过 DiT 前向")
                } else {
                    let fresh = ditX0Guided(
                        dit: dit, forward: forward,
                        vx: vx, ax: ax, sigma: sigma,
                        condV: condV, condA: condA,
                        negV: negV, negA: negA,
                        frozenAudio: frozenAudio,
                        condMask: condMask, cleanV: cleanV,
                        maskTensor: maskTensor, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                        videoPos: videoPos, audioPos: audioPos,
                        rope: rope,
                        cfg: config)
                    x0 = fresh
                    teaCache.store(v: fresh.v, a: fresh.a)
                }
            } else {
                x0 = ditX0Guided(
                    dit: dit, forward: forward,
                    vx: vx, ax: ax, sigma: sigma,
                    condV: condV, condA: condA,
                    negV: negV, negA: negA,
                    frozenAudio: frozenAudio,
                    condMask: condMask, cleanV: cleanV,
                    maskTensor: maskTensor, maskMLX: maskMLX, maskMLXInv: maskMLXInv,
                    videoPos: videoPos, audioPos: audioPos,
                    rope: rope,
                    cfg: config)
            }
            var nvx: MLXArray
            let nax: MLXArray
            if config.ancestral && sigmaNext > 0 {
                // 每步独立可复现噪声（seed 派生：seed + 步号 + 固定偏移，避免与初始 latent 噪声同源）
                MLXRandom.seed(config.seed &+ UInt64(i) &+ 0x9E37_79B9)
                let noiseV = MLXRandom.normal(vx.shape).asType(.bfloat16)
                let noiseA = MLXRandom.normal(ax.shape).asType(.bfloat16)
                nvx = ancestralEulerStep(vx, x0.v, sigma: sigma, sigmaNext: sigmaNext, noise: noiseV)
                nax = ancestralEulerStep(ax, x0.a, sigma: sigma, sigmaNext: sigmaNext, noise: noiseA)
            } else {
                nvx = eulerStep(vx, x0.v, sigma: sigma, sigmaNext: sigmaNext)
                nax = eulerStep(ax, x0.a, sigma: sigma, sigmaNext: sigmaNext)
            }
            // I2V：采样轨迹每步强制写回干净首帧（对齐官方 denoise_mask=0 语义：
            // post_process_latent 于 x_next，SDE 路径首帧 per-token sigma=0 不注入噪声）。
            // 不加此行时 ancestral 步的 renoise 会持续污染首帧，导致"首帧只闪一下"。
            if let mask = condMask, let clean = cleanV {
                let m = maskTensor ?? MLXArray(mask).reshaped([1, nvx.shape[1], nvx.shape[2], nvx.shape[3], 1]).asType(.bfloat16)
                nvx = nvx * m + clean * (1 - m)
            }
            eval(nvx, nax)
            vx = nvx
            ax = nax
        }
        memProfilePoint(String(format: "采样 步%02d后", i + 1))
        let itMs = Date().timeIntervalSince(stepT0)
        monitorLog(String(format: "步%02d  %.2fs/it", i + 1, itMs))
        print(String(format: "    步%02d  %.2fs/it", i + 1, itMs))
    }
    // 采样完成：编译图常驻缓存复用，不释放（避免 Compiled 大图一次性析构卡死）。
    // 仅回收采样过程中产生的空闲 buffer 缓存。
    MLX.Memory.clearCache()
    let totalSec = Int(Date().timeIntervalSince(t0))
    monitorLog(String(format: "✅ 采样完成（%d分%02d秒）", totalSec / 60, totalSec % 60))
    print(String(format: "✅ 采样完成（%d分%02d秒）", totalSec / 60, totalSec % 60))
    return (vx, ax)
}
