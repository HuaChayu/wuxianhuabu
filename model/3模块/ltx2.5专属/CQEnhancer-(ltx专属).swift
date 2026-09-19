// CQEnhancer.swift — LTX-2.5 CQ Video Enhancer LoRA 旁路注入（第二阶段·CQ 清晰度增强通道）
//
// 【为什么新增这条通道】
//   H3 一采 → LTX 像素桥「IC-LoRA 二采」在实机上出现崩坏 / 换脸。改为使用 CQdesign 发布的
//   LTX-2.5 CQ Video Enhancer LoRA（视频版）做**生成式清晰度增强**（官方口径：不是普通放大器，
//   而是增强/修复低质视频）。本通道与既有 IC 像素桥二采**并存、互斥切换**（设置项 videoUseCQEnhancer），
//   不改动、不删除原有二采分支。
//
// 【官方规格】CQdesign/LTX-2.5-CQ-Video-and-Image-Enhancer-LoRAs（README + 配套 ComfyUI 工作流）
//   · LoraLoaderModelOnly: ltx2.5-CQ-enhancer-lora-for-videos-rank128.safetensors, strength = 1.0
//   · 基座 ltx-2.5-22b-dev-transformer（另叠加 distilled-lora-450，strength = 0.5）
//   · VAE 必须用 ltx-2.5-video-vae-conv-bf16.safetensors（卷积版；普通 VAE 会过锐化、降低质量）
//   · 低清视频经 LTXVImgToVideoConditionOnly(bypass) + LTXAddVideoICLoRAGuide 作 in-context 参考
//   · 无需提示词（空文本条件 / prompt 为空）
//   · ManualSigmas = 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0（8 步）
//   · sampler = euler_ancestral；cfg = 1
//
// 【本机权重只读检查事实】~/Downloads/ltx2.5/ltx2.5-CQ-enhancer-lora-for-videos-rank128.safetensors
//   · 1 323 282 248 字节（≈1.23 GiB）、3744 键、无 __metadata__
//   · 键名 ComfyUI 风格：lora_unet_transformer_blocks_{i}_{模块}.lora_down.weight / .lora_up.weight / .alpha
//   · 48 层全覆盖（i = 0..47），**每层 26 个模块**（= 78 个 lora_* 张量键 + 26 个 alpha = 104 键/层），
//     alpha 全部为 0 维 f32 标量，且 alpha / rank 恒 = 1.0（视频流 rank=64 → alpha=64；音频/交叉流 rank=1 → alpha=1），
//     故官方 strength=1.0 时每对等效 scale = 1.0，即等价增量 W' = W + B@A：
//       视频流 attn1 / attn2 的 to_q / to_k / to_v / to_out.0           → 8 个模块（rank 64，in=out=4096）
//       视频流 ff_net_0_proj（4096→16384，rank 64）/ ff_net_2（16384→4096，rank 64） → 2 个模块
//       音频流 audio_attn1 / audio_attn2 的 q / k / v / to_out.0         → 8 个模块（rank 1，2048）
//       音频流 audio_ff_net_0_proj / audio_ff_net_2                      → **本权重中不存在**（视频版未训练音频 FF 增量）
//       AV 交叉 audio_to_video_attn（q∈4096 视频｜kv∈2048 音频｜out 2048→4096）→ 4 个模块（rank 1）
//              video_to_audio_attn（q∈2048 音频｜kv∈4096 视频｜out 2048→2048）→ 4 个模块（rank 1）
//   · 无 to_gate_logits 键；down/up 源 dtype 全为 f32，attach 时统一 cast 到 PrecisionPolicy.defaultMainDType
//   · 与主干槽位形状逐一对齐（LTXGatedAttention: to_q(x) / to_k(context) / to_v(context) / to_out(gate 后 summed)）
//
// 【与 IC-LoRA 的关系】
//   主干（ltx主干-(ltx专属).swift）的 icLoraQ/K/V/O、icLoraIn/Out 槽位是**通用 LoRA 增量槽**，
//   本文件把 CQ 权重挂到同一批槽位，由 setCQActive 全局启停（底层与 setICActive 写同一组 icActive 标志）；
//   两条通道互斥执行（CQ 优先），且启停动作由 runLTXStage2RefineOnLatent 的**函数级 defer** 统一负责
//   （严禁写在挂载处 if 块内 —— defer 会在 if 块闭合处提前触发，导致采样全程 LoRA 不注入）。
//   原生 LTX 路径零参与：通道不激活时 icActive=false，主干不做任何增量计算。
//   每个 pair 自带 scale（IC 通道 = 0.5 官方硬编码；CQ 通道 = 1.0 官方 strength），互不串味。

import Foundation
@preconcurrency import MLX

// MARK: - 常量

/// 官方 CQ Video Enhancer LoRA 权重默认位置（可用环境变量 LTX_CQ_ENHANCER_PATH 覆盖）
let cqEnhancerDefaultPath: String = "\(CommonPaths.modelRoot)/ltx2.5/ltx2.5-CQ-enhancer-lora-for-videos-rank128.safetensors"

/// 官方应用强度：工作流中 LoraLoaderModelOnly 的 strength_model = 1.0
let cqEnhancerStrength: Float = 1.0

/// CQ Enhancer 是否默认启用 ancestral（euler_ancestral）：官方工作流采样器为 euler_ancestral，
/// 故默认 true；置环境变量 LTX_CQ_ANCESTRAL=0 可切确定性 Euler（对齐 IC 二采的确定性语义）。
let cqEnhancerAncestralDefault: Bool = true

// MARK: - 单层权重结构

/// 一个注意力模块的 4 组增量（to_q / to_k / to_v / to_out.0）
struct CQEnhancerAttn {
    let q: ICLoRAPair
    let k: ICLoRAPair
    let v: ICLoRAPair
    let o: ICLoRAPair
    var isPresent: Bool { q.isPresent || k.isPresent || v.isPresent || o.isPresent }
}

/// FFN 的 2 组增量（ff.net.0.proj = proj_in，ff.net.2 = proj_out）
struct CQEnhancerFFN {
    let projIn: ICLoRAPair
    let projOut: ICLoRAPair
    var isPresent: Bool { projIn.isPresent || projOut.isPresent }
}

/// 单层 CQ Enhancer LoRA：官方 26 模块按主干结构归为 8 组（视频流 3 组 + 音频流 3 组 + AV 交叉 2 组）
final class CQEnhancerBlock {
    let layerIndex: Int
    // 视频流：attn1（自注意力）、attn2（视频-文本交叉）、ff
    let attn1: CQEnhancerAttn
    let attn2: CQEnhancerAttn
    let ff: CQEnhancerFFN
    // 音频流：audio_attn1、audio_attn2、audio_ff（本视频版权重无 audio_ff 键 → 恒为空对）
    let audioAttn1: CQEnhancerAttn
    let audioAttn2: CQEnhancerAttn
    let audioFF: CQEnhancerFFN
    // AV 交叉：audio_to_video_attn、video_to_audio_attn
    let audioToVideo: CQEnhancerAttn
    let videoToAudio: CQEnhancerAttn

    init(layerIndex: Int,
         attn1: CQEnhancerAttn, attn2: CQEnhancerAttn, ff: CQEnhancerFFN,
         audioAttn1: CQEnhancerAttn, audioAttn2: CQEnhancerAttn, audioFF: CQEnhancerFFN,
         audioToVideo: CQEnhancerAttn, videoToAudio: CQEnhancerAttn) {
        self.layerIndex = layerIndex
        self.attn1 = attn1; self.attn2 = attn2; self.ff = ff
        self.audioAttn1 = audioAttn1; self.audioAttn2 = audioAttn2; self.audioFF = audioFF
        self.audioToVideo = audioToVideo; self.videoToAudio = videoToAudio
    }
}

// MARK: - 权重解析

/// 解析官方 CQ Video Enhancer LoRA（48 层 × 26 模块）为可挂载的旁路块。
/// - Parameters:
///   - path: safetensors 绝对路径
///   - strength: 应用强度（官方 1.0）；每对的等效 scale = (alpha / rank) × strength
/// - Returns: 长度 48 的数组（layerIndex 0..47）；文件缺失/解析失败返回 nil。
func loadCQEnhancerBlocks(path: String, strength: Float = cqEnhancerStrength) -> [CQEnhancerBlock]? {
    guard FileManager.default.fileExists(atPath: path),
          let arrays = try? MLX.loadArrays(url: URL(fileURLWithPath: path)) else {
        return nil
    }
    let dtype = PrecisionPolicy.defaultMainDType

    /// 取一组 down/up（+ alpha）→ 带 scale 的增量对；缺键返回空对（delta 恒为 nil，等效不注入）
    func pair(_ base: String) -> ICLoRAPair {
        let key = "lora_unet_\(base)"
        guard let down = arrays["\(key).lora_down.weight"],
              let up = arrays["\(key).lora_up.weight"] else {
            return ICLoRAPair(a: nil, b: nil)
        }
        let rank = Float(max(down.shape.first ?? 1, 1))
        // alpha 为 0 维 f32 张量（本权重实测：视频流 64、音频/交叉流 1）；缺失时按 rank 兜底 → scale=1
        var alpha = rank
        if let alphaArr = arrays["\(key).alpha"] {
            alpha = alphaArr.reshaped([-1]).item(Float.self)
        }
        return ICLoRAPair(a: down.asType(dtype), b: up.asType(dtype), scale: (alpha / rank) * strength)
    }

    /// attn 组：to_q / to_k / to_v / to_out.0
    func attn(_ prefix: String) -> CQEnhancerAttn {
        CQEnhancerAttn(q: pair("\(prefix)_to_q"),
                       k: pair("\(prefix)_to_k"),
                       v: pair("\(prefix)_to_v"),
                       o: pair("\(prefix)_to_out_0"))
    }

    /// FFN 组：net.0.proj（proj_in）/ net.2（proj_out）
    func ffn(_ prefix: String) -> CQEnhancerFFN {
        CQEnhancerFFN(projIn: pair("\(prefix)_ff_net_0_proj"),
                      projOut: pair("\(prefix)_ff_net_2"))
    }

    var blocks: [CQEnhancerBlock] = []
    blocks.reserveCapacity(48)
    for layer in 0..<48 {
        let p = "transformer_blocks_\(layer)"
        blocks.append(CQEnhancerBlock(
            layerIndex: layer,
            attn1: attn("\(p)_attn1"),
            attn2: attn("\(p)_attn2"),
            ff: ffn(p),
            audioAttn1: attn("\(p)_audio_attn1"),
            audioAttn2: attn("\(p)_audio_attn2"),
            // 本视频版权重无 audio_ff_* 键：pair() 返回空对（isPresent=false → delta 恒 nil，不注入），
            // 槽位保留以兼容将来含音频 FF 增量的 checkpoint（如图像版 / 后续版本），届时无需改代码。
            audioFF: ffn("\(p)_audio_ff"),
            audioToVideo: attn("\(p)_audio_to_video_attn"),
            videoToAudio: attn("\(p)_video_to_audio_attn")))
    }
    return blocks
}

// MARK: - 常驻缓存

/// CQ Enhancer 权重常驻缓存（≈1.23 GiB 源权重 → cast 后常驻，避免每次采样重复读盘/解析）。
/// 未启用 CQ 通道时**不会**触发加载（调用方在分派前即短路），故不影响既有 IC 二采的内存表现。
enum CQEnhancerCache {
    static var blocks: [CQEnhancerBlock]?
    static var loadedPath: String = ""
    static var loadedModTime: Date?
    static var loadedStrength: Float = -1

    /// 返回与文件/强度匹配的已解析权重；未加载或文件/强度变化则重新解析。
    static func blocks(for path: String, strength: Float = cqEnhancerStrength) -> [CQEnhancerBlock]? {
        let mod = FileManager.default.fileModTime(path)
        if let b = blocks, loadedPath == path, loadedModTime == mod, loadedStrength == strength {
            return b
        }
        guard let b = loadCQEnhancerBlocks(path: path, strength: strength) else { return nil }
        blocks = b
        loadedPath = path
        loadedModTime = mod
        loadedStrength = strength
        return b
    }
}

// MARK: - 主干旁路附着与开关

extension LTXVideoDiT {
    /// 将 48 层 CQ Enhancer 旁路引用挂到主干对应 attn/ff（幂等，可重复调用）。
    /// 与 attachICLoRA 使用同一批通用槽位，但互不冲突（挂载本身不改变计算，各层 icActive 默认 false）。
    func attachCQEnhancer(_ blocks: [CQEnhancerBlock]) {
        guard blocks.count == transformer_blocks.count else { return }
        for (idx, block) in transformer_blocks.enumerated() {
            let lora = blocks[idx]
            // 视频流
            block.attn1.icLoraQ = lora.attn1.q
            block.attn1.icLoraK = lora.attn1.k
            block.attn1.icLoraV = lora.attn1.v
            block.attn1.icLoraO = lora.attn1.o
            block.attn2.icLoraQ = lora.attn2.q
            block.attn2.icLoraK = lora.attn2.k
            block.attn2.icLoraV = lora.attn2.v
            block.attn2.icLoraO = lora.attn2.o
            block.ff.icLoraIn = lora.ff.projIn
            block.ff.icLoraOut = lora.ff.projOut
            // 音频流
            block.audio_attn1.icLoraQ = lora.audioAttn1.q
            block.audio_attn1.icLoraK = lora.audioAttn1.k
            block.audio_attn1.icLoraV = lora.audioAttn1.v
            block.audio_attn1.icLoraO = lora.audioAttn1.o
            block.audio_attn2.icLoraQ = lora.audioAttn2.q
            block.audio_attn2.icLoraK = lora.audioAttn2.k
            block.audio_attn2.icLoraV = lora.audioAttn2.v
            block.audio_attn2.icLoraO = lora.audioAttn2.o
            block.audio_ff.icLoraIn = lora.audioFF.projIn
            block.audio_ff.icLoraOut = lora.audioFF.projOut
            // AV 交叉
            block.audio_to_video_attn.icLoraQ = lora.audioToVideo.q
            block.audio_to_video_attn.icLoraK = lora.audioToVideo.k
            block.audio_to_video_attn.icLoraV = lora.audioToVideo.v
            block.audio_to_video_attn.icLoraO = lora.audioToVideo.o
            block.video_to_audio_attn.icLoraQ = lora.videoToAudio.q
            block.video_to_audio_attn.icLoraK = lora.videoToAudio.k
            block.video_to_audio_attn.icLoraV = lora.videoToAudio.v
            block.video_to_audio_attn.icLoraO = lora.videoToAudio.o
        }
    }

    /// 全局启停 CQ Enhancer 旁路注入：仅 CQ 清晰度增强采样段置 true。
    /// 与 setICActive 使用同一批槽位的 icActive 标志，但两条通道互斥执行（见 runLTXStage2RefineOnLatent 分派）。
    /// ⚠️ 关闭必须由调用方的**函数级 defer** 统一执行（覆盖整段采样+解码），
    ///    严禁在挂载处的 if 块内 defer —— Swift 的 defer 会在该 if 块闭合处提前触发，
    ///    导致采样全程 icActive=false（LoRA 不注入），历史崩坏/换脸即源于此。
    func setCQActive(_ active: Bool) {
        for block in transformer_blocks {
            block.attn1.icActive = active
            block.attn2.icActive = active
            block.ff.icActive = active
            block.audio_attn1.icActive = active
            block.audio_attn2.icActive = active
            block.audio_ff.icActive = active
            block.audio_to_video_attn.icActive = active
            block.video_to_audio_attn.icActive = active
        }
    }
}

// MARK: - 权重元数据：参考降采样因子（reference_downscale_factor）

/// 官方默认参考降采样因子：元数据缺失 / 读取失败 / 非法值时的兜底（同格同位参考）。
let loraRefDownscaleFactorDefault: Int = 1

/// 读取 safetensors LoRA 权重的 `__metadata__.reference_downscale_factor`（官方参考降采样因子）。
///
/// 【官方口径】IC-LoRA / CQ-Enhancer LoRA 在训练时会把自己的「参考 latent 相对目标网格的降采样因子」
/// 写进权重元数据。in-context 注入参考时：参考 latent 所在网格 = 目标网格 / factor，
/// 参考 RoPE 位置 = 参考格像素中点 × factor（映射回目标网格相位）。
///   · IC-LoRA x2 放大器：metadata factor = 2（参考 = 半清格，位置 ×2）
///   · CQ Video Enhancer：权重无 `__metadata__`（本机 3744 键、__metadata__=None）→ 官方默认 1
///     （参考与目标同格同位，不乘任何因子）
///
/// 【为什么必须读元数据而不是沿用 IC 的常量 2】历史缺陷：CQ 通道沿用了 IC 权重放大器的硬编码
/// 因子 2，导致 CQ 的参考网格（半清）与目标网格（×2 升频后的全清）尺寸+相位双重错位，
/// 参考 KV 与主序列不同相位 → 主序列失去锚点、整段重绘（角色数量/形象全变）。
///
/// 【读法】safetensors 头部 = 8 字节小端 u64 头长 + 头 JSON；仅读取这两段，不载入 1.2GB 权重本体。
/// 解析失败一律返回 `fallback`（默认 1），保证「缺元数据 → 同格参考」的官方语义。
func loraReferenceDownscaleFactor(path: String, fallback: Int = loraRefDownscaleFactorDefault) -> Int {
    guard let handle = FileHandle(forReadingAtPath: path) else {
        pipelineLog("⚠️ [LoRA 元数据] 权重不可读：\(path) → reference_downscale_factor 取兜底 \(fallback)")
        return fallback
    }
    defer { try? handle.close() }
    guard let lenData = try? handle.read(upToCount: 8), lenData.count == 8 else { return fallback }
    var headerLenLE: UInt64 = 0
    for (i, byte) in lenData.enumerated() { headerLenLE |= UInt64(byte) << (8 * UInt64(i)) }
    // 头部长度护栏：正常权重头部 ≪ 16MB；异常（截断/非 safetensors/全零）直接兜底
    let headerCap: UInt64 = 16 * 1024 * 1024
    guard headerLenLE > 0, headerLenLE <= headerCap else { return fallback }
    guard let headerData = try? handle.read(upToCount: Int(headerLenLE)), headerData.count == Int(headerLenLE) else {
        return fallback
    }
    guard let any = try? JSONSerialization.jsonObject(with: headerData),
          let obj = any as? [String: Any] else { return fallback }
    let meta = (obj["__metadata__"] as? [String: Any]) ?? (obj["metadata"] as? [String: Any])
    guard let raw = meta?["reference_downscale_factor"] else { return fallback }
    if let s = raw as? String, let v = Int(s), v >= 1 { return v }
    if let n = raw as? NSNumber, n.intValue >= 1 { return n.intValue }
    return fallback
}
