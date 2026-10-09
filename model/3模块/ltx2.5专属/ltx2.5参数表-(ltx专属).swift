//
//  PipelineConfig.swift
//  无限画布 — 管线配置表（2 张大表：Stage1 / Stage2，每张表自包含所有参数）
//
//  设计原则（人类阅读性优先）：
//  - 总共只有 2 张大表：Stage1Config（阶段1）、Stage2Config（阶段2）
//  - 每张表内部：模式枚举、公共路径写在开头，方案工厂用 case 区分
//    不同地址/参数——读一张表 = 完整了解该阶段要哪些文件/参数/开关
//  - 方案工厂返回完整的表值（全部字段都有值），骨干只查表执行
//  - CommonPaths 仅供非流程场景（文本编码 / 音频 / VAE 解码），与两张表无关
//
//  用法：
//    let s1 = Stage1Config.distilled(g: g)   // Stage1 固定走蒸馏；dev 工厂保留不调用（留管线）
//    let s2: Stage2Config? = stage2Refine ? Stage2Config.refine(g: g) : nil

import Foundation
import MLX

// MARK: - 阶段1 大表（所有参数 + 方案 case）

struct Stage1Config {

    // ==================== 模式枚举（方案区分键） ====================

    enum Mode {
        case distilled   // 蒸馏：固定 sigma 表 + ancestral SDE，无 CFG
        case dev         // dev：DynamicShift + 20 步确定性 Euler + CFG 3.0/7.0
    }

    /// Stage1 负向条件策略
    enum NegStrategy {
        case none            // 蒸馏：无 CFG，不编码负向
        case passThrough     // dev：负向条件透传给采样
    }

    // ==================== 公共路径（两方案共用） ====================

    static let base = CommonPaths.ltxServeDir
    static let vaeEncoder = "\(base)/vae_encoder.safetensors"   // I2V 首帧编码

    // ==================== 表字段（所有参数） ====================

    var mode: Mode
    var ditPath: String          // DiT 权重（蒸馏 / dev）
    var vaeEncoderPath: String   // I2V 首帧编码
    var sampler: SamplerConfig   // 步数 / cfg / sigmas / ancestral / seed
    var negStrategy: NegStrategy // on → 编码负向条件并透传（CFG）

    // ==================== 方案工厂（case 区分） ====================

    /// 方案 A：蒸馏
    static func distilled(g: GenConfig) -> Stage1Config {
        Stage1Config(
            mode: .distilled,
            ditPath: "\(base)/transformer-distilled.safetensors",
            vaeEncoderPath: vaeEncoder,
            sampler: SamplerConfig(numSteps: g.numSteps, cfgV: 1.0, cfgA: 1.0,
                                   sigmas: ltx25DistilledSigmas, ancestral: true, seed: g.seed,
                                   teaCache: false), // 蒸馏 8 步小步长表：rel-L1 探针相邻步几乎不变→连跳 block→方块感；关闭（等价旧版无缓存）
            negStrategy: .none)
    }

    /// 方案 B：dev
    static func dev(g: GenConfig) -> Stage1Config {
        Stage1Config(
            mode: .dev,
            ditPath: "\(base)/transformer-dev.safetensors",
            vaeEncoderPath: vaeEncoder,
            sampler: SamplerConfig(numSteps: 20, cfgV: 3.0, cfgA: 7.0,
                                   sigmas: nil, ancestral: false, seed: g.seed,
                                   teaCache: true,
                                   // 官方 dev 默认引导：STG=1.0 @ block28 + modality=3.0（每步 4 次前向：
                                   // cond/uncond/ptb/mod，补回跨模态交互与时空一致性，消除网格感/音频怪异）
                                   stgScaleV: 1.0, stgScaleA: 1.0,
                                   modalityScaleV: 3.0, modalityScaleA: 3.0,
                                   stgBlocksV: [28], stgBlocksA: [28]), // dev 单阶段：官方 rel-L1 语义 TeaCache（特征判据+残差缓存+CFG 分支独立），dev/蒸馏表均适用
            negStrategy: .passThrough)
    }
}

// MARK: - 阶段2 大表（所有参数 + 方案 case）

struct Stage2Config {

    // ==================== 模式枚举（方案区分键） ====================

    enum Mode {
        case refine      // 旧升频：空间 ×2 + 3 步蒸馏 refine
    }

    /// Stage2 guide 构造策略
    enum GuideStrategy {
        case none                  // 无 guide（旧 refine 纯 t2v）
        case firstLastFrames       // I2V 首尾帧重编码钉入（旧 refine）
    }

    /// 位置编码策略
    enum PosStrategy {
        case plain                 // 普通全分辨率位置
    }

    // ==================== 公共路径（两方案共用） ====================

    static let base = CommonPaths.ltxServeDir
    static let spatialUpscaler = "\(base)/spatial_upscaler_x2_v1_1.safetensors"   // 空间 ×2 起步
    static let vaeDecoder = "\(base)/vae_decoder.safetensors"                     // mean/std 统计
    static let vaeEncoder = "\(base)/vae_encoder.safetensors"                     // I2V 首尾帧重编码

    // ==================== 表字段（所有参数） ====================

    var mode: Stage2Config.Mode
    var upscalerPath: String         // 空间升频权重（两分支起步都用）
    var vaeEncoderPath: String?      // I2V 首尾帧重编码（guideStrategy=.firstLastFrames 时需要）
    var vaeDecoderPath: String       // mean/std 统计（升频 norm 需要）
    var sampler: SamplerConfig       // 步数 / cfg / sigmas / ancestral / seed
    var guideStrategy: GuideStrategy // guide 构造方式
    var posStrategy: PosStrategy     // 位置编码方式
    var clearCompiledBefore: Bool    // on → 采样前清编译图（双驻留修复）
    var strength: Float              // guide 强度（1.0 = 完全钉入）

    // ==================== 方案工厂（case 区分） ====================

    /// 方案 A：旧升频 + 3 步蒸馏 refine
    static func refine(g: GenConfig) -> Stage2Config {
        Stage2Config(
            mode: .refine,
            upscalerPath: spatialUpscaler,
            vaeEncoderPath: vaeEncoder,          // 差异：refine 需要首尾帧重编码钉入
            vaeDecoderPath: vaeDecoder,
            sampler: SamplerConfig(numSteps: ltx25Stage2Sigmas.count - 1,
                                   cfgV: 1.0, cfgA: 1.0,
                                   sigmas: ltx25Stage2Sigmas, ancestral: false, seed: g.seed),
            guideStrategy: .firstLastFrames,
            posStrategy: .plain,
            clearCompiledBefore: true,
            strength: 1.0)
    }
}

// MARK: - 非流程公共资源（文本编码 / 音频 / VAE 解码 使用；不属于 Stage1/2 流程表）

enum CommonPaths {
    /// 统一模型根：项目文档地址/model（跟随偏好设置的项目地址，模型不再散落 Downloads）
    static var modelRoot: String {
        URL(fileURLWithPath: AppSettings.shared.canvasRootPath)
            .appendingPathComponent("model").path
    }
    static let ltxServeDir = "\(CommonPaths.modelRoot)/ltx2.5/LTX-2.5-MLX-Serve-4bit"
    static let gemmaDir = "\(CommonPaths.modelRoot)/ltx2.5/gemma4-12b-ltx-v1"
    static let vaeDecoder = "\(ltxServeDir)/vae_decoder.safetensors"
    // LTX-2.5 扩散视频解码器（Diffusion Video Decoder）权重：官方 PyTorch 原版
    // ltx-2.5-video-vae-bf16.safetensors（1.4GB，bf16，decoder 部分为 det_stages+diff_blocks，
    // 键前缀 decoder.、注意力为合并 attn.qkv；DiffusionVideoDecoder 已适配该格式，并兼容旧
    // mlx-community 版 vae_diffusion_decoder.* 前缀 + to_q/to_k/to_v 拆分键）。
    static let vaeDiffusionDecoder = "\(ltxServeDir)/ltx-2.5-video-vae-bf16.safetensors"
    static let spatialUpscaler = "\(ltxServeDir)/spatial_upscaler_x2_v1_1.safetensors"
    static let audioVae = "\(ltxServeDir)/audio_vae.safetensors"
    static let vocoder = "\(ltxServeDir)/vocoder.safetensors"
}

// MARK: - 公共函数：VAE mean/std 轻量读取（Stage2 升频 norm 用）
//   只读 2×128 标量（几十 KB），避免整读 vae_decoder（777MB）；完整权重留给 VAE 解码阶段只读一次盘。

func loadVAEMeanStd(path: String) -> (mean: MLXArray, std: MLXArray)? {
    guard let stats = readSafetensorsFloatArrays(path: path, keys: [
        "vae_decoder.per_channel_statistics.mean",
        "vae_decoder.per_channel_statistics.std"
    ]),
    let mean0 = stats["vae_decoder.per_channel_statistics.mean"],
    let std0 = stats["vae_decoder.per_channel_statistics.std"] else {
        return nil
    }
    let mean = mean0.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType)
    let std = std0.reshaped([1, 1, 1, 1, 128]).asType(PrecisionPolicy.defaultMainDType)
    return (mean, std)
}
