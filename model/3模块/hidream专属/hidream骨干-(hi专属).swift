//
//  HiDreamBackbone（HiDream专用）.swift
//  无限画布 — HiDream-O1-Image 顶层容器（方案 b 类拆分）
//
//  [拆分] 自 HiDreamPipeline.swift：HiDreamBackbone 顶层容器 + hidreamSanitizeWeights。
//  组装 visionTower(Qwen3VL 视觉编码) + languageModel(Qwen3VL 语言模型) +
//  扩散 custom heads（t_embedder1 / x_embedder / final_layer2），
//  权重键对齐 HiDream-O1 safetensors（vision_tower.* / language_model.* / t_embedder1.* / x_embedder.* / final_layer2.*）。
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - 顶层容器

final class HiDreamBackbone: Module {

    @ModuleInfo(key: "vision_tower") var visionTower: HiDreamVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: HiDreamLanguage.LanguageModel
    @ModuleInfo(key: "t_embedder1") var tEmbedder1: HiDreamTimestepEmbedder
    @ModuleInfo(key: "x_embedder") var xEmbedder: HiDreamBottleneckPatchEmbed
    @ModuleInfo(key: "final_layer2") var finalLayer2: HiDreamFinalLayer

    let qwenConfig: Qwen3VLConfiguration

    init(qwenConfig: Qwen3VLConfiguration) {
        self.qwenConfig = qwenConfig
        super.init()
        self.visionTower = HiDreamVision.VisionModel(qwenConfig.visionConfiguration)
        self.languageModel = HiDreamLanguage.LanguageModel(qwenConfig)
        self.tEmbedder1 = HiDreamTimestepEmbedder(
            hiddenSize: qwenConfig.textConfiguration.hiddenSize)
        self.xEmbedder = HiDreamBottleneckPatchEmbed(
            pcaDim: HiDreamDiffusionConstants.bottleneckDim,
            embedDim: qwenConfig.textConfiguration.hiddenSize)
        self.finalLayer2 = HiDreamFinalLayer(
            hiddenSize: qwenConfig.textConfiguration.hiddenSize)
    }
}

// MARK: - 权重预处理

/// 与 py/Qwen3VL 对齐的权重清理：patch_embed.proj.weight 需从
/// [O, C, kT, kH, kW] 转置为 MLX Conv3d 布局 [O, kT, kH, kW, C]。
func hidreamSanitizeWeights(_ weights: inout [String: MLXArray]) {
    for (key, value) in weights {
        if key.contains("patch_embed.proj.weight") {
            if value.ndim == 5 && value.dim(-1) != 3 {
                weights[key] = value.transposed(0, 2, 3, 4, 1)
            }
        }
    }
}

