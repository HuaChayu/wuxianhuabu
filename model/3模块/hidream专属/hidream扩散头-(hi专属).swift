//
//  HiDreamDiffusion.swift
//  无限画布 — HiDream-O1-Image 扩散侧（timestep embedder / patch embed / final layer）
//  对应 Python: scripts/hidream_o1/hidream_model.py + flow_match.py + pipeline_helpers.py
//
//  ============================================================
//  说明：HiDream-O1-Image 生成侧 = Qwen3VL 骨干（vision tower + language model）
//  + 3 个扩散 custom heads（t_embedder1 / x_embedder / final_layer2）。
//  本文件只放扩散侧模块；timestepEmbedding、FlashFlowMatchScheduler、
//  closestResolution 已移入通用算子&函数/通用算子/算子-采样.swift。
//  权重来源：extras/custom_heads.safetensors（键名已与 MLX 对齐，无需 remap）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - 常量

enum HiDreamDiffusionConstants {
    static let patchSize = 32
    static let inChannels = 3
    static let hiddenSize = 4096
    static let bottleneckDim = 1024
    static let frequencyEmbeddingSize = 256
    static let timestepTokenNum = 1
    static let tmsTokenID = 151673
    static let imageTokenID = 151655
    static let videoTokenID = 151656
    static let visionStartTokenID = 151652

    static let tEps: Float = 0.001
    static let noiseScaleDefault: Float = 7.5
    static let numTrainTimesteps = 1000

    /// 训练分辨率表（禁用 snapping 时出现 off-spec 伪影）
    static let predefinedResolutions: [(w: Int, h: Int)] = [
        (2048, 2048),
        (2304, 1728), (1728, 2304),
        (2560, 1440), (1440, 2560),
        (2496, 1664), (1664, 2496),
        (3104, 1312), (1312, 3104),
        (2304, 1792), (1792, 2304),
    ]

    static let defaultTimesteps: [Float] = [
        999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
        707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
    ]
}

// MARK: - TimestepEmbedder (t_embedder1)

/// 时间步嵌入：fc1(256→4096) + silu + fc2(4096→4096)
final class HiDreamTimestepEmbedder: Module, UnaryLayer {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let frequencyEmbeddingSize: Int

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        _fc1.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize)
        _fc2.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let tFreq = timestepEmbedding(t * 1000.0, dim: frequencyEmbeddingSize)
        let h = silu(fc1(tFreq.asType(fc1.weight.dtype)))
        return fc2(h)
    }
}

// MARK: - BottleneckPatchEmbed (x_embedder)

/// 图像 patch 嵌入：proj1(3072→1024, 无 bias) + proj2(1024→4096, 有 bias)
/// 输入 vinputs: [B, N_patch, 32*32*3]
final class HiDreamBottleneckPatchEmbed: Module, UnaryLayer {

    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    init(patchSize: Int = 32, inChannels: Int = 3, pcaDim: Int = 1024, embedDim: Int = 4096) {
        _proj1.wrappedValue = Linear(patchSize * patchSize * inChannels, pcaDim, bias: false)
        _proj2.wrappedValue = Linear(pcaDim, embedDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj2(proj1(x))
    }
}

// MARK: - FinalLayer (final_layer2)

/// 输出头：linear(4096→3072)，将 hidden 映射回 patch 空间
final class HiDreamFinalLayer: Module, UnaryLayer {

    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int = 4096, patchSize: Int = 32, outChannels: Int = 3) {
        _linear.wrappedValue = Linear(hiddenSize, patchSize * patchSize * outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}



