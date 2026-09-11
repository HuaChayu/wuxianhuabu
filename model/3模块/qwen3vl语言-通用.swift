// HiDream-O1-Image 骨干：Qwen3VL（vision tower + language model）移植
// 来源：mlx-swift-lm Libraries/MLXVLM/Models/Qwen3VL.swift 的 Qwen3VLVision / Qwen3VLLanguage
// 原类型为 internal，复制到本目录并加 HiDream 前缀；Qwen3VLConfiguration 复用 MLXVLM public 类型。

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

// [拆分] 自 HiDreamBackbone.swift 语言模型部分（方案 b 类，HiDream 专用）
private let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35vl.ropeDeltas")

// MARK: - Language



enum HiDreamLanguage {

    final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: MLXNN.RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: MLXNN.RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = MLXNN.RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = MLXNN.RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                mropeSection: config.ropeScaling?.mropeSection ?? [24, 20, 20])
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            rotaryCosSin: (MLXArray, MLXArray)? = nil
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask.asType(x.dtype))
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)

            return result
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MLXNN.RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MLXNN.RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = MLXNN.RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = MLXNN.RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            rotaryCosSin: (MLXArray, MLXArray)? = nil
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds,
                rotaryCosSin: rotaryCosSin)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: MLXNN.RMSNorm

        // 层间共享 rope 预计算实例：cos/sin 只依赖 positionIds 与 dtype，
        // 在层循环外算一次供全部 DecoderLayer 复用（避免每层重算 + interleave 逐列循环）。
        let ropeEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            self.ropeEmbedding = RotaryEmbedding(
                headDim: config.headDim,
                base: config.ropeTheta,
                mropeSection: config.ropeScaling?.mropeSection ?? [24, 20, 20])
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = MLXNN.RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            // rope cos/sin 只依赖 positionIds 与 dtype：循环外预计算一次，全部层复用
            let rotaryCosSin: (MLXArray, MLXArray)?
            if let positionIds {
                rotaryCosSin = ropeEmbedding(positionIds: positionIds, dtype: hidden.dtype)
            } else {
                rotaryCosSin = nil
            }

            // deepstack 注入位置索引只依赖 visualMask：循环外算一次，避免每层全量回读
            let deepstackIndices: MLXArray? = (deepstackEmbeds != nil && visualMask != nil)
                ? deepstackIndexArray(visualMask!) : nil

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(
                    hidden, mask: mask, cache: layerCache, positionIds: positionIds,
                    rotaryCosSin: rotaryCosSin)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let idxArr = deepstackIndices
                {

                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        indexArray: idxArr,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            indexArray: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds
            return result
        }

        private func deepstackIndexArray(_ mask: MLXArray) -> MLXArray? {
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [UInt32] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(UInt32(idx))
            }
            guard !indices.isEmpty else { return nil }
            return MLXArray(indices)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3VLConfiguration
        let textConfig: Qwen3VLConfiguration.TextConfiguration
        var kvHeads: [Int]

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers)

            // HiDream 权重不 tie embeddings，lm_head 始终存在（推理不经 lmHead，仅权重对齐）
            _lmHead.wrappedValue = Linear(
                config.textConfiguration.hiddenSize,
                config.textConfiguration.vocabSize,
                bias: false)
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            state: LMOutput.State?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds providedPositionIds: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?
        ) -> LMOutput {
            var state = state ?? .init()

            if pixelValues != nil {
                state[ropeDeltasKey] = nil
            }

            var positionIds = providedPositionIds

            if positionIds == nil && (mask == nil || mask?.ndim == 2) {
                if (cache?.first?.offset ?? 0) == 0 || state[ropeDeltasKey] == nil || cache == nil {
                    if let inputIds {
                        let (computed, deltas) = HiDreamLanguage.getRopeIndex(
                            inputIds: inputIds,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenIndex,
                            videoTokenId: config.videoTokenIndex,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: mask)

                        positionIds = computed
                        state[ropeDeltasKey] = deltas
                    } else if let cache, state[ropeDeltasKey] == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue

                        positionIds = base[.newAxis, 0..., 0...]
                        positionIds = tiled(positionIds!, repetitions: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas = state[ropeDeltasKey] {
                    let batch = (inputIds ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIds ?? inputEmbeddings!).dim(1)

                    let lastCacheOffset = cache.last?.offset ?? 0

                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = base[.newAxis, 0...]
                    base = broadcast(base, to: [batch, seqLength])

                    if delta.dim(0) == 1 && batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }

                    base = base + delta

                    positionIds = base[.newAxis, 0..., 0...]
                    positionIds = broadcast(positionIds!, to: [3, batch, seqLength])
                }
            }

            var output = model(
                inputIds,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIds: positionIds,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds)

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output, state: state)
        }

    }
}
