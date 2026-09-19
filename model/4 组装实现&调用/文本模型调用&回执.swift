//
//  文本模型调用&回执.swift
//  无限画布
//
//  Created by 花茶鱼i on 2026/8/19.
//
//  ============================================================
//  文件作用：Gemma-4-12B-LTX 文本对话调用入口。
//  复用 TextEncoderCache 中已加载的 tokenizer 与 Gemma 权重（常驻缓存），
//  通过 Gemma4TextEncoder.generate 做自回归生成（lm_head 为 tied embedding）。
//  依赖：ltx2.5/Gemma4Text.swift、视频模型调用&回执.swift（TextEncoderCache / pipelineLog）
//  ============================================================
//

import Foundation
@preconcurrency import MLX
@preconcurrency import MLXNN
import Hub
import Tokenizers

// MARK: - Gemma 对话（自回归生成）

/// 与 Gemma-4-12B-LTX 对话：返回生成文本。
/// - Parameters:
///   - prompt: 用户输入文本
///   - maxNewTokens: 最大生成 token 数
///   - temperature: 采样温度（>0；1 为纯随机，越小越确定）
///   - topK: top-k 采样候选数（0 表示关闭）
/// - Returns: 生成文本；加载失败时返回错误说明。
func chatWithGemma(
    _ prompt: String,
    maxNewTokens: Int = 96,
    temperature: Float = 1.0,
    topK: Int = 64
) async -> String {
    let modelDir = CommonPaths.gemmaDir
    let gemmaPath = "\(modelDir)/model.safetensors"
    let t0 = Date()

    // tokenizer（常驻缓存，换模型目录才重载）
    let tokenizer: any Tokenizer
    if let cached = TextEncoderCache.shared.tokenizer(for: modelDir) {
        tokenizer = cached
    } else {
        let config = LanguageModelConfigurationFromHub(
            modelFolder: URL(fileURLWithPath: modelDir))
        guard let tokenizerConfig = try? await config.tokenizerConfig,
              let tokenizerData = try? await config.tokenizerData else {
            return "（tokenizer 加载失败，请检查模型目录）"
        }
        let t = try! AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        TextEncoderCache.shared.storeTokenizer(t, dir: modelDir)
        tokenizer = t
    }

    // Gemma 12B 权重（常驻缓存，权重文件未变则复用）
    let gemmaModel: Gemma4TextEncoder
    if let cached = TextEncoderCache.shared.gemma(for: gemmaPath) {
        gemmaModel = cached
    } else {
        guard let gemmaWeights = try? MLX.loadArrays(url: URL(fileURLWithPath: gemmaPath)) else {
            return "（Gemma 权重加载失败，请检查模型文件）"
        }
        var strippedG: [String: MLXArray] = [:]
        for (k, v) in gemmaWeights {
            strippedG[k.hasPrefix("model.language_model.") ? String(k.dropFirst("model.language_model.".count)) : k] = v
        }
        let m = Gemma4TextEncoder()
        let gemmaNested = NestedDictionary<String, MLXArray>.unflattened(strippedG)
        m.update(parameters: gemmaNested)
        TextEncoderCache.shared.storeGemma(m, path: gemmaPath)
        gemmaModel = m
        pipelineLog("✅ Gemma 权重灌入完成（\(Int(Date().timeIntervalSince(t0)))s）")
    }

    // encode：按官方 chat template 组装（Gemma 4 canonical），
    // 格式：<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    // 不用 addSpecialTokens=false 裸编码，模型训练输入带 turn 标记才能正确对话。
    let rawIds = tokenizer.encode(
        text: "<bos><|turn>user\n\(prompt)<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        addSpecialTokens: false)
    var ids = rawIds.map { Int32($0) }
    if ids.isEmpty { ids = [Gemma4Config.padTokenID] }

    // 自回归生成：对齐官方 generation_config（temperature=1.0, top_k=64, top_p=0.95），
    // EOS 三件套（<eos>=1, <turn|>=106, <|tool_response>=50），抑制多模态结束标记。
    let outIds = gemmaModel.generate(
        ids, maxNewTokens: maxNewTokens,
        temperature: temperature, topK: topK, topP: 0.95,
        eosTokenIDs: [1, 106, 50],
        suppressTokenIDs: [258880, 258881, 258882, 258883, 258884])
    let genIds = Array(outIds.dropFirst(ids.count))
    // 剔除 prompt 尾部未闭合的 thought 通道标记
    var cleaned = genIds
    while cleaned.last == 100 || cleaned.last == 101 { cleaned.removeLast() }
    let text = tokenizer.decode(tokens: cleaned.map(Int.init), skipSpecialTokens: true)
    pipelineLog("✅ Gemma 对话完成：prompt \(ids.count) token，生成 \(genIds.count) token（\(Int(Date().timeIntervalSince(t0)))s）")
    return text
}

