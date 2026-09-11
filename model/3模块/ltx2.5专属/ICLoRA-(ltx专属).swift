// ICLoRA.swift — LTX-2.5 IC-LoRA（Pixel Spatial Upscaler x2）旁路注入
//
// 官方 IC-LoRA 是纯权重增量（LoRA），用法即「把 LoRA 融合进 transformer 对应线性层」。
// 本地主干是 4bit 量化（QuantizedLinear），int4 权重无法直接叠加 bf16 增量，
// 因此采用**旁路注入**：h = W·x + (x·Aᵀ)·Bᵀ，数值上与融合后权重 W' = W + B@A 等价。
//
// 权重事实（ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors）：
//   · 48 层全覆盖，每层 10 组 A/B（20 键），全部为视频流：
//       attn1/attn2.to_q / to_k / to_v / to_out.0（4096→4096）
//       ff.net.0.proj（4096→16384）/ ff.net.2（16384→4096）
//   · 无 to_gate_logits、无音频流、无 AV 交叉、无 patchify 注入
//   · rank=32；A [rank,in]，B [out,rank]（键名 lora_A.weight / lora_B.weight）
//   · metadata 无 scale 字段；官方模型卡 scale=1.0 且权重预缩放 → 直接相加，不乘系数
//   · dtype f16（327MB），attach 时统一 cast 到 PrecisionPolicy.defaultMainDType
//
// 键前缀 diffusion_model.transformer_blocks.N.<suffix>，
// 与主干 strippedW（剥 transformer. 后）的 transformer_blocks.N.* 一一对应。
//
// 注入开关：旁路引用 attach 到 attention/FFN 后，由 LTXVideoDiT.setICActive(_:) 全局启停。
// 原生 LTX 4.5 段 icActive=false，LoRA 零参与；仅 IC 采样段置 true，共享 DiT 缓存不互相污染。

import Foundation
@preconcurrency import MLX

/// LoRA 权重对：delta(x) = (x·Aᵀ)·Bᵀ（数值等价 W' = W + B@A 的秩分解增量）
struct ICLoRAPair {
    let a: MLXArray?   // [rank, in]
    let b: MLXArray?   // [out, rank]
    var isPresent: Bool { a != nil && b != nil }

    /// 对输入 x（特征维在最后）应用旁路增量；A/B 需已 cast 到 x.dtype。
    func delta(_ x: MLXArray) -> MLXArray? {
        guard let a, let b else { return nil }
        // x [.., in] · a.T [in, rank] → [.., rank]；再 · b.T [rank, out] → [.., out]
        return x.matmul(a.transposed()).matmul(b.transposed())
    }
}

/// 单层 IC-LoRA（10 组，视频流 attn1/attn2 + FFN，键名与主干层内结构一一对应）
final class ICLoRABlock {
    let layerIndex: Int
    // attn1：视频自注意力（Q=KV=video）
    let attn1q: ICLoRAPair
    let attn1k: ICLoRAPair
    let attn1v: ICLoRAPair
    let attn1o: ICLoRAPair
    // attn2：视频-文本交叉注意力（Q=video，KV=text）
    let attn2q: ICLoRAPair
    let attn2k: ICLoRAPair
    let attn2v: ICLoRAPair
    let attn2o: ICLoRAPair
    // FFN：net.0.proj = proj_in（4096→16384），net.2 = proj_out（16384→4096）
    let ffIn: ICLoRAPair
    let ffOut: ICLoRAPair

    init(layerIndex: Int,
         attn1q: ICLoRAPair, attn1k: ICLoRAPair, attn1v: ICLoRAPair, attn1o: ICLoRAPair,
         attn2q: ICLoRAPair, attn2k: ICLoRAPair, attn2v: ICLoRAPair, attn2o: ICLoRAPair,
         ffIn: ICLoRAPair, ffOut: ICLoRAPair) {
        self.layerIndex = layerIndex
        self.attn1q = attn1q; self.attn1k = attn1k; self.attn1v = attn1v; self.attn1o = attn1o
        self.attn2q = attn2q; self.attn2k = attn2k; self.attn2v = attn2v; self.attn2o = attn2o
        self.ffIn = ffIn; self.ffOut = ffOut
    }
}

/// 从官方 IC-LoRA safetensors 解析 48 层旁路权重（剥 diffusion_model. 前缀，
/// 按 lora_A/lora_B 配对，cast 到默认主 dtype）。
/// - Returns: 长度 48 的数组（layerIndex 0..47）；解析/文件失败返回 nil。
func loadICLoRABlocks(path: String) -> [ICLoRABlock]? {
    guard FileManager.default.fileExists(atPath: path),
          let arrays = try? MLX.loadArrays(url: URL(fileURLWithPath: path)) else {
        return nil
    }
    let dtype = PrecisionPolicy.defaultMainDType
    func pair(_ base: String) -> ICLoRAPair {
        let a = arrays["\(base).lora_A.weight"]?.asType(dtype)
        let b = arrays["\(base).lora_B.weight"]?.asType(dtype)
        return ICLoRAPair(a: a, b: b)
    }
    var blocks: [ICLoRABlock] = []
    blocks.reserveCapacity(48)
    for layer in 0..<48 {
        let p = "transformer_blocks.\(layer)"
        blocks.append(ICLoRABlock(
            layerIndex: layer,
            attn1q: pair("\(p).attn1.to_q"), attn1k: pair("\(p).attn1.to_k"),
            attn1v: pair("\(p).attn1.to_v"), attn1o: pair("\(p).attn1.to_out.0"),
            attn2q: pair("\(p).attn2.to_q"), attn2k: pair("\(p).attn2.to_k"),
            attn2v: pair("\(p).attn2.to_v"), attn2o: pair("\(p).attn2.to_out.0"),
            ffIn: pair("\(p).ff.net.0.proj"), ffOut: pair("\(p).ff.net.2")))
    }
    return blocks
}

/// IC-LoRA 常驻缓存（权重 327MB，避免每次采样重复读盘/解析）
enum ICLoRACache {
    static var blocks: [ICLoRABlock]?
    static var loadedPath: String = ""
    static var loadedModTime: Date?

    /// 返回与文件匹配的已解析权重；未加载或文件变化则重新解析。
    static func blocks(for path: String) -> [ICLoRABlock]? {
        let mod = FileManager.default.fileModTime(path)
        if let b = blocks, loadedPath == path, loadedModTime == mod {
            return b
        }
        guard let b = loadICLoRABlocks(path: path) else { return nil }
        blocks = b
        loadedPath = path
        loadedModTime = mod
        return b
    }
}

// MARK: - 主干旁路附着与开关

extension LTXVideoDiT {
    /// 将 48 层 IC-LoRA 旁路引用挂到 transformer_blocks 对应 attn/ff（幂等，可重复调用）。
    /// 挂载不改变计算（各层 icActive 默认 false，原生路径零参与）。
    func attachICLoRA(_ blocks: [ICLoRABlock]) {
        guard blocks.count == transformer_blocks.count else { return }
        for (idx, block) in transformer_blocks.enumerated() {
            let lora = blocks[idx]
            block.attn1.icLoraQ = lora.attn1q
            block.attn1.icLoraK = lora.attn1k
            block.attn1.icLoraV = lora.attn1v
            block.attn1.icLoraO = lora.attn1o
            block.attn2.icLoraQ = lora.attn2q
            block.attn2.icLoraK = lora.attn2k
            block.attn2.icLoraV = lora.attn2v
            block.attn2.icLoraO = lora.attn2o
            block.ff.icLoraIn = lora.ffIn
            block.ff.icLoraOut = lora.ffOut
        }
    }

    /// 全局启停 IC-LoRA 旁路注入：仅 IC 采样段置 true，原生 LTX 段置 false。
    func setICActive(_ active: Bool) {
        for block in transformer_blocks {
            block.attn1.icActive = active
            block.attn2.icActive = active
            block.ff.icActive = active
        }
    }
}

// MARK: - 默认权重路径

/// 官方 IC-LoRA（Pixel Spatial Upscaler x2）权重默认位置；
/// 可通过环境变量 LTX_IC_LORA_PATH 覆盖。
let ICLoRADefaultPath: String = NSString(string: "~/Downloads/ltx2.5/ic_lora/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors")
    .expandingTildeInPath
