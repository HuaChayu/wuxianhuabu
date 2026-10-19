import Foundation
import CoreML

/// Real-ESRGAN CoreML 模型封装（x2plus 系列）。
///
/// 参考实现：~/Downloads/real-esrgan-coreml-main/upscale.py
/// 核心策略与 Python 版完全一致：
/// - 模型输入为固定 522x522 的 NCHW float32（512 tile + 10px reflect pre-pad）
/// - 任意尺寸图片按 512 tile + 32 overlap 切块，反射补边后逐 tile 推理
/// - 输出按 overlap 线性斜坡加权融合，消除 tile 边界伪影
/// - 全部推理跑 GPU（RRDBNet 不被 ANE 支持，ComputeUnit 用 ALL 即可）
public final class RealESRGANCoreML {

    public struct TileConfig {
        public var tileSize: Int      // 512
        public var tileOverlap: Int   // 32
        public var prePad: Int        // 10
        public var scale: Int         // 2

        public init(tileSize: Int = 512, tileOverlap: Int = 32, prePad: Int = 10, scale: Int = 2) {
            self.tileSize = tileSize
            self.tileOverlap = tileOverlap
            self.prePad = prePad
            self.scale = scale
        }
    }

    public let config: TileConfig

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let modelSize: Int   // 模型固定输入边长（522）

    public init(modelURL: URL,
                config: TileConfig = TileConfig(),
                computeUnits: MLComputeUnits = .all) throws {
        self.config = config
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        // .mlpackage 需先编译为 .mlmodelc 再加载；编译产物缓存到 Caches 避免每次重复编译
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RealESRGAN_compiled", isDirectory: true)
        let compiledURL = cacheDir.appendingPathComponent("RealESRGAN_x2plus_522_fp16.mlmodelc")
        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
            let tmp = try MLModel.compileModel(at: modelURL)
            try FileManager.default.copyItem(at: tmp, to: compiledURL)
        }
        self.model = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        // 动态读取输入/输出名与模型边长，避免硬编码
        let desc = model.modelDescription
        guard let inputDesc = desc.inputDescriptionsByName.first else {
            throw NSError(domain: "RealESRGANCoreML", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "模型没有输入"])
        }
        guard let outputDesc = desc.outputDescriptionsByName.first else {
            throw NSError(domain: "RealESRGANCoreML", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "模型没有输出"])
        }
        inputName = inputDesc.key
        outputName = outputDesc.key

        let shape = inputDesc.value.multiArrayConstraint?.shape ?? []
        guard shape.count >= 3, let side = shape.last, side.intValue > 0 else {
            throw NSError(domain: "RealESRGANCoreML", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "无法解析模型输入尺寸"])
        }
        modelSize = side.intValue
    }

    // MARK: - 单帧超分

    /// 对一帧 RGB（HWC, float32 [0,1], 行优先）做 scale 倍超分。
    /// 返回超分后的 RGB（HWC, float32 [0,1]）与输出宽高。
    public func upscale(rgb: [Float], width: Int, height: Int) throws -> (rgb: [Float], width: Int, height: Int) {
        let s = config.scale
        let outW = width * s
        let outH = height * s

        // 单 tile 直接处理
        if height <= config.tileSize && width <= config.tileSize {
            let tile = try predictTile(rgb: rgb, w: width, h: height,
                                       y0: 0, x0: 0, th: height, tw: width)
            return (tile, outW, outH)
        }

        // 多 tile：切块 -> 推理 -> overlap 加权融合
        let yStarts = computeTileStarts(total: height)
        let xStarts = computeTileStarts(total: width)

        var output = [Float](repeating: 0, count: outW * outH * 3)
        var weightMap = [Float](repeating: 0, count: outW * outH)

        let rampPixels = config.tileOverlap * s

        for y0 in yStarts {
            for x0 in xStarts {
                let y1 = min(y0 + config.tileSize, height)
                let x1 = min(x0 + config.tileSize, width)
                let th = y1 - y0
                let tw = x1 - x0

                let tileOut = try predictTile(rgb: rgb, w: width, h: height,
                                              y0: y0, x0: x0, th: th, tw: tw)

                // overlap 线性斜坡权重（与 Python _blend_weight 一致）
                var twt = [Float](repeating: 1, count: th * s * tw * s)
                if rampPixels > 0 {
                    var ramp = [Float](repeating: 0, count: rampPixels)
                    for i in 0..<rampPixels { ramp[i] = Float(i) / Float(rampPixels - 1) }

                    if y0 > 0 {
                        for r in 0..<rampPixels {
                            for c in 0..<(tw * s) {
                                twt[r * (tw * s) + c] *= ramp[r]
                            }
                        }
                    }
                    if y1 < height {
                        for r in 0..<rampPixels {
                            let rr = rampPixels - 1 - r
                            for c in 0..<(tw * s) {
                                twt[(th * s - rampPixels + r) * (tw * s) + c] *= ramp[rr]
                            }
                        }
                    }
                    if x0 > 0 {
                        for r in 0..<(th * s) {
                            for c in 0..<rampPixels {
                                twt[r * (tw * s) + c] *= ramp[c]
                            }
                        }
                    }
                    if x1 < width {
                        for r in 0..<(th * s) {
                            for c in 0..<rampPixels {
                                let cc = rampPixels - 1 - c
                                twt[r * (tw * s) + (tw * s - rampPixels + c)] *= ramp[cc]
                            }
                        }
                    }
                }

                let oy0 = y0 * s, ox0 = x0 * s
                for r in 0..<(th * s) {
                    for c in 0..<(tw * s) {
                        let srcIdx = r * (tw * s) + c
                        let dstIdx = (oy0 + r) * outW + (ox0 + c)
                        let w = twt[srcIdx]
                        for ch in 0..<3 {
                            output[dstIdx * 3 + ch] += tileOut[srcIdx * 3 + ch] * w
                        }
                        weightMap[dstIdx] += w
                    }
                }
            }
        }

        // 归一化并 clamp
        for i in 0..<output.count {
            let wi = i / 3
            let inv = 1.0 / max(weightMap[wi], 1e-8)
            output[i] = min(max(output[i] * inv, 0), 1)
        }
        return (output, outW, outH)
    }

    // MARK: - Tile 推理

    /// 对 (y0,x0)-(y1,x1) 区域切 tile：reflect pre-pad -> 补到 modelSize -> NCHW 推理 -> 去 padding。
    /// 返回 th*scale x tw*scale x 3 的 HWC float32 [0,1]。
    private func predictTile(rgb: [Float], w: Int, h: Int,
                             y0: Int, x0: Int, th: Int, tw: Int) throws -> [Float] {
        let p = config.prePad
        let s = config.scale

        // 与 Python upscale.py 完全一致的两步 reflect 填充（均在右下，mode='reflect' 不重复边界）：
        //   1) tile 右下 prepad p 像素 → (th+p, tw+p)
        //   2) 再右下 pad 到 modelSize 正方形
        // reflect 映射：行 r（r>=size）的源行 = 2*size - 1 - r

        // step1: tile + prepad
        let padH1 = th + p
        let padW1 = tw + p
        var t1 = [Float](repeating: 0, count: padH1 * padW1 * 3)

        for r in 0..<th {
            let srcBase = ((y0 + r) * w + x0) * 3
            let dstBase = (r * padW1) * 3
            for c in 0..<tw {
                let s0 = srcBase + c * 3
                let d0 = dstBase + c * 3
                t1[d0] = rgb[s0]; t1[d0+1] = rgb[s0+1]; t1[d0+2] = rgb[s0+2]
            }
        }
        if p > 0 {
            for r in th..<padH1 {
                let sr = 2 * th - 1 - r
                for c in 0..<tw {
                    let s0 = (sr * padW1 + c) * 3
                    let d0 = (r * padW1 + c) * 3
                    t1[d0] = t1[s0]; t1[d0+1] = t1[s0+1]; t1[d0+2] = t1[s0+2]
                }
            }
            for r in 0..<padH1 {
                for c in tw..<padW1 {
                    let sc = 2 * tw - 1 - c
                    let s0 = (r * padW1 + sc) * 3
                    let d0 = (r * padW1 + c) * 3
                    t1[d0] = t1[s0]; t1[d0+1] = t1[s0+1]; t1[d0+2] = t1[s0+2]
                }
            }
        }

        // step2: pad 到 modelSize 正方形
        let inSize = modelSize
        let padH2 = inSize - padH1
        let padW2 = inSize - padW1
        var t2 = [Float](repeating: 0, count: inSize * inSize * 3)

        for r in 0..<padH1 {
            let s0 = (r * padW1) * 3
            let d0 = (r * inSize) * 3
            for c in 0..<padW1 {
                let si = s0 + c * 3
                let di = d0 + c * 3
                t2[di] = t1[si]; t2[di+1] = t1[si+1]; t2[di+2] = t1[si+2]
            }
        }
        if padH2 > 0 {
            for r in padH1..<inSize {
                let sr = 2 * padH1 - 1 - r
                for c in 0..<padW1 {
                    let s0 = (sr * inSize + c) * 3
                    let d0 = (r * inSize + c) * 3
                    t2[d0] = t2[s0]; t2[d0+1] = t2[s0+1]; t2[d0+2] = t2[s0+2]
                }
            }
        }
        if padW2 > 0 {
            for r in 0..<inSize {
                for c in padW1..<inSize {
                    let sc = 2 * padW1 - 1 - c
                    let s0 = (r * inSize + sc) * 3
                    let d0 = (r * inSize + c) * 3
                    t2[d0] = t2[s0]; t2[d0+1] = t2[s0+1]; t2[d0+2] = t2[s0+2]
                }
            }
        }

        // step3: HWC -> NCHW
        let input = try MLMultiArray(shape: [1, 3, NSNumber(value: inSize), NSNumber(value: inSize)],
                                     dataType: .float32)
        let inPtr = UnsafeMutablePointer<Float>(OpaquePointer(input.dataPointer))
        for ch in 0..<3 {
            for r in 0..<inSize {
                for c in 0..<inSize {
                    inPtr[(ch * inSize + r) * inSize + c] = t2[(r * inSize + c) * 3 + ch]
                }
            }
        }

        // step4: 推理
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: input])
        let pred = try model.prediction(from: provider)
        guard let outArr = pred.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "RealESRGANCoreML", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "推理输出缺失"])
        }

        let outSide = modelSize * s
        // 关键坑：此模型输出 strides=[3307392,1102464,1056,1]（GPU 行对齐 1056≠1044），
        // dataPointer 按该布局直接裸读会错位且越界（ASan 证实 heap-buffer-overflow）。
        // MLMultiArray 的线性下标按 shape C 连续展平（strides 仅是内部 hint），
        // 因此必须用下标 out[(ch*outSide+r)*outSide+c] 访问，且全程 < count。

        // step5: NCHW -> HWC，裁掉 square pad 与 prepad（都在右下）→ th*s x tw*s，clamp [0,1]
        let ow = tw * s
        var result = [Float](repeating: 0, count: th * s * ow * 3)
        for r in 0..<(th * s) {
            for c in 0..<ow {
                for ch in 0..<3 {
                    let v = outArr[(ch * outSide + r) * outSide + c].floatValue
                    result[(r * ow + c) * 3 + ch] = min(max(v, 0), 1)
                }
            }
        }
        return result
    }

    // MARK: - 工具

    /// 照抄 Python compute_tile_starts：返回一行/列上所有 tile 起始位置。
    private func computeTileStarts(total: Int) -> [Int] {
        let tile = config.tileSize
        let stride = tile - config.tileOverlap
        if total <= tile { return [0] }
        var positions: [Int] = []
        var pos = 0
        while pos < total {
            if pos + tile >= total {
                positions.append(total - tile)
                break
            }
            positions.append(pos)
            pos += stride
        }
        return positions
    }
}
