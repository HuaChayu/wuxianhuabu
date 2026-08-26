//
//  模型通用加载.swift
//  无限画布 — 通用模型权重加载方法（可跨模型复用）
//
//  ============================================================
//  作用：统一 safetensors / npy 权重加载流程，供 LTX-2.5 及未来
//  其他大模型复用：
//    · loadSafetensors：safetensors → [String: MLXArray]
//    · stripWeightsPrefix：剥离键名前缀（如 "model.language_model."）
//    · remapWeightsKeys：批量键名替换（如 ff.net.0.proj → ff.net.proj0）
//    · unflattenWeights：扁平字典 → NestedDictionary（灌入 Module.parameters）
//    · loadNpy：npy → MLXArray（支持 v1/v2 header，LE/BE 回退）
//  ============================================================

import Foundation
import MLX
import MLXNN

// MARK: - safetensors 加载

/// 从 safetensors 文件加载权重字典 [String: MLXArray]。
func loadSafetensors(_ path: String) -> [String: MLXArray]? {
    try? MLX.loadArrays(url: URL(fileURLWithPath: path))
}

/// 剥离权重键名前缀：匹配则去掉前缀，否则保留原键（兼容已无前缀文件）。
func stripWeightsPrefix(_ weights: [String: MLXArray], prefix: String) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)
    for (k, v) in weights {
        out[k.hasPrefix(prefix) ? String(k.dropFirst(prefix.count)) : k] = v
    }
    return out
}

/// 批量键名替换（用于权重键名 → 模型属性名不一致的 remap）。
/// 按出现顺序依次替换所有匹配子串。
func remapWeightsKeys(_ weights: [String: MLXArray], replacements: [(from: String, to: String)]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    out.reserveCapacity(weights.count)
    for (k, v) in weights {
        var nk = k
        for r in replacements {
            nk = nk.replacingOccurrences(of: r.from, with: r.to)
        }
        out[nk] = v
    }
    return out
}

/// 扁平权重字典 → NestedDictionary，可直接 update(parameters:) 灌入 Module。
func unflattenWeights(_ weights: [String: MLXArray]) -> NestedDictionary<String, MLXArray> {
    NestedDictionary<String, MLXArray>.unflattened(weights)
}

/// 一键组合：加载 → 剥前缀 → unflatten。返回扁平字典与嵌套字典。
func loadModelWeights(
    path: String,
    prefix: String? = nil,
    replacements: [(from: String, to: String)] = []
) -> (flat: [String: MLXArray], nested: NestedDictionary<String, MLXArray>)? {
    guard let raw = loadSafetensors(path) else { return nil }
    var w = raw
    if let prefix { w = stripWeightsPrefix(w, prefix: prefix) }
    if !replacements.isEmpty { w = remapWeightsKeys(w, replacements: replacements) }
    return (w, unflattenWeights(w))
}

// MARK: - npy 加载

/// 读取 npy 文件为 MLXArray（f32；支持 v1/v2 header，LE/BE 长度回退）。
func loadNpy(_ path: String) -> MLXArray? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
        print("❌ 无法读取 \(path)"); return nil
    }
    let bytes = [UInt8](data)
    guard bytes.count > 10,
          bytes[0] == 0x93, bytes[1] == 0x4E, bytes[2] == 0x55, bytes[3] == 0x4D,
          bytes[4] == 0x50, bytes[5] == 0x59 else {
        print("❌ 非 npy 文件"); return nil
    }
    let version = bytes[6]
    var headerStart: Int
    if version == 1 {
        headerStart = 10
    } else {
        headerStart = 12
    }
    // header 长度字段：version 后 2 字节（little-endian，Python numpy 写入；
    // MLX 某些版本写 big-endian，若越界则回退）。
    let headerLenLE = (Int(bytes[headerStart - 1]) << 8) | Int(bytes[headerStart - 2])
    let headerLenBE = (Int(bytes[headerStart - 2]) << 8) | Int(bytes[headerStart - 1])
    // 以第一个 '\n' 截取 header 文本用于解析 dict（不含 padding）
    guard let newlineIdx = bytes[headerStart...].firstIndex(of: 0x0A) else {
        print("❌ 找不到 header 结尾"); return nil
    }
    guard let header = String(data: data[headerStart ..< newlineIdx], encoding: .utf8) else {
        print("❌ header 解析失败"); return nil
    }
    // 解析 dict: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 15, 16, 16, 128), }
    guard let descrRange = header.range(of: #"'descr':\s*'([^']+)'"#, options: .regularExpression),
          let descr = header[descrRange].split(separator: "'", omittingEmptySubsequences: false).dropLast().last else {
        print("❌ 找不到 descr；header=[\(header)] 长度=\(header.count) 字节=\(bytes[headerStart..<min(newlineIdx, headerStart+120)].prefix(40))")
        return nil
    }
    guard let shapeRange = header.range(of: #"'shape':\s*\(([^)]*)\)"#, options: .regularExpression) else {
        print("❌ 找不到 shape"); return nil
    }
    let shapeStr = header[shapeRange].split(separator: "(").last!.split(separator: ")").first!
    let shape = shapeStr.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard !shape.isEmpty else { print("❌ shape 解析失败"); return nil }

    let count = shape.reduce(1, *)
    // 数据起点 = headerStart + header 长度字段；little-endian 越界则试 big-endian
    var dataStart = headerStart + headerLenLE
    if dataStart + count * 4 > bytes.count {
        dataStart = headerStart + headerLenBE
    }
    guard descr == "<f4" || descr == "|f4" else {
        print("❌ 暂不支持 dtype descr=[\(descr)] count=\(descr.count) unicode=\(descr.unicodeScalars.map { $0.value }) header=[\(header)]")
        return nil
    }
    guard dataStart + count * 4 <= bytes.count else {
        print("❌ 数据越界 dataStart=\(dataStart) need=\(count * 4) have=\(bytes.count - dataStart)"); return nil
    }
    let floats = bytes[dataStart ..< (dataStart + count * 4)].withUnsafeBytes {
        Array($0.bindMemory(to: Float.self))
    }
    print("npy: descr=\(descr) shape=\(shape) count=\(count)")
    return MLXArray(floats, shape)
}

// MARK: - 量化参数动态推断（safetensors header）

/// 量化参数推断结果缓存：key=文件标准路径，value=(mtime, 推断结果)
private var _quantParamsCache: [String: (mtime: TimeInterval, params: (bits: Int, groupSize: Int)?)] = [:]

/// 带缓存的量化参数推断：同一文件未变更时直接复用上次结果，避免重复读头。
func cachedQuantParams(from path: String) -> (bits: Int, groupSize: Int)? {
    let key = (path as NSString).standardizingPath
    let mtime = (try? FileManager.default.attributesOfItem(atPath: key)[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
    if let hit = _quantParamsCache[key], hit.mtime == mtime {
        return hit.params
    }
    let params = inferQuantParams(from: key)
    _quantParamsCache[key] = (mtime: mtime, params: params)
    return params
}

/// 从 safetensors 文件头推断量化参数（bits / groupSize），不加载权重数据。
/// 原理：量化层同时含 weight（U32 打包）与 scales：
///   weight.shape[1] = in × bits / 32，scales.shape[1] = in / groupSize
/// 两步确定：
///   1) shape 约束（w2×32/bits == s2×gs）筛候选；groupSize 仅限 MLX 支持的 32/64/128（256 会直接 fatalError）；
///   2) 反量化自洽评分：对候选采样量化层小样本，按 q×scale+bias 反量化，
///      正确参数下权重 std 不会超过 max|bias|（bias=edge≈group 内 max-abs），比值应落在 [0.05, 0.6]；
///      错误参数（bits/groupSize 错位）比值会失调到 2~50 倍。取比值最小的候选。
/// 无量化层（纯 bf16 / fp8 等）返回 nil。
func inferQuantParams(from path: String) -> (bits: Int, groupSize: Int)? {
    // 文件映射读取，只取头部，避免加载整个权重文件
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe),
          data.count > 8 else { return nil }
    let headerLen = data.subdata(in: 0..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    guard headerLen <= data.count - 8 else { return nil }
    guard let json = try? JSONSerialization.jsonObject(with: data.subdata(in: 8 ..< (8 + Int(headerLen)))) as? [String: Any] else {
        return nil
    }

    // 收集量化层样本：同前缀的 weight（U32 打包）+ scales
    var samples: [(key: String, w2: Int, s2: Int)] = []
    for (k, v) in json {
        guard let meta = v as? [String: Any],
              let shape = meta["shape"] as? [Int], shape.count == 2,
              k.hasSuffix(".weight") else { continue }
        let base = String(k.dropLast("weight".count))
        guard let sMeta = json[base + "scales"] as? [String: Any],
              let sShape = sMeta["shape"] as? [Int], sShape.count == 2 else { continue }
        samples.append((key: k, w2: shape[1], s2: sShape[1]))
        if samples.count >= 3 { break }
    }
    guard !samples.isEmpty else { return nil }

    // 候选遍历：MLX 量化支持 2/4/8 bit；groupSize 仅支持 32/64/128（256 会直接 fatalError 卡死）
    var best: (bits: Int, groupSize: Int)?
    var bestScore: Double = .infinity
    for bits in [2, 4, 8] {
        for gs in [32, 64, 128] {
            let shapeOK = samples.allSatisfy { (_, w2, s2) in
                let numer = w2 * 32
                return numer % bits == 0 && numer / bits == s2 * gs
            }
            guard shapeOK else { continue }
            // 反量化自洽评分：多采样层取平均比值，全部落在 [0.05, 0.6] 才接受
            let scores = samples.prefix(2).compactMap { (key, _, _) -> Double? in
                quantSelfConsistencyScore(data: data, json: json, key: key, bits: bits, gs: gs)
            }
            guard !scores.isEmpty, scores.allSatisfy({ $0 >= 0.05 && $0 <= 0.6 }) else { continue }
            let avg = scores.reduce(0, +) / Double(scores.count)
            if avg < bestScore {
                bestScore = avg
                best = (bits, gs)
            }
        }
    }
    return best
}

/// 对单个量化层按候选 (bits, groupSize) 反量化小样本，返回 std/max|bias| 比值。
/// 数据来自文件映射 data（含 header），json 为 header 解析结果；只读前 sampleU32 个 U32。
private func quantSelfConsistencyScore(
    data: Data, json: [String: Any], key: String,
    bits: Int, gs: Int, sampleU32: Int = 256
) -> Double? {
    guard let wMeta = json[key] as? [String: Any],
          let wOff = (wMeta["data_offsets"] as? [Int])?.first else { return nil }
    let base = String(key.dropLast("weight".count))
    guard let sMeta = json[base + "scales"] as? [String: Any],
          let sOff = (sMeta["data_offsets"] as? [Int])?.first,
          let bMeta = json[base + "biases"] as? [String: Any],
          let bOff = (bMeta["data_offsets"] as? [Int])?.first else { return nil }

    let headerLen = data.subdata(in: 0..<8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    let dataStart = 8 + Int(headerLen)
    let wStart = dataStart + wOff
    let wBytes = sampleU32 * 4
    guard wStart + wBytes <= data.count else { return nil }

    // 拆包：MLX 打包为 U32 高位在前（大端段序），逐段取 bits 位
    var qs: [Int] = []
    qs.reserveCapacity(sampleU32 * 32 / bits)
    let u32s = data.subdata(in: wStart ..< (wStart + wBytes)).withUnsafeBytes {
        Array($0.bindMemory(to: UInt32.self))
    }
    let mask = UInt32((1 << bits) - 1)
    for u in u32s {
        for start in stride(from: 0, to: 32, by: bits) {
            qs.append(Int((u >> UInt32(32 - bits - start)) & mask))
        }
    }

    let groupCount = qs.count / gs
    guard groupCount > 0 else { return nil }
    guard let scales = readHeaderFloats(data: data, at: dataStart + sOff, dtype: sMeta["dtype"] as? String, count: groupCount),
          let biases = readHeaderFloats(data: data, at: dataStart + bOff, dtype: bMeta["dtype"] as? String, count: groupCount) else { return nil }

    var sum: Double = 0
    var sumSq: Double = 0
    var n: Double = 0
    var maxBias: Float = 0
    for g in 0 ..< groupCount {
        let sc = scales[g]
        let bs = biases[g]
        maxBias = max(maxBias, abs(bs))
        let lo = g * gs
        let hi = min(lo + gs, qs.count)
        for i in lo ..< hi {
            let d = Float(qs[i]) * sc + bs
            sum += Double(d)
            sumSq += Double(d * d)
            n += 1
        }
    }
    guard n > 0, maxBias > 0 else { return nil }
    let mean = sum / n
    let variance = max(0, sumSq / n - mean * mean)
    return variance.squareRoot() / Double(maxBias)
}

/// 读取 safetensors header 内嵌浮点张量小样本（BF16 / F32），返回 Float 数组。
private func readHeaderFloats(data: Data, at offset: Int, dtype: String?, count: Int) -> [Float]? {
    let dtype = dtype ?? "BF16"
    if dtype == "F32" {
        let need = count * 4
        guard offset + need <= data.count else { return nil }
        return data.subdata(in: offset ..< (offset + need)).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }
    // BF16：UInt16 左移 16 位转 F32
    let need = count * 2
    guard offset + need <= data.count else { return nil }
    let u16s = data.subdata(in: offset ..< (offset + need)).withUnsafeBytes {
        Array($0.bindMemory(to: UInt16.self))
    }
    return u16s.map { Float(bitPattern: UInt32($0) << 16) }
}

/// 文件大小（字节）；读取失败返回 nil。
func fileSizeBytes(_ path: String) -> Int? {
    let attrs = try? FileManager.default.attributesOfItem(atPath: path)
    return (attrs?[.size] as? NSNumber)?.intValue
}
