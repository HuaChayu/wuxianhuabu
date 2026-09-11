//
//  网络算子-位置编码.swift
//  无限画布 — Qwen3VL 系 M-RoPE 位置索引（HiDream 提升，方案 c 类）
//
//  [提升] 自 HiDream-O1-Image/HiDreamPipeline.swift：hidreamGetRopeIndexFixPoint / hidreamGetRopeIndexFixPointMulti
//  （py get_rope_index_fix_point 移植，B=1；token id 硬编码 151_652/151_655、fixPoint 参数化待后续）。
//  [提升] 自 HiDream-O1-Image/HiDreamBackbone.swift：HiDreamLanguage.getRopeIndex
//  （extension 原样保留命名空间，签名不变；依赖 MLXLMCommon.THW）。
//  未来可参数化 visionStartTokenID/imageTokenID/fixPoint 后作为 Qwen3VL 系通用 MRoPE 索引。
//

import MLX
import MLXLMCommon


// MARK: - rope fix-point（py get_rope_index_fix_point 移植，B=1）

func hidreamGetRopeIndexFixPoint(
    inputIdsPad: [Int32],
    imageGridTHW: (t: Int, h: Int, w: Int),
    spatialMergeSize: Int,
    skipVisionStartToken: [Int],
    fixPoint: Int
) -> MLXArray {
    let S = inputIdsPad.count

    // 找第一个 image_token 位置（ed）：vision_start 之后第一个 image_token_id
    var ed = S
    var foundVisionStart = false
    for (i, tok) in inputIdsPad.enumerated() {
        if tok == 151_652 { foundVisionStart = true; continue }
        if foundVisionStart, tok == 151_655 {
            ed = i
            break
        }
    }

    let (t, h, w) = imageGridTHW
    let mergedH = h / spatialMergeSize
    let mergedW = w / spatialMergeSize

    // 文本段长度：ed 前去掉 vision_start 占位
    let skip = skipVisionStartToken.first ?? 0
    let textLen = max(ed - 0 - skip, 0)

    // 文本位置 0..textLen-1（三通道广播）
    let textPos = MLXArray(Array(0 ..< textLen).map { Int32($0) })
        .expandedDimensions(axis: 0)          // [1, textLen]
    let textPos3 = broadcast(textPos, to: [3, textLen])          // [3, textLen]

    // 图像段：t_index / h_index / w_index
    let tArr = MLXArray.zeros([mergedH * mergedW], dtype: .int32)   // t=1 → 全 0
    let hArr = MLXArray(Array(0 ..< mergedH).map { Int32($0) })
        .expandedDimensions(axis: 1)                                // [h, 1]
    let hArr2 = broadcast(hArr, to: [mergedH, mergedW])             // 每行重复 w
        .reshaped([mergedH * mergedW])
    let wArr = MLXArray(Array(0 ..< mergedW).map { Int32($0) })
        .expandedDimensions(axis: 0)                                // [1, w]
    let wArr2 = broadcast(wArr, to: [mergedH, mergedW])             // 每列重复 h
        .reshaped([mergedH * mergedW])

    let gridStack = stacked([tArr, hArr2, wArr2])                   // [3, h*w]
    let visionPos = gridStack + MLXArray(Int32(fixPoint))

    let llmPositions = concatenated([textPos3, visionPos], axis: 1) // [3, S]

    // position_ids 初始 zeros，全量填充
    var positionIds = MLXArray.zeros([3, 1, S], dtype: .int32)
    positionIds[0..., 0, 0...] = llmPositions
    return positionIds
}


// MARK: - 多段 rope fix-point（py get_rope_index_fix_point 移植，B=1）

/// 与 py get_rope_index_fix_point 对齐的多视觉段版本：
/// 视觉段按 input_ids_pad 中 vision_start 出现顺序与 imageGridTHW 一一对应；
/// skipVisionStartToken[i]==1 时该段首 token（vision_start）并入视觉段并从 fixPoint 起算。
func hidreamGetRopeIndexFixPointMulti(
    inputIdsPad: [Int32],
    imageGridTHW: [(t: Int, h: Int, w: Int)],
    spatialMergeSize: Int,
    skipVisionStartToken: [Int],
    fixPoint: Int
) -> MLXArray {
    let S = inputIdsPad.count
    let visionStart = Int32(151_652)
    let visionStarts = inputIdsPad.enumerated().filter { $0.element == visionStart }.map { $0.offset }

    var llmPosIds: [[[Int32]]] = []   // 每段 [3][len]
    var st = 0
    var lastMax = -1
    var localFixPoint = fixPoint

    for (segIdx, vs) in visionStarts.enumerated() {
        guard segIdx < imageGridTHW.count else {
            return MLXArray.zeros([3, 1, S], dtype: .int32)
        }
        let grid = imageGridTHW[segIdx]
        let llmT = grid.t
        let llmH = grid.h / spatialMergeSize
        let llmW = grid.w / spatialMergeSize
        let visionLen = llmT * llmH * llmW
        let edImage = vs + 1   // HiDream 视觉段：vision_start 后紧跟 image_pad 序列
        let skip = segIdx < skipVisionStartToken.count ? skipVisionStartToken[segIdx] : 0
        let textLen = max(edImage - st - skip, 0)
        let stIdx = lastMax + 1

        // 文本段（[st, ed-skip)，skip 掉 vs 本身）
        if textLen > 0 {
            var tPos = [Int32](), hPos = [Int32](), wPos = [Int32]()
            for k in 0 ..< textLen {
                let v = Int32(stIdx + k)
                tPos.append(v); hPos.append(v); wPos.append(v)
            }
            llmPosIds.append([tPos, hPos, wPos])
            lastMax = stIdx + textLen - 1
        }

        // 视觉段网格（行优先：t 外、h 中、w 内）
        var tIndex = [Int32](), hIndex = [Int32](), wIndex = [Int32]()
        for ti in 0 ..< llmT {
            for hi in 0 ..< llmH {
                for wi in 0 ..< llmW {
                    tIndex.append(Int32(ti))
                    hIndex.append(Int32(hi))
                    wIndex.append(Int32(wi))
                }
            }
        }
        let vLen = tIndex.count
        let baseOffset: Int
        if skip > 0 {
            if localFixPoint > 0 {
                localFixPoint -= stIdx
            }
            baseOffset = localFixPoint + stIdx
            localFixPoint = 0
        } else {
            baseOffset = textLen + stIdx
        }
        var tPos2 = [Int32](), hPos2 = [Int32](), wPos2 = [Int32]()
        for k in 0 ..< vLen {
            let off = Int32(baseOffset)
            tPos2.append(tIndex[k] + off)
            hPos2.append(hIndex[k] + off)
            wPos2.append(wIndex[k] + off)
        }
        llmPosIds.append([tPos2, hPos2, wPos2])
        lastMax = max(lastMax, baseOffset + vLen - 1)

        st = edImage + visionLen
    }

    // 尾部文本
    if st < S {
        let stIdx = lastMax + 1
        let textLen = S - st
        var tPos = [Int32](), hPos = [Int32](), wPos = [Int32]()
        for k in 0 ..< textLen {
            let v = Int32(stIdx + k)
            tPos.append(v); hPos.append(v); wPos.append(v)
        }
        llmPosIds.append([tPos, hPos, wPos])
    }

    // 拼接 → [3, S]
    var tAll = [Int32](), hAll = [Int32](), wAll = [Int32]()
    for seg in llmPosIds {
        tAll.append(contentsOf: seg[0])
        hAll.append(contentsOf: seg[1])
        wAll.append(contentsOf: seg[2])
    }
    let totalLen = tAll.count
    var positionIds = MLXArray.zeros([3, 1, S], dtype: .int32)
    if totalLen == S {
        let llmPositions = stacked([MLXArray(tAll), MLXArray(hAll), MLXArray(wAll)])
        positionIds[0..., 0, 0...] = llmPositions
    }
    return positionIds
}


extension HiDreamLanguage {

    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            let positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            let zeros = MLXArray.zeros([batchSize], dtype: .int32)
            return (positionIds3D, zeros)
        }

        positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIds .== MLXArray(visionStartTokenId))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIds))
            let visionStartIdx = argMax(visionStartWeighted).item(Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()).item(
                Int.self)
            let videoNums = ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()).item(
                Int.self)

            let inputTokens = batchInputIds.asArray(Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIdsList.isEmpty {
                let llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIds = newPositions

                let maxPosId = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }
}
