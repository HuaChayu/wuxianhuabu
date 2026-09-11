//
//  视频尺寸入口-通用.swift
//  无限画布 — 跨模型复用的生成目标尺寸入口计算
//
//  ============================================================
//  作用：集中存放"生成目标尺寸入口"计算——比例 × 档位 → 宽高对齐 64 倍数。
//  共用范围（以实测调用点为准）：
//    1) UI 收集阶段（startVideoGeneration）：全部视频模型统一经此处取尺寸，
//       LTX-2.5 与 MiniMax H3 均在此入口取尺寸（与具体模型无关）。
//    2) LTX-2.5 管线兜底（generateVideoTest，I2V 未显式传尺寸时）复用 videoSize。
//    3) 图像侧（HiDream-O1 的 imageResolution）复用本文件的短边对齐内核
//       sizeAlignMultiple / alignedSize，与视频 p720/p1080 档同源。
//  说明：MiniMax H3 管线内部（runH3VideoPipeline）不使用本文件函数，其直接
//       使用 UI 阶段写入 task 的 videoWidth/videoHeight，仅做 ÷2 与 32 对齐。
//  [迁移] 视频尺寸入口自 4 组装实现&调用/ltx2.5-(ltx专属).swift 抽出（原位于
//         该文件"视频生成尺寸对齐公共函数"段，与被 LTX 独占的逻辑混放）。
//  ============================================================

import Foundation

// MARK: - 尺寸对齐内核（视频 / 图像共用）

/// 生成目标尺寸的对齐粒度：64 倍数。
/// 空间升频器 ×2 已接入 VAE 解码前：生成尺寸 = 目标 / 2，除 2 后须仍为 32 倍数
/// （latent 对齐），故目标尺寸必须以 64 倍数对齐，除 2 后天然满足。
func sizeAlignMultiple() -> Int { 64 }

/// 视频生成 latent 尺寸对齐倍数：LTX-2.5 要求宽高为 32 的倍数。
func videoLatentAlignMultiple() -> Int { 32 }

/// 短边向上取到 64 倍数（如 720→768、1080→1088）。
func alignedShortSide(_ raw: Int) -> Int {
    let m = sizeAlignMultiple()
    return max(m, Int(ceil(Double(raw) / Double(m))) * m)
}

/// 目标尺寸内核：给定"短边基准值"与画布比例，长边按比例缩放后取最近 64 倍数。
/// - 短边直接采用传入值（须已是 64 倍数）；
/// - 长边 = round(短边 × 长边比 / 64) × 64，且不小于对齐粒度。
/// 视频 720p / 1080p 档与图像全部档位共用本内核（行为与原两份独立实现完全等价）。
func alignedSize(shortSide: Int, ratio: CanvasStore.Ratio) -> (width: Int, height: Int) {
    let m = sizeAlignMultiple()
    let horizontal = ratio.width >= ratio.height
    let longRatio = horizontal ? ratio.width / ratio.height : ratio.height / ratio.width
    let long = max(m, Int((Double(shortSide) * longRatio / Double(m)).rounded()) * m)
    return horizontal ? (max(long, shortSide), shortSide) : (shortSide, max(long, shortSide))
}

// MARK: - 视频生成尺寸入口

/// 视频生成目标尺寸（动态表）：比例 × 清晰度档位 → 宽高对齐 64 倍数。
/// - standard：最长边 512 基准，等比缩放后每维取最近 64 倍数
/// - p720 / p1080：短边对齐档位（720→768、1080→1088，向上取 64 倍数），长边按比例取最近 64 倍数
///   （短边向上取整与长边对齐统一委托 alignedSize）
func videoResolution(for ratio: CanvasStore.Ratio, quality: VideoQuality) -> (width: Int, height: Int) {
    let m = sizeAlignMultiple()
    switch quality {
    case .standard:
        let horizontal = ratio.width >= ratio.height
        let unit = min(ratio.width, ratio.height)
        let long = max(ratio.width, ratio.height)
        let longSide = m * 8   // 512
        let shortSide = max(m, Int((Double(longSide) * unit / long / Double(m)).rounded()) * m)
        return horizontal ? (longSide, shortSide) : (shortSide, longSide)
    case .p720:
        return alignedSize(shortSide: alignedShortSide(720), ratio: ratio)
    case .p1080:
        return alignedSize(shortSide: alignedShortSide(1080), ratio: ratio)
    }
}

/// 图片 → 动态表中最接近的比例（按宽高比距离取最近，如 1000×1500 的 2:3 命中 3:4）。
func nearestRatio(for width: Int, height: Int) -> CanvasStore.Ratio {
    guard width > 0, height > 0 else { return .ratio16_9 }
    let imgRatio = Double(width) / Double(height)
    return CanvasStore.Ratio.allCases.min { a, b in
        abs(a.width / a.height - imgRatio) < abs(b.width / b.height - imgRatio)
    } ?? .ratio16_9
}

/// 视频生成目标尺寸：有图按图最接近比例查表；无图按传入比例（nil → 16:9）。
/// 供 UI 收集阶段（startVideoGeneration）与管线兜底共用。
func videoSize(imageWidth: Int? = nil, imageHeight: Int? = nil, ratio: CanvasStore.Ratio? = nil, quality: VideoQuality = .standard) -> (width: Int, height: Int) {
    let r: CanvasStore.Ratio
    if let imageWidth, let imageHeight {
        r = nearestRatio(for: imageWidth, height: imageHeight)
    } else {
        r = ratio ?? .ratio16_9
    }
    return videoResolution(for: r, quality: quality)
}
