// 临时自检：H3 video VAE 环回测试（故障域切分）。
// 运行：NA_VAETEST=1 ./无限画布
// 流程 A：encodeImage(原图) → decode → 首帧与原图对比（MSE + png）
// 流程 B：encodeImage → patchifyVideo → unpatchifyVideo → reshape → decode → 对比
import Foundation
import MLX
import CoreGraphics
import ImageIO

enum H3VAETestRun {
    static func run() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1

        let firstImage = "/Users/huachayui/Desktop/000 媒体素材/雷电法王.png"
        let lastImage = "/Users/huachayui/Desktop/蝴蝶/2.png"
        let modelDir = "\(CommonPaths.modelRoot)/minimax h3/MiniMax-H3-FL2VA-MLX-Serve-4bit"

        func log(_ s: String) { print("[H3VAETEST] \(s)"); fflush(stdout) }

        Task {
            do {
                let w = try H3Weights(url: URL(fileURLWithPath: "\(modelDir)/video_vae.safetensors"))
                w.cacheEnabled = false
                let vae = try H3VAE.load(w)

                guard let px = loadImageBCFHW(path: firstImage, width: 256, height: 256) else {
                    throw NSError(domain: "H3VAETEST", code: 1, userInfo: [NSLocalizedDescriptionKey: "图片加载失败"])
                }
                log("input pixels \(px.shape)")  // [1,3,1,256,256] [-1,1]
                h3WriteFramePNG(px, frame: 0, path: "/tmp/h3vae_input.png")

                // ── 流程 A：encode → decode ──
                let z = vae.encodeImage(px)
                MLX.eval(z)
                log("A: latent \(z.shape)")
                let outA = vae.decode(z)
                MLX.eval(outA)
                log("A: decode out \(outA.shape)")
                let mseA = meanSquare(h3Frame(outA, 0), h3Frame(px, 0))
                log("A: frame0 MSE = \(mseA)（0 表示完全还原）")
                h3WriteFramePNG(outA, frame: 0, path: "/tmp/h3vae_a_frame0.png")

                // ── 流程 B：encode → patchify → unpatchify → reshape → decode ──
                let latT = z.shape[2], latH = z.shape[3], latW = z.shape[4], latC = z.shape[1]
                let gridH = latH / 2, gridW = latW / 2
                let nhwc = z.transposed(0, 2, 3, 4, 1).reshaped([latT, latH, latW, latC])
                let rows = H3TensorOps.patchifyVideo(nhwc, gridH: gridH, gridW: gridW)
                let back = H3TensorOps.unpatchifyVideo(rows, gridH: gridH, gridW: gridW)
                let bchw = back.transposed(3, 0, 1, 2).reshaped([1, latC, latT, latH, latW])
                MLX.eval(bchw)
                let mseB = meanSquare(bchw, z)
                log("B: patchify→unpatchify MSE = \(mseB)（0 表示布局自洽）")
                let outB = vae.decode(bchw)
                MLX.eval(outB)
                let mseB2 = meanSquare(h3Frame(outB, 0), h3Frame(px, 0))
                log("B: decode frame0 MSE = \(mseB2)")
                h3WriteFramePNG(outB, frame: 0, path: "/tmp/h3vae_b_frame0.png")

                // ── 流程 C：latentT=5（17 帧）chunk 拼接验证 ──
                // 构造 5-token latent 序列（前 3 token=首帧图，后 2 token=尾帧图），
                // decode 应得到 17 帧：前几帧接近首帧、后几帧接近尾帧的渐变。
                // 若 chunk 拼接/planTemporal 有 bug，会出现网格/噪点/错位。
                guard let lastPx = loadImageBCFHW(path: lastImage, width: 256, height: 256) else {
                    throw NSError(domain: "H3VAETEST", code: 2, userInfo: [NSLocalizedDescriptionKey: "尾帧图片加载失败"])
                }
                let lastLat = vae.encodeImage(lastPx)
                MLX.eval(lastLat)
                log("C: firstLat \(z.shape) lastLat \(lastLat.shape)")
                // [1,24,1,16,16] → 拼成 [1,24,5,16,16]
                let tokSeq = MLX.concatenated([z, z, z, lastLat, lastLat], axis: 2)
                MLX.eval(tokSeq)
                log("C: 5-token latent \(tokSeq.shape)")
                let outC = vae.decode(tokSeq)
                MLX.eval(outC)
                log("C: decode out \(outC.shape)")
                let frames = outC.shape[2]
                var frameMSEs: [Float] = []
                for fi in [0, 4, 8, 12, 16] where fi < frames {
                    let mseF = meanSquare(h3Frame(outC, fi), h3Frame(px, 0))
                    let mseL = meanSquare(h3Frame(outC, fi), h3Frame(lastPx, 0))
                    frameMSEs.append(mseF)
                    log(String(format: "C: frame %d  vs首帧MSE=%.5f  vs尾帧MSE=%.5f", fi, mseF, mseL))
                    h3WriteFramePNG(outC, frame: fi, path: String(format: "/tmp/h3vae_c_f%d.png", fi))
                }
                // 全尾帧 token 的对照：decode 输出应全接近尾帧
                let allLast = MLX.concatenated([lastLat, lastLat, lastLat, lastLat, lastLat], axis: 2)
                let outD = vae.decode(allLast)
                MLX.eval(outD)
                let mseD0 = meanSquare(h3Frame(outD, 0), h3Frame(lastPx, 0))
                let mseD16 = meanSquare(h3Frame(outD, 16), h3Frame(lastPx, 0))
                log(String(format: "C: 全尾帧token frame0 MSE=%.5f frame16 MSE=%.5f（应≈0.x 且一致）", mseD0, mseD16))
                h3WriteFramePNG(outD, frame: 0, path: "/tmp/h3vae_d_f0.png")
                h3WriteFramePNG(outD, frame: 16, path: "/tmp/h3vae_d_f16.png")

                code = 0
            } catch {
                log("❌ \(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 1200)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    private static func h3Frame(_ px: MLXArray, _ f: Int) -> MLXArray {
        let h = px.shape[3], w = px.shape[4]
        return px[0..<1, 0..<3, f..<(f + 1), 0..<h, 0..<w]
    }

    private static func meanSquare(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = ((a.asType(.float32) - b.asType(.float32)) * (a.asType(.float32) - b.asType(.float32))).mean()
        MLX.eval(d)
        return d.item()
    }
}

/// 把 [1,3,T,H,W]（[-1,1] f32）第 frame 帧写成 png（正立）。
func h3WriteFramePNG(_ px: MLXArray, frame: Int, path: String) {
    let h = px.shape[3], w = px.shape[4]
    let f = px[0..<1, 0..<3, frame..<(frame + 1), 0..<h, 0..<w].reshaped([3, h, w])
    MLX.eval(f)
    let floats = f.asArray(Float.self)
    let hw = h * w
    guard floats.count == 3 * hw,
          let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: w * 4,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
    let data = ctx.data!.bindMemory(to: UInt8.self, capacity: h * w * 4)
    for y in 0..<h {
        let dstY = y   // CGContext 位图内存 row0 = 顶部，直接顺序写即正立（与 loadImageBCFHW 读取方向一致）；此前 dstY=h-1-y 曾导致写出的参考帧 PNG 上下颠倒
        for x in 0..<w {
            let o = y * w + x
            let r = max(0, min(255, Int((floats[0 * hw + o] * 0.5 + 0.5) * 255)))
            let g = max(0, min(255, Int((floats[1 * hw + o] * 0.5 + 0.5) * 255)))
            let b = max(0, min(255, Int((floats[2 * hw + o] * 0.5 + 0.5) * 255)))
            let i = (dstY * w + x) * 4
            data[i] = UInt8(r); data[i + 1] = UInt8(g); data[i + 2] = UInt8(b); data[i + 3] = 255
        }
    }
    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                     "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}
