// 临时自检：H3 FL2VA 组装管线端到端测试入口（验证后保留，作为可复现测试命令）。
// 运行：NA_H3TEST=1 ./无限画布
import Foundation
import MLX
import AppKit
import ImageIO

enum H3PipelineRun {
    // NA_H3TEST=13：LTX VAE R/G/B 通道错位探针——合成每通道独立空间模式图，
    // encode→decode 后逐通道位移诊断（像素桥产物 R 通道鬼影问题定位）。
    static func runVaeRgbProbe() -> Int32 {
        let outDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_a6764d958fd9455a953182585f3a61e0/temp/pbdiag/na13"
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        func log(_ s: String) { print("[NA13] \(s)"); fflush(stdout) }
        let H = 64, W = 64, T = 9
        var px = [Float](repeating: 0, count: 3 * T * H * W)
        for t in 0..<T { for y in 0..<H { for x in 0..<W {
            let o = (t * 3 * H * W) + y * W + x
            var r: Float = 0, g: Float = 0, b: Float = 0
            // R：竖向四级阶梯（每 2px 一阶）；G：横向四级阶梯；B：8px 大棋盘
            r = Float(((x / 2) % 4)) * 60
            g = Float(((y / 2) % 4)) * 60
            b = (((x / 8) % 2) ^ ((y / 8) % 2)) == 1 ? 200 : 40
            // 角标：左上 16×16 各通道独特色块（用于肉眼/数值识别通道身份与位移）
            if x < 16 && y < 16 { r = 255; g = 0; b = 0 }
            if x >= 16 && x < 32 && y < 16 { r = 0; g = 255; b = 0 }
            if x >= 32 && x < 48 && y < 16 { r = 0; g = 0; b = 255 }
            px[o] = r / 127.5 - 1; px[o + H * W] = g / 127.5 - 1; px[o + 2 * H * W] = b / 127.5 - 1
        }}}
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        Task {
            do {
                let encW = try MLX.loadArrays(url: URL(fileURLWithPath: Stage2Config.vaeEncoder))
                let decW = try MLX.loadArrays(url: URL(fileURLWithPath: CommonPaths.vaeDecoder))
                log("权重加载完成 enc=\(encW.count) dec=\(decW.count)")
                let inp = MLXArray(px, [1, 3, T, H, W])
                log("输入构造完成 \(inp.shape)")
                let lat = vaeEncodeVideo(weights: encW, pixelsBCFHW: inp)
                log("latent: \(lat.shape)")
                let out = vaeDecode(weights: decW, latentNDHWC: lat)
                log("decode 输出: \(out.shape)")
                h3WriteFramePNG(inp, frame: 4, path: outDir + "/inp_t4.png")
                h3WriteFramePNG(out, frame: 4, path: outDir + "/out_t4.png")
                // 多帧一致性抽查：t0/t8 输出也应与 t4 一致（静态输入）
                h3WriteFramePNG(out, frame: 0, path: outDir + "/out_t0.png")
                h3WriteFramePNG(out, frame: 8, path: outDir + "/out_t8.png")
                log("PNG 已落盘 \(outDir)")
                code = 0
            } catch { log("错误: \(error)"); code = 1 }
            sem.signal()
        }
        sem.wait()
        return code
    }

    // NA_H3TEST=14：真实视频 roundtrip PNG 直出——h3_27_stage1 前 9 帧 → LTX enc → dec，
    // 不经 mp4，直接 PNG 落盘，判定 R 鬼影是在 VAE 像素本身还是 mp4 写读（YUV/色度）环节。
    static func runRealRoundtripProbe() -> Int32 {
        let outDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_a6764d958fd9455a953182585f3a61e0/temp/pbdiag/na14"
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        func log(_ s: String) { print("[NA14] \(s)"); fflush(stdout) }
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        Task {
            do {
                let srcPath = "/Users/huachayui/Documents/无限画布/output/视频/h3_27_stage1.mp4"
                let encW = try MLX.loadArrays(url: URL(fileURLWithPath: Stage2Config.vaeEncoder))
                let decW = try MLX.loadArrays(url: URL(fileURLWithPath: CommonPaths.vaeDecoder))
                log("权重加载完成 enc=\(encW.count) dec=\(decW.count)")
                guard var px = readVideoFramesToBCFHW(videoPath: srcPath)?.pixels else {
                    log("读视频失败"); code = 1; sem.signal(); return
                }
                let fAll = px.shape[2]
                let take = min(fAll, 9)
                px = px[0 ..< 1, 0 ..< 3, 0 ..< take, 0 ..< px.shape[3], 0 ..< px.shape[4]]
                log("输入帧: \(px.shape) (\(srcPath))")
                h3WriteFramePNG(px, frame: 4, path: outDir + "/src_t4.png")
                let lat = vaeEncodeVideo(weights: encW, pixelsBCFHW: px)
                log("latent: \(lat.shape)")
                let out = vaeDecode(weights: decW, latentNDHWC: lat)
                log("decode 输出: \(out.shape) 期望帧=\(8 * lat.shape[1] - 7)")
                h3WriteFramePNG(out, frame: 4, path: outDir + "/rt_t4.png")
                // 帧一致性：静态内容也应时间稳定
                h3WriteFramePNG(out, frame: 0, path: outDir + "/rt_t0.png")
                h3WriteFramePNG(out, frame: 8, path: outDir + "/rt_t8.png")
                log("PNG 已落盘 \(outDir)")
                code = 0
            } catch { log("错误: \(error)"); code = 1 }
            sem.signal()
        }
        sem.wait()
        return code
    }

    // NA_H3TEST=15：NA14 的正确 rt 像素 → decodeLatentToSilentMP4（fillPixelBuffer+h264）
    // → readVideoFramesToBCFHW 回读 → PNG 落盘，对比 mp4 写读是否引入 R 通道位移。
    static func runMp4RoundtripProbe() -> Int32 {
        let outDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_a6764d958fd9455a953182585f3a61e0/temp/pbdiag/na15"
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
        func log(_ s: String) { print("[NA15] \(s)"); fflush(stdout) }
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        Task {
            do {
                let srcPath = "/Users/huachayui/Documents/无限画布/output/视频/h3_27_stage1.mp4"
                let encW = try MLX.loadArrays(url: URL(fileURLWithPath: Stage2Config.vaeEncoder))
                let decW = try MLX.loadArrays(url: URL(fileURLWithPath: CommonPaths.vaeDecoder))
                log("权重加载完成")
                guard var px = readVideoFramesToBCFHW(videoPath: srcPath)?.pixels else {
                    log("读视频失败"); code = 1; sem.signal(); return
                }
                let take = min(px.shape[2], 9)
                px = px[0 ..< 1, 0 ..< 3, 0 ..< take, 0 ..< px.shape[3], 0 ..< px.shape[4]]
                let lat = vaeEncodeVideo(weights: encW, pixelsBCFHW: px)
                log("latent: \(lat.shape)")
                guard let mp4 = decodeLatentToSilentMP4(latentNDHWC: lat, fps: 24, assetPrefix: "na15rt") else {
                    log("写 mp4 失败"); code = 1; sem.signal(); return
                }
                log("mp4: \(mp4)")
                guard let back = readVideoFramesToBCFHW(videoPath: mp4)?.pixels else {
                    log("回读失败"); code = 1; sem.signal(); return
                }
                h3WriteFramePNG(back, frame: 4, path: outDir + "/rt_mp4_t4.png")
                log("回读帧 PNG 已落盘 back=\(back.shape)")
                code = 0
            } catch { log("错误: \(error)"); code = 1 }
            sem.signal()
        }
        sem.wait()
        return code
    }

    // NA_H3TEST=17：纯红/绿/蓝单色帧 → writeMp4(h264 709) → readVideoFramesToBCFHW 回读，
    // 打印三通道均值，一锤定音通道布局与色彩矩阵/range。
    static func runSolidColorProbe() -> Int32 {
        func log(_ s: String) { print("[NA17] \(s)"); fflush(stdout) }
        let W = 672, H = 384, N = 9
        let path = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_a6764d958fd9455a953182585f3a61e0/temp/pbdiag/na17/solid.mp4"
        try? FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        Task {
            defer { sem.signal() }
            do {
                try writeMp4(frameCount: N, width: W, height: H, fps: 24, to: path) { f, base, bytesPerRow, _ in
                    guard let base else { return false }
                    let colors: [(UInt8, UInt8, UInt8)] = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    let (r, g, b) = colors[f % 3]
                    for y in 0..<H {
                        let row = base.advanced(by: y * bytesPerRow).bindMemory(to: UInt8.self, capacity: bytesPerRow)
                        for x in 0..<W {
                            row[x * 4 + 0] = 0          // A=0（对齐项目 fillPixelBuffer：byte0=0）
                            row[x * 4 + 1] = r
                            row[x * 4 + 2] = g
                            row[x * 4 + 3] = b
                        }
                    }
                    return true
                }
                log("写出 solid.mp4，开始回读")
                guard let res = readVideoFramesToBCFHW(videoPath: path) else { log("回读失败"); code = 1; return }
                log("回读成功 w=\(res.width) h=\(res.height) fc=\(res.frameCount)")
                let back = res.pixels
                let raw = Array(back.asArray(Float.self))  // B,C,F,H,W
                let b = back.shape[0], c = back.shape[1], f = back.shape[2], h = back.shape[3], w = back.shape[4]
                func meanOf(_ ci: Int, _ fi: Int) -> Float {
                    var sum: Float = 0
                    let stride = h * w
                    let off = ((0 * c + ci) * f + fi) * stride
                    for k in 0..<stride { sum += raw[off + k] }
                    return sum / Float(stride)
                }
                for i in 0..<3 {
                    log("帧\(i): meanR=\(meanOf(0, i)) meanG=\(meanOf(1, i)) meanB=\(meanOf(2, i))")
                }
                code = 0
            } catch { log("错误: \(error)"); code = 1 }
        }
        sem.wait()
        return code
    }

    static func run() -> Int32 {
        // NA_H3TEST=17：纯色链路通道/矩阵验证
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "17" {
            return runSolidColorProbe()
        }
        // NA_H3TEST=15：h264 mp4 写读链路通道位移隔离
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "15" {
            return runMp4RoundtripProbe()
        }
        // NA_H3TEST=14：真实视频 roundtrip PNG 直出（隔离 mp4 环节）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "14" {
            return runRealRoundtripProbe()
        }
        // NA_H3TEST=13：LTX VAE R/G/B 通道错位探针
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "13" {
            return runVaeRgbProbe()
        }
        // NA_H3TEST=2：单帧 VAE 环回测试（加载原图 → encode → decode → 落盘 + 数值统计）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "2" {
            return runRoundtrip()
        }
        // NA_H3TEST=3：多帧 decode chunk 拼接测试（单帧 latent 沿时间轴重复 5 次 → decode 17 帧）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "3" {
            return runDecodeChunk()
        }
        // NA_H3TEST=10：UI 直出路径回归——目标 1344×768 直接 DiT 6 步（不÷2/不升频/无 Stage2），
        // 复现 Xcode Debug 下 encodeImage 崩溃场景
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "10" {
            return runE2EDirectTarget()
        }
        // NA_H3TEST=11：像素桥 refine 参数回归（不改 H3，直接复用已有 stage1 半清视频测 Stage2）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "11" {
            return runPixelBridgeRetest()
        }
        // NA_H3TEST=12：H3 官方两段式 Stage2 端到端——stage1 低清采样 → latent 几何×2
        // → 按 refineSigmas 加噪 → 3 步 Euler 精修 → VAE decode（全 latent 域直通，无像素环回）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "12" {
            return runStage2E2E()
        }
        // NA_H3TEST=9：ComfyUI sol-attn 移植（H3SolAttn）合成数值自检（不加载模型）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "9" {
            return H3SolAttn.selfTest()
        }
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1

        // 首尾帧：基础站姿 → 舞蹈 pose
        let firstImage = "/Users/huachayui/Downloads/基础.png"
        let lastImage = "/Users/huachayui/Downloads/动作.png"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        var outPath = "\(outDir)/h3_fl2va_dance_480p120f.mp4"

        func log(_ s: String) { print("[H3PipelineRun] \(s)"); fflush(stdout) }

        // MINIMAX_H3_SPARSE：off / mix / spatial / temporal（空 = off，dense 基线；同官方默认）
        let polRaw = ProcessInfo.processInfo.environment["MINIMAX_H3_SPARSE"] ?? "off"
        let sparsePol = SparsePolicy(raw: polRaw)
        if polRaw != "off" {
            outPath = outPath.replacingOccurrences(of: ".mp4", with: "_\(polRaw).mp4")
        }
        log("MINIMAX_H3_SPARSE=\(polRaw) → sparsePolicy 生效，输出 \(outPath)")

        // 官方正向流程：请求帧数 → alignFrameCount → videoLatentT / audio_t
        let requestFrames: UInt32 = 120
        let aligned = H3Const.alignFrameCount(requestFrames)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let audioT = H3Const.audioLatentT(frameCount: aligned)

        log("开始 H3 FL2VA 测试管线（864x480 / turbo 6 步 / 请求 \(requestFrames) 帧 → 对齐 \(aligned) 帧 / latent_t=\(latentT) / audio_t=\(audioT)）")
        Task {
            do {
                let path = try await H3FL2VAPipeline.generateVideo(
                    prompt: "A dancer stands in a relaxed neutral stance, then smoothly transitions into a graceful dance pose, lifting one leg and extending both arms elegantly, body slightly twisted with dynamic ballet posture, smooth continuous motion, full body in frame, clean studio background, gentle stage lighting.",
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outPath,
                    width: 864,
                    height: 480,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: sparsePol,
                    scheduleStyle: .official
                )
                log("✅ 生成完成：\(path)")
                code = 0
            } catch {
                log("❌ 生成失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        // 等 Task 完成（最多 20 分钟）
        _ = sem.wait(timeout: .now() + 1200)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// 单帧 VAE 环回测试：原图 [1,3,1,256,256] → encodeImage → latent [1,24,1,16,16]
    /// → decode → 像素 [-1,1] → 落盘 roundtrip.png，打印数值统计。
    static func runRoundtrip() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let imgPath = "/Users/huachayui/Desktop/000 媒体素材/雷电法王.png"
        let weightsPath = "/Users/huachayui/Downloads/minimax h3/MiniMax-H3-FL2VA-MLX-Serve-4bit/video_vae.safetensors"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let mp4Path = "\(outDir)/roundtrip.mp4"
        let pngPath = "\(outDir)/roundtrip.png"

        func log(_ s: String) { print("[H3Roundtrip] \(s)"); fflush(stdout) }
        func stats(_ label: String, _ a: MLXArray) {
            let f = a.asType(.float32)
            let mn = f.min().item(Float.self)
            let mx = f.max().item(Float.self)
            let mean = f.mean().item(Float.self)
            log("\(label): shape=\(a.shape) min=\(mn) max=\(mx) mean=\(mean)")
        }

        Task {
            do {
                guard let pixels = loadImageBCFHW(path: imgPath, width: 256, height: 256) else {
                    throw H3Error.badFile("无法加载原图 \(imgPath)")
                }
                let px = pixels.shape.count == 5 ? pixels : pixels.reshaped(1, 3, 1, 256, 256)
                stats("input pixels", px)

                let vae = try H3VAE.load(try H3Weights(url: URL(fileURLWithPath: weightsPath)))
                log("VAE 加载完成")

                let latent = vae.encodeImage(px)
                stats("latent", latent)

                let recon = vae.decode(latent)
                stats("recon pixels", recon)

                let frame = recon.reshaped(1, 3, 256, 256)
                try writeMp4(frameCount: 1, width: 256, height: 256, fps: 1, to: mp4Path) { _, base, rowBytes, _ in
                    fillPixelBuffer(frame, width: 256, height: 256, base: base, bytesPerRow: rowBytes)
                    return true
                }
                log("已写 \(mp4Path)")
                code = 0
            } catch {
                log("❌ 环回测试失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 600)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// 多帧 decode chunk 拼接测试：单帧 latent [1,24,1,16,16] 沿时间轴重复 5 次
    /// → [1,24,5,16,16]（latentT=5，对应 17 输出帧）→ H3VAE.decode → 落盘 mp4（fps 24）。
    static func runDecodeChunk() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let imgPath = "/Users/huachayui/Desktop/000 媒体素材/雷电法王.png"
        let weightsPath = "/Users/huachayui/Downloads/minimax h3/MiniMax-H3-FL2VA-MLX-Serve-4bit/video_vae.safetensors"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let mp4Path = "\(outDir)/decode_chunk_test.mp4"

        func log(_ s: String) { print("[H3DecodeChunk] \(s)"); fflush(stdout) }
        func stats(_ label: String, _ a: MLXArray) {
            let f = a.asType(.float32)
            let mn = f.min().item(Float.self)
            let mx = f.max().item(Float.self)
            let mean = f.mean().item(Float.self)
            log("\(label): shape=\(a.shape) min=\(mn) max=\(mx) mean=\(mean)")
        }

        Task {
            do {
                guard let pixels = loadImageBCFHW(path: imgPath, width: 256, height: 256) else {
                    throw H3Error.badFile("无法加载原图 \(imgPath)")
                }
                let px = pixels.shape.count == 5 ? pixels : pixels.reshaped(1, 3, 1, 256, 256)

                let vae = try H3VAE.load(try H3Weights(url: URL(fileURLWithPath: weightsPath)))
                log("VAE 加载完成")

                let latent1 = vae.encodeImage(px)
                stats("latent single", latent1)
                // 沿时间轴重复 5 次 → latentT=5
                let z5 = MLX.concatenated([latent1, latent1, latent1, latent1, latent1], axis: 2)
                stats("latent x5", z5)

                let recon = vae.decode(z5)
                stats("recon pixels", recon)

                try writeMp4(frameCount: recon.shape[2], width: 256, height: 256, fps: 24, to: mp4Path) { i, base, rowBytes, _ in
                    let frame = recon[0, 0..<3, i, 0..<256, 0..<256].reshaped(1, 3, 256, 256)
                    return fillPixelBuffer(frame, width: 256, height: 256, base: base, bytesPerRow: rowBytes)
                }
                log("已写 \(mp4Path)（帧数 \(recon.shape[2])）")
                code = 0
            } catch {
                log("❌ 多帧 decode 测试失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 600)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=10：UI 直出路径回归——目标分辨率 1344×768 直接作为 DiT 生成尺寸
    /// （无升频/无 Stage2、6 步 turbo），复现 Xcode Debug 下
    /// 全分辨率 encodeImage/encodeTiled 崩溃场景。
    static func runE2EDirectTarget() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let firstImage = "/Users/huachayui/Documents/无限画布/资产库/t1.png"
        let lastImage = "/Users/huachayui/Documents/无限画布/资产库/t2.png"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let outPath = "\(outDir)/h3_real_1344x768_direct.mp4"
        let probePng = "\(outDir)/h3_real_1344x768_direct_probe.png"

        func log(_ s: String) { print("[H3Direct] \(s)"); fflush(stdout) }

        let aligned = H3Const.alignFrameCount(124)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        log("直出回归：目标 1344x768 直接生成（6 步，无升频/无 Stage2），latentT=\(latentT)")

        Task {
            do {
                let path = try await H3FL2VAPipeline.generateVideo(
                    prompt: "设计精细的舞蹈动作，后到达尾帧",
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outPath,
                    width: 1344,
                    height: 768,
                    steps: 6,
                    latentT: latentT,
                    scheduleStyle: .official,
                    log: { log("[H3] \($0)") }
                )
                log("✅ 直出生成完成：\(path)")
                probeMp4MiddleFrame(mp4Path: path, pngPath: probePng, log: log)
                code = 0
            } catch {
                log("❌ 直出生成失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 3600)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=12：H3 官方两段式 Stage2 端到端。
    /// stage1 低清 448×256 采样（6 步）→ latent 几何×2 → 896×512 refine（3 步 Euler）
    /// → VAE decode。验证：不崩溃/无 NaN/输出尺寸正确/画面与 stage1 内容一致且更高清。
    static func runStage2E2E() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let firstImage = "/Users/huachayui/Documents/无限画布/资产库/t1.png"
        let lastImage = "/Users/huachayui/Documents/无限画布/资产库/t2.png"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let stage1Path = "\(outDir)/h3_s2_stage1_448x256.mp4"
        let finalPath = "\(outDir)/h3_s2_final_896x512_refine3.mp4"
        let probePng = "\(outDir)/h3_s2_final_probe.png"

        func log(_ s: String) { print("[H3S2] \(s)"); fflush(stdout) }

        let aligned = H3Const.alignFrameCount(90)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let s2cfg = H3Stage2Config(scale: 2,
                                   refineSigmas: [0.9035, 0.6316, 0.3158, 0.0],
                                   refineSeed: 777)
        log("Stage2 端到端：stage1 448×256 6 步 → latent×2 → 896×512 refine 3 步，latentT=\(latentT)")

        Task {
            do {
                // 对照：单遍低清（无 Stage2，旧路径回归）
                let p1 = try await H3FL2VAPipeline.generateVideo(
                    prompt: "保持场景、主体、动作、构图与色彩完全一致，仅提升清晰度与细节质感。",
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: stage1Path,
                    width: 448,
                    height: 256,
                    steps: 6,
                    latentT: latentT,
                    scheduleStyle: .official,
                    log: { log("[stage1] \($0)") }
                )
                log("✅ stage1 对照完成：\(p1)")
                // 两段式：同一 prompt/seed，低清采样后 latent 域放大 + refine
                let p2 = try await H3FL2VAPipeline.generateVideo(
                    prompt: "保持场景、主体、动作、构图与色彩完全一致，仅提升清晰度与细节质感。",
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: finalPath,
                    width: 448,
                    height: 256,
                    steps: 6,
                    latentT: latentT,
                    scheduleStyle: .official,
                    log: { log("[stage2] \($0)") },
                    stage2: s2cfg
                )
                log("✅ Stage2 生成完成：\(p2)")
                probeMp4MiddleFrame(mp4Path: p2, pngPath: probePng, log: log)
                code = 0
            } catch {
                log("❌ Stage2 生成失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 3600)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=11：像素桥 refine 参数回归。
    /// 不改 H3：直接复用已有 stage1 半清视频（默认 h3_27_stage1.mp4，可用环境变量 PIX_IN 覆盖），
    /// 首尾参考图自抽自引用（抽 stage1 首/尾帧），跑 ltxEnhanceExternalVideoWithStage2 全链路，
    /// 产物写 output/视频，随后抽中帧 PNG 供人工比对。
    static func runPixelBridgeRetest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        func log(_ s: String) { print("[PixBridgeRetest] \(s)"); fflush(stdout) }

        let videoDir = outputVideoDirURL.path
        let envIn = ProcessInfo.processInfo.environment["PIX_IN"] ?? ""
        let stage1 = envIn.isEmpty ? "\(videoDir)/h3_27_stage1.mp4" : envIn
        guard FileManager.default.fileExists(atPath: stage1) else {
            log("❌ stage1 视频不存在：\(stage1)")
            return 1
        }
        let refFirst = "\(videoDir)/_pb_ref_first.png"
        let refLast = "\(videoDir)/_pb_ref_last.png"

        // 用项目内置视频读取抽首/尾帧作参考图（自引用，内容与 stage1 严格一致）
        guard let video = readVideoFramesToBCFHW(videoPath: stage1), video.frameCount >= 9 else {
            log("❌ 无法读取 stage1 视频")
            return 1
        }
        h3WriteFramePNG(video.pixels, frame: 0, path: refFirst)
        h3WriteFramePNG(video.pixels, frame: video.frameCount - 1, path: refLast)
        log("✅ 参考帧已抽：首帧/尾帧 #\(video.frameCount - 1) → \(refFirst) / \(refLast)（\(video.width)×\(video.height)）")

        // 改造 B：二采无提示词 —— 无保真引导词；像素桥内部用完全静态空文本条件
        // （loadStaticEmptyTextCond：常量文件读张量 / 缺失全零兜底），全程零模型权重调用，
        // 不加载 Gemma/connector、不自动自举、无需手动设环境变量。

        Task {
            do {
                guard let path = await ltxEnhanceExternalVideoWithStage2(
                    videoPath: stage1,
                    imagePaths: [refFirst, refLast],
                    seed: 42) else {
                    log("❌ 像素桥失败")
                    return
                }
                log("✅ 像素桥完成：\(path)")
                // 抽中帧（与 stage1 同帧号）供比对
                let checkDir = "\(videoDir)/_pb_check"
                try? FileManager.default.createDirectory(atPath: checkDir, withIntermediateDirectories: true)
                if let outV = readVideoFramesToBCFHW(videoPath: path) {
                    let mid = min(video.frameCount - 1, outV.frameCount / 2)
                    let base = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
                    h3WriteFramePNG(outV.pixels, frame: mid, path: "\(checkDir)/\(base)_f\(mid).png")
                    log("✅ 产物中帧 #\(mid) 已抽：\(checkDir)/\(base)_f\(mid).png")
                }
                let srcMid = video.frameCount / 2
                h3WriteFramePNG(video.pixels, frame: srcMid, path: "\(checkDir)/stage1_f\(srcMid).png")
                log("✅ 源视频中帧 #\(srcMid) 已抽：\(checkDir)/stage1_f\(srcMid).png")
                code = 0
            } catch {
                log("❌ 异常：\(error)")
            }
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 3600)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    static func runFFCapture(_ args: [String]) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        do { try p.run(); p.waitUntilExit() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }

    static func probeMp4MiddleFrame(mp4Path: String, pngPath: String, log: (String) -> Void) {
        func runFF(_ args: [String]) -> String? {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            p.arguments = args
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = FileHandle.nullDevice
            do {
                try p.run()
                p.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                return String(data: data, encoding: .utf8)
            } catch {
                log("ffmpeg 调用失败：\(error)")
                return nil
            }
        }
        // 先探测总帧数（估算中间帧号，避免首帧=条件图的假阳性）
        if let out = runFF(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", mp4Path]),
           let total = Int(out.trimmingCharacters(in: .whitespacesAndNewlines)), total > 0 {
            let mid = total / 2
            if runFF(["ffmpeg", "-y", "-i", mp4Path, "-vf", "select=eq(n\\,\(mid))", "-frames:v", "1", pngPath]) != nil {
                log("已抽中帧 #\(mid)/\(total) → \(pngPath)")
                return
            }
        }
        // 兜底：首帧
        _ = runFF(["ffmpeg", "-y", "-i", mp4Path, "-vf", "select=eq(n\\,0)", "-frames:v", "1", pngPath])
        log("已抽首帧（探测失败兜底）→ \(pngPath)")
    }
}
