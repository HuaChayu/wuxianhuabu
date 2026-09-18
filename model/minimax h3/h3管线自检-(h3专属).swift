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
        // NA_H3TEST=21：SelfLift 第三分支 A/B（nearest vs learned upscaler）端到端对比
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "21" {
            return runSelfLiftABTest()
        }
        // NA_H3TEST=23：h3_53 真实场景高分辨率 A/B（nearest vs learned，1344×768 / 62 帧）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "23" {
            return runH3SceneABTest()
        }
        // NA_H3TEST=24：h3_53 真实场景 rho 扫描（learned 提升 × rho 0.25/0.6/0.9，验证像素锚点修正能否消残余轻重影）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "24" {
            return runH3RhoSweepTest()
        }
        // NA_H3TEST=25：h3_53 真实场景 w 软混合扫描（learned + rho=0.9 × wMin/wMax 三档，
        // 验证软混合能否在保留 z_pix 修正的同时用 z_lat 时间平滑性压掉背景小角色帧间抖动）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "25" {
            return runH3WMixSweepTest()
        }
        // NA_H3TEST=26：官方默认路径复刻（learned upscaler + rho=0，不混合 z_pix）。
        // 官方 SelfLiftH3Sampler 默认即 rho=0（纯 z_lat 提升，像素锚点为可选增强）；
        // 若本组后段干净 → 重影由 z_pix 混合引入，默认直接回 rho=0；若不干净 → 问题在 learned 升频器本身。
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "26" {
            return runH3OfficialDefaultTest()
        }
        // NA_H3TEST=27：短场景帧数对照 —— 26 号输入（39 帧）× N=9（6+3，与用户 UI 同参数）。
        // 若本组干净而 UI 长视频重影 → 帧数/时长因素；若本组也重影 → 与帧数无关，查调度/场景。
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "27" {
            setenv("H3_AB_STEPS", "9", 1)
            setenv("H3_AB_OUT", "h3_short_n9.mp4", 1)
            return runH3OfficialDefaultTest()
        }
        // NA_H3TEST=20：ref2va 多参考图通路（多张角色参考图 + 场景描述，多参身份保持）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "20" {
            return runRef2VATest()
        }
        // NA_H3TEST=9：ComfyUI sol-attn 移植（H3SolAttn）合成数值自检（不加载模型）
        if ProcessInfo.processInfo.environment["NA_H3TEST"] == "9" {
            return H3SolAttn.selfTest()
        }
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1

        // 首尾帧：多角色同框场景（可用 H3_FIRST_IMG / H3_LAST_IMG 覆盖）
        // three_shot_a/b.png = 左少女(2.png) + 中狐耳男(3.png) + 右银发男(4.png) 拼成的 864x480 横版合影，
        // b 为 a 中心放大 1.08 的推近版，避免首尾同图导致画面完全静止。
        // 合成脚本：temp/make_three_shot.py
        let envCfg = ProcessInfo.processInfo.environment
        let sceneDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3/assets"
        let firstImage = envCfg["H3_FIRST_IMG"] ?? "\(sceneDir)/three_shot_a.png"
        let lastImage = envCfg["H3_LAST_IMG"] ?? "\(sceneDir)/three_shot_b.png"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        var outPath = "\(outDir)/h3_fl2va_trio_talk_480p124f.mp4"

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
                    // 多角色多参场景：三人同框、各有动作与台词，用于检验模型对多主体的身份保持与对白能力
                    prompt: envCfg["H3_PROMPT"] ?? (
                        "Three characters stand side by side in a clean white studio, full body in frame, soft studio lighting. "
                        + "Left: a girl with short black hair in a light mint-green modern Chinese dress smiles and waves, saying \"Hey! The canvas is finally alive!\" "
                        + "Middle: a young man with brown fox ears and a beige outfit turns his head toward her, eyes wide, and replies \"Wait, you rendered all of this on a laptop?\" "
                        + "Right: a silver-haired man in a black crown and a long dark robe smiles calmly and says \"Impressive. But can it keep all three of us in frame?\" "
                        + "Each character speaks clearly in turn, natural lip-sync, distinct voices, smooth continuous motion, stable identity, no morphing, no extra characters."
                    ),
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

    /// NA_H3TEST=20：ref2va 多参考图通路——多张角色参考图 + 场景描述互动，
    /// 验证移植融合权重的多参能力（身份保持 / 同框互动）。
    /// 默认参考图：~/Downloads/角色/ 1.jpg(红裙金冠女) 2.png(薄荷绿裙少女) 3.png(狐耳男) 4.png(银冠黑袍男)；
    /// 可用 H3_REF_IMAGES（逗号分隔绝对路径）与 H3_PROMPT 覆盖。
    static func runRef2VATest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let envCfg = ProcessInfo.processInfo.environment
        let refDir = "/Users/huachayui/Downloads/角色"
        let refs: [String] = envCfg["H3_REF_IMAGES"]
            .map { $0.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) } }
            ?? ["\(refDir)/1.jpg", "\(refDir)/2.png", "\(refDir)/3.png", "\(refDir)/4.png"]
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let outPath = envCfg["H3_OUT"] ?? "\(outDir)/h3_ref2va_quad_scene_480p124f.mp4"
        // 参考图缩放模式：match = 缩到生成面积（快）；mid = 短边 1024；max = 官方短边 2048（慢）
        let refMode = envCfg["H3_REF_MODE"] ?? "match"
        let refSizing: RefImageSizing = refMode == "max" ? .max : (refMode == "mid" ? .mid : .match)

        func log(_ s: String) { print("[H3Ref2VA] \(s)"); fflush(stdout) }

        let requestFrames: UInt32 = 120
        let aligned = H3Const.alignFrameCount(requestFrames)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        log("ref2va 多参测试：\(refs.count) 张参考图 / 864x480 / turbo 6 步 / latentT=\(latentT)")
        for (i, p) in refs.enumerated() { log("  参考图 \(i + 1)：\(p)") }

        // ★ 2026-09-18：ref2va 自检默认开 SelfLift 解耦（3+3），对齐用户 UI 组合；
        //   NA_H3_FL2VA_AS_REFS=1 时"首尾帧作参考图"即走本通路（软参考 + 视觉块，无 keyframes 硬锚）。
        setenv("NA_H3_SELFLIFT_RHO", "0", 1)
        setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
        setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", "0.7", 1)
        setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "3", 1)

        Task {
            do {
                let path = try await H3FL2VAPipeline.generateVideo(
                    prompt: envCfg["H3_PROMPT"] ?? (
                        "The four characters shown in the reference pictures gather in a bright living room and interact with each other. "
                        + "Picture 1, the woman in the red dress and golden crown, stands in the middle and speaks first: \"Welcome, everyone. Let's make a short film together.\" "
                        + "Picture 2, the girl in the mint-green dress, waves happily and answers: \"I can do the lighting!\" "
                        + "Picture 3, the young man with fox ears, laughs and says: \"Then I will handle the camera.\" "
                        + "Picture 4, the man in the silver crown and dark robe, nods calmly and replies: \"And I will direct.\" "
                        + "They look at each other in turn and talk, natural lip-sync, distinct voices, stable identities, consistent clothing, smooth continuous motion, no morphing, no extra characters."
                    ),
                    firstImagePath: refs[0],
                    lastImagePath: refs[refs.count - 1],
                    outPath: outPath,
                    width: 864,
                    height: 480,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .off,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[H3] \($0)") },
                    referenceImagePaths: refs,
                    referenceSizing: refSizing
                )
                log("✅ ref2va 生成完成：\(path)")
                code = 0
            } catch {
                log("❌ ref2va 生成失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 3600)
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

    /// NA_H3TEST=21：SelfLift 第三分支 A/B 端到端 —— 同一 seed/首尾帧/分辨率，
    /// 分别用 nearest 直接 latent 提升（默认，对照 h3_53 配置）与官方 learned upscaler
    /// （NA_H3_UPSCALER=learned，时间维感知）各生成一段短视频，抽中帧 PNG 供重影比对。
    /// 参数：H3_AB_FRAMES（默认 33）、H3_AB_W/H3_AB_H（默认 448×256）。
    static func runSelfLiftABTest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let env = ProcessInfo.processInfo.environment
        let sceneDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3/assets"
        let firstImage = env["H3_FIRST_IMG"] ?? "/Users/huachayui/Documents/无限画布/资产库/t1.png"
        let lastImage = env["H3_LAST_IMG"] ?? "/Users/huachayui/Documents/无限画布/资产库/t2.png"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let framesRaw = UInt32(env["H3_AB_FRAMES"] ?? "33") ?? 33
        let width = Int(env["H3_AB_W"] ?? "448") ?? 448
        let height = Int(env["H3_AB_H"] ?? "256") ?? 256
        // H3_AB_GROUPS="F,H" 时只跑指定组，跳过其余组（用于大分辨率快速对照）
        let groups = (env["H3_AB_GROUPS"] ?? "A,B,C,D,E,F,G,H").split(separator: ",").map(String.init)
        let aligned = H3Const.alignFrameCount(framesRaw)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let prompt = env["H3_PROMPT"] ?? (
            "A woman in an elegant celadon ancient-style long dress performs a flowing sword dance on a stage under a spotlight. "
            + "She starts raising a glowing sword high, then smoothly turns and lunges into a low bow stance thrusting the sword sideways. "
            + "Continuous fluid motion, stable identity, clear sharp face throughout, no ghosting, no double contours, no extra fingers, stable background."
        )
        let outNearest = "\(outDir)/h3_sl_ab_nearest.mp4"
        let outLearned = "\(outDir)/h3_sl_ab_learned.mp4"
        let outOfficial = "\(outDir)/h3_sl_ab_official.mp4"
        let outRho06 = "\(outDir)/h3_sl_ab_rho06.mp4"
        let outOfficial12 = "\(outDir)/h3_sl_ab_official12.mp4"
        let outDecouple = "\(outDir)/h3_sl_ab_decouple.mp4"
        let outDecoupleShallow = "\(outDir)/h3_sl_ab_decouple_shallow.mp4"
        let outDecoupleLowK03 = "\(outDir)/h3_sl_ab_decouple_lowk03.mp4"
        let outDecoupleHigh3 = "\(outDir)/h3_sl_ab_decouple_high3.mp4"
        let probeNearest = "\(outDir)/h3_sl_ab_nearest_probe.png"
        let probeLearned = "\(outDir)/h3_sl_ab_learned_probe.png"
        let probeOfficial = "\(outDir)/h3_sl_ab_official_probe.png"
        let probeRho06 = "\(outDir)/h3_sl_ab_rho06_probe.png"
        let probeOfficial12 = "\(outDir)/h3_sl_ab_official12_probe.png"
        let probeDecouple = "\(outDir)/h3_sl_ab_decouple_probe.png"
        let probeDecoupleShallow = "\(outDir)/h3_sl_ab_decouple_shallow_probe.png"
        let probeDecoupleLowK03 = "\(outDir)/h3_sl_ab_decouple_lowk03_probe.png"
        let probeDecoupleHigh3 = "\(outDir)/h3_sl_ab_decouple_high3_probe.png"

        func log(_ s: String) { print("[H3SLAB] \(s)"); fflush(stdout) }

        log("SelfLift A/B/C/D/E/F：\(width)×\(height) / \(aligned) 帧 / seed=42；A-D 组 turbo 6 步（nearest → learned → official(rho0+learned) → rho06），E/F 组 12 步公平对照（official12 → 解耦 σ_k=0.7+高清6步）")
        Task {
            do {
                if groups.contains("A") {
                // A 组：nearest（显式回退，对照 h3_53 历史配置）
                setenv("NA_H3_UPSCALER", "nearest", 1)
                let p1 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outNearest,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[A-nearest] \($0)") }
                )
                log("✅ A 组 nearest 完成：\(p1)")
                probeMp4MiddleFrame(mp4Path: p1, pngPath: probeNearest, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("B") {
                // B 组：learned upscaler（时间维感知）
                setenv("NA_H3_UPSCALER", "learned", 1)
                let p2 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outLearned,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[B-learned] \($0)") }
                )
                log("✅ B 组 learned 完成：\(p2)")
                probeMp4MiddleFrame(mp4Path: p2, pngPath: probeLearned, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("C") {
                // C 组：官方默认组合 —— rho=0（纯 latent 提升）+ learned upscaler
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                let p3 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outOfficial,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[C-official] \($0)") }
                )
                log("✅ C 组 official 完成：\(p3)")
                probeMp4MiddleFrame(mp4Path: p3, pngPath: probeOfficial, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("D") {
                // D 组：官方建议起点 rho=0.6 + nearest 提升（像素锚点修正比例最高）
                setenv("NA_H3_UPSCALER", "nearest", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0.6", 1)
                let p4 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outRho06,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[D-rho06] \($0)") }
                )
                log("✅ D 组 rho06 完成：\(p4)")
                probeMp4MiddleFrame(mp4Path: p4, pngPath: probeRho06, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("E") {
                // E 组：12 步官方调度对照（learned + rho=0，与 C 组同配置，仅步数 6→12；
                // 低分 NFE=9 + 高分 NFE=3 = 12 总 NFE，即用户重影场景的默认配置）
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE", "0", 1)
                let p5 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outOfficial12,
                    width: width,
                    height: height,
                    steps: 12,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[E-official12] \($0)") }
                )
                log("✅ E 组 official12 完成：\(p5)")
                probeMp4MiddleFrame(mp4Path: p5, pngPath: probeOfficial12, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("F") {
                // F 组：12 步解耦模式（低清 6 步跑到 σ_k≈0.706 消歧义 + 提升后按 σ_next≈0.706
                // 直接重加噪 + 高清 6 步收细节；总 NFE=12 与 E 组一致，仅 σ 调度解耦）
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", "0.7", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "6", 1)
                let p6 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outDecouple,
                    width: width,
                    height: height,
                    steps: 12,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[F-decouple] \($0)") }
                )
                log("✅ F 组 decouple 完成：\(p6)")
                probeMp4MiddleFrame(mp4Path: p6, pngPath: probeDecouple, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("G") {
                // G 组：12 步解耦、低清更彻底消歧义档（σ_k=0.3 → 低清 L=29 步跑到 σ≈0.3，
                // 运动位置歧义消得更干净；提升后按 σ_next=σ_k≈0.3 直接重加噪 + 高清 6 步收细节）
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", "0.3", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "6", 1)
                let p7 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outDecoupleLowK03,
                    width: width,
                    height: height,
                    steps: 12,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[G-decouple-lowk03] \($0)") }
                )
                log("✅ G 组 decouple-lowk03 完成：\(p7)")
                probeMp4MiddleFrame(mp4Path: p7, pngPath: probeDecoupleLowK03, log: log)
                MLX.Memory.clearCache()
                }

                if groups.contains("H") {
                // H 组：解耦 σ_k=0.7（低清 6 步消歧义）+ 高清只跑 3 步收细节。
                // 与 F 组（高清 6 步）对照：验证低清跑深后高清是否只需 3 步即可收干净、总 NFE 更低。
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", "0.7", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "3", 1)
                let p8 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outDecoupleHigh3,
                    width: width,
                    height: height,
                    steps: 12,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[H-decouple-high3] \($0)") }
                )
                log("✅ H 组 decouple-high3 完成：\(p8)")
                probeMp4MiddleFrame(mp4Path: p8, pngPath: probeDecoupleHigh3, log: log)
                }
                code = 0
            } catch {
                log("❌ SelfLift A/B 失败：\(error)")
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

        // ★ 二采通道 A/B 开关（自检专用，只影响本次 NA_H3TEST=11，不写回用户偏好设置）：
        //   未设 / "1" → 走新的「第二阶段·CQ 清晰度增强」通道（与 App 默认一致）；
        //   PIX_CQ=0   → 强制回原 LTX 像素桥 IC 二采，用于回归验证原分支未被破坏。
        //   优先级：LTX_CQ_ENHANCER（通道强制开关）> PIX_CQ > 偏好设置 videoUseCQEnhancer。
        let pixCQ = ProcessInfo.processInfo.environment["PIX_CQ"] != "0"
        log("ℹ️ 二采通道：\(pixCQ ? "CQ 清晰度增强（PIX_CQ≠0，官方 CQ LoRA）" : "原 IC 像素桥（PIX_CQ=0）")")

        Task {
            do {
                guard let path = await ltxEnhanceExternalVideoWithStage2(
                    videoPath: stage1,
                    imagePaths: [refFirst, refLast],
                    seed: 42,
                    cqEnhancerEnable: pixCQ) else {
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

    /// NA_H3TEST=26：官方默认路径复刻 —— learned + rho=0（不混合 z_pix，纯 z_lat 提升）。
    /// 与 rho=0.6 组同 seed/同帧数/同场景，后段逐帧对比判定重影来源。
    static func runH3OfficialDefaultTest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let env = ProcessInfo.processInfo.environment
        let inDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_1ef541d443984dc7885a9e7586114321/temp/h3_53_ab_input"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let firstImage = env["H3_FIRST_IMG"] ?? "\(inDir)/first.png"
        let lastImage = env["H3_LAST_IMG"] ?? "\(inDir)/last.png"
        let width = Int(env["H3_AB_W"] ?? "1344") ?? 1344
        let height = Int(env["H3_AB_H"] ?? "768") ?? 768
        let framesRaw = UInt32(env["H3_AB_FRAMES"] ?? "33") ?? 33
        let stepsOverride = UInt32(env["H3_AB_STEPS"] ?? "6") ?? 6
        let outName = env["H3_AB_OUT"] ?? "h3_official_default.mp4"
        let aligned = H3Const.alignFrameCount(framesRaw)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let prompt = env["H3_PROMPT"] ?? (
            "integrated_multimodal_description: [Shot 1] 赵阳在摇晃中缓缓睁开眼，入目便是一张清秀的脸庞。"
            + "赵阳 (S1) 看着那双忧郁的眼睛道：<d>[Chinese] 你是？</d> 那张清秀脸庞的主人 (S2) 答："
            + "<d>[Chinese] 果然还活着，哦，我叫江作人。咱们都被抓壮丁了，诶，等等，你自己看吧。</d>"
            + "江作人 (S2) 给他讲述了自己睁开眼看到的一切。赵阳 (S1) 道：<d>[Chinese] 原来是这样吗。</d>"
            + "赵阳顶着大光头，双手抱头，眼神空洞，久久不语。"
        )

        func log(_ s: String) { print("[OFFDEFAULT] \(s)"); fflush(stdout) }

        log("官方默认路径：\(width)×\(height) / \(aligned) 帧 / turbo \(stepsOverride) 步 / seed=42，learned + rho=0（纯 z_lat，不混合 z_pix）")
        Task {
            do {
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_WMIN", "1.0", 1)
                setenv("NA_H3_SELFLIFT_WMAX", "1.0", 1)
                let out = "\(outDir)/\(outName)"
                log("开始 → \(out)")
                let p = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: out,
                    width: width,
                    height: height,
                    steps: stepsOverride,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[official] \($0)") }
                )
                log("✅ 官方默认路径完成：\(p)")
                code = 0
            } catch {
                log("❌ 官方默认路径失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 5400)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=25：h3_53 真实场景 w 软混合扫描 —— learned + rho=0.9 固定，w 三档：
    /// W0 硬掩码（w=1/1，当前默认，=rho=0.9 基线）→ W1 半量（0.5/0.5）→ W2 渐变（0.3/0.8）。
    /// 背景小角色残影来自 z_pix VAE 往返帧间抖动被 90% 全量注入；软混合让分歧小（模糊/背景）区
    /// 保留 z_lat 时间平滑性。参数：H3_AB_FRAMES（默认 33）、H3_AB_W/H3_AB_H（默认 1344×768）。
    static func runH3WMixSweepTest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let env = ProcessInfo.processInfo.environment
        let inDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_1ef541d443984dc7885a9e7586114321/temp/h3_53_ab_input"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let firstImage = env["H3_FIRST_IMG"] ?? "\(inDir)/first.png"
        let lastImage = env["H3_LAST_IMG"] ?? "\(inDir)/last.png"
        let width = Int(env["H3_AB_W"] ?? "1344") ?? 1344
        let height = Int(env["H3_AB_H"] ?? "768") ?? 768
        let framesRaw = UInt32(env["H3_AB_FRAMES"] ?? "33") ?? 33
        let aligned = H3Const.alignFrameCount(framesRaw)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let prompt = env["H3_PROMPT"] ?? (
            "integrated_multimodal_description: [Shot 1] 赵阳在摇晃中缓缓睁开眼，入目便是一张清秀的脸庞。"
            + "赵阳 (S1) 看着那双忧郁的眼睛道：<d>[Chinese] 你是？</d> 那张清秀脸庞的主人 (S2) 答："
            + "<d>[Chinese] 果然还活着，哦，我叫江作人。咱们都被抓壮丁了，诶，等等，你自己看吧。</d>"
            + "江作人 (S2) 给他讲述了自己睁开眼看到的一切。赵阳 (S1) 道：<d>[Chinese] 原来是这样吗。</d>"
            + "赵阳顶着大光头，双手抱头，眼神空洞，久久不语。"
        )

        func log(_ s: String) { print("[WMIXSWEEP] \(s)"); fflush(stdout) }

        log("w 软混合扫描：\(width)×\(height) / \(aligned) 帧 / turbo 6 步 / seed=42，learned+rho0.9 × w 1/1 → 0.5/0.5 → 0.3/0.8")
        Task {
            do {
                let runs: [(wMin: Float, wMax: Float, tag: String)] = [(1.0, 1.0, "w11"), (0.5, 0.5, "w05"), (0.3, 0.8, "w038")]
                for run in runs {
                    setenv("NA_H3_UPSCALER", "learned", 1)
                    setenv("NA_H3_SELFLIFT_RHO", "0.9", 1)
                    setenv("NA_H3_SELFLIFT_WMIN", "\(run.wMin)", 1)
                    setenv("NA_H3_SELFLIFT_WMAX", "\(run.wMax)", 1)
                    let out = "\(outDir)/h3_wmix_\(run.tag).mp4"
                    let probe = "\(outDir)/h3_wmix_\(run.tag)_probe.png"
                    log("组 w=\(run.wMin)/\(run.wMax) 开始 → \(out)")
                    let p = try await H3FL2VAPipeline.generateVideo(
                        prompt: prompt,
                        firstImagePath: firstImage,
                        lastImagePath: lastImage,
                        outPath: out,
                        width: width,
                        height: height,
                        steps: 6,
                        latentT: latentT,
                        seed: 42,
                        sparsePolicy: .mix,
                        scheduleStyle: .official,
                        selfLiftEnabled: true,
                        log: { log("[w=\(run.wMin)/\(run.wMax)] \($0)") }
                    )
                    log("✅ w=\(run.wMin)/\(run.wMax) 完成：\(p)")
                    probeMp4MiddleFrame(mp4Path: p, pngPath: probe, log: log)
                    MLX.Memory.clearCache()
                }
                code = 0
            } catch {
                log("❌ w 软混合扫描失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 5400)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=24：h3_53 真实场景 rho 扫描 —— learned 提升固定，rho 三档 0.25/0.6/0.9，
    /// 验证像素锚点（z_pix VAE 往返真值）修正比例提高能否消除 learned 时间插值残留的轻微重影。
    /// 参数：H3_AB_FRAMES（默认 33）、H3_AB_W/H3_AB_H（默认 1344×768）。
    static func runH3RhoSweepTest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let env = ProcessInfo.processInfo.environment
        let inDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_1ef541d443984dc7885a9e7586114321/temp/h3_53_ab_input"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let firstImage = env["H3_FIRST_IMG"] ?? "\(inDir)/first.png"
        let lastImage = env["H3_LAST_IMG"] ?? "\(inDir)/last.png"
        let width = Int(env["H3_AB_W"] ?? "1344") ?? 1344
        let height = Int(env["H3_AB_H"] ?? "768") ?? 768
        let framesRaw = UInt32(env["H3_AB_FRAMES"] ?? "33") ?? 33
        let aligned = H3Const.alignFrameCount(framesRaw)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let prompt = env["H3_PROMPT"] ?? (
            "integrated_multimodal_description: [Shot 1] 赵阳在摇晃中缓缓睁开眼，入目便是一张清秀的脸庞。"
            + "赵阳 (S1) 看着那双忧郁的眼睛道：<d>[Chinese] 你是？</d> 那张清秀脸庞的主人 (S2) 答："
            + "<d>[Chinese] 果然还活着，哦，我叫江作人。咱们都被抓壮丁了，诶，等等，你自己看吧。</d>"
            + "江作人 (S2) 给他讲述了自己睁开眼看到的一切。赵阳 (S1) 道：<d>[Chinese] 原来是这样吗。</d>"
            + "赵阳顶着大光头，双手抱头，眼神空洞，久久不语。"
        )

        func log(_ s: String) { print("[RHOSWEEP] \(s)"); fflush(stdout) }

        log("rho 扫描：\(width)×\(height) / \(aligned) 帧 / turbo 6 步 / seed=42，learned × rho 0.25→0.6→0.9")
        Task {
            do {
                let runs: [(rho: Double, tag: String)] = [(0.25, "025"), (0.6, "060"), (0.9, "090")]
                for run in runs {
                    setenv("NA_H3_UPSCALER", "learned", 1)
                    setenv("NA_H3_SELFLIFT_RHO", "\(run.rho)", 1)
                    let out = "\(outDir)/h3_rho_\(run.tag).mp4"
                    let probe = "\(outDir)/h3_rho_\(run.tag)_probe.png"
                    log("组 ρ=\(run.rho) 开始 → \(out)")
                    let p = try await H3FL2VAPipeline.generateVideo(
                        prompt: prompt,
                        firstImagePath: firstImage,
                        lastImagePath: lastImage,
                        outPath: out,
                        width: width,
                        height: height,
                        steps: 6,
                        latentT: latentT,
                        seed: 42,
                        sparsePolicy: .mix,
                        scheduleStyle: .official,
                        selfLiftEnabled: true,
                        log: { log("[ρ=\(run.rho)] \($0)") }
                    )
                    log("✅ ρ=\(run.rho) 完成：\(p)")
                    probeMp4MiddleFrame(mp4Path: p, pngPath: probe, log: log)
                    MLX.Memory.clearCache()
                }
                code = 0
            } catch {
                log("❌ rho 扫描失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 5400)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }

    /// NA_H3TEST=23：h3_53 真实场景高分辨率 A/B —— 用资产库 h3_53 首/尾帧 + 原始 prompt，
    /// 1344×768 / 62 帧，A=nearest（默认）、B=learned（NA_H3_UPSCALER=learned），
    /// 抽多个探针帧供面部重影复核。
    static func runH3SceneABTest() -> Int32 {
        let sem = DispatchSemaphore(value: 0)
        var code: Int32 = 1
        let env = ProcessInfo.processInfo.environment
        let inDir = "/Users/huachayui/Library/Application Support/com.tencent.mac.marvis/MarvisData/User/99999343CE761436DB7BAC928475A2A3/workspace/conv_1ef541d443984dc7885a9e7586114321/temp/h3_53_ab_input"
        let outDir = "/Users/huachayui/Desktop/无限画布/model/minimax h3"
        let firstImage = env["H3_FIRST_IMG"] ?? "\(inDir)/first.png"
        let lastImage = env["H3_LAST_IMG"] ?? "\(inDir)/last.png"
        let width = Int(env["H3_AB_W"] ?? "1344") ?? 1344
        let height = Int(env["H3_AB_H"] ?? "768") ?? 768
        let framesRaw = UInt32(env["H3_AB_FRAMES"] ?? "62") ?? 62
        let aligned = H3Const.alignFrameCount(framesRaw)
        let latentT = H3Const.videoLatentT(frameCount: aligned)
        let prompt = env["H3_PROMPT"] ?? (
            "integrated_multimodal_description: [Shot 1] 赵阳在摇晃中缓缓睁开眼，入目便是一张清秀的脸庞。"
            + "赵阳 (S1) 看着那双忧郁的眼睛道：<d>[Chinese] 你是？</d> 那张清秀脸庞的主人 (S2) 答："
            + "<d>[Chinese] 果然还活着，哦，我叫江作人。咱们都被抓壮丁了，诶，等等，你自己看吧。</d>"
            + "江作人 (S2) 给他讲述了自己睁开眼看到的一切。赵阳 (S1) 道：<d>[Chinese] 原来是这样吗。</d>"
            + "赵阳顶着大光头，双手抱头，眼神空洞，久久不语。"
        )
        let outNearest = "\(outDir)/h3_scene_ab_nearest.mp4"
        let outLearned = "\(outDir)/h3_scene_ab_learned.mp4"
        let outDecoupleV2 = "\(outDir)/h3_scene_ab_decouple_v2.mp4"
        let probeNearest = "\(outDir)/h3_scene_ab_nearest_probe.png"
        let probeLearned = "\(outDir)/h3_scene_ab_learned_probe.png"
        let probeDecoupleV2 = "\(outDir)/h3_scene_ab_decouple_v2_probe.png"

        func log(_ s: String) { print("[H3SCENEAB] \(s)"); fflush(stdout) }

        log("h3_53 场景 A/B：\(width)×\(height) / \(aligned) 帧 / turbo 6 步 / seed=42，nearest → learned")
        Task {
            do {
                // A 组：nearest（显式回退，h3_53 历史配置）
                setenv("NA_H3_UPSCALER", "nearest", 1)
                setenv("NA_H3_SELFLIFT_RHO", "", 1)
                let p1 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outNearest,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[A-nearest] \($0)") }
                )
                log("✅ A 组 nearest 完成：\(p1)")
                probeMp4MiddleFrame(mp4Path: p1, pngPath: probeNearest, log: log)
                MLX.Memory.clearCache()

                // B 组：learned upscaler
                setenv("NA_H3_UPSCALER", "learned", 1)
                let p2 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outLearned,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[B-learned] \($0)") }
                )
                log("✅ B 组 learned 完成：\(p2)")
                probeMp4MiddleFrame(mp4Path: p2, pngPath: probeLearned, log: log)
                MLX.Memory.clearCache()

                // C 组：当前 UI v2 解耦档（learned + 解耦 LOW_K=0.9→低清2步 σ_k≈0.9231 + 高清3步等距，
                // 总 NFE=5 < 官方 6 步直出）。对照 B 组：验证重影消除且步数更少。
                setenv("NA_H3_UPSCALER", "learned", 1)
                setenv("NA_H3_SELFLIFT_RHO", "0", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE", "1", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_LOW_K", "0.9", 1)
                setenv("NA_H3_SELFLIFT_DECOUPLE_HIGH_STEPS", "3", 1)
                let p3 = try await H3FL2VAPipeline.generateVideo(
                    prompt: prompt,
                    firstImagePath: firstImage,
                    lastImagePath: lastImage,
                    outPath: outDecoupleV2,
                    width: width,
                    height: height,
                    steps: 6,
                    latentT: latentT,
                    seed: 42,
                    sparsePolicy: .mix,
                    scheduleStyle: .official,
                    selfLiftEnabled: true,
                    log: { log("[C-decouple-v2] \($0)") }
                )
                log("✅ C 组 decouple-v2 完成：\(p3)")
                probeMp4MiddleFrame(mp4Path: p3, pngPath: probeDecoupleV2, log: log)
                code = 0
            } catch {
                log("❌ h3_53 场景 A/B 失败：\(error)")
                code = 1
            }
            sem.signal()
        }

        _ = sem.wait(timeout: .now() + 5400)
        MLX.Memory.clearCache()
        log("退出码 \(code)")
        return code
    }
}
