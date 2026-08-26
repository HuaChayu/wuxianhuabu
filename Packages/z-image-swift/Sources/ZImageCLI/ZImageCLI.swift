// zimage-cli — GPU validation lane for the Z-Image port.
//
//   swift run -c release zimage-cli --size 1024 --steps 8 --seed 42 \
//       [--snapshot <dir>] [--prompt "..."] [--base] [--guidance 4.0] [--fp32-dit] \
//       [--golden-latents <e2e safetensors>] [--out out.png]
//
// Renders on the GPU (bf16 DiT by default), reports per-stage timing and MLX memory.
// --golden-latents injects init_latents from an oracle e2e golden so renders are
// comparable against the reference PNG; otherwise MLXRandom seeding.

import Foundation
import ImageIO
import MLXRandom
import MLX
import MLXToolKit
import MLXZImage
import Tokenizers
import UniformTypeIdentifiers
import ZImage

@main
struct ZImageCLI {
    static func main() async throws {
        var args = Array(CommandLine.arguments.dropFirst())
        func opt(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
            let v = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return v
        }
        func flag(_ name: String) -> Bool {
            if let i = args.firstIndex(of: name) { args.remove(at: i); return true }
            return false
        }

        if flag("--probe-nax") {
            // NAX split-K bf16 GEMM probe (mlx-swift 0.31.3–0.31.6 corrupts bf16 GEMMs
            // at K≥10240, M·N≥2048² on M5-class GPUs). Shapes = the DiT FFN w2 at 1024².
            let (M, K, N) = (4160, 10240, 3840)
            MLXRandom.seed(0)
            let a32 = MLXRandom.normal([M, K]) * 0.02
            let b32 = MLXRandom.normal([K, N]) * 0.02
            let ref = Device.withDefaultDevice(.cpu) { matmul(a32, b32) }
            eval(ref)
            let gpu = matmul(a32.asType(.bfloat16), b32.asType(.bfloat16)).asType(.float32)
            eval(gpu)
            let x = ref.flattened()
            let y = gpu.flattened()
            let cos = MLX.sum(x * y).item(Float.self)
                / (Foundation.sqrt(MLX.sum(x * x).item(Float.self))
                   * Foundation.sqrt(MLX.sum(y * y).item(Float.self)) + 1e-12)
            print("[probe] bf16 GPU GEMM (\(M)x\(K))@(\(K)x\(N)) vs fp32 CPU: cos=\(cos)")
            print(cos > 0.99 ? "[probe] NAX GEMM OK" : "[probe] NAX GEMM BROKEN — row-chunk workaround required")
            return
        }

        // #filePath = .../mlxengine-image/WIP/z-image-swift/Sources/ZImageCLI/ZImageCLI.swift
        // 5 levels up → mlxengine-image, then weights/…
        let defaultSnapshot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("weights/Z-Image-Turbo").path

        // P8 wrapper e2e: drive the real ModelPackage surface (load → run → decode) end to
        // end, proving the engine glue (config resolution, dtype/quant selection, T2IRequest
        // plumbing, PNG encode), not just the raw pipeline. --pkg-e2e [turbo|base].
        if let tier = opt("--pkg-e2e") {
            let snapshot = opt("--snapshot") ?? (tier == "base"
                ? defaultSnapshot.replacingOccurrences(of: "Z-Image-Turbo", with: "Z-Image")
                : defaultSnapshot)
            let quant = opt("--quant").flatMap { Int($0) }
            let q: Quant = quant == 4 ? .int4 : (quant == 8 ? .int8 : .bf16)
            let cfg = tier == "base"
                ? ZImageConfiguration.base(quant: q, snapshotPath: snapshot)
                : ZImageConfiguration.turbo(quant: q, snapshotPath: snapshot)
            let pkg: any ModelPackage = tier == "base"
                ? ZImageT2IPackage(configuration: cfg)
                : ZImageTurboT2IPackage(configuration: cfg)
            print("[pkg-e2e] tier=\(tier) quant=\(q) surface="
                  + "\(type(of: pkg).manifest.surfaces[0].name)")
            let t0 = Date()
            try await pkg.load()
            print("[pkg-e2e] load: \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
            let prompt = opt("--prompt")
                ?? "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, "
                + "warm lamp glow, photorealistic"
            let t1 = Date()
            let resp = try await pkg.run(T2IRequest(
                prompt: prompt, width: 1024, height: 1024, seed: 42)) as! T2IResponse
            print("[pkg-e2e] run: \(String(format: "%.2f", Date().timeIntervalSince(t1)))s "
                  + "→ \(resp.image.width)x\(resp.image.height) \(resp.image.data.count) bytes "
                  + "(.\(resp.image.format))")
            let out = opt("--out") ?? "pkg_e2e_\(tier)_\(q.rawValue).png"
            try resp.image.data.write(to: URL(fileURLWithPath: out))
            await pkg.unload()
            print("[pkg-e2e] MLX peak \(MLX.Memory.peakMemory / (1 << 20))MB; wrote \(out)")
            return
        }

        // P7 quant gate: per-pass cosine vs the fp32 aligned DiT golden + resident floor.
        // Runs on the GPU stream (quantized matmul is Metal-only). `swift run` lane so the
        // metallib is reliable (unlike the SPM test product).
        if let gateBitsStr = opt("--quant-gate") {
            let snapshot = opt("--snapshot") ?? defaultSnapshot
            let goldensDir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("tests/goldens")
            let g = try MLX.loadArrays(
                url: goldensDir.appendingPathComponent("zimage_dit_aligned.safetensors"))
            let inX = g["in_x"]!.asType(.float32)
            let inT = g["in_t"]!.asType(.float32)
            let inCap = g["in_cap"]!.asType(.float32)
            let golden = g["out_final"]!.asType(.float32).flattened()

            for bits in gateBitsStr.split(separator: ",").compactMap({ Int($0) }) {
                let model = try ZImageWeights.loadTransformer(snapshotPath: snapshot, dtype: .bfloat16)
                MLX.Memory.clearCache()
                let bf16MB = MLX.Memory.activeMemory / (1 << 20)
                ZImageWeights.quantizeDiT(model, bits: bits)
                MLX.Memory.clearCache()
                let qMB = MLX.Memory.activeMemory / (1 << 20)
                let out = model([inX], t: inT, capFeats: [inCap])[0].asType(.float32).flattened()
                eval(out)
                let cos = MLX.sum(out * golden).item(Float.self)
                    / (Foundation.sqrt(MLX.sum(out * out).item(Float.self))
                       * Foundation.sqrt(MLX.sum(golden * golden).item(Float.self)) + 1e-12)
                print("[quant-gate] int\(bits): per-pass cos=\(String(format: "%.6f", cos)) "
                      + "| DiT resident bf16=\(bf16MB)MB → int\(bits)=\(qMB)MB")
            }
            return
        }

        let isBase = flag("--base")
        let snapshot = opt("--snapshot") ?? (isBase
            ? defaultSnapshot.replacingOccurrences(of: "Z-Image-Turbo", with: "Z-Image")
            : defaultSnapshot)
        let size = Int(opt("--size") ?? "1024")!
        let steps = Int(opt("--steps") ?? (isBase ? "28" : "8"))!
        let guidance = Float(opt("--guidance") ?? (isBase ? "4.0" : "0.0"))!
        let seed = UInt64(opt("--seed") ?? "42")!
        let prompt = opt("--prompt")
            ?? "A lighthouse on a stormy coast at dusk, dramatic clouds, crashing waves, "
            + "warm lamp glow, photorealistic"
        let negativePrompt = opt("--negative")
        let outPath = opt("--out") ?? "zimage_\(isBase ? "base" : "turbo")_\(size)_s\(seed).png"
        let ditDtype: DType = flag("--fp32-dit") ? .float32 : .bfloat16
        let quantBits = Int(opt("--quant") ?? "0")!   // 0 = none; 8 or 4 = quantize DiT
        let goldenLatents = opt("--golden-latents")

        print("[cli] snapshot=\(snapshot)")
        print("[cli] \(size)x\(size) steps=\(steps) guidance=\(guidance) dit=\(ditDtype) seed=\(seed)")

        func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
            let t0 = Date()
            let r = try body()
            print("[cli] \(label): \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
            return r
        }

        // scheduler config from the snapshot (Turbo shift 3.0 / Base shift 6.0)
        struct SchedCfg: Codable {
            var shift: Float?
            var use_dynamic_shifting: Bool?
            var num_train_timesteps: Int?
        }
        let schedCfg = try JSONDecoder().decode(
            SchedCfg.self,
            from: Data(contentsOf: URL(fileURLWithPath: snapshot)
                .appendingPathComponent("scheduler/scheduler_config.json")))
        let scheduler = FlowMatchEulerDiscreteScheduler(
            numTrainTimesteps: schedCfg.num_train_timesteps ?? 1000,
            shift: schedCfg.shift ?? 3.0,
            useDynamicShifting: schedCfg.use_dynamic_shifting ?? false)

        let transformer = try timed("load DiT (\(ditDtype))") {
            try ZImageWeights.loadTransformer(snapshotPath: snapshot, dtype: ditDtype)
        }
        if quantBits == 8 || quantBits == 4 {
            timed("quantize DiT int\(quantBits)") {
                ZImageWeights.quantizeDiT(transformer, bits: quantBits)
            }
            MLX.Memory.clearCache()
            print("[cli] DiT resident post-quant: \(MLX.Memory.activeMemory / (1 << 20)) MB")
        }
        let vae = try timed("load VAE (fp32)") {
            try ZImageWeights.loadVAE(snapshotPath: snapshot, dtype: .float32)
        }
        let encoder = try timed("load text encoder (bf16)") {
            try ZImageWeights.loadTextEncoder(snapshotPath: snapshot, dtype: .bfloat16)
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: snapshot).appendingPathComponent("tokenizer"))
        let textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: tokenizer)

        var initLatents: MLXArray? = nil
        if let goldenLatents {
            let g = try MLX.loadArrays(url: URL(fileURLWithPath: goldenLatents))
            initLatents = g["init_latents"]
            print("[cli] injected init_latents from \(goldenLatents)")
        }

        // --img2img <input.png> [--strength 0.75]: encode the input → clean latent, renoise+denoise.
        var img2imgClean: MLXArray? = nil
        let strength = Float(opt("--strength") ?? "0.75")!
        if let imgPath = opt("--img2img") {
            let cs = CGColorSpace(name: CGColorSpace.sRGB)!
            let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imgPath) as CFURL, nil)!
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)!
            var buf = [UInt8](repeating: 0, count: size * size * 4)
            let c = CGContext(data: &buf, width: size, height: size, bitsPerComponent: 8,
                bytesPerRow: size * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            c.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))
            var rgb = [Float](repeating: 0, count: 3 * size * size)
            for p in 0..<(size * size) {
                rgb[p] = Float(buf[p*4]) / 127.5 - 1
                rgb[size*size + p] = Float(buf[p*4+1]) / 127.5 - 1
                rgb[2*size*size + p] = Float(buf[p*4+2]) / 127.5 - 1
            }
            let image = MLXArray(rgb, [1, 3, size, size])
            let moments = vae.encodeMoments(image)                 // [1, 32, h, w] (mean|logvar)
            let mean = moments[0..., ..<vae.latentChannels, 0..., 0...]
            img2imgClean = (mean - vae.shiftFactor) * vae.scalingFactor   // model-space clean latent
            print("[cli] img2img from \(imgPath) strength=\(strength)")
        }

        let t0 = Date()
        let result = ZImagePipeline.generate(
            transformer: transformer, vae: vae, textEncoder: textEncoder,
            scheduler: scheduler,
            prompt: prompt,
            height: size, width: size,
            numInferenceSteps: steps, guidanceScale: guidance,
            negativePrompt: negativePrompt,
            seed: seed, initLatents: initLatents,
            img2imgCleanLatent: img2imgClean, strength: strength,
            transformerDtype: ditDtype,
            onStep: { i, n in print("[cli] step \(i)/\(n)") })
        let elapsed = Date().timeIntervalSince(t0)
        print("[cli] generate: \(String(format: "%.2f", elapsed))s "
              + "(\(String(format: "%.2f", elapsed / Double(steps)))s/step incl. encode+decode)")
        print("[cli] MLX peak: \(MLX.Memory.peakMemory / (1 << 20)) MB, "
              + "active: \(MLX.Memory.activeMemory / (1 << 20)) MB")

        // encode PNG: [-1,1] NCHW -> RGBA8
        let img = result.image![0]  // [3, H, W]
        let rgb = MLX.clip(img / 2 + 0.5, min: 0, max: 1).transposed(1, 2, 0) * 255
        let u8 = rgb.asType(.uint8)
        eval(u8)
        let (h, w) = (u8.shape[0], u8.shape[1])
        var bytes = u8.asArray(UInt8.self)
        // RGB -> RGBA
        var rgba = [UInt8](repeating: 255, count: h * w * 4)
        for p in 0..<(h * w) {
            rgba[p * 4] = bytes[p * 3]
            rgba[p * 4 + 1] = bytes[p * 3 + 1]
            rgba[p * 4 + 2] = bytes[p * 3 + 2]
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let cgImage = ctx.makeImage()!
        let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: outPath) as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, cgImage, nil)
        CGImageDestinationFinalize(dest)
        print("[cli] wrote \(outPath)")
    }
}
