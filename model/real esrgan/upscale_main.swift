import Foundation
import CoreML
import AVFoundation
import CoreMedia
import CoreVideo

// 临时 CLI：用内嵌的 RealESRGAN x2plus CoreML 小模型对 h3_439.mov 做 2 倍超分
@main
struct UpscaleCLI {
    static func main() {
        let modelDir = "/Users/huachayui/Desktop/无限画布/Packages/RealESRGAN_x2plus_522_fp16.mlpackage"
        let inputPath = "/Users/huachayui/Documents/无限画布/output/视频/h3_439.mov"
        let outputPath = "/Users/huachayui/Documents/无限画布/output/视频/h3_439_2x.mov"

        let inputURL = URL(fileURLWithPath: inputPath)
        let outputURL = URL(fileURLWithPath: outputPath)

        do {
            print("加载 CoreML 模型…", terminator: ""); fflush(stdout)
            let esrgan = try RealESRGANCoreML(modelURL: URL(fileURLWithPath: modelDir))
            print("完成")

            let pipeline = RealESRGANVideoPipeline(esrgan: esrgan)
            let t0 = Date()
            var lastLog = Date()

            print("开始超分 \(inputPath) -> \(outputPath)")
            try pipeline.upscaleVideo(inputURL: inputURL, outputURL: outputURL) { done, total in
                if Date().timeIntervalSince(lastLog) > 2.0 || done == total {
                    lastLog = Date()
                    let elapsed = Date().timeIntervalSince(t0)
                    let fps = Double(done) / elapsed
                    print(String(format: "帧 %d/%d（%.1f fps, 已用 %.1fs）", done, total, fps, elapsed), terminator: "\r")
                    fflush(stdout)
                }
            }
            let elapsed = Date().timeIntervalSince(t0)
            print(String(format: "\n完成！耗时 %.1fs，输出：%@", elapsed, outputPath))
        } catch {
            print("超分失败：\(error)")
            exit(1)
        }
    }
}
