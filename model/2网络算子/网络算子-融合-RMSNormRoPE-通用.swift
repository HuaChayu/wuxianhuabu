// 网络算子-融合-RMSNormRoPE-通用.swift
//
// H3 DiT P0 融合 kernel（MLXFast.metalKernel 实现，写法/用法对齐
// 网络算子-稀疏注意力-SOL通用.swift 的 kernelV6/V7）。
//
// 1. h3_fused_rmsnorm_rope：融合 q/k 通路的 rmsNormLast + applyRopePub。
//    语义参考 model/minimax h3/H3Common.swift：
//      - rmsNormLast：fp32 规约（mean(x^2)）→ rsqrt(var+eps) → weight 乘 → 回落原 dtype；
//      - applyRopePub：H3 split-half，dims [0, rotHalf) 与 [rotHalf, rot) 配对旋转，
//        [rot, hd) 直通；rot=96、rotHalf=48，cos/sin 形状以 RopeTables 实际定义为准
//        （buildRope 后为 [1, S, 1, rotHalf]，扁平索引 s*rotHalf+d）。
//    输入 x [1, S, H, hd]（H3AttnW.forward 中 normed 的四维形状）、normWeight [hd]、
//    cos/sin 表，输出同形 bf16；调用侧随后仍按原代码 transpose 到 [1, H, S, hd]。
//
// 2. h3_fused_norm_modscale：融合 rmsNormLast + modScaleShift（attention 前调制 m1 通路）。
//    语义参考 H3Common.swift rmsNormLast 与 H3Transformer.swift modScaleShift：
//      out = norm(x) * (scale[modRow] + 1) + shift[modRow]，按 plan.runs 分段取行。
//    调用侧把 runs 展开为 row→modRow 映射表 [S] UInt32（一次 gather 生成，
//    替换旧 per-run slice+concat 循环），kernel 每行按映射取 shift/scale 行，
//    输出 [S, hidden] bf16，与旧实现数值一致。
//
// 接入开关：环境变量 NA_H3_FUSE=0 可回退旧实现（默认开启），便于 A/B 数值对照。

import Foundation
import MLX
import MLXFast

/// H3 融合 kernel 封装。
public enum H3FusedKernels {

    /// NA_H3_FUSE=0 时回退旧实现；默认开启融合。
    public static var fuseEnabled: Bool {
        ProcessInfo.processInfo.environment["NA_H3_FUSE"] != "0"
    }

    // MARK: - kernel 1: h3_fused_rmsnorm_rope

    // source 为纯函数体：签名由 MLXFast.metalKernel 自动生成
    // （输入按元素数自动 const constant/device；shape 以 <name>_shape 注入；
    //   threadgroup_position_in_grid / thread_index_in_threadgroup 为默认属性）。
    private static let mslSourceRmsnormRope = """
        // grid.x = S*H rows；threadgroup = 256 线程；每 row 一行 norm+rope。
        // 输入 x [1,S,H,hd]（normed 四维）、w [hd]、ctab/stab [S,rotHalf] 扁平、params [6]
        const int S = int(params[0]);
        const int H = int(params[1]);
        const int hd = int(params[2]);
        const int rotHalf = int(params[3]);
        const int rot = int(params[4]);
        const float eps = params[5];

        const int row = int(threadgroup_position_in_grid.x); // 0..S*H-1
        const int s = row / H;
        const uint tid = thread_index_in_threadgroup;

        const device T* xr = x + row * hd;
        device T* orow = o + row * hd;

        // fp32 行内规约 mean(x^2)
        float sum = 0.0f;
        for (int j = int(tid); j < hd; j += 256) {
            const float v = float(xr[j]);
            sum += v * v;
        }
        threadgroup float smem[256];
        smem[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float rstd = rsqrt(smem[0] / float(hd) + eps);

        // norm(weight 乘) → 回落 bf16（与 rmsNormLast 一致）→ RoPE（split-half）
        const int cosBase = s * rotHalf;
        for (int j = int(tid); j < hd; j += 256) {
            const float nvRaw = float(xr[j]) * rstd * float(w[j]);
            const float nv = float(T(nvRaw)); // 回落原 dtype
            float outv;
            if (j < rotHalf) {
                const float nv2 = float(T(float(xr[j + rotHalf]) * rstd * float(w[j + rotHalf])));
                const float c = float(ctab[cosBase + j]);
                const float sn = float(stab[cosBase + j]);
                outv = nv * c - nv2 * sn;
            } else if (j < rot) {
                const float nv1 = float(T(float(xr[j - rotHalf]) * rstd * float(w[j - rotHalf])));
                const float c = float(ctab[cosBase + (j - rotHalf)]);
                const float sn = float(stab[cosBase + (j - rotHalf)]);
                outv = nv * c + nv1 * sn;
            } else {
                outv = nv; // [rot, hd) 直通
            }
            orow[j] = T(outv);
        }
        """

    private static let kernelRmsnormRope = MLXFast.metalKernel(
        name: "h3_fused_rmsnorm_rope",
        inputNames: ["x", "w", "ctab", "stab", "params"],
        outputNames: ["o"],
        source: mslSourceRmsnormRope
    )

    /// 融合 rmsNormLast + applyRopePub（q/k 通路）。
    /// - Parameters:
    ///   - x: [1, S, H, hd]（normed 形状），dtype 与原通路一致（bf16）
    ///   - weight: [hd] norm 权重（qNorm / kNorm）
    ///   - cos/sin: RopeTables.cos/.sin（[1, S, 1, rotHalf]）
    ///   - rot: 旋转维度（H3 主模型 96）；rotHalf = rot / 2 在 kernel 内计算
    ///   - eps: norm epsilon（cfg.normEps）
    /// - Returns: [1, S, H, hd] bf16，与 applyRopePub 输出同形
    public static func fusedRmsnormRope(_ x: MLXArray, weight: MLXArray,
                                        cos: MLXArray, sin: MLXArray,
                                        rot: Int, eps: Float) -> MLXArray {
        let shape = x.shape
        let S = shape[1]
        let H = shape[2]
        let hd = shape[3]
        let rotHalf = rot / 2
        let params = MLXArray([Float(S), Float(H), Float(hd), Float(rotHalf), Float(rot), eps])
        let out = kernelRmsnormRope(
            [x, weight, cos, sin, params],
            template: [("T", x.dtype)],
            grid: (S * H * 256, 1, 1),   // grid = 总线程数；threadGroup 256 → threadgroups = S*H
            threadGroup: (256, 1, 1),
            outputShapes: [[1, S, H, hd]],
            outputDTypes: [x.dtype]
        )[0]
        return out
    }

    // MARK: - kernel 2: h3_fused_norm_modscale

    private static let mslSourceNormModscale = """
        // grid.x = S rows；threadgroup = 256 线程；每 row 一行 norm+modscale。
        // 输入 x [S,hidden]、w [hidden]、shift/scale [numModRows,hidden]、map [S]、params [2]
        const int hidden = int(params[0]);
        const float eps = params[1];
        const int row = int(threadgroup_position_in_grid.x);
        const uint tid = thread_index_in_threadgroup;
        const uint modRow = map[row];

        const device T* xr = x + row * hidden;
        const device T* sh = shift + modRow * hidden;
        const device T* sc = scale + modRow * hidden;
        device T* orow = o + row * hidden;

        // fp32 行内规约 mean(x^2)
        float sum = 0.0f;
        for (int j = int(tid); j < hidden; j += 256) {
            const float v = float(xr[j]);
            sum += v * v;
        }
        threadgroup float smem[256];
        smem[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) smem[tid] += smem[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float rstd = rsqrt(smem[0] / float(hidden) + eps);

        // out = norm(x) * (scale[modRow] + 1) + shift[modRow]
        for (int j = int(tid); j < hidden; j += 256) {
            const float nv = float(T(float(xr[j]) * rstd * float(w[j]))); // 回落原 dtype
            const float outv = nv * (float(sc[j]) + 1.0f) + float(sh[j]);
            orow[j] = T(outv);
        }
        """

    private static let kernelNormModscale = MLXFast.metalKernel(
        name: "h3_fused_norm_modscale",
        inputNames: ["x", "w", "shift", "scale", "map", "params"],
        outputNames: ["o"],
        source: mslSourceNormModscale
    )

    // MARK: - kernel 3: h3_fused_mod_gate

    private static let mslSourceModGate = """
        // grid.x = S rows；threadgroup = 256 线程；纯元素级 out = x + other*gate（无规约）。
        // 输入 x/other [S,hidden]、gate [numModRows,hidden]、map [S]、params [1]
        const int hidden = int(params[0]);
        const int row = int(threadgroup_position_in_grid.x);
        const uint tid = thread_index_in_threadgroup;
        const uint modRow = map[row];

        const device T* xr = x + row * hidden;
        const device T* orr = other + row * hidden;
        const device T* gr = gate + modRow * hidden;
        device T* orow = o + row * hidden;

        for (int j = int(tid); j < hidden; j += 256) {
            const float outv = float(xr[j]) + float(orr[j]) * float(gr[j]);
            orow[j] = T(outv);
        }
        """

    private static let kernelModGate = MLXFast.metalKernel(
        name: "h3_fused_mod_gate",
        inputNames: ["x", "other", "gate", "map", "params"],
        outputNames: ["o"],
        source: mslSourceModGate
    )

    // MARK: - 回退实现（NA_H3_FUSE=0 时使用，等价旧全局 slice+concat 循环）
    // 旧 public func modScaleShift / modGate 已删除，回退语义内联于此，开关不依赖全局函数。

    private static func fallbackModScaleShift(_ x: MLXArray, shift: MLXArray,
                                              scale: MLXArray, runs: [ModRun]) -> MLXArray {
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(runs.count)
        for r in runs {
            let seg = x[Int(r.start) ..< Int(r.end)]
            let sc = scale[Int(r.modRow) ..< Int(r.modRow) + 1]
            let sh = shift[Int(r.modRow) ..< Int(r.modRow) + 1]
            pieces.append(seg * (sc + MLXArray(1.0)) + sh)
        }
        return concatenated(pieces, axis: 0)
    }

    private static func fallbackModGate(_ x: MLXArray, gate: MLXArray,
                                        other: MLXArray, runs: [ModRun]) -> MLXArray {
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(runs.count)
        for r in runs {
            let xs = x[Int(r.start) ..< Int(r.end)]
            let os = other[Int(r.start) ..< Int(r.end)]
            let g = gate[Int(r.modRow) ..< Int(r.modRow) + 1]
            pieces.append(xs + os * g)
        }
        return concatenated(pieces, axis: 0)
    }

    // MARK: - 封装入口

    /// 把 plan.runs 展开为 row→modRow 映射表 [S] UInt32（一次 gather 生成）。
    /// - Returns: [S] uint32，每行取 runs 中该行所属段的 modRow。
    public static func buildRowModMap(count: Int, runs: [ModRun]) -> MLXArray {
        var map = [UInt32](repeating: 0, count: count)
        for r in runs {
            let s = max(0, min(count, Int(r.start)))
            let e = max(0, min(count, Int(r.end)))
            if e > s {
                for i in s ..< e {
                    map[i] = r.modRow
                }
            }
        }
        return MLXArray(map)
    }

    /// 融合 rmsNormLast + modScaleShift（attention 前调制 m1 通路 / MLP 前调制 m2 通路）。
    /// - Parameters:
    ///   - x: [S, hidden]（norm 输入，如 h 或 hAttn）
    ///   - weight: [hidden] norm 权重（b.norm1 / b.norm2）
    ///   - shift/scale: [numModRows, hidden]（mods[0]/mods[1] 或 mods[3]/mods[4]）
    ///   - runs: plan.runs（分段取行）
    ///   - eps: norm epsilon
    /// - Returns: [S, hidden] bf16，与 rmsNormLast + modScaleShift 数值一致
    public static func fusedNormModscale(_ x: MLXArray, weight: MLXArray,
                                         shift: MLXArray, scale: MLXArray,
                                         runs: [ModRun], eps: Float) -> MLXArray {
        guard fuseEnabled else {
            return fallbackNormModscale(x, weight: weight, shift: shift, scale: scale,
                                        runs: runs, eps: eps)
        }
        let S = x.shape[0]
        let hidden = x.shape[1]
        let map = buildRowModMap(count: S, runs: runs)
        let params = MLXArray([Float(hidden), eps])
        let out = kernelNormModscale(
            [x, weight, shift, scale, map, params],
            template: [("T", x.dtype)],
            grid: (S * 256, 1, 1),       // grid = 总线程数；threadGroup 256 → threadgroups = S
            threadGroup: (256, 1, 1),
            outputShapes: [[S, hidden]],
            outputDTypes: [x.dtype]
        )[0]
        return out
    }

    private static func fallbackNormModscale(_ x: MLXArray, weight: MLXArray,
                                             shift: MLXArray, scale: MLXArray,
                                             runs: [ModRun], eps: Float) -> MLXArray {
        fallbackModScaleShift(rmsNormLast(x, weight: weight, eps: eps),
                              shift: shift, scale: scale, runs: runs)
    }

    /// 融合 rmsNormLast + 单行 modScaleShift（finalHead 用，映射表全 0）。
    /// shift/scale 为已取出的单行 [1, hidden]；kernel 每行 modRow=0，无 [S,hidden] 中间展开。
    /// - Returns: [S, hidden]，与 nn*(scale+1)+shift 数值一致
    public static func fusedNormModscaleRow(_ x: MLXArray, weight: MLXArray,
                                            shift: MLXArray, scale: MLXArray,
                                            eps: Float) -> MLXArray {
        guard fuseEnabled else {
            let nn = rmsNormLast(x, weight: weight, eps: eps)
            return nn * (scale + MLXArray(1.0)) + shift
        }
        let S = x.shape[0]
        let hidden = x.shape[1]
        let map = MLXArray.zeros([S], dtype: .uint32)   // 全 0 → 每行取 shift/scale 第 0 行
        let params = MLXArray([Float(hidden), eps])
        let out = kernelNormModscale(
            [x, weight, shift, scale, map, params],
            template: [("T", x.dtype)],
            grid: (S * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[S, hidden]],
            outputDTypes: [x.dtype]
        )[0]
        return out
    }

    /// 融合 modGate：out = x + other·gate[modRow]，按 runs 分段取行。
    /// - Parameters:
    ///   - x/other: [S, hidden]（如 h 与 at / mo）
    ///   - gate: [numModRows, hidden]（mods[2] / mods[5]）
    ///   - runs: plan.runs（分段取行）
    /// - Returns: [S, hidden] bf16，与旧 modGate 数值一致
    public static func fusedModGate(_ x: MLXArray, gate: MLXArray,
                                    other: MLXArray, runs: [ModRun]) -> MLXArray {
        guard fuseEnabled else {
            return fallbackModGate(x, gate: gate, other: other, runs: runs)
        }
        let S = x.shape[0]
        let hidden = x.shape[1]
        let map = buildRowModMap(count: S, runs: runs)
        let params = MLXArray([Float(hidden)])
        let out = kernelModGate(
            [x, other, gate, map, params],
            template: [("T", x.dtype)],
            grid: (S * 256, 1, 1),       // grid = 总线程数；threadGroup 256 → threadgroups = S
            threadGroup: (256, 1, 1),
            outputShapes: [[S, hidden]],
            outputDTypes: [x.dtype]
        )[0]
        return out
    }

    // MARK: - 数值自检

    /// 合成数据自检：kernel vs MLX 参考实现，打印逐项 PASS/FAIL。
    /// 返回 0 = 全部通过；非 0 = 有失败项。
    public static func selfTest() -> Int32 {
        print("[FUSETEST] begin"); fflush(stdout)
        var fail = false
        func maxAbs(_ x: MLXArray) -> Float {
            let f = abs(x).asType(.float32).reshaped([-1])
            var mx: Float = 0
            let step = max(1, f.size / 4000)
            for i in stride(from: 0, to: f.size, by: step) {
                mx = max(mx, f[i].item(Float.self))
            }
            return mx
        }
        do {
            // 局部等价参考实现（旧全局 modScaleShift / modGate 已删除，内联于此供数值对照）
            func refModScaleShift(_ x: MLXArray, shift: MLXArray, scale: MLXArray,
                                  runs: [ModRun]) -> MLXArray {
                var pieces: [MLXArray] = []
                for r in runs {
                    let seg = x[Int(r.start) ..< Int(r.end)]
                    let sc = scale[Int(r.modRow) ..< Int(r.modRow) + 1]
                    let sh = shift[Int(r.modRow) ..< Int(r.modRow) + 1]
                    pieces.append(seg * (sc + MLXArray(1.0)) + sh)
                }
                return concatenated(pieces, axis: 0)
            }
            func refModGate(_ x: MLXArray, gate: MLXArray, other: MLXArray,
                            runs: [ModRun]) -> MLXArray {
                var pieces: [MLXArray] = []
                for r in runs {
                    let xs = x[Int(r.start) ..< Int(r.end)]
                    let os = other[Int(r.start) ..< Int(r.end)]
                    let g = gate[Int(r.modRow) ..< Int(r.modRow) + 1]
                    pieces.append(xs + os * g)
                }
                return concatenated(pieces, axis: 0)
            }
            func refNormModscaleRow(_ x: MLXArray, weight: MLXArray,
                                    shift: MLXArray, scale: MLXArray,
                                    eps: Float) -> MLXArray {
                rmsNormLast(x, weight: weight, eps: eps) * (scale + MLXArray(1.0)) + shift
            }

            // ---- Case 1: fusedRmsnormRope ----
            let S = 8, H = 4, hd = 128, rot = 96, rotHalf = 48
            let eps: Float = 1e-5
            let x = MLXRandom.normal([1, S, H, hd]).asType(.bfloat16)
            let w = MLXRandom.normal([hd]).asType(.bfloat16)
            let cosT = MLXRandom.normal([1, S, 1, rotHalf]).asType(.bfloat16)
            let sinT = MLXRandom.normal([1, S, 1, rotHalf]).asType(.bfloat16)
            MLX.eval(x, w, cosT, sinT)

            let fused = fusedRmsnormRope(x, weight: w, cos: cosT, sin: sinT, rot: rot, eps: eps)
            let ref = applyRopePub(rmsNormLast(x, weight: w, eps: eps),
                                   cos: cosT, sin: sinT, rot: rot)
            MLX.eval(fused, ref)
            // 容差 0.25 ≈ 16 个 bf16 ULP（幅值 ~16 元素处），覆盖规约顺序不同导致的舍入边界差
            let mx = maxAbs(fused.asType(.float32) - ref.asType(.float32))
            print(String(format: "[FUSETEST] rmsnorm_rope maxAbs=%.6f => %@", mx, mx < 0.25 ? "PASS" : "FAIL"))
            if mx >= 0.25 { fail = true }

            // ---- Case 2: fusedNormModscale ----
            let S2 = 16, hidden = 64
            let x2 = MLXRandom.normal([S2, hidden]).asType(.bfloat16)
            let w2 = MLXRandom.normal([hidden]).asType(.bfloat16)
            let sh2 = MLXRandom.normal([6, hidden]).asType(.bfloat16)
            let sc2 = MLXRandom.normal([6, hidden]).asType(.bfloat16)
            MLX.eval(x2, w2, sh2, sc2)
            // 3 段 runs（0..5 -> modRow0, 5..11 -> modRow3, 11..16 -> modRow5）
            let runs: [ModRun] = [
                ModRun(start: 0, end: 5, modRow: 0),
                ModRun(start: 5, end: 11, modRow: 3),
                ModRun(start: 11, end: 16, modRow: 5),
            ]
            let fused2 = fusedNormModscale(x2, weight: w2, shift: sh2, scale: sc2, runs: runs, eps: eps)
            let ref2 = refModScaleShift(rmsNormLast(x2, weight: w2, eps: eps),
                                        shift: sh2, scale: sc2, runs: runs)
            MLX.eval(fused2, ref2)
            let mx2 = maxAbs(fused2.asType(.float32) - ref2.asType(.float32))
            print(String(format: "[FUSETEST] norm_modscale maxAbs=%.6f => %@", mx2, mx2 < 0.25 ? "PASS" : "FAIL"))
            if mx2 >= 0.25 { fail = true }

            // ---- Case 3: fusedModGate ----
            let S3 = 16, hidden3 = 64
            let x3 = MLXRandom.normal([S3, hidden3]).asType(.bfloat16)
            let o3 = MLXRandom.normal([S3, hidden3]).asType(.bfloat16)
            let g3 = MLXRandom.normal([6, hidden3]).asType(.bfloat16)
            MLX.eval(x3, o3, g3)
            let fused3 = fusedModGate(x3, gate: g3, other: o3, runs: runs)
            let ref3 = refModGate(x3, gate: g3, other: o3, runs: runs)
            MLX.eval(fused3, ref3)
            let mx3 = maxAbs(fused3.asType(.float32) - ref3.asType(.float32))
            print(String(format: "[FUSETEST] mod_gate maxAbs=%.6f => %@", mx3, mx3 < 0.25 ? "PASS" : "FAIL"))
            if mx3 >= 0.25 { fail = true }

            // ---- Case 4: fusedNormModscaleRow（finalHead 单行调制）----
            let S4 = 12, hidden4 = 64
            let x4 = MLXRandom.normal([S4, hidden4]).asType(.bfloat16)
            let w4 = MLXRandom.normal([hidden4]).asType(.bfloat16)
            let sh4 = MLXRandom.normal([1, hidden4]).asType(.bfloat16)
            let sc4 = MLXRandom.normal([1, hidden4]).asType(.bfloat16)
            MLX.eval(x4, w4, sh4, sc4)
            let fused4 = fusedNormModscaleRow(x4, weight: w4, shift: sh4, scale: sc4, eps: eps)
            let ref4 = refNormModscaleRow(x4, weight: w4, shift: sh4, scale: sc4, eps: eps)
            MLX.eval(fused4, ref4)
            let mx4 = maxAbs(fused4.asType(.float32) - ref4.asType(.float32))
            print(String(format: "[FUSETEST] norm_modscale_row maxAbs=%.6f => %@", mx4, mx4 < 0.25 ? "PASS" : "FAIL"))
            if mx4 >= 0.25 { fail = true }

            // ---- Case 5: block 调制链整体一致性（m1 → modGate → m2 → modGate，
            //    等价 forward 中 NA_H3_FUSE=1 与回退路径逐段数值一致）----
            let S5 = 16, hidden5 = 64
            let h5 = MLXRandom.normal([S5, hidden5]).asType(.bfloat16)
            let at5 = MLXRandom.normal([S5, hidden5]).asType(.bfloat16)
            let mo5 = MLXRandom.normal([S5, hidden5]).asType(.bfloat16)
            let wN1 = MLXRandom.normal([hidden5]).asType(.bfloat16)
            let wN2 = MLXRandom.normal([hidden5]).asType(.bfloat16)
            // forward 中 mods[0..5] 各为完整 [numModRows, hidden] 表（rows addressed by run.modRow）
            let shiftMSA = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            let scaleMSA = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            let gateMSA = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            let shiftMLP = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            let scaleMLP = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            let gateMLP = MLXRandom.normal([6, hidden5]).asType(.bfloat16)
            MLX.eval(h5, at5, mo5, wN1, wN2, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP)
            // fused 链（NA_H3_FUSE=1）
            let m1f = fusedNormModscale(h5, weight: wN1, shift: shiftMSA, scale: scaleMSA, runs: runs, eps: eps)
            let haf = fusedModGate(h5, gate: gateMSA, other: at5, runs: runs)
            let m2f = fusedNormModscale(haf, weight: wN2, shift: shiftMLP, scale: scaleMLP, runs: runs, eps: eps)
            let h1f = fusedModGate(haf, gate: gateMLP, other: mo5, runs: runs)
            // ref 链（NA_H3_FUSE=0 回退语义）
            let m1r = refModScaleShift(rmsNormLast(h5, weight: wN1, eps: eps), shift: shiftMSA, scale: scaleMSA, runs: runs)
            let har = refModGate(h5, gate: gateMSA, other: at5, runs: runs)
            let m2r = refModScaleShift(rmsNormLast(har, weight: wN2, eps: eps), shift: shiftMLP, scale: scaleMLP, runs: runs)
            let h1r = refModGate(har, gate: gateMLP, other: mo5, runs: runs)
            MLX.eval(m1f, haf, m2f, h1f, m1r, har, m2r, h1r)
            for (name, a, b) in [("m1", m1f, m1r), ("hAttn", haf, har), ("m2", m2f, m2r), ("h_out", h1f, h1r)] {
                let mx = maxAbs(a.asType(.float32) - b.asType(.float32))
                print(String(format: "[FUSETEST] chain_%@ maxAbs=%.6f => %@", name, mx, mx < 0.25 ? "PASS" : "FAIL"))
                if mx >= 0.25 { fail = true }
            }

            // ---- Case 6: denseLinear allowBF16 提前降宽（fp32 岛 vs bf16 路径对照，报告性质）----
            // 模拟 patch 投影：随机 fp32 latent + 真形状 patch 权重 [in, 5376]，对照两路径
            // 输出 maxAbs/rms（记录典型误差量级，无强制阈值），并打印 activeMemory 增量差。
            let inDim6 = 128, outDim6 = 5376, nRows6 = 64
            let x6 = MLXRandom.normal([nRows6, inDim6]).asType(.float32)
            let w6 = MLXRandom.normal([inDim6, outDim6]).asType(.float32)
            let b6 = MLXRandom.normal([outDim6]).asType(.float32)
            MLX.eval(x6, w6, b6)
            let memBase = MLX.Memory.activeMemory
            let outF32 = H3TensorOps.denseLinear(x6, w6, b6)
            MLX.eval(outF32)
            let memAfterF32 = MLX.Memory.activeMemory
            let outBF16 = H3TensorOps.denseLinear(x6, w6, b6, allowBF16: true)
            MLX.eval(outBF16)
            let memAfterBF16 = MLX.Memory.activeMemory
            let diff6 = outF32.asType(.float32) - outBF16.asType(.float32)
            let mx6 = maxAbs(diff6)
            let rms6 = diff6.square().mean().sqrt().item(Float.self)
            print(String(format: "[FUSETEST] f32patch_bf16 maxAbs=%.6f rms=%.6f (报告性质，无阈值)", mx6, rms6))
            print(String(format: "[FUSETEST] f32patch_bf16 memDelta f32=+%.1fMB bf16=+%.1fMB (含惰性求值噪声)",
                         Double(memAfterF32 - memBase) / 1048576.0,
                         Double(memAfterBF16 - memBase) / 1048576.0))
        } catch {
            print("[FUSETEST] threw: \(error)"); fflush(stdout)
            fail = true
        }
        print(fail ? "[FUSETEST] FAIL" : "[FUSETEST] PASS"); fflush(stdout)
        return fail ? 1 : 0
    }
}
