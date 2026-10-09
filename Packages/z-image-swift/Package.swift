// swift-tools-version: 6.2
// z-image-swift — Swift/MLX port of Tongyi-MAI Z-Image (Apache 2.0): 6B single-stream
// S3-DiT + Qwen3-4B text conditioner (hidden_states[-2], thinking chat template) +
// FLUX.1-dev AE decode. Serves MLXEngine `textToImage` as Base + Turbo tiers.
// Reference = the official native-PyTorch repo (../../zimage-oracle), ported isomorphic;
// mflux (../../mflux) is the MLX differential probe. See PORTING-SPEC.md — phases gate
// on fp32/CPU goldens in tests/goldens; never advance on a red gate.

import PackageDescription

let package = Package(
    name: "ZImage",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "ZImage", targets: ["ZImage"]),
        // MLXEngine wrappers: the conformant `textToImage` ModelPackages (base + Turbo).
        .library(name: "MLXZImage", targets: ["MLXZImage"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Qwen3 (text encoder backbone) + tokenizer plumbing.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // Shared env-gated performance instrument (MLX_PROFILE=1); zero overhead when unset.
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
        // MLXEngine contract (MLXToolKit) for the wrapper target only; the core `ZImage`
        // target stays engine-agnostic. ≥0.32.0 for engine-executed first-run
        // materialization (contract 1.24.0); the package no longer carries its own
        // WeightMaterializer.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.32.0"),
    ],
    targets: [
        .target(
            name: "ZImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            path: "Sources/ZImage"
        ),
        .target(
            name: "MLXZImage",
            dependencies: [
                "ZImage",
                // Explicit MLX link so unload() can call MLX.Memory.clearCache() directly.
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXZImage"
        ),
        .testTarget(
            name: "ZImageTests",
            dependencies: ["ZImage"],
            path: "Tests/ZImageTests"
        ),
        .testTarget(
            name: "MLXZImageTests",
            dependencies: [
                "MLXZImage",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                // The engine's executable MAT gate, run from this package's own suite.
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                // The engine itself, for the live prepare()-materializes gate (ZIMAGE_MAT_E2E=1).
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXZImageTests"
        ),
        // GPU validation CLI (`swift run zimage-cli`) — the reliable GPU-gate path
        // (SPM test metallib is unreliable for GPU; plain `swift run` works).
        .executableTarget(
            name: "zimage-cli",
            dependencies: [
                "ZImage",
                // The wrapper — for the --pkg-e2e mode that drives the real ModelPackage surface.
                "MLXZImage",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/ZImageCLI"
        ),
    ]
)
