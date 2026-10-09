# z-image-swift

Swift/MLX port of Alibaba Tongyi's **[Z-Image](https://github.com/Tongyi-MAI/Z-Image)**
(Apache-2.0): a 6.15B single-stream **S3-DiT** text-to-image model — Qwen3-4B thinking-template
conditioning (`hidden_states[-2]`) → single-stream DiT → FLUX.1-dev AE decode. Ships MLXEngine's
**`textToImage`** capability in two tiers.

Two products:
- **`ZImage`** — the engine-agnostic generator core (DiT + FLUX.1 AE + Qwen3-4B encoder +
  FlowMatchEuler scheduler + pipeline).
- **`MLXZImage`** — the conformant MLXEngine wrappers: `ZImageT2IPackage` (Base, ~28-step CFG,
  surface `z-image-t2i`) and `ZImageTurboT2IPackage` (Turbo, distilled 8-step no-CFG, surface
  `z-image-turbo-t2i`), both `textToImage`, selected by `PackageID`.

> **Status: complete · wrapped · GPU-validated.** Parity vs the PyTorch goldens (fp32/CPU stream):
> DiT cosine **≥0.9999999** (both aligned + padded cases) · VAE decode **118 dB** · encoder token
> ids **exact** + features cosine **1.0000000** · full-pipeline e2e **105–108 dB** (256²/CPU).
> GPU bf16 1024²/8-step Turbo renders a sharp, prompt-faithful image in **~13 s** (25.7 GB peak at
> int4 — a 16 GB-tier fit). Base + CFG (guidance 4.0, negative prompt) parity-locked at
> cosine 1.000000. Resolution sweep 512²/1024²/1536² all coherent. Quant: **int8** DiT 6.4 GB
> (cos 0.9998), **int4** DiT 3.5 GB (renders sharp). Wrapped as two `textToImage` ModelPackages,
> validated through the real `load → run → decode` surface — the third public T2I backer alongside
> Lens and ERNIE-Image (co-resident, selectable by PackageID).

## Consuming it

Public + version-tagged on github.com/xocialize — add by tagged URL:
`.package(url: "https://github.com/xocialize/z-image-swift", from: "0.1.0")`, then import
`MLXZImage` (the conformant packages) or `ZImage` (the bare generator).

```swift
import MLXZImage
import MLXToolKit

// Turbo tier (fast, 8-step no-CFG). quant: .int4 → ~6 GB pipeline (16 GB Mac); .bf16 → ~20 GB.
let package = ZImageTurboT2IPackage(configuration: .turbo(
    quant: .int4,
    snapshotPath: "<root>/Z-Image-Turbo"))   // or nil → auto-materialize from mlx-community
try await package.load()
let response = try await package.run(T2IRequest(
    prompt: "a lighthouse on a stormy coast at dusk, photorealistic",
    width: 1024, height: 1024, seed: 42)) as! T2IResponse
// response.image: canonical Image (.png)
await package.unload()
```

Base tier: `ZImageT2IPackage(configuration: .base(quant: .bf16, snapshotPath: ...))` — non-distilled,
~28-step with CFG + negative prompts; the quality / LoRA-substrate tier.

**Image-to-image (v0.2.2):** both tiers also expose an `imageEdit` surface — re-generate from an
input image conditioned on a prompt. `metaData["strength"]` (0–1, **default 0.75**) controls how much
of the input is preserved vs redrawn. Encode → renoise at the strength-picked sigma → denoise. As a
rough guide: **~0.5** relight/restyle (holds everything), **~0.75** visible edit + composition kept
(the default), **~0.9** new environment / time-of-day swaps (subject also redraws — e.g. day→night).

> **v0.2.1 fix + v0.2.2 default:** the `strength → t_start` step count floors the *difference* exactly
> like diffusers (`int(steps − steps·strength)`), not the product — the prior off-by-one started one
> denoise step late, so edits came back near-identity on the distilled 8-step Turbo. v0.2.2 also raises
> the default strength from diffusers' 0.6 to **0.75**: on Turbo, 0.6 barely moves global edits (the
> preserved low-frequency content wins), while 0.75 gives a visible edit on most prompts and still
> holds composition.

```swift
let out = try await package.run(IEditRequest(
    images: [inputImage], prompt: "...in an oil-painting style",
    metaData: ["strength": .double(0.9)]))  // omit for the 0.75 default; raise for scene swaps
    as! IEditResponse
```

## Weights

Published bf16 snapshots on Hugging Face (Apache-2.0), materialized automatically by the engine
(`WeightSourcing`) or pointed at via `snapshotPath`:
- Turbo: [`mlx-community/Z-Image-Turbo-bf16`](https://huggingface.co/mlx-community/Z-Image-Turbo-bf16)
- Base: [`mlx-community/Z-Image-bf16`](https://huggingface.co/mlx-community/Z-Image-bf16)

int8/int4 are produced at load time from the bf16 snapshot (correct resident footprint;
pre-quantized repos are a later download-size optimization). Upstream:
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) ·
[Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image).

## Architecture notes (S3-DiT)

- Single stream: image VAE tokens + caption tokens concatenated (image first) through 2 noise-refiner
  + 2 context-refiner blocks, then 30 main blocks. Scale-only tanh-gated AdaLN (256-dim embed),
  SwiGLU FFN, QK-RMSNorm, 3-axis RoPE (θ=256, axes 32/48/48).
- Text encoder = Qwen3-4B, penultimate hidden state via the thinking-mode chat template.
- VAE = FLUX.1-dev AE (16 latent channels, `scaling 0.3611 / shift 0.1159`, force-upcast fp32 decode).
- Turbo: distilled 8-step, guidance 0 (no CFG), scheduler static shift 3.0. Base: ~28-step CFG,
  shift 6.0. **Turbo has low seed variance — a model trait, not a bug** (Base is the diverse/LoRA tier).

## Gates

- Always-on: `swift test --filter P1MathTests` (scheduler + RoPE) and `--filter MLXZImageTests`
  (offline MAT + conformance, no weights).
- Parity/GPU gates need the weights + goldens: `ZIMAGE_PARITY=1 swift test` (fp32/CPU), and the
  GPU lane via `swift run -c release zimage-cli` (`--quant-gate 8,4`, `--pkg-e2e turbo --quant 4`, …).

License: port code MIT; model weights Apache-2.0 (Tongyi-MAI).
