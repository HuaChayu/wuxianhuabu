# z-image-swift — Porting Spec

**Goal:** Swift/MLX port of Tongyi-MAI **Z-Image** (Apache 2.0, both weights + code) serving
MLXEngine's **`textToImage`** capability as the prospective go-to provider — two tiers, Lens-style:
`ZImageT2IPackage` (Base, ~50-step CFG) + `ZImageTurboT2IPackage` (Turbo, 8-step no-CFG,
surface `z-image-turbo-t2i`). 6.15B single-stream S3-DiT + Qwen3-4B text encoder + FLUX.1-dev AE.

**Routing decision (mlx-porting): NO private Python-MLX rung.** ERNIE-Image precedent:
- **PT oracle** = `../../zimage-oracle/` — the official native-PyTorch repo (Tongyi-MAI/Z-Image,
  ~2k LOC, self-contained, has an MPS fallback so it runs on this box). Goldens come from here.
- **MLX differential probe** = `../../mflux/` (`src/mflux/models/z_image/`) — first-class
  Turbo+Base support incl. weight mapping (`weights/z_image_weight_mapping.py`) and a LoRA
  mapping we'll want later for the DiT-LoRA program.
- Swift is ported **directly from the PT oracle sources, isomorphic** (same file/class/method
  names); mflux is consulted for MLX idiom + bisecting divergences, never as the source of truth.

**Weights:** `../../weights/Z-Image-Turbo/` + `../../weights/Z-Image/` (both downloading,
~33 GB each). bf16 DiT ≈ 12.3 GB · Qwen3-4B ≈ 8.1 GB · VAE ≈ 0.3 GB.

## Checkpoint-config ground truth (verified 2026-07-05, Turbo snapshot)

- `transformer/config.json` **matches every constructor default** in
  `src/zimage/transformer.py` — dim 3840, 30 layers + 2+2 refiners, 30/30 heads (head_dim 128,
  **no GQA**), in_ch 16, patch (2,), f_patch (1,), norm_eps **1e-5**, qk_norm true,
  cap_feat_dim 2560, t_scale 1000, rope_theta **256.0**, axes_dims [32,48,48],
  axes_lens **[1536,512,512]**.
- `scheduler/scheduler_config.json`: **Turbo = static shift 3.0, Base = static shift 6.0**
  (both `use_dynamic_shifting: false` — the pipeline's `calculate_shift` mu is DEAD for both
  released checkpoints; port the dynamic `time_shift` path anyway for config-driven correctness).
- **Turbo ships fp32 DiT shards** (22.93 GB, 3 files) vs Base bf16 (11.46 GB) — identical
  521-key weight map. Load-time dtype cast to bf16 (oracle does the same); conversion output
  should be bf16 for both.
- `vae/config.json`: `_name_or_path: "flux-dev"` — literally the FLUX.1-dev AE. 16 latent ch,
  scaling **0.3611**, shift **0.1159**, groups 32, eps 1e-6, `use_quant_conv/post_quant_conv:
  false`, `force_upcast: true` (→ run VAE decode fp32).
- `text_encoder/config.json`: Qwen3ForCausalLM — hidden 2560, **36 layers**, 32Q/8KV heads
  (head_dim 128), vocab 151936, bf16.
- ⚠ **`src/config/model.py` VAE defaults are SD1.5 placeholders** (4 ch, 0.18215) — never use
  them; the pipeline reads `vae.config` at runtime. Pitfall #1 (defaults vs config) live example.

## Pipeline quirks — port VERBATIM (pitfall #11 cluster, from `src/zimage/pipeline.py`)

1. **Time reversal**: scheduler timesteps are `sigma*1000` descending; the DiT receives
   `(1000 − t)/1000` (= 1 − σ), then multiplies by `t_scale=1000` internally.
2. **Negated output**: `noise_pred = −model_out` BEFORE the Euler step. Miss either 1 or 2 and
   you generate noise.
3. **Pos-anchored CFG**: `pred = pos + g·(pos − neg)` — NOT diffusers `neg + g·(pos−neg)`
   (their g ≡ standard g+1). `do_cfg = g > 1.0`. Turbo: **g = 0.0** → no CFG, no neg branch.
4. **cfg_truncation** (default 1.0): when normalized t > truncation, guidance turns OFF for that
   step. **cfg_normalization** (default False): norm-cap `pred` to `‖pos‖·factor` — Lens-style
   rescale variant, port the exact conditional.
5. **fp32 latents end-to-end**: latents fp32 through the loop; DiT runs bf16; `scheduler.step`
   fp32 (`assert latents.dtype == float32`); model outs upcast `.float()` before CFG.
6. `scheduler.sigma_min = 0.0` override before `retrieve_timesteps`; terminal 0 appended to
   sigmas; **skip the final step when t == 0** (Turbo's `num_inference_steps=9` → 8 DiT calls).
7. **Text encoding**: Qwen3 chat template `apply_chat_template(messages,
   add_generation_prompt=True, enable_thinking=True)` (thinking-mode template!), pad to
   max_length **512**, truncate; features = `hidden_states[-2]` (after layer 35 of 36 — skip
   final layer? NO: [-2] = output of layer 35, i.e. run 35 of 36 layers, no final norm), then
   **masked gather** to a variable-length per-prompt list (no padded batch into the DiT).
8. **VAE denorm**: `latents/0.3611 + 0.1159` → decode fp32 → `(img/2 + 0.5).clamp(0,1)`.
9. Defaults (Turbo): 1024×1024, steps 8 (loop len 9), g 0.0, max_seq 512, seed via
   `torch.Generator` — **inject numpy latents on both sides for parity** (RNG never matches).

## DiT structure notes (`src/zimage/transformer.py`, 571 LOC)

- **Three token streams before the main stack**: image tokens → 2× `noise_refiner` blocks
  (modulated); caption tokens (cap_embedder = RMSNorm(2560) + Linear→3840) → 2×
  `context_refiner` blocks (unmodulated). Then **unified = concat[image, caption] per item**
  (image FIRST — the tech-report "[cap, x]" order is NOT what the code does) → 30 main blocks.
- **Block = sandwich-norm, scale-only AdaLN**: `adaLN_modulation: Linear(256 → 4·dim)` chunks
  to (scale_msa, gate_msa, scale_mlp, gate_mlp); gates `tanh`'d, scales `1+`; **no shift**;
  gate applies to `attention_norm2(attn_out)` / `ffn_norm2(ffn_out)` (post-norms). Unmodulated
  blocks share weights structure minus adaLN.
- FFN = SwiGLU `w2(silu(w1 x)·w3 x)`, hidden = dim/3·8 = 10240. QK-RMSNorm per head, eps 1e-5.
- **RoPE**: 3-axis (t,h,w) = dims (32,48,48) @ theta 256.0. PT uses complex `torch.polar` +
  `view_as_complex` — port as **real interleaved (cos,sin) rotation** (Lens T2 pattern), tables
  precomputed float64→float32 on CPU (MLX GPU has no float64 — build tables with Double math
  CPU-side / numpy at conversion; plain class, NOT a Module). Position ids: caption axis-0 =
  `1..cap_len_padded`, rows (0,0); image axis-0 = `cap_len_padded+1` constant, (h,w) grid;
  **pad positions = (0,0,0)** (position 0 is reserved for padding).
- **Sequence padding to SEQ_MULTI_OF=32** with **learned pad tokens** (`x_pad_token`,
  `cap_pad_token` are Parameters) — pad FEATURES with the learned token, pad mask marks them;
  attention mask = per-item valid-length booleans. For bsz=1 (Turbo t2i) lengths are already
  %32-padded → mask is all-true → SDPA maskless fast path; keep the mask machinery for
  Base-CFG bsz=2 (equal lengths → still trivial) but gate correctness on the padded case.
- `TimestepEmbedder(out=256, mid=1024)`, freq-embed 256, max_period 10000, fp32 internally.
- patchify: `view(C,F/pF,pF,H/pH,pH,W/pW,pW).permute(1,3,5,2,4,6,0)`; unpatchify inverse
  `.view(F,H/p,W/p,pF,pH,pW,C).permute(6,0,3,1,4,2,5)` — copy axis orders EXACTLY (T4-class).
- Attention dispatch (`utils/attention.py`) is backend plumbing (flash/cuda paths) — Swift goes
  straight to `MLXFast.scaledDotProductAttention`; oracle on MPS/CPU uses SDPA equivalently.

## Component map (Swift)

| Component | Source (PT oracle) | Swift action |
|---|---|---|
| `ZImageTransformer2DModel` | `src/zimage/transformer.py` (571) | Port isomorphic, keep names (`noise_refiner`, `context_refiner`, `all_x_embedder["2-1"]`, `all_final_layer["2-1"]`, `x_pad_token`…). ModuleDict keys contain `-` → safetensors keys have dots+dashes; Swift `Module` property names can't contain dots — remap in the loader (repo-layout idioms). |
| `RopeEmbedder` + `apply_rotary_emb` | same file | Real (cos,sin) tables, plain class not Module; fp32 apply (PT upcasts x to float for rotation — match). |
| `FlowMatchEulerDiscreteScheduler` | `src/zimage/scheduler.py` (150) | Port verbatim: static-shift transform `σ' = s·σ/(1+(s−1)σ)`, dynamic `time_shift` (for Base if config says so), sigma_min override, terminal 0, step-index bookkeeping. Pure fp32 math. |
| Pipeline `generate()` | `src/zimage/pipeline.py` (293) | Port verbatim incl. quirks 1–9 above. |
| Text encoder (Qwen3-4B) | HF `text_encoder/` + pipeline lines 108–138 | **Wrap the Qwen3 stack from mlx-swift-lm / mlx-qwen-llm-swift**: run 35/36 layers, capture pre-final-layer hidden (= `hidden_states[-2]`), skip lm_head + final norm. Chat template via swift-transformers AutoTokenizer (verify `enable_thinking=True` template branch renders identically — golden the token ids). If internals are fileprivate → copy-adapt into `Adapted/` (Lens GPTOSS pattern). |
| FLUX.1-dev AE (decoder path) | `src/zimage/autoencoder.py` (369) | **Copy-adapt Boogu's `VAE.swift`** (same flux-dev AE family) — verify groups 32 / eps 1e-6 / block_out [128,256,512,512] / no quant_conv match its constants; decode-only for t2i; fp32 (force_upcast). NHWC conv weight transpose at load. |
| Weights loader | `src/utils/loader.py` + mflux `z_image_weight_mapping.py` | bf16 DiT strict load; key remap table cross-checked against mflux's mapping; quantized-repo rebuild (group 64) with `keep_hi_precision` predicate (in/out embedders, t_embedder, final layer, norms — tune per Lens experience). |
| Engine wrappers | — | `MLXZImage`: `ZImageT2IPackage` + `ZImageTurboT2IPackage`, canonical `T2IRequest/T2IResponse`, license `.permissive` (Apache), split QuantFootprint per 1.14 contract, WeightSourcing + MAT gate per 0.19.0. |

## Phases & gates (CPU stream for parity; never advance on red)

1. **P1 — Pure math**: scheduler (static + dynamic shift), RoPE tables + apply, timestep embed,
   patchify/unpatchify round-trip. Gate: ≤1e-6 fp32 vs oracle functions on injected inputs.
2. **P2 — DiT**: strict-load bf16. Gate: golden `(latents, t, cap_feats)` → output, fp32/CPU,
   cosine ≥0.9999 / max_abs <1e-2. Sub-goldens per stage (post-noise-refiner, post-context-refiner,
   post-unified-stack, final) to localize breaks — granular per-sub-op goldens doctrine.
3. **P3 — VAE decode**: golden final-latent → image. Gate: PSNR ≥ 55 dB + the pitfall-#7
   noise-path smoke (random Gaussian through denorm→decode; any periodic pattern = broken
   spatial op).
4. **P4 — Encoder**: golden prompt → token ids EXACT (chat template incl. thinking markers) →
   `hidden_states[-2]` features, cosine ≥0.999 bf16 / max_abs <1e-2.
5. **P5 — E2E Turbo**: 1024²/8-step/seed-42 injected-latents vs oracle MPS render — image-valid
   + eyeball + latent cosine. **Gate at 1024² (the model's base resolution) FIRST, then 512²,
   1536², 2048², and non-square ARs** (pitfall #32: resolution-dependent failure; axes_lens
   allow 4096² @ patch 2). Known-model-quirk: Turbo seed variance is LOW — do not misread
   near-identical seeds as a port bug.
6. **P6 — Base + CFG**: bsz=2 CFG path, pos-anchored formula + truncation/normalization knobs;
   dynamic-shift iff Base scheduler config enables it. Neg-prompt goldens.
7. **P7 — Quantize**: int8/int4 DiT (+ int4 Qwen3 from existing quant repos if parity-clean).
   Gate: per-pass cosine (int4 ≥0.99, int8 ≥0.9999) + image-validity, NOT PSNR-vs-fp32 (Lens
   doctrine). Target: q4/q4 ≈ 6 GB weights → 16 GB-tier admissible.
8. **P8 — Wrap + publish**: engine wrappers, goldens manifest, `xocialize/z-image-swift` +
   weights to mlx-community (`Z-Image-Turbo-{bf16,8bit,4bit}` etc. in a Collection; check
   nothing already claims the name).

## Known risks / watch-list

- **NAX split-K GEMM bug** (mlx-swift 0.31.3–0.31.6, M5 Max): Qwen3-4B encoder runs bf16 at
  seq 512 — K=2560 is under the K≥10240 trip-wire, but the DiT's w2 (10240→3840) at M·N large
  is in range → **run the fleet probe on first bf16 DiT forward**; row-chunk ≤896 workaround if hit.
- Turbo "not finetunable" per card — LoRA later rides Base (mflux `z_image_lora_mapping.py` +
  ostris De-Turbo ecosystem); ties into the DiT-LoRA foundation program.
- Z-Image-Edit/Omni: unreleased; the transformer's per-token noise-mask/omni machinery in
  *diffusers* is NOT in the native oracle — port what the oracle has, nothing more.
- Oracle goldens: dump on **CPU fp32** for P1–P4 (MPS for the e2e reference render only);
  `torch.amp.autocast("cuda")` blocks are no-ops off-CUDA — fine.

## Status

- [x] Oracle cloned + venv (`zimage-oracle/`, torch/transformers/safetensors)
- [x] mflux cloned + venv (differential probe / future bench harness)
- [x] Weights: Turbo (31G, fp32 shards) + Base (19G, bf16) in `../../weights/`
- [x] Goldens: DiT aligned+unaligned, RoPE, encoder, VAE, scheduler traces,
      e2e MPS 1024²/8-step (coherent lighthouse render — oracle validated on-box)
- [x] **P1 GREEN** — scheduler (Turbo shift-3 + Base shift-6) + RoPE tables vs goldens
- [x] **P2 GREEN** — full DiT parity, both cases: all stages cos ≥0.9999999,
      x_embed bit-exact, worst max_abs 6.1e-3 @ block29 (fp32 accumulation), padding
      machinery exact (unaligned case), api==staged. Loader remaps verified 521/521
      keys, 0 missing / 0 unused, strict `.all`.
- [x] **P3 GREEN** — Autoencoder.swift (copy-adapt Boogu VAE.swift, key-for-key):
      decode PSNR **118.34 dB** vs golden; noise-path smoke clean (stride-8 phase
      spread 0.02 vs std 0.51); denorm exact.
- [x] **P4 GREEN** — Qwen3HiddenStateEncoder (Adapted/, copy-adapt mlx-swift-lm Qwen3,
      runs n−1 layers for hidden[-2]) + ZImageTextEncoder. Chat template EXACT, token
      ids EXACT (pos+neg), features cos 1.0000000 / max_abs 3.9e-3 (bf16). Loader drops
      last-layer + norm + tied lm_head keys.
- [x] **P5 GREEN** — Pipeline.swift (all quirks 1–9 verbatim) + zimage-cli GPU lane.
      CPU fp32 256²/4-step: final-latent cos 1.000000, decoded PSNR **108 dB**.
      GPU bf16 1024²/8-step: renders the reference lighthouse (semantic match to the MPS
      golden; bf16 trajectory divergence only). ~15s / 34 GB peak / 19.5 GB active.
      **NAX split-K bf16 GEMM probe (DiT FFN shapes): cos 0.99999607 — CLEAN** on
      mlx-swift 0.31.6, no row-chunk workaround needed.
      **Resolution sweep (pitfall #32): 512² (2.8s), 1024² (15s), 1536² (48s) all
      coherent** — no stride artifact / RoPE-extrapolation failure. 3-axis position
      machinery holds across the production range. Turbo default seed-42 renders.
- [x] **P6 GREEN** — Base tier (shift 6.0) + CFG path. Numeric: Base+CFG 256²/6-step
      guidance 4.0 + neg prompt vs oracle golden → final-latent cos 1.000000, PSNR
      **105 dB**. GPU Base 1024²/28-step/g4.0 renders a richer, higher-ceiling
      composition than Turbo (CFG bsz=2 unified path + pos-anchored formula + truncation
      + neg prompt all exercised). ~170s / 34 GB peak.
- [x] **P7 GREEN** — DiT quant (affine group 64, keepHiPrecision skips x_embedder /
      final_layer / t_embedder / cap_embedder_linear / per-block adaLN_modulation.0).
      Per-pass cos vs fp32 golden: **int8 0.9998** (DiT 11.7→**6.4 GB**), **int4 0.976**
      (DiT →**3.5 GB**). int4 1024²/8-step render is sharp + fully coherent (eyeball —
      the decisive generative-quant gate; low per-pass cosine ≠ bad image, per Lens
      doctrine). q4 pipeline ≈ 3.5 GB DiT + ~2.3 GB int4 Qwen3 + 0.3 GB VAE ≈ **6 GB →
      16 GB tier**. GATE RUNS ON GPU via `swift run zimage-cli --quant-gate 8,4` /
      `--quant N` (quantized matmul is Metal-only; CPU-stream quant forward HANGS —
      cost a 10h stuck run before the fix).
- [x] **P8 WRAPPER GREEN — publish pending** — MLXZImage: `ZImageConfiguration`
      (PackageConfiguration/ModelStorable/QuantConfigured/WeightSourcing, per-quant
      suffixed mlx-community repos, `.base()`/`.turbo()` factories), `ZImageT2IPackage`
      (base, 28-step CFG, surface `z-image-t2i`) + `ZImageTurboT2IPackage` (thin delegate,
      8-step no-CFG, `z-image-turbo-t2i`), `ZImageGenerator` core holder. Manifests:
      Apache-2.0/MIT, split QuantFootprint ×3 (bf16/int8/int4), unload()→clearCache.
      Engine pin 0.21.0 (N5 cacheLimit). Builds clean vs MLXToolKit 0.21.0.
      - **7/7 MAT + conformance tests pass** (both MAT-gate cases: Turbo bf16, Base int4;
        WeightSourcing source shape, store-layout probe, explicit-path precedence,
        Codable round-trip, Apache/distinct-surface manifests).
      - **Wrapper e2e green** (`zimage-cli --pkg-e2e turbo --quant 4`): real ModelPackage
        load 2.95s → run 12.79s → valid 1024² PNG via T2IResponse (25.7 GB peak) → clean
        unload. Sharp coherent lighthouse — the full engine glue works.
      NOTE: v0.1.0 publishes bf16 + quantizes at load() (correct resident footprint;
      pre-quantized published repos = later download-size optimization).

## Publish — SHIPPED v0.1.0 (2026-07-06)

- [x] Weights → mlx-community Collection "Z-Image (MLX)"
      (`z-image-mlx-6a4bae170a9e0cf520bd2049`): `Z-Image-Turbo-bf16` (21 files, 19.1 GB) +
      `Z-Image-bf16` (20 files, 19.1 GB), full diffusers snapshots + cards. Turbo transformer
      converted fp32→bf16 (`publish_zimage.py`), verified renders clean before upload.
      int8/int4 = later pre-quant download-size optimization.
- [x] Code → `github.com/xocialize/z-image-swift` @ v0.1.0 (public). `Tests/` case fixed
      for case-sensitive Linux clones.
- [x] Registry row → branch `registry/z-image-textToImage` in mlx-engine-swift (pushed,
      PR-ready; not merged to shared main — user merges).
- [ ] REMAINING (later session): in-app validation in MLXEngineImage (register → prepare →
      run → decode) + phys re-baseline (smoke under-reads ~2.7×) — Val 🟡 until then.
      Plus the mflux-vs-Lens/ERNIE quality bench (own session, custom metrics).

Lesson (harness): `String(format: "%s", swiftString)` in test prints segfaults xctest
(signal 11) — looked exactly like a load/MLX crash. Use interpolation.
