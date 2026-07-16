# Complete Wan2.1 video generation and DiT-only timing

This document is the primary performance and quality result for this
repository. It covers complete Wan2.1-T2V-1.3B generations through T5, the
30-layer DiT, classifier-free guidance (CFG), UniPC scheduling, the official
Wan VAE decoder, and media creation. The latency metric intentionally isolates
the DiT and is labeled throughout as **DiT-only, T5/VAE excluded**.

The older one-forward study remains an independent microbenchmark in
[`report.md`](report.md). Its `697.726 ms` CODA number uses fixed synthetic
inputs, 10 warmups, and 30 trials. It should not be compared directly with the
50-step totals in this document.

## Result

On the fixed complete-generation protocol, CODA reaches a median summed
DiT-only time of `72,339.065 ms`, a **1.286× speedup over eager** and a
**1.124× speedup over `torch.compile(max-autotune)`**.

| Variant | Median DiT-only total | Avg. / 50-step scheduler step | Avg. / CFG branch forward | vs eager | vs compile | Denoising peak allocated | Full-pipeline peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 93,046.214 ms | 1,860.924 ms | 930.462 ms | 1.000× | 0.873× | 5.179 GiB | 13.995 GiB |
| compile | 81,273.185 ms | 1,625.464 ms | 812.732 ms | 1.145× | 1.000× | **4.492 GiB** | 13.995 GiB |
| **CODA** | **72,339.065 ms** | **1,446.781 ms** | **723.391 ms** | **1.286×** | **1.124×** | 4.932 GiB | 14.010 GiB |

The denoising peak is reset and sampled over the denoising phase. It includes
live scheduler and CFG temporaries, even though the event-timed latency covers
only DiT forwards. The full-pipeline peak includes T5 and VAE phases and is
provided for context; it is not part of the speed calculation.

### Raw generation totals

One complete unmeasured generation warmed each variant. The following are all
three measured complete-generation sums, in run order:

| Variant | Run 0 | Run 1 | Run 2 | Median-selected run |
|---|---:|---:|---:|---:|
| eager | 93,046.214 ms | 93,051.005 ms | 93,030.396 ms | run 0 |
| compile | 81,273.185 ms | 81,248.272 ms | 81,317.091 ms | run 0 |
| CODA | 72,301.856 ms | 72,348.153 ms | 72,339.065 ms | run 2 |

The representative media and decoded frame array for each variant come from
the measured run nearest that variant's median. Every warmup and measured run
executes prompt encoding, all denoising steps, and VAE decoding; media files are
written from the representative measured run.

## Exact protocol

| Item | Value |
|---|---|
| Model | Wan2.1-T2V-1.3B, pretrained 30-layer DiT |
| Prompt | `Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.` |
| Negative prompt | Wan2.1-T2V-1.3B default negative prompt |
| Seed | `20260715` |
| Output | width `832`, height `480`, 81 frames, 16 fps |
| Latent | `[16, 21, 60, 104]`; DiT grid `21×30×52`; sequence length `32760` |
| Sampler | UniPC, 50 steps, shift `5.0` |
| CFG | scale `5.0`; conditional then unconditional, each batch 1 |
| Attention | FlashAttention-2 forced for self- and cross-attention |
| Precision | DiT bf16; T5 bf16; official Wan VAE decoder fp32 |
| Optimization | inference only; no training/backward; no FP8 |
| Warmup / trials | 1 complete generation / 3 complete generations per variant |
| Compile variant | `torch.compile(mode="max-autotune", fullgraph=False)` around the DiT model |
| CODA variant | all 30 Wan blocks replaced once by `CodaFullBlock`; fixed RoPE table for `21×30×52` |

The exact Wan default negative prompt used in every generation was:

```text
色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走
```

### Timing boundary

For every scheduler step, the script records one CUDA start/end event around
the conditional `WanModel.forward`, then another pair around the unconditional
`WanModel.forward`. It checks that the generation produced exactly 100 event
pairs. After denoising, it synchronizes and sums all 100 event durations.

Consequently:

- total = sum of 100 DiT forward event durations;
- average scheduler-step DiT time = total / 50 and includes both CFG branches;
- average CFG-branch forward time = total / 100;
- the reported result is the median of the three per-generation totals.

T5 encoding, CFG arithmetic, scheduler operations, transfers/model movement,
VAE decoding, and media export are outside the event intervals. CUDA events are
recorded on the current CUDA stream. No prompt embedding or VAE output is
reused between complete generations.

## Decoded-video regression check

Eager run 0 and CODA run 2 were compared over all aligned decoded frames. The
comparison domain is the VAE output clipped from `[-1, 1]`, mapped to RGB uint8
`[0, 255]`, with shape `[81, 480, 832, 3]`, before MP4 encoding.

| Metric | Value |
|---|---:|
| Global MSE on uint8 RGB | 235.9393 |
| Global PSNR | **24.4028 dB** |
| Mean per-frame PSNR | 24.8077 dB |
| Minimum per-frame PSNR | 20.3516 dB |
| Mean per-frame SSIM | **0.927721** |
| Minimum per-frame SSIM | 0.891530 |

SSIM uses scikit-image Gaussian-weighted SSIM with `sigma=1.5`, RGB channel
axis, data range 255, and population covariance. The operational regression
gate is global PSNR `≥20 dB` and mean frame SSIM `≥0.90`; this result passes.
The pass supports **high structural similarity** for this fixed generation. It
does not establish pixel equality, temporal perceptual equivalence, or quality
across prompts and seeds.

## Artifacts

All repository artifacts live under [`results/full-video/`](../results/full-video/).

| Artifact | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Aggregate result | [`dit_only_results.json`](../results/full-video/dit_only_results.json) | — | `de0eed96646649e77cf9d727410042d76c69dd965a1f26df7311ae50ab2974d4` |
| eager MP4 | [`eager.mp4`](../results/full-video/artifacts/videos/eager.mp4) | 3,693,677 | `3172d991f2aba2543cbf88cfef5323b6edb00294a0687744d0d347a2a2d1ee9d` |
| compile MP4 | [`compile.mp4`](../results/full-video/artifacts/videos/compile.mp4) | 3,726,393 | `cbb870ad7afc4a0b184067818101b24b07bdbc8603d941314f82dc4e3d4fea7c` |
| CODA MP4 | [`coda.mp4`](../results/full-video/artifacts/videos/coda.mp4) | 3,700,405 | `335996275699a6a1548ede2f72563621f67c3931191aa3ccfa830a6862d6d2e7` |
| eager GIF | [`eager.gif`](../results/full-video/artifacts/gifs/eager.gif) | 292,021 | `d156e9269372ab4c3aafd1196c0515ae10873102829dadf89d37d467b0394729` |
| CODA GIF | [`coda.gif`](../results/full-video/artifacts/gifs/coda.gif) | 292,985 | `abe940a7fae5929774bccb390fc96973ee599f7aa1a975b16e3334c4b60bbe3e` |

Both GIFs are 256×148, contain 16 uniformly sampled source frames, and are
well below 3 MB. Aligned eager/CODA PNGs are included for source frames
`0, 11, 23, 34, 46, 57, 69, 80`.

The aggregate JSON is the exact machine-readable record. It includes all 300
per-forward timings per variant, hashes, validation gates, environment fields,
generation configuration, per-phase memory, and source provenance.

## NRT provenance

The runs were executed on NRT, and the server workdir—not a pre-existing local
implementation—was the source of truth.

| Item | Value |
|---|---|
| Workdir | `.` |
| Result directory | `results/full_video_final` |
| Checkpoint | `models/Wan2.1-T2V-1.3B` |
| Python | `python` |
| Slurm | job `<job-id>`, node `<gpu-node>` |
| GPU | NVIDIA H100 80GB HBM3, UUID `GPU-35e47ebd-386b-8a91-35f3-4684de558503` |
| Software | driver 550.127.05; CUDA 12.6; PyTorch 2.7.0+cu126; FlashAttention 2.7.4.post1; Python 3.10.20 |
| Wan2.1 commit | `9737cba9c1c3c4d04b33fcad41c111989865d315` |

The checkpoint audit records a `5,676,070,424`-byte DiT safetensors file, an
`11,361,920,418`-byte bf16 T5 checkpoint, a `507,609,880`-byte VAE checkpoint,
and the tokenizer model. `config.json` has SHA-256
`ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce`.

### Server source SHA-256

| Server source | SHA-256 | Role |
|---|---|---|
| `scripts/generate_video.py` at generation time | `f4471404cee5f33ead29a0a067155e91c6929678a2500fef06ec0b43721e0aa3` | Executed for all 12 complete generations |
| `scripts/generate_video.py` final aggregation copy | `bebc42aaab0bbb868d80f4bfaf74ce0155f675e7bffa9055ae3b95f69c3fc53d` | Produced/validated the aggregate suite after adding final provenance handling |
| `scripts/wan_full_coda_v3_bench.py` | `61fbb339deedabba2e24366d07b996766e98fcc2fa65efc43b1549d6fc5ce437` | `CodaFullBlock` implementation |
| `scripts/wan_coda_kernels_v3.py` | `6734efacc95288ceb72dc639f33fb86a7ad939264634d7ef14601d173a0182de` | CODA kernels and RoPE table builder |
| `scripts/wan_gate_residual_coda_v3.py` | `154d01c0e0ff4daedf5f534533804c96b03f5917e205fd5656cc29ac3ca36780` | Gate/residual epilogues |

The repository entrypoint is a packaging adaptation of the server-tested
script, so its Git-blob hash can differ. The result JSON preserves the exact
server-generation and aggregation hashes above.

### Commands executed on NRT

With `PY` set to the Python path and `WORK` to the workdir above, each command
used the same checkpoint and result directory:

```bash
"${PY}" "${WORK}/scripts/generate_video.py" \
  --variant eager \
  --ckpt-dir "${WORK}/models/Wan2.1-T2V-1.3B" \
  --output-dir "${WORK}/results/full_video_final" \
  --warmup 1 --trials 3

"${PY}" "${WORK}/scripts/generate_video.py" \
  --variant compile \
  --ckpt-dir "${WORK}/models/Wan2.1-T2V-1.3B" \
  --output-dir "${WORK}/results/full_video_final" \
  --warmup 1 --trials 3

"${PY}" "${WORK}/scripts/generate_video.py" \
  --variant coda \
  --ckpt-dir "${WORK}/models/Wan2.1-T2V-1.3B" \
  --output-dir "${WORK}/results/full_video_final" \
  --warmup 1 --trials 3
```

## Interpretation and limits

The full experiment confirms that the single-forward kernel optimization
survives the real scheduler trajectory, sequential CFG, full prompt encoding,
and VAE decode. The honest strong-baseline result is 1.124× versus compile.

This remains a single-shape, single-seed, single-prompt regression study on one
GPU. The DiT timing excludes substantial complete-pipeline work by design, and
three measured generations provide a median rather than a broad performance
distribution. PSNR/SSIM is useful for catching output regressions here, but it
does not replace multi-prompt human or perceptual evaluation. CODA also remains
fixed to batch 1 and grid `21×30×52`; other shapes require re-profiling and
kernel validation.
