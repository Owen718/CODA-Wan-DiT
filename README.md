# CODA-Wan-DiT

CODA GEMM-epilogue fusion integrated into the complete
[Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1) generation pipeline:
T5 text encoding, a 30-layer DiT, UniPC scheduling, classifier-free guidance,
the official Wan VAE decoder, and video export.

The primary result is a real `832×480`, 81-frame, 50-step generation on one
H100 80GB. With the same prompt, negative prompt, seed, initial noise,
checkpoint, scheduler, and FlashAttention-2 backend, CODA reduces the median
summed DiT time from `81,273.19 ms` with
`torch.compile(mode="max-autotune")` to `72,339.07 ms`: **1.124× vs compile**
and **1.286× vs eager**. This latency is always **DiT-only, T5/VAE excluded**.
All three variants nevertheless execute the complete pipeline on every warmup
and measured run, and each exports a representative real video.

The earlier `697.73 ms` result remains useful, but it is a separate
single-forward microbenchmark with synthetic fixed inputs. It is documented in
[`docs/report.md`](docs/report.md) and must not be mixed with the 50-step totals
below. The complete-pipeline protocol and raw measurements are in
[`docs/end_to_end.md`](docs/end_to_end.md).

> This is a shape-specific exploratory study, not a general Wan acceleration
> package. See [Limitations](#limitations).

---

## Complete-pipeline result

Each variant ran one unmeasured complete generation followed by three measured
complete generations. A generation contains 50 scheduler steps and two
sequential batch-1 DiT forwards per step (conditional, then unconditional), for
100 timed forwards. CUDA events surround only each `WanModel.forward`; the 100
event durations are summed and the median of the three generation totals is
reported.

| Variant | Median DiT-only total | Avg. per scheduler step | Avg. per CFG forward | vs eager | vs compile | Denoising peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| eager (stock Wan) | 93,046.21 ms | 1,860.92 ms | 930.46 ms | 1.000× | 0.873× | 5.179 GiB |
| `torch.compile(max-autotune)` | 81,273.19 ms | 1,625.46 ms | 812.73 ms | 1.145× | 1.000× | **4.492 GiB** |
| **CODA** | **72,339.07 ms** | **1,446.78 ms** | **723.39 ms** | **1.286×** | **1.124×** | 4.932 GiB |

**Timing label: DiT-only, T5/VAE excluded.** CFG combination, scheduler work,
model movement, T5, VAE, and media export are outside the CUDA-event intervals.
The memory column is the peak allocation during denoising, so it includes live
scheduler/CFG tensors even though latency times only DiT forwards.

All runs used seed `20260715`, CFG `5.0`, UniPC shift `5.0`, bf16 DiT and T5,
the official fp32 Wan VAE decoder, and latent grid `21×30×52` (`N=32760`). The
machine was an NVIDIA H100 80GB HBM3 with CUDA 12.6, PyTorch 2.7.0, and
FlashAttention 2.7.4.

### Video previews and data

| Stock eager | CODA |
|---|---|
| [![Stock eager preview](results/full-video/artifacts/gifs/eager.gif)](results/full-video/artifacts/videos/eager.mp4) | [![CODA preview](results/full-video/artifacts/gifs/coda.gif)](results/full-video/artifacts/videos/coda.mp4) |

- MP4: [eager](results/full-video/artifacts/videos/eager.mp4) ·
  [compile](results/full-video/artifacts/videos/compile.mp4) ·
  [CODA](results/full-video/artifacts/videos/coda.mp4)
- Optimized GIF: [eager](results/full-video/artifacts/gifs/eager.gif) ·
  [CODA](results/full-video/artifacts/gifs/coda.gif)
- Aligned decoded frames:
  [eager](results/full-video/artifacts/aligned_frames/eager/) ·
  [CODA](results/full-video/artifacts/aligned_frames/coda/)
- Machine-readable measurements and provenance:
  [`dit_only_results.json`](results/full-video/dit_only_results.json)

Eager versus CODA was compared over all 81 aligned VAE-decoded RGB uint8
frames before MP4 encoding. Global PSNR is `24.403 dB` and mean frame SSIM is
`0.9277`. This passes the operational regression gate of PSNR `≥20 dB` and
mean SSIM `≥0.90`, supporting **high structural similarity** for this fixed
case. The gate is not a claim of pixel equality or a substitute for broader
perceptual evaluation.

---

## What CODA changes

[CODA](https://arxiv.org/abs/2605.19269) rewrites memory-bound non-attention
operators as GEMM epilogues so intermediate activations can be consumed while
their output tile is still on chip. In this Wan block, the shipped path covers:

| Block seam | CODA rewrite |
|---|---|
| QKV projection | LayerNorm/modulation fusion plus partial Q/K mean-square reduction; a small auxiliary applies RMS weights, 3D RoPE, and cast |
| Self-attention output | Bias, dynamic gate, and residual in the output-projection epilogue |
| Cross-attention output | Bias and residual in the output-projection epilogue |
| FFN input projection | LayerNorm/modulation and tanh-GELU fused around GEMM1 |
| FFN output projection | Bias, dynamic gate, and residual in GEMM2's epilogue |

The implementation also uses per-shape GEMM autotuning and prepacked weights
that alias the original storage. Self- and cross-attention remain
FlashAttention-2. The algebra, profiling buckets, accepted levers, and rolled
back experiments are covered in the [microbenchmark technical
report](docs/report.md) and [implementation walkthrough](docs/walkthrough.md).

---

## Reproduce complete generation

Requirements:

- NVIDIA H100/H200 with a compatible CUDA 12 toolchain
- PyTorch 2.7, FlashAttention-2, CuTeDSL/CUTLASS, Quack, Triton, and the CODA
  kernel dependencies (`rapier.*` and `kernels.gens.*`)
- Wan2.1 available on `PYTHONPATH`
- A complete Wan2.1-T2V-1.3B checkpoint containing DiT, UMT5, tokenizer, and
  VAE weights

From the repository root, run every variant into the same output directory:

```bash
export PYTHONPATH="/path/to/Wan2.1:/path/to/coda-kernels:${PYTHONPATH}"

for variant in eager compile coda; do
  python generate_video.py \
    --variant "${variant}" \
    --ckpt-dir /path/to/Wan2.1-T2V-1.3B \
    --output-dir results/full-video \
    --warmup 1 \
    --trials 3
done
```

The defaults reproduce the measured generation configuration: prompt and seed
from this report, width `832`, height `480`, 81 frames, 50 steps, CFG `5.0`,
shift `5.0`, and 16 fps. `generate_video.py` validates that each measured run
contains 100 timed DiT forwards. Once all three variant JSON files exist in the
output directory, it writes the aggregate `dit_only_results.json`, aligned
PNGs, and decoded-video PSNR/SSIM.

The first CODA invocation may spend several minutes populating the CuTeDSL
autotuning cache. Keep that cache stable between warmup and measurement.

For the independent single-forward microbenchmark, use
`coda_wan/run_wan_coda.py`; its inputs, warmup count, trial count, metrics, and
scope differ from this complete-pipeline experiment.

---

## Repository layout

```text
generate_video.py                         # complete pipeline + DiT-only timing
coda_wan/
  run_wan_coda.py                         # independent single-forward benchmark
  wan_coda_kernels.py                     # QKV/FFN fusion and RoPE helpers
  wan_gate_residual_coda.py               # gate/bias/residual epilogues
docs/
  end_to_end.md                           # complete protocol, raw totals, provenance
  report.md                               # single-forward microbenchmark report
  walkthrough.md                          # block-level implementation narrative
results/full-video/
  dit_only_results.json                   # aggregate measurements and validation
  {eager,compile,coda}_result.json         # raw per-variant runs
  artifacts/videos/{eager,compile,coda}.mp4
  artifacts/gifs/{eager,coda}.gif
  artifacts/aligned_frames/{eager,coda}/
```

---

## Independent microbenchmark

The original study isolates one 30-layer DiT forward at `N=32760` using fixed
synthetic latent/context inputs, 10 warmups, and 30 CUDA-event trials. It found
`913.690 ms` eager, `796.299 ms` compile, and `697.726 ms` CODA: 1.141× versus
compile. Those numbers explain where the kernel win comes from; they are not
complete-generation totals and do not include CFG's second forward. See
[`docs/report.md`](docs/report.md).

---

## Limitations

- The complete result covers one prompt, one negative prompt, one seed, one
  checkpoint, one H100, batch-1 CFG branches, and one fixed
  `832×480×81`/50-step configuration. Other shapes and models require new
  profiling and validation.
- Latency intentionally isolates DiT. It is not complete-pipeline wall time;
  T5, VAE, scheduler work, transfers, and export are excluded.
- The denoising memory peak includes live scheduler/CFG allocations and should
  not be interpreted as a kernel-only allocation measurement.
- PSNR/SSIM is a fixed-case regression signal on pre-encoding decoded frames.
  It is not a human preference study, temporal metric, LPIPS evaluation, or
  broad quality benchmark.
- The complete benchmark uses three measured generations per variant. More
  repetitions and multiple nodes are needed for a distributional performance
  claim.
- CODA currently assumes batch 1 and grid `21×30×52`; CFG is therefore executed
  as two sequential batch-1 forwards.
- Inference only: no training/backward and no FP8. DiT and T5 are bf16; the
  official Wan VAE decoder remains fp32.
- The kernel stack targets Hopper and is a research reference implementation,
  not a drop-in package.

---

## Credits and license

- **CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs** — Guo,
  Zhang, Menon, Guessous, Thakkar, Kim, Dao (2026).
  [arXiv:2605.19269](https://arxiv.org/abs/2605.19269) ·
  [coda-kernels](https://github.com/HanGuo97/coda-kernels)
- **Wan2.1** — Alibaba Wan Team.
  [github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
- **CUTLASS / CuTeDSL** — NVIDIA

The fusion code in this repository is released under [Apache-2.0](LICENSE). It
depends on, but does not vendor, CODA and Wan2.1; install those projects under
their respective licenses.
