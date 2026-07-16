# Technical Report — CODA GEMM-Epilogue Fusion on Wan2.1-T2V-1.3B

> [!IMPORTANT]
> **Scope boundary:** this document is the independent **single-forward
> microbenchmark** (`N=32760`, fixed synthetic latent/context). The repository's
> primary result is now the [complete 50-step Wan generation
> experiment](end_to_end.md), where CODA records `72,339.065 ms` summed
> DiT-only time and 1.124× versus compile. The `697.726 ms` CODA result below is
> one forward under a different protocol; it is not a complete-generation
> total.

A profile-gated study of how far CODA-style epilogue fusion accelerates the **sampling** forward pass of Wan2.1-T2V-1.3B, at the real video token length, in bf16, on a single H100. Attention is kept as FlashAttention-2 throughout and is excluded from fusion by design.

---

## 1. Setup

| Item | Value |
|---|---|
| Model | Wan2.1-T2V-1.3B pretrained DiT — 30 layers, dim 1536, 12 heads, head_dim 128, FFN 8960 |
| Real shape | latent `[16, 21, 60, 104]`, grid `21×30×52`, `N = 32760`, text context `[512, 4096]` |
| Task | one denoising forward, `t = 900`; sampling-only, bf16, no backward |
| GPU / stack | NVIDIA H100 80GB, CUDA 12.6, PyTorch 2.7.0, FlashAttention-2 2.7.4 |
| dtype | bf16 compute; Wan's fp32 residual / modulation semantics preserved |
| Benchmark discipline | fixed seed; 10 warmups; 30 trials; CUDA-event median |
| Primary baseline | `torch.compile(mode="max-autotune")`. eager kept only as a mechanism reference. |
| Profiler | Nsight Systems with a `cudaProfilerApi` capture range over one warmed real block |

Inputs are fixed-seed synthetic latent/context; weights are the real pretrained DiT. T5/VAE are not loaded — the goal is to isolate the DiT block and its GEMM epilogues. GPU clock locking was unavailable (no permission); the warmup + median discipline is retained instead.

---

## 2. The block, and what is fusible

Wan's attention block forward (modulation `e[0..5]` = shift/scale/gate for {self-attn, FFN}, broadcast over tokens):

```
e = (block.modulation + e0).chunk(6, dim=1)        # e[i]: (B,1,dim), per-channel
y = self_attn( norm1(x)*(1+e[1]) + e[0] );  x = x + y*e[2]     # norm1 = LayerNorm(affine=False)
x = x + cross_attn( norm3(x), context )                        # norm3 = LayerNorm(affine=True)
y = ffn( norm2(x)*(1+e[4]) + e[3] );        x = x + y*e[5]     # ffn = Linear→GELU(tanh)→Linear
# self_attn internal: q,k → WanRMSNorm (qk-norm) → 3D RoPE → FlashAttention
```

Every non-attention side-op here is, in stock PyTorch, a separate memory-bound kernel that reads and writes the full `[N,1536]` (or `[N,8960]`) activation. CODA folds them into the epilogue of the neighbouring GEMM. The named fusion seams:

- **K1** — FFN: `norm2+modulation` fused; GEMM1 epilogue does `bias + tanh-GELU`; GEMM2 epilogue does `bias + gate + residual`.
- **K2** — self-attn: `norm1+modulation` fused; out-proj epilogue does `(A@B)*gate + bias*gate + residual`.
- **K3** — QKV: QK GEMM epilogue emits a partial mean-square reduction; a light auxiliary kernel applies WanRMSNorm weights + 3D RoPE + bf16 cast. Cross-attention Q/K RMSNorm is also fused.
- **E** — cross-attn out-proj: `bias + residual` epilogue.
- **D** — per-shape exhaustive GEMM autotuning (mainloop quality, not epilogue fusion).

---

## 3. Phase 0 — five-bucket profiling of the CODA run

Every kernel in one warmed real block is attributed to five buckets:

1. **FA kernels** — the FlashAttention kernels themselves (self + cross).
2. **GEMM mainloop** — production GEMM time minus a same-shape no-op-epilogue reference.
3. **folded epilogue** — the increment a production epilogue adds over its no-op mainloop.
4. **independent aux** — kernels still launched separately (norm, modulation, K3 aux, casts, residual).
5. **other** — scheduling, scan/copy, unclassified tail.

Buckets 3 + 4 are the *recoverable surface*. Before/after applying the accepted levers (E + D):

| bucket | before (ms/block) | before % | after E+D (ms/block) | after % |
|---|---:|---:|---:|---:|
| FA kernels | 17.328 | 75.85% | 17.337 | 77.29% |
| GEMM mainloop | 3.496 | 15.30% | 3.564 | 15.89% |
| folded epilogue | 0.433 | 1.90% | 0.155 | 0.69% |
| independent aux | 1.107 | 4.84% | 0.857 | 3.82% |
| other | 0.482 | 2.11% | 0.520 | 2.32% |
| **block total** | **22.847** | 100% | **22.432** | 100% |

- Recoverable before = bucket 3 + 4 = **1.540 ms/block** (≈ 46.2 ms over 30 blocks).
- Recoverable after E+D = **1.011 ms/block** (≈ 30.3 ms). ~34% of the recoverable surface eliminated.
- FA itself is not optimized; the 17.33 ms is stable, and its rising *share* is purely the denominator shrinking as non-attention work is removed.

Post-E+D `independent aux` decomposes as: K3 QK RMS+RoPE `0.287`, LN1+LN2 modulation `0.201`, cross norm3 `0.173`, cross RMSNorm `0.072`, cross-q cast `0.099`, modulation add `0.008`, gate prep `0.017` ms/block.

### GEMM cross-check (why lever D exists)

| shape | torch mm | CODA no-op (old fixed config) | fused | note |
|---|---:|---:|---:|---|
| QK `1536→3072` | 0.399 | 0.425 | 0.451 | fusion overhead ~0.026 ms |
| FFN1 `1536→8960` | 1.128 | 1.180 | 1.407 | needs per-shape tile |
| FFN2 `8960→1536` | 1.074 | 1.201 | 1.301 | needs per-shape tile |
| projection `1536→1536` | 0.209 | 0.229 | 0.310 | epilogue saving must beat fusion cost |

A single generic `128×128, cluster 2×2` config left GEMM performance on the table; hence per-shape autotune (D).

---

## 4. Lever gates, implementation, and rollbacks

Each candidate lever is only shipped if it clears a **≥ 0.15 ms/block** net gain, measured, not assumed.

| Lever | Phase-0 target | Measured | Decision |
|---|---:|---|---|
| **E** — cross out-proj + bias + residual | ~0.423 ms/block | old cross GEMM + residual `0.445` → fused `0.244`, saves `0.201`; full step `723.10 → 703.95 ms` | **accept** |
| **D** — per-shape exhaustive autotune | ~0.226 ms/block | FFN1+FFN2 tuners save ~`0.193` combined; full step −`4.61 ms` | **accept** |
| **A** — three-phase LayerNorm | ~0.199 ms/block | numerically passes, but block `22.432 → 22.825 ms`, a `0.392` **regression** | **rollback** (evidence kept) |
| **B** — shrink K3 RMS+RoPE aux | ~0.288 ms/block | wide-row / pair / two `ROWS=2` variants save only `0.015–0.043` ms | **rollback** |
| **C** — dtype seam | 0.099 ms/block | below gate; merging with norm3 also pushed block max-abs to ~0.038 | **skip** |

### Why A (the biggest theoretical lever) was rolled back

A is CODA's canonical cross-GEMM three-phase pattern: the cross-out epilogue emits residual + partial sum + partial mean-square; an O(M) aux computes `rstd` and `-mean*rstd`; then the FFN1 weight is dynamically scaled and the FFN1 epilogue applies the deferred `rstd`, the mean rank-1 correction, and GELU. The math is correct — block parity passes (max abs `0.0237`).

But **folding the dynamic per-step modulation into the weight requires transforming the `[1536×8960]` FFN1 weight on every step (~0.35 ms/block)**, which exceeds the LayerNorm it removes (~0.10 ms/block). Net regression `0.392 ms/block`. This is the crisp reason DiT differs from LLaMA: LLaMA's RMSNorm weights are static, so the three-phase pays; Wan's modulation is dynamic per denoising step, so the reparametrization tax dominates. The per-seam broadcast forms of modulation (K1/K2/K3) remain clean wins — it is specifically the *weight-fold* path that fails.

### Why B could not shrink K3

K3's global per-row RMS reduction defeats a simple "wider tile". The best variants moved the aux from ~`0.287` to ~`0.232` ms/block — far below the `0.15` net-gain gate — with no stable full-model improvement. The auxiliary RMS+RoPE pass before FlashAttention is essentially irreducible given the closed FA kernel boundary (q/k must be materialized before FA consumes them).

---

## 5. Independent one-forward 30-layer A/B

Pretrained, seed fixed, 10 warmups, 30 trials, CUDA-event median, FA2.

| Variant | median ms | tokens/s | peak GiB | vs compile | vs eager | numerical vs eager |
|---|---:|---:|---:|---:|---:|---|
| eager reference | 913.690 | 35,855 | 4.623 | 0.872× | 1.000× | reference |
| `torch.compile(max-autotune)` | 796.299 | 41,140 | 3.936 | 1.000× | 1.147× | allclose, 63.87 dB |
| CODA K1+K2+K3 (v2) | 723.096 | 45,305 | 4.471 | 1.101× | 1.264× | allclose, 64.72 dB |
| CODA + E | 703.952 | 46,537 | 4.377 | 1.131× | 1.298× | allclose, 65.86 dB |
| **CODA + E + D (final)** | **697.726** | **46,953** | **4.377** | **1.141×** | **1.310×** | allclose, **65.86 dB** |

Three independent 30-trial medians for the final config: `699.34`, `698.65`, `697.73 ms` (spread 0.23%); the reported row is the last, not cherry-picked.

**Speedup-gain capture.** Against the eager-relative recoverable budget (46.2 ms/step), the final path captures **72.96%** relative to eager and **28.02%** relative to compile. The `vs compile` figure is the honest one; `vs eager` is inflated by Wan's slow eager RoPE (see §7).

---

## 6. Numerics, memory, storage audit

| Metric | Final (E+D) vs eager |
|---|---:|
| allclose `rtol=2e-2, atol=2e-2` | **pass** |
| pass fraction | 1.0 |
| max abs | 0.0234 |
| mean abs | 0.00174 |
| latent PSNR | **65.86 dB** |
| peak allocated | **4.377 GiB** (eager 4.623; −5.33%) |

The comparison is CODA-bf16 vs eager-bf16 (not fp32), so max-abs `0.023` sits at the bf16 rounding floor. Notably CODA (65.86 dB) is *closer* to eager than `torch.compile` (63.87 dB). Precision-losing fusions were explicitly refused — lever C's norm3+cast merge was rejected partly because it pushed max-abs to ~0.038.

**Storage audit:** unique CUDA storage grows by exactly `16,773,120` bytes — solely the one shared RoPE table. All 30 layers' prepacked parameters share their original weight storage (`transpose_duplicate_bytes = 0`); q/k/self-o/cross-o/FFN1/FFN2 alias checks all pass. The earlier per-layer transpose-copy regression is eliminated.

---

## 7. Stop conditions and ceilings

| Condition | Value | Met |
|---|---:|---|
| independent aux < 1% of step | 3.82% / block | no |
| CODA ≤ 685 ms | 697.73 ms (short by 12.7 ms) | no |
| memory ≤ eager 4.623 GiB | 4.377 GiB | yes |
| allclose + latent PSNR | pass / 65.86 dB | yes |

Even zeroing all remaining bucket 3+4 (30.3 ms/step) only reaches a 667.4 ms floor — and that floor is unreachable because A regressed, B's reduction is bounded, and C is below gate. The aspirational ~652 ms figure is a loose theoretical lower bound, not achievable here. **Closing the last ~12.7 ms would require a structural multi-seam program that removes several memory seams at once, not more small kernels** — and the one attempt at that (A) regressed.

The `vs eager` numbers overstate CODA's distinctive contribution: Wan's eager 3D RoPE (complex-tensor ops) is pathologically slow, and `torch.compile` alone speeds the K3 micro-kernel ~5× by fusion. So most of the eager-relative "recoverable" pie is inefficiency the compiler also removes; CODA's marginal win over the compiler is ~12% (1.14×).

---

## 8. Findings

1. **Attention-bound, not method-limited.** FA (~55–58%) + GEMM FLOPs (~16%) are an untouchable floor. 1.14× is near the physical ceiling on the addressable ~28% of the block.
2. **DiT's dynamic modulation is a double-edged sword** — a clean win as per-seam epilogue broadcast, but it blocks CODA's cross-GEMM weight-fold three-phase (lever A) because the per-step weight transform outweighs the norm it removes. This is the structural difference from LLaMA.
3. **Profile-gating is decisive.** The first pass mis-bucketed QK-norm+RoPE (17.8%, the largest seam) as "attention support" and concluded a 1.13× ceiling. Re-profiling the CODA run five-bucketed and gating each lever at ≥0.15 ms/block turned 1.01× → 1.14× and correctly killed three dead-end levers.

## 9. Later complete-pipeline validation and remaining scope

The follow-on [complete-pipeline experiment](end_to_end.md) now runs T5,
50-step UniPC denoising with sequential CFG, the official Wan VAE decoder, and
media export for eager, compile, and CODA. It reports decoded-frame PSNR/SSIM
for eager versus CODA and retains this document as the kernel-focused
microbenchmark. Its strong-baseline result is 1.124× versus compile, with high
structural similarity under the fixed regression gate.

Training/backward (recoverable only ~7.5% at this shape), 200-step loss curves,
LPIPS, multi-prompt or multi-seed quality evaluation, and FP8 remain out of
scope. The only remaining pure-CODA direction with real headroom (K3's
~8.6 ms/forward auxiliary work) needs a QKV epilogue that produces
FA-consumable layout while co-locating RMSNorm+RoPE—still without touching the
FA mainloop. Moving RoPE *inside* FlashAttention would break CODA's "attention
excluded" premise and become a different project. Any change to resolution,
frames, batch, or architecture (Wan2.2, MMDiT) requires re-running Phase 0.
