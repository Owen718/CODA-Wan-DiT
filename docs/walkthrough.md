# Walkthrough — from the stock Wan block to a GEMM-epilogue program

This is the narrative companion to [`report.md`](report.md): what CODA's idea *is*, how each piece of a Wan DiT block maps onto it, and — just as important — where it stops paying off and why. If you are studying CODA, this is the part worth reading slowly.

---

## 1. The one idea

CODA is not a kernel DSL. It is one observation plus a small set of restricted primitives:

> A Transformer block is a few high-throughput **GEMMs** (matmuls) surrounded by a swarm of **memory-bound side-operators** — normalization, activation, residual add, gating, reductions. In a stock framework each side-op is its own CUDA kernel that reads the whole `[N, d]` activation from HBM, does a little arithmetic, and writes it back. That data movement, not the arithmetic, is the bottleneck.

CODA's move: **leave the tuned GEMM mainloop untouched, and make its *epilogue* programmable.** The side-ops are algebraically rewritten so they execute while the GEMM's output tile is still in registers / shared memory — *before* it is written to HBM. The GEMM is a compute-bound vehicle; the extra epilogue arithmetic hides in its compute shadow, essentially for free. What you save is HBM bandwidth and kernel launches.

Two premises to keep in mind, both deliberate:
- **Attention is excluded.** It stays FlashAttention (one of CODA's authors is Tri Dao; the division of labour is on purpose). CODA fuses everything *except* attention.
- **The GEMM FLOPs are not the target.** CODA never claims to make matmul faster. It removes the memory traffic *around* the matmul.

The epilogue exposes a handful of composable primitives: rank-1 vector loads (per-channel/per-token, broadcast over the tile), tile loads/stores (residual streams), tile reductions (row/col partials merged later by a light aux kernel), and elementwise/pairwise maps (activations, rotations, gating).

---

## 2. Why a DiT is an interesting target

A DiT block has *more* memory-bound side-structure than a plain Transformer, because of **adaLN-style modulation**. Wan's block:

```
e = (block.modulation + e0).chunk(6, dim=1)   # 6 vectors: shift/scale/gate × {attn, ffn}
y = self_attn( norm1(x)*(1+e[1]) + e[0] );  x = x + y*e[2]
x = x + cross_attn( norm3(x), context )
y = ffn( norm2(x)*(1+e[4]) + e[3] );        x = x + y*e[5]
# q,k → WanRMSNorm → 3D RoPE → FlashAttention
```

Every `scale`, `shift`, `gate` is a per-channel vector broadcast over the token dimension — *exactly* CODA's rank-1-load-then-broadcast primitive. The modulation and gating that a DiT adds on top of a plain Transformer are precisely the pure-memory-movement ops CODA is best at absorbing, and they appear twice per block. So on paper, a DiT is a *fatter* target than an LLM block.

On paper. Whether it pays depends on how much of the block is *not* attention — and for a long-sequence video DiT, that is the catch (§6).

---

## 3. The scheme, GEMM by GEMM

The block has five non-attention GEMMs. Stock Wan surrounds each with separate kernels; the CODA rewrite absorbs them into epilogues.

**QKV projection** — input is `norm1(x)*(1+e[1])+e[0]`, output feeds attention through RMSNorm and RoPE.
- *Stock:* LayerNorm → mul → add (modulation) feed a cuBLAS GEMM; then a RMSNorm kernel on q, another on k; then complex-valued 3D-RoPE kernels.
- *CODA (K2 + K3):* the norm1+modulation is fused; the QK GEMM epilogue emits a **partial mean-square reduction** per row; a single light auxiliary kernel merges it into `rstd`, applies the RMSNorm weight, rotates with 3D-RoPE, and casts to bf16. A cluster of ~6 kernels collapses to one GEMM epilogue + one light aux.

**self-attn out-proj** — `out * gate(e[2]) + x`.
- *Stock:* GEMM → gate-mul kernel → residual-add kernel (two full `[N,1536]` round-trips).
- *CODA (K2):* `(A@B)*gate + bias*gate + residual` in one epilogue.

**cross-attn out-proj** — `+ x`.
- *Stock:* GEMM → residual-add kernel.
- *CODA (E):* `bias + residual` fused into the epilogue.

**FFN Linear1** (1536→8960) — input `norm2(x)*(1+e[4])+e[3]`, output through GELU.
- *Stock:* LayerNorm → mul → add feed the GEMM; then a separate GELU kernel over the *wide* `[N,8960]` activation.
- *CODA (K1):* norm2+modulation fused; GEMM1 epilogue inlines `bias + tanh-GELU`, killing the widest round-trip in the block.

**FFN Linear2** (8960→1536) — `out * gate(e[5]) + x`.
- *Stock:* GEMM → gate-mul → residual-add.
- *CODA (K1):* `bias + gate + residual` in the GEMM2 epilogue.

Two supporting moves make it real:
- **Per-shape GEMM autotune (D):** the five GEMMs have very different shapes; one generic tile config leaves performance on the table. Each gets its own tile/cluster (e.g. FFN1 `128×192/c2×1`, FFN2 `128×192/c1×2`).
- **Weight prepack + storage aliasing:** weights are transposed once at load into the CuTeDSL GEMM's `(K,N)` layout, and the original `nn.Linear.weight` is replaced with a *view* of that same storage — zero duplicate bytes, which is what keeps peak memory ≤ eager.

At the kernel-launch level: a stock block fires 5 GEMMs + 2 FA + dozens of tiny norm/mul/add/gate/residual/GELU/RMSNorm/RoPE/cast kernels. The CODA block fires ~5 fused GEMMs + 2 FA + one irreducible light aux before FA.

---

## 4. The three-phase norm trick

Normalization needs a per-row statistic, but a single GEMM tile only sees part of a row — it cannot finish `rstd` in the epilogue. CODA splits it across the GEMM boundary, exploiting that a per-row scalar **commutes** with the following matmul:

1. GEMM *N*'s epilogue emits a **partial** sum-of-squares per tile (tile reduction primitive).
2. A tiny O(M) aux kernel merges the partials into `rstd = 1/√(mean + ε)`.
3. The scale is **deferred** and applied in GEMM *N+1*'s epilogue.

One activation-sized norm kernel becomes tile-local epilogue work plus a reduction over O(M) partials instead of O(M·N) activations. Wan's *affine-free* LayerNorm needs one extra step over the RMSNorm in the paper — the mean subtraction — handled as a rank-1 correction (`GEMM result − μ·(column-sum of W)`).

---

## 5. Where it stops — and the one deep finding

Three levers were implemented and **rolled back** by the profile gate. One of them is the interesting one.

**Lever A — cross-GEMM three-phase LayerNorm.** This is CODA's most powerful structural pattern: turn the stacked per-seam kernels into one continuous program where each GEMM's epilogue feeds the next. It was implemented, and it is *numerically correct*. It still regressed by 0.39 ms/block. Why?

To defer the modulation scale, A folds `(1+scale)` into the FFN1 weight: `(h⊙(1+scale)) @ W = h @ (diag(1+scale)·W)`. But the modulation is **dynamic — it changes every denoising step** — so `diag(1+scale)·W` must be recomputed each step, a `[1536×8960]` weight transform costing ~0.35 ms/block. That is *more* than the LayerNorm it eliminates (~0.10 ms). The algebra is valid; the reparametrization is uneconomic.

This is the crisp structural difference between a DiT and an LLM:
- In **LLaMA**, RMSNorm weights are static, so the weight-fold three-phase pays — it is one of CODA's headline wins.
- In **Wan**, the modulation is dynamic per step, so folding it into weights taxes you every step. DiT's *extra* modulation is a clean win as an epilogue **broadcast** (K1/K2/K3), but it **blocks** the weight-fold path.

So a DiT is simultaneously a fatter *and* a more constrained target than a plain Transformer — richer per-seam broadcast structure, but a closed door on the one pattern that would fuse whole GEMM chains.

**Lever B — shrink the K3 aux.** The RMS+RoPE pass before FlashAttention is essentially irreducible: FA is a closed kernel, so q/k must be fully materialized (normed and rotated) before it runs. You cannot defer into FA's epilogue. The best re-tilings saved 0.02–0.04 ms — below the gate.

**Lever C — dtype seam.** Small (0.099 ms/block), and merging the cast into norm3 broke numerical parity. Rejected on principle: no trading precision for speed.

---

## 6. Honest reading of the number

The final path is **1.14× vs `torch.compile(max-autotune)`**, **1.31× vs eager**, passes its bf16 latent-parity gate, and is memory-neutral. How to read that:

- **Report `vs compile`.** The `vs eager` number is inflated because Wan's eager 3D-RoPE (complex-tensor ops) is pathologically slow — `torch.compile` alone recovers ~5× of the K3 micro-kernel by fusion. Most of the eager-relative "recoverable" pie is inefficiency the compiler also removes; CODA's marginal win over the compiler is ~12%.
- **The ceiling is set by the workload, not the method.** At `N = 32760`, FlashAttention is ~55–58% of the block and the GEMM FLOPs are ~16% — a ~70% floor CODA physically cannot touch. On the addressable ~28%, 1.14× is close to the ceiling.
- **This is a video-length result.** At shorter sequences the non-attention share is larger (a single-frame profile put it near 17%), so an image DiT or short clip would show a bigger relative win. Video's long sequence is the *least* favourable home for CODA — it stacks *with* FlashAttention, it does not replace it.

The takeaway is not "CODA gives 12% on Wan." It is: **CODA transfers cleanly to a DiT's modulation/gate/residual/norm structure (the kernels are correct and 2×+ on their own slices), but on a long-sequence video sampler the addressable surface is small, and DiT's dynamic modulation closes the one structural door that would let you fuse further.** The method works; this workload just doesn't have much for it to lift.

---

## 7. The methodology that mattered

The single most important process lesson: **profile the fused run itself, five-bucketed, and gate every change on measured ms — never on "one fewer kernel".**

The first profiling pass mis-attributed QK-norm + RoPE (17.8% of the block, the *largest* fusible seam) into an "attention support" bucket and concluded a 1.13× ceiling — nearly writing the whole direction off. Re-profiling the CODA run, separating foldable work from the genuinely-irreducible FA plumbing (which turned out to be <1%), is what revealed the real ceiling and turned 1.01× into 1.14×. The same gate then correctly killed three dead-end levers (A, B, C) instead of grinding GPU hours into them.

If you take one thing from this study: an honest five-bucket profile of your *own* fused kernel, with a hard per-lever gate, is worth more than any amount of clever epilogue code written before you know where the time actually is.
