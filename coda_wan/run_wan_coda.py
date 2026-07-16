#!/usr/bin/env python3
"""Full 30-block Wan eager, max-autotune, or K1+K2+K3 CODA benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

import wan.modules.model as wan_model_module
from wan.modules.model import WanModel

if __package__:
    from .wan_coda_kernels import (
        DIM,
        build_wan_rope_table,
        layernorm_modulation_bf16,
        linear_bias_gelu_fixed,
        prepack_linear_inplace,
        prepack_qk_inplace,
        qk_coda_rms_rope,
        rmsnorm_fused,
        unique_storage_bytes,
    )
    from .wan_gate_residual_coda import (
        linear_bias_residual_fixed,
        linear_gate_bias_residual_fixed,
    )
else:
    from wan_coda_kernels import (
        DIM,
        build_wan_rope_table,
        layernorm_modulation_bf16,
        linear_bias_gelu_fixed,
        prepack_linear_inplace,
        prepack_qk_inplace,
        qk_coda_rms_rope,
        rmsnorm_fused,
        unique_storage_bytes,
    )
    from wan_gate_residual_coda import (
        linear_bias_residual_fixed,
        linear_gate_bias_residual_fixed,
    )


SEED = 20260715
SEQ_LEN = 32760
LATENT_SHAPE = (16, 21, 60, 104)
TEXT_SHAPE = (512, 4096)
GRID_FHW = (21, 30, 52)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("eager", "compile", "coda"), required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def force_fa2() -> None:
    original = wan_model_module.flash_attention

    def wrapped(*args, **kwargs):
        kwargs["version"] = 2
        return original(*args, **kwargs)

    wan_model_module.flash_attention = wrapped


def storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class CodaFullBlock(nn.Module):
    """Sampling-only Wan block with K1, K2, and K3 fused seams."""

    def __init__(self, inner: nn.Module, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        super().__init__()
        self.inner = inner
        qk_weight, qk_bias = prepack_qk_inplace(
            inner.self_attn.q, inner.self_attn.k
        )
        self.register_buffer("qk_weight_kn", qk_weight, persistent=False)
        self.register_buffer("qk_bias", qk_bias, persistent=False)
        self.register_buffer(
            "self_o_weight_kn",
            prepack_linear_inplace(inner.self_attn.o),
            persistent=False,
        )
        self.register_buffer(
            "cross_o_weight_kn",
            prepack_linear_inplace(inner.cross_attn.o),
            persistent=False,
        )
        self.register_buffer(
            "ffn1_weight_kn",
            prepack_linear_inplace(inner.ffn[0]),
            persistent=False,
        )
        self.register_buffer(
            "ffn2_weight_kn",
            prepack_linear_inplace(inner.ffn[2]),
            persistent=False,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def alias_audit(self) -> dict:
        q = self.inner.self_attn.q
        k = self.inner.self_attn.k
        o = self.inner.self_attn.o
        co = self.inner.cross_attn.o
        f1 = self.inner.ffn[0]
        f2 = self.inner.ffn[2]
        checks = {
            "q_weight": storage_ptr(q.weight) == storage_ptr(self.qk_weight_kn),
            "k_weight": storage_ptr(k.weight) == storage_ptr(self.qk_weight_kn),
            "q_bias": storage_ptr(q.bias) == storage_ptr(self.qk_bias),
            "k_bias": storage_ptr(k.bias) == storage_ptr(self.qk_bias),
            "self_o_weight": storage_ptr(o.weight)
            == storage_ptr(self.self_o_weight_kn),
            "cross_o_weight": storage_ptr(co.weight)
            == storage_ptr(self.cross_o_weight_kn),
            "ffn1_weight": storage_ptr(f1.weight)
            == storage_ptr(self.ffn1_weight_kn),
            "ffn2_weight": storage_ptr(f2.weight)
            == storage_ptr(self.ffn2_weight_kn),
        }
        return {"checks": checks, "all_share_storage": all(checks.values())}

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        del freqs
        assert x.shape[0] == 1 and tuple(grid_sizes[0].tolist()) == GRID_FHW
        e = (self.inner.modulation.float() + e.float()).chunk(6, dim=1)

        z = layernorm_modulation_bf16(x, e[1], e[0], self.inner.eps)
        q, k = qk_coda_rms_rope(
            z.reshape(-1, DIM),
            self.qk_weight_kn,
            self.qk_bias,
            self.inner.self_attn.norm_q.weight,
            self.inner.self_attn.norm_k.weight,
            self.rope_cos,
            self.rope_sin,
            self.inner.self_attn.eps,
        )
        b, s, n, d = 1, x.shape[1], self.inner.num_heads, DIM // self.inner.num_heads
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = self.inner.self_attn.v(z).view(b, s, n, d)
        y = wan_model_module.flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.inner.window_size,
        )
        x = linear_gate_bias_residual_fixed(
            y.flatten(2).reshape(-1, DIM).bfloat16(),
            self.self_o_weight_kn,
            x.reshape(-1, DIM).float(),
            e[2].reshape(-1),
            self.inner.self_attn.o.bias,
            inplace=True,
        ).view(b, s, DIM)

        ca = self.inner.cross_attn
        cross_in = self.inner.norm3(x)
        cq = rmsnorm_fused(ca.q(cross_in), ca.norm_q.weight, ca.eps).view(
            b, -1, n, d
        )
        ck = rmsnorm_fused(ca.k(context), ca.norm_k.weight, ca.eps).view(
            b, -1, n, d
        )
        cv = ca.v(context).view(b, -1, n, d)
        cy = wan_model_module.flash_attention(cq, ck, cv, k_lens=context_lens)
        x = linear_bias_residual_fixed(
            cy.flatten(2).reshape(-1, DIM).bfloat16(),
            self.cross_o_weight_kn,
            x.reshape(-1, DIM).float(),
            ca.o.bias,
            inplace=True,
        ).view(b, s, DIM)

        z = layernorm_modulation_bf16(x, e[4], e[3], self.inner.eps)
        h = linear_bias_gelu_fixed(
            z.reshape(-1, DIM), self.ffn1_weight_kn, self.inner.ffn[0].bias
        )
        x = linear_gate_bias_residual_fixed(
            h,
            self.ffn2_weight_kn,
            x.reshape(-1, DIM).float(),
            e[5].reshape(-1),
            self.inner.ffn[2].bias,
            inplace=True,
        ).view(b, s, DIM)
        return x


def make_model(model_dir: Path) -> WanModel:
    weights = model_dir / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"pretrained Wan weights required: {weights}")
    return WanModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda().eval()


def numerical(got: torch.Tensor, ref: torch.Tensor) -> dict:
    got = got.float()
    ref = ref.float()
    diff = (got - ref).abs()
    tol = 0.02 + 0.02 * ref.abs()
    mse = (got - ref).square().mean()
    data_range = ref.max() - ref.min()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "pass_fraction": (diff <= tol).float().mean().item(),
        "allclose_rtol_2e-2_atol_2e-2": torch.allclose(
            got, ref, rtol=2e-2, atol=2e-2
        ),
        "latent_psnr_db": float(20 * torch.log10(data_range / torch.sqrt(mse))),
    }


def main() -> None:
    a = parse_args()
    assert a.warmup >= 10 and a.trials >= 30
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    force_fa2()
    model = make_model(a.model_dir)
    x = [torch.randn(*LATENT_SHAPE, device="cuda", dtype=torch.bfloat16)]
    t = torch.tensor([900], device="cuda", dtype=torch.long)
    context = [torch.randn(*TEXT_SHAPE, device="cuda", dtype=torch.bfloat16)]

    def run(current_model):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return current_model(x, t, context, SEQ_LEN)[0]

    reference = run(model).detach().clone()
    storage = None
    if a.variant == "coda":
        before = unique_storage_bytes(model, [])
        if model.freqs.device.type != "cuda":
            model.freqs = model.freqs.cuda()
        rope_cos, rope_sin = build_wan_rope_table(GRID_FHW, model.freqs)
        model.blocks = nn.ModuleList(
            [CodaFullBlock(block, rope_cos, rope_sin) for block in model.blocks]
        )
        after = unique_storage_bytes(model, [])
        audits = [block.alias_audit() for block in model.blocks]
        storage = {
            "unique_cuda_storage_before_bytes": before,
            "unique_cuda_storage_after_bytes": after,
            "delta_bytes": after - before,
            "shared_rope_table_bytes": (
                rope_cos.untyped_storage().nbytes() + rope_sin.untyped_storage().nbytes()
            ),
            "prepacked_layer_count": len(audits),
            "all_prepacked_parameters_share_storage": all(
                row["all_share_storage"] for row in audits
            ),
            "transpose_duplicate_bytes": 0,
            "first_layer_alias_audit": audits[0],
        }
        bench_model = model
    elif a.variant == "compile":
        bench_model = torch.compile(model, mode="max-autotune", fullgraph=False)
    else:
        bench_model = model

    candidate = run(bench_model)
    accuracy = numerical(candidate, reference)
    del candidate, reference
    torch.cuda.empty_cache()

    for _ in range(a.warmup):
        run(bench_model)
    torch.cuda.synchronize()
    rows, peaks = [], []
    for _ in range(a.trials):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run(bench_model)
        end.record()
        torch.cuda.synchronize()
        rows.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
        del out

    median_ms = statistics.median(rows)
    payload = {
        "seed": SEED,
        "variant": a.variant,
        "model_source": "pretrained",
        "shape": {
            "latent": list(LATENT_SHAPE),
            "seq_len": SEQ_LEN,
            "text": list(TEXT_SHAPE),
            "grid_fhw": list(GRID_FHW),
            "layers": 30,
            "dim": DIM,
            "ffn_dim": 8960,
        },
        "dtype": "sampling-only bf16; Wan fp32 residual/modulation preserved",
        "attention": "FlashAttention-2",
        "compile_mode": "max-autotune" if a.variant == "compile" else None,
        "warmup": a.warmup,
        "trials": a.trials,
        "median_ms": median_ms,
        "tokens_per_second": SEQ_LEN / (median_ms / 1e3),
        "peak_allocated_gib_median": statistics.median(peaks),
        "numerical_vs_eager_same_seed_same_weights": accuracy,
        "storage_audit": storage,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
