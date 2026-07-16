"""CODA K1/K3 kernels and inference-only zero-duplicate prepacking for Wan."""

from __future__ import annotations

import math
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import torch
import torch.nn as nn
import triton
import triton.language as tl

from kernels.gens.epilogue.kernel_0 import _create_mean_sq_reduction_op
from kernels.gens.gpt import _dispatch
from rapier.epilogue import (
    EVTColBlockReductionStore,
    EVTList,
    EVTRowOrColBias,
)
from rapier.gemm.gemm_interface import (
    gemm_epilogue,
    preprocess_tensor,
    preprocess_vector,
)
from rapier.ops import misc_utils


DIM = 1536
HEAD_DIM = 128
RMS_BLOCK = 128


@triton.jit
def _layernorm_modulation_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / D
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = centered * tl.rsqrt(var + EPS)
    y = y * (1.0 + scale) + shift
    tl.store(out_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)


def layernorm_modulation_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fuse Wan's affine-free LayerNorm and modulation, storing bf16 once."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    assert x2.shape[1] == DIM
    assert scale.numel() == shift.numel() == DIM
    out = torch.empty_like(x2, dtype=torch.bfloat16)
    _layernorm_modulation_kernel[(x2.shape[0],)](
        x2,
        scale.reshape(-1),
        shift.reshape(-1),
        out,
        D=DIM,
        EPS=eps,
        BLOCK=2048,
        num_warps=8,
    )
    return out.view(shape)


class EVTGELUBias(EVTRowOrColBias):
    """Add a row bias and apply Wan's tanh-GELU in accumulator registers."""

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EVTRowOrColBias.EpilogueParams,
        epi_tensors_loop: EVTRowOrColBias.EpilogueTensorsLoop,
    ) -> EVTRowOrColBias.EpilogueTensorsLoop:
        row = misc_utils.static_assert_is_Tensor(
            epi_tensors_loop.tDrRowVec_epi
        )
        for i in cutlass.range_constexpr(cute.size(row)):
            x = tRS_rD[i] + row[i]
            z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            tRS_rD[i] = 0.5 * x * (1.0 + cute.math.tanh(z))
        return self.EpilogueTensorsLoop(
            tDrRowVec_epi=row,
            tDrColVec_epi=epi_tensors_loop.tDrColVec_epi,
        )


def _prepare_gelu_epilogue(shape_mnkl, tile_shape_mn, bias):
    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTGELUBias(
        acc_dtype=acc_dtype,
        tile_shape_mnk=tile_shape_mnk,
    )
    epi_args = EVTGELUBias.EpilogueArguments(mRowVec=bias, mColVec=None)
    return epi_cls, epi_args, {}, (bias.dtype, EVTGELUBias)


GELU_KERNEL = SimpleNamespace(prepare_epilogue=_prepare_gelu_epilogue)


def linear_bias_gelu_fixed(
    a: torch.Tensor,
    b_kn: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute GELU-tanh(A @ B + bias), storing only the activated bf16 output."""
    assert a.ndim == b_kn.ndim == 2 and bias.ndim == 1
    assert a.shape[1] == b_kn.shape[0] and b_kn.shape[1] == bias.shape[0]
    out = torch.empty(
        (a.shape[0], b_kn.shape[1]), dtype=a.dtype, device=a.device
    )
    a3 = preprocess_tensor(a, permute=True, transpose=False)
    b3 = preprocess_tensor(b_kn, permute=True, transpose=True)
    out3 = preprocess_tensor(out, permute=True, transpose=False)
    bias2 = preprocess_vector(bias, permute=False)
    epi_cls, epi_args, _, epi_keys = _prepare_gelu_epilogue(
        (a.shape[0], b_kn.shape[1], a.shape[1], 1),
        (128, 192),
        bias=bias2,
    )
    gemm_epilogue(
        A=a3,
        B=b3,
        C=out3,
        epi_cls=epi_cls,
        epi_args=epi_args,
        epi_keys=epi_keys,
        pingpong=True,
        tile_shape_mn=(128, 192),
        cluster_shape_mn=(2, 1),
        add_to_output=False,
    )
    return out


def _prepare_bias_partial_rms(shape_mnkl, tile_shape_mn, bias, partial):
    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList(
        [
            EVTRowOrColBias(
                acc_dtype=acc_dtype,
                tile_shape_mnk=tile_shape_mnk,
            ),
            EVTColBlockReductionStore(
                reduction_op=_create_mean_sq_reduction_op(
                    element_type=acc_dtype,
                    inv_block_size=1.0 / tile_shape_mnk[1],
                ),
                tile_shape_mnk=tile_shape_mnk,
            ),
        ]
    )
    epi_args = EVTList.EpilogueArguments(
        [
            EVTRowOrColBias.EpilogueArguments(
                mRowVec=bias,
                mColVec=None,
            ),
            EVTColBlockReductionStore.EpilogueArguments(mColVec=partial),
        ]
    )
    keys = (
        bias.dtype,
        partial.dtype,
        EVTRowOrColBias,
        EVTColBlockReductionStore,
    )
    return epi_cls, epi_args, {}, keys


QK_PARTIAL_KERNEL = SimpleNamespace(prepare_epilogue=_prepare_bias_partial_rms)


def qk_linear_bias_partial_rms(
    a: torch.Tensor,
    qk_weight_kn: torch.Tensor,
    qk_bias: torch.Tensor,
    block_size: int = RMS_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Q+K GEMM with bias and CODA epilogue partial mean-square reduction."""
    m, k = a.shape
    assert qk_weight_kn.shape == (k, 2 * DIM)
    assert qk_bias.shape == (2 * DIM,)
    assert (2 * DIM) % block_size == 0
    qk = torch.empty((m, 2 * DIM), dtype=a.dtype, device=a.device)
    partial = torch.empty(
        (m, 2 * DIM // block_size), dtype=torch.float32, device=a.device
    )
    bias2 = preprocess_vector(qk_bias, permute=False)
    partial3 = preprocess_tensor(partial, permute=False)
    a3 = preprocess_tensor(a, permute=True, transpose=False)
    b3 = preprocess_tensor(qk_weight_kn, permute=True, transpose=True)
    qk3 = preprocess_tensor(qk, permute=True, transpose=False)
    epi_cls, epi_args, _, epi_keys = _prepare_bias_partial_rms(
        (m, 2 * DIM, k, 1),
        (128, block_size),
        bias=bias2,
        partial=partial3,
    )
    gemm_epilogue(
        A=a3,
        B=b3,
        C=qk3,
        epi_cls=epi_cls,
        epi_args=epi_args,
        epi_keys=epi_keys,
        pingpong=True,
        tile_shape_mn=(128, block_size),
        cluster_shape_mn=(2, 2),
        add_to_output=False,
    )
    return qk, partial3.squeeze(0)


@triton.jit
def _qk_rms_rope_kernel(
    qk_ptr,
    partial_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    PARTS: tl.constexpr,
    HD: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    half = chunk // (D // BLOCK)
    d = (chunk % (D // BLOCK)) * BLOCK + offs
    valid = d < D

    part_idx = tl.arange(0, 16)
    part = tl.load(
        partial_ptr + row * (2 * PARTS) + half * PARTS + part_idx,
        mask=part_idx < PARTS,
        other=0.0,
    )
    mean_sq = tl.sum(part, axis=0) / PARTS
    rstd = tl.rsqrt(mean_sq + EPS)

    base = row * (2 * D) + half * D
    x = tl.load(qk_ptr + base + d, mask=valid, other=0.0).to(tl.float32)
    partner_d = d ^ 1
    partner = tl.load(
        qk_ptr + base + partner_d, mask=valid, other=0.0
    ).to(tl.float32)
    qw = tl.load(q_weight_ptr + d, mask=valid, other=0.0).to(tl.float32)
    kw = tl.load(k_weight_ptr + d, mask=valid, other=0.0).to(tl.float32)
    w = tl.where(half == 0, qw, kw)
    qw_partner = tl.load(
        q_weight_ptr + partner_d, mask=valid, other=0.0
    ).to(tl.float32)
    kw_partner = tl.load(
        k_weight_ptr + partner_d, mask=valid, other=0.0
    ).to(tl.float32)
    w_partner = tl.where(half == 0, qw_partner, kw_partner)

    # Match WanRMSNorm's bf16 rounding before affine weight multiplication.
    y = (x * rstd).to(tl.bfloat16).to(tl.float32)
    yp = (partner * rstd).to(tl.bfloat16).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)
    yp = (yp * w_partner).to(tl.bfloat16).to(tl.float32)

    pair = (d % HD) // 2
    cos = tl.load(cos_ptr + row * (HD // 2) + pair, mask=valid, other=1.0)
    sin = tl.load(sin_ptr + row * (HD // 2) + pair, mask=valid, other=0.0)
    is_even = (d & 1) == 0
    even = tl.where(is_even, y, yp)
    odd = tl.where(is_even, yp, y)
    rotated = tl.where(
        is_even,
        even * cos - odd * sin,
        even * sin + odd * cos,
    )
    tl.store(
        q_out_ptr + row * OUT_ROW_STRIDE + d,
        rotated,
        mask=valid & (half == 0),
    )
    tl.store(
        k_out_ptr + row * OUT_ROW_STRIDE + d,
        rotated,
        mask=valid & (half == 1),
    )


def qk_rms_rope_from_partials(
    qk: torch.Tensor,
    partial: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = qk.shape[0]
    assert qk.shape == (m, 2 * DIM)
    parts = DIM // RMS_BLOCK
    assert partial.shape == (m, 2 * parts)
    assert cos.shape == sin.shape == (m, HEAD_DIM // 2)
    # The auxiliary kernel consumes each 256-wide pair chunk entirely before
    # storing, so its outputs can safely overwrite the corresponding Q/K
    # halves.  This removes a 2*M*D bf16 runtime duplicate (~192 MiB at Wan's
    # real sequence length).
    q = qk[:, :DIM]
    k = qk[:, DIM:]
    block = 256
    _qk_rms_rope_kernel[(m, 2 * DIM // block)](
        qk,
        partial,
        q_weight,
        k_weight,
        cos,
        sin,
        q,
        k,
        M=m,
        D=DIM,
        PARTS=parts,
        HD=HEAD_DIM,
        EPS=eps,
        BLOCK=block,
        OUT_ROW_STRIDE=q.stride(0),
        num_warps=4,
    )
    return q, k


def qk_coda_rms_rope(
    a: torch.Tensor,
    qk_weight_kn: torch.Tensor,
    qk_bias: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qk, partial = qk_linear_bias_partial_rms(a, qk_weight_kn, qk_bias)
    return qk_rms_rope_from_partials(
        qk, partial, q_weight, k_weight, cos, sin, eps
    )


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / D
    y = (x * tl.rsqrt(mean_sq + EPS)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(out_ptr + row * D + offs, out, mask=mask)


def rmsnorm_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    assert x2.shape[1] == DIM
    out = torch.empty_like(x2)
    _rmsnorm_kernel[(x2.shape[0],)](
        x2,
        weight,
        out,
        M=x2.shape[0],
        D=DIM,
        EPS=eps,
        BLOCK=2048,
        num_warps=8,
    )
    return out.view(shape)


def build_wan_rope_table(
    grid_fhw: tuple[int, int, int],
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one shared per-token/per-head-pair table, not a per-layer D-wide copy."""
    f, h, w = grid_fhw
    c = HEAD_DIM // 2
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    table = torch.cat(
        [
            parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, c)
    return table.real.float().contiguous(), table.imag.float().contiguous()


def replace_parameter_with_view(
    module: nn.Module,
    name: str,
    view: torch.Tensor,
) -> None:
    module._parameters[name] = nn.Parameter(view, requires_grad=False)


def prepack_linear_inplace(linear: nn.Linear) -> torch.Tensor:
    """Transpose once and make Linear.weight a view of that same storage."""
    packed = linear.weight.detach().T.contiguous()
    replace_parameter_with_view(linear, "weight", packed.T)
    return packed


def prepack_qk_inplace(
    q: nn.Linear,
    k: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack Q and K into one Kx2N allocation; original parameters become views."""
    packed = torch.cat(
        [q.weight.detach().T, k.weight.detach().T], dim=1
    ).contiguous()
    bias = torch.cat([q.bias.detach(), k.bias.detach()]).contiguous()
    replace_parameter_with_view(q, "weight", packed[:, :DIM].T)
    replace_parameter_with_view(k, "weight", packed[:, DIM:].T)
    replace_parameter_with_view(q, "bias", bias[:DIM])
    replace_parameter_with_view(k, "bias", bias[DIM:])
    return packed, bias


def unique_storage_bytes(module: nn.Module, extras: list[torch.Tensor]) -> int:
    """Count unique CUDA storage capacity across parameters, buffers and extras."""
    seen: set[tuple[int, int]] = set()
    total = 0
    tensors = list(module.parameters()) + list(module.buffers()) + extras
    for tensor in tensors:
        if tensor.device.type != "cuda":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total
