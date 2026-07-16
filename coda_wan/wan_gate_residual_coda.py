"""Wan gate+residual GEMM epilogue implemented on CODA/Rapier CuTeDSL."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import cutlass
import cutlass.cute as cute

from quack.cute_dsl_utils import torch2cute_dtype_map
from rapier.epilogue import EVTList, EVTResidual, EVTRowOrColBias, EpilogueVisitorTree
from rapier.ops import misc_utils
from rapier.gemm.gemm_interface import (
    gemm_epilogue,
    preprocess_tensor,
    preprocess_vector,
)
from kernels.gens.gpt import _dispatch


class EVTRowScale(EVTRowOrColBias):
    """Reuse CODA's rank-1 loader but multiply the accumulator by row vector."""

    @cute.jit
    def consumer_visit(
        self,
        tRS_rD: cute.Tensor,
        shape_mnk: cute.Shape,
        epi_params: EVTRowOrColBias.EpilogueParams,
        epi_tensors_loop: EVTRowOrColBias.EpilogueTensorsLoop,
    ) -> EVTRowOrColBias.EpilogueTensorsLoop:
        if cutlass.const_expr(epi_tensors_loop.tDrRowVec_epi is not None):
            row = misc_utils.static_assert_is_Tensor(
                epi_tensors_loop.tDrRowVec_epi
            )
            for i in cutlass.range_constexpr(cute.size(row)):
                tRS_rD[i] = tRS_rD[i] * row[i]
        else:
            row = epi_tensors_loop.tDrRowVec_epi
        # This Wan kernel only needs a per-channel row vector.
        col = epi_tensors_loop.tDrColVec_epi
        return self.EpilogueTensorsLoop(
            tDrRowVec_epi=row,
            tDrColVec_epi=col,
        )


def prepare_epilogue(shape_mnkl, tile_shape_mn, C, G):
    """Compose `(A @ B) * G + C` in accumulator registers."""
    epi_dtype = torch2cute_dtype_map[C.dtype]
    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList(
        [
            EVTRowScale(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk),
            EVTResidual(
                acc_dtype=acc_dtype,
                epi_dtype=epi_dtype,
                tile_shape_mnk=tile_shape_mnk,
                buffer_align_bytes=buffer_align_bytes,
            ),
        ]
    )
    epi_args = EVTList.EpilogueArguments(
        [
            EVTRowScale.EpilogueArguments(mRowVec=G, mColVec=None),
            EVTResidual.EpilogueArguments(mMatrix=C),
        ]
    )
    return epi_cls, epi_args, {}, (C.dtype, G.dtype, EVTRowScale, EVTResidual)


KERNEL = SimpleNamespace(prepare_epilogue=prepare_epilogue)


def prepare_epilogue_bias_residual(shape_mnkl, tile_shape_mn, C, bias):
    """Compose `A @ B + bias + C` without a synthetic all-ones gate."""
    epi_dtype = torch2cute_dtype_map[C.dtype]
    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList(
        [
            EVTRowOrColBias(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk),
            EVTResidual(
                acc_dtype=acc_dtype,
                epi_dtype=epi_dtype,
                tile_shape_mnk=tile_shape_mnk,
                buffer_align_bytes=buffer_align_bytes,
            ),
        ]
    )
    epi_args = EVTList.EpilogueArguments(
        [
            EVTRowOrColBias.EpilogueArguments(mRowVec=bias, mColVec=None),
            EVTResidual.EpilogueArguments(mMatrix=C),
        ]
    )
    keys = (C.dtype, bias.dtype, EVTRowOrColBias, EVTResidual)
    return epi_cls, epi_args, {}, keys


def linear_bias_residual_fixed(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Compute `A @ B + bias + C` in one CODA GEMM epilogue."""
    assert A.ndim == B.ndim == C.ndim == 2
    assert bias.shape == (B.shape[1],)
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1]
    out = C if inplace else torch.empty_like(C)
    A3 = preprocess_tensor(A, permute=True, transpose=False)
    B3 = preprocess_tensor(B, permute=True, transpose=True)
    C3 = preprocess_tensor(C, permute=True, transpose=False)
    out3 = preprocess_tensor(out, permute=True, transpose=False)
    bias2 = preprocess_vector(bias, permute=False)
    epi_cls, epi_args, _, epi_keys = prepare_epilogue_bias_residual(
        (A.shape[0], B.shape[1], A.shape[1], 1),
        (128, 128),
        C=C3,
        bias=bias2,
    )
    gemm_epilogue(
        A=A3,
        B=B3,
        C=out3,
        epi_cls=epi_cls,
        epi_args=epi_args,
        epi_keys=epi_keys,
        pingpong=True,
        tile_shape_mn=(128, 128),
        cluster_shape_mn=(2, 2),
        add_to_output=False,
    )
    return out


def prepare_epilogue_bias(shape_mnkl, tile_shape_mn, C, G, BG):
    """Compose `(A @ B) * G + bias * G + C`."""
    epi_dtype = torch2cute_dtype_map[C.dtype]
    epi_cls = lambda acc_dtype, tile_shape_mnk, buffer_align_bytes: EVTList(
        [
            EVTRowScale(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk),
            EVTRowOrColBias(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk),
            EVTResidual(
                acc_dtype=acc_dtype,
                epi_dtype=epi_dtype,
                tile_shape_mnk=tile_shape_mnk,
                buffer_align_bytes=buffer_align_bytes,
            ),
        ]
    )
    epi_args = EVTList.EpilogueArguments(
        [
            EVTRowScale.EpilogueArguments(mRowVec=G, mColVec=None),
            EVTRowOrColBias.EpilogueArguments(mRowVec=BG, mColVec=None),
            EVTResidual.EpilogueArguments(mMatrix=C),
        ]
    )
    keys = (C.dtype, G.dtype, BG.dtype, EVTRowScale, EVTRowOrColBias, EVTResidual)
    return epi_cls, epi_args, {}, keys


def linear_gate_residual(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Compute `(A @ B) * gate + C`; B uses CODA's `(K,N)` convention."""
    assert A.ndim == B.ndim == C.ndim == 2
    assert gate.ndim == 1
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1] == gate.shape[0]
    out = torch.empty_like(C)
    C3 = preprocess_tensor(C, permute=True)
    G2 = preprocess_vector(gate, permute=False)
    _dispatch(
        kernel=KERNEL,
        A=A,
        B=B,
        out=out,
        add_to_output=False,
        block_size_m=None,
        block_size_n=None,
        C=C3,
        G=G2,
    )
    return out


def linear_gate_bias_residual_fixed(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Exact Wan Linear+bias+gate+residual with per-shape tuned H100 tiles.

    FFN2 uses 128x192/cluster 1x2; the 1536-wide projection uses the
    independently tuned 128x128/cluster 2x2 configuration.
    """
    assert gate.shape == bias.shape == (B.shape[1],)
    # GEMM's output and the residual input may alias: each epilogue tile loads
    # its residual fragment before storing the result, matching standard C=D
    # GEMM semantics.  Sampling-only Wan can therefore update the residual
    # buffer in place and avoid one full fp32 MxN allocation.
    out = C if inplace else torch.empty_like(C)
    C3 = preprocess_tensor(C, permute=True)
    G2 = preprocess_vector(gate, permute=False)
    BG2 = preprocess_vector((bias.float() * gate).contiguous(), permute=False)
    A3 = preprocess_tensor(A, permute=True, transpose=False)
    B3 = preprocess_tensor(B, permute=True, transpose=True)
    out3 = preprocess_tensor(out, permute=True, transpose=False)
    # Exhaustive real-shape H100 tuning selects a wider N tile for Wan FFN2;
    # the 1536->1536 projection keeps its independently tuned square tile.
    is_ffn2 = A.shape[1] == 8960 and B.shape[1] == 1536
    tile_shape_mn = (128, 192) if is_ffn2 else (128, 128)
    cluster_shape_mn = (1, 2) if is_ffn2 else (2, 2)
    epi_cls, epi_args, _, epi_keys = prepare_epilogue_bias(
        (A.shape[0], B.shape[1], A.shape[1], 1),
        tile_shape_mn,
        C=C3,
        G=G2,
        BG=BG2,
    )
    gemm_epilogue(
        A=A3,
        B=B3,
        C=out3,
        epi_cls=epi_cls,
        epi_args=epi_args,
        epi_keys=epi_keys,
        pingpong=True,
        tile_shape_mn=tile_shape_mn,
        cluster_shape_mn=cluster_shape_mn,
        add_to_output=False,
    )
    return out
