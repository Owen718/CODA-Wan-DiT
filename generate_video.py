#!/usr/bin/env python3
"""Generate complete Wan2.1-T2V-1.3B videos and time only the DiT forwards.

The authoritative CODA implementation is imported from the NRT v3 benchmark
modules.  Every trial performs the complete T5 -> 50-step DiT -> VAE pipeline.
CUDA events cover only the two complete WanModel forwards (conditional and
unconditional) in each denoising step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import structural_similarity

from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

try:
    # NRT server layout; this is the source of truth used for GPU validation.
    from wan_full_coda_v3_bench import (
        CodaFullBlock,
        GRID_FHW,
        force_fa2,
        unique_storage_bytes,
    )
    from wan_coda_kernels_v3 import build_wan_rope_table
    CODA_IMPORT_BACKEND = "server_v3"
except ModuleNotFoundError as exc:
    if exc.name != "wan_full_coda_v3_bench":
        raise
    # Repository layout after the server-tested sources are pulled back.
    from coda_wan.run_wan_coda import (
        CodaFullBlock,
        GRID_FHW,
        force_fa2,
        unique_storage_bytes,
    )
    from coda_wan.wan_coda_kernels import build_wan_rope_table
    CODA_IMPORT_BACKEND = "repo_pulled_from_server"


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)
DEFAULT_SEED = 20260715
TIMING_LABEL = "DiT-only, T5/VAE excluded"
EXPECTED_GRID = (21, 30, 52)
EXPECTED_LATENT_SHAPE = (16, 21, 60, 104)
EXPECTED_SEQ_LEN = 32760
FRAME_INDICES = (0, 11, 23, 34, 46, 57, 69, 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("eager", "compile", "coda"), required=True
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Skip MP4/GIF/PNG work (intended only for short smoke tests).",
    )
    return parser.parse_args()


def run_command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit_for_source(source: Path) -> str | None:
    """Return the nearest enclosing Git commit for an imported source file."""
    for candidate in (source.parent, *source.parents):
        if (candidate / ".git").exists():
            return run_command(["git", "rev-parse", "HEAD"], cwd=candidate)
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(array)))
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def gib(value: int | float) -> float:
    return float(value) / (2**30)


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def checkpoint_audit(checkpoint_dir: Path) -> dict[str, Any]:
    required = {
        "dit_config": checkpoint_dir / "config.json",
        "dit_weights": checkpoint_dir / "diffusion_pytorch_model.safetensors",
        "t5_weights": checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        "vae_weights": checkpoint_dir / "Wan2.1_VAE.pth",
        "tokenizer_model": checkpoint_dir / "google/umt5-xxl/spiece.model",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete Wan checkpoint; missing: {missing}")
    return {
        key: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if key in {"dit_config"} else None,
        }
        for key, path in required.items()
    }


def source_audit() -> dict[str, Any]:
    import inspect

    gate_module_name = (
        "wan_gate_residual_coda_v3"
        if CODA_IMPORT_BACKEND == "server_v3"
        else "coda_wan.wan_gate_residual_coda"
    )
    files = {
        "generate_video": Path(__file__).resolve(),
        "coda_full_block": Path(inspect.getsourcefile(CodaFullBlock)).resolve(),
        "coda_rope_builder": Path(inspect.getsourcefile(build_wan_rope_table)).resolve(),
        "coda_gate_residual": Path(sys.modules[gate_module_name].__file__).resolve(),
    }
    return {
        key: {"path": str(path), "sha256": sha256_file(path)}
        for key, path in files.items()
    }


def environment_metadata(device: int) -> dict[str, Any]:
    gpu_query = run_command(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_name = torch.cuda.get_device_name(device)
    gpu_uuid = None
    driver = None
    if gpu_query:
        fields = [item.strip() for item in gpu_query.splitlines()[0].split(",")]
        if len(fields) == 3:
            gpu_name, gpu_uuid, driver = fields
    try:
        flash_attn_version = importlib.metadata.version("flash-attn")
    except importlib.metadata.PackageNotFoundError:
        flash_attn_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn_version,
        "python": sys.version.split()[0],
        "attention_backend": "FlashAttention-2 (forced by server v3 force_fa2)",
        "dit_dtype": "bfloat16",
        "t5_dtype": "bfloat16",
        "vae_dtype": "float32 (official Wan decoder)",
        "training": False,
        "fp8": False,
    }


def install_variant(
    pipeline: WanT2V, variant: str, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    model = pipeline.model.to(device=device, dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    first_dtype = str(next(model.parameters()).dtype)
    if first_dtype != "torch.bfloat16":
        raise AssertionError(f"DiT must be bf16, got {first_dtype}")

    build: dict[str, Any] = {
        "variant": variant,
        "torch_compile": None,
        "compiled_component": None,
        "coda": False,
        "coda_block_count": 0,
        "batch_per_cfg_forward": 1,
    }
    if variant == "coda":
        before = unique_storage_bytes(model, [])
        if model.freqs.device.type != "cuda":
            model.freqs = model.freqs.to(device)
        rope_cos, rope_sin = build_wan_rope_table(EXPECTED_GRID, model.freqs)
        model.blocks = nn.ModuleList(
            [CodaFullBlock(block, rope_cos, rope_sin) for block in model.blocks]
        )
        audits = [block.alias_audit() for block in model.blocks]
        after = unique_storage_bytes(model, [])
        if len(model.blocks) != 30:
            raise AssertionError(f"expected 30 CODA blocks, got {len(model.blocks)}")
        if not all(row["all_share_storage"] for row in audits):
            raise AssertionError("CODA prepacked tensors do not all alias parameters")
        build.update(
            {
                "coda": True,
                "coda_block_count": len(model.blocks),
                "unique_cuda_storage_before_bytes": before,
                "unique_cuda_storage_after_bytes": after,
                "storage_delta_bytes": after - before,
                "all_prepacked_parameters_share_storage": True,
                "transpose_duplicate_bytes": 0,
                "first_layer_alias_audit": audits[0],
            }
        )
        return model, build
    if variant == "compile":
        compiled = torch.compile(model, mode="max-autotune", fullgraph=False)
        build.update(
            {
                "torch_compile": "max-autotune",
                "compiled_component": "DiT only",
            }
        )
        return compiled, build
    return model, build


def create_scheduler(
    pipeline: WanT2V, steps: int, shift: float
) -> tuple[FlowUniPCMultistepScheduler, torch.Tensor]:
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipeline.num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=pipeline.device, shift=shift)
    return scheduler, scheduler.timesteps


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return sha256_array(array)


def generate_once(
    pipeline: WanT2V,
    model: nn.Module,
    args: argparse.Namespace,
    *,
    measured: bool,
    keep_frames: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    device = pipeline.device
    target_shape = (
        pipeline.vae.model.z_dim,
        (args.frames - 1) // pipeline.vae_stride[0] + 1,
        args.height // pipeline.vae_stride[1],
        args.width // pipeline.vae_stride[2],
    )
    seq_len = math.ceil(
        target_shape[2]
        * target_shape[3]
        / (pipeline.patch_size[1] * pipeline.patch_size[2])
        * target_shape[1]
    )
    if args.variant == "coda" and (
        tuple(target_shape) != EXPECTED_LATENT_SHAPE or seq_len != EXPECTED_SEQ_LEN
    ):
        raise AssertionError(
            f"CODA requires latent={EXPECTED_LATENT_SHAPE}, seq={EXPECTED_SEQ_LEN}; "
            f"got latent={target_shape}, seq={seq_len}"
        )

    negative_prompt = (
        pipeline.sample_neg_prompt
        if args.negative_prompt is None
        else args.negative_prompt
    )
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()

    # T5 phase: encode both CFG branches, then offload only T5. DiT remains resident.
    torch.cuda.reset_peak_memory_stats(device)
    pipeline.text_encoder.model.to(device)
    context = pipeline.text_encoder([args.prompt], device)
    context_null = pipeline.text_encoder([negative_prompt], device)
    torch.cuda.synchronize(device)
    t5_peak_allocated = torch.cuda.max_memory_allocated(device)
    t5_peak_reserved = torch.cuda.max_memory_reserved(device)
    pipeline.text_encoder.model.cpu()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    noise = [
        torch.randn(
            *target_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ]
    initial_noise_sha256 = tensor_sha256(noise[0])
    scheduler, timesteps = create_scheduler(pipeline, args.steps, args.shift)
    latents = noise
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    event_rows: list[dict[str, Any]] = []
    pending_events: list[tuple[int, str, torch.cuda.Event, torch.cuda.Event]] = []
    model.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.param_dtype):
        for step_index, timestep_value in enumerate(timesteps):
            timestep = torch.stack([timestep_value])

            if measured:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(device))
            noise_pred_cond = model(latents, t=timestep, **arg_c)[0]
            if measured:
                end.record(torch.cuda.current_stream(device))
                pending_events.append((step_index, "cond", start, end))

            if measured:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(torch.cuda.current_stream(device))
            noise_pred_uncond = model(latents, t=timestep, **arg_null)[0]
            if measured:
                end.record(torch.cuda.current_stream(device))
                pending_events.append((step_index, "uncond", start, end))

            noise_pred = noise_pred_uncond + args.guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            next_latent = scheduler.step(
                noise_pred.unsqueeze(0),
                timestep_value,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=generator,
            )[0]
            latents = [next_latent.squeeze(0)]

        torch.cuda.synchronize(device)
        dit_peak_allocated = torch.cuda.max_memory_allocated(device)
        dit_peak_reserved = torch.cuda.max_memory_reserved(device)
        if measured:
            for step_index, branch, start, end in pending_events:
                event_rows.append(
                    {
                        "step": step_index,
                        "branch": branch,
                        "ms": float(start.elapsed_time(end)),
                    }
                )
            expected_calls = args.steps * 2
            if len(event_rows) != expected_calls:
                raise AssertionError(
                    f"expected {expected_calls} DiT forwards, got {len(event_rows)}"
                )

        # Official Wan VAE decode, outside every DiT event.
        torch.cuda.reset_peak_memory_stats(device)
        videos = pipeline.vae.decode(latents)
        torch.cuda.synchronize(device)
        vae_peak_allocated = torch.cuda.max_memory_allocated(device)
        vae_peak_reserved = torch.cuda.max_memory_reserved(device)

    wall_seconds = time.perf_counter() - wall_start
    frames_u8 = None
    decoded_sha256 = None
    if keep_frames:
        video = videos[0]
        frames_u8 = (
            ((video.clamp(-1, 1) + 1.0) * 127.5)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        decoded_sha256 = sha256_array(frames_u8)

    total_ms = float(sum(item["ms"] for item in event_rows)) if measured else None
    row = {
        "seed": args.seed,
        "initial_noise_sha256": initial_noise_sha256,
        "dit_forward_calls": len(event_rows) if measured else args.steps * 2,
        "dit_calls": event_rows,
        "dit_only_total_ms": total_ms,
        "dit_only_avg_step_ms": total_ms / args.steps if measured else None,
        "dit_only_avg_forward_ms": total_ms / (args.steps * 2) if measured else None,
        "dit_phase_peak_allocated_bytes": dit_peak_allocated,
        "dit_phase_peak_reserved_bytes": dit_peak_reserved,
        "t5_phase_peak_allocated_bytes": t5_peak_allocated,
        "t5_phase_peak_reserved_bytes": t5_peak_reserved,
        "vae_phase_peak_allocated_bytes": vae_peak_allocated,
        "vae_phase_peak_reserved_bytes": vae_peak_reserved,
        "full_pipeline_peak_allocated_bytes": max(
            t5_peak_allocated, dit_peak_allocated, vae_peak_allocated
        ),
        "full_pipeline_wall_seconds": wall_seconds,
        "decoded_rgb_sha256": decoded_sha256,
    }
    del videos, latents, noise, context, context_null, scheduler
    return row, frames_u8


def save_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to create MP4: {path}")


def save_gif(frames: np.ndarray, path: Path) -> dict[str, Any]:
    indices = np.rint(np.linspace(0, len(frames) - 1, 16)).astype(int).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = ((256, 128), (256, 96), (224, 96), (224, 64), (192, 64))
    for width, colors in attempts:
        height = int(round(frames.shape[1] * width / frames.shape[2]))
        images = []
        for index in indices:
            rgb = Image.fromarray(frames[index], mode="RGB").resize(
                (width, height), Image.Resampling.LANCZOS
            )
            images.append(
                rgb.quantize(
                    colors=colors,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.FLOYDSTEINBERG,
                )
            )
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=333,
            loop=0,
            optimize=True,
            disposal=2,
        )
        if path.stat().st_size <= 3 * 1024 * 1024:
            break
    with Image.open(path) as gif:
        metadata = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "width": gif.width,
            "height": gif.height,
            "frames": gif.n_frames,
            "source_frame_indices": indices,
            "sha256": sha256_file(path),
        }
    if metadata["width"] > 256 or metadata["frames"] != 16:
        raise AssertionError(f"invalid optimized GIF: {metadata}")
    if metadata["bytes"] > 3 * 1024 * 1024:
        raise AssertionError(f"GIF exceeds 3 MiB: {metadata}")
    return metadata


def save_aligned_pngs(
    output_dir: Path, eager: np.ndarray, coda: np.ndarray
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"eager": [], "coda": []}
    for variant, frames in (("eager", eager), ("coda", coda)):
        folder = output_dir / "artifacts/aligned_frames" / variant
        folder.mkdir(parents=True, exist_ok=True)
        for index in FRAME_INDICES:
            path = folder / f"frame_{index:03d}.png"
            Image.fromarray(frames[index], mode="RGB").save(path, optimize=True)
            result[variant].append(relative(path, output_dir))
    return result


def video_similarity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"decoded video shapes differ: {reference.shape} vs {candidate.shape}"
        )
    squared_error_sum = 0.0
    element_count = int(reference.size)
    frame_psnr: list[float | None] = []
    frame_ssim: list[float] = []
    for ref_frame, got_frame in zip(reference, candidate):
        difference = ref_frame.astype(np.float32) - got_frame.astype(np.float32)
        frame_mse_u8 = float(np.mean(difference * difference))
        squared_error_sum += float(np.sum(difference * difference, dtype=np.float64))
        if frame_mse_u8 == 0.0:
            frame_psnr.append(None)
        else:
            frame_psnr.append(10.0 * math.log10((255.0**2) / frame_mse_u8))
        frame_ssim.append(
            float(
                structural_similarity(
                    ref_frame,
                    got_frame,
                    data_range=255,
                    channel_axis=-1,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
            )
        )
    global_mse_u8 = squared_error_sum / element_count
    global_infinite = global_mse_u8 == 0.0
    global_psnr = (
        None
        if global_infinite
        else 10.0 * math.log10((255.0**2) / global_mse_u8)
    )
    finite_psnr = [item for item in frame_psnr if item is not None]
    return {
        "reference": "eager",
        "candidate": "coda",
        "domain": (
            "all aligned VAE-decoded RGB uint8 frames, clipped to [-1,1] and "
            "mapped to [0,255], before MP4 encoding"
        ),
        "shape_t_h_w_c": list(reference.shape),
        "data_range": 255,
        "global_mse_u8": global_mse_u8,
        "global_psnr_db": global_psnr,
        "psnr_is_infinite": global_infinite,
        "frame_psnr_mean_db": (
            statistics.mean(finite_psnr) if finite_psnr else None
        ),
        "frame_psnr_min_db": min(finite_psnr) if finite_psnr else None,
        "infinite_psnr_frame_count": len(frame_psnr) - len(finite_psnr),
        "frame_ssim_mean": statistics.mean(frame_ssim),
        "frame_ssim_min": min(frame_ssim),
        "ssim_method": (
            "skimage Gaussian-weighted SSIM, sigma=1.5, channel_axis RGB, "
            "data_range=255, population covariance"
        ),
    }


def generation_metadata(args: argparse.Namespace, negative_prompt: str) -> dict[str, Any]:
    latent_shape = (
        16,
        (args.frames - 1) // 4 + 1,
        args.height // 8,
        args.width // 8,
    )
    grid = (latent_shape[1], latent_shape[2] // 2, latent_shape[3] // 2)
    return {
        "task": "t2v-1.3B",
        "prompt": args.prompt,
        "negative_prompt": negative_prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "sampling_steps": args.steps,
        "solver": "unipc",
        "shift": args.shift,
        "cfg_scale": args.guide_scale,
        "cfg_execution": "sequential cond then uncond, batch=1 each",
        "latent_shape": list(latent_shape),
        "dit_grid_fhw": list(grid),
        "dit_seq_len": int(np.prod(grid)),
    }


def write_suite_results(output_dir: Path) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in ("eager", "compile", "coda"):
        path = output_dir / f"{variant}_result.json"
        if path.is_file():
            variants[variant] = json.loads(path.read_text())

    medians = {
        variant: payload["aggregate"]["median_dit_only_total_ms"]
        for variant, payload in variants.items()
    }
    for variant, payload in variants.items():
        current = medians[variant]
        payload["aggregate"]["speedup_vs_eager"] = (
            medians["eager"] / current if "eager" in medians else None
        )
        payload["aggregate"]["speedup_vs_compile"] = (
            medians["compile"] / current if "compile" in medians else None
        )

    similarity = None
    aligned_pngs = None
    raw_paths = {
        variant: output_dir / "raw" / f"{variant}_vae_decoded_rgb_u8.npy"
        for variant in ("eager", "coda")
    }
    if {"eager", "coda"}.issubset(variants) and all(
        path.is_file() for path in raw_paths.values()
    ):
        eager = np.load(raw_paths["eager"], mmap_mode="r")
        coda = np.load(raw_paths["coda"], mmap_mode="r")
        similarity = video_similarity(eager, coda)
        similarity["source_runs"] = {
            "eager": variants["eager"]["aggregate"]["representative_run_index"],
            "coda": variants["coda"]["aggregate"]["representative_run_index"],
        }
        aligned_pngs = save_aligned_pngs(output_dir, eager, coda)

        # This is a transparent delivery regression gate, not a losslessness claim.
        quality_pass = (
            (
                similarity["psnr_is_infinite"]
                or similarity["global_psnr_db"] >= 20.0
            )
            and similarity["frame_ssim_mean"] >= 0.90
        )
        similarity["quality_gate"] = {
            "global_psnr_db_min": 20.0,
            "frame_ssim_mean_min": 0.90,
            "pass": quality_pass,
            "interpretation": (
                "operational decoded-video regression gate; passing indicates high "
                "structural similarity, not pixel identity or losslessness"
            ),
        }

    common = next(iter(variants.values())) if variants else None
    environment_rows = {
        variant: payload["environment"] for variant, payload in variants.items()
    }
    noise_hashes = {
        row["initial_noise_sha256"]
        for payload in variants.values()
        for row in payload["runs"]
    }

    def all_equal(values: list[Any]) -> bool:
        return bool(values) and all(value == values[0] for value in values[1:])

    def json_signature(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    def source_sha_signature(payload: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (key, value["sha256"])
                for key, value in payload["provenance"]["sources"].items()
            )
        )

    def environment_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
        environment = payload["environment"]
        return tuple(
            environment.get(key)
            for key in (
                "gpu_name",
                "gpu_uuid",
                "driver",
                "torch",
                "cuda",
                "flash_attn",
                "attention_backend",
                "dit_dtype",
            )
        )

    def artifact_valid(metadata: Any) -> bool:
        if not isinstance(metadata, dict) or not metadata.get("path"):
            return False
        path = output_dir / metadata["path"]
        if not path.is_file():
            return False
        if metadata.get("bytes") is not None and path.stat().st_size != metadata["bytes"]:
            return False
        if metadata.get("sha256") is not None and sha256_file(path) != metadata["sha256"]:
            return False
        return True

    expected_variants = {"eager", "compile", "coda"}
    exact_variant_set = set(variants) == expected_variants
    generation_consistent = all_equal(
        [json_signature(payload["generation"]) for payload in variants.values()]
    )
    checkpoint_consistent = all_equal(
        [
            json_signature(payload["provenance"]["checkpoint"])
            for payload in variants.values()
        ]
    )
    environment_consistent = all_equal(
        [environment_signature(payload) for payload in variants.values()]
    )
    sources_consistent = all_equal(
        [source_sha_signature(payload) for payload in variants.values()]
    )
    protocols_valid = exact_variant_set and all(
        payload["timing_protocol"].get("protocol_valid") is True
        for payload in variants.values()
    )
    runs_valid = exact_variant_set and all(
        3 <= len(payload["runs"]) <= 5
        and len(payload["runs"])
        == payload["timing_protocol"]["measured_full_generations"]
        and all(
            row["dit_forward_calls"]
            == payload["timing_protocol"]["expected_dit_forward_calls_per_generation"]
            and len(row["dit_calls"])
            == payload["timing_protocol"]["expected_dit_forward_calls_per_generation"]
            for row in payload["runs"]
        )
        for payload in variants.values()
    )
    videos_valid = exact_variant_set and all(
        artifact_valid(variants[variant]["artifacts"].get("mp4"))
        for variant in expected_variants
    )
    gifs_valid = {"eager", "coda"}.issubset(variants) and all(
        artifact_valid(variants[variant]["artifacts"].get("gif"))
        and variants[variant]["artifacts"]["gif"].get("width", 257) <= 256
        and variants[variant]["artifacts"]["gif"].get("frames") == 16
        and variants[variant]["artifacts"]["gif"].get("bytes", 3 * 1024 * 1024 + 1)
        <= 3 * 1024 * 1024
        for variant in ("eager", "coda")
    )
    aligned_pngs_valid = aligned_pngs is not None and all(
        len(paths) == len(FRAME_INDICES)
        and all((output_dir / path).is_file() for path in paths)
        for paths in aligned_pngs.values()
    )
    quality_gate_pass = (
        similarity is not None
        and similarity.get("quality_gate", {}).get("pass") is True
    )
    validation = {
        "exact_variant_set": exact_variant_set,
        "protocols_valid": protocols_valid,
        "runs_valid": runs_valid,
        "same_generation_config": generation_consistent,
        "same_checkpoint": checkpoint_consistent,
        "same_gpu_and_software": environment_consistent,
        "same_source_shas": sources_consistent,
        "same_prompt": generation_consistent,
        "same_negative_prompt": generation_consistent,
        "same_seed": generation_consistent,
        "same_initial_noise": len(noise_hashes) == 1,
        "same_scheduler": generation_consistent,
        "required_videos_present_and_hashed": videos_valid,
        "required_gifs_present_and_hashed": gifs_valid,
        "aligned_pngs_present": aligned_pngs_valid,
        "decoded_video_quality_gate_pass": quality_gate_pass,
        "fa2_forced": True,
        "dit_bf16": True,
        "official_vae_decoder_float32": True,
        "no_training": True,
        "no_fp8": True,
    }
    complete = all(validation.values())
    suite = {
        "schema_version": "1.0",
        "complete": complete,
        "metric_label": TIMING_LABEL,
        "provenance": {
            "server_authoritative": all(
                payload["provenance"].get("server_authoritative") is True
                for payload in variants.values()
            ),
            "wan_pipeline": "Wan2.1/wan/text2video.py",
            "coda_implementation": "CodaFullBlock source and SHA recorded per variant",
            "commands": {
                variant: payload["provenance"]["command"]
                for variant, payload in variants.items()
            },
            "sources": {
                variant: payload["provenance"]["sources"]
                for variant, payload in variants.items()
            },
            "aggregation_sources": source_audit(),
            "checkpoint": common["provenance"]["checkpoint"] if common else None,
        },
        "environment": environment_rows,
        "generation": common["generation"] if common else None,
        "timing_protocol": (
            {
                **common["timing_protocol"],
                "memory_scope": (
                    "denoising-phase peak allocation; includes live scheduler/CFG "
                    "temporaries, while the latency metric remains DiT-only"
                ),
            }
            if common
            else None
        ),
        "variants": variants,
        "similarity": similarity,
        "artifacts": {
            "videos": {
                variant: payload["artifacts"].get("mp4")
                for variant, payload in variants.items()
            },
            "gifs": {
                variant: payload["artifacts"].get("gif")
                for variant, payload in variants.items()
                if payload["artifacts"].get("gif") is not None
            },
            "aligned_png_frame_indices": list(FRAME_INDICES),
            "aligned_pngs": aligned_pngs,
        },
        "validation": validation,
    }
    json_write(output_dir / "dit_only_results.json", suite)
    return suite


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.trials < 1:
        raise ValueError("warmup must be >=0 and trials must be >=1")
    if args.frames % 4 != 1:
        raise ValueError("frame count must be 4n+1")
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    force_fa2()

    checkpoint = checkpoint_audit(args.ckpt_dir)
    environment = environment_metadata(args.device)
    sources = source_audit()
    print(
        f"[setup] variant={args.variant} gpu={environment['gpu_name']} "
        f"checkpoint={args.ckpt_dir}",
        flush=True,
    )

    config = WAN_CONFIGS["t2v-1.3B"]
    pipeline = WanT2V(
        config=config,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    model, build = install_variant(pipeline, args.variant, device)
    negative_prompt = (
        pipeline.sample_neg_prompt
        if args.negative_prompt is None
        else args.negative_prompt
    )

    for warmup_index in range(args.warmup):
        print(
            f"[warmup] variant={args.variant} full_generation={warmup_index + 1}/"
            f"{args.warmup}",
            flush=True,
        )
        generate_once(
            pipeline, model, args, measured=False, keep_frames=False
        )

    rows: list[dict[str, Any]] = []
    decoded_videos: list[np.ndarray] = []
    for trial_index in range(args.trials):
        print(
            f"[measure] variant={args.variant} full_generation={trial_index + 1}/"
            f"{args.trials}",
            flush=True,
        )
        row, frames = generate_once(
            pipeline, model, args, measured=True, keep_frames=True
        )
        row["run_index"] = trial_index
        rows.append(row)
        assert frames is not None
        decoded_videos.append(frames)
        print(
            f"[measure] run={trial_index} dit_only_total_ms="
            f"{row['dit_only_total_ms']:.3f} peak_gib="
            f"{gib(row['dit_phase_peak_allocated_bytes']):.3f}",
            flush=True,
        )

    totals = [row["dit_only_total_ms"] for row in rows]
    median_total = float(statistics.median(totals))
    representative_index = min(
        range(len(rows)), key=lambda index: abs(totals[index] - median_total)
    )
    representative_frames = decoded_videos[representative_index]

    raw_path = args.output_dir / "raw" / f"{args.variant}_vae_decoded_rgb_u8.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, representative_frames, allow_pickle=False)

    artifacts: dict[str, Any] = {
        "representative_run_index": representative_index,
        "decoded_rgb_u8_npy": relative(raw_path, args.output_dir),
        "decoded_rgb_u8_sha256": sha256_file(raw_path),
        "mp4": None,
        "gif": None,
    }
    if not args.no_media:
        mp4_path = args.output_dir / "artifacts/videos" / f"{args.variant}.mp4"
        save_mp4(representative_frames, mp4_path, args.fps)
        artifacts["mp4"] = {
            "path": relative(mp4_path, args.output_dir),
            "bytes": mp4_path.stat().st_size,
            "sha256": sha256_file(mp4_path),
            "fps": args.fps,
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
        }
        if args.variant in {"eager", "coda"}:
            gif_path = args.output_dir / "artifacts/gifs" / f"{args.variant}.gif"
            gif_metadata = save_gif(representative_frames, gif_path)
            gif_metadata["path"] = relative(gif_path, args.output_dir)
            artifacts["gif"] = gif_metadata

    aggregate = {
        "median_dit_only_total_ms": median_total,
        "median_dit_only_avg_step_ms": median_total / args.steps,
        "median_dit_only_avg_forward_ms": median_total / (args.steps * 2),
        "speedup_vs_eager": 1.0 if args.variant == "eager" else None,
        "speedup_vs_compile": 1.0 if args.variant == "compile" else None,
        "median_dit_phase_peak_allocated_gib": statistics.median(
            gib(row["dit_phase_peak_allocated_bytes"]) for row in rows
        ),
        "max_dit_phase_peak_allocated_gib": max(
            gib(row["dit_phase_peak_allocated_bytes"]) for row in rows
        ),
        "median_full_pipeline_peak_allocated_gib": statistics.median(
            gib(row["full_pipeline_peak_allocated_bytes"]) for row in rows
        ),
        "representative_run_index": representative_index,
    }
    protocol_valid = (
        args.width == 832
        and args.height == 480
        and args.frames == 81
        and args.steps == 50
        and args.warmup >= 1
        and 3 <= args.trials <= 5
        and args.guide_scale == 5.0
        and args.shift == 5.0
    )
    payload = {
        "schema_version": "1.0",
        "metric_label": TIMING_LABEL,
        "variant": args.variant,
        "provenance": {
            "server_authoritative": CODA_IMPORT_BACKEND == "server_v3",
            "coda_import_backend": CODA_IMPORT_BACKEND,
            "command": " ".join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()),
            "checkpoint": checkpoint,
            "sources": sources,
            "wan_git_commit": git_commit_for_source(
                Path(sys.modules[WanT2V.__module__].__file__).resolve()
            ),
        },
        "environment": environment,
        "generation": generation_metadata(args, negative_prompt),
        "timing_protocol": {
            "label": TIMING_LABEL,
            "clock": "CUDA events on current CUDA stream",
            "scope": (
                "complete WanModel.forward; CFG combine, scheduler, T5, VAE, "
                "model movement and export excluded"
            ),
            "warmup_full_generations": args.warmup,
            "measured_full_generations": args.trials,
            "expected_dit_forward_calls_per_generation": args.steps * 2,
            "aggregation": "median of per-generation summed DiT event time",
            "memory_scope": (
                "denoising-phase peak allocation; includes live scheduler/CFG "
                "temporaries, while the latency metric remains DiT-only"
            ),
            "protocol_valid": protocol_valid,
        },
        "build": build,
        "runs": rows,
        "aggregate": aggregate,
        "artifacts": artifacts,
    }
    json_write(args.output_dir / f"{args.variant}_result.json", payload)
    suite = write_suite_results(args.output_dir)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "median_dit_only_total_ms": median_total,
                "median_dit_only_avg_step_ms": median_total / args.steps,
                "representative_run_index": representative_index,
                "suite_complete": suite["complete"],
                "result": str(args.output_dir / f"{args.variant}_result.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
