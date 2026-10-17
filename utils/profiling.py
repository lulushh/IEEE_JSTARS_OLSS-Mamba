"""Complexity and runtime profiling for patch-based LST-Mamba.

Reported FLOPs are analytical estimates.  The official Mamba selective scan is
implemented by fused kernels, so generic FLOP hook tools do not count it
reliably.  This file counts the main projections, depthwise causal convolution,
state projections, output projection, FFN layers and an approximate state-scan
term.  FLOPs are reported as 2 * MACs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


def sync_device(device: torch.device) -> None:
    """Synchronize CUDA so wall-clock timing is not affected by async kernels."""
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Return total/trainable/non-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "non_trainable_params": int(total - trainable),
    }


def human_number(value: float) -> str:
    """Compact number formatting used in console logs."""
    value = float(value)
    if abs(value) >= 1e12:
        return f"{value / 1e12:.3f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f}G"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def _get_cfg(model: torch.nn.Module):
    if hasattr(model, "cfg"):
        return model.cfg
    raise TypeError("estimate_lst_mamba_complexity expects a model with a `.cfg` attribute.")


def estimate_lst_mamba_complexity(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    full_image_shape: Tuple[int, int] | None = None,
    convention: str = "2flops_per_mac",
) -> Dict[str, Any]:
    """Analytically estimate Params, MACs and FLOPs.

    Args:
        model: LSTMambaPatchClassifier instance.
        input_shape: one patch input shape [B, P, P, C].
        full_image_shape: optional [H, W].  If provided, the report also includes
            estimated full-scene sliding-window FLOPs = per-patch FLOPs * H * W.
        convention: only ``2flops_per_mac`` is supported.
    """
    if convention != "2flops_per_mac":
        raise ValueError("Only convention='2flops_per_mac' is supported.")

    cfg = _get_cfg(model)
    batch, patch_h, patch_w, bands = [int(v) for v in input_shape]
    if bands != int(cfg.in_bands):
        raise ValueError(f"input bands {bands} != model cfg.in_bands {cfg.in_bands}")

    d = int(cfg.dim)
    k = int(cfg.num_tokens)
    r = int(cfg.rank)
    classes = int(cfg.num_classes)
    pp = patch_h * patch_w

    macs: Dict[str, float] = {}

    # Patch stem.
    macs["stem_pointwise_in"] = batch * pp * bands * d
    macs["stem_depthwise_3x3"] = batch * pp * d * 3 * 3
    macs["stem_pointwise_out"] = batch * pp * d * d

    # Low-rank TokenLearner.
    macs["token_spatial_depthwise_3x3"] = batch * pp * d * 3 * 3
    macs["token_spatial_pointwise"] = batch * pp * d * k * r
    hidden = max(16, d // 2)
    macs["token_spectral_conv1d_k5"] = batch * bands * hidden * 5
    macs["token_spectral_conv1d_1x1"] = batch * bands * hidden * k * r
    macs["token_spatial_pool_einsum"] = batch * k * r * d * pp
    macs["token_spectral_response_einsum"] = batch * k * r * bands * pp
    macs["token_spectral_spatial_reduce"] = batch * k * r * pp
    macs["token_spectral_value_einsum"] = batch * k * r * bands * d
    macs["token_projection"] = batch * k * d * d

    # Tiny Mamba sequence mixer.
    depth = int(cfg.mamba_depth)
    directions = 2 if bool(cfg.bidirectional) else 1
    expand = int(cfg.expand)
    d_inner = d * expand
    d_state = int(cfg.d_state)
    d_conv = int(cfg.d_conv)
    dt_rank = max(1, (d + 15) // 16)  # mamba-ssm default dt_rank='auto'
    token_len = k
    per_direction = 0.0
    per_direction += batch * token_len * d * (2 * d_inner)
    per_direction += batch * token_len * d_inner * d_conv
    per_direction += batch * token_len * d_inner * (dt_rank + 2 * d_state)
    per_direction += batch * token_len * dt_rank * d_inner
    per_direction += batch * token_len * d_inner * d_state * 4
    per_direction += batch * token_len * d_inner * d
    hidden_ffn = int(d * 2.0)
    per_block_ffn = batch * token_len * d * hidden_ffn + batch * token_len * hidden_ffn * d
    macs["tiny_mamba_blocks"] = depth * (directions * per_direction + per_block_ffn)

    # Center-token fusion and classifier.
    macs["center_token_fusion"] = batch * (2 * d) * d + batch * d * d
    macs["classifier_linear"] = batch * d * classes

    total_macs = float(sum(macs.values()))
    total_flops = total_macs * 2.0
    params = count_parameters(model)
    report: Dict[str, Any] = {
        **params,
        "input_shape": list(input_shape),
        "input_meaning": "one patch [B,P,P,C]; FLOPs are per-patch unless full_scene_* fields are shown",
        "macs": int(total_macs),
        "flops": int(total_flops),
        "macs_million": total_macs / 1e6,
        "flops_million": total_flops / 1e6,
        "macs_breakdown": {key: int(value) for key, value in macs.items()},
        "flop_convention": "FLOPs = 2 * MACs; elementwise ops mostly ignored; Mamba scan approximated.",
    }

    if full_image_shape is not None:
        height, width = [int(v) for v in full_image_shape]
        pixels = height * width
        report.update(
            {
                "full_scene_shape": [height, width],
                "full_scene_pixels": int(pixels),
                "full_scene_macs_estimated": int(total_macs * pixels),
                "full_scene_flops_estimated": int(total_flops * pixels),
                "full_scene_gflops_estimated": float(total_flops * pixels / 1e9),
                "full_scene_flops_note": "Estimated as per-patch FLOPs multiplied by H*W for sliding-window full-scene inference.",
            }
        )
    return report


def print_complexity_report(report: Dict[str, Any]) -> None:
    """Print a compact complexity summary."""
    print("-" * 90)
    print("Model complexity summary")
    print("-" * 90)
    print(f"Input shape        : {report['input_shape']}  ([B,P,P,C])")
    print(f"Total params       : {report['total_params']} ({human_number(report['total_params'])})")
    print(f"Trainable params   : {report['trainable_params']} ({human_number(report['trainable_params'])})")
    print(f"MACs / patch       : {report['macs']} ({human_number(report['macs'])})")
    print(f"FLOPs / patch      : {report['flops']} ({human_number(report['flops'])})")
    if "full_scene_gflops_estimated" in report:
        print(f"Full-scene FLOPs   : {report['full_scene_gflops_estimated']:.3f} GFLOPs (sliding-window estimate)")
    print(f"Convention         : {report['flop_convention']}")
    print("-" * 90)


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    data_iter: Iterable,
    device: torch.device,
    warmup: int = 5,
    repeat: int = 20,
) -> Dict[str, float]:
    """Measure forward-only time for one patch batch.

    For full-image inference time, use ``predict_full_image`` / ``generate_png``.
    This function measures the first DataLoader batch and is useful for reporting
    batch latency and patch throughput.
    """
    model.eval()
    batch = next(iter(data_iter))
    patches, _labels, _coords = batch
    patches = patches.to(device, non_blocking=True)

    for _ in range(max(0, warmup)):
        _ = model(patches)
    sync_device(device)

    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        _ = model(patches)
    sync_device(device)
    elapsed = time.perf_counter() - start
    repeat = max(1, int(repeat))
    batch_size = int(patches.shape[0])
    return {
        "patch_batch_inference_warmup": int(warmup),
        "patch_batch_inference_repeat": int(repeat),
        "patch_batch_size": int(batch_size),
        "patch_batch_inference_total_time_sec": float(elapsed),
        "patch_batch_inference_time_per_forward_sec": float(elapsed / repeat),
        "patch_batch_inference_time_per_forward_ms": float(elapsed * 1000.0 / repeat),
        "patches_per_sec": float(batch_size * repeat / max(elapsed, 1e-12)),
        # Backward-compatible names, now explicitly mean one patch batch.
        "inference_time_per_forward_ms": float(elapsed * 1000.0 / repeat),
        "inference_forward_per_sec": float(repeat / max(elapsed, 1e-12)),
    }


def save_json_report(report: Dict[str, Any], path: str | Path) -> None:
    """Save an experiment report as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def append_text_report(report: Dict[str, Any], path: str | Path) -> None:
    """Append complexity and timing information to the txt record file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Complexity and runtime summary\n")
        f.write("=" * 80 + "\n")
        for key in [
            "total_params",
            "trainable_params",
            "non_trainable_params",
            "macs",
            "flops",
            "full_scene_flops_estimated",
            "full_scene_gflops_estimated",
            "train_total_time_sec",
            "train_avg_epoch_time_sec",
            "train_best_epoch",
            "test_metric_time_sec",
            "patch_batch_inference_time_per_forward_ms",
            "patches_per_sec",
            "full_image_inference_time_sec",
            "full_image_inference_time_ms",
            "full_image_inference_pixels_per_sec",
            "full_image_inference_time_per_pixel_ms",
        ]:
            if key in report:
                f.write(f"{key}: {report[key]}\n")
        f.write("\nFLOP convention: FLOPs = 2 * MACs; Mamba scan is analytically approximated.\n")
