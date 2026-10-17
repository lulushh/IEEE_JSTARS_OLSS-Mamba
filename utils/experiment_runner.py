"""Shared experiment runner for patch-based LST-Mamba main scripts."""

from __future__ import annotations

import datetime
import os
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import optim

from model import build_lst_mamba

from . import metricss, record
from .generate_pic import generate_png
from .losses import LSTTokenLoss
from .patch_data import generate_patch_iter, predict_loader
from .profiling import (
    append_text_report,
    estimate_lst_mamba_complexity,
    measure_inference_time,
    print_complexity_report,
    save_json_report,
    sync_device,
)
from .train import train


def model_structure(model: torch.nn.Module) -> None:
    """Print parameter names, shapes and counts."""
    print("-" * 90)
    print(f"{'weight name':45s} {'weight shape':30s} {'number':>12s}")
    print("-" * 90)
    num_para = 0
    for name, param in model.named_parameters():
        count = int(np.prod(param.shape))
        num_para += count
        print(f"{name:45s} {str(tuple(param.shape)):30s} {count:12d}")
    print("-" * 90)
    print(f"Total parameters: {num_para} ({num_para / 1e6:.4f} M)")
    print("-" * 90)


@torch.no_grad()
def evaluate_on_loader(net, test_iter, device, num_classes: int):
    """Return OA, AA, Kappa and class-wise accuracy on patch test samples."""
    pred, true, _coords = predict_loader(net, test_iter, device)
    oa = metricss.th_overall_accuracy_score(true, pred)
    aa, acc_per_class = metricss.th_average_accuracy_score(true, pred, num_classes, return_accuracys=True)
    kappa = metricss.th_cohen_kappa_score(true, pred, num_classes)
    return oa, aa, kappa, acc_per_class


def run_patch_experiment(
    dataset_name: str,
    whole_data: np.ndarray,
    gt: np.ndarray,
    train_indices: Sequence[Tuple[int, int]],
    val_indices: Sequence[Tuple[int, int]],
    test_indices: Sequence[Tuple[int, int]],
    device: torch.device,
    patch_size: int = 11,
    batch_size: int = 64,
    test_batch_size: int = 256,
    num_workers: int = 0,
    epochs: int = 300,
    early_num: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    model_kwargs: Dict | None = None,
    loss_kwargs: Dict | None = None,
):
    """Train, evaluate, profile and save a patch-based experiment.

    The function centralizes the workflow used by Houston2013, Indian Pines and
    PaviaU so that dataset-specific main files only need to load data and define
    the split.
    """
    os.makedirs("net", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    os.makedirs("classification_maps", exist_ok=True)

    h, w, band = whole_data.shape
    num_classes = int(gt.max())
    model_kwargs = dict(model_kwargs or {})
    loss_kwargs = dict(loss_kwargs or {})

    train_iter, val_iter, test_iter = generate_patch_iter(
        train_indices=train_indices,
        test_indices=test_indices,
        val_indices=val_indices,
        whole_data=whole_data,
        gt=gt,
        patch_size=patch_size,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )

    net = build_lst_mamba(
        in_bands=band,
        num_classes=num_classes,
        patch_size=patch_size,
        **model_kwargs,
    )
    model_structure(net)

    complexity_report = estimate_lst_mamba_complexity(
        net,
        input_shape=(1, patch_size, patch_size, band),
        full_image_shape=(h, w),
    )
    print_complexity_report(complexity_report)

    loss_fn = LSTTokenLoss(**loss_kwargs)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_history = train(
        net,
        train_iter,
        val_iter,
        loss_fn,
        optimizer,
        device,
        epochs=epochs,
        early_stopping=True,
        early_num=early_num,
    )

    sync_device(device)
    tic = time.perf_counter()
    oa, aa, kappa, acc_per_class = evaluate_on_loader(net, test_iter, device, num_classes)
    sync_device(device)
    test_time = time.perf_counter() - tic

    print(f"OA {oa:.6f}, AA {aa:.6f}, Kappa {kappa:.6f}")
    print("Class-wise accuracy:", acc_per_class)

    # Patch-batch latency measures one mini-batch forward.  Full-image inference
    # time is measured below when the classification map is generated.
    patch_inference_report = measure_inference_time(net, test_iter, device, warmup=5, repeat=20)
    print(
        f"Patch-batch inference: {patch_inference_report['patch_batch_inference_time_per_forward_ms']:.3f} ms/batch, "
        f"{patch_inference_report['patches_per_sec']:.1f} patches/s"
    )

    map_flag = f"{dataset_name.lower()}_lst_mamba_patch"
    _pred_map, full_image_report = generate_png(
        net,
        whole_data,
        device,
        patch_size=patch_size,
        batch_size=test_batch_size,
        flag=map_flag,
        num_workers=num_workers,
        dataset_name=dataset_name,
        gt=gt,
    )
    print(
        f"Full-image inference: {full_image_report['full_image_inference_time_sec']:.3f}s, "
        f"{full_image_report['full_image_inference_pixels_per_sec']:.1f} pixels/s"
    )

    torch.save(net.state_dict(), f"./net/lst_mamba_patch_{dataset_name.lower()}_oa_{float(oa):.3f}.pt")
    day_str = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    record_path = f"records/{day_str}_{dataset_name.lower()}_lst_mamba_patch.txt"
    json_path = f"records/{day_str}_{dataset_name.lower()}_lst_mamba_patch_summary.json"

    record.record_output(
        [float(oa)],
        [float(aa)],
        [float(kappa)],
        acc_per_class.view(1, -1).cpu().numpy(),
        [float(train_history["total_time_sec"])],
        [float(test_time)],
        record_path,
    )

    experiment_report = {
        "dataset": dataset_name,
        "protocol": "patch-based training; batched sliding-window full-scene inference",
        "patch_size": int(patch_size),
        "batch_size": int(batch_size),
        "test_batch_size": int(test_batch_size),
        "num_train_pixels": int(len(train_indices)),
        "num_val_pixels": int(len(val_indices)),
        "num_test_pixels": int(len(test_indices)),
        "oa": float(oa),
        "aa": float(aa),
        "kappa": float(kappa),
        "class_wise_accuracy": [float(x) for x in acc_per_class.cpu().view(-1)],
        "test_metric_time_sec": float(test_time),
        "train_total_time_sec": float(train_history["total_time_sec"]),
        "train_avg_epoch_time_sec": float(train_history["avg_epoch_time_sec"]),
        "train_num_epochs_ran": int(train_history["num_epochs_ran"]),
        "train_best_epoch": int(train_history["best_epoch"]),
        "train_best_val_loss": float(train_history["best_val_loss"]),
        **complexity_report,
        **patch_inference_report,
        **full_image_report,
    }
    save_json_report(experiment_report, json_path)
    append_text_report(experiment_report, record_path)
    print(f"Saved experiment summary to {json_path}")
    return experiment_report
