"""Patch-based Pavia University experiment for LST-Mamba.

Expected dataset files under ``./datasets``:
    datasets/PaviaU.mat       usually key: paviaU
    datasets/PaviaU_gt.mat    usually key: paviaU_gt

Training is patch-based.  Full-scene inference is performed after testing by
batched sliding-window prediction, so the JSON report contains both patch-batch
latency and true full-image inference time.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from sklearn import preprocessing

from mbmamba_source.utils.false_color import generate_hsi_pseudocolor
from utils import random_sample, run_patch_experiment


def set_seed(seed: int = 1) -> None:
    """Set all random seeds used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mat_array(mat_path: Path, preferred_keys: Sequence[str], ndim: int) -> np.ndarray:
    """Load a MATLAB array with robust key handling."""
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {mat_path}\n"
            f"Please put PaviaU.mat and PaviaU_gt.mat under {mat_path.parent}."
        )
    mat_dict = loadmat(mat_path)
    for key in preferred_keys:
        value = mat_dict.get(key)
        if isinstance(value, np.ndarray) and value.ndim == ndim:
            return value
    for key, value in mat_dict.items():
        if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == ndim:
            print(f"[Warning] Use array key '{key}' from {mat_path.name}; expected one of {preferred_keys}.")
            return value
    raise KeyError(f"No {ndim}D array found in {mat_path}.")


def load_paviau(folder: str = "./datasets") -> Tuple[np.ndarray, np.ndarray]:
    """Load Pavia University image and ground truth."""
    folder_path = Path(folder)
    img = _load_mat_array(folder_path / "PaviaU.mat", preferred_keys=("paviaU", "PaviaU"), ndim=3)
    gt = _load_mat_array(folder_path / "PaviaU_gt.mat", preferred_keys=("paviaU_gt", "PaviaU_gt"), ndim=2)
    img = img.astype(np.float32)
    gt = gt.astype(np.int64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    pseudo_rgb = generate_hsi_pseudocolor(
        img,
        "paviaU",
        rgb_bands=(60, 30, 10)
    )

    return img, gt


def standardize_hsi(img: np.ndarray) -> np.ndarray:
    """Band-wise standardization over all pixels."""
    h, w, c = img.shape
    flat = preprocessing.scale(img.reshape(h * w, c))
    return flat.reshape(h, w, c).astype(np.float32)


def build_ratio_split(gt: np.ndarray, train_ratio: float = 0.01, val_ratio: float = 0.01, min_per_class: int = 5):
    """Build per-class train/validation sample counts from ratios."""
    num_classes = int(gt.max())
    train_sample = []
    val_sample = []
    for cls in range(1, num_classes + 1):
        n_cls = int(np.sum(gt == cls))
        if n_cls <= 0:
            train_sample.append(0)
            val_sample.append(0)
            continue
        n_train = max(min_per_class, int(round(n_cls * train_ratio)))
        n_val = max(min_per_class, int(round(n_cls * val_ratio)))
        max_train = max(n_cls - 2, 1)
        n_train = min(n_train, max_train)
        n_val = min(n_val, max(n_cls - n_train - 1, 0))
        train_sample.append(n_train)
        val_sample.append(n_val)
    return tuple(train_sample), tuple(val_sample)


def main() -> None:
    set_seed(1)
    os.makedirs("net", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    os.makedirs("classification_maps", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img, gt = load_paviau("./datasets")
    whole_data = standardize_hsi(img)
    h, w, band = whole_data.shape
    num_classes = int(gt.max())
    print(f"HSI shape: {whole_data.shape}, classes: {num_classes}, device: {device}")

    train_sample, val_sample = build_ratio_split(gt, train_ratio=0.01, val_ratio=0.01, min_per_class=5)
    train_indices, val_indices, test_indices = random_sample(train_sample, val_sample, gt)
    print("Train samples per class:", train_sample)
    print("Val samples per class:  ", val_sample)

    run_patch_experiment(
        dataset_name="PaviaU",
        whole_data=whole_data,
        gt=gt,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        device=device,
        patch_size=7,
        batch_size=64,
        test_batch_size=256,
        num_workers=0,
        epochs=300,
        early_num=50,
        lr=1e-3,
        weight_decay=1e-4,
        model_kwargs={
            "dim": 64,
            "num_tokens": 8,
            "rank": 1,
            "mamba_depth": 1,
            "d_state": 8,
            "d_conv": 3,
            "expand": 1,
            "dropout": 0.05,
            "bidirectional": True,
        },
        loss_kwargs={
            "lambda_spatial_div": 0.01,
            "lambda_spectral_div": 0.01,
            "lambda_center": 0.01,
        },
    )


if __name__ == "__main__":
    main()
