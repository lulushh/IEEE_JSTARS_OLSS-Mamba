"""Patch datasets and full-scene inference utilities for HSI classification.

Training is patch-based: every sample is a local ``P x P x C`` cube centered on a
labeled pixel.  Full-scene inference is still supported by batching patches for
all image coordinates and stitching the center-pixel predictions back into an
``H x W`` map.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as Data

from .profiling import sync_device

Coord = Tuple[int, int]


class HSIPatchDataset(Data.Dataset):
    """Dataset that extracts a spatial patch around each requested coordinate.

    Args:
        whole_data: HSI cube with shape [H, W, C], already normalized.
        gt: 2-D label map with labels 1..num_classes and 0 for unlabeled pixels.
        coords: Sequence of (row, col) coordinates.  Labels are read from ``gt``.
        patch_size: Odd spatial window size.  The center pixel is the target.
        pad_mode: Numpy padding mode.  ``reflect`` usually works well for HSI.

    Returns per item:
        patch: float tensor [P, P, C]
        label: long tensor scalar, zero-based class id; -1 if unlabeled
        coord: long tensor [2], original image coordinate
    """

    def __init__(
        self,
        whole_data: np.ndarray,
        gt: np.ndarray,
        coords: Sequence[Coord],
        patch_size: int = 11,
        pad_mode: str = "reflect",
    ) -> None:
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd so the target pixel has a unique center.")
        if whole_data.ndim != 3:
            raise ValueError(f"whole_data must be [H,W,C], got {whole_data.shape}")
        if gt.ndim != 2:
            raise ValueError(f"gt must be [H,W], got {gt.shape}")
        if whole_data.shape[:2] != gt.shape:
            raise ValueError(f"whole_data spatial shape {whole_data.shape[:2]} != gt shape {gt.shape}")

        self.whole_data = whole_data.astype(np.float32, copy=False)
        self.gt = gt.astype(np.int64, copy=False)
        self.coords = [(int(r), int(c)) for r, c in coords]
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.padded = np.pad(
            self.whole_data,
            ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
            mode=pad_mode,
        )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        row, col = self.coords[idx]
        # Because the image is padded by radius pixels, the patch around original
        # (row, col) starts at padded[row, col] and has size P x P.
        patch = self.padded[row : row + self.patch_size, col : col + self.patch_size, :]
        label = int(self.gt[row, col]) - 1
        return (
            torch.from_numpy(np.ascontiguousarray(patch)).float(),
            torch.tensor(label, dtype=torch.long),
            torch.tensor([row, col], dtype=torch.long),
        )


def make_all_coords(height: int, width: int) -> List[Coord]:
    """Return coordinates for every pixel in row-major order."""
    return [(r, c) for r in range(int(height)) for c in range(int(width))]


def coords_from_mask(mask: np.ndarray) -> List[Coord]:
    """Return coordinates where ``mask`` is true/nonzero."""
    rows, cols = np.nonzero(mask)
    return list(zip(rows.tolist(), cols.tolist()))


def generate_patch_iter(
    train_indices: Sequence[Coord],
    test_indices: Sequence[Coord],
    val_indices: Sequence[Coord],
    whole_data: np.ndarray,
    gt: np.ndarray,
    patch_size: int = 11,
    batch_size: int = 64,
    test_batch_size: int | None = None,
    num_workers: int = 0,
):
    """Build train/validation/test DataLoaders for patch classification."""
    test_batch_size = int(test_batch_size or batch_size)
    train_set = HSIPatchDataset(whole_data, gt, train_indices, patch_size=patch_size)
    val_set = HSIPatchDataset(whole_data, gt, val_indices, patch_size=patch_size)
    test_set = HSIPatchDataset(whole_data, gt, test_indices, patch_size=patch_size)

    train_loader = Data.DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = Data.DataLoader(
        val_set,
        batch_size=int(test_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = Data.DataLoader(
        test_set,
        batch_size=int(test_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def random_sample(train_sample: Sequence[int], validate_sample: Sequence[int], labels: np.ndarray):
    """Per-class random coordinate split.

    Args:
        train_sample: number of training pixels per class.
        validate_sample: number of validation pixels per class.
        labels: 2-D ground-truth map, with labels starting from 1.
    """
    num_classes = int(np.max(labels))
    train_indices: List[Coord] = []
    val_indices: List[Coord] = []
    test_indices: List[Coord] = []

    for cls in range(1, num_classes + 1):
        rows, cols = np.where(labels == cls)
        class_coords = list(zip(rows.tolist(), cols.tolist()))
        np.random.shuffle(class_coords)
        n_train = min(int(train_sample[cls - 1]), len(class_coords))
        n_val = min(int(validate_sample[cls - 1]), max(len(class_coords) - n_train, 0))
        train_indices.extend(class_coords[:n_train])
        val_indices.extend(class_coords[n_train : n_train + n_val])
        test_indices.extend(class_coords[n_train + n_val :])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    return train_indices, val_indices, test_indices


@torch.no_grad()
def predict_loader(net, data_iter, device: torch.device):
    """Predict labels for a patch DataLoader.

    Returns:
        pred: 1-D tensor of predicted labels in 1..num_classes.
        true: 1-D tensor of ground-truth labels in 1..num_classes.
        coords: [N,2] tensor of image coordinates.
    """
    net.eval()
    pred_all, true_all, coord_all = [], [], []
    for patches, labels, coords in data_iter:
        patches = patches.to(device, non_blocking=True)
        logits, _ = net(patches)
        pred_all.append(logits.argmax(dim=1).cpu() + 1)
        true_all.append(labels.cpu() + 1)
        coord_all.append(coords.cpu())
    return torch.cat(pred_all), torch.cat(true_all), torch.cat(coord_all, dim=0)


@torch.no_grad()
def predict_full_image(
    net,
    whole_data: np.ndarray,
    device: torch.device,
    patch_size: int = 11,
    batch_size: int = 256,
    num_workers: int = 0,
):
    """Run batched sliding-window inference over the full HSI scene.

    This function performs true full-scene inference for a patch model: it builds
    patches for every pixel, predicts the center label of each patch, and writes
    the result back to an H x W map.

    Returns:
        pred_map: uint8 array [H,W] with labels in 1..num_classes.
        report: timing dictionary with total full-image inference time and
                throughput information.
    """
    h, w, _ = whole_data.shape
    coords = make_all_coords(h, w)
    dummy_gt = np.ones((h, w), dtype=np.int64)  # labels are not used for prediction
    dataset = HSIPatchDataset(whole_data, dummy_gt, coords, patch_size=patch_size)
    loader = Data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    pred_map = np.zeros((h, w), dtype=np.uint8)
    net.eval()
    sync_device(device)
    start = time.perf_counter()
    total_pixels = 0
    for patches, _labels, batch_coords in loader:
        patches = patches.to(device, non_blocking=True)
        logits, _ = net(patches)
        preds = (logits.argmax(dim=1).cpu().numpy() + 1).astype(np.uint8)
        coords_np = batch_coords.numpy()
        pred_map[coords_np[:, 0], coords_np[:, 1]] = preds
        total_pixels += int(preds.shape[0])
    sync_device(device)
    elapsed = time.perf_counter() - start
    report = {
        "full_image_inference_time_sec": float(elapsed),
        "full_image_inference_time_ms": float(elapsed * 1000.0),
        "full_image_pixels": int(total_pixels),
        "full_image_inference_pixels_per_sec": float(total_pixels / max(elapsed, 1e-12)),
        "full_image_inference_time_per_pixel_ms": float(elapsed * 1000.0 / max(total_pixels, 1)),
        "full_image_inference_batch_size": int(batch_size),
    }
    return pred_map, report
