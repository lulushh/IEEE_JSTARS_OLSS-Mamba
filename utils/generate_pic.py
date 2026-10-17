"""Classification-map generation helpers for patch-based LST-Mamba.

The original project used one fixed repeated RGB table for all datasets, which
makes maps from Houston, Indian Pines, PaviaU and custom datasets visually
inconsistent.  This file now provides dataset-aware color palettes:

    label 0  -> background, always black
    label 1+ -> dataset-specific class colors

To customize a dataset, only edit ``DATASET_PALETTES`` below.  The values are
RGB triples in the 0..255 range and the list index is the class label.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

from .patch_data import predict_full_image, random_sample


# -----------------------------------------------------------------------------
# Dataset-aware palettes
# -----------------------------------------------------------------------------
# Notes:
#   1. Index 0 must be [0, 0, 0] because GT/prediction label 0 is background.
#   2. Class labels in this project are assumed to be 1..num_classes.
#   3. The palettes below are intentionally high-contrast and avoid the old
#      pure red/green/blue repeated style.
#   4. Dioni is a placeholder palette because you said you will handle the data
#      key yourself.  If Dioni has a fixed class definition, just replace the
#      Dioni RGB list with your preferred class colors.
DATASET_PALETTES: Dict[str, List[List[int]]] = {
    # Houston2013: 15 classes.
    "houston2013": [
        [0, 0, 0],  # 0 Background
        [0, 153, 0],  # 1 Healthy grass  ok
        [3, 100, 3],  # 2 Stressed grass  ok
        [78, 95, 72],  # 3 Synthetic grass
        [16, 55, 11],  # 4 Tree   ok
        [131, 72, 42],  # 5 Soil
        [90, 178, 198],  # 6 Water  0 4 118 ok
        [186, 66, 65],  # 7 Residential (White) ok
        [193, 194, 189],  # 8 Commercial  ok
        [119, 102, 86],  # 9 Road   ok
        [26, 24, 26],  # 10 Highway   ok
        [5, 3, 146],  # 11 Railway  ok
        [203, 167, 131],  # 12 Parking Lot 1  ok
        [182, 134, 73],  # 13 Parking Lot 2  ok
        [254, 137, 163],  # 14 Tennis Court   ok
        [185, 222, 82],  # 15 Running Track   ok
    ],
    "houston": [],  # alias filled after the dict definition

    # Indian Pines: 16 classes.
    "indianpines": [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ],
    "indian_pines": [],
    "id1992": [],

    # Pavia University: 9 classes.
    "paviau": [
        [0, 0, 0],
        [192, 192, 192],    # Asphalt
        [0, 255, 0],   # Meadows
        [0, 255, 255],  # Gravel
        [0, 128, 0],  # Trees
        [255, 0, 255],   # Painted metal sheets
        [165, 82, 41],   # Bare Soil
        [128, 0, 128],  # Bitumen
        [255, 0, 0],  # Self-Blocking Bricks
        [255, 255, 0],    # Shadows
    ],
    "pavia_u": [],
    "paviauniversity": [],

    # Dioni: placeholder qualitative palette.  Replace these RGB values with
    # your dataset-specific colors if Dioni has an official/expected map style.
    "dioni": [
        [0, 0, 0],
        [102, 194, 165],
        [252, 141, 98],
        [141, 160, 203],
        [231, 138, 195],
        [166, 216, 84],
        [255, 217, 47],
        [229, 196, 148],
        [179, 179, 179],
        [127, 201, 127],
        [190, 174, 212],
        [253, 192, 134],
        [255, 255, 153],
        [56, 108, 176],
        [240, 2, 127],
        [191, 91, 23],
        [102, 102, 102],
        [27, 158, 119],
        [217, 95, 2],
        [117, 112, 179],
        [231, 41, 138],
    ],
}

# Fill aliases without duplicating the long lists above.
DATASET_PALETTES["houston"] = DATASET_PALETTES["houston2013"]
DATASET_PALETTES["indian_pines"] = DATASET_PALETTES["indianpines"]
DATASET_PALETTES["id1992"] = DATASET_PALETTES["indianpines"]
DATASET_PALETTES["pavia_u"] = DATASET_PALETTES["paviau"]
DATASET_PALETTES["paviauniversity"] = DATASET_PALETTES["paviau"]


# Fallback palette used when the dataset name is unknown or a dataset has more
# classes than its predefined table.  It is copied from a qualitative palette
# family and kept here explicitly so the code does not depend on matplotlib
# colormap internals.
FALLBACK_EXTRA_COLORS: List[List[int]] = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
    [174, 199, 232],
    [255, 187, 120],
    [152, 223, 138],
    [255, 152, 150],
    [197, 176, 213],
    [196, 156, 148],
    [247, 182, 210],
    [199, 199, 199],
    [219, 219, 141],
    [158, 218, 229],
]


def _normalize_dataset_name(dataset_name: Optional[str]) -> str:
    """Normalize dataset names used by main scripts and output flags."""
    if dataset_name is None:
        return "default"
    return str(dataset_name).lower().replace("-", "_").replace(" ", "_")


def get_dataset_palette(dataset_name: Optional[str], num_classes: int) -> np.ndarray:
    """Return an RGB palette whose row index equals the class label.

    Args:
        dataset_name: Dataset name such as ``Houston2013``, ``IndianPines``,
            ``PaviaU`` or ``Dioni``.  Unknown names use the fallback palette.
        num_classes: Maximum foreground class label.  The returned palette has
            at least ``num_classes + 1`` rows, including row 0 for background.

    Returns:
        ``np.ndarray`` with shape ``[num_classes + 1, 3]`` and dtype uint8.
    """
    key = _normalize_dataset_name(dataset_name)
    base = DATASET_PALETTES.get(key)
    if base is None:
        base = [[0, 0, 0]]

    # Copy to avoid mutating the global palette when we extend it.
    palette: List[List[int]] = [list(c) for c in base]
    if len(palette) == 0 or palette[0] != [0, 0, 0]:
        palette.insert(0, [0, 0, 0])

    needed = int(num_classes) + 1
    extra_idx = 0
    while len(palette) < needed:
        palette.append(FALLBACK_EXTRA_COLORS[extra_idx % len(FALLBACK_EXTRA_COLORS)])
        extra_idx += 1

    return np.asarray(palette[:needed], dtype=np.uint8)


def labels_to_rgb(labels: np.ndarray, dataset_name: Optional[str] = None, num_classes: Optional[int] = None) -> np.ndarray:
    """Convert an integer label map to an RGB image using dataset colors.

    Args:
        labels: Integer map with shape ``[H, W]``.  0 is background.
        dataset_name: Dataset name used to choose the palette.
        num_classes: Optional number of classes.  If omitted, ``labels.max()``
            is used.

    Returns:
        RGB uint8 image with shape ``[H, W, 3]``.
    """
    labels = labels.astype(np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) if labels.size else 0
    palette = get_dataset_palette(dataset_name, num_classes)

    # Unknown/overflow labels are wrapped only after reserving row 0 as black.
    # In normal experiments labels should already be within 0..num_classes.
    safe_labels = np.clip(labels, 0, len(palette) - 1)
    return palette[safe_labels]


def _classification_map(rgb: np.ndarray, save_path: str, dpi: int = 300) -> None:
    """Save an RGB classification map without axes or padding."""
    h, w = rgb.shape[:2]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w * 2.0 / dpi, h * 2.0 / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgb)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_colorized_label_map(
    labels: np.ndarray,
    save_path: str,
    dataset_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    dpi: int = 300,
) -> None:
    """Save a label map as a dataset-colored PNG image."""
    rgb = labels_to_rgb(labels, dataset_name=dataset_name, num_classes=num_classes)
    _classification_map(rgb, save_path, dpi=dpi)


def save_palette_file(save_path: str, dataset_name: Optional[str], num_classes: int) -> None:
    """Save the RGB table used in the current experiment for reproducibility."""
    palette = get_dataset_palette(dataset_name, num_classes)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset_name}\n")
        f.write("label,r,g,b\n")
        for label, (r, g, b) in enumerate(palette.tolist()):
            f.write(f"{label},{r},{g},{b}\n")


@torch.no_grad()
def generate_png(
    net,
    whole_data: np.ndarray,
    device,
    patch_size: int,
    batch_size: int,
    flag: str = "lst_mamba_patch",
    num_workers: int = 0,
    dataset_name: Optional[str] = None,
    gt: Optional[np.ndarray] = None,
):
    """Generate a full-scene map using patch-based sliding-window inference.

    Args:
        net: Trained patch classifier.
        whole_data: HSI cube with shape ``[H, W, C]``.
        device: PyTorch device.
        patch_size: Patch size used during training and inference.
        batch_size: Patch batch size for sliding-window prediction.
        flag: File-name suffix.
        num_workers: DataLoader workers for full-image prediction.
        dataset_name: Dataset name used to choose visualization colors.
        gt: Optional ground-truth map.  If provided, a colorized GT PNG is saved
            with the same palette as the prediction.

    Returns:
        ``pred_map, report`` where ``report`` is returned by
        ``predict_full_image`` and contains full-image inference time.
    """
    os.makedirs("classification_maps", exist_ok=True)
    pred_map, report = predict_full_image(
        net,
        whole_data,
        device,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Prefer explicit dataset_name.  If not provided, use the file flag so old
    # calls still receive dataset-aware colors whenever the flag contains the
    # dataset name, e.g. "paviau_lst_mamba_patch".
    palette_name = dataset_name or flag.split("_lst_mamba")[0]
    num_classes = int(max(pred_map.max(), gt.max() if gt is not None else 0))

    day_str = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    tif_path = f"classification_maps/{day_str}_{flag}.tif"
    png_path = f"classification_maps/{day_str}_{flag}.png"
    palette_path = f"classification_maps/{day_str}_{flag}_palette.csv"

    # Save raw integer prediction map for quantitative reuse.
    PIL.Image.fromarray(pred_map.astype(np.uint8)).save(tif_path)

    # Save colorized prediction map with dataset-specific palette.
    save_colorized_label_map(pred_map, png_path, dataset_name=palette_name, num_classes=num_classes)
    save_palette_file(palette_path, dataset_name=palette_name, num_classes=num_classes)

    if gt is not None:
        gt_png_path = f"classification_maps/{day_str}_{flag}_gt.png"
        save_colorized_label_map(gt, gt_png_path, dataset_name=palette_name, num_classes=num_classes)
        print(f"------Saved colorized GT map: {gt_png_path}-------")

    print(f"------Get classification maps successful: {png_path}-------")
    print(f"------Palette file saved: {palette_path}-------")
    return pred_map, report


