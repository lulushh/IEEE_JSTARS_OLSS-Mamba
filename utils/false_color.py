import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image


def normalize_to_uint8(img):
    """
    将图像归一化到 0~255，并转成 uint8
    输入可以是 H×W 或 H×W×3
    """
    img = img.astype(np.float32)

    min_val = np.percentile(img, 2)
    max_val = np.percentile(img, 98)

    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val + 1e-8)
    img = img * 255

    return img.astype(np.uint8)


def generate_hsi_pseudocolor(hsi_data,dataset, rgb_bands=(60, 30, 10)):
    """
    根据指定波段生成高光谱伪彩图

    参数：
        hsi_data: H×W×C 高光谱数据
        rgb_bands: 用作 R、G、B 的波段索引

    注意：
        Python 索引从 0 开始。
        rgb_bands=(60, 30, 10) 表示：
        R 通道取第 61 个波段
        G 通道取第 31 个波段
        B 通道取第 11 个波段
    """
    h, w, c = hsi_data.shape

    r_band, g_band, b_band = rgb_bands

    assert r_band < c and g_band < c and b_band < c, \
        f"波段索引超出范围，当前数据只有 {c} 个波段"

    r = hsi_data[:, :, r_band]
    g = hsi_data[:, :, g_band]
    b = hsi_data[:, :, b_band]

    rgb = np.stack([r, g, b], axis=-1)
    rgb = normalize_to_uint8(rgb)

    # pseudo_rgb_path = os.path.join("../false", "PaviaU_pseudo_rgb.png")
    pseudo_rgb_path = os.path.join("././classification_maps", f"{dataset}_pseudo_rgb.png")

    Image.fromarray(rgb).save(pseudo_rgb_path)

    print("伪彩图已保存到:", pseudo_rgb_path)

    return rgb
