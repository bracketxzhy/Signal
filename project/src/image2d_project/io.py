from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data, img_as_float


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return img_as_float(image)


def load_builtin_images() -> dict[str, np.ndarray]:
    return {
        "camera": _to_gray_float(data.camera()),
        "coins": _to_gray_float(data.coins()),
        "astronaut": _to_gray_float(data.astronaut()),
        "synthetic": generate_synthetic_image(),
    }


def generate_synthetic_image(size: int = 256) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=float)
    canvas[24:104, 24:132] = 0.72
    canvas[144:216, 60:212] = 0.45

    y, x = np.ogrid[:size, :size]
    circle = (x - 176) ** 2 + (y - 76) ** 2 <= 34 ** 2
    ring = (x - 70) ** 2 + (y - 186) ** 2 <= 28 ** 2
    inner = (x - 70) ** 2 + (y - 186) ** 2 <= 15 ** 2
    canvas[circle] = 0.9
    canvas[ring] = 0.95
    canvas[inner] = 0.08

    diagonal = np.abs(y - (0.72 * x + 14)) < 2
    canvas[diagonal] = 1.0
    canvas[:, 198:202] = np.linspace(0.1, 0.95, size)[:, None]
    return np.clip(canvas, 0.0, 1.0)


def save_grayscale_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)
    return path
