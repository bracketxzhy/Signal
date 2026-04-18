from __future__ import annotations

import numpy as np
from skimage import feature

from image2d_project.filters import sobel_gradients


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(image, dtype=float)
    return (image - min_value) / (max_value - min_value)


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    grad_x, grad_y = sobel_gradients(image)
    magnitude = np.hypot(grad_x, grad_y)
    return normalize_image(magnitude)


def threshold_edges(magnitude: np.ndarray, quantile: float = 0.88) -> np.ndarray:
    level = float(np.quantile(magnitude, quantile))
    return magnitude >= level


def canny_edges(
    image: np.ndarray,
    sigma: float = 1.6,
    low_threshold: float = 0.08,
    high_threshold: float = 0.2,
) -> np.ndarray:
    return feature.canny(
        image,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
