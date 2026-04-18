from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.copy()
    return ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")


def sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=float,
    )
    kernel_y = kernel_x.T
    return kernel_x, kernel_y


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return signal.convolve2d(image, kernel, mode="same", boundary="symm")


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel_x, kernel_y = sobel_kernels()
    grad_x = convolve2d(image, kernel_x)
    grad_y = convolve2d(image, kernel_y)
    return grad_x, grad_y
