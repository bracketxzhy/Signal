from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_pipeline_panel(
    path: Path,
    title: str,
    clean_image: np.ndarray,
    clean_sobel: np.ndarray,
    noisy_image: np.ndarray,
    noisy_sobel: np.ndarray,
    smoothed_sobel: np.ndarray,
    canny_map: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    items = [
        ("Original image", clean_image),
        ("Clean Sobel magnitude", clean_sobel),
        ("Noisy image", noisy_image),
        ("Naive Sobel on noise", noisy_sobel),
        ("Gaussian + Sobel", smoothed_sobel),
        ("Canny edge map", canny_map.astype(float)),
    ]
    for axis, (label, image) in zip(axes.flat, items):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(label, fontsize=11)
        axis.axis("off")
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout()
    return _save(fig, path)


def save_sobel_kernel_panel(path: Path, kernel_x: np.ndarray, kernel_y: np.ndarray) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 6.2))
    for axis, matrix, title in zip(
        axes,
        (kernel_x, kernel_y),
        ("Horizontal-change kernel", "Vertical-change kernel"),
    ):
        image = axis.imshow(matrix, cmap="coolwarm")
        axis.set_title(title, fontsize=11)
        axis.set_xticks(range(3))
        axis.set_yticks(range(3))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axis.text(col, row, f"{matrix[row, col]:.0f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, path)


def save_heatmap(
    path: Path,
    heatmap: np.ndarray,
    noise_levels: list[float],
    smoothing_sigmas: list[float],
) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    image = ax.imshow(heatmap, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Edge quality tradeoff: noise level vs smoothing sigma", fontsize=14, weight="bold")
    ax.set_xlabel("Gaussian smoothing sigma")
    ax.set_ylabel("Noise sigma")
    ax.set_xticks(range(len(smoothing_sigmas)), [f"{value:.1f}" for value in smoothing_sigmas])
    ax.set_yticks(range(len(noise_levels)), [f"{value:.2f}" for value in noise_levels])
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax.text(col, row, f"{heatmap[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="F1 score against clean-edge reference")
    fig.tight_layout()
    return _save(fig, path)


def save_gallery(path: Path, images: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5))
    for axis, (name, image) in zip(axes.flat, images.items()):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name.capitalize(), fontsize=12)
        axis.axis("off")
    fig.suptitle("Project image set", fontsize=16, weight="bold")
    fig.tight_layout()
    return _save(fig, path)
