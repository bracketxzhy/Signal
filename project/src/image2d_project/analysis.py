from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from image2d_project.edges import sobel_magnitude, threshold_edges
from image2d_project.filters import gaussian_blur
from image2d_project.noise import add_gaussian_noise


def f1_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = np.count_nonzero(prediction & target)
    fp = np.count_nonzero(prediction & ~target)
    fn = np.count_nonzero(~prediction & target)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return float((2 * tp) / denominator)


def build_tradeoff_heatmap(
    clean_image: np.ndarray,
    noise_levels: list[float],
    smoothing_sigmas: list[float],
    quantile: float = 0.9,
    seed: int = 23,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    reference_magnitude = sobel_magnitude(clean_image)
    reference_edges = threshold_edges(reference_magnitude, quantile=quantile)

    heatmap = np.zeros((len(noise_levels), len(smoothing_sigmas)), dtype=float)
    rows: list[dict[str, float]] = []

    for row_index, noise_sigma in enumerate(noise_levels):
        noisy = add_gaussian_noise(clean_image, sigma=noise_sigma, seed=seed + row_index)
        for col_index, blur_sigma in enumerate(smoothing_sigmas):
            prepared = gaussian_blur(noisy, sigma=blur_sigma)
            candidate_mag = sobel_magnitude(prepared)
            candidate_edges = threshold_edges(candidate_mag, quantile=quantile)
            score = f1_score(candidate_edges, reference_edges)
            heatmap[row_index, col_index] = score
            rows.append(
                {
                    "noise_sigma": noise_sigma,
                    "blur_sigma": blur_sigma,
                    "f1_score": score,
                }
            )
    return heatmap, rows


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["noise_sigma", "blur_sigma", "f1_score"])
        writer.writeheader()
        writer.writerows(rows)
    return path
