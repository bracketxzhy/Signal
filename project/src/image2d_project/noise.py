from __future__ import annotations

import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(loc=0.0, scale=sigma, size=image.shape)
    return np.clip(noisy, 0.0, 1.0)


def add_salt_and_pepper_noise(image: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    total = image.size
    flips = max(1, int(total * amount))
    indices = rng.choice(total, size=flips, replace=False)
    values = rng.integers(0, 2, size=flips)
    flat = noisy.reshape(-1)
    flat[indices] = values
    return noisy
