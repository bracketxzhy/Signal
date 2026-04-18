from __future__ import annotations

import json

from image2d_project.analysis import build_tradeoff_heatmap, write_metrics_csv
from image2d_project.edges import canny_edges, sobel_magnitude
from image2d_project.filters import gaussian_blur, sobel_kernels
from image2d_project.io import load_builtin_images
from image2d_project.noise import add_gaussian_noise
from image2d_project.paths import PROJECT_ROOT, ensure_output_dirs
from image2d_project.visualization import (
    save_gallery,
    save_heatmap,
    save_pipeline_panel,
    save_sobel_kernel_panel,
)


def build_project_outputs() -> dict[str, list[str] | str]:
    output_dirs = ensure_output_dirs()
    figures_dir = output_dirs["figures"]
    data_dir = output_dirs["data"]

    images = load_builtin_images()
    gallery_path = save_gallery(figures_dir / "image_gallery.png", images)
    artifact_paths = [gallery_path.relative_to(PROJECT_ROOT).as_posix()]

    demo_noise_sigma = 0.12
    demo_blur_sigma = 1.4

    for name, image in images.items():
        noisy = add_gaussian_noise(image, sigma=demo_noise_sigma, seed=17)
        clean_sobel = sobel_magnitude(image)
        noisy_sobel = sobel_magnitude(noisy)
        smoothed = gaussian_blur(noisy, sigma=demo_blur_sigma)
        smoothed_sobel = sobel_magnitude(smoothed)
        canny_map = canny_edges(noisy, sigma=2.2, low_threshold=0.12, high_threshold=0.28)

        panel_path = save_pipeline_panel(
            figures_dir / f"{name}_pipeline.png",
            f"{name.capitalize()}: edge detection in clean and noisy conditions",
            image,
            clean_sobel,
            noisy,
            noisy_sobel,
            smoothed_sobel,
            canny_map,
        )
        artifact_paths.append(panel_path.relative_to(PROJECT_ROOT).as_posix())

    kernel_x, kernel_y = sobel_kernels()
    kernel_path = save_sobel_kernel_panel(figures_dir / "sobel_kernels.png", kernel_x, kernel_y)
    artifact_paths.append(kernel_path.relative_to(PROJECT_ROOT).as_posix())

    noise_levels = [0.02, 0.05, 0.08, 0.12, 0.18, 0.24]
    smoothing_sigmas = [0.0, 0.6, 1.0, 1.4, 1.8, 2.4]
    heatmap, rows = build_tradeoff_heatmap(images["camera"], noise_levels, smoothing_sigmas)
    heatmap_path = save_heatmap(figures_dir / "tradeoff_heatmap.png", heatmap, noise_levels, smoothing_sigmas)
    metrics_path = write_metrics_csv(data_dir / "tradeoff_metrics.csv", rows)
    artifact_paths.extend(
        [
            heatmap_path.relative_to(PROJECT_ROOT).as_posix(),
            metrics_path.relative_to(PROJECT_ROOT).as_posix(),
        ]
    )

    summary = {
        "topic": "Edge Detection in Noisy Images Using 2D Filtering",
        "primary_demo_image": "camera",
        "artifacts": artifact_paths,
        "noise_levels": noise_levels,
        "smoothing_sigmas": smoothing_sigmas,
    }
    summary_path = data_dir / "artifact_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifact_paths.append(summary_path.relative_to(PROJECT_ROOT).as_posix())

    return {"artifacts": artifact_paths, "primary_demo": "assets/generated/figures/camera_pipeline.png"}
