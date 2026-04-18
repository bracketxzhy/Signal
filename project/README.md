# Edge Detection in Noisy Images Using 2D Filtering

This project studies a narrow signals-and-systems question: **why naive edge detection fails in noisy images, and how 2D filtering improves the result**. The core ideas are 2D convolution, LTI filtering, derivative operators, and the tradeoff between noise suppression and edge preservation. The formal deliverables are written in LaTeX: one report PDF and one Beamer slide PDF.

## Why this topic fits the course

- It uses 2D filtering as the central system concept.
- It shows how derivative-based operators emphasize high-frequency content.
- It demonstrates that noise and true edges compete in the same frequency region.
- It includes a short demo that can be explained in under two minutes.

## Project structure

- `src/image2d_project/`: reusable Python modules
- `scripts/build_project.py`: generate all figures and data artifacts
- `scripts/build_project.py`: generate all figures and compile both TeX PDFs
- `scripts/run_demo.py`: print the presentation-order demo sequence
- `scripts/self_check.py`: verify project completeness and compliance
- `assets/generated/`: generated figures and metrics
- `docs/project_report.tex`: formal project report source
- `docs/project_report.pdf`: compiled report
- `slides/slides.tex`: Beamer slide source
- `slides/slides.pdf`: compiled slide deck

## Requirements

This project uses Python 3.12 with these libraries:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `scikit-image`

## How to run

From the `project/` directory:

```bash
python scripts/build_project.py
python scripts/run_demo.py
python scripts/self_check.py
```

## Generated outputs

The build script creates these main artifacts:

- `assets/generated/figures/image_gallery.png`
- `assets/generated/figures/camera_pipeline.png`
- `assets/generated/figures/coins_pipeline.png`
- `assets/generated/figures/astronaut_pipeline.png`
- `assets/generated/figures/synthetic_pipeline.png`
- `assets/generated/figures/sobel_kernels.png`
- `assets/generated/figures/tradeoff_heatmap.png`
- `assets/generated/data/tradeoff_metrics.csv`
- `assets/generated/data/artifact_manifest.json`
- `docs/project_report.pdf`
- `slides/slides.pdf`

## Two-minute demo

The demo sequence is intentionally short:

1. Show the clean image and its Sobel response.
2. Add Gaussian noise and show how naive edge detection breaks down.
3. Apply Gaussian smoothing before Sobel and show the recovery.
4. Compare with Canny as a stronger edge-detection pipeline.

## Design choices

- Sample images come from `skimage.data` plus one synthetic image generated in code.
- The Sobel operator is implemented with explicit 2D convolution kernels.
- Canny is included as a comparison, not the main theory.
- CNN and diffusion are mentioned only as a modern connection in the slides.
- The report and slides are authored in TeX rather than Markdown or HTML.

## References

See `docs/project_report.tex` and `slides/slides.tex` for the cited project sources.

## Team responsibilities

Fill in the member names in `docs/project_report.tex` and `slides/slides.tex` before submission.
