from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image2d_project.pipeline import build_project_outputs


def main() -> None:
    result = build_project_outputs()
    lines = [
        "Two-minute demo sequence",
        "1. Start with the clean image and show that Sobel highlights meaningful boundaries.",
        "2. Add Gaussian noise and show how the raw derivative responds to noise as if it were detail.",
        "3. Apply Gaussian smoothing before Sobel and point out the cleaner edge structure.",
        "4. Finish with Canny to show how smoothing plus threshold logic gives a stronger pipeline.",
        f"Primary visual: {result['primary_demo']}",
        "Slides PDF: slides/slides.pdf",
        "Report PDF: docs/project_report.pdf",
    ]
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
