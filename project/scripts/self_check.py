from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image2d_project.paths import GENERATED_DIR


TEXT_SUFFIXES = {".py", ".tex", ".md", ".txt", ".json", ".csv", ".log", ".aux", ".nav", ".out", ".snm", ".toc"}
EXPECTED_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "project_report.tex",
    ROOT / "docs" / "project_report.pdf",
    ROOT / "slides" / "slides.tex",
    ROOT / "slides" / "slides.pdf",
    GENERATED_DIR / "figures" / "camera_pipeline.png",
    GENERATED_DIR / "figures" / "coins_pipeline.png",
    GENERATED_DIR / "figures" / "astronaut_pipeline.png",
    GENERATED_DIR / "figures" / "synthetic_pipeline.png",
    GENERATED_DIR / "figures" / "sobel_kernels.png",
    GENERATED_DIR / "figures" / "tradeoff_heatmap.png",
    GENERATED_DIR / "data" / "tradeoff_metrics.csv",
    GENERATED_DIR / "data" / "artifact_manifest.json",
]


def collect_text_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES]


def find_absolute_path_literals(paths: list[Path]) -> list[str]:
    pattern = re.compile(r"[A-Za-z]:\\")
    findings: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            findings.append(path.relative_to(ROOT).as_posix())
    return findings


def main() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in EXPECTED_FILES if not path.exists()]
    text_files = collect_text_files(ROOT)
    absolute_literals = find_absolute_path_literals(text_files)

    slides_text = (ROOT / "slides" / "slides.tex").read_text(encoding="utf-8")
    report_text = (ROOT / "docs" / "project_report.tex").read_text(encoding="utf-8")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    checks = {
        "references_section": "References" in slides_text and "References" in report_text,
        "team_section": "Team Responsibilities" in slides_text and "Team Responsibilities" in report_text,
        "narrow_topic": "Edge Detection in Noisy Images" in readme_text,
        "demo_scope": "Two-minute demo" in readme_text,
        "tex_only_delivery": "slides.html" not in readme_text and "references.md" not in readme_text,
    }

    failed = []
    if missing:
        failed.append(f"Missing expected outputs: {', '.join(missing)}")
    if absolute_literals:
        failed.append(f"Absolute path literals found in: {', '.join(absolute_literals)}")
    for label, passed in checks.items():
        if not passed:
            failed.append(f"Check failed: {label}")

    if failed:
        print("Self-check failed.")
        for line in failed:
            print(f" - {line}")
        raise SystemExit(1)

    print("Self-check passed.")
    for label in checks:
        print(f" - {label}: ok")
    print(" - outputs: ok")
    print(" - relative paths: ok")


if __name__ == "__main__":
    main()
