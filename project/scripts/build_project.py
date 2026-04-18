from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image2d_project.pipeline import build_project_outputs


def compile_tex(tex_path: Path, passes: int = 2) -> Path:
    for _ in range(passes):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex_path.name,
            ],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    result = build_project_outputs()
    report_pdf = compile_tex(ROOT / "docs" / "project_report.tex")
    slides_pdf = compile_tex(ROOT / "slides" / "slides.tex")
    print("Build complete.")
    for artifact in result["artifacts"]:
        print(f" - {artifact}")
    print(f" - {report_pdf.relative_to(ROOT).as_posix()}")
    print(f" - {slides_pdf.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
