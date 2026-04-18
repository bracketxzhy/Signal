from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
GENERATED_DIR = ASSETS_DIR / "generated"
DOCS_DIR = PROJECT_ROOT / "docs"
SLIDES_DIR = PROJECT_ROOT / "slides"


def ensure_output_dirs() -> dict[str, Path]:
    figures_dir = GENERATED_DIR / "figures"
    data_dir = GENERATED_DIR / "data"
    for path in (ASSETS_DIR, GENERATED_DIR, figures_dir, data_dir, DOCS_DIR, SLIDES_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {"figures": figures_dir, "data": data_dir}
