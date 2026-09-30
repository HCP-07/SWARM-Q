"""Packages clean distribution zip file: adaptive_swarm_intelligence_solution.zip."""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_OUT = ROOT / "adaptive_swarm_intelligence_solution.zip"

EXCLUDE_DIRS = {
    "venv", ".venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".git", ".idea", ".vscode", "scratch"
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".DS_Store"}


def build_clean_zip():
    print(f"Building clean distribution archive at: {ZIP_OUT}")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    included_count = 0
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in sorted(files):
                if any(f.endswith(ext) for ext in EXCLUDE_EXTS) or f.startswith("._") or f == ".env":
                    continue
                if f.endswith(".zip"):
                    continue
                file_path = Path(root) / f
                arc_name = file_path.relative_to(ROOT)
                zf.write(file_path, arc_name)
                included_count += 1

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {included_count} files into {ZIP_OUT} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    build_clean_zip()
