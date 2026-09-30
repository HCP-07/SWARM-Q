"""Packages the competition-ready solution into a clean, portable zip archive."""

from __future__ import annotations
import os
import zipfile
import sys


def package_solution(output_zip: str = "adaptive_swarm_intelligence_solution.zip") -> None:
    exclude_dirs = {".git", "venv", ".pytest_cache", "__pycache__", ".vscode", ".idea"}
    exclude_files = {".DS_Store", output_zip}

    print(f"Packaging project into: {output_zip}...")
    total_files = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file in exclude_files or file.endswith(".pyc"):
                    continue
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, ".")
                zf.write(file_path, archive_name)
                total_files += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Archive created successfully: {output_zip} ({total_files} files, {size_mb:.2f} MB)")


if __name__ == "__main__":
    package_solution()
