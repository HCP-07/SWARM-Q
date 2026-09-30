"""Security audit unit tests for credentials, environment files, and CORS settings."""

import os
import re
import pytest
from pathlib import Path


def test_no_env_file_in_repository():
    """Ensures no active .env file containing live secrets is present in repository."""
    root = Path(__file__).resolve().parent.parent.parent
    env_file = root / ".env"
    assert not env_file.exists(), (
        f"Active .env file found at {env_file}! Secrets must not be committed."
    )


def test_no_hardcoded_secrets_in_codebase():
    """Scans all Python and configuration files for exposed API key signatures."""
    root = Path(__file__).resolve().parent.parent.parent
    secret_patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),
        re.compile(r"ghp_[a-zA-Z0-9]{36}"),
        re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{25,}", re.IGNORECASE),
    ]

    scanned_exts = {".py", ".json", ".yaml", ".yml", ".md", ".sh", ".toml"}
    excluded_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "results"}

    violations = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ext in scanned_exts:
                fpath = os.path.join(dirpath, f)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                        for pattern in secret_patterns:
                            matches = pattern.findall(content)
                            if matches:
                                violations.append(f"{fpath}: matched {pattern.pattern}")
                except Exception:
                    pass

    assert not violations, f"Potential secret exposure detected in:\n" + "\n".join(violations)


def test_server_cors_configuration():
    """Validates FastAPI CORS middleware settings in server/app.py."""
    server_app_path = Path(__file__).resolve().parent.parent.parent / "server" / "app.py"
    with open(server_app_path, "r", encoding="utf-8") as fh:
        code = fh.read()

    # Must not allow wildcard origin with credentials
    assert 'allow_origins=["*"]' not in code or "allow_credentials=True" not in code, (
        "Insecure CORS: wildcard origin cannot be combined with credentials"
    )
