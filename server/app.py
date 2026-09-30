"""FastAPI application factory with lifespan bootstrap and hardened CORS security."""

from __future__ import annotations
import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import router, RESULTS_DIR, BASE_DIR


def bootstrap_results() -> None:
    """Executes initial baseline simulation in background worker thread if results are absent."""
    if not (RESULTS_DIR / "trajectory.csv").exists():
        cmd = [
            sys.executable,
            str(BASE_DIR / "main.py"),
            "--scenario", "stress",
            "--agents", "15",
            "--iterations", "120",
            "--visualize"
        ]
        try:
            subprocess.run(cmd, check=False, timeout=900)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated on_event."""
    if not (RESULTS_DIR / "trajectory.csv").exists():
        asyncio.get_running_loop().run_in_executor(None, bootstrap_results)
    yield


def create_app() -> FastAPI:
    """Creates configured FastAPI application."""
    app = FastAPI(
        title="Adaptive Decentralized QPSO (AD-QPSO) - Mission Control",
        description="Autonomous Multi-Agent Decision-Making and Trajectory Optimization Platform",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS hardening: never wildcard with credentials, strict localhost default
    raw_origins = os.getenv("ADSO_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    app.include_router(router)
    return app
