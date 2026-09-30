"""Shim re-exporting QPSOTrajectoryEngine, qpso_update, and conflict_penalty from src.optimization.qpso."""

from __future__ import annotations
from src.optimization.qpso import QPSOTrajectoryEngine, qpso_update, conflict_penalty

__all__ = ["QPSOTrajectoryEngine", "qpso_update", "conflict_penalty"]
