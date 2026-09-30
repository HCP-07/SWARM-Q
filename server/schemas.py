"""Pydantic schemas and authentication dependencies for Mission Control API."""

from __future__ import annotations
import os
import secrets
from typing import Literal, Optional
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

MAX_AGENTS = int(os.getenv("ADSO_MAX_AGENTS", "100"))
MAX_TICKS = int(os.getenv("ADSO_MAX_TICKS", "500"))


class SimRequest(BaseModel):
    """Validated simulation execution payload."""
    scenario: Literal[
        "static",
        "dynamic_obstacle",
        "multiple_moving_obstacles",
        "communication_dropout",
        "agent_failure",
        "stress",
        "full_stress"
    ] = "stress"
    agents: int = Field(12, ge=1, le=MAX_AGENTS, description="Agent swarm population")
    ticks: int = Field(60, ge=1, le=MAX_TICKS, description="Number of simulation steps")
    seed: int = Field(42, ge=0, le=2**31 - 1, description="Deterministic random seed")
    latency_budget: float = Field(25.0, gt=0, le=5000.0, description="Real-time latency budget in ms")


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Optional constant-time API key verification for endpoint security."""
    expected = os.getenv("ADSO_API_KEY")
    if expected:
        if not x_api_key or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
