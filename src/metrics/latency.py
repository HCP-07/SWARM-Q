"""High-precision per-tick latency instrumentation and SLA compliance checking."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from configs.config import LatencyBudgetConfig


@dataclass
class LatencyReport:
    """Statistical summary of per-tick decision latencies."""
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    budget_ms: float
    passed: bool
    total_ticks: int


class LatencyTracker:
    """Records high-resolution tick timings and assesses real-time performance."""

    def __init__(self, config: Any):
        if isinstance(config, (int, float)):
            self.config = LatencyBudgetConfig(budget_ms=float(config))
        else:
            self.config = config
        self.tick_latencies_ms: List[float] = []

    def record_tick(self, latency_ms: float) -> None:
        self.tick_latencies_ms.append(float(latency_ms))

    def compute_report(self) -> LatencyReport:
        if not self.tick_latencies_ms:
            return LatencyReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.config.budget_ms, True, 0)

        data = np.array(self.tick_latencies_ms, dtype=np.float64)
        mean_v = float(np.mean(data))
        median_v = float(np.median(data))
        p95_v = float(np.percentile(data, 95))
        p99_v = float(np.percentile(data, 99))
        max_v = float(np.max(data))
        min_v = float(np.min(data))

        passed = p95_v <= self.config.budget_ms

        return LatencyReport(
            mean_ms=mean_v,
            median_ms=median_v,
            p95_ms=p95_v,
            p99_ms=p99_v,
            max_ms=max_v,
            min_ms=min_v,
            budget_ms=self.config.budget_ms,
            passed=passed,
            total_ticks=len(data)
        )
