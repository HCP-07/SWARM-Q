"""Unit tests verifying real-time latency budget tracking and SLA enforcement."""

import pytest
import numpy as np
from configs.config import LatencyBudgetConfig
from src.metrics.latency import LatencyTracker, LatencyReport


def test_latency_budget_sla_pass():
    """Ensures that latency under budget evaluates to passed=True."""
    budget = LatencyBudgetConfig(budget_ms=100.0)
    tracker = LatencyTracker(config=budget)

    # Record 100 ticks with latencies between 10ms and 40ms
    for i in range(100):
        lat = 10.0 + (i % 30)
        tracker.record_tick(lat)

    report = tracker.compute_report()
    assert report.p95_ms <= budget.budget_ms
    assert report.passed is True


def test_latency_budget_sla_fail():
    """Ensures that latency exceeding budget strictly reports passed=False without fabrication."""
    budget = LatencyBudgetConfig(budget_ms=30.0)
    tracker = LatencyTracker(config=budget)

    # Record 100 ticks with latencies of 150ms
    for _ in range(100):
        tracker.record_tick(150.0)

    report = tracker.compute_report()
    assert report.p95_ms > budget.budget_ms
    assert report.passed is False


def test_custom_budget_environment_override(monkeypatch):
    """Verifies that ADSO_DEFAULT_LATENCY_BUDGET_MS environment variable can configure the budget."""
    monkeypatch.setenv("ADSO_DEFAULT_LATENCY_BUDGET_MS", "45.0")
    from main import _env_float
    overridden = _env_float("ADSO_DEFAULT_LATENCY_BUDGET_MS", 25.0, 1.0, 5000.0)
    assert overridden == 45.0
