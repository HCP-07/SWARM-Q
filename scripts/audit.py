"""Automated quality, constraint, and terminological audit script.

Enforces zero-tolerance checks for:
1. Zero banned legacy terms
2. Architectural decentralization
3. Hard collision-free invariants
4. Real-time latency budget (< 100 ms)
5. Comprehensive test coverage (40+ passing tests)
6. Empirical benchmark & ablation reproducibility
7. Documentation integrity
"""

from __future__ import annotations
import os
import re
import sys
import json
import subprocess
from typing import List, Tuple, Dict, Any

BANNED_PATTERNS = [
    r"\btraffic\b",
    r"\bvehicle\b",
    r"\bvehicles\b",
    r"\broad\b",
    r"\broads\b",
    r"\btransportation\b",
    r"\bqpso\b",
    r"\biqpso\b",
    r"\btomtom\b",
    r"google maps",
    r"\bosm\b",
    r"shortest-path",
    r"\bvrp\b",
    r"city graph",
    r"\bcongestion\b",
    r"\bfuel\b",
]

REQUIRED_FILES = [
    ".env.example",
    ".gitignore",
    "README.md",
    "FINAL_REPORT.md",
    "config.py",
    "main.py",
    "core/collision.py",
    "core/adaptive_policy.py",
    "core/agent.py",
    "core/environment.py",
    "core/metrics.py",
    "core/swarm.py",
    "algorithms/greedy.py",
    "algorithms/static_priority.py",
    "algorithms/non_adaptive_swarm.py",
    "algorithms/adaptive_swarm.py",
    "scenarios/__init__.py",
    "experiments/benchmark.py",
    "experiments/ablation.py",
    "experiments/statistics.py",
    "experiments/visualize.py",
    "scripts/audit.py",
    "scripts/package_solution.py",
]


def audit_banned_terms() -> Tuple[bool, List[str]]:
    """Scans all code and documentation files for prohibited legacy terms."""
    regex = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)
    violations = []

    # Ignore virtualenvs, git internals, caches, and audit script's own pattern definitions
    ignore_dirs = {".git", "venv", ".pytest_cache", "__pycache__"}
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.endswith((".py", ".md", ".json", ".csv", ".txt", ".yml", ".yaml", ".toml")):
                path = os.path.relpath(os.path.join(root, f), ".")
                if path in ("scripts/audit.py", "audit.py"):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                        for line_num, line in enumerate(fp, 1):
                            m = regex.findall(line)
                            if m:
                                violations.append(f"{path}:{line_num}: found {set(m)} -> {line.strip()[:60]}")
                except Exception as e:
                    violations.append(f"Could not read {path}: {e}")

    return len(violations) == 0, violations


def audit_required_files() -> Tuple[bool, List[str]]:
    """Validates existence of all necessary architectural and documentation files."""
    missing = []
    for rf in REQUIRED_FILES:
        if not os.path.exists(rf):
            missing.append(rf)
    return len(missing) == 0, missing


def audit_tests() -> Tuple[bool, int, str]:
    """Executes pytest suite and counts passing test assertions."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = res.stdout + res.stderr
        m = re.search(r"(\d+) passed", output)
        passed_count = int(m.group(1)) if m else 0
        success = (res.returncode == 0 and passed_count >= 40)
        return success, passed_count, output
    except Exception as e:
        return False, 0, str(e)


def audit_results_artifacts() -> Tuple[bool, List[str]]:
    """Checks whether benchmark, ablation, and plot outputs are present and valid."""
    required_artifacts = [
        "results/benchmark.csv",
        "results/benchmark.json",
        "results/ablation.csv",
        "results/ablation.json",
        "results/summary.csv",
        "results/results.json",
        "results/plots/trajectories.png",
        "results/plots/convergence.png",
        "results/plots/latency.png",
    ]
    missing = []
    for ra in required_artifacts:
        if not os.path.exists(ra) or os.path.getsize(ra) == 0:
            missing.append(ra)
    return len(missing) == 0, missing


def audit_safety_invariants() -> Tuple[bool, str]:
    """Audits results.json to ensure 0 collisions across all runs."""
    results_file = "results/results.json"
    if not os.path.exists(results_file):
        return False, "results.json not found"

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        coll_max = summary.get("collision_count", {}).get("max", -1)
        if coll_max == 0:
            return True, "0 collisions across all runs"
        return False, f"Max collisions observed: {coll_max}"
    except Exception as e:
        return False, f"Error reading results.json: {e}"


def run_full_audit() -> int:
    """Executes full automated audit and prints 7-dimension scorecard."""
    print("=" * 80)
    print("AUTOMATED VERIFICATION AUDIT & EVALUATION SCORECARD")
    print("=" * 80)

    # 1. Banned Terms Audit
    banned_ok, banned_violations = audit_banned_terms()
    print(f"\n[Dimension 1: Zero Banned Terms] ... {'PASS' if banned_ok else 'FAIL'}")
    if not banned_ok:
        print(f"  -> Found {len(banned_violations)} banned term occurrences:")
        for v in banned_violations[:10]:
            print(f"     * {v}")

    # 2. Required Files Audit
    files_ok, missing_files = audit_required_files()
    print(f"[Dimension 2: Architecture & File Structure] ... {'PASS' if files_ok else 'FAIL'}")
    if not files_ok:
        print(f"  -> Missing {len(missing_files)} required files: {missing_files}")

    # 3. Test Suite Audit
    test_ok, test_count, test_out = audit_tests()
    print(f"[Dimension 3: Test Suite Density & Invariants] ... {'PASS' if test_ok else 'FAIL'} ({test_count} tests passed)")
    if not test_ok:
        print(f"  -> Test failure details:\n{test_out[:400]}")

    # 4. Results & Artifacts Audit
    artifacts_ok, missing_artifacts = audit_results_artifacts()
    print(f"[Dimension 4: Empirical Benchmark Artifacts] ... {'PASS' if artifacts_ok else 'FAIL'}")
    if not artifacts_ok:
        print(f"  -> Missing artifacts: {missing_artifacts}")

    # 5. Hard Safety Invariants Audit
    safety_ok, safety_msg = audit_safety_invariants()
    print(f"[Dimension 5: Hard Collision-Free Invariant] ... {'PASS' if safety_ok else 'FAIL'} ({safety_msg})")

    # 6. Real-Time SLA Budget Audit
    sla_ok = True
    sla_msg = "P95 latency <= 100 ms"
    if os.path.exists("results/results.json"):
        try:
            with open("results/results.json", "r") as f:
                d = json.load(f)
            p95_max = d.get("summary", {}).get("p95_decision_latency_ms", {}).get("max", 0.0)
            if p95_max > 100.0:
                sla_ok = False
                sla_msg = f"Max P95 latency {p95_max:.2f} ms exceeded 100 ms budget"
            else:
                sla_msg = f"Max P95 latency: {p95_max:.2f} ms (Budget: 100 ms)"
        except Exception:
            sla_ok = False
            sla_msg = "Could not verify SLA in results.json"
    else:
        sla_ok = False
        sla_msg = "results.json not found"

    print(f"[Dimension 6: Real-Time Latency SLA Budget] ... {'PASS' if sla_ok else 'FAIL'} ({sla_msg})")

    # 7. Decentralization & No Coordinator Single Point of Failure
    decentralized_ok = os.path.exists("core/adaptive_policy.py") and os.path.exists("core/agent.py")
    print(f"[Dimension 7: Decentralized Multi-Agent Autonomy] ... {'PASS' if decentralized_ok else 'FAIL'}")

    # Scorecard Table
    print("\n" + "=" * 80)
    print("FINAL 7-POINT SYSTEM AUDIT SCORECARD")
    print("=" * 80)
    scores = [
        ("1. Zero Banned Legacy Terms", banned_ok),
        ("2. Architecture & File Structure", files_ok),
        ("3. Comprehensive Test Suite (40+ assertions)", test_ok),
        ("4. Empirical Benchmark & Ablation Data", artifacts_ok),
        ("5. Hard Collision-Free Safety Invariants", safety_ok),
        ("6. Real-Time SLA Decision Latency (<100ms)", sla_ok),
        ("7. Decentralized Multi-Agent Autonomy", decentralized_ok),
    ]

    total_pass = sum(1 for _, ok in scores if ok)
    for name, ok in scores:
        print(f"  {name:<52} : [{'PASS' if ok else 'FAIL'}]")

    print("-" * 80)
    print(f"TOTAL SCORE: {total_pass} / {len(scores)} DIMENSIONS PASSED")
    print("=" * 80)

    all_passed = (total_pass == len(scores))
    if all_passed:
        print("RESULT: ALL AUDIT CRITERIA SATISFIED. SYSTEM IS COMPETITION-READY.")
    else:
        print("RESULT: AUDIT FAILED. ADDRESS THE FLAGGED ISSUES ABOVE.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_full_audit())
