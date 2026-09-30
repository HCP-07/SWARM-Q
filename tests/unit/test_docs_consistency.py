"""Unit tests ensuring documentation consistency, absence of legacy traffic terms, and exact naming."""

import pytest
from pathlib import Path


def test_no_legacy_traffic_terms_in_docs():
    """Scans all Markdown documentation to verify no legacy traffic or SUMO terminology remains."""
    root = Path(__file__).resolve().parent.parent.parent
    banned_terms = ["sumo", "traci", "traffic light", "intersection controller"]
    doc_paths = list(root.glob("docs/*.md")) + [root / "README.md"]

    violations = []
    for doc in doc_paths:
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8").lower()
        for term in banned_terms:
            if term in text:
                violations.append(f"{doc.name} contains banned term '{term}'")

    assert not violations, f"Documentation contains legacy remnants:\n" + "\n".join(violations)


def test_proposed_method_naming_consistency():
    """Verifies that the proposed algorithm is systematically documented as AD-QPSO."""
    readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        assert "AD-QPSO" in text, "README must document the primary proposed algorithm as AD-QPSO"
