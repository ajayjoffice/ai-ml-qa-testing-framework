"""Deterministic synthetic evaluation; no external LLM provider is called."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .classification import classify_failures
from .junit import parse_junit
from .models import BuildMetadata
from .release import assess_release

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "triage_eval.json"


def _category_scores(gold: list[str], predicted: list[str]) -> dict[str, object]:
    labels = sorted(set(gold) | set(predicted))
    true_positive = sum(left == right for left, right in zip(gold, predicted))
    per_category: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(left == label and right == label for left, right in zip(gold, predicted))
        fp = sum(left != label and right == label for left, right in zip(gold, predicted))
        fn = sum(left == label and right != label for left, right in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_category[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
            "support": sum(item == label for item in gold),
        }
    precision_micro = true_positive / len(predicted) if predicted else 0.0
    recall_micro = true_positive / len(gold) if gold else 0.0
    return {
        "precision_micro": round(precision_micro, 4),
        "recall_micro": round(recall_micro, 4),
        "f1_micro": round(2 * precision_micro * recall_micro / (precision_micro + recall_micro), 4) if precision_micro + recall_micro else 0.0,
        "accuracy": round(true_positive / max(len(gold), 1), 4),
        "per_category": per_category,
    }


def evaluate(path: Path = DATASET) -> dict[str, object]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    gold_categories: list[str] = []
    predicted_categories: list[str] = []
    gold_gates: list[str] = []
    predicted_gates: list[str] = []
    citation_count = valid_citations = 0
    abstention_count = correct_abstentions = 0
    failures = 0
    rows = []
    start = time.perf_counter()
    for case in cases:
        try:
            metadata = BuildMetadata(
                changed_files=case.get("changed_files", []),
                critical_tests=case.get("critical_tests", []),
                repeated_runs=case.get("repeated_runs", {}),
            )
            report = parse_junit(case["xml"], run_id=case["id"], build=metadata)
            findings = classify_failures(report)
            release = assess_release(report, findings)
            category = findings[0].category.value if findings else None
            expected_category = case["expected_category"]
            if expected_category:
                gold_categories.append(expected_category)
                predicted_categories.append(category or "no_finding")
            else:
                abstention_count += 1
                correct_abstentions += not findings
            gold_gates.append(case["expected_release"])
            predicted_gates.append(release.recommendation.value)
            evidence_refs = {item.reference for finding in findings for item in finding.evidence}
            for finding in findings:
                for ref in finding.root_cause.evidence_refs:
                    citation_count += 1
                    valid_citations += ref in evidence_refs
            rows.append({
                "id": case["id"],
                "expected_category": expected_category,
                "predicted_category": category,
                "expected_release": case["expected_release"],
                "predicted_release": release.recommendation.value,
                "pass": category == expected_category and release.recommendation.value == case["expected_release"],
            })
        except Exception as exc:  # Each bad evaluation record is counted as a failure, not hidden.
            failures += 1
            rows.append({"id": case.get("id", "unknown"), "error": type(exc).__name__, "pass": False})
    elapsed = time.perf_counter() - start
    gate_accuracy = sum(left == right for left, right in zip(gold_gates, predicted_gates)) / max(len(gold_gates), 1)
    category_metrics = _category_scores(gold_categories, predicted_categories)
    return {
        "mode": "deterministic; no LLM calls",
        "dataset": path.name,
        "case_count": len(cases),
        "labelled_category_case_count": len(gold_categories),
        "category_metrics": category_metrics,
        "root_cause_hypothesis_accuracy": category_metrics["accuracy"],
        "evidence_grounding_rate": round(valid_citations / citation_count, 4) if citation_count else None,
        "correct_insufficient_evidence_escalation_rate": round(correct_abstentions / abstention_count, 4) if abstention_count else None,
        "release_gate_accuracy": round(gate_accuracy, 4),
        "test_selection_metrics": None,
        "test_selection_note": "No labeled test-selection ground truth is supplied, so ranking precision/recall are not reported.",
        "elapsed_seconds": round(elapsed, 4),
        "case_failure_rate": round(failures / max(len(cases), 1), 4),
        "limitations": [
            "Small synthetic labels are authored to exercise rule behavior and do not estimate production accuracy.",
            "Root-cause hypothesis accuracy uses expected category as a coarse proxy, not verified causal labels.",
            "LLM-dependent quality is not measured here; external provider behavior is covered by mocked tests.",
        ],
        "results": rows,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
