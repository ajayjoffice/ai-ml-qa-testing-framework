"""Deterministic signature grouping and first-pass defect classification."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .models import Category, Finding, Hypothesis, NormalizedReport, Outcome, Priority, Severity, TestResult


def _normalise(text: str) -> str:
    text = re.sub(r"0x[0-9a-f]+", "<addr>", text.lower())
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
    text = re.sub(r"(/[\w./-]+)+", "<path>", text)
    return re.sub(r"\s+", " ", text).strip()[:500]


def group_failures(report: NormalizedReport) -> list[list[TestResult]]:
    groups: defaultdict[str, list[TestResult]] = defaultdict(list)
    for case in report.test_cases:
        if case.outcome not in {Outcome.FAILED, Outcome.ERROR}:
            continue
        final_line = (case.stack_trace or "").splitlines()[-1] if case.stack_trace else ""
        key = "|".join((case.exception_type or "", _normalise(case.message or ""), _normalise(final_line)))
        groups[hashlib.sha1(key.encode()).hexdigest()[:10]].append(case)
    return list(groups.values())


def _category(cases: list[TestResult], report: NormalizedReport) -> Category:
    text = " ".join(f"{case.exception_type or ''} {case.message or ''} {case.stack_trace or ''}" for case in cases).lower()
    for case in cases:
        runs = [value.lower() for value in report.build.repeated_runs.get(case.original_id, [])]
        has_pass = any(value in {"passed", "pass"} for value in runs)
        has_fail = any(value in {"failed", "fail", "error"} for value in runs)
        if has_pass and has_fail:
            return Category.POTENTIAL_FLAKY
    if any(word in text for word in ("connection refused", "timed out", "timeout", "no such host", "network is unreachable", "service unavailable")):
        return Category.INFRASTRUCTURE
    if any(word in text for word in ("environment variable", "configuration", "config file", "permission denied", "missing setting")):
        return Category.ENVIRONMENT
    if "assertionerror" in text or "assertion failed" in text or any((case.source_file or "").startswith(("test_", "tests/")) for case in cases):
        return Category.TEST_ISSUE
    changed = [path.replace("\\", "/").lower() for path in report.build.changed_files]
    for case in cases:
        context = f"{case.source_file or ''} {case.classname or ''} {case.stack_trace or ''}".replace("\\", "/").lower()
        if any(path and path in context for path in changed):
            return Category.REGRESSION
    if not any(case.message or case.stack_trace for case in cases):
        return Category.UNKNOWN
    return Category.APPLICATION


def classify_failures(report: NormalizedReport) -> list[Finding]:
    findings: list[Finding] = []
    for index, cases in enumerate(group_failures(report), start=1):
        category = _category(cases, report)
        critical = any(case.original_id in report.build.critical_tests for case in cases)
        if critical:
            severity, priority = Severity.CRITICAL, Priority.P0
        elif category in {Category.APPLICATION, Category.REGRESSION, Category.ENVIRONMENT, Category.INFRASTRUCTURE}:
            severity, priority = Severity.HIGH, Priority.P1
        elif category in {Category.TEST_ISSUE, Category.POTENTIAL_FLAKY}:
            severity, priority = Severity.MEDIUM, Priority.P2
        else:
            severity, priority = Severity.UNKNOWN, Priority.P2
        evidence = [item for case in cases for item in case.evidence]
        observed = [f"{case.test_id} ended with {case.outcome.value}." + (f" {case.exception_type}: {case.message}" if case.message else "") for case in cases]
        if category == Category.POTENTIAL_FLAKY:
            summary = "Mixed pass/fail repeated-run history suggests a potential flaky test; verify with additional runs."
            cause = "Intermittent behavior is possible from supplied mixed history; cause is not confirmed."
            confidence = 0.55
        elif category == Category.UNKNOWN:
            summary = "Failure evidence is insufficient to determine a root cause."
            cause = "Insufficient diagnostic evidence to identify a root cause."
            confidence = 0.1
        else:
            summary = f"{len(cases)} test(s) share a diagnostic signature; classified as {category.value}."
            cause = f"Observed evidence is consistent with {category.value}; this remains a hypothesis."
            confidence = 0.65 if evidence else 0.25
        findings.append(Finding(
            group_id=f"failure-{index}",
            test_ids=[case.test_id for case in cases],
            category=category,
            severity=severity,
            priority=priority,
            summary=summary,
            observed_facts=observed,
            root_cause=Hypothesis(summary=cause, evidence_refs=[item.reference for item in evidence], confidence=confidence, verified=False),
            evidence=evidence,
            debugging_steps=["Reproduce the failing test with the same commit and build configuration.", "Inspect the referenced error and stack evidence before changing code."],
            remediation_proposal="Investigate the suspected failure using the cited evidence; this is a proposal, not a verified fix.",
            human_review_required=True,
        ))
    return findings
