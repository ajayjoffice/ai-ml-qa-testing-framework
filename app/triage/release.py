"""Auditable release gate based only on submitted results and configured policy."""

from __future__ import annotations

from .models import Category, Finding, NormalizedReport, Recommendation, ReleaseAssessment, Severity


def assess_release(report: NormalizedReport, findings: list[Finding], block_on_any_failure: bool = False) -> ReleaseAssessment:
    counts = report.counts
    total = len(report.test_cases)
    failed = counts["failed"] + counts["error"]
    rate = failed / total if total else 0.0
    critical = [finding.group_id for finding in findings if finding.severity == Severity.CRITICAL]
    high = [finding.group_id for finding in findings if finding.severity == Severity.HIGH]
    regressions = [finding.group_id for finding in findings if finding.category == Category.REGRESSION]
    infrastructure = [finding.group_id for finding in findings if finding.category == Category.INFRASTRUCTURE]
    missing: list[str] = []
    limitations: list[str] = []
    trace: list[str] = []
    if not total:
        missing.append("Report contains no test cases.")
    if any(case.outcome.value in {"failed", "error"} and not (case.message or case.stack_trace) for case in report.test_cases):
        missing.append("At least one failed test has no message or stack trace.")
    if critical:
        recommendation = Recommendation.BLOCK
        trace.append("BLOCK: at least one failed test is configured as critical.")
    elif block_on_any_failure and failed:
        recommendation = Recommendation.BLOCK
        trace.append("BLOCK: QA_BLOCK_ON_ANY_FAILURE policy applies to a failed/error result.")
    elif failed or missing:
        recommendation = Recommendation.REVIEW
        trace.append("REVIEW: failures or missing diagnostic evidence require human assessment.")
    elif not total or counts["passed"] == 0:
        recommendation = Recommendation.REVIEW
        trace.append("REVIEW: no passing test evidence is available.")
    else:
        recommendation = Recommendation.PASS
        trace.append("PASS: no failures or errors were observed in the submitted report.")
        limitations.append("A passing test report does not prove that software is defect-free.")
    return ReleaseAssessment(
        run_id=report.run_id,
        recommendation=recommendation,
        counts=counts,
        total=total,
        failure_rate=round(rate, 4),
        critical_or_high_open_findings=critical + high,
        potential_regressions=regressions,
        infrastructure_issues=infrastructure,
        missing_evidence=missing,
        rule_trace=trace,
        limitations=limitations + report.warnings,
    )
