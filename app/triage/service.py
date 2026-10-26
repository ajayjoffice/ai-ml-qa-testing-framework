"""Triage orchestration around the existing churn QA application."""

from __future__ import annotations

import uuid

from .analysis import Analyzer, DeterministicAnalyzer
from .classification import classify_failures
from .junit import parse_junit
from .models import BuildMetadata, Evidence, Finding, Hypothesis, NormalizedReport, TriageResponse
from .release import assess_release
from .selection import recommend_tests


class TriageService:
    def __init__(self, analyzer: Analyzer | None = None, block_on_any_failure: bool = False):
        self.analyzer = analyzer or DeterministicAnalyzer()
        self.block_on_any_failure = block_on_any_failure
        self.runs: dict[str, TriageResponse] = {}

    @staticmethod
    def _attach_log_evidence(report: NormalizedReport, findings: list[Finding]) -> None:
        if not report.application_logs:
            return
        cases = {case.test_id: case for case in report.test_cases}
        for finding in findings:
            terms = {
                term.lower()
                for test_id in finding.test_ids
                if test_id in cases
                for term in (cases[test_id].name, cases[test_id].exception_type or "", (cases[test_id].message or "")[:100])
                if len(term.strip()) >= 4
            }
            for line_number, line in enumerate(report.application_logs.splitlines(), start=1):
                if terms and any(term in line.lower() for term in terms):
                    evidence = Evidence(reference=f"application_logs:line-{line_number}", kind="application_log_line", text=line[:2000], line_start=line_number, line_end=line_number)
                    finding.evidence.append(evidence)
                    finding.root_cause.evidence_refs.append(evidence.reference)
                    if sum(item.kind == "application_log_line" for item in finding.evidence) >= 10:
                        break

    def analyze(self, xml: str, *, logs: str = "", build: BuildMetadata | None = None, run_id: str | None = None) -> TriageResponse:
        report = parse_junit(xml, run_id=run_id or str(uuid.uuid4()), application_logs=logs, build=build)
        findings = classify_failures(report)
        self._attach_log_evidence(report, findings)
        warnings = list(report.warnings)
        analyzed: list[Finding] = []
        for finding in findings:
            try:
                candidate = self.analyzer.enrich(finding.model_copy(deep=True))
                valid_refs = {item.reference for item in finding.evidence}
                refs = set(candidate.root_cause.evidence_refs)
                refs.update(ref for alt in candidate.alternatives for ref in alt.evidence_refs)
                if not refs or not refs.issubset(valid_refs):
                    raise ValueError("analysis cited absent evidence")
                # Only provider-generated narrative fields may change. Keep policy fields,
                # test IDs, evidence, and the unverified hypothesis status deterministic.
                analyzed.append(finding.model_copy(update={
                    "summary": candidate.summary,
                    "root_cause": Hypothesis(summary=candidate.root_cause.summary, evidence_refs=candidate.root_cause.evidence_refs, confidence=candidate.root_cause.confidence, verified=False),
                    "alternatives": candidate.alternatives,
                    "debugging_steps": candidate.debugging_steps,
                    "remediation_proposal": candidate.remediation_proposal,
                    "human_review_required": True,
                    "analysis_mode": getattr(self.analyzer, "name", "configured_provider"),
                }))
            except Exception as exc:  # Provider-specific outages and output validation failures.
                warnings.append(f"LLM analysis for {finding.group_id} failed; deterministic result used ({type(exc).__name__}).")
                analyzed.append(finding)
        release = assess_release(report, analyzed, self.block_on_any_failure)
        response = TriageResponse(run_id=report.run_id, findings=analyzed, release=release, selection=recommend_tests(report), warnings=warnings)
        self.runs[response.run_id] = response
        if len(self.runs) > 100:
            self.runs.pop(next(iter(self.runs)))
        return response

    def get(self, run_id: str) -> TriageResponse | None:
        return self.runs.get(run_id)
