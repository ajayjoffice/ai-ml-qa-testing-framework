"""Validated request, evidence, finding, and release-report schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Outcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class Category(str, Enum):
    APPLICATION = "application_defect"
    REGRESSION = "regression"
    TEST_ISSUE = "test_assertion_or_test_code_issue"
    ENVIRONMENT = "environment_or_configuration_issue"
    INFRASTRUCTURE = "dependency_or_infrastructure_failure"
    POTENTIAL_FLAKY = "potential_flaky_test"
    UNKNOWN = "unknown_or_insufficient_evidence"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Evidence(BaseModel):
    reference: str
    kind: str
    text: str
    line_start: int | None = None
    line_end: int | None = None


class TestResult(BaseModel):
    test_id: str
    original_id: str
    name: str
    classname: str | None = None
    outcome: Outcome
    duration_seconds: float = 0.0
    message: str | None = None
    exception_type: str | None = None
    stack_trace: str | None = None
    source_file: str | None = None
    source_line: int | None = None
    duplicate_index: int = 1
    evidence: list[Evidence] = Field(default_factory=list)


class BuildMetadata(BaseModel):
    commit: str | None = None
    branch: str | None = None
    build_id: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    available_tests: list[str] = Field(default_factory=list)
    test_module_map: dict[str, list[str]] = Field(default_factory=dict)
    critical_tests: list[str] = Field(default_factory=list)
    history: dict[str, dict[str, int]] = Field(default_factory=dict)
    repeated_runs: dict[str, list[str]] = Field(default_factory=dict)
    test_durations: dict[str, float] = Field(default_factory=dict)


class NormalizedReport(BaseModel):
    run_id: str
    test_cases: list[TestResult]
    application_logs: str = ""
    build: BuildMetadata = Field(default_factory=BuildMetadata)
    warnings: list[str] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {item.value: sum(case.outcome == item for case in self.test_cases) for item in Outcome}


class Hypothesis(BaseModel):
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    verified: bool = False


class Finding(BaseModel):
    group_id: str
    test_ids: list[str]
    category: Category
    severity: Severity
    priority: Priority
    summary: str
    observed_facts: list[str] = Field(default_factory=list)
    root_cause: Hypothesis
    alternatives: list[Hypothesis] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    debugging_steps: list[str] = Field(default_factory=list)
    remediation_proposal: str | None = None
    human_review_required: bool = True
    analysis_mode: str = "deterministic"


class Recommendation(str, Enum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class ReleaseAssessment(BaseModel):
    run_id: str
    recommendation: Recommendation
    counts: dict[str, int]
    total: int
    failure_rate: float
    critical_or_high_open_findings: list[str]
    potential_regressions: list[str]
    infrastructure_issues: list[str]
    missing_evidence: list[str]
    rule_trace: list[str]
    limitations: list[str]


class AnalyzeRequest(BaseModel):
    junit_xml: str = Field(min_length=1, max_length=10_000_000)
    application_logs: str = Field(default="", max_length=2_000_000)
    build: BuildMetadata = Field(default_factory=BuildMetadata)
    run_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class TriageResponse(BaseModel):
    run_id: str
    findings: list[Finding]
    release: ReleaseAssessment
    selection: list[dict[str, Any]]
    warnings: list[str]
