from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.triage.analysis import LLMOutput, OpenAICompatibleAnalyzer
from app.triage.classification import classify_failures, group_failures
from app.triage.evaluate import evaluate
from app.triage.junit import MAX_LOG_CHARS, MAX_REPORT_BYTES, ReportError, parse_junit
from app.triage.models import BuildMetadata, Category, Recommendation, Severity
from app.triage.release import assess_release
from app.triage.selection import recommend_tests
from app.triage.service import TriageService

FIXTURE = Path(__file__).parent.parent / "fixtures" / "triage_junit.xml"


def parse(xml: str | None = None, **kwargs):
    return parse_junit(xml if xml is not None else FIXTURE.read_text(), **kwargs)


def failure_xml(name="test_failure", message="RuntimeError: broken", exception="RuntimeError", classname="app.feature"):
    return f'<testsuite><testcase classname="{classname}" name="{name}" time="0.1"><failure message="{message}" type="{exception}">{message}</failure></testcase></testsuite>'


def test_junit_parser_normalizes_all_outcomes_and_traceback_evidence():
    report = parse()
    assert report.counts == {"passed": 1, "failed": 2, "error": 2, "skipped": 1}
    assert report.test_cases[0].source_line == 24
    assert any(item.kind == "stack_trace_line" for item in report.test_cases[0].evidence)
    assert report.test_cases[-1].outcome.value == "skipped"


def test_missing_test_name_is_assigned_with_a_warning():
    report = parse_junit('<testsuite><testcase/></testsuite>')
    assert report.test_cases[0].test_id == "unnamed-test-1"
    assert "had no name" in report.warnings[0]


def test_duplicate_identifiers_are_preserved_with_stable_suffixes():
    report = parse_junit('<testsuite><testcase name="same"/><testcase name="same"/></testsuite>')
    assert [case.test_id for case in report.test_cases] == ["same", "same#duplicate-2"]
    assert report.test_cases[1].duplicate_index == 2


@pytest.mark.parametrize("xml", ["", "<testsuite>", "<testsuite/>", "<!DOCTYPE x [<!ENTITY e 'bad'>]><testsuite><testcase name='x'/></testsuite>"])
def test_empty_malformed_and_entity_reports_are_rejected(xml):
    with pytest.raises(ReportError):
        parse_junit(xml)


def test_large_report_is_rejected_and_logs_are_truncated():
    with pytest.raises(ReportError, match="exceeds"):
        parse_junit(" " * (MAX_REPORT_BYTES + 1))
    report = parse_junit('<testsuite><testcase name="ok"/></testsuite>', application_logs="x" * (MAX_LOG_CHARS + 20))
    assert len(report.application_logs) == MAX_LOG_CHARS
    assert "retained the tail" in report.warnings[0]


def test_common_failures_are_grouped_by_normalized_signature():
    xml = "<testsuite>" + "".join(
        f'<testcase classname="app.orders" name="test_{n}"><failure message="RuntimeError: status {status}" type="RuntimeError">RuntimeError: status {status}</failure></testcase>'
        for n, status in (("a", 500), ("b", 503))
    ) + "</testsuite>"
    groups = group_failures(parse_junit(xml))
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_classifier_identifies_assertion_infrastructure_and_environment():
    report = parse()
    categories = {finding.category for finding in classify_failures(report)}
    assert Category.TEST_ISSUE in categories
    assert Category.INFRASTRUCTURE in categories
    assert Category.ENVIRONMENT in categories


def test_regression_requires_changed_file_evidence():
    xml = failure_xml(message="RuntimeError: payment failed")
    with_trace = xml.replace("RuntimeError: payment failed</failure>", 'File "src/payments.py", line 25\nRuntimeError: payment failed</failure>')
    report = parse_junit(with_trace, build=BuildMetadata(changed_files=["src/payments.py"]))
    assert classify_failures(report)[0].category == Category.REGRESSION


def test_flaky_classification_requires_mixed_repeated_history():
    xml = failure_xml(classname="app.cache", name="test_cache")
    without_history = classify_failures(parse_junit(xml))[0]
    with_history = classify_failures(parse_junit(xml, build=BuildMetadata(repeated_runs={"app.cache::test_cache": ["passed", "failed"]})))[0]
    assert without_history.category != Category.POTENTIAL_FLAKY
    assert with_history.category == Category.POTENTIAL_FLAKY


def test_no_diagnostic_content_is_classified_unknown_and_requires_review():
    report = parse_junit('<testsuite><testcase name="unknown"><error type="Error"/></testcase></testsuite>')
    finding = classify_failures(report)[0]
    assert finding.category == Category.UNKNOWN
    assert finding.human_review_required
    assert finding.root_cause.verified is False


def test_release_gate_counts_and_blocks_critical_configured_failure():
    build = BuildMetadata(critical_tests=["app.feature::test_failure"])
    report = parse_junit(failure_xml(), build=build)
    finding = classify_failures(report)
    result = assess_release(report, finding)
    assert finding[0].severity == Severity.CRITICAL
    assert result.recommendation == Recommendation.BLOCK
    assert result.counts["error"] == 0
    assert result.failure_rate == 1.0


def test_release_gate_reviews_failure_and_can_block_all_failures():
    report = parse_junit(failure_xml())
    findings = classify_failures(report)
    assert assess_release(report, findings).recommendation == Recommendation.REVIEW
    strict = assess_release(report, findings, block_on_any_failure=True)
    assert strict.recommendation == Recommendation.BLOCK


def test_clean_report_passes_with_nonzero_test_evidence_and_limitations():
    report = parse_junit('<testsuite><testcase name="test_ok"/></testsuite>')
    result = assess_release(report, [])
    assert result.recommendation == Recommendation.PASS
    assert result.limitations


def test_test_selection_ranks_changes_history_and_criticality_explainably():
    report = parse(build=BuildMetadata(
        changed_files=["src/payments.py"],
        available_tests=["tests.test_payments::test_refund", "tests.test_auth::test_login"],
        test_module_map={"tests.test_payments::test_refund": ["src/payments.py"]},
        history={"tests.test_payments::test_refund": {"runs": 10, "failures": 2}},
        critical_tests=["tests.test_auth::test_login"],
    ))
    results = recommend_tests(report)
    ids = [item["test_id"] for item in results]
    assert ids.index("tests.test_payments::test_refund") < ids.index("tests.test_auth::test_login")
    assert any("explicit module map" in reason for reason in results[0]["reasons"])
    assert all(item["reasons"] for item in results)


def test_test_selection_uses_documented_baseline_when_metadata_missing():
    report = parse_junit('<testsuite><testcase name="test_one"/><testcase name="test_two"/></testsuite>')
    results = recommend_tests(report)
    assert len(results) == 2
    assert all("baseline fallback" in item["reasons"][0] for item in results)


def test_llm_output_schema_rejects_incomplete_or_invalid_confidence():
    with pytest.raises(ValidationError):
        LLMOutput.model_validate({"summary": "partial"})
    with pytest.raises(ValidationError):
        LLMOutput.model_validate({"summary": "x", "root_cause": "x", "evidence_refs": ["x"], "confidence": 2, "remediation_proposal": "x"})


def _llm_response(ref: str) -> bytes:
    content = {
        "summary": "The failure is likely in the application handler.",
        "root_cause": "A runtime defect may have caused the observed error.",
        "alternatives": [],
        "evidence_refs": [ref],
        "debugging_steps": ["Reproduce the test."],
        "remediation_proposal": "Inspect the handler; proposed, not verified.",
        "confidence": 0.6,
    }
    return json.dumps({"choices": [{"message": {"content": json.dumps(content)}}]}).encode()


def test_openai_compatible_analysis_accepts_valid_citations_and_rejects_fabricated():
    finding = classify_failures(parse_junit(failure_xml()))[0]
    analyzer = OpenAICompatibleAnalyzer("http://localhost:11434/v1", "local")
    with patch("app.triage.analysis.urlopen", return_value=BytesIO(_llm_response("app.feature::test_failure:message"))):
        enriched = analyzer.enrich(finding)
    assert enriched.analysis_mode == "llm"
    assert enriched.root_cause.verified is False
    with patch("app.triage.analysis.urlopen", return_value=BytesIO(_llm_response("fabricated:line-99"))), pytest.raises(RuntimeError, match="unknown evidence"):
        analyzer.enrich(finding)


def test_provider_outage_returns_safe_error_for_deterministic_service_fallback():
    finding = classify_failures(parse_junit(failure_xml()))[0]
    analyzer = OpenAICompatibleAnalyzer("http://localhost:11434/v1", "local")
    with patch("app.triage.analysis.urlopen", side_effect=URLError("offline")), pytest.raises(RuntimeError, match="LLM analysis unavailable"):
        analyzer.enrich(finding)
    response = TriageService(analyzer=UnavailableAnalyzer()).analyze(failure_xml())
    assert response.findings[0].analysis_mode == "deterministic"
    assert "offline" not in response.warnings[0]


class UnavailableAnalyzer:
    name = "mock-provider"

    def enrich(self, finding):
        raise RuntimeError("secret endpoint is unavailable")


def test_service_attaches_log_lines_and_falls_back_on_fabricated_refs():
    class FabricatedAnalyzer:
        name = "mock-provider"

        def enrich(self, finding):
            finding.root_cause.evidence_refs = ["fake:999"]
            return finding

    response = TriageService(analyzer=FabricatedAnalyzer()).analyze(
        failure_xml(), logs="startup\ntest_failure returned 500"
    )
    finding = response.findings[0]
    assert finding.analysis_mode == "deterministic"
    assert finding.evidence[-1].reference == "application_logs:line-2"
    assert "secret endpoint" not in str(response.warnings)


def test_api_adds_triage_routes_and_keeps_prediction_routes():
    previous = app.state.triage_service
    app.state.triage_service = TriageService()
    try:
        client = TestClient(app)
        response = client.post("/qa/analyze", json={"junit_xml": FIXTURE.read_text(), "run_id": "test-run"})
        assert response.status_code == 200
        assert response.json()["release"]["recommendation"] == "REVIEW"
        assert client.get("/qa/runs/test-run").status_code == 200
        assert client.get("/qa/runs/test-run/release").status_code == 200
        assert client.get("/qa/runs/test-run/test-selection").status_code == 200
        assert client.get("/qa/runs/unknown").status_code == 404
        assert client.get("/model-info").status_code == 200
    finally:
        app.state.triage_service = previous


def test_api_rejects_malformed_and_sensitive_invalid_input_safely():
    client = TestClient(app)
    assert client.post("/qa/analyze", json={"junit_xml": "<bad"}).status_code == 400
    response = client.post("/qa/analyze", json={"junit_xml": "<testsuite><testcase name='ok'/></testsuite>", "run_id": "SECRET\nVALUE"})
    assert response.status_code == 422
    assert "SECRET" not in response.text


def test_synthetic_evaluation_reports_all_case_results_and_limitations():
    metrics = evaluate()
    assert metrics["case_count"] == 9
    assert metrics["case_failure_rate"] == 0.0
    assert metrics["test_selection_metrics"] is None
    assert len(metrics["results"]) == 9
