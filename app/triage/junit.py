"""Bounded parser for Pytest JUnit XML with explicit source evidence."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections import defaultdict

from .models import BuildMetadata, Evidence, NormalizedReport, Outcome, TestResult

MAX_REPORT_BYTES = 10_000_000
MAX_LOG_CHARS = 2_000_000


class ReportError(ValueError):
    """Invalid, unsafe, or empty JUnit report."""


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "\n".join(line.rstrip() for line in "".join(element.itertext()).splitlines()).strip()
    return value or None


def _duration(value: str | None) -> float:
    try:
        duration = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return duration if math.isfinite(duration) and duration >= 0 else 0.0


def parse_junit(
    xml_text: str,
    *,
    run_id: str | None = None,
    application_logs: str = "",
    build: BuildMetadata | None = None,
) -> NormalizedReport:
    if len(xml_text.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ReportError(f"JUnit XML exceeds {MAX_REPORT_BYTES} bytes")
    if not xml_text.strip():
        raise ReportError("JUnit XML is empty")
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        raise ReportError("DTD and entity declarations are not allowed")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ReportError(f"Malformed JUnit XML: {exc}") from exc
    elements = [element for element in root.iter() if _tag(element) == "testcase"]
    if not elements:
        raise ReportError("JUnit XML contains no testcase elements")

    duplicates: defaultdict[str, int] = defaultdict(int)
    cases: list[TestResult] = []
    warnings: list[str] = []
    for position, element in enumerate(elements, start=1):
        name = (element.get("name") or "").strip()
        classname = (element.get("classname") or "").strip() or None
        if not name:
            name = f"unnamed-test-{position}"
            warnings.append(f"Test case at position {position} had no name; assigned {name}.")
        original_id = f"{classname}::{name}" if classname else name
        duplicates[original_id] += 1
        duplicate_index = duplicates[original_id]
        test_id = original_id if duplicate_index == 1 else f"{original_id}#duplicate-{duplicate_index}"
        result_node = next((child for child in element if _tag(child) in {"failure", "error", "skipped"}), None)
        if result_node is None:
            outcome = Outcome.PASSED
        else:
            outcome = {"failure": Outcome.FAILED, "error": Outcome.ERROR, "skipped": Outcome.SKIPPED}[_tag(result_node)]
        message = (result_node.get("message") if result_node is not None else None) or _element_text(result_node)
        stack = _element_text(result_node)
        exception = result_node.get("type") if result_node is not None and outcome in {Outcome.FAILED, Outcome.ERROR} else None
        source_file = element.get("file")
        source_line: int | None = None
        line_value = element.get("line")
        for prop in element.iter():
            if _tag(prop) == "property":
                key, value = prop.get("name", ""), prop.get("value", "")
                if key in {"file", "filename", "path"}:
                    source_file = source_file or value
                if key in {"line", "lineno"}:
                    line_value = line_value or value
        try:
            source_line = int(line_value) if line_value else None
        except ValueError:
            source_line = None

        evidence: list[Evidence] = []
        if message:
            evidence.append(Evidence(reference=f"{test_id}:message", kind="test_message", text=message[:8000]))
        if stack:
            evidence.append(Evidence(reference=f"{test_id}:traceback", kind="stack_trace", text=stack[:20_000]))
            for line_number, line in enumerate(stack.splitlines(), start=1):
                if line.strip():
                    evidence.append(Evidence(reference=f"{test_id}:trace-line-{line_number}", kind="stack_trace_line", text=line[:2000], line_start=line_number, line_end=line_number))
        if source_file:
            evidence.append(Evidence(reference=f"{test_id}:source", kind="test_source", text=source_file, line_start=source_line, line_end=source_line))
        cases.append(TestResult(
            test_id=test_id,
            original_id=original_id,
            name=name,
            classname=classname,
            outcome=outcome,
            duration_seconds=_duration(element.get("time")),
            message=message[:8000] if message else None,
            exception_type=exception,
            stack_trace=stack[:20_000] if stack else None,
            source_file=source_file,
            source_line=source_line,
            duplicate_index=duplicate_index,
            evidence=evidence,
        ))

    if len(application_logs) > MAX_LOG_CHARS:
        application_logs = application_logs[-MAX_LOG_CHARS:]
        warnings.append(f"Application logs exceeded {MAX_LOG_CHARS} characters; retained the tail.")
    identifier = run_id or hashlib.sha256(xml_text.encode("utf-8")).hexdigest()[:16]
    return NormalizedReport(run_id=identifier, test_cases=cases, application_logs=application_logs, build=build or BuildMetadata(), warnings=warnings)
