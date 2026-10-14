"""Run the test suite and save a factual test, coverage, and model report."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    junit_path = REPORTS / "junit.xml"
    coverage_path = REPORTS / "coverage.xml"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={junit_path}",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True)
    test_root = ET.parse(junit_path).getroot()
    suite = test_root if test_root.tag == "testsuite" else test_root.find("testsuite")
    tests = int(suite.attrib.get("tests", 0))
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    coverage_root = ET.parse(coverage_path).getroot()
    line_rate = float(coverage_root.attrib.get("line-rate", 0))

    # Build the metrics report only after tests have passed.
    model_metrics = {}
    regression_result = None
    if completed.returncode == 0:
        from validation.model_validation import validate_model_file
        from validation.regression_validation import compare_with_baseline, train_candidate_model

        model_result = validate_model_file()
        regression_result = compare_with_baseline(train_candidate_model())
        model_metrics = model_result.metrics
        if not model_result.is_valid or not regression_result["is_valid"]:
            completed.returncode = 1
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests": {"total": tests, "passed": tests - failures - errors - skipped, "failed": failures + errors, "skipped": skipped},
        "coverage_percent": round(line_rate * 100, 2),
        "model_validation": model_result.as_dict() if completed.returncode == 0 or model_metrics else None,
        "regression": regression_result,
        "quality_gate": "PASSED" if completed.returncode == 0 else "FAILED",
    }
    (REPORTS / "test_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nAI/ML QA Test Report")
    print("-------------------")
    print(f"Tests: {report['tests']['passed']} passed, {report['tests']['failed']} failed, {skipped} skipped ({tests} total)")
    print(f"Coverage: {report['coverage_percent']:.2f}%")
    if regression_result:
        print(f"Baseline F1: {regression_result['baseline']['f1']:.3f}")
        print(f"Current F1:  {regression_result['candidate']['f1']:.3f}")
    print(f"Quality gate: {report['quality_gate']}")
    print(f"JSON report: {REPORTS / 'test_report.json'}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
