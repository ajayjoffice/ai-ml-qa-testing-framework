"""Transparent test ranking using changes, explicit maps, history, criticality, and cost."""

from __future__ import annotations

from pathlib import PurePosixPath

from .models import NormalizedReport


def recommend_tests(report: NormalizedReport, limit: int = 20) -> list[dict[str, object]]:
    changed = [path.replace("\\", "/").strip("/").lower() for path in report.build.changed_files]
    by_id = {case.test_id: case for case in report.test_cases}
    by_id.update({case.original_id: case for case in report.test_cases})
    candidates = list(dict.fromkeys([case.test_id for case in report.test_cases] + report.build.available_tests))
    ranked: list[tuple[float, str, list[str]]] = []
    for test_id in candidates:
        case = by_id.get(test_id)
        original_id = case.original_id if case else test_id
        paths = report.build.test_module_map.get(test_id, report.build.test_module_map.get(original_id, []))
        paths = [path.replace("\\", "/").strip("/").lower() for path in paths]
        component = ((case.source_file or case.classname or "") if case else test_id).replace("\\", "/").lower()
        score = 0.0
        reasons: list[str] = []
        for path in changed:
            if not path:
                continue
            if path in paths or any(path.endswith(item) or item.endswith(path) for item in paths):
                score += 5.0
                reasons.append(f"explicit module map matches changed file {path}")
            elif component and (path in component or component.endswith(path)):
                score += 3.0
                reasons.append(f"test component matches changed file {path}")
            elif PurePosixPath(path).stem in test_id.lower():
                score += 1.5
                reasons.append(f"test ID mentions changed module {PurePosixPath(path).stem}")
        history = report.build.history.get(test_id, report.build.history.get(original_id, {}))
        runs = max(0, int(history.get("runs", 0)))
        failures = max(0, int(history.get("failures", 0)))
        if runs and failures:
            score += min(2.0, 2.0 * failures / runs)
            reasons.append(f"recorded failure history: {failures}/{runs}")
        if test_id in report.build.critical_tests or original_id in report.build.critical_tests:
            score += 4.0
            reasons.append("test is marked critical")
        duration = report.build.test_durations.get(test_id, report.build.test_durations.get(original_id, 0.0)) or (case.duration_seconds if case else 0.0)
        if duration > 0:
            score -= min(0.25, duration / 1000)
            reasons.append(f"known duration {duration:.2f}s (minor cost penalty)")
        if not reasons:
            score = 0.1
            reasons.append("baseline fallback: no change mapping or history available")
        ranked.append((score, test_id, reasons))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [{"test_id": test_id, "score": round(score, 3), "reasons": reasons} for score, test_id, reasons in ranked[:limit]]
