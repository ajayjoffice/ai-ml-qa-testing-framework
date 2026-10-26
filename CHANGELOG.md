# Changelog

Notable project updates are recorded here.

## 2.0.0 — 2026-10-26

### Added

- Added a test failure triage API that accepts bounded Pytest JUnit XML and optional build metadata.
- Added deterministic failure grouping and classification for application, regression, test, environment, infrastructure, potential flaky, and insufficient-evidence cases.
- Added evidence-linked findings, severity and priority, an auditable `PASS` / `REVIEW` / `BLOCK` release recommendation, and explainable test-selection suggestions.
- Added optional OpenAI-compatible LLM enrichment with validated structured output, evidence-reference checks, human-review labeling, and deterministic fallback when the provider is unavailable or invalid.
- Added a synthetic triage evaluation command and CI artifact, with clear caveats that fixture scores are not production accuracy estimates.
- Expanded automated coverage with parser, classification, service, API, provider-fallback, release-gate, and evaluation tests.
- Expanded the README with setup, configuration, API usage, evaluation, architecture, and limitations for triage.

### Preserved

- Existing churn prediction, data validation, model evaluation, regression gates, and GitHub Actions workflow remain available.
- The model's recorded holdout metrics remain ROC-AUC 0.922 and F1 0.618.
