"""QA triage API router, integrated into the existing FastAPI application."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .analysis import DeterministicAnalyzer, OpenAICompatibleAnalyzer
from .junit import ReportError
from .models import AnalyzeRequest, ReleaseAssessment, TriageResponse
from .service import TriageService

LOG = logging.getLogger("app.triage")
router = APIRouter(prefix="/qa", tags=["QA triage"])


def make_service() -> TriageService:
    analyzer = OpenAICompatibleAnalyzer.from_environment() or DeterministicAnalyzer()
    block = os.getenv("QA_BLOCK_ON_ANY_FAILURE", "false").lower() in {"1", "true", "yes"}
    return TriageService(analyzer=analyzer, block_on_any_failure=block)


def get_service(request: Request) -> TriageService:
    return request.app.state.triage_service


@router.post("/analyze", response_model=TriageResponse)
def analyze(payload: AnalyzeRequest, request: Request) -> TriageResponse:
    try:
        result = get_service(request).analyze(payload.junit_xml, logs=payload.application_logs, build=payload.build, run_id=payload.run_id)
    except ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    LOG.info("qa_triage_complete run_id=%s findings=%d recommendation=%s", result.run_id, len(result.findings), result.release.recommendation.value)
    return result


@router.get("/runs/{run_id}", response_model=TriageResponse)
def get_run(run_id: str, request: Request) -> TriageResponse:
    result = get_service(request).get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Triage run not found")
    return result


@router.get("/runs/{run_id}/release", response_model=ReleaseAssessment)
def get_release(run_id: str, request: Request) -> ReleaseAssessment:
    result = get_run(run_id, request)
    return result.release


@router.get("/runs/{run_id}/test-selection")
def get_test_selection(run_id: str, request: Request) -> dict[str, object]:
    result = get_run(run_id, request)
    return {"run_id": run_id, "recommendations": result.selection}


async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Do not echo invalid payloads (which may contain diagnostic logs or secrets).
    errors = [{"loc": item.get("loc", []), "msg": item.get("msg", "Invalid input"), "type": item.get("type", "validation_error")} for item in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})
