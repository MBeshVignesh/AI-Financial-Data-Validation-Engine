from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from hierarchy_migration_validation_agent import __version__
from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.config import get_settings
from hierarchy_migration_validation_agent.schemas import (
    ExceptionReport,
    HealthResponse,
    IngestionResponse,
    ValidationRequest,
    ValidationResponse,
)
from hierarchy_migration_validation_agent.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
workflow = ValidationWorkflow(settings)

app = FastAPI(title=settings.app_name, version=__version__)


@app.post("/ingest-excel", response_model=IngestionResponse)
async def ingest_excel(
    source_files: list[UploadFile] | None = File(default=None),
    target_files: list[UploadFile] | None = File(default=None),
) -> IngestionResponse:
    saved_source_paths: list[Path] = []
    saved_target_paths: list[Path] = []
    if source_files:
        for uploaded_file in source_files:
            destination = settings.source_dir / uploaded_file.filename
            contents = await uploaded_file.read()
            destination.write_bytes(contents)
            saved_source_paths.append(destination)
    if target_files:
        for uploaded_file in target_files:
            destination = settings.target_upload_dir / uploaded_file.filename
            contents = await uploaded_file.read()
            destination.write_bytes(contents)
            saved_target_paths.append(destination)
    return workflow.ingest_excel_files(
        source_file_paths=saved_source_paths or None,
        target_file_paths=saved_target_paths or None,
        auto_build_index=True,
    )


@app.post("/build-index")
def build_index() -> dict[str, int | str]:
    return workflow.build_index()


@app.post("/validate", response_model=ValidationResponse)
def validate(validation_request: ValidationRequest) -> ValidationResponse:
    report = workflow.validate(validation_request)
    return ValidationResponse(
        run_id=report.run_id,
        overall_status=report.overall_status,
        summary=report.summary,
        json_report_path=report.json_report_path or "",
        markdown_report_path=report.markdown_report_path or "",
        agent_explanation=report.agent_explanation,
    )


@app.get("/validation-report/{run_id}", response_model=ExceptionReport)
def validation_report(run_id: str) -> ExceptionReport:
    try:
        return workflow.get_validation_report(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        generated_reports=workflow.repository.list_run_count(),
    )


def run() -> None:
    uvicorn.run(
        "hierarchy_migration_validation_agent.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
