from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class FileArtifact(BaseModel):
    name: str
    path: str
    row_count: int | None = None


class HierarchyNode(BaseModel):
    dimension: str
    member_name: str
    parent_name: str | None = None
    level: int
    leaf_flag: bool
    sort_order: int
    member_code: str | None = None
    member_description: str | None = None
    source_system: str = "Oracle Smart View"


class MappingRecord(BaseModel):
    dimension: str
    source_member_name: str
    target_member_name: str
    mapping_status: str = "active"
    mapping_rule: str = "exact_match"
    notes: str | None = None


class ValidationRule(BaseModel):
    rule_id: str
    dimension: str
    rule_name: str
    check_type: str
    severity: str = "medium"
    enabled: bool = True
    optional: bool = False
    description: str
    business_rationale: str


class RagDocument(BaseModel):
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureRecord(BaseModel):
    dimension: str
    rule_name: str
    member_name: str | None = None
    issue: str
    source_value: str | None = None
    target_value: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    rule_id: str
    rule_name: str
    check_type: str
    dimension: str
    severity: str
    status: str
    passed_count: int
    failed_count: int
    failed_records: list[FailureRecord] = Field(default_factory=list)
    retrieved_context: list[str] = Field(default_factory=list)
    likely_cause: str
    recommended_action: str


class ExceptionReport(BaseModel):
    run_id: str
    created_at: datetime
    request: str
    overall_status: str
    dimensions: list[str]
    summary: dict[str, Any]
    results: list[ValidationResult]
    retrieved_context: list[str] = Field(default_factory=list)
    likely_root_causes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    agent_explanation: str = ""
    json_report_path: str | None = None
    markdown_report_path: str | None = None


class ValidationRequest(BaseModel):
    message: str = "Validate Oracle Smart View account and entity hierarchies against ADLS target"
    dimensions: list[str] | None = None
    rebuild_index: bool = False


class ValidationResponse(BaseModel):
    run_id: str
    overall_status: str
    summary: dict[str, Any]
    json_report_path: str
    markdown_report_path: str
    agent_explanation: str


class IngestionResponse(BaseModel):
    ingested_at: datetime
    normalized_files: list[FileArtifact]
    source_files: list[str]
    target_files: list[str] = Field(default_factory=list)
    rag_document_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    generated_reports: int
