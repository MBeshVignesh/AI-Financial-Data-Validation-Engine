from __future__ import annotations

from dataclasses import dataclass, field
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hierarchy_migration_validation_agent.config import Settings, get_settings
from hierarchy_migration_validation_agent.ingestion.excel_ingestor import ExcelIngestionService, ParsedWorkbookPayload
from hierarchy_migration_validation_agent.rag.indexer import RagIndexService
from hierarchy_migration_validation_agent.reporting.report_writer import ReportWriter
from hierarchy_migration_validation_agent.schemas import (
    ExceptionReport,
    IngestionResponse,
    ValidationRequest,
    ValidationRule,
)
from hierarchy_migration_validation_agent.storage.repository import RunRepository
from hierarchy_migration_validation_agent.utils.ids import generate_run_id
from hierarchy_migration_validation_agent.utils.io import read_json, read_text
from hierarchy_migration_validation_agent.validation.checks import HierarchyValidator
from hierarchy_migration_validation_agent.validation.rule_catalog import default_rules_for_dimensions
from .reasoner import AgentReasoner

LOGGER = logging.getLogger(__name__)


@dataclass
class ActiveWorkbookContext:
    source_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    target_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_measure_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    target_measure_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    mapping_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rules_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_files: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    rag_collection_name: str = "hierarchy_context"
    cleanup_file_paths: list[str] = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ValidationWorkflow:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.ingestor = ExcelIngestionService(self.settings)
        self.rag_service = RagIndexService(self.settings)
        self.validator = HierarchyValidator()
        self.report_writer = ReportWriter(self.settings)
        self.repository = RunRepository(self.settings)
        self.reasoner = AgentReasoner(self.settings)
        self._active_context: ActiveWorkbookContext | None = None

    def ingest_excel_files(
        self,
        source_file_paths: list[Path] | None = None,
        target_file_paths: list[Path] | None = None,
        *,
        auto_build_index: bool = True,
    ) -> IngestionResponse:
        artifacts = []
        warnings: list[str] = []
        uploaded_source_files: list[str] = []
        uploaded_target_files: list[str] = []
        rag_collection_name = (
            self._active_context.rag_collection_name if self._active_context else self.rag_service.collection_name
        )
        cleanup_file_paths = list(self._active_context.cleanup_file_paths) if self._active_context else []
        source_frames = self._copy_frame_map(self._active_context.source_frames) if self._active_context else self._empty_frame_map()
        target_frames = self._copy_frame_map(self._active_context.target_frames) if self._active_context else self._empty_frame_map()
        source_measure_frames = (
            self._copy_frame_map(self._active_context.source_measure_frames) if self._active_context else self._empty_frame_map()
        )
        target_measure_frames = (
            self._copy_frame_map(self._active_context.target_measure_frames) if self._active_context else self._empty_frame_map()
        )
        mapping_df = self._active_context.mapping_df.copy() if self._active_context is not None else self._empty_mapping_frame()
        rules_df = self._active_context.rules_df.copy() if self._active_context is not None else pd.DataFrame()

        if source_file_paths is not None:
            rag_collection_name = generate_run_id("hierarchy_context")
            source_payload = self.ingestor.parse_files(
                source_file_paths,
                role="source",
            )
            artifacts.extend(source_payload.artifacts)
            warnings.extend(source_payload.warnings)
            uploaded_source_files = source_payload.uploaded_files
            cleanup_file_paths = self._merge_cleanup_paths(cleanup_file_paths, uploaded_source_files)
            source_frames = {
                "account": source_payload.frames.get("account_source", pd.DataFrame()),
                "entity": source_payload.frames.get("entity_source", pd.DataFrame()),
            }
            source_measure_frames = {
                "account": source_payload.frames.get("account_measure_source", pd.DataFrame()),
                "entity": source_payload.frames.get("entity_measure_source", pd.DataFrame()),
            }
            mapping_df = source_payload.frames.get("mapping", self._empty_mapping_frame())
            rules_df = source_payload.frames.get("rules", pd.DataFrame())

        if target_file_paths is not None:
            if source_file_paths is None:
                rag_collection_name = generate_run_id("hierarchy_context")
            target_payload = self.ingestor.parse_files(
                target_file_paths,
                role="target",
            )
            artifacts.extend(target_payload.artifacts)
            warnings.extend(target_payload.warnings)
            uploaded_target_files = target_payload.uploaded_files
            cleanup_file_paths = self._merge_cleanup_paths(cleanup_file_paths, uploaded_target_files)
            target_frames = {
                "account": target_payload.frames.get("account_target", pd.DataFrame()),
                "entity": target_payload.frames.get("entity_target", pd.DataFrame()),
            }
            target_measure_frames = {
                "account": target_payload.frames.get("account_measure_target", pd.DataFrame()),
                "entity": target_payload.frames.get("entity_measure_target", pd.DataFrame()),
            }

        if source_file_paths is None and target_file_paths is None:
            rag_collection_name = generate_run_id("hierarchy_context")
            source_payload = self._parse_directory_payload(role="source", raise_if_missing=True)
            target_payload = self._parse_directory_payload(role="target", raise_if_missing=True)
            artifacts.extend(source_payload.artifacts)
            artifacts.extend(target_payload.artifacts)
            warnings.extend(source_payload.warnings)
            warnings.extend(target_payload.warnings)
            uploaded_source_files = source_payload.uploaded_files
            uploaded_target_files = target_payload.uploaded_files
            source_frames = {
                "account": source_payload.frames.get("account_source", pd.DataFrame()),
                "entity": source_payload.frames.get("entity_source", pd.DataFrame()),
            }
            source_measure_frames = {
                "account": source_payload.frames.get("account_measure_source", pd.DataFrame()),
                "entity": source_payload.frames.get("entity_measure_source", pd.DataFrame()),
            }
            mapping_df = source_payload.frames.get("mapping", self._empty_mapping_frame())
            rules_df = source_payload.frames.get("rules", pd.DataFrame())
            target_frames = {
                "account": target_payload.frames.get("account_target", pd.DataFrame()),
                "entity": target_payload.frames.get("entity_target", pd.DataFrame()),
            }
            target_measure_frames = {
                "account": target_payload.frames.get("account_measure_target", pd.DataFrame()),
                "entity": target_payload.frames.get("entity_measure_target", pd.DataFrame()),
            }
            cleanup_file_paths = self._merge_cleanup_paths([], uploaded_source_files + uploaded_target_files)

        self.rag_service.set_collection_name(rag_collection_name)
        self._active_context = ActiveWorkbookContext(
            source_frames=source_frames,
            target_frames=target_frames,
            source_measure_frames=source_measure_frames,
            target_measure_frames=target_measure_frames,
            mapping_df=mapping_df,
            rules_df=rules_df,
            source_files=uploaded_source_files or (self._active_context.source_files if self._active_context else []),
            target_files=uploaded_target_files or (self._active_context.target_files if self._active_context else []),
            rag_collection_name=rag_collection_name,
            cleanup_file_paths=cleanup_file_paths,
            artifacts=artifacts,
            warnings=warnings,
        )

        rag_document_count = 0
        if auto_build_index:
            try:
                rag_document_count = int(self.build_index()["document_count"])
            except Exception as exc:
                LOGGER.warning("Automatic RAG build failed during ingest: %s", exc)
                warnings.append(f"Automatic RAG build failed during ingest: {exc}")

        return IngestionResponse(
            ingested_at=datetime.now(timezone.utc),
            normalized_files=artifacts,
            source_files=uploaded_source_files,
            target_files=uploaded_target_files,
            rag_document_count=rag_document_count,
            warnings=warnings,
        )

    def build_index(self) -> dict[str, Any]:
        source_frames = self._load_source_frames()
        target_frames = self._load_target_frames()
        source_measure_frames = self._load_source_measure_frames()
        target_measure_frames = self._load_target_measure_frames()
        available_dimensions = self._available_dimensions(
            source_frames,
            target_frames,
            source_measure_frames=source_measure_frames,
            target_measure_frames=target_measure_frames,
            union=True,
        )
        rules = self._load_rules(available_dimensions)
        documents = self.rag_service.build_index(
            mapping_df=self._load_mapping_frame(),
            rules_df=pd.DataFrame([rule.model_dump() for rule in rules]) if rules else pd.DataFrame(),
            source_frames=source_frames,
            target_frames=target_frames,
            transformation_notes=self._load_transformation_notes(),
            prior_exception_log=self._load_prior_exception_log(),
        )
        status = self.rag_service.storage_status()
        return {
            "collection_name": self.rag_service.collection_name,
            "document_count": len(documents),
            "client_mode": status["mode"],
            "index_path": status["path"],
            "index_file_count": status["file_count"],
        }

    def validate(self, validation_request: ValidationRequest) -> ExceptionReport:
        source_frames = self._load_source_frames()
        target_frames = self._load_target_frames()
        source_measure_frames = self._load_source_measure_frames()
        target_measure_frames = self._load_target_measure_frames()
        mapping_df = self._load_mapping_frame()
        available_dimensions = self._available_dimensions(
            source_frames,
            target_frames,
            source_measure_frames=source_measure_frames,
            target_measure_frames=target_measure_frames,
        )
        if not available_dimensions:
            raise ValueError(
                "No matching source and target datasets are loaded. "
                "Please ingest a source workbook and a target workbook first."
            )
        rules = self._load_rules(available_dimensions)

        if validation_request.rebuild_index or not self.rag_service.collection_exists():
            self.rag_service.build_index(
                mapping_df=mapping_df,
                rules_df=pd.DataFrame([rule.model_dump() for rule in rules]),
                source_frames=source_frames,
                target_frames=target_frames,
                transformation_notes=self._load_transformation_notes(),
                prior_exception_log=self._load_prior_exception_log(),
            )

        requested_dimensions = validation_request.dimensions or self._determine_dimensions(validation_request.message)
        dimensions = [dimension for dimension in requested_dimensions if dimension in available_dimensions] or available_dimensions
        results = []
        retrieved_context: list[str] = []

        for dimension in dimensions:
            LOGGER.info("Running validation for dimension=%s", dimension)
            dimension_rules = self._select_rules_for_dimension(
                dimension=dimension,
                rules=rules,
                mapping_df=mapping_df.loc[mapping_df["dimension"] == dimension] if not mapping_df.empty else mapping_df,
                source_df=source_frames[dimension],
                target_df=target_frames[dimension],
                source_measure_df=source_measure_frames[dimension],
                target_measure_df=target_measure_frames[dimension],
            )
            dimension_context = [
                document.content
                for document in self.rag_service.retrieve(
                    f"{validation_request.message}. Focus on {dimension} hierarchy validation rules and mappings.",
                    dimension=dimension,
                    n_results=4,
                )
            ]
            retrieved_context.extend(dimension_context)
            results.extend(
                self.validator.run_dimension_checks(
                    dimension=dimension,
                    source_df=source_frames[dimension],
                    target_df=target_frames[dimension],
                    source_measure_df=source_measure_frames[dimension],
                    target_measure_df=target_measure_frames[dimension],
                    mapping_df=mapping_df.loc[mapping_df["dimension"] == dimension],
                    rules=dimension_rules,
                    retrieved_context=dimension_context,
                )
            )

        summary = self._build_summary(results)
        report = ExceptionReport(
            run_id=generate_run_id("validation"),
            created_at=datetime.now(timezone.utc),
            request=validation_request.message,
            overall_status="FAILED" if summary["failed_checks"] else "PASSED",
            dimensions=dimensions,
            summary=summary,
            results=results,
            retrieved_context=self._deduplicate(retrieved_context)[:8],
            likely_root_causes=self._deduplicate([result.likely_cause for result in results if result.failed_count]),
            recommended_actions=self._deduplicate(
                [result.recommended_action for result in results if result.failed_count]
            ),
        )
        report.agent_explanation = self.reasoner.explain(report)
        report = self.report_writer.write(report)
        self.repository.save_report(report)
        return report

    def clear_runtime_state(
        self,
        *,
        clear_index: bool = False,
        clear_uploaded_files: bool = True,
        clear_supporting_docs: bool = False,
    ) -> None:
        if clear_uploaded_files:
            self._clear_directory_files(self.settings.source_dir)
            self._clear_directory_files(self.settings.target_upload_dir)
        if clear_supporting_docs:
            self._clear_directory_files(self.settings.supporting_dir)
        self._active_context = None
        if clear_index:
            self.rag_service.clear_index()

    def get_validation_report(self, run_id: str) -> ExceptionReport:
        report_paths = self.repository.get_report_paths(run_id)
        if report_paths is None:
            raise FileNotFoundError(f"Unknown validation run: {run_id}")
        json_path, _ = report_paths
        payload = read_json(json_path)
        return ExceptionReport.model_validate(payload)

    def preview_dataframes(self) -> dict[str, pd.DataFrame]:
        if self._active_context is not None:
            return {
                "source_account": self._active_context.source_frames.get("account", pd.DataFrame()),
                "source_entity": self._active_context.source_frames.get("entity", pd.DataFrame()),
                "target_account": self._active_context.target_frames.get("account", pd.DataFrame()),
                "target_entity": self._active_context.target_frames.get("entity", pd.DataFrame()),
                "source_account_measures": self._active_context.source_measure_frames.get("account", pd.DataFrame()),
                "source_entity_measures": self._active_context.source_measure_frames.get("entity", pd.DataFrame()),
                "target_account_measures": self._active_context.target_measure_frames.get("account", pd.DataFrame()),
                "target_entity_measures": self._active_context.target_measure_frames.get("entity", pd.DataFrame()),
            }
        source_frames = self._load_source_frames()
        target_frames = self._load_target_frames()
        source_measure_frames = self._load_source_measure_frames()
        target_measure_frames = self._load_target_measure_frames()
        return {
            "source_account": source_frames["account"],
            "source_entity": source_frames["entity"],
            "target_account": target_frames["account"],
            "target_entity": target_frames["entity"],
            "source_account_measures": source_measure_frames["account"],
            "source_entity_measures": source_measure_frames["entity"],
            "target_account_measures": target_measure_frames["account"],
            "target_entity_measures": target_measure_frames["entity"],
        }

    def _load_source_frames(self) -> dict[str, pd.DataFrame]:
        if self._active_context is not None:
            return self._copy_frame_map(self._active_context.source_frames)
        payload = self._parse_directory_payload(role="source", raise_if_missing=False)
        if payload is None:
            return self._empty_frame_map()
        return {
            "account": payload.frames.get("account_source", pd.DataFrame()),
            "entity": payload.frames.get("entity_source", pd.DataFrame()),
        }

    def _load_target_frames(self) -> dict[str, pd.DataFrame]:
        if self._active_context is not None:
            return self._copy_frame_map(self._active_context.target_frames)
        payload = self._parse_directory_payload(role="target", raise_if_missing=False)
        if payload is None:
            return self._empty_frame_map()
        return {
            "account": payload.frames.get("account_target", pd.DataFrame()),
            "entity": payload.frames.get("entity_target", pd.DataFrame()),
        }

    def _load_source_measure_frames(self) -> dict[str, pd.DataFrame]:
        if self._active_context is not None:
            return self._copy_frame_map(self._active_context.source_measure_frames)
        payload = self._parse_directory_payload(role="source", raise_if_missing=False)
        if payload is None:
            return self._empty_frame_map()
        return {
            "account": payload.frames.get("account_measure_source", pd.DataFrame()),
            "entity": payload.frames.get("entity_measure_source", pd.DataFrame()),
        }

    def _load_target_measure_frames(self) -> dict[str, pd.DataFrame]:
        if self._active_context is not None:
            return self._copy_frame_map(self._active_context.target_measure_frames)
        payload = self._parse_directory_payload(role="target", raise_if_missing=False)
        if payload is None:
            return self._empty_frame_map()
        return {
            "account": payload.frames.get("account_measure_target", pd.DataFrame()),
            "entity": payload.frames.get("entity_measure_target", pd.DataFrame()),
        }

    def _load_mapping_frame(self) -> pd.DataFrame:
        if self._active_context is not None:
            return self._active_context.mapping_df.copy()
        payload = self._parse_directory_payload(role="source", raise_if_missing=False)
        if payload is None:
            return self._empty_mapping_frame()
        return payload.frames.get("mapping", self._empty_mapping_frame()).copy()

    def _load_rules(self, dimensions: list[str]) -> list[ValidationRule]:
        if self._active_context is not None and not self._active_context.rules_df.empty:
            rules = [ValidationRule.model_validate(record) for record in self._active_context.rules_df.to_dict(orient="records")]
            filtered = [rule for rule in rules if rule.dimension in dimensions]
            if filtered:
                return filtered
        payload = self._parse_directory_payload(role="source", raise_if_missing=False)
        if payload is not None and "rules" in payload.frames:
            frame = payload.frames["rules"]
            rules = [ValidationRule.model_validate(record) for record in frame.to_dict(orient="records")]
            filtered = [rule for rule in rules if rule.dimension in dimensions]
            if filtered:
                return filtered
        return default_rules_for_dimensions(dimensions)

    def _load_transformation_notes(self) -> str:
        notes_path = self.settings.supporting_dir / "transformation_notes.md"
        return read_text(notes_path) if notes_path.exists() else ""

    def _load_prior_exception_log(self) -> pd.DataFrame:
        path = self.settings.supporting_dir / "prior_exception_log.csv"
        if not path.exists():
            return pd.DataFrame(columns=["run_date", "dimension", "rule_name", "member_name", "issue", "root_cause"])
        return pd.read_csv(path)

    @staticmethod
    def _merge_cleanup_paths(existing_paths: list[str], new_paths: list[str]) -> list[str]:
        merged = list(existing_paths)
        for path in new_paths:
            if path not in merged:
                merged.append(path)
        return merged

    @staticmethod
    def _clear_directory_files(directory: Path) -> None:
        for path in directory.glob("**/*"):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except Exception as exc:
                LOGGER.warning("Unable to remove file %s: %s", path, exc)

    @staticmethod
    def _determine_dimensions(message: str) -> list[str]:
        lowered = message.lower()
        dimensions = []
        if "account" in lowered:
            dimensions.append("account")
        if "entity" in lowered:
            dimensions.append("entity")
        return dimensions or ["account", "entity"]

    @staticmethod
    def _available_dimensions(
        source_frames: dict[str, pd.DataFrame],
        target_frames: dict[str, pd.DataFrame],
        *,
        source_measure_frames: dict[str, pd.DataFrame] | None = None,
        target_measure_frames: dict[str, pd.DataFrame] | None = None,
        union: bool = False,
    ) -> list[str]:
        source_dimensions = {dimension for dimension, frame in source_frames.items() if not frame.empty}
        target_dimensions = {dimension for dimension, frame in target_frames.items() if not frame.empty}
        if source_measure_frames is not None:
            source_dimensions.update(
                dimension for dimension, frame in source_measure_frames.items() if not frame.empty
            )
        if target_measure_frames is not None:
            target_dimensions.update(
                dimension for dimension, frame in target_measure_frames.items() if not frame.empty
            )
        available = source_dimensions | target_dimensions if union else source_dimensions & target_dimensions
        return sorted(available)

    def _select_rules_for_dimension(
        self,
        *,
        dimension: str,
        rules: list[ValidationRule],
        mapping_df: pd.DataFrame,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
    ) -> list[ValidationRule]:
        selected: list[ValidationRule] = []
        has_mapping = not mapping_df.empty and {"mapping_status", "source_member_name"}.issubset(mapping_df.columns)
        has_hierarchy = not source_df.empty or not target_df.empty
        has_measures = not source_measure_df.empty or not target_measure_df.empty
        for rule in rules:
            if rule.dimension != dimension or not rule.enabled:
                continue
            if rule.check_type in {
                "missing_members_in_target",
                "parent_existence",
                "parent_mismatch",
                "duplicate_members",
                "leaf_flag_consistency",
                "level_consistency",
                "row_level_match",
            } and not has_hierarchy:
                continue
            if rule.check_type == "mapping_completeness" and not has_mapping:
                continue
            if rule.check_type == "numeric_value_match" and not has_measures:
                continue
            if rule.check_type == "rollup_preservation" and (dimension != "account" or not has_hierarchy):
                continue
            selected.append(rule)
        return selected

    @staticmethod
    def _build_summary(results: list) -> dict[str, int]:
        failed_results = [result for result in results if result.failed_count]
        return {
            "total_checks": len(results),
            "passed_checks": len(results) - len(failed_results),
            "failed_checks": len(failed_results),
            "failed_records": sum(result.failed_count for result in results),
        }

    @staticmethod
    def _deduplicate(items: list[str]) -> list[str]:
        deduplicated: list[str] = []
        for item in items:
            if item and item not in deduplicated:
                deduplicated.append(item)
        return deduplicated

    @staticmethod
    def _copy_frame_map(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {
            "account": frames.get("account", pd.DataFrame()).copy(),
            "entity": frames.get("entity", pd.DataFrame()).copy(),
        }

    @staticmethod
    def _empty_frame_map() -> dict[str, pd.DataFrame]:
        return {"account": pd.DataFrame(), "entity": pd.DataFrame()}

    @staticmethod
    def _empty_mapping_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["dimension", "source_member_name", "target_member_name", "mapping_status", "mapping_rule", "notes"]
        )

    def _parse_directory_payload(self, *, role: str, raise_if_missing: bool) -> ParsedWorkbookPayload | None:
        directory = self.settings.source_dir if role == "source" else self.settings.target_upload_dir
        files = sorted(directory.glob("*.xlsx"))
        if not files:
            if raise_if_missing:
                raise FileNotFoundError(f"No Excel workbooks found in {directory}")
            return None
        try:
            return self.ingestor.parse_files(files, role=role)
        except FileNotFoundError:
            if raise_if_missing:
                raise
            return None
