from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.normalization.hierarchy_normalizer import HierarchyNormalizer
from hierarchy_migration_validation_agent.schemas import FileArtifact, IngestionResponse
from hierarchy_migration_validation_agent.utils.text import to_snake_case

LOGGER = logging.getLogger(__name__)


@dataclass
class ParsedWorkbookPayload:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: list[FileArtifact] = field(default_factory=list)
    uploaded_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ExcelIngestionService:
    SOURCE_WORKBOOK_TYPES = {
        "account_hierarchy_source.xlsx": "account_source",
        "entity_hierarchy_source.xlsx": "entity_source",
        "hierarchy_mapping.xlsx": "mapping",
        "validation_rules.xlsx": "rules",
    }
    TARGET_WORKBOOK_TYPES = {
        "account_hierarchy_target.xlsx": "account_target",
        "entity_hierarchy_target.xlsx": "entity_target",
        "dim_account_hierarchy.xlsx": "account_target",
        "dim_entity_hierarchy.xlsx": "entity_target",
    }
    OUTPUT_ORDER_BY_ROLE = {
        "source": [
            "account_source",
            "entity_source",
            "account_measure_source",
            "entity_measure_source",
            "mapping",
            "rules",
        ],
        "target": ["account_target", "entity_target", "account_measure_target", "entity_measure_target"],
    }
    ACCOUNT_TOKENS = {
        "account",
        "revenue",
        "expense",
        "gl",
        "coa",
        "subscription",
        "license",
        "asset",
        "liability",
        "equity",
        "income",
        "ledger",
        "cogs",
    }
    ENTITY_TOKENS = {
        "entity",
        "corp",
        "business",
        "region",
        "europe",
        "america",
        "bu",
        "company",
        "division",
        "department",
        "cost_center",
        "team",
        "function",
        "location",
        "branch",
        "country",
        "market",
    }
    SHEET_HIERARCHY_HINTS = {"hierarchy", "org", "spine", "structure", "tree", "rollup", "parent", "node"}
    HIERARCHY_NAME_TOKENS = {
        "entity",
        "business_unit",
        "department",
        "cost_center",
        "region",
        "division",
        "function",
        "team",
        "company",
        "legal_entity",
        "organization",
        "org",
        "unit",
        "group",
        "branch",
        "country",
        "market",
    }
    EXCLUDED_GENERIC_HIERARCHY_TOKENS = {
        "default",
        "typical",
        "metric",
        "value",
        "snapshot",
        "load",
        "owner",
        "refresh",
        "worker",
        "employee",
        "supervisor",
        "manager",
        "salary",
        "hourly",
        "annual",
        "bonus",
        "currency",
        "status",
        "date",
        "timestamp",
        "meaning",
        "example",
        "type",
        "pattern",
        "risk",
        "population",
        "grade",
        "job",
        "performance",
        "project",
        "code",
        "location",
        "name",
        "roster",
        "dictionary",
        "field",
        "dashboard",
    }
    PATH_COLUMN_ORDER = ["entity", "business_unit", "department", "cost_center", "account"]
    STRUCTURAL_NUMERIC_COLUMNS = {
        "level",
        "lvl",
        "depth",
        "hierarchy_level",
        "generation",
        "level_number",
        "sort_order",
        "sort",
        "sequence",
        "order",
        "position",
        "leaf_flag",
        "is_leaf",
        "leaf",
        "terminal",
        "is_terminal",
        "leaf_node",
    }
    EXCLUDED_PATH_TOKENS = {
        "employee",
        "manager",
        "salary",
        "currency",
        "status",
        "amount",
        "total",
        "grade",
        "title",
        "email",
        "phone",
        "description",
        "note",
        "comment",
        "address",
    }
    MAPPING_SOURCE_COLUMNS = {
        "source_member_name",
        "source_member",
        "source_value",
        "source_member_value",
        "source_hierarchy_member",
        "smartview_member",
        "from_member",
    }
    MAPPING_TARGET_COLUMNS = {
        "target_member_name",
        "target_member",
        "target_value",
        "target_member_value",
        "target_hierarchy_member",
        "curated_member",
        "adls_member",
        "to_member",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.normalizer = HierarchyNormalizer()

    def ingest_directory(self, source_dir: Path | None = None, *, role: str = "source") -> IngestionResponse:
        directory = source_dir or (self.settings.source_dir if role == "source" else self.settings.target_upload_dir)
        files = sorted(directory.glob("*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No Excel workbooks found in {directory}")
        payload = self.parse_files(files, role=role)
        return IngestionResponse(
            ingested_at=datetime.now(timezone.utc),
            normalized_files=payload.artifacts,
            source_files=payload.uploaded_files if role == "source" else [],
            target_files=payload.uploaded_files if role == "target" else [],
            warnings=payload.warnings,
        )

    def parse_files(self, files: list[Path], *, role: str = "source") -> ParsedWorkbookPayload:
        output_order = self.OUTPUT_ORDER_BY_ROLE[role]
        workbook_types = self.SOURCE_WORKBOOK_TYPES if role == "source" else self.TARGET_WORKBOOK_TYPES
        collected_frames, uploaded_files, warnings = self._collect_frames(
            files,
            role=role,
            output_order=output_order,
            workbook_types=workbook_types,
        )

        parsed_frames: dict[str, pd.DataFrame] = {}
        artifacts: list[FileArtifact] = []
        for workbook_type in output_order:
            frames = collected_frames[workbook_type]
            if not frames:
                continue
            combined_frame = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
            normalized_frame = self._normalize_workbook(combined_frame, workbook_type)
            normalized_frame = self._deduplicate_normalized_frame(normalized_frame, workbook_type)
            parsed_frames[workbook_type] = normalized_frame
            artifacts.append(
                FileArtifact(
                    name=workbook_type,
                    path=f"in-memory://{workbook_type}",
                    row_count=len(normalized_frame),
                )
            )

        if not artifacts:
            raise FileNotFoundError(
                f"No supported {role} hierarchy, mapping, or rule sheets were found. "
                "Expected either standard hierarchy columns like member_name / parent_name with optional level / leaf_flag, "
                "or path-style hierarchy sheets like Level 1 / Level 2 / Level 3, "
                "or ordered hierarchy columns such as Entity / Business Unit / Department / Cost Center "
                "or Company / Division / Team, or parent-child aliases such as Child / Parent / Depth."
            )

        return ParsedWorkbookPayload(
            frames=parsed_frames,
            artifacts=artifacts,
            uploaded_files=uploaded_files,
            warnings=warnings,
        )

    def ingest_files(self, files: list[Path], *, role: str = "source") -> tuple[list[FileArtifact], list[str], list[str]]:
        payload = self.parse_files(files, role=role)
        return payload.artifacts, payload.uploaded_files, payload.warnings

    def _collect_frames(
        self,
        files: list[Path],
        *,
        role: str,
        output_order: list[str],
        workbook_types: dict[str, str],
    ) -> tuple[dict[str, list[pd.DataFrame]], list[str], list[str]]:
        collected_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in output_order}
        uploaded_files: list[str] = []
        warnings: list[str] = []
        for path in files:
            extracted_frames = self._extract_workbook_frames(path, role=role, workbook_types=workbook_types)
            if not extracted_frames:
                warning = f"Skipping workbook with no recognized {role} sheets: {path.name}"
                LOGGER.warning(warning)
                warnings.append(warning)
                continue

            for workbook_type, frame in extracted_frames:
                LOGGER.info("Ingesting workbook %s as %s", path.name, workbook_type)
                collected_frames[workbook_type].append(frame)
            uploaded_files.append(str(path))
        return collected_frames, uploaded_files, warnings

    def _extract_workbook_frames(
        self,
        path: Path,
        *,
        role: str,
        workbook_types: dict[str, str],
    ) -> list[tuple[str, pd.DataFrame]]:
        expected_type = workbook_types.get(path.name.lower())
        if expected_type:
            frame = self._read_sheet(path, 0)
            return [(expected_type, frame)]

        workbook = pd.ExcelFile(path)
        extracted_frames: list[tuple[str, pd.DataFrame]] = []
        for sheet_name in workbook.sheet_names:
            frame = self._read_sheet(path, sheet_name)
            extracted_frames.extend(self._infer_sheet_types(frame, sheet_name, role=role))
        return extracted_frames

    def _read_sheet(self, path: Path, sheet_name: str | int) -> pd.DataFrame:
        raw_frame = pd.read_excel(path, sheet_name=sheet_name, header=None)
        return self._prepare_sheet_frame(raw_frame)

    def _prepare_sheet_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        prepared = raw_frame.dropna(how="all").reset_index(drop=True)
        if prepared.empty:
            return pd.DataFrame()

        header_row_index = self._detect_header_row_index(prepared)
        header_values = prepared.iloc[header_row_index].tolist()
        headers = self._normalize_headers(header_values)
        data = prepared.iloc[header_row_index + 1 :].copy()
        data.columns = headers
        data = data.dropna(how="all").reset_index(drop=True)
        data = data.infer_objects(copy=False)
        return data

    def _detect_header_row_index(self, frame: pd.DataFrame) -> int:
        if self._looks_like_primary_header_row(frame.iloc[0].tolist(), len(frame.columns)):
            return 0
        candidate_limit = min(5, len(frame))
        scored_rows = [
            (self._header_candidate_score(frame.iloc[index].tolist()), index)
            for index in range(1, candidate_limit)
        ]
        if not scored_rows:
            return 0
        scored_rows.sort(reverse=True)
        return scored_rows[0][1]

    def _looks_like_primary_header_row(self, values: list[object], total_columns: int) -> bool:
        texts = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
        if len(texts) < 2:
            return False
        non_null_ratio = len(texts) / max(total_columns, 1)
        if len(texts) == 1 or non_null_ratio < 0.5:
            return False
        data_like_count = sum(self._looks_like_data_value(text) for text in texts)
        return data_like_count <= max(1, len(texts) // 3)

    def _header_candidate_score(self, values: list[object]) -> int:
        texts = [str(value).strip() for value in values if pd.notna(value) and str(value).strip()]
        if len(texts) < 2:
            return -1000

        keyword_hits = 0
        short_alpha_count = 0
        numeric_like_count = 0
        for text in texts:
            normalized = to_snake_case(text)
            if any(token in normalized for token in self.HIERARCHY_NAME_TOKENS | self.SHEET_HIERARCHY_HINTS):
                keyword_hits += 1
            if any(char.isalpha() for char in text) and len(text) <= 40:
                short_alpha_count += 1
            if self._looks_like_data_value(text):
                numeric_like_count += 1

        return (keyword_hits * 5) + (short_alpha_count * 2) + len(texts) - (numeric_like_count * 3)

    @staticmethod
    def _looks_like_data_value(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if re.fullmatch(r"\d+(\.\d+)?", stripped):
            return True
        if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?( \d{2}:\d{2}:\d{2})?", stripped):
            return True
        if re.fullmatch(r"[A-Z]{1,6}[-_]?\d{2,}", stripped):
            return True
        return False

    @staticmethod
    def _normalize_headers(values: list[object]) -> list[str]:
        headers: list[str] = []
        counts: dict[str, int] = {}
        for index, value in enumerate(values):
            base = to_snake_case(str(value)) if pd.notna(value) and str(value).strip() else f"unnamed_{index}"
            if not base:
                base = f"unnamed_{index}"
            suffix = counts.get(base, 0)
            counts[base] = suffix + 1
            header = base if suffix == 0 else f"{base}_{suffix}"
            headers.append(header)
        return headers

    def _infer_sheet_types(
        self,
        frame: pd.DataFrame,
        sheet_name: str,
        *,
        role: str,
    ) -> list[tuple[str, pd.DataFrame]]:
        standardized = frame.copy()
        standardized.columns = [to_snake_case(column) for column in frame.columns]
        normalized_columns = set(standardized.columns)
        sheet_key = to_snake_case(sheet_name)

        if role == "source" and self._looks_like_mapping_sheet(normalized_columns, sheet_key):
            return [("mapping", self._prepare_mapping_frame(standardized, sheet_key))]

        if role == "source" and self._looks_like_rules_sheet(normalized_columns):
            return [("rules", frame)]

        detected_types: list[tuple[str, pd.DataFrame]] = []

        measure_sheet = self._extract_measure_sheet(standardized, sheet_key, role=role)
        if measure_sheet is not None:
            detected_types.append(measure_sheet)

        path_hierarchy = self._extract_path_hierarchy_sheet(standardized, sheet_key, role=role)
        if path_hierarchy is not None:
            detected_types.append(path_hierarchy)
            return detected_types

        if not self._looks_like_hierarchy_sheet(normalized_columns):
            return detected_types

        if "dimension" in standardized.columns:
            dimension_series = standardized["dimension"].astype(str).str.strip().str.lower()
            for dimension in ["account", "entity"]:
                mask = dimension_series == dimension
                if mask.any():
                    detected_types.append(
                        (
                            f"{dimension}_{role}",
                            standardized.loc[mask].copy(),
                        )
                    )
            if detected_types:
                return detected_types

        inferred_dimension = self._infer_dimension_from_sheet(standardized, sheet_key)
        if inferred_dimension:
            detected_types.append((f"{inferred_dimension}_{role}", frame))
        return detected_types

    def _looks_like_mapping_sheet(self, normalized_columns: set[str], sheet_key: str) -> bool:
        has_source = any(column in normalized_columns for column in self.MAPPING_SOURCE_COLUMNS)
        has_target = any(column in normalized_columns for column in self.MAPPING_TARGET_COLUMNS)
        if has_source and has_target:
            return True
        return (
            any(token in sheet_key for token in {"mapping", "crosswalk", "xref"})
            and has_source
            and has_target
        )

    @staticmethod
    def _looks_like_rules_sheet(normalized_columns: set[str]) -> bool:
        return {"rule_id", "rule_name", "check_type"}.issubset(normalized_columns)

    @staticmethod
    def _looks_like_hierarchy_sheet(normalized_columns: set[str]) -> bool:
        return (
            any(
                column in normalized_columns
                for column in {"member_name", "member", "node", "node_name", "child", "child_name", "member_label"}
            )
            and any(column in normalized_columns for column in {"parent_name", "parent", "parent_member", "parent_node"})
        )

    def _infer_dimension_from_sheet(self, standardized: pd.DataFrame, sheet_key: str) -> str | None:
        if any(token in sheet_key for token in ("account", "gl", "coa")):
            return "account"
        if any(token in sheet_key for token in ("entity", "org", "legal")):
            return "entity"

        if "member_code" in standardized.columns:
            member_codes = standardized["member_code"].dropna().astype(str).str.upper()
            if member_codes.str.startswith("ACC_").any():
                return "account"
            if member_codes.str.startswith("ENT_").any():
                return "entity"

        if "member_name" in standardized.columns:
            member_names = " ".join(standardized["member_name"].dropna().astype(str).str.lower().tolist())
            account_hits = sum(token in member_names for token in self.ACCOUNT_TOKENS)
            entity_hits = sum(token in member_names for token in self.ENTITY_TOKENS)
            if account_hits > entity_hits and account_hits > 0:
                return "account"
            if entity_hits > account_hits and entity_hits > 0:
                return "entity"
            if account_hits == 0 and entity_hits == 0:
                return "entity"

        if any(
            column in standardized.columns
            for column in {"child", "child_name", "node", "node_name", "parent", "parent_name", "depth", "is_terminal"}
        ):
            return "entity"

        return None

    def _extract_path_hierarchy_sheet(
        self,
        standardized: pd.DataFrame,
        sheet_key: str,
        role: str,
    ) -> tuple[str, pd.DataFrame] | None:
        path_columns = self._detect_path_columns(standardized, sheet_key)
        if not path_columns:
            return None

        inferred_dimension = self._infer_dimension_from_path_columns(path_columns, sheet_key)
        if inferred_dimension is None:
            return None

        hierarchy_frame = self._path_columns_to_hierarchy(standardized, path_columns, inferred_dimension)
        if hierarchy_frame.empty:
            return None
        return f"{inferred_dimension}_{role}", hierarchy_frame

    def _prepare_mapping_frame(self, standardized: pd.DataFrame, sheet_key: str) -> pd.DataFrame:
        mapping_frame = standardized.copy()
        if "dimension" not in mapping_frame.columns:
            inferred_dimension = self._infer_mapping_dimension(mapping_frame, sheet_key)
            if inferred_dimension:
                mapping_frame["dimension"] = inferred_dimension
        return mapping_frame

    def _extract_measure_sheet(
        self,
        standardized: pd.DataFrame,
        sheet_key: str,
        role: str,
    ) -> tuple[str, pd.DataFrame] | None:
        inferred_dimension = self._infer_measure_dimension(standardized, sheet_key)
        if inferred_dimension is None:
            return None

        measure_columns = self._detect_measure_columns(standardized)
        if not measure_columns:
            return None

        measure_frame = standardized.copy()
        measure_frame["dataset_name"] = sheet_key
        return f"{inferred_dimension}_measure_{role}", measure_frame

    def _infer_measure_dimension(self, standardized: pd.DataFrame, sheet_key: str) -> str | None:
        path_columns = self._detect_path_columns(standardized, sheet_key)
        if path_columns:
            inferred_dimension = self._infer_dimension_from_path_columns(path_columns, sheet_key)
            if inferred_dimension:
                return inferred_dimension
        inferred_from_columns = self._infer_dimension_from_column_names(standardized.columns.tolist())
        if inferred_from_columns:
            return inferred_from_columns
        return self._infer_dimension_from_sheet(standardized, sheet_key)

    def _infer_dimension_from_column_names(self, columns: list[str]) -> str | None:
        combined = " ".join(columns)
        account_hits = sum(token in combined for token in self.ACCOUNT_TOKENS)
        entity_hits = sum(token in combined for token in self.ENTITY_TOKENS)
        if entity_hits > account_hits and entity_hits > 0:
            return "entity"
        if account_hits > entity_hits and account_hits > 0:
            return "account"
        return None

    def _detect_measure_columns(self, standardized: pd.DataFrame) -> list[str]:
        return [
            column
            for column in standardized.columns
            if column not in {"dimension", "dataset_name"}
            and column not in self.STRUCTURAL_NUMERIC_COLUMNS
            and pd.api.types.is_numeric_dtype(standardized[column])
        ]

    def _infer_mapping_dimension(self, standardized: pd.DataFrame, sheet_key: str) -> str | None:
        if any(token in sheet_key for token in ("account", "gl", "coa")):
            return "account"
        if any(token in sheet_key for token in ("entity", "org", "legal", "hcm")):
            return "entity"

        candidate_columns = [
            column
            for column in standardized.columns
            if column in self.MAPPING_SOURCE_COLUMNS | self.MAPPING_TARGET_COLUMNS
        ]
        if not candidate_columns:
            return None

        flattened = " ".join(
            value
            for column in candidate_columns
            for value in standardized[column].dropna().astype(str).str.lower().tolist()
        )
        account_hits = sum(token in flattened for token in self.ACCOUNT_TOKENS)
        entity_hits = sum(token in flattened for token in self.ENTITY_TOKENS)
        if account_hits > entity_hits and account_hits > 0:
            return "account"
        if entity_hits > account_hits and entity_hits > 0:
            return "entity"
        return None

    def _detect_path_columns(self, standardized: pd.DataFrame, sheet_key: str) -> list[str]:
        columns = list(standardized.columns)
        level_columns = []
        for column in columns:
            match = re.match(r"^level_(\d+)(?:_.+)?$", column)
            if match:
                level_columns.append((int(match.group(1)), column))
        if len(level_columns) >= 2:
            return [column for _, column in sorted(level_columns)]

        ordered = [column for column in self.PATH_COLUMN_ORDER if column in columns]
        if len(ordered) >= 3:
            return ordered
        if self._has_parent_child_structure(set(columns)):
            return []
        generic = self._detect_generic_path_columns(standardized, sheet_key)
        if len(generic) >= 2:
            return generic
        return []

    @staticmethod
    def _has_parent_child_structure(normalized_columns: set[str]) -> bool:
        member_columns = {"member_name", "member", "node", "node_name", "child", "child_name", "member_label"}
        parent_columns = {"parent_name", "parent", "parent_member", "parent_node", "parent_label"}
        return any(column in normalized_columns for column in member_columns) and any(
            column in normalized_columns for column in parent_columns
        )

    def _detect_generic_path_columns(self, standardized: pd.DataFrame, sheet_key: str) -> list[str]:
        candidate_columns: list[str] = []
        total_columns = len(standardized.columns)
        for column in standardized.columns:
            series = standardized[column]
            if self._is_measure_or_metadata_column(column, series):
                continue
            if not self._looks_like_categorical_path_column(series, total_columns):
                continue
            candidate_columns.append(column)
        if not self._sheet_name_suggests_hierarchy(sheet_key):
            return []
        hierarchy_named_columns = [column for column in candidate_columns if self._is_hierarchy_named_column(column)]
        if len(hierarchy_named_columns) >= 2:
            return hierarchy_named_columns
        if 2 <= len(candidate_columns) <= 5:
            return candidate_columns
        return []

    def _is_measure_or_metadata_column(self, column: str, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series) and column not in self.STRUCTURAL_NUMERIC_COLUMNS:
            return True
        lowered = column.lower()
        if any(token in lowered for token in self.EXCLUDED_PATH_TOKENS):
            return True
        if lowered.endswith("_id") and not any(token in lowered for token in {"member", "node", "account", "cost_center"}):
            return True
        return False

    def _is_hierarchy_named_column(self, column: str) -> bool:
        lowered = column.lower()
        if any(token in lowered for token in self.EXCLUDED_GENERIC_HIERARCHY_TOKENS):
            return False
        if any(token in lowered for token in self.HIERARCHY_NAME_TOKENS):
            return True
        return False

    def _sheet_name_suggests_hierarchy(self, sheet_key: str) -> bool:
        return any(token in sheet_key for token in self.SHEET_HIERARCHY_HINTS)

    @staticmethod
    def _looks_like_categorical_path_column(series: pd.Series, total_columns: int) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        if pd.api.types.is_numeric_dtype(non_null):
            return False
        unique_ratio = non_null.astype(str).nunique() / max(len(non_null), 1)
        return unique_ratio < 0.9 or total_columns <= 6

    def _infer_dimension_from_path_columns(self, path_columns: list[str], sheet_key: str) -> str | None:
        combined = " ".join([sheet_key, *path_columns])
        if "account" in combined or "gl" in combined or "coa" in combined:
            return "account"
        if any(token in combined for token in ["entity", "business_unit", "department", "cost_center", "org", "hcm"]):
            return "entity"
        return "entity"

    def _path_columns_to_hierarchy(
        self,
        standardized: pd.DataFrame,
        path_columns: list[str],
        dimension: str,
    ) -> pd.DataFrame:
        unique_nodes: dict[tuple[str, str | None, int], int] = {}
        sort_order = 1
        for row in standardized[path_columns].itertuples(index=False, name=None):
            values = []
            for value in row:
                if pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    values.append(text)
            for level, member_name in enumerate(values):
                parent_name = values[level - 1] if level > 0 else None
                key = (member_name, parent_name, level)
                if key not in unique_nodes:
                    unique_nodes[key] = sort_order
                    sort_order += 1

        if not unique_nodes:
            return pd.DataFrame()

        parent_names = {parent_name for _, parent_name, _ in unique_nodes if parent_name}
        rows = []
        for (member_name, parent_name, level), node_sort_order in unique_nodes.items():
            rows.append(
                {
                    "dimension": dimension,
                    "member_code": None,
                    "member_name": member_name,
                    "parent_name": parent_name,
                    "level": level,
                    "leaf_flag": member_name not in parent_names,
                    "sort_order": node_sort_order,
                    "member_description": None,
                    "source_system": "Oracle Smart View",
                }
            )
        return pd.DataFrame(rows)

    def _normalize_workbook(self, frame: pd.DataFrame, workbook_type: str) -> pd.DataFrame:
        if workbook_type == "account_source":
            normalized = self.normalizer.normalize_hierarchy(frame, "account")
        elif workbook_type == "entity_source":
            normalized = self.normalizer.normalize_hierarchy(frame, "entity")
        elif workbook_type == "account_target":
            normalized = self.normalizer.normalize_hierarchy(frame, "account")
        elif workbook_type == "entity_target":
            normalized = self.normalizer.normalize_hierarchy(frame, "entity")
        elif workbook_type == "account_measure_source":
            normalized = self.normalizer.normalize_measure(frame, "account", self._dataset_name(frame))
        elif workbook_type == "entity_measure_source":
            normalized = self.normalizer.normalize_measure(frame, "entity", self._dataset_name(frame))
        elif workbook_type == "account_measure_target":
            normalized = self.normalizer.normalize_measure(frame, "account", self._dataset_name(frame))
        elif workbook_type == "entity_measure_target":
            normalized = self.normalizer.normalize_measure(frame, "entity", self._dataset_name(frame))
        elif workbook_type == "mapping":
            default_dimension = None
            if "dimension" in frame.columns and frame["dimension"].notna().any():
                default_dimension = str(frame["dimension"].dropna().iloc[0]).strip().lower()
            normalized = self.normalizer.normalize_mapping(frame, default_dimension=default_dimension)
        elif workbook_type == "rules":
            normalized = self.normalizer.normalize_rules(frame)
        else:
            raise ValueError(f"Unsupported workbook type: {workbook_type}")
        return normalized

    @staticmethod
    def _deduplicate_normalized_frame(frame: pd.DataFrame, workbook_type: str) -> pd.DataFrame:
        if workbook_type in {"account_source", "entity_source", "account_target", "entity_target"}:
            deduplicated = frame.drop_duplicates(
                subset=["dimension", "member_name", "parent_name", "level"],
                keep="first",
            ).reset_index(drop=True)
            deduplicated["sort_order"] = range(1, len(deduplicated) + 1)
            return deduplicated
        return frame.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _dataset_name(frame: pd.DataFrame) -> str | None:
        if "dataset_name" in frame.columns and frame["dataset_name"].notna().any():
            return str(frame["dataset_name"].dropna().iloc[0]).strip()
        return None
