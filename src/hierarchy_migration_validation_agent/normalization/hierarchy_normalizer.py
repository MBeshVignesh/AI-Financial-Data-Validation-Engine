from __future__ import annotations

from typing import Iterable

import pandas as pd

from hierarchy_migration_validation_agent.utils.text import coerce_bool, normalize_optional_str, to_snake_case


class HierarchyNormalizer:
    MEASURE_METADATA_COLUMNS = {"dimension", "dataset_name"}
    HIERARCHY_ALIASES = {
        "dimension": "dimension",
        "member_name": "member_name",
        "member": "member_name",
        "node": "member_name",
        "node_name": "member_name",
        "child": "member_name",
        "child_name": "member_name",
        "member_label": "member_name",
        "hierarchy_member": "member_name",
        "member_code": "member_code",
        "member_id": "member_code",
        "node_id": "member_code",
        "code": "member_code",
        "member_description": "member_description",
        "description": "member_description",
        "parent_name": "parent_name",
        "parent": "parent_name",
        "parent_member": "parent_name",
        "parent_node": "parent_name",
        "parent_label": "parent_name",
        "level": "level",
        "lvl": "level",
        "depth": "level",
        "hierarchy_level": "level",
        "generation": "level",
        "level_number": "level",
        "leaf_flag": "leaf_flag",
        "is_leaf": "leaf_flag",
        "leaf": "leaf_flag",
        "terminal": "leaf_flag",
        "is_terminal": "leaf_flag",
        "leaf_node": "leaf_flag",
        "sort_order": "sort_order",
        "sort": "sort_order",
        "sequence": "sort_order",
        "order": "sort_order",
        "position": "sort_order",
        "source_system": "source_system",
    }
    MAPPING_ALIASES = {
        "dimension": "dimension",
        "source_member_name": "source_member_name",
        "source_member": "source_member_name",
        "source_value": "source_member_name",
        "source_member_value": "source_member_name",
        "source_hierarchy_member": "source_member_name",
        "smartview_member": "source_member_name",
        "from_member": "source_member_name",
        "target_member_name": "target_member_name",
        "target_member": "target_member_name",
        "target_value": "target_member_name",
        "target_member_value": "target_member_name",
        "target_hierarchy_member": "target_member_name",
        "curated_member": "target_member_name",
        "adls_member": "target_member_name",
        "to_member": "target_member_name",
        "mapping_status": "mapping_status",
        "mapping_rule": "mapping_rule",
        "notes": "notes",
    }
    RULE_ALIASES = {
        "rule_id": "rule_id",
        "dimension": "dimension",
        "rule_name": "rule_name",
        "check_type": "check_type",
        "severity": "severity",
        "enabled": "enabled",
        "optional": "optional",
        "description": "description",
        "business_rationale": "business_rationale",
    }

    def normalize_hierarchy(self, frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
        normalized = self._rename_columns(frame, self.HIERARCHY_ALIASES)
        self._require_columns(normalized.columns, ["member_name"])
        normalized["dimension"] = dimension
        normalized["member_name"] = normalized["member_name"].map(normalize_optional_str)
        normalized = normalized.loc[normalized["member_name"].notna()].copy()
        if "parent_name" not in normalized.columns:
            normalized["parent_name"] = None
        normalized["parent_name"] = normalized["parent_name"].map(normalize_optional_str)
        derived_levels = self._derive_levels(normalized)
        if "level" in normalized.columns:
            normalized["level"] = pd.to_numeric(normalized["level"], errors="coerce").fillna(derived_levels).astype(int)
        else:
            normalized["level"] = derived_levels

        derived_leaf_flags = self._derive_leaf_flags(normalized)
        if "leaf_flag" in normalized.columns:
            normalized["leaf_flag"] = [
                self._coerce_bool_or_default(value, default)
                for value, default in zip(normalized["leaf_flag"].tolist(), derived_leaf_flags.tolist(), strict=False)
            ]
        else:
            normalized["leaf_flag"] = derived_leaf_flags

        default_sort_order = pd.Series(range(1, len(normalized) + 1), index=normalized.index)
        normalized["sort_order"] = pd.to_numeric(
            normalized.get("sort_order", default_sort_order),
            errors="coerce",
        ).fillna(default_sort_order).astype(int)
        normalized["member_code"] = normalized.get(
            "member_code",
            pd.Series([None] * len(normalized), index=normalized.index),
        ).map(normalize_optional_str)
        normalized["member_description"] = normalized.get(
            "member_description",
            pd.Series([None] * len(normalized), index=normalized.index),
        ).map(normalize_optional_str)
        normalized["source_system"] = normalized.get(
            "source_system",
            pd.Series(["Oracle Smart View"] * len(normalized), index=normalized.index),
        ).fillna("Oracle Smart View")
        return normalized[
            [
                "dimension",
                "member_code",
                "member_name",
                "parent_name",
                "level",
                "leaf_flag",
                "sort_order",
                "member_description",
                "source_system",
            ]
        ].copy()

    def _derive_levels(self, normalized: pd.DataFrame) -> pd.Series:
        parent_lookup: dict[str, str | None] = {}
        for member_name, parent_name in normalized[["member_name", "parent_name"]].itertuples(index=False, name=None):
            if member_name not in parent_lookup:
                parent_lookup[member_name] = parent_name
            elif parent_lookup[member_name] is None and parent_name is not None:
                parent_lookup[member_name] = parent_name

        cache: dict[str, int] = {}
        visiting: set[str] = set()

        def resolve_member_level(member_name: str | None) -> int:
            if member_name is None:
                return 0
            if member_name in cache:
                return cache[member_name]
            if member_name in visiting:
                return 0

            visiting.add(member_name)
            parent_name = parent_lookup.get(member_name)
            if parent_name in {None, member_name}:
                level = 0
            else:
                level = resolve_member_level(parent_name) + 1
            visiting.discard(member_name)
            cache[member_name] = level
            return level

        derived_levels = []
        for member_name, parent_name in normalized[["member_name", "parent_name"]].itertuples(index=False, name=None):
            if parent_name is None:
                derived_levels.append(0)
                continue
            parent_level = resolve_member_level(parent_name)
            if parent_name not in parent_lookup:
                derived_levels.append(1)
                continue
            if parent_name == member_name:
                derived_levels.append(0)
                continue
            derived_levels.append(parent_level + 1)
        return pd.Series(derived_levels, index=normalized.index, dtype=int)

    @staticmethod
    def _derive_leaf_flags(normalized: pd.DataFrame) -> pd.Series:
        parent_names = set(normalized["parent_name"].dropna().tolist())
        return normalized["member_name"].map(lambda member_name: member_name not in parent_names).astype(bool)

    @staticmethod
    def _coerce_bool_or_default(value: object, default: bool) -> bool:
        normalized = normalize_optional_str(value)
        if normalized is None:
            return bool(default)
        return coerce_bool(value)

    def normalize_mapping(self, frame: pd.DataFrame, default_dimension: str | None = None) -> pd.DataFrame:
        normalized = self._rename_columns(frame, self.MAPPING_ALIASES)
        if "dimension" not in normalized.columns and default_dimension:
            normalized["dimension"] = default_dimension
        self._require_columns(normalized.columns, ["dimension", "source_member_name", "target_member_name"])
        normalized["dimension"] = normalized["dimension"].map(str).str.strip().str.lower()
        normalized["source_member_name"] = normalized["source_member_name"].map(str).str.strip()
        normalized["target_member_name"] = normalized["target_member_name"].map(str).str.strip()
        normalized["mapping_status"] = normalized.get("mapping_status", pd.Series(["active"] * len(normalized))).fillna("active")
        normalized["mapping_rule"] = normalized.get("mapping_rule", pd.Series(["exact_match"] * len(normalized))).fillna("exact_match")
        normalized["notes"] = normalized.get("notes", pd.Series([None] * len(normalized))).map(normalize_optional_str)
        return normalized[
            ["dimension", "source_member_name", "target_member_name", "mapping_status", "mapping_rule", "notes"]
        ].copy()

    def normalize_rules(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = self._rename_columns(frame, self.RULE_ALIASES)
        self._require_columns(
            normalized.columns,
            ["rule_id", "dimension", "rule_name", "check_type", "description", "business_rationale"],
        )
        normalized["dimension"] = normalized["dimension"].map(str).str.strip().str.lower()
        normalized["enabled"] = normalized.get("enabled", pd.Series([True] * len(normalized))).map(coerce_bool)
        normalized["optional"] = normalized.get("optional", pd.Series([False] * len(normalized))).map(coerce_bool)
        normalized["severity"] = normalized.get("severity", pd.Series(["medium"] * len(normalized))).fillna("medium")
        return normalized[
            [
                "rule_id",
                "dimension",
                "rule_name",
                "check_type",
                "severity",
                "enabled",
                "optional",
                "description",
                "business_rationale",
            ]
        ].copy()

    def normalize_measure(self, frame: pd.DataFrame, dimension: str, dataset_name: str | None = None) -> pd.DataFrame:
        normalized = self._rename_columns(frame, {})
        normalized["dimension"] = dimension
        if dataset_name is not None and "dataset_name" not in normalized.columns:
            normalized["dataset_name"] = dataset_name

        for column in normalized.columns:
            if column in self.MEASURE_METADATA_COLUMNS:
                continue
            if pd.api.types.is_object_dtype(normalized[column]) or pd.api.types.is_string_dtype(normalized[column]):
                normalized[column] = normalized[column].map(normalize_optional_str)

        ordered_columns = []
        for column in ["dimension", "dataset_name"]:
            if column in normalized.columns:
                ordered_columns.append(column)
        ordered_columns.extend(column for column in normalized.columns if column not in ordered_columns)
        return normalized[ordered_columns].copy()

    @staticmethod
    def _rename_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
        renamed = frame.copy()
        renamed.columns = [aliases.get(to_snake_case(column), to_snake_case(column)) for column in renamed.columns]
        return renamed

    @staticmethod
    def _require_columns(columns: Iterable[str], required: list[str]) -> None:
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
