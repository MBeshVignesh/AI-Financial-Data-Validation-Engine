from __future__ import annotations

from collections import defaultdict
from typing import Callable

import pandas as pd

from hierarchy_migration_validation_agent.schemas import FailureRecord, ValidationResult, ValidationRule
from hierarchy_migration_validation_agent.utils.text import normalize_optional_str


class HierarchyValidator:
    LIKELY_CAUSES = {
        "missing_members_in_target": "The target load likely filtered out valid source members or the curated insert step missed a subset of records.",
        "parent_existence": "A transformation rule likely produced a parent reference that never landed in the curated hierarchy.",
        "parent_mismatch": "The parent-child remapping logic is likely outdated relative to the Smart View source hierarchy.",
        "duplicate_members": "The curated target likely merged multiple loads without a deterministic de-duplication key.",
        "leaf_flag_consistency": "Leaf flags appear to have been copied or derived incorrectly after parent-child relationships changed.",
        "level_consistency": "Hierarchy depth appears to have been recalculated incorrectly during normalization or flattening.",
        "row_level_match": "The normalized source and target hierarchy rows do not align, which usually points to sheet-selection, normalization, or load logic drift.",
        "numeric_value_match": "Numeric measure data differs between source and target, which usually points to an aggregation, currency, or row-level load mismatch.",
        "mapping_completeness": "The mapping workbook or curated mapping table is incomplete for one or more source members.",
        "rollup_preservation": "Root rollups changed because missing or mis-parented members altered the descendant leaf set.",
    }
    RECOMMENDED_ACTIONS = {
        "missing_members_in_target": "Re-run the target load for the missing members and review any filters applied during normalization.",
        "parent_existence": "Fix unresolved parent references in the transformation layer and reload the affected hierarchy records.",
        "parent_mismatch": "Compare the mapping logic to the current source hierarchy and correct the parent remap rules.",
        "duplicate_members": "Add or repair de-duplication logic before inserting into the curated target tables.",
        "leaf_flag_consistency": "Recompute leaf indicators from the final hierarchy graph instead of preserving stale flags.",
        "level_consistency": "Recalculate hierarchy levels from the resolved parent-child structure during normalization.",
        "row_level_match": "Compare normalized source and target rows side by side and reconcile any missing, extra, or structurally different hierarchy rows.",
        "numeric_value_match": "Compare source and target measure rows on the shared business keys, then reconcile any mismatched numeric values or missing measure rows.",
        "mapping_completeness": "Add the missing mapping records and re-run the migration validation.",
        "rollup_preservation": "Review affected parent-child relationships for revenue and expense totals before approving the migration.",
    }

    def __init__(self) -> None:
        self._checks: dict[str, Callable[..., ValidationResult]] = {
            "missing_members_in_target": self._check_missing_members,
            "parent_existence": self._check_parent_existence,
            "parent_mismatch": self._check_parent_mismatch,
            "duplicate_members": self._check_duplicate_members,
            "leaf_flag_consistency": self._check_leaf_flag_consistency,
            "level_consistency": self._check_level_consistency,
            "row_level_match": self._check_row_level_match,
            "numeric_value_match": self._check_numeric_value_match,
            "mapping_completeness": self._check_mapping_completeness,
            "rollup_preservation": self._check_rollup_preservation,
        }

    def run_rule(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str] | None = None,
    ) -> ValidationResult:
        check = self._checks.get(rule.check_type)
        if check is None:
            raise ValueError(f"Unsupported check type: {rule.check_type}")
        return check(
            dimension=dimension,
            source_df=self._prepare_frame(source_df),
            target_df=self._prepare_frame(target_df),
            source_measure_df=self._prepare_frame(source_measure_df),
            target_measure_df=self._prepare_frame(target_measure_df),
            mapping_df=self._prepare_frame(mapping_df),
            rule=rule,
            retrieved_context=retrieved_context or [],
        )

    def run_dimension_checks(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rules: list[ValidationRule],
        retrieved_context: list[str] | None = None,
    ) -> list[ValidationResult]:
        return [
            self.run_rule(
                dimension=dimension,
                source_df=source_df,
                target_df=target_df,
                source_measure_df=source_measure_df,
                target_measure_df=target_measure_df,
                mapping_df=mapping_df,
                rule=rule,
                retrieved_context=retrieved_context,
            )
            for rule in rules
            if rule.enabled
        ]

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if "parent_name" in prepared.columns:
            prepared["parent_name"] = prepared["parent_name"].map(normalize_optional_str)
        if "member_name" in prepared.columns:
            prepared["member_name"] = prepared["member_name"].map(str)
        return prepared

    def _check_missing_members(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        missing = source_df.loc[~source_df["member_name"].isin(target_df["member_name"])]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Source member missing from target hierarchy",
                source_value=self._optional_string(row.member_name),
            )
            for row in missing.itertuples()
        ]
        return self._build_result(rule, dimension, len(source_df) - len(failures), failures, retrieved_context)

    def _check_parent_existence(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_df, source_measure_df, target_measure_df, mapping_df
        valid_members = set(target_df["member_name"])
        invalid = target_df.loc[
            target_df["parent_name"].notna() & ~target_df["parent_name"].isin(valid_members)
        ]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Target parent does not exist",
                target_value=self._optional_string(row.parent_name),
            )
            for row in invalid.itertuples()
        ]
        return self._build_result(rule, dimension, len(target_df) - len(failures), failures, retrieved_context)

    def _check_parent_mismatch(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        source_deduped = source_df.drop_duplicates(subset=["member_name"], keep="first")
        target_deduped = target_df.drop_duplicates(subset=["member_name"], keep="first")
        joined = source_deduped[["member_name", "parent_name"]].merge(
            target_deduped[["member_name", "parent_name"]],
            on="member_name",
            how="inner",
            suffixes=("_source", "_target"),
        )
        joined["parent_name_source"] = joined["parent_name_source"].map(normalize_optional_str)
        joined["parent_name_target"] = joined["parent_name_target"].map(normalize_optional_str)
        comparison = joined[["parent_name_source", "parent_name_target"]].fillna("__ROOT__")
        mismatch = joined.loc[comparison["parent_name_source"] != comparison["parent_name_target"]]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Source parent differs from target parent",
                source_value=self._optional_string(row.parent_name_source),
                target_value=self._optional_string(row.parent_name_target),
            )
            for row in mismatch.itertuples()
        ]
        return self._build_result(rule, dimension, len(joined) - len(failures), failures, retrieved_context)

    def _check_duplicate_members(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_df, source_measure_df, target_measure_df, mapping_df
        duplicate_counts = target_df.groupby(["member_name", "parent_name", "level"], dropna=False).size()
        duplicate_counts = duplicate_counts.loc[duplicate_counts > 1]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=member_name,
                issue="Duplicate member found in target hierarchy",
                target_value=str(count),
                details={
                    "duplicate_count": int(count),
                    "parent_name": normalize_optional_str(parent_name),
                    "level": int(level),
                },
            )
            for (member_name, parent_name, level), count in duplicate_counts.items()
        ]
        distinct_rows = target_df.drop_duplicates(subset=["member_name", "parent_name", "level"])
        return self._build_result(rule, dimension, len(distinct_rows) - len(failures), failures, retrieved_context)

    def _check_leaf_flag_consistency(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        source_deduped = source_df.drop_duplicates(subset=["member_name"], keep="first")
        target_deduped = target_df.drop_duplicates(subset=["member_name"], keep="first")
        joined = source_deduped[["member_name", "leaf_flag"]].merge(
            target_deduped[["member_name", "leaf_flag"]],
            on="member_name",
            how="inner",
            suffixes=("_source", "_target"),
        )
        mismatch = joined.loc[joined["leaf_flag_source"] != joined["leaf_flag_target"]]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Leaf flag differs between source and target",
                source_value=str(bool(row.leaf_flag_source)),
                target_value=str(bool(row.leaf_flag_target)),
            )
            for row in mismatch.itertuples()
        ]
        return self._build_result(rule, dimension, len(joined) - len(failures), failures, retrieved_context)

    def _check_level_consistency(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        source_deduped = source_df.drop_duplicates(subset=["member_name"], keep="first")
        target_deduped = target_df.drop_duplicates(subset=["member_name"], keep="first")
        joined = source_deduped[["member_name", "level"]].merge(
            target_deduped[["member_name", "level"]],
            on="member_name",
            how="inner",
            suffixes=("_source", "_target"),
        )
        mismatch = joined.loc[joined["level_source"] != joined["level_target"]]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Source level differs from target level",
                source_value=str(row.level_source),
                target_value=str(row.level_target),
            )
            for row in mismatch.itertuples()
        ]
        return self._build_result(rule, dimension, len(joined) - len(failures), failures, retrieved_context)

    def _check_mapping_completeness(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del target_df, source_measure_df, target_measure_df
        active_mappings = set(
            mapping_df.loc[mapping_df["mapping_status"].str.lower() == "active", "source_member_name"].tolist()
        )
        missing = source_df.loc[~source_df["member_name"].isin(active_mappings)]
        failures = [
            FailureRecord(
                dimension=dimension,
                rule_name=rule.rule_name,
                member_name=row.member_name,
                issue="Source member has no active mapping record",
                source_value=self._optional_string(row.member_name),
            )
            for row in missing.itertuples()
        ]
        return self._build_result(rule, dimension, len(source_df) - len(failures), failures, retrieved_context)

    def _check_numeric_value_match(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_df, target_df, mapping_df
        if source_measure_df.empty and target_measure_df.empty:
            return self._build_result(rule, dimension, 0, [], retrieved_context)
        if source_measure_df.empty:
            return self._build_result(
                rule,
                dimension,
                0,
                [
                    FailureRecord(
                        dimension=dimension,
                        rule_name=rule.rule_name,
                        member_name=dimension,
                        issue="Target workbook contains numeric measure data but the source workbook does not",
                    )
                ],
                retrieved_context,
            )
        if target_measure_df.empty:
            return self._build_result(
                rule,
                dimension,
                0,
                [
                    FailureRecord(
                        dimension=dimension,
                        rule_name=rule.rule_name,
                        member_name=dimension,
                        issue="Source workbook contains numeric measure data but the target workbook does not",
                    )
                ],
                retrieved_context,
            )

        source_dataset_names = set(source_measure_df.get("dataset_name", pd.Series(dtype="object")).dropna().tolist())
        target_dataset_names = set(target_measure_df.get("dataset_name", pd.Series(dtype="object")).dropna().tolist())
        dataset_names = sorted(source_dataset_names | target_dataset_names)

        failures: list[FailureRecord] = []
        passed_count = 0

        for dataset_name in dataset_names:
            source_dataset = source_measure_df.loc[source_measure_df["dataset_name"] == dataset_name].copy()
            target_dataset = target_measure_df.loc[target_measure_df["dataset_name"] == dataset_name].copy()

            if source_dataset.empty:
                failures.append(
                    FailureRecord(
                        dimension=dimension,
                        rule_name=rule.rule_name,
                        member_name=dataset_name,
                        issue="Target measure dataset has no matching source dataset",
                        target_value=dataset_name,
                    )
                )
                continue
            if target_dataset.empty:
                failures.append(
                    FailureRecord(
                        dimension=dimension,
                        rule_name=rule.rule_name,
                        member_name=dataset_name,
                        issue="Source measure dataset is missing from the target workbook",
                        source_value=dataset_name,
                    )
                )
                continue

            source_measure_columns = self._measure_columns(source_dataset)
            target_measure_columns = self._measure_columns(target_dataset)
            shared_measure_columns = sorted(set(source_measure_columns) & set(target_measure_columns))
            if not shared_measure_columns:
                continue

            key_columns = self._shared_measure_key_columns(source_dataset, target_dataset, shared_measure_columns)
            source_aggregated = self._aggregate_measure_frame(source_dataset, key_columns, shared_measure_columns)
            target_aggregated = self._aggregate_measure_frame(target_dataset, key_columns, shared_measure_columns)
            merged = source_aggregated.merge(
                target_aggregated,
                on=key_columns,
                how="outer",
                suffixes=("_source", "_target"),
                indicator=True,
            )

            for _, row in merged.iterrows():
                row_identifier = self._build_row_identifier(row, key_columns) or dataset_name
                if row["_merge"] == "left_only":
                    failures.append(
                        FailureRecord(
                            dimension=dimension,
                            rule_name=rule.rule_name,
                            member_name=row_identifier,
                            issue=f"Measure row from dataset '{dataset_name}' is missing from the target workbook",
                            source_value=row_identifier,
                            details={"dataset_name": dataset_name},
                        )
                    )
                    continue
                if row["_merge"] == "right_only":
                    failures.append(
                        FailureRecord(
                            dimension=dimension,
                            rule_name=rule.rule_name,
                            member_name=row_identifier,
                            issue=f"Target workbook contains an extra measure row in dataset '{dataset_name}'",
                            target_value=row_identifier,
                            details={"dataset_name": dataset_name},
                        )
                    )
                    continue

                for measure_column in shared_measure_columns:
                    source_value = row[f"{measure_column}_source"]
                    target_value = row[f"{measure_column}_target"]
                    if not self._numeric_values_equal(source_value, target_value):
                        failures.append(
                            FailureRecord(
                                dimension=dimension,
                                rule_name=rule.rule_name,
                                member_name=row_identifier,
                                issue=f"Numeric column '{measure_column}' differs between source and target",
                                source_value=self._stringify_numeric(source_value),
                                target_value=self._stringify_numeric(target_value),
                                details={"dataset_name": dataset_name, "measure_column": measure_column},
                            )
                        )
                    else:
                        passed_count += 1

        return self._build_result(rule, dimension, passed_count, failures, retrieved_context)

    def _check_row_level_match(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        compare_columns = ["member_name", "parent_name", "level", "leaf_flag"]
        source_rows = self._row_key_set(source_df, compare_columns)
        target_rows = self._row_key_set(target_df, compare_columns)

        missing_in_target = sorted(source_rows - target_rows, key=self._row_sort_key)
        extra_in_target = sorted(target_rows - source_rows, key=self._row_sort_key)
        failures: list[FailureRecord] = []

        for member_name, parent_name, level, leaf_flag in missing_in_target:
            failures.append(
                FailureRecord(
                    dimension=dimension,
                    rule_name=rule.rule_name,
                    member_name=member_name,
                    issue="Normalized source row is missing from the target hierarchy",
                    source_value=self._format_row_signature(parent_name, level, leaf_flag),
                )
            )

        for member_name, parent_name, level, leaf_flag in extra_in_target:
            failures.append(
                FailureRecord(
                    dimension=dimension,
                    rule_name=rule.rule_name,
                    member_name=member_name,
                    issue="Target contains a normalized row not found in the source hierarchy",
                    target_value=self._format_row_signature(parent_name, level, leaf_flag),
                )
            )

        return self._build_result(
            rule,
            dimension,
            len(source_rows & target_rows),
            failures,
            retrieved_context,
        )

    def _check_rollup_preservation(
        self,
        *,
        dimension: str,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_measure_df: pd.DataFrame,
        target_measure_df: pd.DataFrame,
        mapping_df: pd.DataFrame,
        rule: ValidationRule,
        retrieved_context: list[str],
    ) -> ValidationResult:
        del source_measure_df, target_measure_df, mapping_df
        roots = source_df.loc[source_df["member_name"].str.startswith("Total_"), "member_name"].tolist()
        source_tree = self._build_tree(source_df)
        target_tree = self._build_tree(target_df.drop_duplicates(subset=["member_name"], keep="first"))
        failures: list[FailureRecord] = []
        for root in roots:
            source_descendants = self._leaf_descendants(root, source_tree)
            target_descendants = self._leaf_descendants(root, target_tree)
            if source_descendants != target_descendants:
                failures.append(
                    FailureRecord(
                        dimension=dimension,
                        rule_name=rule.rule_name,
                        member_name=root,
                        issue="Leaf descendants under the total changed after migration",
                        source_value=", ".join(sorted(source_descendants)),
                        target_value=", ".join(sorted(target_descendants)),
                        details={
                            "missing_descendants": sorted(source_descendants - target_descendants),
                            "extra_descendants": sorted(target_descendants - source_descendants),
                        },
                    )
                )
        return self._build_result(rule, dimension, len(roots) - len(failures), failures, retrieved_context)

    @staticmethod
    def _build_tree(frame: pd.DataFrame) -> dict[str | None, list[str]]:
        tree: dict[str | None, list[str]] = defaultdict(list)
        for row in frame.itertuples():
            tree[row.parent_name].append(row.member_name)
        return tree

    def _leaf_descendants(self, root: str, tree: dict[str | None, list[str]]) -> set[str]:
        children = tree.get(root, [])
        if not children:
            return {root}
        descendants: set[str] = set()
        for child in children:
            descendants.update(self._leaf_descendants(child, tree))
        return descendants

    def _build_result(
        self,
        rule: ValidationRule,
        dimension: str,
        passed_count: int,
        failures: list[FailureRecord],
        retrieved_context: list[str],
    ) -> ValidationResult:
        has_failures = bool(failures)
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            check_type=rule.check_type,
            dimension=dimension,
            severity=rule.severity,
            status="FAILED" if has_failures else "PASSED",
            passed_count=max(0, passed_count),
            failed_count=len(failures),
            failed_records=failures,
            retrieved_context=retrieved_context[:4] if has_failures else [],
            likely_cause=self.LIKELY_CAUSES[rule.check_type] if has_failures else "",
            recommended_action=self.RECOMMENDED_ACTIONS[rule.check_type] if has_failures else "",
        )

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return normalize_optional_str(value)

    @staticmethod
    def _row_key_set(frame: pd.DataFrame, compare_columns: list[str]) -> set[tuple[str, str | None, int, bool]]:
        comparable = frame.loc[:, compare_columns].copy()
        comparable["parent_name"] = comparable["parent_name"].map(normalize_optional_str)
        comparable["level"] = comparable["level"].fillna(0).astype(int)
        comparable["leaf_flag"] = comparable["leaf_flag"].astype(bool)
        comparable = comparable.drop_duplicates()
        return {
            (
                str(row.member_name),
                normalize_optional_str(row.parent_name),
                int(row.level),
                bool(row.leaf_flag),
            )
            for row in comparable.itertuples(index=False)
        }

    @staticmethod
    def _format_row_signature(parent_name: str | None, level: int, leaf_flag: bool) -> str:
        return (
            f"parent={parent_name or 'ROOT'}, "
            f"level={level}, "
            f"leaf_flag={leaf_flag}"
        )

    @staticmethod
    def _row_sort_key(row: tuple[str, str | None, int, bool]) -> tuple[str, str, int, bool]:
        member_name, parent_name, level, leaf_flag = row
        return member_name, parent_name or "", level, leaf_flag

    @staticmethod
    def _measure_columns(frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if column not in {"dimension", "dataset_name"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

    @staticmethod
    def _shared_measure_key_columns(
        source_frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        measure_columns: list[str],
    ) -> list[str]:
        candidate_columns = [
            column
            for column in source_frame.columns
            if column in target_frame.columns and column not in {"dimension", "dataset_name", *measure_columns}
        ]
        return candidate_columns or ["dataset_name"]

    @staticmethod
    def _aggregate_measure_frame(
        frame: pd.DataFrame,
        key_columns: list[str],
        measure_columns: list[str],
    ) -> pd.DataFrame:
        aggregation_frame = frame.copy()
        return aggregation_frame.groupby(key_columns, dropna=False)[measure_columns].sum().reset_index()

    @staticmethod
    def _build_row_identifier(row, key_columns: list[str]) -> str:
        values = []
        for column in key_columns:
            value = row[column]
            if pd.isna(value):
                continue
            values.append(f"{column}={value}")
        return ", ".join(values)

    @staticmethod
    def _numeric_values_equal(source_value: object, target_value: object, tolerance: float = 1e-9) -> bool:
        if pd.isna(source_value) and pd.isna(target_value):
            return True
        if pd.isna(source_value) or pd.isna(target_value):
            return False
        return abs(float(source_value) - float(target_value)) <= tolerance

    @staticmethod
    def _stringify_numeric(value: object) -> str:
        if pd.isna(value):
            return "null"
        return str(value)
