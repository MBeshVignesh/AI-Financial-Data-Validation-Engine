from __future__ import annotations

import pandas as pd

from hierarchy_migration_validation_agent.schemas import ValidationRule
from hierarchy_migration_validation_agent.validation.checks import HierarchyValidator


def _rule(frame, dimension: str, check_type: str) -> ValidationRule:
    record = frame.loc[(frame["dimension"] == dimension) & (frame["check_type"] == check_type)].iloc[0].to_dict()
    return ValidationRule.model_validate(record)


def test_missing_members_rule_detects_subscription_gap(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "missing_members_in_target"),
    )

    assert result.status == "FAILED"
    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Subscription_Rev"


def test_parent_existence_rule_detects_missing_target_parent(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="entity",
        source_df=seeded_frames["entity_source"],
        target_df=seeded_frames["entity_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "entity"],
        rule=_rule(seeded_frames["rules"], "entity", "parent_existence"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "California_BU"


def test_parent_mismatch_rule_detects_wrong_parent(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "parent_mismatch"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "License_Rev"


def test_duplicate_members_rule_detects_duplicate_target_member(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "duplicate_members"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Salary_Expense"


def test_leaf_consistency_rule_detects_invalid_flag(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "leaf_flag_consistency"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Total_Expense"


def test_level_consistency_rule_detects_level_mismatch(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "level_consistency"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Rent_Expense"


def test_level_consistency_rule_ignores_duplicate_member_rows_with_same_level():
    validator = HierarchyValidator()
    rule = ValidationRule(
        rule_id="ENTITY_010",
        dimension="entity",
        rule_name="Level consistency",
        check_type="level_consistency",
        severity="medium",
        enabled=True,
        optional=False,
        description="Hierarchy levels must remain consistent after normalization.",
        business_rationale="Duplicate rows from multiple parsed sheets should not create false level mismatches.",
    )
    source_df = pd.DataFrame(
        [
            {"member_name": "Europe", "level": 1},
            {"member_name": "Europe", "level": 1},
            {"member_name": "Germany_BU", "level": 2},
            {"member_name": "Germany_BU", "level": 2},
        ]
    )
    target_df = source_df.copy()

    result = validator.run_rule(
        dimension="entity",
        source_df=source_df,
        target_df=target_df,
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=pd.DataFrame(),
        rule=rule,
    )

    assert result.status == "PASSED"
    assert result.failed_count == 0


def test_mapping_completeness_rule_detects_missing_mapping(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="entity",
        source_df=seeded_frames["entity_source"],
        target_df=seeded_frames["entity_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "entity"],
        rule=_rule(seeded_frames["rules"], "entity", "mapping_completeness"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Germany_BU"


def test_row_level_match_rule_detects_structural_difference(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "row_level_match"),
    )

    assert result.failed_count >= 1
    assert {failure.member_name for failure in result.failed_records} >= {"Subscription_Rev", "License_Rev"}


def test_rollup_preservation_rule_detects_total_revenue_shift(seeded_frames):
    validator = HierarchyValidator()
    result = validator.run_rule(
        dimension="account",
        source_df=seeded_frames["account_source"],
        target_df=seeded_frames["account_target"],
        source_measure_df=pd.DataFrame(),
        target_measure_df=pd.DataFrame(),
        mapping_df=seeded_frames["mapping"].loc[seeded_frames["mapping"]["dimension"] == "account"],
        rule=_rule(seeded_frames["rules"], "account", "rollup_preservation"),
    )

    assert result.failed_count == 1
    assert result.failed_records[0].member_name == "Total_Revenue"


def test_numeric_value_match_rule_detects_changed_salary():
    validator = HierarchyValidator()
    rules_frame = pd.DataFrame(
        [
            {
                "rule_id": "ENTITY_999",
                "dimension": "entity",
                "rule_name": "Numeric value match",
                "check_type": "numeric_value_match",
                "severity": "high",
                "enabled": True,
                "optional": False,
                "description": "Comparable numeric values should match between source and target measure sheets.",
                "business_rationale": "Measure totals should not drift silently during migration.",
            }
        ]
    )
    source_measure = pd.DataFrame(
        [
            {
                "dimension": "entity",
                "dataset_name": "hcm_employee_data",
                "entity": "GlobalCorp",
                "business_unit": "Europe",
                "department": "Legal",
                "cost_center": "LEG210",
                "employee_id": "E1000",
                "salary": 94696,
            }
        ]
    )
    target_measure = source_measure.copy()
    target_measure.loc[0, "salary"] = 120000

    result = validator.run_rule(
        dimension="entity",
        source_df=pd.DataFrame(),
        target_df=pd.DataFrame(),
        source_measure_df=source_measure,
        target_measure_df=target_measure,
        mapping_df=pd.DataFrame(),
        rule=ValidationRule.model_validate(rules_frame.iloc[0].to_dict()),
    )

    assert result.status == "FAILED"
    assert result.failed_count == 1
    assert result.failed_records[0].details["measure_column"] == "salary"
