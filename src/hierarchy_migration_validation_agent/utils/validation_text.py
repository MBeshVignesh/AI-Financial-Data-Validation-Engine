from __future__ import annotations

from hierarchy_migration_validation_agent.schemas import ValidationResult


CHECK_DESCRIPTIONS = {
    "missing_members_in_target": "Checked that every source hierarchy member exists in the target hierarchy.",
    "parent_existence": "Checked that every target parent reference points to a valid target member.",
    "parent_mismatch": "Checked that each source member rolls up to the same parent in the target hierarchy.",
    "duplicate_members": "Checked that the target hierarchy does not contain duplicate hierarchy rows.",
    "leaf_flag_consistency": "Checked that leaf flags match the actual child relationships in the hierarchy.",
    "level_consistency": "Checked that hierarchy depth stays consistent between source and target.",
    "row_level_match": "Checked that normalized source and target hierarchy rows match exactly.",
    "numeric_value_match": "Checked that comparable source and target measure rows have matching numeric values.",
    "mapping_completeness": "Checked that every source member has an active mapping record.",
    "rollup_preservation": "Checked that important account rollups preserve the same descendant leaf members.",
}

PASS_OUTCOMES = {
    "missing_members_in_target": "All required source members were found in the target.",
    "parent_existence": "All referenced parents were present in the target hierarchy.",
    "parent_mismatch": "All compared members rolled up to the expected target parent.",
    "duplicate_members": "No duplicate hierarchy rows were found in the target.",
    "leaf_flag_consistency": "Leaf and non-leaf flags aligned with the hierarchy structure.",
    "level_consistency": "Hierarchy levels remained consistent between source and target.",
    "row_level_match": "Source and target hierarchy rows matched without extra or missing rows.",
    "numeric_value_match": "Comparable source and target measure rows matched numerically.",
    "mapping_completeness": "All source members had active mapping coverage.",
    "rollup_preservation": "The compared account rollups preserved the same descendant leaves.",
}


def describe_checked_rule(result: ValidationResult) -> str:
    return CHECK_DESCRIPTIONS.get(
        result.check_type,
        f"Checked the {result.rule_name.lower()} rule for the {result.dimension} dataset.",
    )


def describe_passed_rule(result: ValidationResult) -> str:
    baseline = PASS_OUTCOMES.get(result.check_type, "The validation passed without any flagged issues.")
    if result.passed_count > 0:
        comparison_label = "comparison" if result.passed_count == 1 else "comparisons"
        return f"{baseline} {result.passed_count} {comparison_label} passed."
    return baseline
