from __future__ import annotations

from hierarchy_migration_validation_agent.schemas import ValidationRule


COMMON_RULES = [
    ("missing_members_in_target", "Missing members in target", "high", False, "Source members must all exist in the curated target hierarchy."),
    ("parent_existence", "Parent existence", "high", False, "Every target parent reference must resolve to an existing target member."),
    ("parent_mismatch", "Source parent vs target parent mismatch", "high", False, "Target parent-child relationships must match the Smart View source extract."),
    ("duplicate_members", "Duplicate members", "high", False, "Each hierarchy member should appear once in the curated target."),
    ("leaf_flag_consistency", "Leaf/non-leaf consistency", "medium", False, "Leaf indicators must align to the actual presence or absence of children."),
    ("level_consistency", "Level consistency", "medium", False, "Hierarchy levels must remain consistent after normalization."),
    ("row_level_match", "Row-level hierarchy match", "medium", False, "Normalized hierarchy rows should match exactly between source and target."),
    ("numeric_value_match", "Numeric value match", "high", False, "Comparable numeric values should match between source and target measure sheets."),
    ("mapping_completeness", "Mapping completeness", "high", False, "Every source member must have a valid mapping record before migration is accepted."),
]


def default_rules_for_dimensions(dimensions: list[str]) -> list[ValidationRule]:
    rules: list[ValidationRule] = []
    for dimension in dimensions:
        for index, (check_type, rule_name, severity, optional, description) in enumerate(COMMON_RULES, start=1):
            rules.append(
                ValidationRule(
                    rule_id=f"{dimension.upper()}_{index:03d}",
                    dimension=dimension,
                    rule_name=rule_name,
                    check_type=check_type,
                    severity=severity,
                    enabled=True,
                    optional=optional,
                    description=description,
                    business_rationale=(
                        f"{dimension.title()} hierarchy reconciliation should stay transparent and explainable "
                        "for business users and data engineering."
                    ),
                )
            )
        if dimension == "account":
            rules.append(
                ValidationRule(
                    rule_id="ACCOUNT_099",
                    dimension="account",
                    rule_name="Rollup preservation",
                    check_type="rollup_preservation",
                    severity="medium",
                    enabled=True,
                    optional=True,
                    description="Critical account totals should preserve the same descendant leaf members after migration.",
                    business_rationale="Revenue and expense totals should not silently shift because of hierarchy migration defects.",
                )
            )
    return rules
