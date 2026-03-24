from __future__ import annotations

from hierarchy_migration_validation_agent.schemas import ValidationResult
from hierarchy_migration_validation_agent.utils.validation_text import describe_checked_rule, describe_passed_rule


def test_passed_rule_descriptions_are_human_readable():
    result = ValidationResult(
        rule_id="ENTITY_001",
        rule_name="Missing members in target",
        check_type="missing_members_in_target",
        dimension="entity",
        severity="high",
        status="PASSED",
        passed_count=47,
        failed_count=0,
        likely_cause="",
        recommended_action="",
    )

    assert describe_checked_rule(result) == "Checked that every source hierarchy member exists in the target hierarchy."
    assert describe_passed_rule(result) == "All required source members were found in the target. 47 comparisons passed."
