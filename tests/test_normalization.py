from __future__ import annotations

import pandas as pd

from hierarchy_migration_validation_agent.normalization.hierarchy_normalizer import HierarchyNormalizer


def test_hierarchy_normalizer_standardizes_columns_and_types():
    frame = pd.DataFrame(
        [
            {
                "Member Name": "Test_Node",
                "Parent": "",
                "Level": "0",
                "Leaf Flag": "Y",
            }
        ]
    )

    normalized = HierarchyNormalizer().normalize_hierarchy(frame, "account")

    assert list(normalized.columns) == [
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
    assert normalized.loc[0, "dimension"] == "account"
    assert normalized.loc[0, "parent_name"] is None
    assert bool(normalized.loc[0, "leaf_flag"]) is True
    assert normalized.loc[0, "level"] == 0
