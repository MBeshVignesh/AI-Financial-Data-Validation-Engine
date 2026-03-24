from __future__ import annotations

import pandas as pd
import pytest

from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.config import Settings


def _account_source_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "account", "member_code": "ACC_100", "member_name": "Total_Revenue", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Top revenue rollup"},
        {"dimension": "account", "member_code": "ACC_110", "member_name": "Product_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "All product revenue"},
        {"dimension": "account", "member_code": "ACC_120", "member_name": "Service_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": True, "sort_order": 3, "member_description": "Professional services revenue"},
        {"dimension": "account", "member_code": "ACC_111", "member_name": "Subscription_Rev", "parent_name": "Product_Revenue", "level": 2, "leaf_flag": True, "sort_order": 4, "member_description": "Recurring subscription revenue"},
        {"dimension": "account", "member_code": "ACC_112", "member_name": "License_Rev", "parent_name": "Product_Revenue", "level": 2, "leaf_flag": True, "sort_order": 5, "member_description": "Perpetual license revenue"},
        {"dimension": "account", "member_code": "ACC_200", "member_name": "Total_Expense", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 6, "member_description": "Top expense rollup"},
        {"dimension": "account", "member_code": "ACC_210", "member_name": "Operating_Expense", "parent_name": "Total_Expense", "level": 1, "leaf_flag": False, "sort_order": 7, "member_description": "Operating expense bucket"},
        {"dimension": "account", "member_code": "ACC_211", "member_name": "Salary_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 8, "member_description": "Payroll expense"},
        {"dimension": "account", "member_code": "ACC_212", "member_name": "Rent_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 9, "member_description": "Facilities expense"},
    ]


def _entity_source_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "entity", "member_code": "ENT_100", "member_name": "Global_Corp", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Global legal hierarchy"},
        {"dimension": "entity", "member_code": "ENT_110", "member_name": "North_America", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "North America rollup"},
        {"dimension": "entity", "member_code": "ENT_120", "member_name": "Europe", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 3, "member_description": "Europe rollup"},
        {"dimension": "entity", "member_code": "ENT_111", "member_name": "US_Business", "parent_name": "North_America", "level": 2, "leaf_flag": False, "sort_order": 4, "member_description": "US operating entities"},
        {"dimension": "entity", "member_code": "ENT_112", "member_name": "California_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 5, "member_description": "California business unit"},
        {"dimension": "entity", "member_code": "ENT_113", "member_name": "Texas_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 6, "member_description": "Texas business unit"},
        {"dimension": "entity", "member_code": "ENT_121", "member_name": "Germany_BU", "parent_name": "Europe", "level": 2, "leaf_flag": True, "sort_order": 7, "member_description": "Germany business unit"},
    ]


def _mapping_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for member in _account_source_rows():
        rows.append(
            {
                "dimension": member["dimension"],
                "source_member_name": member["member_name"],
                "target_member_name": member["member_name"],
                "mapping_status": "active",
                "mapping_rule": "exact_match",
                "notes": "Smart View account members should land one-to-one in the curated hierarchy table.",
            }
        )
    for member in _entity_source_rows():
        if member["member_name"] == "Germany_BU":
            continue
        rows.append(
            {
                "dimension": member["dimension"],
                "source_member_name": member["member_name"],
                "target_member_name": member["member_name"],
                "mapping_status": "active",
                "mapping_rule": "exact_match",
                "notes": "Entity members preserve business-unit names exactly unless a legal rename is approved.",
            }
        )
    return rows


def _rule_rows() -> list[dict[str, object]]:
    common_rules = [
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
    rows: list[dict[str, object]] = []
    for dimension in ["account", "entity"]:
        for index, (check_type, rule_name, severity, optional, description) in enumerate(common_rules, start=1):
            rows.append(
                {
                    "rule_id": f"{dimension.upper()}_{index:03d}",
                    "dimension": dimension,
                    "rule_name": rule_name,
                    "check_type": check_type,
                    "severity": severity,
                    "enabled": True,
                    "optional": optional,
                    "description": description,
                    "business_rationale": f"{dimension.title()} hierarchy reconciliation should be explainable to finance operations and data engineering.",
                }
            )
    rows.append(
        {
            "rule_id": "ACCOUNT_099",
            "dimension": "account",
            "rule_name": "Rollup preservation",
            "check_type": "rollup_preservation",
            "severity": "medium",
            "enabled": True,
            "optional": True,
            "description": "Critical account totals should preserve the same descendant leaf members after migration.",
            "business_rationale": "Revenue and expense totals must not silently shift due to hierarchy migration defects.",
        }
    )
    return rows


def _account_target_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "account", "member_code": "ACC_100", "member_name": "Total_Revenue", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Top revenue rollup"},
        {"dimension": "account", "member_code": "ACC_110", "member_name": "Product_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "All product revenue"},
        {"dimension": "account", "member_code": "ACC_120", "member_name": "Service_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": True, "sort_order": 3, "member_description": "Professional services revenue"},
        {"dimension": "account", "member_code": "ACC_112", "member_name": "License_Rev", "parent_name": "Total_Revenue", "level": 2, "leaf_flag": True, "sort_order": 4, "member_description": "Wrong parent defect"},
        {"dimension": "account", "member_code": "ACC_200", "member_name": "Total_Expense", "parent_name": None, "level": 0, "leaf_flag": True, "sort_order": 5, "member_description": "Invalid leaf flag defect"},
        {"dimension": "account", "member_code": "ACC_210", "member_name": "Operating_Expense", "parent_name": "Total_Expense", "level": 1, "leaf_flag": False, "sort_order": 6, "member_description": "Operating expense bucket"},
        {"dimension": "account", "member_code": "ACC_211", "member_name": "Salary_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 7, "member_description": "First duplicate row"},
        {"dimension": "account", "member_code": "ACC_211_DUP", "member_name": "Salary_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 8, "member_description": "Duplicate member defect"},
        {"dimension": "account", "member_code": "ACC_212", "member_name": "Rent_Expense", "parent_name": "Operating_Expense", "level": 1, "leaf_flag": True, "sort_order": 9, "member_description": "Level mismatch defect"},
    ]


def _entity_target_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "entity", "member_code": "ENT_100", "member_name": "Global_Corp", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Global legal hierarchy"},
        {"dimension": "entity", "member_code": "ENT_110", "member_name": "North_America", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "North America rollup"},
        {"dimension": "entity", "member_code": "ENT_120", "member_name": "Europe", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 3, "member_description": "Europe rollup"},
        {"dimension": "entity", "member_code": "ENT_111", "member_name": "US_Business", "parent_name": "North_America", "level": 2, "leaf_flag": False, "sort_order": 4, "member_description": "US operating entities"},
        {"dimension": "entity", "member_code": "ENT_112", "member_name": "California_BU", "parent_name": "West_Region", "level": 3, "leaf_flag": True, "sort_order": 5, "member_description": "Parent existence defect"},
        {"dimension": "entity", "member_code": "ENT_113", "member_name": "Texas_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 6, "member_description": "Texas business unit"},
        {"dimension": "entity", "member_code": "ENT_121", "member_name": "Germany_BU", "parent_name": "Europe", "level": 2, "leaf_flag": True, "sort_order": 7, "member_description": "Germany business unit"},
    ]


@pytest.fixture
def test_settings(tmp_path):
    settings = Settings.from_root(tmp_path)
    settings.embedding_provider = "nomic"
    settings.ensure_directories()
    return settings


@pytest.fixture(autouse=True)
def stub_local_embedding(monkeypatch):
    class DummyEmbeddingFunction:
        model_name = "dummy-local-nomic"

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in input]

        @staticmethod
        def name() -> str:
            return "sentence_transformer::dummy-local-nomic"

        @staticmethod
        def is_legacy() -> bool:
            return False

        @staticmethod
        def default_space() -> str:
            return "cosine"

        @staticmethod
        def supported_spaces() -> list[str]:
            return ["cosine"]

        @classmethod
        def build_from_config(cls, config: dict[str, str | bool]) -> "DummyEmbeddingFunction":
            del config
            return cls()

        def get_config(self) -> dict[str, str | bool]:
            return {
                "name": self.name(),
                "model_name": self.model_name,
                "device": "cpu",
                "local_files_only": True,
                "space": self.default_space(),
            }

        def embed_query(self, text: str) -> list[float]:
            length = float(max(len(text), 1))
            return [length, 1.0]

        def embed_documents(self, documents: list[str]) -> list[list[float]]:
            return [self.embed_query(document) for document in documents]

    monkeypatch.setattr(
        "hierarchy_migration_validation_agent.rag.indexer.create_embedding_function",
        lambda settings: DummyEmbeddingFunction(),
    )


@pytest.fixture
def seeded_frames():
    return {
        "account_source": pd.DataFrame(_account_source_rows()),
        "entity_source": pd.DataFrame(_entity_source_rows()),
        "account_target": pd.DataFrame(_account_target_rows()),
        "entity_target": pd.DataFrame(_entity_target_rows()),
        "mapping": pd.DataFrame(_mapping_rows()),
        "rules": pd.DataFrame(_rule_rows()),
    }


@pytest.fixture
def workflow(test_settings):
    account_source_path = test_settings.source_dir / "Account_Hierarchy_Source.xlsx"
    entity_source_path = test_settings.source_dir / "Entity_Hierarchy_Source.xlsx"
    mapping_path = test_settings.source_dir / "Hierarchy_Mapping.xlsx"
    rules_path = test_settings.source_dir / "Validation_Rules.xlsx"
    account_target_path = test_settings.target_upload_dir / "Account_Hierarchy_Target.xlsx"
    entity_target_path = test_settings.target_upload_dir / "Entity_Hierarchy_Target.xlsx"

    with pd.ExcelWriter(account_source_path, engine="openpyxl") as writer:
        pd.DataFrame(_account_source_rows()).to_excel(writer, index=False, sheet_name="AccountHierarchy")
    with pd.ExcelWriter(entity_source_path, engine="openpyxl") as writer:
        pd.DataFrame(_entity_source_rows()).to_excel(writer, index=False, sheet_name="EntityHierarchy")
    with pd.ExcelWriter(mapping_path, engine="openpyxl") as writer:
        pd.DataFrame(_mapping_rows()).to_excel(writer, index=False, sheet_name="HierarchyMappings")
    with pd.ExcelWriter(rules_path, engine="openpyxl") as writer:
        pd.DataFrame(_rule_rows()).to_excel(writer, index=False, sheet_name="ValidationRules")
    with pd.ExcelWriter(account_target_path, engine="openpyxl") as writer:
        pd.DataFrame(_account_target_rows()).to_excel(writer, index=False, sheet_name="AccountHierarchy")
    with pd.ExcelWriter(entity_target_path, engine="openpyxl") as writer:
        pd.DataFrame(_entity_target_rows()).to_excel(writer, index=False, sheet_name="EntityHierarchy")

    validation_workflow = ValidationWorkflow(test_settings)
    validation_workflow.ingest_excel_files(auto_build_index=False)
    return validation_workflow
