from __future__ import annotations

import pandas as pd

from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.ingestion.excel_ingestor import ExcelIngestionService
from hierarchy_migration_validation_agent.schemas import ValidationRequest


def _account_source_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "account", "member_code": "ACC_100", "member_name": "Total_Revenue", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Top revenue rollup"},
        {"dimension": "account", "member_code": "ACC_110", "member_name": "Product_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "All product revenue"},
        {"dimension": "account", "member_code": "ACC_120", "member_name": "Service_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": True, "sort_order": 3, "member_description": "Professional services revenue"},
        {"dimension": "account", "member_code": "ACC_111", "member_name": "Subscription_Rev", "parent_name": "Product_Revenue", "level": 2, "leaf_flag": True, "sort_order": 4, "member_description": "Recurring subscription revenue"},
    ]


def _entity_source_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "entity", "member_code": "ENT_100", "member_name": "Global_Corp", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Global legal hierarchy"},
        {"dimension": "entity", "member_code": "ENT_110", "member_name": "North_America", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "North America rollup"},
        {"dimension": "entity", "member_code": "ENT_111", "member_name": "US_Business", "parent_name": "North_America", "level": 2, "leaf_flag": False, "sort_order": 3, "member_description": "US operating entities"},
        {"dimension": "entity", "member_code": "ENT_112", "member_name": "California_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 4, "member_description": "California business unit"},
    ]


def _mapping_rows() -> list[dict[str, object]]:
    return [
        {"dimension": "account", "source_member_name": "Total_Revenue", "target_member_name": "Total_Revenue", "mapping_status": "active", "mapping_rule": "exact_match", "notes": "Account mapping"},
        {"dimension": "account", "source_member_name": "Product_Revenue", "target_member_name": "Product_Revenue", "mapping_status": "active", "mapping_rule": "exact_match", "notes": "Account mapping"},
        {"dimension": "entity", "source_member_name": "Global_Corp", "target_member_name": "Global_Corp", "mapping_status": "active", "mapping_rule": "exact_match", "notes": "Entity mapping"},
        {"dimension": "entity", "source_member_name": "North_America", "target_member_name": "North_America", "mapping_status": "active", "mapping_rule": "exact_match", "notes": "Entity mapping"},
    ]


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
    rows = []
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
    return rows


def test_ingest_directory_detects_workbook_datasets(test_settings):
    account_path = test_settings.source_dir / "Account_Hierarchy_Source.xlsx"
    entity_path = test_settings.source_dir / "Entity_Hierarchy_Source.xlsx"
    mapping_path = test_settings.source_dir / "Hierarchy_Mapping.xlsx"
    rules_path = test_settings.source_dir / "Validation_Rules.xlsx"

    with pd.ExcelWriter(account_path, engine="openpyxl") as writer:
        pd.DataFrame(_account_source_rows()).to_excel(writer, index=False, sheet_name="AccountHierarchy")
    with pd.ExcelWriter(entity_path, engine="openpyxl") as writer:
        pd.DataFrame(_entity_source_rows()).to_excel(writer, index=False, sheet_name="EntityHierarchy")
    with pd.ExcelWriter(mapping_path, engine="openpyxl") as writer:
        pd.DataFrame(_mapping_rows()).to_excel(writer, index=False, sheet_name="HierarchyMappings")
    with pd.ExcelWriter(rules_path, engine="openpyxl") as writer:
        pd.DataFrame(_rule_rows()).to_excel(writer, index=False, sheet_name="ValidationRules")

    ingestion = ExcelIngestionService(test_settings).ingest_directory()
    normalized_names = {artifact.name for artifact in ingestion.normalized_files}

    assert normalized_names == {"account_source", "entity_source", "mapping", "rules"}
    assert all(artifact.path.startswith("in-memory://") for artifact in ingestion.normalized_files)
    assert len(ingestion.source_files) == 4


def test_ingest_single_multi_sheet_workbook(test_settings):
    workbook_path = test_settings.source_dir / "Finance_Migration_Input.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(_account_source_rows()).to_excel(
            writer,
            index=False,
            sheet_name="smartview_report_account",
        )
        pd.DataFrame(_entity_source_rows()).to_excel(
            writer,
            index=False,
            sheet_name="smartview_report_entity",
        )
        pd.DataFrame(_mapping_rows()).to_excel(
            writer,
            index=False,
            sheet_name="hierarchy_mapping_tab",
        )
        pd.DataFrame(_rule_rows()).to_excel(
            writer,
            index=False,
            sheet_name="validation_rules_tab",
        )

    artifacts, uploaded_files, warnings = ExcelIngestionService(test_settings).ingest_files([workbook_path])
    normalized_names = {artifact.name for artifact in artifacts}

    assert normalized_names == {"account_source", "entity_source", "mapping", "rules"}
    assert all(artifact.path.startswith("in-memory://") for artifact in artifacts)
    assert uploaded_files == [str(workbook_path)]
    assert warnings == []


def test_ingest_path_style_hierarchy_workbook(test_settings):
    workbook_path = test_settings.source_dir / "HCM_SmartView_100Rows.xlsx"
    hierarchy_view = pd.DataFrame(
        [
            {
                "Level 1 (Entity)": "GlobalCorp",
                "Level 2 (BU)": "North America",
                "Level 3 (Dept)": "Finance",
                "Level 4 (Cost Center)": "FIN100",
            },
            {
                "Level 1 (Entity)": "GlobalCorp",
                "Level 2 (BU)": "Europe",
                "Level 3 (Dept)": "HR",
                "Level 4 (Cost Center)": "HR200",
            },
        ]
    )
    smartview_report = pd.DataFrame(
        [
            {
                "Entity": "GlobalCorp",
                "Business Unit": "North America",
                "Department": "Finance",
                "Headcount": 10,
                "Total Salary": 1000000,
            }
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        hierarchy_view.to_excel(writer, index=False, sheet_name="Hierarchy_View")
        smartview_report.to_excel(writer, index=False, sheet_name="SmartView_Report")

    payload = ExcelIngestionService(test_settings).parse_files([workbook_path])
    normalized_names = {artifact.name for artifact in payload.artifacts}
    entity_source = payload.frames["entity_source"]
    entity_measures = payload.frames["entity_measure_source"]

    assert normalized_names == {"entity_source", "entity_measure_source"}
    assert payload.uploaded_files == [str(workbook_path)]
    assert payload.warnings == []
    assert set(entity_source["member_name"]) >= {"GlobalCorp", "North America", "Europe", "Finance", "HR", "FIN100", "HR200"}
    assert set(entity_measures["dataset_name"]) == {"smartview_report"}
    assert set(entity_measures.columns) >= {"entity", "business_unit", "department", "headcount", "total_salary"}


def test_ingest_source_and_target_workbooks_auto_builds_rag_and_validates(test_settings):
    workflow = ValidationWorkflow(test_settings)

    source_workbook_path = test_settings.source_dir / "Source_Hierarchy.xlsx"
    target_workbook_path = test_settings.target_upload_dir / "Target_Hierarchy.xlsx"

    source_hierarchy = pd.DataFrame(
        [
            {
                "Level 1 (Entity)": "GlobalCorp",
                "Level 2 (BU)": "North America",
                "Level 3 (Dept)": "Finance",
                "Level 4 (Cost Center)": "FIN100",
            },
            {
                "Level 1 (Entity)": "GlobalCorp",
                "Level 2 (BU)": "Europe",
                "Level 3 (Dept)": "HR",
                "Level 4 (Cost Center)": "HR200",
            },
        ]
    )
    target_hierarchy = source_hierarchy.copy()

    with pd.ExcelWriter(source_workbook_path, engine="openpyxl") as writer:
        source_hierarchy.to_excel(writer, index=False, sheet_name="Hierarchy_View")
    with pd.ExcelWriter(target_workbook_path, engine="openpyxl") as writer:
        target_hierarchy.to_excel(writer, index=False, sheet_name="Hierarchy_View")

    ingestion = workflow.ingest_excel_files(
        source_file_paths=[source_workbook_path],
        target_file_paths=[target_workbook_path],
        auto_build_index=True,
    )
    report = workflow.validate(ValidationRequest(message="Validate uploaded hierarchies"))
    previews = workflow.preview_dataframes()

    assert {artifact.name for artifact in ingestion.normalized_files} == {"entity_source", "entity_target"}
    assert all(artifact.path.startswith("in-memory://") for artifact in ingestion.normalized_files)
    assert ingestion.rag_document_count > 0
    assert report.dimensions == ["entity"]
    assert report.summary["total_checks"] >= 6
    assert previews["source_entity"].empty is False
    assert previews["target_entity"].empty is False
    assert not (test_settings.hierarchies_dir / "dim_entity_hierarchy.csv").exists()


def test_ingest_mapping_sheet_without_dimension_infers_entity(test_settings):
    workbook_path = test_settings.source_dir / "Entity_Multi_Tab.xlsx"
    hierarchy_view = pd.DataFrame(
        [
            {
                "Level 1 (Entity)": "GlobalCorp",
                "Level 2 (BU)": "North America",
                "Level 3 (Dept)": "Finance",
                "Level 4 (Cost Center)": "FIN100",
            }
        ]
    )
    mapping_sheet = pd.DataFrame(
        [
            {
                "Source Member": "Finance",
                "Target Member": "Finance",
                "Mapping Status": "active",
            },
            {
                "Source Member": "FIN100",
                "Target Member": "FIN100",
                "Mapping Status": "active",
            },
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        hierarchy_view.to_excel(writer, index=False, sheet_name="Hierarchy_View")
        mapping_sheet.to_excel(writer, index=False, sheet_name="Entity_Mapping")

    payload = ExcelIngestionService(test_settings).parse_files([workbook_path])
    normalized_names = {artifact.name for artifact in payload.artifacts}
    mapping = payload.frames["mapping"]

    assert normalized_names == {"entity_source", "mapping"}
    assert payload.warnings == []
    assert set(mapping["dimension"]) == {"entity"}
    assert set(mapping["source_member_name"]) == {"Finance", "FIN100"}


def test_ingest_generic_path_hierarchy_columns_as_entity(test_settings):
    workbook_path = test_settings.source_dir / "Generic_Org_Hierarchy.xlsx"
    generic_hierarchy = pd.DataFrame(
        [
            {"Company": "Acme", "Division": "Retail", "Team": "Store Ops"},
            {"Company": "Acme", "Division": "Retail", "Team": "Merchandising"},
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        generic_hierarchy.to_excel(writer, index=False, sheet_name="OrgStructure")

    payload = ExcelIngestionService(test_settings).parse_files([workbook_path])

    assert {artifact.name for artifact in payload.artifacts} == {"entity_source"}
    assert set(payload.frames["entity_source"]["member_name"]) >= {"Acme", "Retail", "Store Ops", "Merchandising"}


def test_ingest_parent_child_alias_hierarchy_columns(test_settings):
    workbook_path = test_settings.source_dir / "Alias_Hierarchy.xlsx"
    alias_hierarchy = pd.DataFrame(
        [
            {"Child": "Acme", "Parent": None, "Depth": 0, "Is Terminal": False},
            {"Child": "Retail", "Parent": "Acme", "Depth": 1, "Is Terminal": False},
            {"Child": "Store Ops", "Parent": "Retail", "Depth": 2, "Is Terminal": True},
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        alias_hierarchy.to_excel(writer, index=False, sheet_name="HierarchyAlias")

    payload = ExcelIngestionService(test_settings).parse_files([workbook_path])

    assert {artifact.name for artifact in payload.artifacts} == {"entity_source"}
    assert set(payload.frames["entity_source"]["member_name"]) == {"Acme", "Retail", "Store Ops"}


def test_ingest_parent_child_hierarchy_without_explicit_level_or_leaf(test_settings):
    workbook_path = test_settings.source_dir / "Parent_Child_Minimal.xlsx"
    parent_child = pd.DataFrame(
        [
            {"Node Name": "GlobalCorp", "Parent Node": None},
            {"Node Name": "North America", "Parent Node": "GlobalCorp"},
            {"Node Name": "Finance", "Parent Node": "North America"},
        ]
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        parent_child.to_excel(writer, index=False, sheet_name="OrgNodes")

    payload = ExcelIngestionService(test_settings).parse_files([workbook_path])
    entity_source = payload.frames["entity_source"].sort_values("level").reset_index(drop=True)

    assert {artifact.name for artifact in payload.artifacts} == {"entity_source"}
    assert entity_source[["member_name", "parent_name", "level", "leaf_flag"]].to_dict(orient="records") == [
        {"member_name": "GlobalCorp", "parent_name": None, "level": 0, "leaf_flag": False},
        {"member_name": "North America", "parent_name": "GlobalCorp", "level": 1, "leaf_flag": False},
        {"member_name": "Finance", "parent_name": "North America", "level": 2, "leaf_flag": True},
    ]


def test_title_row_org_workbook_same_source_and_target_passes(test_settings):
    workflow = ValidationWorkflow(test_settings)
    source_workbook_path = test_settings.source_dir / "Asteron_Like_Source.xlsx"
    target_workbook_path = test_settings.target_upload_dir / "Asteron_Like_Target.xlsx"

    org_spine_rows = [
        ["Organization Spine Mapping", None, None, None, None, None, None],
        [None, None, None, None, None, None, None],
        ["Region", "Division", "Function", "Team", "Default Ledger", "Typical Entity", "Default Currency"],
        ["Americas", "Corporate Services", "Finance Ops", "Payroll Control", "CORP_US", "Asteron Holdings US", "USD"],
        ["Americas", "Corporate Services", "People Operations", "Talent Acquisition", "CORP_US", "Asteron Holdings US", "USD"],
    ]

    with pd.ExcelWriter(source_workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(org_spine_rows).to_excel(writer, index=False, header=False, sheet_name="Org_Spine_Map")
    with pd.ExcelWriter(target_workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(org_spine_rows).to_excel(writer, index=False, header=False, sheet_name="Org_Spine_Map")

    ingestion = workflow.ingest_excel_files(
        source_file_paths=[source_workbook_path],
        target_file_paths=[target_workbook_path],
        auto_build_index=False,
    )
    report = workflow.validate(ValidationRequest(message="Validate uploaded hierarchies"))
    previews = workflow.preview_dataframes()

    assert {artifact.name for artifact in ingestion.normalized_files} == {"entity_source", "entity_target"}
    assert report.overall_status == "PASSED"
    assert report.summary["failed_checks"] == 0
    assert "0.05" not in set(previews["source_entity"]["member_name"])
    assert "Region" not in set(previews["source_entity"]["member_name"])


def test_measure_only_workbook_detects_changed_target_value(test_settings):
    workflow = ValidationWorkflow(test_settings)
    source_workbook_path = test_settings.source_dir / "Measure_Only_Source.xlsx"
    target_workbook_path = test_settings.target_upload_dir / "Measure_Only_Target.xlsx"

    source_rows = pd.DataFrame(
        [
            {
                "Legal Entity": "GlobalCorp US",
                "Company Code": "US01",
                "Worker ID": "E1001",
                "Salary": 95000,
                "Bonus Target": 5000,
            }
        ]
    )
    target_rows = source_rows.copy()
    target_rows.loc[0, "Salary"] = 120000

    with pd.ExcelWriter(source_workbook_path, engine="openpyxl") as writer:
        source_rows.to_excel(writer, index=False, sheet_name="SmartView_Report")
    with pd.ExcelWriter(target_workbook_path, engine="openpyxl") as writer:
        target_rows.to_excel(writer, index=False, sheet_name="SmartView_Report")

    ingestion = workflow.ingest_excel_files(
        source_file_paths=[source_workbook_path],
        target_file_paths=[target_workbook_path],
        auto_build_index=False,
    )
    report = workflow.validate(ValidationRequest(message="Validate uploaded source and target workbooks"))

    assert {artifact.name for artifact in ingestion.normalized_files} == {"entity_measure_source", "entity_measure_target"}
    assert report.dimensions == ["entity"]
    assert report.overall_status == "FAILED"
    assert any(result.check_type == "numeric_value_match" and result.failed_count == 1 for result in report.results)
