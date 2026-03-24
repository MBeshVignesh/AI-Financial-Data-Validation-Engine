from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hierarchy_migration_validation_agent.agent.reasoner import AgentReasoner
from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.rag import indexer as rag_indexer
from hierarchy_migration_validation_agent.schemas import ValidationRequest
from hierarchy_migration_validation_agent.utils.io import write_json, write_text


class DummyEmbeddingFunction:
    """Fast deterministic local embedding stub for reproducible evaluation runs."""

    model_name = "dummy-eval-embed"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in input]

    @staticmethod
    def name() -> str:
        return "sentence_transformer::dummy-eval-embed"

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
    def build_from_config(cls, config: dict[str, Any]) -> "DummyEmbeddingFunction":
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
        checksum = float(sum(ord(char) for char in text) % 997)
        return [length, checksum]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.embed_query(document) for document in documents]


ACCOUNT_ROWS = [
    {"dimension": "account", "member_code": "ACC_100", "member_name": "Total_Revenue", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Top revenue rollup"},
    {"dimension": "account", "member_code": "ACC_110", "member_name": "Product_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "All product revenue"},
    {"dimension": "account", "member_code": "ACC_120", "member_name": "Service_Revenue", "parent_name": "Total_Revenue", "level": 1, "leaf_flag": True, "sort_order": 3, "member_description": "Services revenue"},
    {"dimension": "account", "member_code": "ACC_111", "member_name": "Subscription_Rev", "parent_name": "Product_Revenue", "level": 2, "leaf_flag": True, "sort_order": 4, "member_description": "Subscription revenue"},
    {"dimension": "account", "member_code": "ACC_112", "member_name": "License_Rev", "parent_name": "Product_Revenue", "level": 2, "leaf_flag": True, "sort_order": 5, "member_description": "License revenue"},
    {"dimension": "account", "member_code": "ACC_200", "member_name": "Total_Expense", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 6, "member_description": "Top expense rollup"},
    {"dimension": "account", "member_code": "ACC_210", "member_name": "Operating_Expense", "parent_name": "Total_Expense", "level": 1, "leaf_flag": False, "sort_order": 7, "member_description": "Operating expense bucket"},
    {"dimension": "account", "member_code": "ACC_211", "member_name": "Salary_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 8, "member_description": "Salary expense"},
    {"dimension": "account", "member_code": "ACC_212", "member_name": "Rent_Expense", "parent_name": "Operating_Expense", "level": 2, "leaf_flag": True, "sort_order": 9, "member_description": "Rent expense"},
]

ENTITY_ROWS = [
    {"dimension": "entity", "member_code": "ENT_100", "member_name": "Global_Corp", "parent_name": None, "level": 0, "leaf_flag": False, "sort_order": 1, "member_description": "Global legal hierarchy"},
    {"dimension": "entity", "member_code": "ENT_110", "member_name": "North_America", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 2, "member_description": "North America rollup"},
    {"dimension": "entity", "member_code": "ENT_120", "member_name": "Europe", "parent_name": "Global_Corp", "level": 1, "leaf_flag": False, "sort_order": 3, "member_description": "Europe rollup"},
    {"dimension": "entity", "member_code": "ENT_111", "member_name": "US_Business", "parent_name": "North_America", "level": 2, "leaf_flag": False, "sort_order": 4, "member_description": "US business"},
    {"dimension": "entity", "member_code": "ENT_112", "member_name": "California_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 5, "member_description": "California business unit"},
    {"dimension": "entity", "member_code": "ENT_113", "member_name": "Texas_BU", "parent_name": "US_Business", "level": 3, "leaf_flag": True, "sort_order": 6, "member_description": "Texas business unit"},
    {"dimension": "entity", "member_code": "ENT_121", "member_name": "Germany_BU", "parent_name": "Europe", "level": 2, "leaf_flag": True, "sort_order": 7, "member_description": "Germany business unit"},
]

ENTITY_MEASURE_ROWS = [
    {"Entity": "Global_Corp", "Business Unit": "North_America", "Department": "US_Business", "Cost Center": "California_BU", "Headcount": 120, "Total Salary": 1450000},
    {"Entity": "Global_Corp", "Business Unit": "North_America", "Department": "US_Business", "Cost Center": "Texas_BU", "Headcount": 95, "Total Salary": 990000},
    {"Entity": "Global_Corp", "Business Unit": "Europe", "Department": "Europe", "Cost Center": "Germany_BU", "Headcount": 88, "Total Salary": 1015000},
]


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    title: str
    source_sheets: list[tuple[str, pd.DataFrame, bool]]
    target_sheets: list[tuple[str, pd.DataFrame, bool]]
    expected_status: str
    expected_rule_keys: set[str]
    expected_members: set[str]


def main() -> None:
    args = _parse_args()
    eval_result = run_evaluation(judge_mode=args.judge)

    if args.output:
        output_path = Path(args.output)
        write_json(output_path, eval_result)
        print(f"Saved JSON evaluation report to {output_path}")

    if args.markdown:
        markdown_path = Path(args.markdown)
        write_text(markdown_path, render_markdown(eval_result))
        print(f"Saved markdown evaluation report to {markdown_path}")

    if args.format == "json":
        print(json.dumps(eval_result, indent=2))
    else:
        print(render_console_summary(eval_result))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 5-case AI Financial Data Validation Engine (Hierarchical and Row-Level) benchmark."
    )
    parser.add_argument(
        "--judge",
        choices=["auto", "ollama", "rubric"],
        default="auto",
        help="Use Ollama as judge when available, or fall back to the deterministic rubric.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Console output format.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a JSON report. Defaults to data/reports/evaluations/<timestamp>.json when omitted.",
    )
    parser.add_argument(
        "--markdown",
        help="Optional path for a markdown report. Defaults to data/reports/evaluations/<timestamp>.md when omitted.",
    )
    args = parser.parse_args()

    if not args.output or not args.markdown:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = ROOT_DIR / "data" / "reports" / "evaluations"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if not args.output:
            args.output = str(eval_dir / f"eval_{timestamp}.json")
        if not args.markdown:
            args.markdown = str(eval_dir / f"eval_{timestamp}.md")
    return args


def run_evaluation(*, judge_mode: str) -> dict[str, Any]:
    rag_indexer.create_embedding_function = lambda settings: DummyEmbeddingFunction()

    base_settings = Settings.from_root(ROOT_DIR)
    ollama_available = _ollama_available(base_settings)
    if judge_mode in {"auto", "ollama"} and not ollama_available:
        AgentReasoner._try_ollama = lambda self, report: None  # type: ignore[method-assign]

    cases = build_cases()
    case_results: list[dict[str, Any]] = []

    for case in cases:
        result = _run_case(case)
        if judge_mode == "rubric":
            judge = score_case_with_rubric(case, result)
        elif judge_mode == "ollama":
            judge = score_case_with_ollama(base_settings, case, result)
        else:
            judge = score_case_with_ollama(base_settings, case, result) if ollama_available else score_case_with_rubric(case, result)
        result["judge"] = judge
        case_results.append(result)

    total_score = sum(case["judge"]["score"] for case in case_results)
    max_score = len(case_results) * 10
    accuracy_percentage = round((total_score / max_score) * 100, 2) if max_score else 0.0

    return {
        "generated_at": datetime.now().isoformat(),
        "judge_mode_requested": judge_mode,
        "judge_mode_used": "ollama" if any(case["judge"]["judge_method"] == "ollama" for case in case_results) else "rubric",
        "embedding_mode": "deterministic_dummy_for_reproducible_eval",
        "cases": case_results,
        "aggregate": {
            "cases_run": len(case_results),
            "total_score": total_score,
            "max_score": max_score,
            "accuracy_percentage": accuracy_percentage,
        },
    }


def build_cases() -> list[CaseDefinition]:
    from hierarchy_migration_validation_agent.validation.rule_catalog import default_rules_for_dimensions

    shared_rules = pd.DataFrame([rule.model_dump() for rule in default_rules_for_dimensions(["account", "entity"])])

    case_1 = CaseDefinition(
        name="case_1_happy_path_multi_hierarchy",
        title="Happy path multi-hierarchy",
        source_sheets=[
            ("AccountHierarchy", pd.DataFrame(ACCOUNT_ROWS), True),
            ("EntityHierarchy", pd.DataFrame(ENTITY_ROWS), True),
            ("HierarchyMappings", pd.DataFrame(mapping_rows()), True),
            ("ValidationRules", shared_rules, True),
            ("SmartView_Report", pd.DataFrame(ENTITY_MEASURE_ROWS), True),
        ],
        target_sheets=[
            ("AccountHierarchy", pd.DataFrame(ACCOUNT_ROWS), True),
            ("EntityHierarchy", pd.DataFrame(ENTITY_ROWS), True),
            ("SmartView_Report", pd.DataFrame(ENTITY_MEASURE_ROWS), True),
        ],
        expected_status="PASSED",
        expected_rule_keys=set(),
        expected_members=set(),
    )

    account_defects = pd.DataFrame(ACCOUNT_ROWS).copy()
    account_defects = account_defects.loc[account_defects["member_name"] != "Subscription_Rev"].copy()
    account_defects.loc[account_defects["member_name"] == "License_Rev", "parent_name"] = "Total_Revenue"
    account_defects.loc[account_defects["member_name"] == "Total_Expense", "leaf_flag"] = True
    account_defects.loc[account_defects["member_name"] == "Rent_Expense", "level"] = 1
    account_defects = pd.concat(
        [
            account_defects,
            account_defects.loc[account_defects["member_name"] == "Salary_Expense"].assign(
                member_code="ACC_211_DUP",
                member_description="duplicate salary",
            ),
        ],
        ignore_index=True,
    )
    entity_defects = pd.DataFrame(ENTITY_ROWS).copy()
    entity_defects.loc[entity_defects["member_name"] == "California_BU", "parent_name"] = "West_Region"
    case_2 = CaseDefinition(
        name="case_2_structural_defects",
        title="Structural defect mix",
        source_sheets=[
            ("AccountHierarchy", pd.DataFrame(ACCOUNT_ROWS), True),
            ("EntityHierarchy", pd.DataFrame(ENTITY_ROWS), True),
            ("HierarchyMappings", pd.DataFrame(mapping_rows()), True),
        ],
        target_sheets=[
            ("AccountHierarchy", account_defects, True),
            ("EntityHierarchy", entity_defects, True),
        ],
        expected_status="FAILED",
        expected_rule_keys={
            "account:missing_members_in_target",
            "account:parent_mismatch",
            "account:duplicate_members",
            "account:leaf_flag_consistency",
            "account:level_consistency",
            "account:row_level_match",
            "account:rollup_preservation",
            "entity:parent_existence",
            "entity:parent_mismatch",
            "entity:row_level_match",
        },
        expected_members={
            "Subscription_Rev",
            "License_Rev",
            "Salary_Expense",
            "Total_Expense",
            "Rent_Expense",
            "Total_Revenue",
            "California_BU",
        },
    )

    hcm_source = pd.DataFrame(
        [
            {"Level 1 (Entity)": "GlobalCorp", "Level 2 (BU)": "Corporate Services", "Level 3 (Dept)": "Finance Ops", "Level 4 (Cost Center)": "Payroll Control"},
            {"Level 1 (Entity)": "GlobalCorp", "Level 2 (BU)": "Corporate Services", "Level 3 (Dept)": "People Ops", "Level 4 (Cost Center)": "Talent Acquisition"},
        ]
    )
    hcm_measures_source = pd.DataFrame(
        [
            {"Entity": "GlobalCorp", "Business Unit": "Corporate Services", "Department": "Finance Ops", "Cost Center": "Payroll Control", "Headcount": 42, "Total Salary": 4200000},
            {"Entity": "GlobalCorp", "Business Unit": "Corporate Services", "Department": "People Ops", "Cost Center": "Talent Acquisition", "Headcount": 18, "Total Salary": 1620000},
        ]
    )
    hcm_measures_target = hcm_measures_source.copy()
    hcm_measures_target.loc[0, "Headcount"] = 39
    hcm_measures_target.loc[1, "Total Salary"] = 1710000
    case_3 = CaseDefinition(
        name="case_3_hcm_numeric_mismatch",
        title="HCM numeric mismatch",
        source_sheets=[
            ("Hierarchy_View", hcm_source, True),
            ("SmartView_Report", hcm_measures_source, True),
        ],
        target_sheets=[
            ("Hierarchy_View", hcm_source, True),
            ("SmartView_Report", hcm_measures_target, True),
        ],
        expected_status="FAILED",
        expected_rule_keys={"entity:numeric_value_match"},
        expected_members={
            "entity=GlobalCorp, business_unit=Corporate Services, department=Finance Ops, cost_center=Payroll Control",
            "entity=GlobalCorp, business_unit=Corporate Services, department=People Ops, cost_center=Talent Acquisition",
        },
    )

    case_4 = CaseDefinition(
        name="case_4_mapping_gap_only",
        title="Mapping completeness gap",
        source_sheets=[
            ("AccountHierarchy", pd.DataFrame(ACCOUNT_ROWS), True),
            ("EntityHierarchy", pd.DataFrame(ENTITY_ROWS), True),
            ("HierarchyMappings", pd.DataFrame(mapping_rows(missing={"Germany_BU", "License_Rev"})), True),
        ],
        target_sheets=[
            ("AccountHierarchy", pd.DataFrame(ACCOUNT_ROWS), True),
            ("EntityHierarchy", pd.DataFrame(ENTITY_ROWS), True),
        ],
        expected_status="FAILED",
        expected_rule_keys={"account:mapping_completeness", "entity:mapping_completeness"},
        expected_members={"License_Rev", "Germany_BU"},
    )

    org_spine_source = pd.DataFrame(
        [
            ["Organization Spine Mapping", None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            ["Region", "Division", "Function", "Team", "Default Ledger", "Typical Entity", "Default Currency"],
            ["Americas", "Corporate Services", "Finance Ops", "Payroll Control", "CORP_US", "Asteron Holdings US", "USD"],
            ["Americas", "Corporate Services", "People Operations", "Talent Acquisition", "CORP_US", "Asteron Holdings US", "USD"],
            ["EMEA", "Shared Services", "Finance Ops", "Treasury Control", "CORP_UK", "Asteron Holdings UK", "GBP"],
        ]
    )
    org_spine_target = pd.DataFrame(
        [
            ["Organization Spine Mapping", None, None, None, None, None, None],
            [None, None, None, None, None, None, None],
            ["Region", "Division", "Function", "Team", "Default Ledger", "Typical Entity", "Default Currency"],
            ["Americas", "Corporate Services", "Finance Ops", "Payroll Control", "CORP_US", "Asteron Holdings US", "USD"],
            ["Americas", "Corporate Services", "People Operations", "Talent Acquisition", "CORP_US", "Asteron Holdings US", "USD"],
            ["EMEA", "Shared Services", "Finance Ops", "Cash Ops", "CORP_UK", "Asteron Holdings UK", "GBP"],
            ["APAC", "Operations", "Finance Ops", "Sydney Control", "CORP_AU", "Asteron Holdings AU", "AUD"],
        ]
    )
    case_5 = CaseDefinition(
        name="case_5_title_row_edge_case",
        title="Title-row edge case",
        source_sheets=[("Org_Spine", org_spine_source, False)],
        target_sheets=[("Org_Spine", org_spine_target, False)],
        expected_status="FAILED",
        expected_rule_keys={"entity:row_level_match"},
        expected_members={"APAC", "Operations", "Finance Ops"},
    )

    return [case_1, case_2, case_3, case_4, case_5]


def mapping_rows(*, missing: set[str] | None = None) -> list[dict[str, str]]:
    missing = missing or set()
    rows: list[dict[str, str]] = []
    for row in ACCOUNT_ROWS + ENTITY_ROWS:
        if row["member_name"] in missing:
            continue
        rows.append(
            {
                "dimension": row["dimension"],
                "source_member_name": row["member_name"],
                "target_member_name": row["member_name"],
                "mapping_status": "active",
                "mapping_rule": "exact_match",
                "notes": "Evaluation mapping",
            }
        )
    return rows


def _run_case(case: CaseDefinition) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"{case.name}_") as tmpdir:
        root = Path(tmpdir)
        settings = Settings.from_root(root)
        settings.ensure_directories()
        workflow = ValidationWorkflow(settings)

        source_path = settings.source_dir / f"{case.name}_source.xlsx"
        target_path = settings.target_upload_dir / f"{case.name}_target.xlsx"
        _write_workbook(source_path, case.source_sheets)
        _write_workbook(target_path, case.target_sheets)

        ingestion = workflow.ingest_excel_files(
            source_file_paths=[source_path],
            target_file_paths=[target_path],
            auto_build_index=True,
        )
        report = workflow.validate(
            ValidationRequest(
                message="Validate uploaded source and target hierarchy workbooks across account and entity hierarchies",
                rebuild_index=True,
            )
        )
        actual_rule_keys = {
            f"{result.dimension}:{result.check_type}"
            for result in report.results
            if result.failed_count
        }
        actual_members = {
            failure.member_name
            for result in report.results
            for failure in result.failed_records
            if failure.member_name
        }
        return {
            "name": case.name,
            "title": case.title,
            "ingestion_warnings": ingestion.warnings,
            "rag_document_count": ingestion.rag_document_count,
            "overall_status": report.overall_status,
            "summary": report.summary,
            "dimensions": report.dimensions,
            "actual_rule_keys": sorted(actual_rule_keys),
            "actual_members": sorted(actual_members),
            "agent_explanation": report.agent_explanation,
            "json_report_path": report.json_report_path,
            "markdown_report_path": report.markdown_report_path,
        }


def _write_workbook(path: Path, sheets: list[tuple[str, pd.DataFrame, bool]]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame, include_header in sheets:
            frame.to_excel(writer, index=False, header=include_header, sheet_name=sheet_name)


def score_case_with_rubric(case: CaseDefinition, result: dict[str, Any]) -> dict[str, Any]:
    status_points = 4.0 if result["overall_status"] == case.expected_status else 0.0
    rule_points = round(3.0 * _f1(case.expected_rule_keys, set(result["actual_rule_keys"])), 2)
    member_points = round(2.0 * _f1(case.expected_members, set(result["actual_members"])), 2)
    explanation_points = 1.0 if _explanation_is_usable(result["agent_explanation"], case.expected_status) else 0.0
    score = round(status_points + rule_points + member_points + explanation_points, 2)

    misses: list[str] = []
    missing_rules = sorted(case.expected_rule_keys - set(result["actual_rule_keys"]))
    unexpected_rules = sorted(set(result["actual_rule_keys"]) - case.expected_rule_keys)
    if missing_rules:
        misses.append(f"Missing expected failed checks: {', '.join(missing_rules)}")
    if unexpected_rules and case.expected_status == "PASSED":
        misses.append(f"Unexpected failed checks: {', '.join(unexpected_rules)}")
    if result["overall_status"] != case.expected_status:
        misses.append(f"Expected overall status {case.expected_status}, got {result['overall_status']}")

    strengths = []
    if status_points:
        strengths.append("Overall pass/fail outcome matched expectation.")
    if rule_points >= 2.5:
        strengths.append("Primary failed checks were detected accurately.")
    if member_points >= 1.5:
        strengths.append("Row-level member attribution was directionally correct.")
    if explanation_points:
        strengths.append("Generated explanation was usable for triage.")

    return {
        "judge_method": "rubric",
        "score": score,
        "score_breakdown": {
            "status_points": status_points,
            "rule_points": rule_points,
            "member_points": member_points,
            "explanation_points": explanation_points,
        },
        "strengths": strengths,
        "misses": misses,
        "reason": "Deterministic rubric used because Ollama judge was unavailable or not requested.",
    }


def score_case_with_ollama(settings: Settings, case: CaseDefinition, result: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        "You are evaluating a hierarchy migration validation system.\n"
        "Score this case from 0 to 10 using this rubric:\n"
        "- 4 points: overall pass/fail status matches expectation\n"
        "- 3 points: primary failed checks match expectation\n"
        "- 2 points: key row-level members/issues match expectation\n"
        "- 1 point: explanation quality is useful and business-friendly\n"
        "Return JSON only with keys score, strengths, misses, reason.\n"
        f"Case title: {case.title}\n"
        f"Expected: {json.dumps({'status': case.expected_status, 'rule_keys': sorted(case.expected_rule_keys), 'members': sorted(case.expected_members)})}\n"
        f"Actual: {json.dumps({'status': result['overall_status'], 'rule_keys': result['actual_rule_keys'], 'members': result['actual_members'], 'summary': result['summary'], 'agent_explanation': result['agent_explanation']})}\n"
    )
    endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {"model": settings.ollama_model, "stream": False, "prompt": prompt}
    try:
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        response_text = body.get("response", "").strip()
        parsed = json.loads(_extract_json_object(response_text))
        parsed["judge_method"] = "ollama"
        parsed["score"] = float(parsed["score"])
        return parsed
    except Exception:
        return score_case_with_rubric(case, result)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge response.")
    return text[start : end + 1]


def _f1(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected:
        return 0.0 if actual else 1.0
    if not actual:
        return 0.0
    intersection = len(expected & actual)
    if intersection == 0:
        return 0.0
    precision = intersection / len(actual)
    recall = intersection / len(expected)
    return 2 * precision * recall / (precision + recall)


def _explanation_is_usable(explanation: str, expected_status: str) -> bool:
    if not explanation.strip():
        return False
    required_sections = ["What Was Tested"]
    if expected_status == "PASSED":
        required_sections.append("Passed")
    else:
        required_sections.extend(["What Failed", "Likely Cause", "Recommended Action"])
    return all(section in explanation for section in required_sections)


def _ollama_available(settings: Settings) -> bool:
    endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with request.urlopen(endpoint, timeout=2) as response:
            return response.status == 200
    except (error.URLError, TimeoutError):
        return False


def render_console_summary(eval_result: dict[str, Any]) -> str:
    lines = [
        "AI Financial Data Validation Engine (Hierarchical and Row-Level)",
        f"Judge mode used: {eval_result['judge_mode_used']}",
        f"Aggregate accuracy: {eval_result['aggregate']['accuracy_percentage']}% "
        f"({eval_result['aggregate']['total_score']}/{eval_result['aggregate']['max_score']})",
        "",
    ]
    for case in eval_result["cases"]:
        judge = case["judge"]
        lines.extend(
            [
                f"{case['title']}: {judge['score']}/10",
                f"  Status: {case['overall_status']}",
                f"  Failed checks: {case['summary']['failed_checks']}",
                f"  Judge: {judge['judge_method']}",
                f"  Reason: {judge['reason']}",
            ]
        )
        if judge["misses"]:
            lines.append(f"  Misses: {' | '.join(judge['misses'])}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_markdown(eval_result: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at: `{eval_result['generated_at']}`",
        f"- Judge mode used: `{eval_result['judge_mode_used']}`",
        f"- Aggregate accuracy: `{eval_result['aggregate']['accuracy_percentage']}%`",
        f"- Total score: `{eval_result['aggregate']['total_score']}/{eval_result['aggregate']['max_score']}`",
        "",
        "## Cases",
        "",
    ]
    for case in eval_result["cases"]:
        judge = case["judge"]
        lines.extend(
            [
                f"### {case['title']}",
                "",
                f"- Score: `{judge['score']}/10`",
                f"- Overall status: `{case['overall_status']}`",
                f"- Failed checks: `{case['summary']['failed_checks']}`",
                f"- Failed records: `{case['summary']['failed_records']}`",
                f"- Judge method: `{judge['judge_method']}`",
                f"- Reason: {judge['reason']}",
            ]
        )
        if judge["strengths"]:
            lines.append(f"- Strengths: {'; '.join(judge['strengths'])}")
        if judge["misses"]:
            lines.append(f"- Misses: {'; '.join(judge['misses'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    main()
