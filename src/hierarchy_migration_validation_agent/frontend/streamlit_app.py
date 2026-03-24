from __future__ import annotations

from pathlib import Path
import sys
import json

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.config import get_settings
from hierarchy_migration_validation_agent.schemas import ValidationRequest
from hierarchy_migration_validation_agent.utils.validation_text import describe_checked_rule, describe_passed_rule

settings = get_settings()


def _get_workflow() -> ValidationWorkflow:
    if "workflow" not in st.session_state:
        st.session_state["workflow"] = ValidationWorkflow(settings)
    return st.session_state["workflow"]


def _flatten_failures(report) -> pd.DataFrame:
    rows = []
    for result in report.results:
        for failure in result.failed_records:
            rows.append(
                {
                    "dimension": failure.dimension,
                    "rule_name": failure.rule_name,
                    "member_name": failure.member_name,
                    "issue": failure.issue,
                    "source_value": failure.source_value,
                    "target_value": failure.target_value,
                }
            )
    return pd.DataFrame(rows)


def _build_validation_summary(report) -> str:
    tested_checks = [
        f"{result.dimension.title()} - {result.rule_name}"
        for result in report.results
    ]
    if report.overall_status == "PASSED":
        return "\n".join(
            [
                "### What Was Tested",
                f"- {', '.join(tested_checks) if tested_checks else 'No validation checks were executed.'}",
                "",
                "### Passed",
                f"- All {report.summary['passed_checks']} passed checks completed successfully.",
                f"- Validation succeeded for {', '.join(report.dimensions)} with no flagged records.",
                "- No failed checks or row-level exceptions were found.",
            ]
        )

    failed_checks = [
        f"{result.dimension.title()} - {result.rule_name} ({result.failed_count} failed records)"
        for result in report.results
        if result.failed_count
    ]
    causes = report.likely_root_causes[:3]
    actions = report.recommended_actions[:3]
    return "\n".join(
        [
            "### What Was Tested",
            f"- {', '.join(tested_checks) if tested_checks else 'No validation checks were executed.'}",
            "",
            "### What Failed",
            f"- {', '.join(failed_checks) if failed_checks else 'No failed checks were recorded.'}",
            f"- {report.summary['failed_checks']} failed checks and {report.summary['failed_records']} failed records were identified.",
            "",
            "### Likely Cause",
            f"- {', '.join(causes) if causes else 'No likely cause was derived.'}",
            "",
            "### Recommended Action",
            f"- {', '.join(actions) if actions else 'No recommended action was derived.'}",
        ]
    )


def _hierarchy_stats(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"nodes": 0, "roots": 0, "leaves": 0, "levels": 0}
    return {
        "nodes": int(len(frame)),
        "roots": int(frame["parent_name"].isna().sum()) if "parent_name" in frame.columns else 0,
        "leaves": int(frame["leaf_flag"].astype(bool).sum()) if "leaf_flag" in frame.columns else 0,
        "levels": int(frame["level"].max() + 1) if "level" in frame.columns and not frame["level"].empty else 0,
    }


def _business_preview_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    preview = frame.copy()
    if "sort_order" in preview.columns:
        preview = preview.sort_values(["sort_order", "level", "member_name"]).reset_index(drop=True)
    tree_node = []
    for row in preview.itertuples(index=False):
        level = int(getattr(row, "level", 0) or 0)
        member_name = str(getattr(row, "member_name", ""))
        tree_node.append(f"{'  ' * level}{member_name}")
    business = pd.DataFrame(
        {
            "Tree Node": tree_node,
            "Node": preview["member_name"],
            "Parent": preview["parent_name"].fillna("ROOT"),
            "Level": preview["level"],
            "Node Type": preview["leaf_flag"].map(lambda value: "Leaf" if bool(value) else "Rollup"),
        }
    )
    return business


def _render_hierarchy_preview(title: str, preview_items: list[tuple[str, pd.DataFrame]], empty_message: str) -> None:
    st.subheader(title)
    rendered = False
    for label, frame in preview_items:
        if frame.empty:
            continue
        rendered = True
        st.markdown(f"**{label}**")
        stats = _hierarchy_stats(frame)
        stat_cols = st.columns(4)
        stat_cols[0].metric("Nodes", stats["nodes"])
        stat_cols[1].metric("Roots", stats["roots"])
        stat_cols[2].metric("Leaves", stats["leaves"])
        stat_cols[3].metric("Levels", stats["levels"])
        st.caption("Tree Node is indented by hierarchy level so you can read the structure more easily.")
        st.dataframe(_business_preview_frame(frame), width="stretch", height=260, hide_index=True)
    if not rendered:
        st.info(empty_message)


def main() -> None:
    st.set_page_config(page_title="AI Financial Data Validation Engine (Hierarchical and Row-Level)", layout="wide")
    workflow = _get_workflow()
    st.title("AI Financial Data Validation Engine (Hierarchical and Row-Level)")
    st.caption("Validate Oracle Smart View hierarchy extracts against local ADLS-style curated targets.")
    cleanup_notice = st.session_state.pop("cleanup_notice", None)
    if cleanup_notice:
        st.info(cleanup_notice)

    with st.sidebar:
        st.subheader("Uploads")
        uploaded_source_file = st.file_uploader(
            "Upload Source Workbook",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Upload the Oracle Smart View source hierarchy workbook.",
        )
        uploaded_target_file = st.file_uploader(
            "Upload Target Workbook",
            type=["xlsx"],
            accept_multiple_files=False,
            help="Upload the target hierarchy workbook exported from the curated target.",
        )
        if st.button("Ingest Uploaded Excel", width="stretch"):
            try:
                source_paths = None
                target_paths = None
                if uploaded_source_file:
                    source_destination = settings.source_dir / uploaded_source_file.name
                    source_destination.write_bytes(uploaded_source_file.getvalue())
                    source_paths = [source_destination]
                if uploaded_target_file:
                    target_destination = settings.target_upload_dir / uploaded_target_file.name
                    target_destination.write_bytes(uploaded_target_file.getvalue())
                    target_paths = [target_destination]

                ingestion = workflow.ingest_excel_files(
                    source_file_paths=source_paths,
                    target_file_paths=target_paths,
                    auto_build_index=True,
                )
                st.session_state.pop("report", None)
                rag_status = workflow.rag_service.storage_status()
                rag_build_failed = any(
                    warning.startswith("Automatic RAG build failed during ingest:")
                    for warning in ingestion.warnings
                )
                detected_datasets = ", ".join(artifact.name for artifact in ingestion.normalized_files)
                st.success(
                    f"Parsed {len(ingestion.normalized_files)} workbook datasets directly and indexed "
                    f"{ingestion.rag_document_count} RAG documents."
                )
                if detected_datasets:
                    st.caption(f"Detected datasets: {detected_datasets}")
                if not rag_build_failed:
                    st.caption(
                        f"RAG collection `{rag_status['collection_name']}` "
                        f"with {rag_status['file_count']} files."
                    )
                for warning in ingestion.warnings:
                    st.warning(warning)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.info(
                    "Supported layouts include standard hierarchy sheets "
                    "with member/parent/level columns, or path-style tabs "
                    "such as Level 1/Level 2/Level 3, Entity/Business Unit/Department/Cost Center, "
                    "or other ordered hierarchy columns like Company/Division/Team."
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
        default_prompt = "Validate the uploaded source hierarchy workbook against the uploaded target hierarchy workbook"
        validation_prompt = st.text_area("Validation Request", value=default_prompt, height=110)
        if st.button("Run Validation", type="primary", width="stretch"):
            try:
                st.session_state["report"] = workflow.validate(
                    ValidationRequest(message=validation_prompt, rebuild_index=True)
                )
                st.session_state["cleanup_after_render"] = True
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

    previews = workflow.preview_dataframes()
    source_col, target_col = st.columns(2)
    with source_col:
        _render_hierarchy_preview(
            "Source Preview",
            [
                ("Entity Hierarchy", previews["source_entity"]),
                ("Account Hierarchy", previews["source_account"]),
            ],
            "No source hierarchy loaded yet.",
        )

    with target_col:
        _render_hierarchy_preview(
            "Target Preview",
            [
                ("Entity Hierarchy", previews["target_entity"]),
                ("Account Hierarchy", previews["target_account"]),
            ],
            "No target hierarchy loaded yet.",
        )

    report = st.session_state.get("report")
    if report:
        st.divider()
        summary_1, summary_2, summary_3 = st.columns(3)
        summary_1.metric("Overall Status", report.overall_status)
        summary_2.metric("Failed Checks", report.summary["failed_checks"])
        summary_3.metric("Failed Records", report.summary["failed_records"])

        st.subheader("Validation Summary")
        st.markdown(_build_validation_summary(report))

        st.subheader("Agent Explanation")
        st.markdown(report.agent_explanation)

        st.subheader("Row-Level Failures")
        failure_frame = _flatten_failures(report)
        if failure_frame.empty:
            st.success("No row-level failures detected.")
        else:
            st.dataframe(failure_frame, width="stretch", height=240)

        st.subheader("Validation Checks")
        for result in report.results:
            with st.expander(f"{result.dimension.title()} - {result.rule_name} ({result.status})", expanded=False):
                st.write(f"Severity: {result.severity}")
                st.write(f"What was checked: {describe_checked_rule(result)}")
                st.write(f"Rows passed: {result.passed_count}")
                st.write(f"Rows failed: {result.failed_count}")
                if result.status == "FAILED":
                    st.write(f"Likely cause: {result.likely_cause}")
                    st.write(f"Recommended action: {result.recommended_action}")
                    if result.retrieved_context:
                        st.markdown("**Retrieved context**")
                        for snippet in result.retrieved_context:
                            st.write(f"- {snippet}")
                else:
                    st.success(describe_passed_rule(result))

        st.subheader("Generated Reports")
        st.markdown("**JSON Preview**")
        st.json(report.model_dump(mode="json"))

        json_path = Path(report.json_report_path) if report.json_report_path else None
        markdown_path = Path(report.markdown_report_path) if report.markdown_report_path else None
        if json_path:
            st.caption(f"JSON report: {json_path}")
        if markdown_path:
            st.caption(f"Markdown report: {markdown_path}")
        download_col_1, download_col_2 = st.columns(2)
        with download_col_1:
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report.model_dump(mode="json"), indent=2),
                file_name=json_path.name if json_path else "validation_report.json",
                mime="application/json",
                width="stretch",
            )
        with download_col_2:
            markdown_data = markdown_path.read_text(encoding="utf-8") if markdown_path and markdown_path.exists() else ""
            st.download_button(
                "Download Markdown Report",
                data=markdown_data,
                file_name=markdown_path.name if markdown_path else "validation_report.md",
                mime="text/markdown",
                width="stretch",
            )

    if st.session_state.pop("cleanup_after_render", False):
        workflow.clear_runtime_state(clear_index=False, clear_uploaded_files=True, clear_supporting_docs=True)
        st.session_state["cleanup_notice"] = (
            "Validation finished. Source files, target files, and supporting docs were cleared. "
            "The Chroma index was retained, and the next ingest will use a new upload-scoped collection."
        )


if __name__ == "__main__":
    main()
