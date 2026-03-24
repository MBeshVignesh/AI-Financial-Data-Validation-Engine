from __future__ import annotations

from pathlib import Path

from hierarchy_migration_validation_agent.schemas import ValidationRequest


def test_end_to_end_validation_flow(workflow):
    workflow.build_index()
    report = workflow.validate(
        ValidationRequest(
            message="Validate Oracle Smart View account and entity hierarchies against ADLS target",
            rebuild_index=True,
        )
    )

    assert report.overall_status == "FAILED"
    assert report.summary["failed_checks"] >= 7
    assert Path(report.json_report_path).exists()
    assert Path(report.markdown_report_path).exists()
    assert "failed checks" in report.agent_explanation.lower()
