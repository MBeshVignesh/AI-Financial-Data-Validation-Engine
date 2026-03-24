from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.schemas import ExceptionReport
from hierarchy_migration_validation_agent.utils.io import write_json, write_text
from hierarchy_migration_validation_agent.utils.validation_text import describe_checked_rule, describe_passed_rule


class ReportWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()

    def write(self, report: ExceptionReport) -> ExceptionReport:
        json_path = self.settings.reports_dir / f"{report.run_id}.json"
        markdown_path = self.settings.reports_dir / f"{report.run_id}.md"
        report.json_report_path = str(json_path)
        report.markdown_report_path = str(markdown_path)
        write_json(json_path, report.model_dump(mode="json"))
        write_text(markdown_path, self.to_markdown(report))
        self.append_to_prior_exception_log(report)
        return report

    def to_markdown(self, report: ExceptionReport) -> str:
        lines = [
            "# Hierarchy Migration Validation Report",
            "",
            "## Run Summary",
            f"- Run ID: `{report.run_id}`",
            f"- Generated At: {report.created_at.isoformat()}",
            f"- Request: {report.request}",
            f"- Overall Status: **{report.overall_status}**",
            f"- Dimensions: {', '.join(report.dimensions)}",
            f"- Total Checks: {report.summary['total_checks']}",
            f"- Failed Checks: {report.summary['failed_checks']}",
            f"- Total Failed Records: {report.summary['failed_records']}",
            "",
            "## Validation Narrative",
            report.agent_explanation or "No agent explanation available.",
            "",
            "## Checks",
        ]

        for result in report.results:
            lines.extend(
                [
                    "",
                    f"### {result.dimension.title()} - {result.rule_name}",
                    f"- Status: **{result.status}**",
                    f"- Severity: {result.severity}",
                    f"- What Was Checked: {describe_checked_rule(result)}",
                    f"- Passed Count: {result.passed_count}",
                    f"- Failed Count: {result.failed_count}",
                ]
            )
            if result.status == "FAILED":
                lines.extend(
                    [
                        f"- Likely Cause: {result.likely_cause}",
                        f"- Recommended Action: {result.recommended_action}",
                    ]
                )
            else:
                lines.append(f"- Outcome: {describe_passed_rule(result)}")
            if result.status == "FAILED" and result.retrieved_context:
                lines.append("- Retrieved Context:")
                for snippet in result.retrieved_context:
                    lines.append(f"  - {snippet}")
            if result.failed_records:
                lines.append("- Failed Records:")
                for failure in result.failed_records:
                    lines.append(
                        "  - "
                        f"member=`{failure.member_name or 'n/a'}` issue={failure.issue} "
                        f"source={failure.source_value or 'n/a'} target={failure.target_value or 'n/a'}"
                    )

        if report.likely_root_causes:
            lines.extend(["", "## Likely Root Causes"])
            for cause in report.likely_root_causes:
                lines.append(f"- {cause}")

        if report.recommended_actions:
            lines.extend(["", "## Recommended Next Actions"])
            for action in report.recommended_actions:
                lines.append(f"- {action}")

        return "\n".join(lines)

    def append_to_prior_exception_log(self, report: ExceptionReport) -> None:
        prior_exception_path = self.settings.supporting_dir / "prior_exception_log.csv"
        rows = []
        for result in report.results:
            for failure in result.failed_records:
                rows.append(
                    {
                        "run_date": datetime.now(timezone.utc).date().isoformat(),
                        "dimension": failure.dimension,
                        "rule_name": failure.rule_name,
                        "member_name": failure.member_name,
                        "issue": failure.issue,
                        "root_cause": result.likely_cause,
                    }
                )
        if not rows:
            return

        current = pd.read_csv(prior_exception_path) if prior_exception_path.exists() else pd.DataFrame()
        updated = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        updated.to_csv(prior_exception_path, index=False)
