from __future__ import annotations

import sqlite3
from pathlib import Path

from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.schemas import ExceptionReport


class RunRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.db_path = self.settings.db_path
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    overall_status TEXT NOT NULL,
                    dimensions TEXT NOT NULL,
                    json_report_path TEXT NOT NULL,
                    markdown_report_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    agent_explanation TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save_report(self, report: ExceptionReport) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO validation_runs (
                    run_id,
                    created_at,
                    request_text,
                    overall_status,
                    dimensions,
                    json_report_path,
                    markdown_report_path,
                    summary_json,
                    agent_explanation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.created_at.isoformat(),
                    report.request,
                    report.overall_status,
                    ",".join(report.dimensions),
                    report.json_report_path or "",
                    report.markdown_report_path or "",
                    report.summary.__repr__(),
                    report.agent_explanation,
                ),
            )
            connection.commit()

    def list_run_count(self) -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute("SELECT COUNT(*) FROM validation_runs")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def get_report_paths(self, run_id: str) -> tuple[Path, Path] | None:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                """
                SELECT json_report_path, markdown_report_path
                FROM validation_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Path(row[0]), Path(row[1])
