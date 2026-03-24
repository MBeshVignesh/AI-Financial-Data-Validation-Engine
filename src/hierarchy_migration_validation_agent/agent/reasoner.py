from __future__ import annotations

import json
import logging
from urllib import error, request

from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.schemas import ExceptionReport

LOGGER = logging.getLogger(__name__)


class AgentReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def explain(self, report: ExceptionReport) -> str:
        explanation = self._try_ollama(report)
        if explanation:
            return explanation
        return self._fallback(report)

    def _try_ollama(self, report: ExceptionReport) -> str | None:
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "prompt": self._prompt(report),
        }
        endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        try:
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body.get("response")
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            LOGGER.info("Ollama explanation unavailable, falling back to template reasoning: %s", exc)
            return None

    def _prompt(self, report: ExceptionReport) -> str:
        failure_payload = [
            {
                "dimension": result.dimension,
                "rule_name": result.rule_name,
                "failed_count": result.failed_count,
                "sample_failures": [failure.model_dump() for failure in result.failed_records[:3]],
                "retrieved_context": result.retrieved_context[:2],
            }
            for result in report.results
            if result.failed_count
        ]
        context = report.retrieved_context[:6]
        return (
            "You are explaining a finance hierarchy migration validation run.\n"
            "Return markdown with these sections: What Was Tested, What Failed or Passed, Likely Cause, Recommended Action.\n"
            "Summarize the failures for business users, include passed or failed check counts, "
            "name likely root causes, and recommend concise next actions.\n"
            "Be concrete and avoid generic AI language.\n"
            f"Request: {report.request}\n"
            f"Context: {json.dumps(context)}\n"
            f"Failures: {json.dumps(failure_payload)}"
        )

    def _fallback(self, report: ExceptionReport) -> str:
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
                    f"- All {report.summary['passed_checks']} passed checks completed successfully for {', '.join(report.dimensions)}.",
                    "- No hierarchy defects were detected in the uploaded source-versus-target comparison.",
                    "- No failed checks or flagged records were produced in this run.",
                ]
            )

        top_failures = [
            f"{result.dimension.title()} - {result.rule_name} ({result.failed_count} failed records)"
            for result in report.results
            if result.failed_count
        ][:4]
        root_causes = report.likely_root_causes[:3]
        actions = report.recommended_actions[:3]
        return "\n".join(
            [
                "### What Was Tested",
                f"- {', '.join(tested_checks) if tested_checks else 'No validation checks were executed.'}",
                "",
                "### What Failed",
                f"- {', '.join(top_failures)}",
                f"- {report.summary['failed_checks']} failed checks were recorded and {report.summary['failed_records']} records were flagged.",
                "",
                "### Likely Cause",
                f"- {', '.join(root_causes) if root_causes else 'No likely cause was derived.'}",
                "",
                "### Recommended Action",
                f"- {', '.join(actions) if actions else 'No recommended action was derived.'}",
            ]
        )
