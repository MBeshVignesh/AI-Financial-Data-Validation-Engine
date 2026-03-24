from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def generate_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"
