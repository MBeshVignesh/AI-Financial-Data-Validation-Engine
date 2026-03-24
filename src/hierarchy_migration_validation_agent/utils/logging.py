from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        return

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
