from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _PACKAGE_ROOT.parent / "src" / "hierarchy_migration_validation_agent"

if _SRC_PACKAGE.exists():
    __path__ = [str(_SRC_PACKAGE), str(_PACKAGE_ROOT)]
else:
    __path__ = [str(_PACKAGE_ROOT)]

__version__ = "0.1.0"
