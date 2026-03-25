from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str = "AI Financial Data Validation Engine (Hierarchical and Row-Level)"
    root_dir: Path
    data_dir: Path
    source_dir: Path
    target_upload_dir: Path
    curated_dir: Path
    hierarchies_dir: Path
    reports_dir: Path
    index_dir: Path
    supporting_dir: Path
    state_dir: Path
    db_path: Path
    ollama_base_url: str
    ollama_model: str
    embedding_provider: str
    embedding_model_name: str
    embedding_device: str
    embedding_cache_dir: Path
    embedding_local_files_only: bool
    embedding_trust_remote_code: bool
    log_level: str

    @classmethod
    def from_root(cls, root_dir: Path | None = None) -> "Settings":
        root = (root_dir or Path(__file__).resolve().parents[2]).resolve()
        data_dir = _resolve_data_dir(root)
        return cls(
            root_dir=root,
            data_dir=data_dir,
            source_dir=data_dir / "source",
            target_upload_dir=data_dir / "target",
            curated_dir=data_dir / "curated",
            hierarchies_dir=data_dir / "curated" / "hierarchies",
            reports_dir=data_dir / "reports",
            index_dir=data_dir / "indexes" / "chroma",
            supporting_dir=data_dir / "supporting",
            state_dir=data_dir / "state",
            db_path=data_dir / "state" / "app.sqlite",
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "nomic"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            embedding_cache_dir=data_dir / "models",
            embedding_local_files_only=_env_flag("EMBEDDING_LOCAL_FILES_ONLY", False),
            embedding_trust_remote_code=_env_flag("EMBEDDING_TRUST_REMOTE_CODE", True),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.source_dir,
            self.target_upload_dir,
            self.curated_dir,
            self.hierarchies_dir,
            self.reports_dir,
            self.index_dir,
            self.embedding_cache_dir,
            self.supporting_dir,
            self.state_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_root()
    settings.ensure_directories()
    return settings


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_data_dir(root_dir: Path) -> Path:
    configured_data_dir = os.getenv("APP_DATA_DIR")
    if configured_data_dir:
        configured_path = Path(configured_data_dir).expanduser().resolve()
        if _can_write_to_directory(configured_path):
            return configured_path

    default_data_dir = root_dir / "data"
    if _can_write_to_directory(default_data_dir):
        return default_data_dir

    fallback_data_dir = Path(tempfile.gettempdir()) / "ai_financial_data_validation_engine"
    if _can_write_to_directory(fallback_data_dir):
        return fallback_data_dir

    raise OSError(
        "No writable application data directory is available. "
        "Set APP_DATA_DIR to a writable path for Chroma, uploads, reports, and local state."
    )


def _can_write_to_directory(path: Path) -> bool:
    probe_path = path / ".write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False
