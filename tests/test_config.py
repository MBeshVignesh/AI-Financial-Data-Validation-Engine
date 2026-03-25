from __future__ import annotations

from pathlib import Path

from hierarchy_migration_validation_agent.config import Settings


def test_settings_uses_configured_app_data_dir(monkeypatch, tmp_path):
    configured_dir = tmp_path / "streamlit-data"
    monkeypatch.setenv("APP_DATA_DIR", str(configured_dir))

    settings = Settings.from_root(tmp_path / "repo-root")

    assert settings.data_dir == configured_dir
    assert settings.index_dir == configured_dir / "indexes" / "chroma"
    assert settings.embedding_cache_dir == configured_dir / "models"


def test_settings_falls_back_to_temp_when_repo_data_dir_is_not_writable(monkeypatch, tmp_path):
    fallback_dir = tmp_path / "ai_financial_data_validation_engine"

    def fake_can_write_to_directory(path: Path) -> bool:
        return path == fallback_dir

    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    monkeypatch.setattr(
        "hierarchy_migration_validation_agent.config.tempfile.gettempdir",
        lambda: str(tmp_path),
    )
    monkeypatch.setattr(
        "hierarchy_migration_validation_agent.config._can_write_to_directory",
        fake_can_write_to_directory,
    )

    settings = Settings.from_root(tmp_path / "repo-root")

    assert settings.data_dir == fallback_dir
    assert settings.source_dir == fallback_dir / "source"
