from __future__ import annotations

import pandas as pd
import pytest

from hierarchy_migration_validation_agent.agent.workflow import ValidationWorkflow
from hierarchy_migration_validation_agent.rag.indexer import RagIndexService
from hierarchy_migration_validation_agent.schemas import RagDocument
from hierarchy_migration_validation_agent.schemas import ValidationRequest


def test_build_index_raises_when_persistent_writes_fail(test_settings):
    service = RagIndexService(test_settings)
    recreate_calls: list[str] = []

    def flaky_recreate_collection(documents):
        del documents
        recreate_calls.append("attempt")
        raise RuntimeError("attempt to write a readonly database")

    service._recreate_collection = flaky_recreate_collection  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Persistent Chroma build failed while using the local Nomic embedding model"):
        service.build_index(
            mapping_df=pd.DataFrame(),
            rules_df=pd.DataFrame(),
            source_frames={"account": pd.DataFrame(), "entity": pd.DataFrame()},
            target_frames={"account": pd.DataFrame(), "entity": pd.DataFrame()},
            transformation_notes="",
            prior_exception_log=pd.DataFrame(),
        )

    assert recreate_calls == ["attempt", "attempt"]


def test_build_index_raises_when_embedding_setup_fails(test_settings):
    service = RagIndexService(test_settings)
    recreate_calls: list[str] = []

    def embedding_failure(documents):
        del documents
        recreate_calls.append("attempt")
        raise RuntimeError("missing embedding dependency")

    service._recreate_collection = embedding_failure  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Persistent Chroma build failed while using the local Nomic embedding model"):
        service.build_index(
            mapping_df=pd.DataFrame(),
            rules_df=pd.DataFrame(),
            source_frames={"account": pd.DataFrame(), "entity": pd.DataFrame()},
            target_frames={"account": pd.DataFrame(), "entity": pd.DataFrame()},
            transformation_notes="",
            prior_exception_log=pd.DataFrame(),
        )

    assert recreate_calls == ["attempt", "attempt"]


def test_recreate_collection_batches_document_adds(test_settings):
    service = RagIndexService(test_settings)

    class FakeCollection:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def add(self, ids, documents, metadatas) -> None:
            self.batch_sizes.append(len(ids))

    fake_collection = FakeCollection()

    class FakeClient:
        def __init__(self, collection) -> None:
            self.collection = collection

        def delete_collection(self, name: str) -> None:
            return None

        def get_or_create_collection(self, name: str, embedding_function=None):
            return self.collection

        def get_max_batch_size(self) -> int:
            return 2

    service.client = FakeClient(fake_collection)
    documents = [
        RagDocument(document_id=f"doc-{index}", content=f"content-{index}", metadata={"dimension": "entity"})
        for index in range(5)
    ]

    service._recreate_collection(documents)

    assert fake_collection.batch_sizes == [2, 2, 1]


def test_validate_rebuild_index_builds_without_preclear(test_settings):
    workflow = ValidationWorkflow(test_settings)
    source_path = test_settings.source_dir / "Current_Source.xlsx"
    target_path = test_settings.target_upload_dir / "Current_Target.xlsx"
    hierarchy_frame = pd.DataFrame(
        [{"Level 1 (Entity)": "GlobalCorp", "Level 2 (BU)": "North America", "Level 3 (Dept)": "Finance"}]
    )

    with pd.ExcelWriter(source_path, engine="openpyxl") as writer:
        hierarchy_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")
    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        hierarchy_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")

    workflow.ingest_excel_files(auto_build_index=False)
    calls: list[str] = []

    def fake_build_index(*args, **kwargs):
        calls.append("build")
        return []

    def fake_collection_exists() -> bool:
        calls.append("exists")
        return True

    workflow.rag_service.build_index = fake_build_index  # type: ignore[method-assign]
    workflow.rag_service.collection_exists = fake_collection_exists  # type: ignore[method-assign]
    workflow.rag_service.retrieve = lambda *args, **kwargs: []  # type: ignore[method-assign]

    workflow.validate(ValidationRequest(message="Validate current upload", rebuild_index=True))

    assert calls == ["build"]


def test_clear_runtime_state_removes_ingested_files_and_supporting_docs(test_settings):
    workflow = ValidationWorkflow(test_settings)
    source_path = test_settings.source_dir / "Uploaded_Source.xlsx"
    target_path = test_settings.target_upload_dir / "Uploaded_Target.xlsx"
    notes_path = test_settings.supporting_dir / "transformation_notes.md"
    history_path = test_settings.supporting_dir / "prior_exception_log.csv"

    source_frame = pd.DataFrame(
        [{"Level 1 (Entity)": "GlobalCorp", "Level 2 (BU)": "North America", "Level 3 (Dept)": "Finance"}]
    )
    target_frame = source_frame.copy()

    with pd.ExcelWriter(source_path, engine="openpyxl") as writer:
        source_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")
    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        target_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")
    notes_path.write_text("temporary notes", encoding="utf-8")
    history_path.write_text("run_date,dimension\n", encoding="utf-8")

    workflow.ingest_excel_files(source_file_paths=[source_path], target_file_paths=[target_path], auto_build_index=False)
    workflow.clear_runtime_state(clear_index=False, clear_uploaded_files=True, clear_supporting_docs=True)

    assert workflow._active_context is None
    assert not source_path.exists()
    assert not target_path.exists()
    assert not notes_path.exists()
    assert not history_path.exists()


def test_storage_status_reports_persistent_chroma_path(test_settings):
    service = RagIndexService(test_settings)

    status = service.storage_status()

    assert status["mode"] == "persistent"
    assert status["collection_name"] == "hierarchy_context"
    assert status["path"] == str(test_settings.index_dir)


def test_ingest_assigns_new_upload_scoped_collection_name(test_settings):
    workflow = ValidationWorkflow(test_settings)
    source_path = test_settings.source_dir / "Scoped_Source.xlsx"
    target_path = test_settings.target_upload_dir / "Scoped_Target.xlsx"

    hierarchy_frame = pd.DataFrame(
        [{"Level 1 (Entity)": "GlobalCorp", "Level 2 (BU)": "North America", "Level 3 (Dept)": "Finance"}]
    )

    with pd.ExcelWriter(source_path, engine="openpyxl") as writer:
        hierarchy_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")
    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        hierarchy_frame.to_excel(writer, index=False, sheet_name="Hierarchy_View")

    workflow.ingest_excel_files(source_file_paths=[source_path], target_file_paths=[target_path], auto_build_index=False)
    first_collection = workflow.rag_service.collection_name

    workflow.ingest_excel_files(source_file_paths=[source_path], target_file_paths=[target_path], auto_build_index=False)
    second_collection = workflow.rag_service.collection_name

    assert first_collection.startswith("hierarchy_context_")
    assert second_collection.startswith("hierarchy_context_")
    assert first_collection != second_collection
    assert workflow._active_context is not None
    assert workflow._active_context.rag_collection_name == second_collection
