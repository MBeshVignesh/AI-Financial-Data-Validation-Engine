from __future__ import annotations

import logging
import shutil
from pathlib import Path

import chromadb
import pandas as pd

from hierarchy_migration_validation_agent.config import Settings
from hierarchy_migration_validation_agent.rag.embeddings import create_embedding_function
from hierarchy_migration_validation_agent.schemas import RagDocument
from hierarchy_migration_validation_agent.utils.io import read_text

LOGGER = logging.getLogger(__name__)


class RagIndexService:
    DEFAULT_MAX_BATCH_SIZE = 1000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.collection_name = "hierarchy_context"
        self.embedding_function = create_embedding_function(settings)
        self.client = self._create_persistent_client()

    def set_collection_name(self, collection_name: str) -> None:
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
            return True
        except Exception as exc:
            LOGGER.warning("Unable to inspect Chroma collection state: %s", exc)
            return False

    def clear_index(self) -> None:
        self._reset_index_storage()

    def storage_status(self) -> dict[str, str | int]:
        file_count = sum(1 for path in self.settings.index_dir.glob("**/*") if path.is_file())
        return {
            "mode": "persistent",
            "collection_name": self.collection_name,
            "path": str(self.settings.index_dir),
            "file_count": file_count,
        }

    def build_index(
        self,
        mapping_df: pd.DataFrame | None = None,
        rules_df: pd.DataFrame | None = None,
        source_frames: dict[str, pd.DataFrame] | None = None,
        target_frames: dict[str, pd.DataFrame] | None = None,
        transformation_notes: str | None = None,
        prior_exception_log: pd.DataFrame | None = None,
    ) -> list[RagDocument]:
        documents = self._build_documents(
            mapping_df=mapping_df if mapping_df is not None else self._load_mapping(),
            rules_df=rules_df if rules_df is not None else self._load_rules(),
            source_frames=source_frames if source_frames is not None else self._load_source_frames(),
            target_frames=target_frames if target_frames is not None else self._load_target_frames(),
            transformation_notes=transformation_notes if transformation_notes is not None else self._load_transformation_notes(),
            prior_exception_log=prior_exception_log if prior_exception_log is not None else self._load_prior_exception_log(),
        )
        try:
            self._recreate_collection(documents)
        except Exception as exc:
            LOGGER.warning("Primary Chroma rebuild failed, resetting local index storage: %s", exc)
            try:
                self._reset_index_storage()
                self._recreate_collection(documents)
            except Exception as reset_exc:
                raise RuntimeError(
                    "Persistent Chroma build failed while using the local Nomic embedding model. "
                    f"Original error: {reset_exc}"
                ) from reset_exc

        LOGGER.info("Built RAG index with %s documents", len(documents))
        return documents

    def retrieve(self, query: str, *, dimension: str | None = None, n_results: int = 4) -> list[RagDocument]:
        try:
            if not self.collection_exists():
                self.build_index()

            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            results = collection.query(
                query_embeddings=[self._query_embedding(query)],
                n_results=max(n_results * 2, n_results),
            )
        except Exception as exc:
            LOGGER.warning("Chroma retrieval failed, rebuilding index and retrying once: %s", exc)
            try:
                self._reset_index_storage()
                self.build_index()
                collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                )
                results = collection.query(
                    query_embeddings=[self._query_embedding(query)],
                    n_results=max(n_results * 2, n_results),
                )
            except Exception as retry_exc:
                raise RuntimeError(
                    "Chroma retrieval failed while using the local Nomic embedding model. "
                    f"Original error: {retry_exc}"
                ) from retry_exc
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved: list[RagDocument] = []
        for document_id, content, metadata in zip(ids, documents, metadatas, strict=False):
            metadata = metadata or {}
            metadata_dimension = metadata.get("dimension")
            if dimension and metadata_dimension not in {dimension, "all"}:
                continue
            retrieved.append(RagDocument(document_id=document_id, content=content, metadata=metadata))
            if len(retrieved) >= n_results:
                break
        return retrieved

    def _recreate_collection(self, documents: list[RagDocument]) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

        if documents:
            self._add_documents_in_batches(collection, documents)

    def _reset_index_storage(self) -> None:
        if self.settings.index_dir.exists():
            shutil.rmtree(self.settings.index_dir, ignore_errors=True)
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._create_persistent_client()

    def _query_embedding(self, query: str) -> list[float]:
        embed_query = getattr(self.embedding_function, "embed_query", None)
        if callable(embed_query):
            embedding = embed_query(query)
            if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                return embedding[0]
            return embedding
        return self.embedding_function([query])[0]

    def _create_persistent_client(self):
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.settings.index_dir))

    def _add_documents_in_batches(self, collection, documents: list[RagDocument]) -> None:
        batch_size = self._max_batch_size()
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            collection.add(
                ids=[document.document_id for document in batch],
                documents=[document.content for document in batch],
                metadatas=[document.metadata for document in batch],
            )

    def _max_batch_size(self) -> int:
        get_max_batch_size = getattr(self.client, "get_max_batch_size", None)
        if callable(get_max_batch_size):
            try:
                max_batch_size = int(get_max_batch_size())
                if max_batch_size > 0:
                    return max_batch_size
            except Exception as exc:
                LOGGER.warning("Unable to read Chroma max batch size, using default: %s", exc)
        return self.DEFAULT_MAX_BATCH_SIZE

    def _build_documents(
        self,
        *,
        mapping_df: pd.DataFrame,
        rules_df: pd.DataFrame,
        source_frames: dict[str, pd.DataFrame],
        target_frames: dict[str, pd.DataFrame],
        transformation_notes: str,
        prior_exception_log: pd.DataFrame,
    ) -> list[RagDocument]:
        documents: list[RagDocument] = []
        documents.extend(self._hierarchy_documents(source_frames, doc_type="source_hierarchy"))
        documents.extend(self._hierarchy_documents(target_frames, doc_type="target_hierarchy"))

        for row_number, row in enumerate(mapping_df.itertuples(), start=1):
            documents.append(
                RagDocument(
                    document_id=f"mapping-{row_number}",
                    content=(
                        f"Mapping record for {row.dimension} member {row.source_member_name}: "
                        f"target member {row.target_member_name}, status {row.mapping_status}, "
                        f"rule {row.mapping_rule}. Notes: {row.notes or 'None'}"
                    ),
                    metadata={"doc_type": "mapping", "dimension": row.dimension},
                )
            )

        for row_number, row in enumerate(rules_df.itertuples(), start=1):
            documents.append(
                RagDocument(
                    document_id=f"rule-{row_number}",
                    content=(
                        f"Validation rule for {row.dimension}: {row.rule_name}. "
                        f"Check type {row.check_type}. Description: {row.description}. "
                        f"Business rationale: {row.business_rationale}."
                    ),
                    metadata={"doc_type": "rule", "dimension": row.dimension},
                )
            )

        for index, section in enumerate(
            [section.strip() for section in transformation_notes.split("\n\n") if section.strip()],
            start=1,
        ):
            documents.append(
                RagDocument(
                    document_id=f"note-{index}",
                    content=section,
                    metadata={"doc_type": "transformation_note", "dimension": "all"},
                )
            )

        for row_number, row in enumerate(prior_exception_log.itertuples(), start=1):
            documents.append(
                RagDocument(
                    document_id=f"history-{row_number}",
                    content=(
                        f"Prior exception on {row.run_date} for {row.dimension}: "
                        f"rule {row.rule_name}, member {row.member_name}, issue {row.issue}. "
                        f"Likely root cause: {row.root_cause}."
                    ),
                    metadata={"doc_type": "prior_exception", "dimension": row.dimension},
                )
            )
        return documents

    def _hierarchy_documents(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        doc_type: str,
    ) -> list[RagDocument]:
        documents: list[RagDocument] = []
        for dimension, frame in frames.items():
            if frame.empty:
                continue
            origin = "source" if doc_type == "source_hierarchy" else "target"
            documents.append(
                RagDocument(
                    document_id=f"{origin}-summary-{dimension}",
                    content=(
                        f"{origin.title()} {dimension} hierarchy contains {len(frame)} normalized members. "
                        f"Columns available: {', '.join(frame.columns)}."
                    ),
                    metadata={"doc_type": doc_type, "dimension": dimension},
                )
            )
            for row_number, row in enumerate(frame.itertuples(), start=1):
                parent_name = getattr(row, "parent_name", None) or "ROOT"
                documents.append(
                    RagDocument(
                        document_id=f"{origin}-{dimension}-{row_number}",
                        content=(
                            f"{origin.title()} {dimension} hierarchy member {getattr(row, 'member_name', 'unknown')} "
                            f"has parent {parent_name}, level {getattr(row, 'level', 'unknown')}, "
                            f"leaf flag {getattr(row, 'leaf_flag', 'unknown')}."
                        ),
                        metadata={"doc_type": doc_type, "dimension": dimension},
                    )
                )
        return documents

    def _load_mapping(self) -> pd.DataFrame:
        payload = self._load_source_payload()
        if payload is None:
            return pd.DataFrame(columns=["dimension", "source_member_name", "target_member_name", "mapping_status", "mapping_rule", "notes"])
        return payload.frames.get(
            "mapping",
            pd.DataFrame(columns=["dimension", "source_member_name", "target_member_name", "mapping_status", "mapping_rule", "notes"]),
        )

    def _load_rules(self) -> pd.DataFrame:
        payload = self._load_source_payload()
        if payload is None:
            return pd.DataFrame(columns=["rule_id", "dimension", "rule_name", "check_type", "severity", "enabled", "optional", "description", "business_rationale"])
        return payload.frames.get(
            "rules",
            pd.DataFrame(columns=["rule_id", "dimension", "rule_name", "check_type", "severity", "enabled", "optional", "description", "business_rationale"]),
        )

    def _load_source_frames(self) -> dict[str, pd.DataFrame]:
        payload = self._load_source_payload()
        if payload is None:
            return {"account": pd.DataFrame(), "entity": pd.DataFrame()}
        return {
            "account": payload.frames.get("account_source", pd.DataFrame()),
            "entity": payload.frames.get("entity_source", pd.DataFrame()),
        }

    def _load_target_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "account": self._read_optional_csv(self.settings.hierarchies_dir / "dim_account_hierarchy.csv"),
            "entity": self._read_optional_csv(self.settings.hierarchies_dir / "dim_entity_hierarchy.csv"),
        }

    def _load_transformation_notes(self) -> str:
        path = self.settings.supporting_dir / "transformation_notes.md"
        return read_text(path) if path.exists() else ""

    def _load_prior_exception_log(self) -> pd.DataFrame:
        path = self.settings.supporting_dir / "prior_exception_log.csv"
        if not path.exists():
            return pd.DataFrame(columns=["run_date", "dimension", "rule_name", "member_name", "issue", "root_cause"])
        return pd.read_csv(path)

    def _load_source_payload(self):
        from hierarchy_migration_validation_agent.ingestion.excel_ingestor import ExcelIngestionService

        files = sorted(self.settings.source_dir.glob("*.xlsx"))
        if not files:
            return None
        try:
            return ExcelIngestionService(self.settings).parse_files(files, role="source")
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_optional_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
