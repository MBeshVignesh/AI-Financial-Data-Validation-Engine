from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class SentenceTransformerEmbeddingFunction:
    """Local sentence-transformers embedding function with query/document prompts."""

    QUERY_PROMPT_CANDIDATES = ("query", "search_query")
    DOCUMENT_PROMPT_CANDIDATES = ("document", "search_document")

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self._model = None

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def name(self) -> str:
        return f"sentence_transformer::{self.model_name}"

    @staticmethod
    def is_legacy() -> bool:
        return False

    @staticmethod
    def default_space() -> str:
        return "cosine"

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine"]

    def get_config(self) -> dict[str, str | bool]:
        return {
            "name": self.name(),
            "model_name": self.model_name,
            "device": self.device,
            "local_files_only": self.local_files_only,
            "space": self.default_space(),
        }

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._encode(documents, prompt_candidates=self.DOCUMENT_PROMPT_CANDIDATES)

    def embed_query(self, input: list[str] | str) -> list[list[float]] | list[float]:
        if isinstance(input, list):
            return self._encode(input, prompt_candidates=self.QUERY_PROMPT_CANDIDATES)
        return self._encode([input], prompt_candidates=self.QUERY_PROMPT_CANDIDATES)[0]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install project dependencies to use local Nomic embeddings."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.cache_dir is not None:
            load_kwargs["cache_folder"] = str(self.cache_dir)
        if self.device and self.device != "auto":
            load_kwargs["device"] = self.device

        self._model = SentenceTransformer(self.model_name, **load_kwargs)
        return self._model

    def _encode(self, texts: list[str], *, prompt_candidates: tuple[str, ...]) -> list[list[float]]:
        model = self._ensure_model()
        encode_kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
        resolved_candidates = self._resolve_prompt_candidates(model, prompt_candidates)
        last_prompt_error: Exception | None = None

        for prompt_name in resolved_candidates:
            try:
                vectors = model.encode(texts, prompt_name=prompt_name, **encode_kwargs)
                return vectors.tolist()
            except TypeError:
                prompt_text = self._prompt_text_for_name(model, prompt_name)
                if prompt_text is not None:
                    vectors = model.encode(texts, prompt=prompt_text, **encode_kwargs)
                else:
                    prefixed_texts = [f"{prompt_name}: {text}" for text in texts]
                    vectors = model.encode(prefixed_texts, **encode_kwargs)
                return vectors.tolist()
            except ValueError as exc:
                if "Prompt name" in str(exc):
                    last_prompt_error = exc
                    continue
                raise

        prompt_name = resolved_candidates[0]
        prompt_text = self._prompt_text_for_name(model, prompt_name)
        if prompt_text is not None:
            try:
                vectors = model.encode(texts, prompt=prompt_text, **encode_kwargs)
                return vectors.tolist()
            except TypeError:
                pass

        if last_prompt_error is not None:
            LOGGER.warning("Falling back to prefixed embedding text after prompt lookup failure: %s", last_prompt_error)
        prefixed_texts = [f"{prompt_name}: {text}" for text in texts]
        vectors = model.encode(prefixed_texts, **encode_kwargs)
        return vectors.tolist()

    @staticmethod
    def _prompt_text_for_name(model, prompt_name: str) -> str | None:
        prompts = getattr(model, "prompts", None)
        if isinstance(prompts, dict):
            prompt_value = prompts.get(prompt_name)
            if isinstance(prompt_value, str):
                return prompt_value
        return None

    @staticmethod
    def _resolve_prompt_candidates(model, prompt_candidates: tuple[str, ...]) -> tuple[str, ...]:
        prompts = getattr(model, "prompts", None)
        if not isinstance(prompts, dict) or not prompts:
            return prompt_candidates

        matching = tuple(candidate for candidate in prompt_candidates if candidate in prompts)
        if matching:
            return matching
        return prompt_candidates


def create_embedding_function(settings) -> SentenceTransformerEmbeddingFunction:
    provider = settings.embedding_provider.lower()
    if provider in {"sentence_transformers", "nomic_local", "nomic"}:
        return SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
            trust_remote_code=settings.embedding_trust_remote_code,
        )

    raise ValueError(
        f"Unsupported embedding provider '{settings.embedding_provider}'. "
        "This app is configured for local sentence-transformers / Nomic embeddings only."
    )
