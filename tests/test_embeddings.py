from __future__ import annotations

from hierarchy_migration_validation_agent.rag.embeddings import SentenceTransformerEmbeddingFunction


class FakeVectors(list):
    def tolist(self):
        return list(self)


def test_embedding_uses_model_document_and_query_prompt_names():
    class FakeModel:
        prompts = {"query": "query: ", "document": "document: "}

        def __init__(self) -> None:
            self.calls: list[tuple[str | None, str | None, list[str]]] = []

        def encode(self, texts, prompt_name=None, prompt=None, **kwargs):
            del kwargs
            self.calls.append((prompt_name, prompt, list(texts)))
            return FakeVectors([[1.0, 0.0] for _ in texts])

    embedding = SentenceTransformerEmbeddingFunction(model_name="nomic-ai/nomic-embed-text-v1.5")
    fake_model = FakeModel()
    embedding._model = fake_model

    embedding.embed_documents(["member one"])
    embedding.embed_query("member one")

    assert fake_model.calls[0][0] == "document"
    assert fake_model.calls[1][0] == "query"


def test_embedding_falls_back_to_prompt_text_when_prompt_name_is_unsupported():
    class FakeModel:
        prompts = {"document": "document: "}

        def __init__(self) -> None:
            self.calls: list[tuple[str | None, str | None, list[str]]] = []

        def encode(self, texts, prompt_name=None, prompt=None, **kwargs):
            del kwargs
            self.calls.append((prompt_name, prompt, list(texts)))
            if prompt_name is not None:
                raise TypeError("encode() got an unexpected keyword argument 'prompt_name'")
            return FakeVectors([[1.0, 0.0] for _ in texts])

    embedding = SentenceTransformerEmbeddingFunction(model_name="nomic-ai/nomic-embed-text-v1.5")
    fake_model = FakeModel()
    embedding._model = fake_model

    embedding.embed_documents(["member one"])

    assert fake_model.calls[0][0] == "document"
    assert fake_model.calls[1][1] == "document: "
