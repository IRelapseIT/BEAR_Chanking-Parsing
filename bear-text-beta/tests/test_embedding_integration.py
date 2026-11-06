"""Integration of the imported embedding layer with the hardened BEAR runtime."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bear_text_beta import runtime
from bear_text_beta.embedding_pipeline import EmbeddingPipeline
from bear_text_beta.embeddings import UnavailableEmbeddingProvider
from bear_text_beta.models import ProcessingProfile


@pytest.mark.parametrize("model_id,expected_model", [
    ("intfloat/multilingual-e5-small", "bear-e5-small"),
    ("BAAI/bge-m3", "BAAI/bge-m3"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
])
def test_services_wires_pipeline_without_replacing_parser_guards(
        tmp_path, monkeypatch, profile, counter, context, model_id, expected_model):
    selected = replace(profile, model_id=model_id)
    counter.profile = selected
    monkeypatch.delenv("BEAR_CHUNKER_URL", raising=False)
    monkeypatch.delenv("BEAR_INFINITY_MODEL", raising=False)
    monkeypatch.delenv("BEAR_QDRANT_COLLECTION", raising=False)
    monkeypatch.setattr(runtime, "load_profile", lambda path: selected)
    monkeypatch.setattr(runtime, "HFTokenCounter", lambda path, profile: counter)
    calls = []

    class Provider:
        max_tokens = 480
        profile = selected
        closed = False

        def embed_batch(self, texts, purpose):
            calls.append((texts, purpose))
            return [[3.0, 4.0, 0.0, 0.0] for _ in texts]

        def close(self):
            self.closed = True

    provider = Provider()

    def create_provider(url, profile, counter, **kwargs):
        assert kwargs["server_model"] == expected_model
        return provider

    monkeypatch.setattr(runtime, "InfinityEmbeddingProvider", create_provider)
    monkeypatch.setattr(runtime, "QdrantVectorStore", lambda url, collection, dimension, **kwargs:
                        SimpleNamespace(collection=collection, close=lambda: None))
    services = runtime.Services(tmp_path, "unused-model-dir")
    try:
        assert isinstance(services.embeddings, EmbeddingPipeline)
        assert services.chunker.provider is services.embeddings
        assert services.indexer.embeddings is services.embeddings
        assert services.indexer.external_validators == services.chunker.external_validators
        assert services.vectors.collection == (
            "bear_text_e5_small_v1" if model_id.startswith("intfloat/")
            else ("bear_bge_m3_" if model_id == "BAAI/bge-m3" else "bear_text_") + selected.identifier())
        # Uses the current native TXT route, without an upstream Dedoc service.
        result = services.process("Условия SSL.\n\nВозврат за 12 дней.".encode(), context,
                                  ProcessingProfile(mode="token"))
        embedded = services.embeddings.vectorize_chunks(result)
        assert [item.chunk.id for item in embedded] == [c.id for c in result.chunks]
        assert all(item.vector == [0.6, 0.8, 0.0, 0.0] for item in embedded)
        services.embeddings.embed_batch(["  Как\n настроить SSL? "], "query")
        assert calls[-1] == (["Как настроить SSL?"], "query")
    finally:
        services.close()
    assert provider.closed


def test_offline_runtime_does_not_create_inference_provider(tmp_path, monkeypatch, profile, counter):
    monkeypatch.delenv("BEAR_CHUNKER_URL", raising=False)
    monkeypatch.setattr(runtime, "load_profile", lambda path: profile)
    monkeypatch.setattr(runtime, "HFTokenCounter", lambda path, profile: counter)

    def unexpected_provider(*args, **kwargs):
        pytest.fail("Offline mode must not create an inference client")

    monkeypatch.setattr(runtime, "InfinityEmbeddingProvider", unexpected_provider)
    services = runtime.Services(tmp_path, "unused-model-dir", offline=True)
    try:
        assert isinstance(services.embeddings, UnavailableEmbeddingProvider)
    finally:
        services.close()
