import asyncio
import copy
import json
import math
from dataclasses import asdict, replace

import httpx
import pytest

from bear_text_beta.embedding_pipeline import EmbeddingPipeline, MemoryVectorCache
from bear_text_beta.embeddings import InfinityEmbeddingProvider
from bear_text_beta.errors import EmbeddingError, ValidationError
from bear_text_beta.indexing import IndexingService, QdrantVectorStore
from bear_text_beta.models import ProcessingProfile
from bear_text_beta.storage import SQLiteManifest


class Provider:
    def __init__(self, profile):
        self.profile = profile
        self.calls = []
        self.closed = False

    def embed_batch(self, texts, purpose):
        self.calls.append((list(texts), purpose))
        return [[3.0, 4.0, 0.0, 0.0] for _ in texts]

    def close(self):
        self.closed = True


@pytest.fixture
def pipeline(profile, counter):
    return EmbeddingPipeline(Provider(profile), counter, profile, cache=MemoryVectorCache(), batch_size=2)


def test_query_normalization_trace_and_unit_vector(pipeline):
    trace = pipeline.vectorize_query("  Как\n настроить  SSL в NGINX? \u0438\u0306  ")
    assert trace.query.original.startswith("  Как")
    assert trace.query.text == "Как настроить SSL в NGINX? й"
    assert trace.query.builder_version == "nfc-whitespace-v1"
    assert trace.vector == [0.6, 0.8, 0.0, 0.0]
    assert trace.elapsed_ms >= 0
    assert trace.embedding_profile_id == pipeline.profile.identifier()


def test_custom_transform_once_per_entry_point(profile, counter):
    calls = []
    def expand(text):
        calls.append(text)
        return text + " TLS"
    pipeline = EmbeddingPipeline(Provider(profile), counter, profile,
                                 query_transform=expand, builder_version="tls-expansion-v1")
    pipeline.vectorize_query("SSL")
    pipeline.embed_batch(["SSL"], "query")
    assert calls == ["SSL", "SSL"]
    assert pipeline.provider.calls == [(["SSL TLS"], "query"), (["SSL TLS"], "query")]


def test_transform_requires_version(profile, counter):
    with pytest.raises(ValidationError, match="builder_version"):
        EmbeddingPipeline(Provider(profile), counter, profile, query_transform=lambda text: text)


@pytest.mark.parametrize("text", ["", " \n\t", None, 123, "x" * 1000])
def test_invalid_query_fails_before_inference(pipeline, text):
    with pytest.raises(ValidationError):
        pipeline.embed(text)
    assert pipeline.provider.calls == []


def test_cache_batch_deduplication_order_and_ownership(pipeline):
    texts = ["a", "b", "a", "c", "d", "e"]
    first = pipeline.embed_batch(texts, "document")
    assert [len(batch) for batch, _ in pipeline.provider.calls] == [2, 2, 1]
    first[0][0] = 999
    second = pipeline.embed_batch(texts, "document")
    assert len(pipeline.provider.calls) == 3
    assert second == [[0.6, 0.8, 0.0, 0.0]] * 6
    assert first[2][0] == 0.6


def test_cache_separates_purpose_and_model_revision(pipeline, profile, counter):
    pipeline.embed_batch(["SSL"], "document")
    pipeline.embed("SSL")
    assert len(pipeline.provider.calls) == 2
    changed = replace(profile, model_revision="other-fixed")
    other_counter = copy.copy(counter)
    other_counter.profile = changed
    other = EmbeddingPipeline(Provider(changed), other_counter, changed, cache=pipeline.cache)
    other.embed("SSL")
    assert len(other.provider.calls) == 1


def test_cache_is_bounded():
    cache = MemoryVectorCache(1)
    cache.put("a", [1])
    cache.put("b", [2])
    assert cache.get("a") is None
    assert cache.get("b") == [2]


@pytest.mark.parametrize("vector", [[0.0] * 4, [math.nan] * 4, [True] * 4, [1.0]])
def test_invalid_provider_vectors_rejected(pipeline, vector):
    pipeline.provider.embed_batch = lambda texts, purpose: [vector for _ in texts]
    with pytest.raises(EmbeddingError):
        pipeline.embed("SSL")


def test_all_lengths_checked_before_any_document_inference(pipeline):
    with pytest.raises(ValidationError):
        pipeline.embed_batch(["short", "x" * 1000], "document")
    assert pipeline.provider.calls == []


def test_profile_agreement_and_revisions(profile, counter):
    with pytest.raises(ValidationError, match="profiles differ"):
        EmbeddingPipeline(Provider(replace(profile, model_id="other")), counter, profile)
    with pytest.raises(ValidationError, match="Pin"):
        EmbeddingPipeline(Provider(profile), counter, replace(profile, model_revision=""))


def test_real_chunker_input_preserves_ids_source_and_metadata(pipeline, parsed, chunker):
    result = chunker.process(parsed("Раздел\n\nТекст о настройке SSL."), ProcessingProfile(mode="token"))
    original = copy.deepcopy(asdict(result))
    embedded = pipeline.vectorize_chunks(result)
    assert asdict(result) == original
    assert [entry.chunk for entry in embedded] == result.chunks
    assert all(entry.vector == [0.6, 0.8, 0.0, 0.0] for entry in embedded)
    assert pipeline.provider.calls[0] == ([chunk.embedding_text for chunk in result.chunks], "document")
    result.metrics["embedding_profile_id"] = "wrong"
    with pytest.raises(ValidationError):
        pipeline.vectorize_chunks(result)


def test_document_text_layout_is_not_query_normalized(pipeline):
    text = "# Заголовок\n    def code():\n        return 1\n"
    pipeline.embed_batch([text], "document")
    assert pipeline.provider.calls == [([text], "document")]


def test_fusion_experimental_envelope_is_not_silently_accepted(pipeline, parsed, chunker):
    result = chunker.process(parsed("Текст."))
    envelope = {"accepted_for_indexing": False, "source_fidelity": "unverified", "result": asdict(result)}
    with pytest.raises(ValidationError, match="acceptance"):
        pipeline.vectorize_envelope(envelope)
    assert pipeline.provider.calls == []


def test_metrics_events_and_optional_text_logging(profile, counter, caplog):
    events = []
    pipeline = EmbeddingPipeline(Provider(profile), counter, profile, on_event=events.append,
                                 cache=MemoryVectorCache())
    pipeline.embed("secret query")
    pipeline.embed(" secret query ")
    assert [(event.cache_hits, event.inferred_count) for event in events] == [(0, 1), (1, 0)]
    assert "secret query" not in caplog.text
    with caplog.at_level("INFO"):
        pipeline.log_query_text = True
        pipeline.embed("  SSL  ")
    assert caplog.records[-1].original_query == "  SSL  "
    assert caplog.records[-1].prepared_query == "SSL"


def test_async_and_close(pipeline):
    assert asyncio.run(pipeline.aembed("SSL")) == [0.6, 0.8, 0.0, 0.0]
    pipeline.close()
    pipeline.close()
    assert pipeline.provider.closed
    assert pipeline.health()["ready"] is False
    with pytest.raises(EmbeddingError, match="closed"):
        pipeline.embed("SSL")


def test_actual_infinity_and_qdrant_adapters_with_pipeline(parsed, chunker, profile, counter, tmp_path):
    """Real adapters, real parsing/chunking/manifest; HTTP service responses are mocked."""
    infinity_requests, qdrant_requests, points = [], [], {}
    def inference(request):
        body = json.loads(request.content)
        infinity_requests.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [3.0, 4.0, 0.0, 0.0]}
            for i in reversed(range(len(body["input"])))
        ]})
    def qdrant(request):
        body = json.loads(request.content) if request.content else {}
        qdrant_requests.append(body)
        if request.url.path.endswith("/points/query"):
            found = [dict(point, score=0.9) for point in points.values() if all(
                (point["payload"].get(c["key"]) == c["match"]["value"] if "value" in c["match"]
                 else point["payload"].get(c["key"]) in c["match"]["any"])
                for c in body["filter"]["must"])]
            return httpx.Response(200, json={"result": {"points": found[:body["limit"]]}})
        if request.url.path.endswith("/points"):
            if request.method == "PUT":
                points.update({point["id"]: point for point in body["points"]})
                return httpx.Response(200, json={"result": {"status": "completed"}})
            return httpx.Response(200, json={"result": [points[key] for key in body["ids"] if key in points]})
        return httpx.Response(200, json={"result": {"config": {"params": {
            "vectors": {"size": 4, "distance": "Cosine"}}}}})
    provider = InfinityEmbeddingProvider("", profile, counter, client=httpx.Client(
        base_url="http://infinity/v1/", transport=httpx.MockTransport(inference)))
    pipeline = EmbeddingPipeline(provider, counter, profile, max_tokens=provider.max_tokens)
    store = QdrantVectorStore("", "test", 4, client=httpx.Client(
        base_url="http://qdrant/", transport=httpx.MockTransport(qdrant)))
    indexer = IndexingService(SQLiteManifest(tmp_path / "state.db"), store, pipeline, counter, profile)
    result = chunker.process(parsed("Текст про SSL."), ProcessingProfile(mode="token"))
    assert indexer.index(result)["status"] == "indexed"
    hits = indexer.search("  Как\n настроить SSL? ", "tenant-a", "space-a", top_k=50)
    assert hits[0].id == result.chunks[0].id
    assert hits[0].payload["source_spans"] == asdict(result.chunks[0])["source_spans"]
    assert infinity_requests[-1]["input"] == ["query: Как настроить SSL?"]
    assert infinity_requests[0]["input"][0] == "passage: " + result.chunks[0].embedding_text
    assert qdrant_requests[-1]["query"] == [0.6, 0.8, 0.0, 0.0]
    assert indexer.search("SSL", "different-tenant", "space-a") == []
    assert indexer.search("SSL", "tenant-a", "different-space") == []
    pipeline.close()
    store.close()
