import json
import os
from uuid import uuid4

import httpx
import pytest

from bear_text_beta.errors import EmbeddingError, StorageError, ValidationError
from bear_text_beta.indexing import IndexingService, QdrantVectorStore
from bear_text_beta.models import EmbeddedChunk, SearchHit


def test_lexical_vectors_are_stable_and_match_normalized_terms():
    from bear_text_beta.retrieval import LexicalSparseProvider

    provider = LexicalSparseProvider()
    assert provider.encode("КОД   ＡＢＣ, 007") == provider.encode("007 abc код")
    assert provider.encode("  —! ") is None
    single = provider.encode("пример")
    repeated = provider.encode("пример пример")
    assert single["indices"] == repeated["indices"]
    assert single["values"] == repeated["values"] == [1.0]
    assert provider.encode("пример другой")["indices"] == sorted(provider.encode("пример другой")["indices"])


def test_runtime_provider_settings(monkeypatch):
    from bear_text_beta.retrieval import configured_reranker, configured_sparse_provider

    monkeypatch.delenv("BEAR_SPARSE_PROVIDER", raising=False)
    monkeypatch.delenv("BEAR_RERANK_MODEL", raising=False)
    assert configured_sparse_provider().id == "lexical-v1"
    assert configured_reranker() is None
    monkeypatch.setenv("BEAR_SPARSE_PROVIDER", "off")
    assert configured_sparse_provider() is None
    monkeypatch.setenv("BEAR_SPARSE_PROVIDER", "invalid")
    with pytest.raises(ValidationError, match="BEAR_SPARSE_PROVIDER"):
        configured_sparse_provider()
    monkeypatch.setenv("BEAR_RERANK_MODEL", "rerank-name")
    monkeypatch.setenv("BEAR_RERANK_URL", "http://localhost:9999/v1")
    assert configured_reranker().id == "infinity:rerank-name"


class RetrievalStore:
    def __init__(self, dense=(), sparse=(), failure=False):
        from bear_text_beta.retrieval import LexicalSparseProvider

        self.sparse_provider = LexicalSparseProvider()
        self.dense, self.sparse, self.failure = dense, sparse, failure
        self.calls = []

    def search(self, vector, filters, top_k):
        self.calls.append(("dense", filters, top_k))
        return self.dense

    def search_sparse(self, query, filters, top_k):
        self.calls.append(("sparse", filters, top_k))
        if self.failure:
            raise StorageError("simulated optional branch failure")
        return self.sparse, {"indexed_points": 2, "total_points": 3}


class Manifest:
    def __init__(self):
        self.active = ["active"]
        self.result = None

    def active_generations(self, tenant, space):
        return list(self.active)

    def load(self, generation):
        assert generation == "active"
        if self.result is None:
            raise StorageError("missing")
        return self.result


def make_hit(hit_id, profile, *, document="doc-a", generation="active", tenant="tenant", score=0.7):
    return SearchHit(hit_id, score, {"text": hit_id, "generation_id": generation,
        "tenant_id": tenant, "space_id": "space", "document_id": document,
        "embedding_profile_id": profile.identifier(), "section_path": ["Глава"]})


def test_detailed_search_preserves_scope_and_fuses_only_current_hits(profile, counter, embeddings):
    def hit(name, **options):
        return make_hit(name, profile, **options)
    store = RetrievalStore(
        [hit("foreign", tenant="outsider"), hit("dense"), hit("both"), hit("both"),
         hit("wrong-doc", document="doc-b")],
        [hit("stale", generation="stale"), hit("nan", score=float("nan")), hit("both"), hit("sparse")],
    )
    index = IndexingService(Manifest(), store, embeddings, counter, profile)
    response = index.search_detailed("пример", "tenant", "space", 3, document_ids=["doc-a"],
                                     prefetch=4, include_parents=False)
    assert response["hits"][0]["id"] == "both"
    assert {item["id"] for item in response["hits"]} == {"dense", "both", "sparse"}
    trace = response["trace"]
    assert [item["id"] for item in trace["dense"]] == ["dense", "both"]
    assert [item["rank"] for item in trace["sparse"]] == [1, 2]
    assert trace["fusion"][0]["dense_rank"] == 2
    assert trace["fusion"][0]["sparse_rank"] == 1
    assert trace["limits"]["prefetch"] == 4
    assert trace["providers"]["sparse"]["status"] == "partial_coverage"
    assert embeddings.calls == [(["пример"], "query")]
    expect = {"key": "document_id", "match": {"any": ["doc-a"]}}
    assert all(expect in filters["must"] for _, filters, _ in store.calls)


def test_unavailable_branches_degrade_dense_and_empty_scope_does_not_infer(profile, counter, embeddings):
    store = RetrievalStore([make_hit("dense", profile)], failure=True)
    manifest = Manifest()
    index = IndexingService(manifest, store, embeddings, counter, profile)
    response = index.search_detailed("пример", "tenant", "space", use_rerank=True, include_parents=False)
    assert response["hits"][0]["id"] == "dense"
    assert response["trace"]["providers"]["sparse"]["status"] == "unavailable"
    assert response["trace"]["providers"]["reranker"]["status"] == "not_configured"
    assert response["trace"]["fusion"] == []
    with pytest.raises(ValidationError):
        index.search_detailed("пример", "tenant", "space", top_k=5, prefetch=2)
    manifest.active = []
    response = index.search_detailed("пример", "tenant", "space")
    assert response["hits"] == []
    assert len(embeddings.calls) == 1


def test_sparse_only_does_not_require_dense_inference(profile, counter, embeddings):
    store = RetrievalStore(sparse=[make_hit("sparse", profile)])
    embeddings.fail = True
    index = IndexingService(Manifest(), store, embeddings, counter, profile)
    response = index.search_detailed("пример", "tenant", "space", mode="sparse", include_parents=False)
    assert [hit["id"] for hit in response["hits"]] == ["sparse"]
    assert not embeddings.calls
    assert response["trace"]["providers"]["dense"]["status"] == "disabled"
    with pytest.raises(EmbeddingError):
        index.search_detailed("пример", "tenant", "space", mode="dense")


def test_rerank_records_rank_changes_and_rechecks_activation(profile, counter, embeddings):
    store = RetrievalStore([make_hit("a", profile), make_hit("b", profile)])
    manifest = Manifest()

    class Reranker:
        id = "example"
        fail = False
        remove = False

        def score(self, query, passages):
            assert query == "пример"
            assert passages == ["a", "b"]
            if self.fail:
                raise EmbeddingError("unavailable")
            if self.remove:
                manifest.active = []
            return [0.1, 0.9]

    reranker = Reranker()
    index = IndexingService(manifest, store, embeddings, counter, profile, reranker=reranker)
    options = {"mode": "dense", "top_k": 1, "rerank_candidates": 2, "use_rerank": True, "include_parents": False}
    response = index.search_detailed("пример", "tenant", "space", **options)
    assert response["hits"][0]["id"] == "b"
    assert response["trace"]["rerank"][0]["original_rank"] == 2
    assert response["trace"]["rerank"][0]["rank"] == 1
    reranker.fail = True
    response = index.search_detailed("пример", "tenant", "space", **options)
    assert response["hits"][0]["id"] == "a"
    assert response["trace"]["providers"]["reranker"]["status"] == "unavailable"
    reranker.fail, reranker.remove = False, True
    response = index.search_detailed("пример", "tenant", "space", **options)
    assert response["hits"] == []
    assert response["trace"]["dense"] == response["trace"]["rerank"] == []


def test_parent_expansion_uses_only_saved_scoped_generation(parsed, chunker, profile, counter, embeddings):
    from bear_fusion.parents import attach_parents

    result = chunker.process(parsed("Текст из сохранённого разбора."))
    result.generation_id = "active"
    result.context.tenant_id = "tenant"
    result.context.space_id = "space"
    result.metrics["parents"] = attach_parents(result)
    manifest = Manifest()
    manifest.result = result
    hit = make_hit(result.chunks[0].id, profile)
    index = IndexingService(manifest, RetrievalStore([hit]), embeddings, counter, profile)
    response = index.search_detailed("пример", "tenant", "space", mode="dense")
    parent = response["trace"]["parents"][0]
    assert parent["text"] == result.chunks[0].text
    assert parent["sources"]
    assert not parent["truncated"]
    result.context.tenant_id = "foreign"
    response = index.search_detailed("пример", "tenant", "space", mode="dense")
    assert not response["trace"]["parents"]


def test_infinity_reranker_restores_request_order_and_rejects_duplicates():
    from bear_text_beta.retrieval import InfinityReranker

    results = [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.3}]

    def handler(request):
        assert request.url.path == "/v1/rerank"
        assert request.headers["Authorization"] == "Bearer token"
        assert json.loads(request.content) == {"model": "model", "query": "q",
            "documents": ["first", "second"], "return_documents": False}
        return httpx.Response(200, json={"results": results})

    provider = InfinityReranker("http://infinity/v1", "model", api_key="token",
                                transport=httpx.MockTransport(handler))
    assert provider.score("q", ["first", "second"]) == [0.3, 0.8]
    results[1]["index"] = 1
    with pytest.raises(EmbeddingError):
        provider.score("q", ["first", "second"])


def test_qdrant_old_collection_gains_optional_sparse_vectors(parsed, chunker, profile):
    from bear_text_beta.retrieval import LexicalSparseProvider

    points, requests, config = {}, [], {"vectors": {"size": 4, "distance": "Cosine"}}

    def handler(request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request.method, request.url.path, body))
        match request.url.path.split("/beta", 1)[1]:
            case "":
                return httpx.Response(200, json={"result": {"config": {"params": config}}})
            case "/vectors/lexical_v1":
                assert body == {"sparse": {"modifier": "idf"}}
                config["sparse_vectors"] = {"lexical_v1": {"modifier": "idf"}}
            case "/points":
                points.update({point["id"]: point for point in body["points"]})
            case "/points/count":
                sparse = any("has_vector" in entry for entry in body["filter"]["must"])
                return httpx.Response(200, json={"result": {"count": 1 if sparse else 2}})
            case "/points/query":
                assert body["using"] == "lexical_v1"
                assert body["query"] == provider.encode("чай")
                return httpx.Response(200, json={"result": {"points": []}})
            case _:
                pytest.fail(f"Unexpected route {request.url.path}")
        return httpx.Response(200, json={"result": True})

    provider = LexicalSparseProvider()
    store = QdrantVectorStore("", "beta", 4, sparse_provider=provider, client=httpx.Client(
        base_url="http://qdrant/", transport=httpx.MockTransport(handler)))
    result = chunker.process(parsed("Чай с мятой."))
    embedded = EmbeddedChunk(result.chunks[0], [1.0, 0, 0, 0], profile.identifier())
    store.upsert([embedded], result)
    assert points[result.chunks[0].id]["vector"][""] == embedded.vector
    assert "lexical_v1" in points[result.chunks[0].id]["vector"]
    assert store.search_sparse("чай", {"must": []}, 5) == ([], {"indexed_points": 1, "total_points": 2})
    assert sum(path.endswith("/vectors/lexical_v1") for _, path, _ in requests) == 1


def test_sparse_write_failure_retries_dense_points(parsed, chunker, profile):
    from bear_text_beta.retrieval import LexicalSparseProvider

    writes = []

    def handler(request):
        data = json.loads(request.content) if request.content else {}
        if request.method == "GET":
            return httpx.Response(200, json={"result": {"config": {"params": {
                "vectors": {"size": 4, "distance": "Cosine"},
                "sparse_vectors": {"lexical_v1": {"modifier": "idf"}},
            }}}})
        writes.append(data["points"][0]["vector"])
        return httpx.Response(400 if isinstance(writes[-1], dict) else 200, json={"result": {}})

    store = QdrantVectorStore("", "beta", 4, sparse_provider=LexicalSparseProvider(), client=httpx.Client(
        base_url="http://qdrant/", transport=httpx.MockTransport(handler)))
    result = chunker.process(parsed("Текст."))
    embedded = EmbeddedChunk(result.chunks[0], [1.0, 0, 0, 0], profile.identifier())
    assert store.upsert([embedded], result) == "sparse_index_unavailable"
    assert writes[0][""] == writes[1] == embedded.vector
    assert store.search_sparse("?!", {"must": []}, 5) == ([], None)


@pytest.mark.skipif(not all(os.getenv(key) for key in (
    "BEAR_QDRANT_TEST_URL", "BEAR_INFINITY_TEST_URL", "BEAR_TEST_MODEL_DIR",
)), reason="requires a live local Qdrant, Infinity and BGE-M3 model")
def test_live_bge_m3_dense_sparse_hybrid(tmp_path, monkeypatch):
    from bear_text_beta.models import DocumentContext, ProcessingProfile
    from bear_text_beta.runtime import Services

    collection = f"bear_live_retrieval_{uuid4().hex}"
    endpoint = os.environ["BEAR_QDRANT_TEST_URL"]
    monkeypatch.setenv("BEAR_QDRANT_URL", endpoint)
    monkeypatch.setenv("BEAR_QDRANT_COLLECTION", collection)
    monkeypatch.setenv("BEAR_INFINITY_URL", os.environ["BEAR_INFINITY_TEST_URL"])
    monkeypatch.setenv("BEAR_INFINITY_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("BEAR_SPARSE_PROVIDER", "lexical")
    with httpx.Client(base_url=endpoint, timeout=30, trust_env=False) as client:
        services = Services(tmp_path, os.environ["BEAR_TEST_MODEL_DIR"])
        try:
            ctx = DocumentContext("t", "s", "arctic", 1, "notes.txt")
            text = "Арктический договор определяет судоходство в северных водах."
            result = services.process(text.encode(), ctx, ProcessingProfile())
            assert services.profile.model_id == "BAAI/bge-m3"
            assert services.indexer.index(result)["status"] == "indexed"
            query = "Арктический договор"
            modes = {}
            for mode in ("dense", "sparse", "hybrid"):
                response = services.indexer.search_detailed(query, "t", "s", document_ids=["arctic"], mode=mode)
                assert response["hits"][0]["payload"]["text"] == text
                assert response["trace"]["parents"][0]["text"] == text
                modes[mode] = response["trace"]["timings_ms"]
                if mode != "dense":
                    assert response["trace"]["sparse_coverage"] == {"indexed_points": 1, "total_points": 1}
                if mode == "hybrid":
                    assert response["trace"]["fusion"][0]["dense_rank"] == 1
                    assert response["trace"]["fusion"][0]["sparse_rank"] == 1
            assert not services.indexer.search_detailed(query, "other-tenant", "s")["hits"]
            assert not services.indexer.search_detailed(query, "t", "s", document_ids=["other-doc"])["hits"]
            print(json.dumps({"embedding": services.profile.model_id, "dimension": services.profile.vector_dimension,
                              "modes_ms": modes}, ensure_ascii=False))
        finally:
            services.close()
            deleted = client.delete(f"/collections/{collection}")
            assert deleted.status_code in {200, 404}
