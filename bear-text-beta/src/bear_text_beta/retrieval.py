"""Scoped dense, sparse and hybrid retrieval over one active index."""

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict
from hashlib import blake2s
from time import perf_counter
from typing import Protocol, TypedDict

import httpx

from .embeddings import validate_vectors
from .errors import BearError, EmbeddingError, StorageError, ValidationError
from .models import SearchHit


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class SparseProvider(Protocol):
    id: str
    vector_name: str

    def encode(self, text: str) -> SparseVector | None: ...


class Reranker(Protocol):
    id: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class LexicalSparseProvider:
    """Unicode words and numbers with normalized term frequencies."""

    id = "lexical-v1"
    vector_name = "lexical_v1"
    words = re.compile(r"[^\W_]+", re.UNICODE)

    def encode(self, text: str) -> SparseVector | None:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        counts = Counter(
            int.from_bytes(blake2s(word.group().encode(), digest_size=4).digest())
            for word in self.words.finditer(normalized)
        )
        if not counts:
            return None
        indices = sorted(counts)
        weights = [1.0 + math.log(counts[index]) for index in indices]
        norm = math.sqrt(sum(value * value for value in weights))
        return {"indices": indices, "values": [value / norm for value in weights]}


class InfinityReranker:
    def __init__(self, base_url, model, *, api_key=None, transport=None):
        self.id = f"infinity:{model}"
        self.model = model
        self.url = base_url.rstrip("/") + "/rerank"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.transport = transport

    def score(self, query: str, passages: list[str]) -> list[float]:
        try:
            with httpx.Client(headers=self.headers, timeout=30, transport=self.transport) as client:
                response = client.post(self.url, json={"model": self.model, "query": query,
                    "documents": passages, "return_documents": False})
                response.raise_for_status()
            indexed = {}
            for item in response.json()["results"]:
                index, score = item["index"], item["relevance_score"]
                if (type(index) is not int or not 0 <= index < len(passages) or index in indexed
                        or not _valid_score(score)):
                    raise ValueError("invalid rerank result")
                indexed[index] = score
            return [indexed[index] for index in range(len(passages))]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise EmbeddingError("Реранкер Infinity недоступен или вернул некорректный ответ") from exc


def configured_sparse_provider() -> SparseProvider | None:
    name = os.getenv("BEAR_SPARSE_PROVIDER", "lexical").strip().lower()
    if name == "lexical":
        return LexicalSparseProvider()
    if name == "off":
        return None
    raise ValidationError("BEAR_SPARSE_PROVIDER должен быть lexical или off")


def configured_reranker() -> Reranker | None:
    model = os.getenv("BEAR_RERANK_MODEL", "").strip()
    if not model:
        return None
    return InfinityReranker(
        os.getenv("BEAR_RERANK_URL") or os.getenv("BEAR_INFINITY_URL", "http://127.0.0.1:7998/v1"),
        model, api_key=os.getenv("BEAR_RERANK_API_KEY") or os.getenv("BEAR_INFINITY_API_KEY"),
    )


def _valid_score(score):
    return type(score) in (int, float) and math.isfinite(score)


def _elapsed(started):
    return (perf_counter() - started) * 1000


def _rrf(dense, sparse, k=60):
    ranks = [{hit.id: rank for rank, hit in enumerate(branch, 1)} for branch in (dense, sparse)]
    by_id = {hit.id: hit for hit in (*dense, *sparse)}
    rows = []
    for key in by_id:
        d, s = (branch.get(key) for branch in ranks)
        score = sum(1 / (k + rank) for rank in (d, s) if rank is not None)
        rows.append((SearchHit(key, score, by_id[key].payload), d, s))
    order = sorted(rows, key=lambda row: (-row[0].score, row[1] or math.inf, row[2] or math.inf, row[0].id))
    return [row[0] for row in order], [
        {"id": hit.id, "rank": rank, "score": hit.score, "dense_rank": d, "sparse_rank": s}
        for rank, (hit, d, s) in enumerate(order, 1)
    ]


class RetrievalService:
    def __init__(self, manifest, vectors, embeddings, counter, profile, reranker):
        self.manifest = manifest
        self.vectors = vectors
        self.embeddings = embeddings
        self.counter = counter
        self.profile = profile
        self.reranker = reranker

    def search(self, query, tenant_id, space_id, top_k=5, *, document_ids=(), mode="hybrid",
               prefetch=None, rerank_candidates=None, use_rerank=False, include_parents=True,
               context_max_tokens=480):
        started = perf_counter()
        query = query.strip()
        if not 1 <= len(query) <= 16_000 or not tenant_id or not space_id or not 1 <= top_k <= 100:
            raise ValidationError("Некорректный поисковый запрос")
        prefetch = max(top_k, 20) if prefetch is None else prefetch
        candidates = min(prefetch, max(top_k, 20)) if rerank_candidates is None else rerank_candidates
        if not top_k <= candidates <= prefetch <= 200:
            raise ValidationError("Требуется top_k ≤ rerank_candidates ≤ prefetch ≤ 200")
        if mode not in {"dense", "sparse", "hybrid"} or not 1 <= context_max_tokens <= 4096:
            raise ValidationError("Некорректный режим поиска или лимит контекста")
        if isinstance(document_ids, str) or len(document_ids) > 100 or any(
            not isinstance(key, str) or not key.strip() for key in document_ids
        ):
            raise ValidationError("Некорректный список документов")
        documents = frozenset(document_ids)
        generations = set(self.manifest.active_generations(tenant_id, space_id))
        profile_id = self.profile.identifier()
        fields = {"tenant_id": tenant_id, "space_id": space_id, "embedding_profile_id": profile_id}
        conditions = [{"key": key, "match": {"value": value}} for key, value in fields.items()]
        conditions.append({"key": "generation_id", "match": {"any": sorted(generations)}})
        if documents:
            conditions.append({"key": "document_id", "match": {"any": sorted(documents)}})
        filters = {"must": conditions}
        sparse_provider = getattr(self.vectors, "sparse_provider", None)
        providers = {
            "dense": {"id": self.profile.model_id, "profile": profile_id,
                      "status": "disabled" if mode == "sparse" else "empty_scope"},
            "sparse": {"id": sparse_provider.id if sparse_provider else None,
                       "status": "disabled" if mode == "dense" else
                       "empty_scope" if sparse_provider else "not_configured"},
            "reranker": {"id": self.reranker.id if self.reranker else None,
                         "status": "disabled" if not use_rerank else
                         "no_candidates" if self.reranker else "not_configured"},
        }
        timings = {}
        warnings = []
        dense, sparse, coverage = [], [], None

        def scoped(hits, active):
            seen = set()
            result = []
            for hit in hits:
                if (hit.id in seen or not _valid_score(hit.score)
                        or hit.payload.get("generation_id") not in active
                        or (documents and hit.payload.get("document_id") not in documents)
                        or any(hit.payload.get(key) != value for key, value in fields.items())):
                    continue
                seen.add(hit.id)
                result.append(hit)
            return result

        if generations and mode != "sparse":
            phase = perf_counter()
            vector = self.embeddings.embed_batch([query], "query")
            validate_vectors(vector, 1, self.profile.vector_dimension)
            timings["embedding"] = _elapsed(phase)
            phase = perf_counter()
            dense = self.vectors.search(vector[0], filters, prefetch)
            timings["dense"] = _elapsed(phase)
            providers["dense"]["status"] = "used"
        if generations and mode != "dense" and sparse_provider:
            phase = perf_counter()
            try:
                sparse, coverage = self.vectors.search_sparse(query, filters, prefetch)
                providers["sparse"]["status"] = (
                    "no_terms" if coverage is None else "partial_coverage"
                    if coverage["indexed_points"] < coverage["total_points"] else "used"
                )
            except (EmbeddingError, StorageError):
                providers["sparse"]["status"] = "unavailable"
                warnings.append("sparse_unavailable")
            timings["sparse"] = _elapsed(phase)
        active = set(self.manifest.active_generations(tenant_id, space_id)) if generations else generations
        dense, sparse = scoped(dense, active), scoped(sparse, active)

        fusion = []
        if dense and sparse:
            phase = perf_counter()
            ranked, fusion = _rrf(dense, sparse)
            ranked, fusion = ranked[:prefetch], fusion[:prefetch]
            timings["fusion"] = _elapsed(phase)
        else:
            ranked = dense or sparse

        rerank_rows = []
        if use_rerank and self.reranker and ranked:
            phase = perf_counter()
            initial = ranked[:candidates]
            passages = [hit.payload.get("embedding_text") or hit.payload["text"] for hit in initial]
            try:
                scores = self.reranker.score(query, passages)
                if len(scores) != len(initial) or not all(_valid_score(score) for score in scores):
                    raise EmbeddingError("Недопустимые оценки реранкера")
                scored = sorted(zip(scores, range(1, len(initial) + 1), initial, strict=True),
                                key=lambda entry: (-entry[0], entry[1]))
                ranked = [SearchHit(hit.id, score, hit.payload) for score, _, hit in scored]
                rerank_rows = [{"id": hit.id, "rank": rank, "original_rank": original,
                                "previous_score": hit.score, "score": score}
                               for rank, (score, original, hit) in enumerate(scored, 1)]
                providers["reranker"]["status"] = "used"
            except (EmbeddingError, StorageError):
                providers["reranker"]["status"] = "unavailable"
                warnings.append("reranker_unavailable")
                rerank_rows = [{"id": hit.id, "rank": rank, "original_rank": rank,
                                "previous_score": hit.score, "score": None}
                               for rank, hit in enumerate(initial, 1)]
            timings["rerank"] = _elapsed(phase)

        parents = []
        if include_parents and ranked:
            phase = perf_counter()
            parents = self._parents(ranked[:top_k], tenant_id, space_id, context_max_tokens, warnings)
            timings["parents"] = _elapsed(phase)

        # The trace and the expanded context must obey the same scope as the final hits.
        active = set(self.manifest.active_generations(tenant_id, space_id)) if generations else generations
        dense, sparse = scoped(dense, active), scoped(sparse, active)
        final = scoped(ranked, active)[:top_k]
        remaining = {hit.id for hit in (*dense, *sparse)}
        final_ids = {hit.id for hit in final}
        timings["total"] = _elapsed(started)

        def branch(hits):
            return [{"id": hit.id, "rank": rank, "score": hit.score,
                     "document_id": hit.payload.get("document_id"),
                     "generation_id": hit.payload["generation_id"], "text": hit.payload.get("text")}
                    for rank, hit in enumerate(hits, 1)]

        return {"hits": [asdict(hit) for hit in final], "trace": {
            "dense": branch(dense), "sparse": branch(sparse),
            "fusion": [item for item in fusion if item["id"] in remaining],
            "rerank": [item for item in rerank_rows if item["id"] in remaining],
            "parents": [parent for parent in parents if parent["seed_id"] in final_ids],
            "providers": providers, "timings_ms": timings, "warnings": warnings,
            "sparse_coverage": coverage,
            "limits": {"top_k": top_k, "prefetch": prefetch, "rerank_candidates": candidates,
                       "context_max_tokens": context_max_tokens, "rrf_k": 60},
        }}

    def _parents(self, hits, tenant_id, space_id, max_tokens, warnings):
        try:
            from bear_fusion.parents import expand_context
        except ModuleNotFoundError:
            warnings.append("parent_expansion_unavailable")
            return []
        results, by_generation, seen = [], {}, set()
        for rank, hit in enumerate(hits):
            by_generation.setdefault(hit.payload["generation_id"], []).append((rank, hit))
        incomplete = False
        for generation, seeds in by_generation.items():
            try:
                result = self.manifest.load(generation)
            except BearError:
                incomplete = True
                continue
            context = result.context
            if (context.tenant_id, context.space_id, result.generation_id) != (tenant_id, space_id, generation):
                incomplete = True
                continue
            envelope = {"result": {"chunks": [asdict(chunk) for chunk in result.chunks],
                                   "context": asdict(context), "generation_id": generation}}
            sources = {block.id: block.source for block in result.parsed_document.blocks}
            for rank, hit in seeds:
                if hit.payload.get("document_id") != context.document_id:
                    incomplete = True
                    continue
                try:
                    parent = expand_context(envelope, hit.id, self.counter, max_tokens)
                except ValueError:
                    incomplete = True
                    continue
                key = (generation, tuple(parent["parent_ids"]) or (hit.id,))
                if key not in seen:
                    seen.add(key)
                    parent["sources"] = [{**span, **asdict(sources[span["block_id"]])}
                                         for span in parent["sources"] if span["block_id"] in sources]
                    results.append((rank, {**parent, "seed_id": hit.id}))
        if incomplete:
            warnings.append("parent_context_incomplete")
        return [item[1] for item in sorted(results)]
