"""Application layer between parsed chunks, query building and embedding inference.

No model weights, parsing logic or vector database ownership lives here.
The class implements the existing EmbeddingProvider protocol.
"""

import asyncio
import hashlib
import json
import logging
import math
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Protocol

from .embeddings import validate_vectors
from .errors import EmbeddingError, ValidationError
from .models import EmbeddedChunk, EmbeddingProfile, ProcessingResult
from .ports import EmbeddingProvider, TokenCounter


@dataclass(frozen=True)
class PreparedQuery:
    original: str
    normalized: str
    text: str
    token_count: int
    builder_version: str


@dataclass(frozen=True)
class VectorizedQuery:
    query: PreparedQuery
    vector: list[float]
    embedding_profile_id: str
    elapsed_ms: float


@dataclass(frozen=True)
class EmbeddingEvent:
    purpose: str
    input_count: int
    cache_hits: int
    inferred_count: int
    elapsed_ms: float
    embedding_profile_id: str


class VectorCache(Protocol):
    def get(self, key: str) -> list[float] | None: ...
    def put(self, key: str, vector: list[float]) -> None: ...


class MemoryVectorCache:
    """Bounded per-process LRU. Stores vectors, never source text or search hits."""

    def __init__(self, capacity: int = 4096):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Cache capacity must be a positive integer")
        self.capacity = capacity
        self._items = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return list(value)

    def put(self, key, vector):
        with self._lock:
            self._items[key] = tuple(vector)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)


class EmbeddingPipeline:
    """Receive ready text, build queries and produce validated unit dense vectors.

    ``embed``/``embed_batch`` satisfy the presentation/provider contracts;
    ``vectorize_chunks`` accepts the parser/chunker's actual ProcessingResult;
    ``vectorize_query`` returns a reproducible trace for the metrics layer.

    The injected provider adds model-specific prefixes exactly once. Documents
    retain their embedding_text verbatim, including breadcrumbs and code layout.
    """

    VERSION = "embedding-pipeline-v1"

    def __init__(
        self,
        provider: EmbeddingProvider,
        counter: TokenCounter,
        profile: EmbeddingProfile,
        *,
        batch_size: int = 16,
        max_tokens: int | None = None,
        cache: VectorCache | None = None,
        query_transform: Callable[[str], str] | None = None,
        builder_version: str = "nfc-whitespace-v1",
        on_event: Callable[[EmbeddingEvent], None] | None = None,
        log_query_text: bool = False,
        logger: logging.Logger | None = None,
    ):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(profile.vector_dimension) is not int or profile.vector_dimension < 1:
            raise ValueError("Invalid vector dimension")
        if not profile.model_revision or not profile.tokenizer_revision:
            raise ValidationError("Pin model and tokenizer revisions before vectorization")
        if not builder_version or (query_transform and builder_version == "nfc-whitespace-v1"):
            raise ValidationError("Custom query transforms need an explicit builder_version")
        for component in (provider, counter):
            actual = getattr(component, "profile", profile)
            if actual != profile:
                raise ValidationError("Provider, tokenizer and pipeline profiles differ")
        budget = profile.max_input_tokens if max_tokens is None else max_tokens
        if type(budget) is not int or budget < 1 or budget > profile.max_input_tokens:
            raise ValueError("Invalid max_tokens")
        if budget > getattr(provider, "max_tokens", budget):
            raise ValidationError("Pipeline budget exceeds provider budget")
        self.provider, self.counter, self.profile = provider, counter, profile
        self.batch_size, self.max_tokens = batch_size, budget
        self.cache = cache
        self.query_transform = query_transform
        self.builder_version = builder_version
        self.on_event = on_event
        self.log_query_text = log_query_text
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._closed = False

    @property
    def metrics(self):
        """Preserve the upstream evaluator's inference-counter contract."""
        return getattr(self.provider, "metrics", {})

    def _check_open(self):
        if self._closed:
            raise EmbeddingError("EmbeddingPipeline is closed")

    def _count(self, text, purpose):
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Embedding input must be non-empty text")
        count = self.counter.full_count(text, purpose)
        if count > self.max_tokens:
            raise ValidationError(
                f"{purpose} input has {count} tokens; limit is {self.max_tokens}; truncation is disabled"
            )
        return count

    def build_query(self, text: str) -> PreparedQuery:
        self._check_open()
        if not isinstance(text, str):
            raise ValidationError("Query must be text")
        normalized = " ".join(unicodedata.normalize("NFC", text).split())
        if not normalized:
            raise ValidationError("Query is empty after normalization")
        built = self.query_transform(normalized) if self.query_transform else normalized
        count = self._count(built, "query")
        trace = PreparedQuery(text, normalized, built, count, self.builder_version)
        if self.log_query_text:
            self.logger.info("BEAR query built", extra={"original_query": text, "prepared_query": built,
                                                      "builder_version": self.builder_version})
        return trace

    def _key(self, text, purpose):
        identity = [self.VERSION, self.profile.identifier(), self.max_tokens, "l2", purpose, text]
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()

    def embed_batch(self, texts: list[str], purpose: str = "document") -> list[list[float]]:
        self.profile.prefix(purpose)  # Reject unknown purpose even for an empty batch.
        if not isinstance(texts, list):
            raise ValidationError("texts must be a list")
        prepared = [self.build_query(text).text for text in texts] if purpose == "query" else texts
        for text in prepared:
            self._count(text, purpose)
        return self._encode(prepared, purpose)

    def _encode(self, texts, purpose):
        started = time.perf_counter()
        with self._lock:
            self._check_open()
            keys = [self._key(text, purpose) for text in texts]
            values, missing, hits = {}, {}, 0
            for key, text in zip(keys, texts, strict=True):
                if key in values or key in missing:
                    continue
                cached = self.cache.get(key) if self.cache is not None else None
                if cached is None:
                    missing[key] = text
                else:
                    validate_vectors([cached], 1, self.profile.vector_dimension)
                    values[key] = self._unit(cached)
            hits = sum(key in values for key in keys)
            pending = list(missing.items())
            # Validate and normalize the entire response before caching any batch.
            for start in range(0, len(pending), self.batch_size):
                batch = pending[start:start + self.batch_size]
                vectors = self.provider.embed_batch([text for _, text in batch], purpose)
                validate_vectors(vectors, len(batch), self.profile.vector_dimension)
                normalized = [self._unit(vector) for vector in vectors]
                for (key, _), vector in zip(batch, normalized, strict=True):
                    values[key] = vector
                    if self.cache is not None:
                        self.cache.put(key, vector)
            output = [list(values[key]) for key in keys]
        event = EmbeddingEvent(purpose, len(texts), hits, len(pending),
                               (time.perf_counter() - started) * 1000, self.profile.identifier())
        if self.on_event:
            self.on_event(event)
        return output

    @staticmethod
    def _unit(vector):
        norm = math.hypot(*(float(value) for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise EmbeddingError("Cannot normalize embedding vector")
        return [float(value) / norm for value in vector]

    def embed(self, text: str) -> list[float]:
        return self.vectorize_query(text).vector

    def vectorize_query(self, text: str) -> VectorizedQuery:
        started = time.perf_counter()
        query = self.build_query(text)
        # Do not apply query_transform twice.
        vector = self._encode([query.text], "query")[0]
        return VectorizedQuery(query, vector, self.profile.identifier(),
                               (time.perf_counter() - started) * 1000)

    def vectorize_chunks(self, result: ProcessingResult) -> list[EmbeddedChunk]:
        """Return downstream DTOs without changing chunks or activating a generation."""
        if result.metrics.get("embedding_profile_id") != self.profile.identifier():
            raise ValidationError("ProcessingResult was produced with another embedding profile")
        if result.context != result.parsed_document.context:
            raise ValidationError("ParsedDocument context mismatch")
        if len({chunk.id for chunk in result.chunks}) != len(result.chunks):
            raise ValidationError("Duplicate chunk identifiers")
        vectors = self.embed_batch([chunk.embedding_text for chunk in result.chunks], "document")
        return [EmbeddedChunk(chunk, vector, self.profile.identifier())
                for chunk, vector in zip(result.chunks, vectors, strict=True)]

    def vectorize_envelope(self, envelope: dict) -> list[EmbeddedChunk]:
        """Honor FusionPipeline's explicit acceptance boundary; never approve it here."""
        if (envelope.get("accepted_for_indexing") is not True
                or envelope.get("source_fidelity") != "verified"):
            raise ValidationError("Fusion result requires explicit source-fidelity acceptance before indexing")
        from .serialization import load_result
        return self.vectorize_chunks(load_result(envelope["result"]))

    async def aembed(self, text: str) -> list[float]:
        """Keep sync HTTP work off the event loop; one instance serializes inference."""
        return await asyncio.to_thread(self.embed, text)

    def health(self):
        if self._closed:
            return {"ready": False, "closed": True}
        if hasattr(self.provider, "health"):
            return self.provider.health()
        return {"ready": None, "reason": "provider_has_no_health_probe"}

    def close(self):
        with self._lock:
            if not self._closed and hasattr(self.provider, "close"):
                self.provider.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
