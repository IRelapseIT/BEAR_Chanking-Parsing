from dataclasses import asdict
from urllib.parse import quote

import httpx

from .chunking import ChunkValidator
from .embeddings import CharacterTokenCounter, validate_vectors
from .errors import EmbeddingError, StorageError, ValidationError
from .models import EmbeddedChunk, ProcessingProfile, SearchHit
from .retrieval import RetrievalService, SparseProvider


class QdrantVectorStore:
    def __init__(self, base_url, collection, dimension=384, *, client=None, api_key=None,
                 sparse_provider: SparseProvider | None = None):
        self.collection = quote(collection, safe="")
        self.dimension = dimension
        self.sparse_provider = sparse_provider
        self._collection_ready = False
        self._sparse_ready = False
        self.client = client or httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=30,
                                            headers={"api-key": api_key} if api_key else {})

    def close(self):
        self.client.close()

    def request(self, method, path, **kwargs):
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise StorageError("Qdrant недоступен или вернул ошибку") from exc

    def ensure_collection(self):
        if self._collection_ready:
            return
        path = f"collections/{self.collection}"
        try:
            response = self.client.get(path)
            if response.status_code == 404:
                # Another worker may create the collection between our GET and PUT.
                created = self.client.put(path, json={"vectors": {
                    "size": self.dimension, "distance": "Cosine"}})
                if not created.is_success and created.status_code not in {400, 409}:
                    created.raise_for_status()
                response = self.client.get(path)
            response.raise_for_status()
            config = response.json()["result"]["config"]["params"]["vectors"]
            if config["size"] != self.dimension or config["distance"] != "Cosine":
                raise StorageError("Коллекция Qdrant имеет несовместимые vectors")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise StorageError("Не удалось проверить конфигурацию Qdrant") from exc
        self._collection_ready = True

    def ensure_sparse(self):
        if self._sparse_ready:
            return
        name = self.sparse_provider.vector_name
        path = f"collections/{self.collection}"
        self.ensure_collection()

        def schema():
            try:
                params = self.request("GET", path)["result"]["config"]["params"]
                return (params.get("sparse_vectors") or {}).get(name)
            except (AttributeError, KeyError, TypeError) as exc:
                raise StorageError("Не удалось прочитать схему sparse-вектора Qdrant") from exc

        existing = schema()
        try:
            if existing is None:
                # Qdrant's global IDF can change scoped scores when another tenant adds points.
                self.request("PUT", f"{path}/vectors/{quote(name, safe='')}", json={"sparse": {}})
        except StorageError:
            # Another worker may have added the same name between GET and PUT.
            existing = schema()
            if existing is None:
                raise
        if existing is not None and (not isinstance(existing, dict) or
                                      str(existing.get("modifier")).lower() != "none"):
            raise StorageError("Несовместимая схема sparse-вектора Qdrant")
        self._sparse_ready = True

    def upsert(self, chunks, result):
        sources = {block.id: block.source for block in result.parsed_document.blocks}
        self.ensure_collection()
        use_sparse = self.sparse_provider is not None
        sparse_unavailable = False
        if use_sparse:
            try:
                self.ensure_sparse()
            except StorageError:
                use_sparse, sparse_unavailable = False, True
        context = result.context
        for index in range(0, len(chunks), 64):
            points = []
            for embedded in chunks[index:index + 64]:
                payload = asdict(embedded.chunk)
                payload["sources"] = [{"block_id": span.block_id, "start_char": span.start_char,
                                       "end_char": span.end_char, **asdict(sources[span.block_id])}
                                      for span in embedded.chunk.source_spans]
                payload["acceptance"] = result.metrics.get("acceptance", {})
                payload.update(tenant_id=context.tenant_id, space_id=context.space_id,
                               document_id=context.document_id, document_version=context.document_version,
                               generation_id=result.generation_id, chunk_id=embedded.chunk.id,
                               processing_profile_id=result.processing_profile_id,
                               embedding_profile_id=embedded.embedding_profile_id,
                               completeness=result.completeness)
                points.append({"id": embedded.chunk.id, "vector": embedded.vector, "payload": payload})
            path = f"collections/{self.collection}/points"
            if use_sparse:
                try:
                    hybrid = []
                    for point in points:
                        sparse = self.sparse_provider.encode(point["payload"]["embedding_text"])
                        hybrid.append({**point, "vector": {"": point["vector"], self.sparse_provider.vector_name: sparse}}
                                      if sparse is not None else point)
                    self.request("PUT", path, params={"wait": "true"}, json={"points": hybrid})
                    continue
                except (EmbeddingError, StorageError):
                    use_sparse, sparse_unavailable = False, True
                    self._sparse_ready = False
            self.request("PUT", path, params={"wait": "true"}, json={"points": points})
        if sparse_unavailable:
            return "sparse_index_unavailable"
        return None

    def verify_generation(self, generation_id, point_ids):
        for index in range(0, len(point_ids), 128):
            ids = point_ids[index:index + 128]
            body = self.request("POST", f"collections/{self.collection}/points",
                                json={"ids": ids, "with_payload": True, "with_vector": False})
            points = body["result"]
            if {str(point["id"]) for point in points} != set(ids):
                return False
            if any(point["payload"].get("generation_id") != generation_id for point in points):
                return False
        return True

    def delete_generation(self, generation_id):
        self.request("POST", f"collections/{self.collection}/points/delete", params={"wait": "true"},
                     json={"filter": {"must": [{"key": "generation_id", "match": {"value": generation_id}}]}})

    def search(self, vector, filters, top_k):
        return self._query(vector, filters, top_k)

    def _query(self, vector, filters, top_k, using=None):
        options = {"using": using} if using is not None else {}
        body = self.request("POST", f"collections/{self.collection}/points/query",
                            json={"query": vector, "filter": filters, "limit": top_k,
                                  "with_payload": True, "with_vector": False, **options})
        try:
            return [SearchHit(str(point["id"]), point["score"], point["payload"])
                    for point in body["result"]["points"] if isinstance(point.get("payload"), dict)]
        except (AttributeError, KeyError, TypeError) as exc:
            raise StorageError("Некорректная поисковая выдача Qdrant") from exc

    def search_sparse(self, query, filters, top_k):
        vector = self.sparse_provider.encode(query)
        if vector is None:
            return [], None
        self.ensure_sparse()
        name = self.sparse_provider.vector_name
        hits = self._query(vector, filters, top_k, using=name)
        indexed_filter = {**filters, "must": [*filters["must"], {"has_vector": name}]}
        count_path = f"collections/{self.collection}/points/count"
        try:
            counts = [self.request("POST", count_path, json={"exact": True, "filter": condition})["result"]["count"]
                      for condition in (indexed_filter, filters)]
        except (KeyError, TypeError) as exc:
            raise StorageError("Некорректный подсчёт точек Qdrant") from exc
        if any(type(count) is not int or count < 0 for count in counts):
            raise StorageError("Некорректный подсчёт точек Qdrant")
        return hits, {"indexed_points": counts[0], "total_points": counts[1]}

    def health(self):
        try:
            return {"ready": self.client.get("healthz").is_success}
        except httpx.HTTPError:
            return {"ready": False}


class IndexingService:
    def __init__(self, manifest, vectors, embeddings, counter, embedding_profile,
                 external_validators=None, *, reranker=None):
        self.manifest = manifest
        self.vectors = vectors
        self.embeddings = embeddings
        self.counter = counter
        self.embedding_profile = embedding_profile
        self.external_validators = external_validators or {}
        self.retrieval = RetrievalService(manifest, vectors, embeddings, counter, embedding_profile, reranker)

    def index(self, result, processing_claim=None):
        from .acceptance import assess
        if (result.metrics.get("acceptance", {}).get("accepted_for_indexing") is False
                or not assess(result)["accepted_for_indexing"]):
            raise ValidationError("Документ не прошёл установленную политику приёмки")
        if isinstance(self.counter, CharacterTokenCounter):
            raise ValidationError("Тестовый символьный токенизатор нельзя использовать для индексации")
        if result.metrics.get("embedding_profile_id") != self.embedding_profile.identifier():
            raise ValidationError("Checkpoint создан с другим embedding-профилем")
        profile = ProcessingProfile(**result.metrics["processing_profile"])
        from .chunking import ChunkingService
        expected = ChunkingService(self.counter, self.embeddings, self.embedding_profile).profile_id(
            profile, result.parsed_document.parser_profile)
        if result.processing_profile_id != expected or result.metrics.get("token_counter") != type(self.counter).__name__:
            raise ValidationError("Processing profile checkpoint несовместим с текущим runtime")
        ChunkValidator(self.counter, min(profile.max_tokens, self.embedding_profile.max_input_tokens),
                       external_validators=self.external_validators).validate(result)
        if not result.chunks:
            raise ValidationError("Нет пригодных чанков для индексации")
        self.manifest.save(result)
        if not self.manifest.begin_index(result.generation_id, processing_claim):
            return {"generation_id": result.generation_id, "status": "already_indexed",
                    "completeness": result.completeness}
        retrieval_warnings = set()
        try:
            for start in range(0, len(result.chunks), 16):
                batch = result.chunks[start:start + 16]
                vectors = self.embeddings.embed_batch([chunk.embedding_text for chunk in batch], "document")
                validate_vectors(vectors, len(batch), self.embedding_profile.vector_dimension)
                if processing_claim:
                    self.manifest.set_processing_phase(*processing_claim, "indexing")
                embedded = [EmbeddedChunk(chunk, vector, self.embedding_profile.identifier())
                            for chunk, vector in zip(batch, vectors, strict=True)]
                sparse_warning = self.vectors.upsert(embedded, result)
                if sparse_warning == "sparse_index_unavailable":
                    retrieval_warnings.add(sparse_warning)
                if processing_claim:
                    self.manifest.set_processing_phase(*processing_claim, "embedding")
            if not self.vectors.verify_generation(result.generation_id, [chunk.id for chunk in result.chunks]):
                raise StorageError("Поколение Qdrant записано не полностью")
            previous = self.manifest.activate(result.generation_id, processing_claim)
        except Exception as exc:
            self.manifest.fail(result.generation_id, exc)
            raise
        cleanup_warning = None
        if previous and previous != result.generation_id:
            try:
                self.vectors.delete_generation(previous)
            except StorageError:
                cleanup_warning = "old_generation_cleanup_pending"
        return {"generation_id": result.generation_id, "status": "indexed", "chunks": len(result.chunks),
                "completeness": result.completeness, "cleanup_warning": cleanup_warning,
                "retrieval_warnings": sorted(retrieval_warnings)}

    def retry(self, generation_id):
        return self.index(self.manifest.load(generation_id))

    def search(self, query, tenant_id, space_id, top_k=5):
        result = self.search_detailed(query, tenant_id, space_id, top_k, mode="dense",
                                      prefetch=top_k, include_parents=False)
        return [SearchHit(**hit) for hit in result["hits"]]

    def search_detailed(self, query, tenant_id, space_id, top_k=5, **options):
        return self.retrieval.search(query, tenant_id, space_id, top_k, **options)
