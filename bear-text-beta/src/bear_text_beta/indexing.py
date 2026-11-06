from dataclasses import asdict
from urllib.parse import quote

import httpx

from .chunking import ChunkValidator
from .embeddings import CharacterTokenCounter, validate_vectors
from .errors import StorageError, ValidationError
from .models import EmbeddedChunk, ProcessingProfile, SearchHit


class QdrantVectorStore:
    def __init__(self, base_url, collection, dimension=384, *, client=None, api_key=None):
        self.collection = quote(collection, safe="")
        self.dimension = dimension
        self._collection_ready = False
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

    def upsert(self, chunks, result):
        sources = {block.id: block.source for block in result.parsed_document.blocks}
        self.ensure_collection()
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
            self.request("PUT", f"collections/{self.collection}/points", params={"wait": "true"},
                         json={"points": points})

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
        body = self.request("POST", f"collections/{self.collection}/points/query",
                            json={"query": vector, "filter": filters, "limit": top_k,
                                  "with_payload": True, "with_vector": False})
        return [SearchHit(str(point["id"]), point["score"], point["payload"])
                for point in body["result"]["points"]]

    def health(self):
        try:
            return {"ready": self.client.get("healthz").is_success}
        except httpx.HTTPError:
            return {"ready": False}


class IndexingService:
    def __init__(self, manifest, vectors, embeddings, counter, embedding_profile, external_validators=None):
        self.manifest = manifest
        self.vectors = vectors
        self.embeddings = embeddings
        self.counter = counter
        self.embedding_profile = embedding_profile
        self.external_validators = external_validators or {}

    def index(self, result, processing_claim=None):
        if result.metrics.get("acceptance", {}).get("accepted_for_indexing") is False:
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
        try:
            for start in range(0, len(result.chunks), 16):
                batch = result.chunks[start:start + 16]
                vectors = self.embeddings.embed_batch([chunk.embedding_text for chunk in batch], "document")
                validate_vectors(vectors, len(batch), self.embedding_profile.vector_dimension)
                if processing_claim:
                    self.manifest.set_processing_phase(*processing_claim, "indexing")
                embedded = [EmbeddedChunk(chunk, vector, self.embedding_profile.identifier())
                            for chunk, vector in zip(batch, vectors, strict=True)]
                self.vectors.upsert(embedded, result)
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
                "completeness": result.completeness, "cleanup_warning": cleanup_warning}

    def retry(self, generation_id):
        return self.index(self.manifest.load(generation_id))

    def search(self, query, tenant_id, space_id, top_k=5):
        if not query.strip() or not tenant_id or not space_id or not 1 <= top_k <= 100:
            raise ValidationError("Некорректный поисковый запрос")
        generations = self.manifest.active_generations(tenant_id, space_id)
        if not generations:
            return []
        vector = self.embeddings.embed_batch([query], "query")
        validate_vectors(vector, 1, self.embedding_profile.vector_dimension)
        filters = {"must": [
            {"key": "tenant_id", "match": {"value": tenant_id}},
            {"key": "space_id", "match": {"value": space_id}},
            {"key": "generation_id", "match": {"any": generations}},
            {"key": "embedding_profile_id", "match": {"value": self.embedding_profile.identifier()}},
        ]}
        hits = self.vectors.search(vector[0], filters, top_k)
        # Recheck activation after retrieval so concurrent deletion/reindexing cannot expose stale results.
        current = set(self.manifest.active_generations(tenant_id, space_id))
        return [hit for hit in hits if hit.payload.get("tenant_id") == tenant_id
                and hit.payload.get("space_id") == space_id and hit.payload.get("generation_id") in current
                and hit.payload.get("embedding_profile_id") == self.embedding_profile.identifier()]
