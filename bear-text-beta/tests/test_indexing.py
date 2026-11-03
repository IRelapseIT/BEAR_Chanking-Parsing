import copy
import json
from dataclasses import asdict

import httpx
import pytest

from bear_text_beta.errors import EmbeddingError, StorageError, ValidationError
from bear_text_beta.indexing import IndexingService, QdrantVectorStore
from bear_text_beta.models import SearchHit
from bear_text_beta.storage import SQLiteManifest
from bear_text_beta.workflow import IndexingWorker, OutboxPublisher


class FakeVectors:
    def __init__(self):
        self.points = {}
        self.fail_partial = False
        self.fail_cleanup = False

    def upsert(self, chunks, result):
        for i, embedded in enumerate(chunks):
            payload = asdict(embedded.chunk)
            payload.update(asdict(result.context))
            payload.update(generation_id=result.generation_id, embedding_profile_id=embedded.embedding_profile_id)
            self.points[embedded.chunk.id] = SearchHit(embedded.chunk.id, 0.9, payload)
            if self.fail_partial and i == 0:
                raise StorageError("partial write")

    def verify_generation(self, generation_id, ids):
        return all(key in self.points and self.points[key].payload["generation_id"] == generation_id for key in ids)

    def delete_generation(self, generation_id):
        if self.fail_cleanup:
            raise StorageError("cleanup failure")
        self.points = {key: point for key, point in self.points.items()
                       if point.payload["generation_id"] != generation_id}

    def search(self, vector, filters, top_k):
        def matches(point):
            for condition in filters["must"]:
                actual = point.payload.get(condition["key"])
                match = condition["match"]
                if "value" in match and actual != match["value"]:
                    return False
                if "any" in match and actual not in match["any"]:
                    return False
            return True
        return [point for point in self.points.values() if matches(point)][:top_k]

    def health(self):
        return {"ready": True}


@pytest.fixture
def indexing(tmp_path, counter, embeddings, profile):
    manifest = SQLiteManifest(tmp_path / "manifest.sqlite3")
    store = FakeVectors()
    return IndexingService(manifest, store, embeddings, counter, profile)


def test_retry_duplicate_delivery_and_payload(parsed, chunker, indexing):
    result = chunker.process(parsed("Текст документа."))
    assert indexing.index(result)["status"] == "indexed"
    assert indexing.retry(result.generation_id)["status"] == "already_indexed"
    assert len(indexing.vectors.points) == len(result.chunks)
    hit = indexing.search("Вопрос", "tenant-a", "space-a")[0]
    assert hit.payload["text"] == result.chunks[0].text
    assert indexing.search("Вопрос", "other", "space-a") == []
    assert indexing.search("Вопрос", "tenant-a", "other") == []


def test_partial_upsert_keeps_previous_generation(parsed, chunker, indexing):
    first = chunker.process(parsed("Старый текст."))
    indexing.index(first)
    document = parsed("Новый текст." * 80)
    document.context.document_version = 2
    second = chunker.process(document)
    indexing.vectors.fail_partial = True
    with pytest.raises(StorageError):
        indexing.index(second)
    hits = indexing.search("Вопрос", "tenant-a", "space-a")
    assert {hit.payload["generation_id"] for hit in hits} == {first.generation_id}
    indexing.vectors.fail_partial = False
    indexing.retry(second.generation_id)
    assert indexing.manifest.active_generations("tenant-a", "space-a") == [second.generation_id]
    assert len(indexing.vectors.points) == len(second.chunks)


def test_embedding_failure_and_three_attempt_limit(parsed, chunker, indexing, embeddings):
    result = chunker.process(parsed("Текст."))
    embeddings.fail = True
    for _ in range(3):
        with pytest.raises(EmbeddingError):
            indexing.index(result)
    with pytest.raises(StorageError, match="три"):
        indexing.retry(result.generation_id)
    assert not indexing.manifest.active_generations("tenant-a", "space-a")
    assert indexing.manifest.load(result.generation_id).chunks == result.chunks


def test_deletion_before_activation_cannot_resurrect(parsed, chunker, indexing):
    result = chunker.process(parsed("Текст."))
    indexing.manifest.save(result)
    indexing.manifest.begin_index(result.generation_id)
    indexing.manifest.delete_document("tenant-a", "space-a", "doc-a")
    with pytest.raises(StorageError):
        indexing.manifest.activate(result.generation_id)
    with pytest.raises(StorageError):
        indexing.index(result)
    assert indexing.manifest.active_generations("tenant-a", "space-a") == []


def test_stale_generation_is_not_activated(parsed, chunker, indexing):
    first = chunker.process(parsed("Текст."))
    second = chunker.process(parsed("Текст."))
    for result in (first, second):
        indexing.manifest.save(result)
        indexing.manifest.begin_index(result.generation_id)
    with pytest.raises(StorageError):
        indexing.manifest.activate(first.generation_id)
    indexing.manifest.activate(second.generation_id)


def test_checkpoint_immutable_and_version_checksum(parsed, chunker, indexing):
    result = chunker.process(parsed("Текст."))
    indexing.manifest.save(result)
    changed = copy.deepcopy(result)
    changed.warnings.append("changed")
    with pytest.raises(StorageError, match="неизменяем"):
        indexing.manifest.save(changed)
    with pytest.raises(StorageError, match="версии"):
        indexing.manifest.save(chunker.process(parsed("Другой текст.")))


def test_cleanup_failure_does_not_hide_new_generation(parsed, chunker, indexing):
    old = chunker.process(parsed("Текст."))
    indexing.index(old)
    new = chunker.process(parsed("Текст."))
    indexing.vectors.fail_cleanup = True
    response = indexing.index(new)
    assert response["cleanup_warning"] == "old_generation_cleanup_pending"
    assert {hit.payload["generation_id"] for hit in indexing.search("?", "tenant-a", "space-a")} == {new.generation_id}


def test_runtime_profile_mismatch(parsed, chunker, indexing):
    result = chunker.process(parsed("Текст."))
    result.metrics["embedding_profile_id"] = "other"
    with pytest.raises(ValidationError):
        indexing.index(result)


def test_outbox_replay_and_worker_ack(parsed, chunker, indexing):
    class Delivery:
        def __init__(self):
            self.published, self.acks = [], []
        def publish(self, job_id):
            self.published.append(job_id)
        def acknowledge(self, message_id):
            self.acks.append(message_id)
    delivery = Delivery()
    result = chunker.process(parsed("Текст."))
    indexing.manifest.save(result)
    publisher = OutboxPublisher(indexing.manifest, delivery)
    assert publisher.dispatch() == 1
    assert publisher.dispatch() == 0
    worker = IndexingWorker(indexing, delivery)
    worker.handle(result.generation_id, "message-1")
    worker.handle(result.generation_id, "message-2")
    assert delivery.acks == ["message-1", "message-2"]
    assert len(indexing.vectors.points) == 1


def test_worker_does_not_ack_transient_failure(parsed, chunker, indexing, embeddings):
    class Delivery:
        def acknowledge(self, message_id):
            raise AssertionError("Must not ACK failed transient work")
    result = chunker.process(parsed("Текст."))
    indexing.manifest.save(result)
    embeddings.fail = True
    with pytest.raises(EmbeddingError):
        IndexingWorker(indexing, Delivery()).handle(result.generation_id, "message")


def test_worker_acks_terminal_attempt_and_outbox_publication_replay(parsed, chunker, indexing, embeddings, monkeypatch):
    published, acks = [], []
    delivery = type("Delivery", (), {"publish": lambda self, key: published.append(key),
                                     "acknowledge": lambda self, key: acks.append(key)})()
    result = chunker.process(parsed("Текст."))
    indexing.manifest.save(result)
    mark = indexing.manifest.mark_published
    monkeypatch.setattr(indexing.manifest, "mark_published", lambda key: (_ for _ in ()).throw(StorageError("crash")))
    with pytest.raises(StorageError):
        OutboxPublisher(indexing.manifest, delivery).dispatch()
    monkeypatch.setattr(indexing.manifest, "mark_published", mark)
    OutboxPublisher(indexing.manifest, delivery).dispatch()
    assert published == [result.generation_id, result.generation_id]
    embeddings.fail = True
    worker = IndexingWorker(indexing, delivery)
    for attempt in range(3):
        with pytest.raises(EmbeddingError):
            worker.handle(result.generation_id, f"message-{attempt}")
    assert acks == ["message-2"]


def test_qdrant_http_wire_contract(parsed, chunker, counter, embeddings, profile, tmp_path):
    points, requests = {}, []
    def handler(request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request.method, request.url.path, body))
        if request.url.path.endswith("/points/query"):
            matches = []
            for point in points.values():
                if all((point["payload"].get(c["key"]) == c["match"]["value"] if "value" in c["match"]
                        else point["payload"].get(c["key"]) in c["match"]["any"]) for c in body["filter"]["must"]):
                    matches.append({**point, "score": 0.9})
            return httpx.Response(200, json={"result": {"points": matches[:body["limit"]]}})
        if request.url.path.endswith("/points"):
            if request.method == "PUT":
                points.update({point["id"]: point for point in body["points"]})
                return httpx.Response(200, json={"result": {"status": "completed"}})
            return httpx.Response(200, json={"result": [points[key] for key in body["ids"] if key in points]})
        return httpx.Response(200, json={"result": {"config": {"params": {
            "vectors": {"size": 4, "distance": "Cosine"}}}}})
    store = QdrantVectorStore("", "beta", 4, client=httpx.Client(
        base_url="http://qdrant/", transport=httpx.MockTransport(handler)))
    indexer = IndexingService(SQLiteManifest(tmp_path / "state.db"), store, embeddings, counter, profile)
    result = chunker.process(parsed("Текст документа."))
    indexer.index(result)
    hits = indexer.search("Вопрос", "tenant-a", "space-a")
    assert hits[0].payload["text"] == result.chunks[0].text
    query = requests[-1][2]
    assert query["with_payload"] is True
    assert {item["key"] for item in query["filter"]["must"]} == {
        "tenant_id", "space_id", "generation_id", "embedding_profile_id"}
