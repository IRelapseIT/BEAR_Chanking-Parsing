import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from bear_text_beta.api import create_app
from bear_text_beta.cli import main
from bear_text_beta.embeddings import HFTokenCounter, load_profile
from bear_text_beta.file_validation import FileValidator
from bear_text_beta.models import ProcessingProfile
from bear_text_beta.object_storage import LocalObjectStorage
from bear_text_beta.storage import SQLiteManifest


def test_api_process_and_validation(tmp_path, parser, chunker, counter, embeddings, profile, context):
    manifest = SQLiteManifest(tmp_path / "state.db")
    storage = LocalObjectStorage(tmp_path / "objects")
    validator = FileValidator()
    def upload_stream(stream, ctx):
        return validator.validate_and_store(stream, ctx.filename, "originals/test", storage)
    def process_stored(stored, ctx, policy):
        with storage.get_stream(stored.object_key) as source:
            result = chunker.process(parser.parse(source.read(), ctx), policy)
        manifest.save(result)
        return result
    runtime = SimpleNamespace(upload_stream=upload_stream, process_stored=process_stored,
                              parser=parser, manifest=manifest)
    with TestClient(create_app(runtime)) as client:
        response = client.post("/beta/process", files={"file": ("document.txt", "Текст.".encode())},
                               data={"context": json.dumps(asdict(context))})
        assert response.status_code == 200
        body = response.json()
        assert body["schema_version"] == 1
        assert body["chunks"][0]["text"] == "Текст."
        assert manifest.load(body["generation_id"]).chunks[0].text == "Текст."
        malformed = client.post("/beta/process", files={"file": ("bad.txt", b"data")},
                                data={"context": "not JSON"})
        assert malformed.status_code == 422


def test_cli_process_export_and_compare(tmp_path, capsys):
    source = tmp_path / "input.txt"
    output = tmp_path / "result.json"
    source.write_text("Текст документа.", encoding="utf-8")
    args = ["--state-dir", str(tmp_path / "state"), "--test-counter", "process", str(source),
            "--document", "demo", "--output", str(output), "--benchmark"]
    assert main(args) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["chunks"][0]["text"] == "Текст документа."
    assert "python_peak_allocated_bytes" in result["metrics"]
    assert main(["compare", str(output), str(output)]) == 0
    assert "same_boundaries" in capsys.readouterr().out


def test_real_e5_tokenizer_preservation(tmp_path, parser, context):
    model_dir = Path(__file__).parents[1] / "models" / "e5"
    if not (model_dir / "bear-model.json").exists():
        pytest.skip("Optional cached tokenizer: run bear prepare-model")
    from bear_text_beta.chunking import ChunkingService, ChunkValidator
    from bear_text_beta.embeddings import UnavailableEmbeddingProvider
    profile = load_profile(model_dir)
    counter = HFTokenCounter(model_dir, profile)
    text = "  Т.е. проверка выполняется каждые 30 дней. Используйте LAN-1; значение 3.14.\n\n" * 80
    document = parser.parse(text.encode(), context)
    result = ChunkingService(counter, UnavailableEmbeddingProvider(), profile).process(
        document, ProcessingProfile(mode="token"))
    ChunkValidator(counter, 480).validate(result)
    assert sum(s.end_char - s.start_char for c in result.chunks for s in c.source_spans) == sum(
        len(block.text) for block in document.blocks)
    assert all(counter.full_count(c.embedding_text) <= 480 for c in result.chunks)


def test_api_index_search_health_and_errors(parsed, chunker, embeddings):
    from bear_text_beta.errors import EmbeddingError
    from bear_text_beta.models import SearchHit
    calls = []
    result = chunker.process(parsed("Ответ из payload."))
    class Indexer:
        def index(self, value):
            calls.append(value)
            return {"status": "indexed"}
        def retry(self, generation_id):
            return {"generation_id": generation_id}
        def search(self, query, tenant, space, top_k):
            if query == "failure":
                raise EmbeddingError("unavailable")
            return [SearchHit("point", 0.9, {"text": "Ответ из payload.", "source_spans": []})]
    runtime = SimpleNamespace(indexer=Indexer(), embeddings=embeddings,
                              vectors=SimpleNamespace(health=lambda: {"ready": True}))
    with TestClient(create_app(runtime)) as client:
        assert client.post("/beta/index", json={"result": asdict(result)}).json() == {"status": "indexed"}
        assert calls[0].chunks[0].text == "Ответ из payload."
        assert client.post("/beta/index", json={"generation_id": result.generation_id}).status_code == 200
        assert client.post("/beta/index", json={}).status_code == 422
        assert client.post("/beta/index", json={"result": {"invalid": True}}).status_code == 422
        query = {"query": "вопрос", "tenant_id": "tenant-a", "space_id": "space-a"}
        assert client.post("/beta/search", json=query).json()["results"][0]["text"] == "Ответ из payload."
        query["query"] = "failure"
        assert client.post("/beta/search", json=query).status_code == 503
        assert client.get("/health").json()["qdrant"]["ready"]


def test_real_tokenizer_with_semantic_adapter(parser, context):
    model_dir = Path(__file__).parents[1] / "models" / "e5"
    if not (model_dir / "bear-model.json").exists():
        pytest.skip("Optional cached tokenizer")
    from conftest import TestEmbeddings

    from bear_text_beta.chunking import ChunkingService
    profile = load_profile(model_dir)
    counter = HFTokenCounter(model_dir, profile)
    provider = TestEmbeddings(profile.vector_dimension)
    text = " ".join(f"Предложение {i} объясняет правила работы с документами компании." for i in range(40))
    result = ChunkingService(counter, provider, profile).process(parser.parse(text.encode(), context))
    assert provider.calls
    assert any(chunk.strategy == "semantic" for chunk in result.chunks)
    assert all(chunk.token_count <= 480 for chunk in result.chunks)
    assert "".join(chunk.text for chunk in result.chunks) == text
