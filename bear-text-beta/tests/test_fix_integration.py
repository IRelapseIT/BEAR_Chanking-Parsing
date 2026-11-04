import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from bear_text_beta.api import create_app
from bear_text_beta.cli import main
from bear_text_beta.errors import ParseError, StorageError, ValidationError
from bear_text_beta.file_validation import FileValidator
from bear_text_beta.indexing import IndexingService, QdrantVectorStore
from bear_text_beta.models import PreparedFragment, SourceSpan
from bear_text_beta.object_storage import LocalObjectStorage
from bear_text_beta.parsing import HTMLParser, ParsingPipeline, TXTParser


def test_html_inline_breaks_preserve_word_fragments_and_code(context):
    raw = b'<p>Account<br>Bal<span>ance</span><!--omit--></p><pre>if x:<br>    return "a b"</pre>'
    doc = HTMLParser().parse(raw, context)
    assert [b.text for b in doc.blocks] == ['Account\nBalance', 'if x:\n    return "a b"']


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_bom_upload_and_default_parser_agree(tmp_path, context, encoding):
    raw = "Пример текста".encode(encoding)
    store = LocalObjectStorage(tmp_path / "objects")
    FileValidator().validate_and_store(io.BytesIO(raw), "document.txt", "input", store)
    pipeline = ParsingPipeline(tmp_path / "assets")
    try:
        assert isinstance(pipeline.parsers[".txt"], TXTParser)
        assert isinstance(pipeline.parsers[".html"], HTMLParser)
        assert pipeline.parse(raw, context).blocks[0].text == "Пример текста"
    finally:
        pipeline.parsers[".docx"].close()


def test_binary_control_still_rejected(tmp_path):
    with pytest.raises(ParseError, match="Бинарное"):
        FileValidator().validate_and_store(io.BytesIO(b"xx\0yy"), "a.txt", "input",
                                           LocalObjectStorage(tmp_path))


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_qdrant_ensure_transport_error_is_typed(failure):
    def respond(request):
        raise failure("deliberate outage", request=request)
    with httpx.Client(base_url="http://qdrant/", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(StorageError):
            QdrantVectorStore("http://qdrant", "test", client=client).ensure_collection()


def test_qdrant_creation_race_verifies_the_actual_schema():
    calls = []
    def respond(request):
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(404)
        if request.method == "PUT":
            return httpx.Response(409)
        return httpx.Response(200, json={"result": {"config": {"params": {
            "vectors": {"size": 384, "distance": "Cosine"}}}}})
    with httpx.Client(base_url="http://qdrant/", transport=httpx.MockTransport(respond)) as client:
        QdrantVectorStore("http://qdrant", "test", client=client).ensure_collection()
    assert calls == ["GET", "PUT", "GET"]


def test_http_readiness_does_not_claim_missing_model_is_ready():
    unavailable = SimpleNamespace(health=lambda: {"ready": False})
    runtime = SimpleNamespace(parser=SimpleNamespace(parsers={}), counter=object(),
                              chunker=object(), embeddings=unavailable, vectors=unavailable)
    with TestClient(create_app(runtime)) as client:
        assert client.get("/live").json() == {"status": "ok"}
        state = client.get("/health").json()
        assert state["model_ready"] is False
        assert client.get("/ready?capability=index").status_code == 503
        assert client.get("/ready?capability=parse").status_code == 200
        assert client.get("/ready?capability=chunk").status_code == 200


def test_index_boundary_requires_external_validator(parsed, counter, embeddings, profile):
    from bear_text_beta.chunking import ChunkingService
    class Tables:
        def process(self, block, context):
            return [PreparedFragment("A: 1", [SourceSpan(block.id, 0, 0)])]
        def validate(self, document, chunks, counter, maximum):
            if [c.text for c in chunks] != ["A: 1"]:
                raise ValidationError("Invalid table")
    service = ChunkingService(counter, embeddings, profile, {"TABLE": Tables()})
    result = service.process(parsed("| A |\n|---|\n| 1 |", "document.md"))
    writes = []
    manifest = SimpleNamespace(save=lambda value: writes.append(value))
    indexer = IndexingService(manifest, None, embeddings, counter, profile)
    with pytest.raises(ValidationError):
        indexer.index(result)
    assert not writes


def test_doctor_does_not_need_models_or_external_services(tmp_path, capsys):
    assert main(["--model-dir", str(tmp_path / "missing"), "doctor"]) == 0
    assert '"tokenizer_ready": false' in capsys.readouterr().out
    assert main(["--model-dir", str(tmp_path / "missing"), "doctor", "--require-model"]) == 1


def test_http_rejects_invalid_generation_before_initializing_services(monkeypatch):
    from bear_text_beta import chunker_api
    def unexpected_service():
        raise AssertionError("Invalid generation must not initialize model services")
    monkeypatch.setattr(chunker_api, "get_service", unexpected_service)
    with TestClient(chunker_api.app) as client:
        assert client.post("/chunk", json={"document": {}, "profile": {},
                                           "generation_id": "invalid"}).status_code == 422
    with TestClient(create_app()) as client:
        assert client.post("/beta/index", json={"generation_id": "invalid"}).status_code == 422
