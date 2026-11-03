import io
import json
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from docx import Document
from fastapi.testclient import TestClient

from bear_text_beta.api import create_app
from bear_text_beta.chunking import AdaptiveChunkingStrategy, Region
from bear_text_beta.errors import ParseError, ServiceUnavailableError
from bear_text_beta.file_validation import FileLimits, FileValidator, StoredObject
from bear_text_beta.models import DocumentContext, ProcessingProfile
from bear_text_beta.object_storage import LocalObjectStorage
from bear_text_beta.parsing import ParsingPipeline
from bear_text_beta.runtime import Services
from bear_text_beta.storage import SQLiteManifest


def test_file_validator_hashes_during_storage_read(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    source = io.BytesIO(b"hello")
    stored = FileValidator(FileLimits(max_file_bytes=5)).validate_and_store(
        source, "a.txt", "originals/one", storage)
    assert stored.size_bytes == 5
    assert stored.checksum == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert storage.get_stream(stored.object_key).read() == b"hello"
    with pytest.raises(ParseError):
        FileValidator(FileLimits(max_file_bytes=4)).validate_and_store(
            io.BytesIO(b"hello"), "a.txt", "originals/oversize", storage)
    assert not storage._path("originals/oversize").exists()


def test_docx_attachment_and_zip_limits():
    document = Document()
    document.add_paragraph("Text")
    output = io.BytesIO()
    document.save(output)
    FileValidator().validate_docx(output.getvalue())
    with pytest.raises(ParseError):
        FileValidator(FileLimits(max_docx_entries=1)).validate_docx(output.getvalue())


def test_dedoc_is_default_for_txt_html_docx_and_md_is_local(tmp_path):
    parser = ParsingPipeline(tmp_path)
    assert parser.parsers[".txt"] is parser.parsers[".docx"]
    assert parser.parsers[".html"] is parser.parsers[".docx"]
    assert parser.parsers[".md"] is not parser.parsers[".docx"]
    payload = {"schema_version": 1, "blocks": [{"type": "TEXT", "role": "paragraph",
               "text": "Привет", "node_id": "0.0", "line_id": 0, "page": 1}],
               "attachments": [], "warnings": []}
    parser.parsers[".docx"].client = httpx.Client(
        base_url="http://dedoc", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)))
    result = parser.parse("Привет".encode(), DocumentContext("t", "s", "d", 1, "a.txt"))
    assert result.blocks[0].text == "Привет"
    assert result.parser_profile == "dedoc-2.8-v1"


def test_dedoc_service_error_is_retryable(tmp_path):
    parser = ParsingPipeline(tmp_path)
    parser.parsers[".docx"].client = httpx.Client(
        base_url="http://dedoc", transport=httpx.MockTransport(
            lambda request: httpx.Response(503)))
    with pytest.raises(ServiceUnavailableError):
        parser.parse(b"hello", DocumentContext("t", "s", "d", 1, "a.txt"))


def test_router_does_not_treat_section_path_alone_as_structure():
    region = Region("Long text. " * 50, [(0, 550, object())], ["Section"], False, "paragraph")
    decision = AdaptiveChunkingStrategy.choose(region, ProcessingProfile(), lambda _: False, len)
    assert decision.strategy == "semantic"
    structured = Region(region.text, region.ranges, region.section_path, True, region.role)
    assert AdaptiveChunkingStrategy.choose(structured, ProcessingProfile(), lambda _: False, len).strategy == (
        "structural")


def test_processing_jobs_reclaim_and_complete(tmp_path):
    manifest = SQLiteManifest(tmp_path / "manifest.db")
    job_id = str(uuid4())
    stored = StoredObject("originals/one", 12, "a" * 64)
    context = DocumentContext("t", "s", "d", 1, "a.txt")
    manifest.submit_processing(job_id, stored, context, ProcessingProfile())
    claimed = manifest.claim_next_processing()
    assert claimed["id"] == job_id
    assert claimed["stored"] == asdict(stored)
    manifest.complete_processing(job_id)
    assert manifest.processing_status(job_id)["status"] == "complete"
    assert manifest.claim_next_processing() is None


def test_large_upload_is_queued(tmp_path):
    services = Services(tmp_path / "state", tmp_path / "unused-model", test_counter=True)
    try:
        with TestClient(create_app(services)) as client:
            context = DocumentContext("t", "s", "big", 1, "big.txt")
            response = client.post("/beta/process", files={"file": ("big.txt", b"x" * (8 * 1024 * 1024 + 1))},
                                   data={"context": json.dumps(asdict(context))})
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            assert client.get(f"/beta/jobs/{job_id}").json()["status"] == "pending"
    finally:
        services.close()


def test_api_rejects_oversized_body_before_parsing():
    with TestClient(create_app()) as client:
        response = client.post("/beta/process", content=b"x", headers={
            "content-length": str(52 * 1024 * 1024), "content-type": "application/octet-stream"})
        assert response.status_code == 413


def test_dedoc_tree_to_contract_preserves_table_order(tmp_path):
    from dedoc_service import _dedoc_to_contract

    def node(identifier, text, line_id, annotations=None, children=None, line_type="raw_text"):
        return SimpleNamespace(node_id=identifier, text=text, annotations=annotations or [],
            metadata=SimpleNamespace(page_id=0, line_id=line_id,
                hierarchy_level=SimpleNamespace(line_type=line_type, level_2=1)),
            subparagraphs=children or [])

    table_schema = SimpleNamespace(model_dump=lambda mode: {"cells": [["value"]]})
    table = SimpleNamespace(metadata=SimpleNamespace(uid="u1", page_id=0),
                            to_api_schema=lambda: table_schema)
    before = node("0.0", "Before", 0, [SimpleNamespace(name="table", value="u1")])
    after = node("0.1", "After", 1)
    root = node("0", "", 0, children=[before, after], line_type="root")
    parsed = SimpleNamespace(content=SimpleNamespace(structure=root, tables=[table], warnings=[]),
                             attachments=[])
    result = _dedoc_to_contract(parsed, tmp_path, FileLimits())
    assert [block["type"] for block in result["blocks"]] == ["TEXT", "TABLE", "TEXT"]
    assert result["blocks"][1]["metadata"]["dedoc_uid"] == "u1"
