import io
from dataclasses import replace

import pytest
from bear_fusion.markdown_adapter import StructuredMarkdownParser
from docx import Document
from docx.oxml import OxmlElement
from fastapi.testclient import TestClient
from test_indexing import FakeVectors
from test_ingestion import submit, system  # noqa: F401 - imported pytest fixture

from bear_text_beta.acceptance import assess
from bear_text_beta.api import create_app
from bear_text_beta.errors import ValidationError
from bear_text_beta.indexing import IndexingService
from bear_text_beta.parsing import MDParser
from bear_text_beta.process_worker import run_once
from bear_text_beta.resources import is_status_badge
from bear_text_beta.storage import SQLiteManifest
from bear_text_beta.unified_parsing import DOCXParser


@pytest.mark.parametrize("path", ["/beta/process", "/beta/index", "/beta/search", "/beta/jobs/unknown"])
def test_legacy_paths_cannot_initialize_runtime(path):
    with TestClient(create_app()) as client:
        assert client.post(path, json={"result": {"metrics": {}}}).status_code == 410
        assert all(not p.startswith("/beta/") for p in client.get("/openapi.json").json()["paths"])


@pytest.mark.parametrize("decision", [None, {"accepted_for_indexing": True}])
def test_internal_indexer_reassesses_unresolved_content(
    tmp_path, chunker, counter, embeddings, profile, context, decision
):
    document = MDParser().parse(b"Text\n\n![chart](https://example.invalid/chart.png)", context)
    result = chunker.process(document)
    if decision is not None:
        result.metrics["acceptance"] = decision
    indexer = IndexingService(SQLiteManifest(tmp_path / "m.db"), FakeVectors(), embeddings, counter, profile)
    with pytest.raises(ValidationError, match="приёмки"):
        indexer.index(result)


def test_status_badges_are_optional_but_diagrams_are_not(chunker, context):
    badge = b"TypeScript docs.\n\n![CI](https://github.com/microsoft/TypeScript/actions/workflows/ci.yml/badge.svg)"
    result = chunker.process(StructuredMarkdownParser().parse(badge, context))
    decision = assess(result)
    assert result.completeness == "partial"  # Do not relabel missing remote values as fully extracted.
    assert decision["accepted_for_indexing"] and decision["optional_references"]
    assert decision["optional_references"][0]["values_verified"] is False
    assert decision["limitations"]
    result = chunker.process(
        StructuredMarkdownParser().parse(badge + b"\n\n![Plan](https://example.org/diagram.png)", context)
    )
    assert not assess(result)["accepted_for_indexing"]
    assert not is_status_badge("https://github.com.evil.test/a/b/actions/workflows/test.yml/badge.svg")
    assert not is_status_badge("https://github.com/a/b/diagram.svg")


def test_docx_content_control_preserves_text_and_table_order(context):
    document = Document()
    document.add_paragraph("Before")
    table = document.add_table(rows=2, cols=1)
    table.cell(0, 0).text = "Label"
    table.cell(1, 0).text = "42"
    # Wrap the table in a standard block-level content control.
    body = document.element.body
    body.remove(table._tbl)
    wrapper = OxmlElement("w:sdt")
    content = OxmlElement("w:sdtContent")
    paragraph = OxmlElement("w:p")
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "CONTROL TEXT"
    run.append(text)
    paragraph.append(run)
    content.append(paragraph)
    content.append(table._tbl)
    wrapper.append(content)
    body.insert(1, wrapper)
    document.add_paragraph("After")
    buffer = io.BytesIO()
    document.save(buffer)
    parsed = DOCXParser().parse(buffer.getvalue(), replace(context, filename="control.docx"))
    assert [b.role for b in parsed.blocks] == ["paragraph", "paragraph", "table", "paragraph"]
    assert parsed.blocks[1].text == "CONTROL TEXT"
    assert "42" in parsed.blocks[2].text
    assert "/sdtContent/" in parsed.blocks[1].source.element_path


def test_unknown_docx_container_is_flagged(context):
    document = Document()
    document.add_paragraph("Known")
    unknown = OxmlElement("w:smartTag")
    p = OxmlElement("w:p")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "Unknown"
    r.append(t)
    p.append(r)
    unknown.append(p)
    document.element.body.insert(1, unknown)
    buffer = io.BytesIO()
    document.save(buffer)
    parsed = DOCXParser().parse(buffer.getvalue(), replace(context, filename="unknown.docx"))
    assert any(w.startswith("docx_unsupported_body_element") for w in parsed.warnings)


def test_reprocess_uses_original_and_new_parser_instead_of_old_checkpoint(request, context):
    services, client = request.getfixturevalue("system")
    services.parser.parsers[".md"] = MDParser()
    context = replace(context, filename="readme.md")
    data = b"TypeScript\n\n![CI](https://github.com/microsoft/TypeScript/actions/workflows/ci.yml/badge.svg)"
    old = submit(client, context, data).json()["job_id"]
    run_once(services)
    assert client.get(f"/v1/jobs/{old}").json()["status"] == "needs_review"
    old_result = services.manifest.load(old)
    services.parser.parsers[".md"] = StructuredMarkdownParser()
    response = client.post(f"/v1/jobs/{old}/reprocess")
    assert response.status_code == 202
    new = response.json()["job_id"]
    assert new != old
    run_once(services)
    assert client.get(f"/v1/jobs/{new}").json()["status"] == "searchable"
    assert (
        services.manifest.load(old).parsed_document.parser_profile
        == old_result.parsed_document.parser_profile
    )
    assert (
        services.manifest.load(new).metrics["original_object_key"]
        == old_result.metrics["original_object_key"]
    )


def test_docx_content_control_in_cell_preserves_value(context):
    document = Document()
    table = document.add_table(rows=2, cols=1)
    table.cell(0, 0).text = "Header"
    table.cell(1, 0).text = "CELL-CONTROL-42"
    cell = table.cell(1, 0)._tc
    paragraph = cell.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
    cell.remove(paragraph)
    control = OxmlElement("w:sdt")
    content = OxmlElement("w:sdtContent")
    content.append(paragraph)
    control.append(content)
    cell.append(control)
    marker = OxmlElement("w:bookmarkEnd")
    cell.append(marker)
    buffer = io.BytesIO()
    document.save(buffer)
    parsed = DOCXParser().parse(buffer.getvalue(), replace(context, filename="cell.docx"))
    assert "CELL-CONTROL-42" in parsed.blocks[0].text
