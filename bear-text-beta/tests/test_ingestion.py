import io
import json
from dataclasses import asdict, replace

import pytest
from fastapi.testclient import TestClient
from test_indexing import FakeVectors

from bear_text_beta import runtime
from bear_text_beta.api import create_app
from bear_text_beta.errors import StorageError
from bear_text_beta.file_validation import StoredObject
from bear_text_beta.models import ProcessingProfile
from bear_text_beta.process_worker import run_once


@pytest.fixture
def system(tmp_path, monkeypatch, profile, counter, embeddings):
    monkeypatch.delenv("BEAR_CHUNKER_URL", raising=False)
    monkeypatch.setattr(runtime, "load_profile", lambda path: profile)
    monkeypatch.setattr(runtime, "HFTokenCounter", lambda path, p: counter)
    embeddings.max_tokens = 480
    monkeypatch.setattr(runtime, "InfinityEmbeddingProvider", lambda *a, **kw: embeddings)
    monkeypatch.setattr(runtime, "QdrantVectorStore", lambda *a, **kw: FakeVectors())
    services = runtime.Services(tmp_path, "unused")
    with TestClient(create_app(services)) as client:
        yield services, client
    services.close()


def submit(client, context, data=b"Refund within 12 days."):
    return client.post("/v1/documents", files={"file": (context.filename, data)},
                       data={"context": json.dumps(asdict(context))})


def test_upload_automatically_becomes_searchable_and_deduplicates(system, context):
    services, client = system
    response = submit(client, context)
    assert response.status_code == 202
    job = response.json()["job_id"]
    assert submit(client, context).json()["job_id"] == job
    assert client.get(f"/v1/jobs/{job}").json()["status"] == "queued"
    assert run_once(services)
    assert client.get(f"/v1/jobs/{job}").json()["status"] == "searchable"
    found = client.post("/v1/search", json={"query": "refund", "tenant_id": context.tenant_id,
                                          "space_id": context.space_id}).json()["results"]
    assert found[0]["document_id"] == context.document_id
    assert submit(client, context).json()["job_id"] == job
    assert not run_once(services)
    assert submit(client, context, b"different").status_code == 503
    assert client.post("/v1/search", json={"query": "refund", "tenant_id": "other",
                                          "space_id": context.space_id}).json()["results"] == []


@pytest.mark.parametrize("suffix", [".txt", ".md", ".html", ".htm", ".docx", ".pdf"])
def test_supported_formats_use_same_worker(system, context, suffix):
    services, client = system
    data = b"Refund within 12 days."
    if suffix in {".html", ".htm"}:
        data = b"<h1>Policy</h1><p>Refund within 12 days.</p>"
    elif suffix == ".md":
        data = b"# Policy\n\nRefund within 12 days."
    elif suffix == ".docx":
        from docx import Document
        doc = Document()
        doc.add_heading("Policy", 1)
        doc.add_paragraph("Refund within 12 days.")
        table = doc.add_table(rows=2, cols=2)
        for cell, text in zip([c for r in table.rows for c in r.cells], ["Year", "Days", "2025", "12"], strict=True):
            cell.text = text
        output = io.BytesIO()
        doc.save(output)
        data = output.getvalue()
    elif suffix == ".pdf":
        from reportlab.pdfgen import canvas
        output = io.BytesIO()
        doc = canvas.Canvas(output)
        doc.drawString(40, 700, "Refund within 12 days.")
        doc.save()
        data = output.getvalue()
    response = submit(client, replace(context, filename="document" + suffix), data)
    assert response.status_code == 202
    job = response.json()["job_id"]
    assert run_once(services)
    state = client.get(f"/v1/jobs/{job}").json()
    assert state["status"] == "searchable", state
    result = services.manifest.load(job)
    assert result.metrics["acceptance"]["accepted_for_indexing"]


def test_failed_update_preserves_old_and_retry_activates_new(system, context):
    services, client = system
    first = submit(client, context).json()["job_id"]
    run_once(services)
    second = submit(client, replace(context, document_version=2), b"Refund within 90 days.").json()["job_id"]
    services.vectors.fail_partial = True
    for _ in range(3):
        run_once(services)
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == [first]
    assert client.get(f"/v1/jobs/{second}").json()["status"] == "failed"
    services.vectors.fail_partial = False
    assert client.post(f"/v1/jobs/{second}/retry").status_code == 202
    run_once(services)
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == [second]
    deleted = client.delete(f"/v1/documents/{context.document_id}",
                            params={"tenant_id": context.tenant_id, "space_id": context.space_id})
    assert deleted.status_code == 200
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == []


def test_stale_lease_cannot_activate_auto_generation(system, context):
    services, client = system
    job_id = submit(client, context).json()["job_id"]
    job = services.manifest.claim_next_processing()
    result = services.process_stored(StoredObject(**job["stored"]), context, ProcessingProfile(),
        job_id, processing_claim=(job_id, job["claim_token"]))
    with services.manifest.connection() as db:
        db.execute("UPDATE processing_jobs SET lease_until=0 WHERE id=?", (job_id,))
    with pytest.raises(StorageError, match="lease"):
        services.indexer.index(result, processing_claim=(job_id, job["claim_token"]))
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == []


def test_docx_headers_lists_merges_long_values_and_embedded_object(system, context):
    from docx import Document
    from PIL import Image
    services, client = system
    doc = Document()
    doc.add_heading("Refund policy", 1)
    doc.add_paragraph("First condition", style="List Bullet")
    doc.sections[0].header.paragraphs[0].text = "Company header"
    doc.sections[0].footer.paragraphs[0].text = "Company footer"
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Refund conditions"
    table.cell(1, 0).text = "Nord"
    table.cell(1, 1).text = "Return within 12 days. " * 60 + "END-VALUE"
    table.cell(2, 0).text = "South"
    table.cell(2, 1).text = "Return within 90 days."
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(image, format="PNG")
    image.seek(0)
    doc.add_picture(image)
    output = io.BytesIO()
    doc.save(output)
    job = submit(client, replace(context, filename="document.docx"), output.getvalue()).json()["job_id"]
    run_once(services)
    state = client.get(f"/v1/jobs/{job}").json()
    assert state["status"] == "needs_review"
    result = services.manifest.load(job)
    assert any(b.role == "list_item" for b in result.parsed_document.blocks)
    assert any(b.text == "Company header" for b in result.parsed_document.blocks)
    assert any(b.text == "Company footer" for b in result.parsed_document.blocks)
    assert any("END-VALUE" in c.text for c in result.chunks)
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == []


def test_malformed_pdf_is_failed_without_damaging_other_document(system, context):
    services, client = system
    job = submit(client, replace(context, filename="bad.pdf"), b"not pdf").json()["job_id"]
    run_once(services)
    assert client.get(f"/v1/jobs/{job}").json()["status"] == "failed"


def test_new_submission_fences_old_queued_generation(system, context):
    services, client = system
    old = submit(client, context).json()["job_id"]
    new = submit(client, replace(context, document_version=2), b"New policy").json()["job_id"]
    for _ in range(5):
        run_once(services)
    assert services.manifest.active_generations(context.tenant_id, context.space_id) == [new]
    assert old != new


def test_successful_docling_fallback_cannot_erase_original_ocr_risk(parsed, chunker):
    from bear_text_beta.acceptance import assess
    document = parsed("SCAN POLICY: Refund within 27 days.")
    document.warnings.append("ocr_unverified:page:1; numeric_and_layout_review_required")
    document.blocks[0].metadata.update(parser="docling", pdf_route={"status": "heuristic_checks_passed"})
    result = chunker.process(document)
    assert assess(result)["accepted_for_indexing"] is False
    assert "ocr_requires_review" in assess(result)["reasons"]
