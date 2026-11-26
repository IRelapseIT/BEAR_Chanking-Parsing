"""Opt-in workflow on a running local API + worker + Infinity + Qdrant."""

import io
import json
import os
from time import monotonic, sleep
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import pytest
from test_stand_formats import make_xlsx
from test_stand_images import make_ocr_image
from test_stand_office import make_doc, make_odt


@pytest.fixture(scope="module")
def live_client():
    url = os.getenv("BEAR_STAND_URL")
    if not url:
        pytest.skip("Set BEAR_STAND_URL to run against a local stand")
    address = urlparse(url)
    assert address.hostname in {"127.0.0.1", "::1", "localhost"} and address.scheme == "http"
    with httpx.Client(base_url=url, timeout=40.0, trust_env=False) as client:
        ready = client.get("/ready")
        assert ready.is_success and ready.json()["model_ready"], ready.text
        formats = client.get("/v1/formats").raise_for_status().json()
        assert formats["embedding_profile"]["model_id"] == "BAAI/bge-m3"
        yield client


@pytest.fixture(scope="module")
def live_scope():
    return {"tenant_id": "format-verification", "space_id": f"test-{uuid4().hex[:12]}"}


def format_fixture(extension):
    if extension == "csv":
        return b'Entity,Note\r\nMarket,"vector lighthouse\r\nsecond line"\r\n', "vector lighthouse"
    if extension == "tsv":
        return b'Entity\tNote\nMarket\t"silver orchard"\n', "silver orchard"
    if extension == "xlsx":
        return make_xlsx(), "Hidden datum"
    if extension == "xls":
        import xlwt

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("Warehouse notes")
        sheet.write_merge(0, 1, 0, 1, "copper ledger")
        sheet.write(20, 8, "violet month")
        output = io.BytesIO()
        workbook.save(output)
        return output.getvalue(), "violet month"
    if extension == "xml":
        return b'<policy region="north"><note>premium cabin access</note></policy>', "premium cabin access"
    if extension == "rtf":
        return br"{\rtf1\ansi Quiet forest policy.}", "Quiet forest policy"
    if extension == "odt":
        return make_odt("Field supervisor note."), "Field supervisor note"
    if extension == "doc":
        return make_doc("Emerald warehouse procedure."), "Emerald warehouse procedure"
    image_format = "PNG" if extension == "png" else "JPEG"
    phrase = {"png": "BEAR SCAN CHECK", "jpg": "ORCHARD RECORD", "jpeg": "CLEAR ROUTE MAP"}[extension]
    return make_ocr_image(image_format, phrase=phrase), phrase


@pytest.mark.parametrize("extension", ["csv", "tsv", "xlsx", "xls", "png", "jpg", "jpeg", "xml", "doc", "odt", "rtf"])
def test_real_format_full_workflow(live_client, live_scope, extension):
    data, phrase = format_fixture(extension)
    filename = "control." + extension
    context = {**live_scope, "document_id": extension, "document_version": 1, "filename": filename}
    started = monotonic()
    queued = live_client.post("/v1/documents", files={"file": (filename, data)},
                              data={"context": json.dumps(context)})
    assert queued.status_code == 202, queued.text
    status_url = queued.json()["status_url"]
    while True:
        state = live_client.get(status_url).raise_for_status().json()
        if state["worker_status"] in {"complete", "failed"} or monotonic() - started > 240:
            break
        sleep(0.5)
    assert state["outcome"] in {"completed", "completed_with_warnings"}, state
    assert state["search_available"], state
    result = live_client.get(status_url + "/result").raise_for_status().json()
    assert any(phrase in block["text"] for block in result["parsed_document"]["blocks"])
    assert any(phrase in chunk["text"] and chunk["source_spans"] for chunk in result["chunks"])
    hits = live_client.post("/v1/search", json={**live_scope, "query": phrase, "top_k": 12}).raise_for_status().json()
    assert any(hit["document_id"] == extension and phrase in hit["text"] and hit["sources"]
               for hit in hits["results"]), hits
    print(json.dumps({"extension": extension, "seconds": round(monotonic() - started, 3),
                      "chunks": len(result["chunks"]), "outcome": state["outcome"],
                      "job_id": queued.json()["job_id"], "scope": live_scope}, ensure_ascii=False))
