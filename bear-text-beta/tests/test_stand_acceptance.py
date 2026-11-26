from test_ingestion import submit, system  # noqa: F401

from bear_text_beta.acceptance import assess
from bear_text_beta.process_worker import run_once


def test_nonempty_ocr_is_indexable(parsed, chunker):
    document = parsed("There are 36 days for a refund.")
    document.blocks[0].metadata.update(extraction="ocr", engine="tesseract")
    decision = assess(chunker.process(document))
    assert decision["accepted_for_indexing"]
    assert decision["outcome"] == "completed_with_warnings"
    assert decision["warnings"][0]["code"] == "ocr_requires_review"


def test_selected_fallback_resolves_historical_diagnostic(parsed, chunker):
    document = parsed("Restored text from the first page.")
    document.warnings += [
        "pdf_page_failed:page:1",
        'pdf_page_route:{"page":1,"selected_parser":"docling","fallback_status":"selected",'
        '"primary_reasons":["pdf_page_failed"],"remaining_reasons":[],"status":"heuristic_checks_passed"}',
    ]
    decision = assess(chunker.process(document))
    historical = next(w for w in decision["warnings"] if w["code"] == "pdf_page_failed")
    assert historical["resolved"] and historical["resolution"] == "docling"
    assert decision["outcome"] == "completed"


def test_index_failure_has_separate_status_and_preserves_result(system, context):
    services, client = system
    services.vectors.fail_partial = True
    job = submit(client, context).json()["job_id"]
    for _ in range(3):
        run_once(services)
    state = client.get(f"/v1/jobs/{job}").json()
    assert state["outcome"] == "completed"
    assert state["index"]["status"] == "failed"
    assert not state["search_available"]
    assert client.get(f"/v1/jobs/{job}/result").json()["chunks"]
    services.vectors.fail_partial = False
    assert client.post(f"/v1/jobs/{job}/retry").status_code == 202
    run_once(services)
    assert client.get(f"/v1/jobs/{job}").json()["search_available"]


def test_empty_document_outcome_failed_with_checkpoint(system, context):
    services, client = system
    job = submit(client, context, b" \n").json()["job_id"]
    run_once(services)
    assert client.get(f"/v1/jobs/{job}").json()["outcome"] == "failed"
    assert client.get(f"/v1/jobs/{job}/result").json()["chunks"] == []
