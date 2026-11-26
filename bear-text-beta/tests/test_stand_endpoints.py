from test_ingestion import submit, system  # noqa: F401

from bear_text_beta.process_worker import run_once


def test_full_stand_exports_scope_history_and_run_deletion(system, context):
    services, client = system
    job = submit(client, context).json()["job_id"]
    run_once(services)
    scope = {"tenant_id": context.tenant_id, "space_id": context.space_id}
    assert client.get("/v1/formats").json()["extensions"]
    assert client.get("/v1/jobs", params=scope).json()["jobs"][0]["id"] == job
    for artifact in ("original", "parsed_document.json", "document.md", "document.txt", "chunks.jsonl",
                     "warnings.json", "configuration.json", "report.md"):
        assert client.get(f"/v1/jobs/{job}/artifacts/{artifact}", params=scope).status_code == 200
    request = {**scope, "query": "refund", "expected_fragment": "Refund", "mode": "dense"}
    found = client.post("/v1/search/trace", json=request)
    assert found.status_code == 200, found.text
    assert found.json()["expectation"]["matched"]
    assert client.get("/v1/search/history", params=scope).json()["history"]
    assert client.get(f"/v1/jobs/{job}/compare/{job}", params=scope).json()["same_boundaries"]
    assert client.get(f"/v1/jobs/{job}/page/1", params=scope).status_code == 415
    assert client.get(f"/v1/jobs/{job}", params={**scope, "space_id": "other"}).status_code == 404
    assert client.delete(f"/v1/jobs/{job}/index", params=scope).status_code == 200
    assert not client.get(f"/v1/jobs/{job}").json()["search_available"]
    assert client.post(f"/v1/jobs/{job}/retry", params=scope).status_code == 202
    run_once(services)
    assert client.get(f"/v1/jobs/{job}").json()["search_available"]
    assert client.delete(f"/v1/jobs/{job}", params=scope).status_code == 200
    assert client.get("/v1/jobs", params=scope).json()["jobs"] == []
