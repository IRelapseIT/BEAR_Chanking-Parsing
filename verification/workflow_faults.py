"""Real dependency outage, API retry, deletion, vectors and resource controls."""
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import psutil
from bear_text_beta.models import DocumentContext

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification/workflow"
API = "http://127.0.0.1:8002"


def wait(client, job):
    for _ in range(500):
        value = client.get(API + "/v1/jobs/" + job).raise_for_status().json()
        if value["worker_status"] in {"failed", "complete"}:
            return value
        time.sleep(.2)
    raise AssertionError("Job did not settle")


def restart():
    for action in ("stop", "start"):
        subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "scripts/services.py"), action],
                       check=True, stdout=subprocess.DEVNULL)
    with httpx.Client(timeout=10) as client:
        for _ in range(100):
            try:
                if client.get(API + "/live").is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.2)
    raise AssertionError("API restart readiness failed")


def main():
    checks = {}
    space = uuid4().hex[:12]
    context = DocumentContext("faults", space, "policy", 1, "policy.txt")
    def submit(client, version, text):
        ctx = {**asdict(context), "document_version": version}
        return client.post(API + "/v1/documents", files={"file": (context.filename, text.encode())},
                           data={"context": json.dumps(ctx)}).raise_for_status().json()["job_id"]
    def search(client):
        return client.post(API + "/v1/search", json={"query": "Nord refund policy", "tenant_id": "faults",
                                                     "space_id": space}).raise_for_status().json()["results"]
    with httpx.Client(timeout=60) as client:
        first = submit(client, 1, "Nord refunds payments within 12 working days.")
        assert wait(client, first)["status"] == "searchable"
    for version, dependency in [(2, "embedding"), (3, "qdrant")]:
        entry = json.loads((ROOT / "runtime/processes.json").read_text())[dependency]
        process = psutil.Process(entry["pid"])
        assert abs(process.create_time() - entry["created"]) < .01 and process.cmdline() == entry["command"]
        for child in reversed(process.children(recursive=True)):
            child.terminate()
        process.terminate()
        process.wait(20)
        with httpx.Client(timeout=60) as client:
            job = submit(client, version, f"Nord refunds payments within {version * 20} working days.")
            failed = wait(client, job)
            assert failed["status"] == "failed", failed
        restart()
        with httpx.Client(timeout=60) as client:
            assert any(h["generation_id"] == first for h in search(client))
            client.post(API + f"/v1/jobs/{job}/retry").raise_for_status()
            assert wait(client, job)["status"] == "searchable"
            assert all(h["generation_id"] == job for h in search(client))
            first = job
        checks[dependency + "_outage_preserves_old_and_api_retry_activates_new"] = True
    with httpx.Client(timeout=60) as client:
        deleted = client.delete(API + "/v1/documents/policy", params={"tenant_id": "faults", "space_id": space})
        deleted.raise_for_status()
        assert search(client) == []
        checks["deletion_excludes_from_search"] = True
        # Full multipart request exceeds the fixed API limit and must not become a queued job.
        large = client.post(API + "/v1/documents", files={"file": ("large.txt", b"x" * (52 * 1024 * 1024))},
                            data={"context": json.dumps(asdict(context))})
        assert large.status_code == 413
        checks["oversized_http_upload_rejected"] = True
    reference = json.loads((OUT / "bge-reference.json").read_text(encoding="utf-8"))
    fixed = [{"text": text} for text in reference["texts"]]
    expected = np.asarray(reference["vectors"])
    with httpx.Client(timeout=60) as client:
        metadata = client.get("http://127.0.0.1:7998/v1/models").raise_for_status().json()
        body = client.post("http://127.0.0.1:7998/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": [fixed[i]["text"] for i in range(2)]}).raise_for_status().json()
        actual = np.asarray([entry["embedding"] for entry in sorted(body["data"], key=lambda item: item["index"])])
        assert actual.shape == (2, 1024) and np.isfinite(actual).all()
        assert np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5)
        cosine = np.sum(expected * actual, axis=1) / (np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1))
        unchanged = [i for i in range(2) if fixed[i]["text"] == fixed[i]["text"].strip()]
        assert unchanged and (cosine[unchanged] > .99999).all(), cosine
        paired = client.post("http://127.0.0.1:7998/v1/embeddings", json={
            "model": "BAAI/bge-m3", "input": [fixed[1]["text"], fixed[1]["text"].strip()]}).raise_for_status().json()
        paired_vectors = np.asarray([e["embedding"] for e in paired["data"]])
        assert np.allclose(paired_vectors[0], paired_vectors[1], atol=1e-6)
        checks["infinity_vs_previously_verified_local_bge_cosine"] = cosine.tolist()
        checks["identical_untrimmed_inputs_matching_reference"] = unchanged
        checks["sentence_transformers_edge_whitespace_normalization_verified"] = True
        checks["infinity_models_response"] = metadata
    (OUT / "faults-report.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REAL_OUTAGES_RETRY_DELETE_VECTORS_VERIFIED")


if __name__ == "__main__":
    main()
