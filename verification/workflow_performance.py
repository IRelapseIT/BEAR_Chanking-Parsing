"""Bounded local load observation, not a scalability benchmark."""
import concurrent.futures
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8002"


def main():
    report = json.loads((ROOT / "verification/workflow/api-corpus-report.json").read_text(encoding="utf-8"))
    space = report["run"]
    unique = uuid4().hex[:6]
    queries = [{"query": "Срок возврата денег Норд. Проверка " + unique + str(i),
                "tenant_id": "workflow", "space_id": space} for i in range(4)]
    def search(query):
        start = time.perf_counter()
        with httpx.Client(timeout=60) as client:
            response = client.post(API + "/v1/search", json=query)
            response.raise_for_status()
        return time.perf_counter() - start
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        cold = list(pool.map(search, queries))
        warm = list(pool.map(search, queries))
    text = ("Long processing control: retention of archive backups is 43 days. " * 500).encode()
    context = {"tenant_id": "load", "space_id": space, "document_id": unique, "document_version": 1,
               "filename": "long.txt"}
    with httpx.Client(timeout=60) as client:
        job = client.post(API + "/v1/documents", files={"file": ("long.txt", text)},
                          data={"context": json.dumps(context)}).raise_for_status().json()["job_id"]
        for _ in range(500):
            state = client.get(API + "/v1/jobs/" + job).json()
            if state["status"] in {"chunking", "embedding", "indexing", "searchable"}:
                break
            time.sleep(.1)
        observed_phase = state["status"]
        start = time.perf_counter()
        client.get(API + "/live").raise_for_status()
        live_seconds = time.perf_counter() - start
        during = search({**queries[0], "query": queries[0]["query"] + " дополнительный запрос"})
        for _ in range(1200):
            state = client.get(API + "/v1/jobs/" + job).json()
            if state["worker_status"] in {"complete", "failed"}:
                break
            time.sleep(.2)
        assert state["status"] == "searchable", state
    output = {"parallelism": 4, "cold_search_seconds": cold, "cached_search_seconds": warm,
              "long_document_bytes": len(text), "job_status": state, "phase_at_probe": observed_phase,
              "live_during_processing_seconds": live_seconds, "uncached_search_during_processing_seconds": during,
              "scope": "One bounded CPU run. Query inference is serialized per API EmbeddingPipeline instance."}
    (ROOT / "verification/workflow/performance-report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BOUNDED_LOAD_VERIFIED")


if __name__ == "__main__":
    main()
