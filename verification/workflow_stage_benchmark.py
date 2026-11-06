"""Separate two-document model/storage microbenchmark; not end-to-end stage tracing."""
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
name = "workflow_stage_" + uuid4().hex
ids = [str(uuid4()), str(uuid4())]
output = {"documents": 2, "batch_size": 2, "scope": "Two short documents, real model and own temporary collection"}
with httpx.Client(timeout=60) as client:
    start = time.perf_counter()
    vectors = client.post("http://127.0.0.1:7998/v1/embeddings", json={"model": "BAAI/bge-m3",
        "input": ["Refunds take twelve working days.", "Архив хранится сорок три дня."]}).raise_for_status().json()["data"]
    output["embedding_seconds"] = time.perf_counter() - start
    base = "http://127.0.0.1:6335/collections/" + name
    client.put(base, json={"vectors": {"size": 1024, "distance": "Cosine"}}).raise_for_status()
    try:
        start = time.perf_counter()
        client.put(base + "/points", params={"wait": "true"}, json={"points": [
            {"id": ids[i], "vector": vector["embedding"], "payload": {"scope": "own-microbenchmark"}}
            for i, vector in enumerate(sorted(vectors, key=lambda v: v["index"]))]}).raise_for_status()
        found = client.post(base + "/points", json={"ids": ids, "with_payload": True}).raise_for_status().json()
        assert len(found["result"]) == 2
        output["qdrant_upsert_and_verification_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        client.post(base + "/points/query", json={"query": vectors[0]["embedding"], "limit": 2}).raise_for_status()
        output["qdrant_query_seconds"] = time.perf_counter() - start
    finally:
        client.delete(base).raise_for_status()
(ROOT / "verification/workflow/stage-benchmark.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print("MODEL_STORAGE_TIMINGS_MEASURED")
