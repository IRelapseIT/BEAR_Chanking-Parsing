"""Evaluate the predeclared questions on one final fixed active index."""
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "verification/workflow/api-corpus-report.json"
report = json.loads(path.read_text(encoding="utf-8"))
with httpx.Client(timeout=60) as client:
    for case in report["cases"]:
        hits = client.post("http://127.0.0.1:8002/v1/search", json={"query": case["query"],
            "tenant_id": "workflow", "space_id": report["run"], "top_k": 5}).raise_for_status().json()["results"]
        case["rank_during_incremental_upload"] = case["rank_at_5"]
        case["rank_at_5"] = next((i + 1 for i, hit in enumerate(hits)
            if hit["document_id"] == case["document_id"] and case["answer"].casefold() in hit["text"].casefold()), 0)
        case["has_sources"] = any(h["sources"] for h in hits if h["document_id"] == case["document_id"])
        case["automatic_success"] = (case["state"]["status"] == "searchable"
                                      and case["rank_at_5"] > 0 and case["has_sources"])
n = len(report["cases"])
report.update(retrieval_protocol="fixed_final_active_index_after_all_submissions",
              automatic_successes=sum(c["automatic_success"] for c in report["cases"]),
              success_fraction=sum(c["automatic_success"] for c in report["cases"]) / n,
              recall_at_5_all_cases=sum(c["rank_at_5"] > 0 for c in report["cases"]) / n,
              mrr_at_5_all_cases=sum(1 / c["rank_at_5"] if c["rank_at_5"] else 0 for c in report["cases"]) / n)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("FIXED_INDEX_RETRIEVAL_EVALUATED")
