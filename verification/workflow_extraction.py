"""Independent known-value checks against actual HTTP checkpoints, no re-upload."""
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification/workflow"
report = json.loads((OUT / "api-corpus-report.json").read_text(encoding="utf-8"))
expected_tables = {
    "table.md": ["Year", "Profit", "2025", "150 million RUB", "2026", "180 million RUB"],
    "table.htm": ["Office", "Hours", "East", "09:30"],
    "delivery.docx": ["Delivery regions", "Nord", "7 working days", "South", "18 working days"],
}
checks = []
with httpx.Client(timeout=30) as client:
    for case in report["cases"]:
        if case["state"]["status"] == "failed":
            continue
        result = client.get("http://127.0.0.1:8002/v1/jobs/" + case["job_id"] + "/result").raise_for_status().json()
        blocks = result["parsed_document"]["blocks"]
        values = [cell["text"] for block in blocks for cell in block["metadata"].get("table_cells", [])]
        expected = expected_tables.get(case["file"], [])
        assert all(v in values for v in expected), (case["file"], expected, values)
        if case["file"] == "text.pdf":
            assert any(b["source"]["page"] == 1 and b["source"]["bbox"] for b in blocks)
        if case["file"] == "ssl.md":
            assert any(b["role"] == "code" and "ssl_certificate /etc/ssl/bear.pem;" in b["text"] for b in blocks)
        assert all(c["source_spans"] for c in result["chunks"])
        checks.append({"file": case["file"], "known_table_values_checked": len(expected),
                       "expected_answer_in_extracted_blocks": any(case["answer"].casefold() in b["text"].casefold() for b in blocks),
                       "source_spans_present": True, "coverage": result["completeness"]})
(OUT / "extraction-report.json").write_text(json.dumps({"checks": checks,
    "scope": "Known fragments/cells/code/page coordinates only. No global whole-source or arbitrary reading-order oracle."},
    ensure_ascii=False, indent=2), encoding="utf-8")
print("EXTRACTION_TABLES_SOURCES_VERIFIED")
