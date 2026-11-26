"""Run document ingestion, source inspection and retrieval against a running BEAR stand."""
import argparse
import hashlib
import io
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    body: bytes
    query: str
    expected: str
    pages: int
    kind: str = "generated"
    check: str = ""
    provenance: str = ""


def pdf(draw):
    stream = io.BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    draw(canvas)
    canvas.save()
    return stream.getvalue()


def scan(canvas, text, font, y=560):
    with Image.new("RGB", (1200, 220), "white") as image:
        ImageDraw.Draw(image).text((30, 60), text, font=ImageFont.truetype(font, 46), fill="black")
        canvas.drawImage(ImageReader(image), 37, y, width=510, height=94)


def merged_table(canvas):
    xs = (45, 180, 315, 450)
    for y in (580, 625, 670, 715):
        canvas.line(xs[0], y, xs[-1], y)
    for x in xs:
        canvas.line(x, 580, x, 715 if x in {xs[0], xs[-1]} else 670)
    canvas.drawString(55, 689, "Morava warehouse")
    for row, values in enumerate((("Depot", "Delivery days", "Unit"), ("Morava", "47", "calendar days"))):
        for col, value in enumerate(values):
            canvas.drawString(xs[col] + 8, 645 - row * 45, value)
    canvas.showPage()


def long_table(canvas):
    for page in range(3):
        for row in range(21):
            y = 751 - row * 31
            canvas.line(50, y, 470, y)
        for x in (50, 200, 340, 470):
            canvas.line(x, 131, x, 751)
        for col, text in enumerate(("Product", "Warehouse", "Quota kg")):
            canvas.drawString((60, 214, 350)[col], 730, text)
        for row in range(19):
            number = page * 19 + row + 1
            marker = "Last quota BEAR-9847" if number == 57 else f"Batch {number:03}"
            for col, text in enumerate((marker, "Borealis", str(500 + number))):
                canvas.drawString((60, 214, 350)[col], 699 - row * 31, text)
        canvas.showPage()


def matrix(font, public):
    def mixed(canvas):
        canvas.drawString(40, 730, "The policy annex contains a scanned warehouse code.")
        scan(canvas, "Warehouse trace code HUMMINGBIRD 761.", font)
        canvas.showPage()

    def columns(canvas):
        for row, (left, right) in enumerate(zip(
            ("Left forecast column", "The Boreal programme opens in October.", "Use an early reservation."),
            ("Right retention column", "Crosswind records are kept for 63 months.", "Archive the original copy."),
            strict=True,
        )):
            canvas.drawString(35, 750 - 22 * row, left)
            canvas.drawString(310, 750 - 22 * row, right)
        canvas.showPage()

    def rotated(canvas):
        canvas.setPageRotation(90)
        canvas.drawString(55, 520, "ROTATED INSPECTION. Relief valve pressure is 74 bar.")
        canvas.showPage()

    def isolated_error(canvas):
        canvas.drawString(40, 730, "Page one contains provision MOUNTAIN 440.")
        canvas.showPage()
        # The 216 dpi raster has more than 30M pixels. BEAR's existing preflight rejects
        # the render before allocating it. Native text on neighbouring pages remains real.
        canvas.setPageSize((1850, 1850))
        scan(canvas, "Oversize scanned page in the test corpus.", font)
        canvas.showPage()
        canvas.setPageSize((600, 800))
        canvas.drawString(40, 730, "Page three contains provision STONEBRIDGE 863.")
        canvas.showPage()

    def order_phrase(canvas):
        canvas.drawString(40, 730, "This guide explains what an order number means in the archive.")
        canvas.drawString(40, 702, "Training materials are kept for 58 months.")
        canvas.showPage()

    def searchable_scan(canvas):
        scan(canvas, "Scan policy. Refund within 27 calendar days.", font)
        canvas.showPage()

    def empty(canvas):
        canvas.showPage()

    dummy = ROOT / "verification/workflow/corpus/dummy.pdf"
    cases = [
        Case(dummy.name, dummy.read_bytes(), "Dummy PDF file", "Dummy", 1, "public",
             provenance="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"),
        Case("ocr.pdf", pdf(searchable_scan), "Refund time in the scanned policy?", "27 calendar days", 1, check="ocr"),
        Case("mixed.pdf", pdf(mixed), "What is the warehouse trace code?", "HUMMINGBIRD 761", 1),
        Case("columns.pdf", pdf(columns), "How long are Crosswind records kept?", "63 months", 1, check="columns"),
        Case("merged.pdf", pdf(merged_table), "Morava warehouse delivery time?", "47", 1, check="merged"),
        Case("long-table.pdf", pdf(long_table), "Last quota at Borealis?", "BEAR-9847", 3, check="long"),
        Case("rotated.pdf", pdf(rotated), "What is the valve pressure?", "74 bar", 1, check="rotated"),
        Case("isolated-page-error.pdf", pdf(isolated_error), "Which code applies on page three?", "STONEBRIDGE 863",
             3, check="isolated"),
        Case("native-order-phrase.pdf", pdf(order_phrase), "How long are training materials kept?", "58 months", 1,
             check="order"),
        Case("empty.pdf", pdf(empty), "", "", 1, check="empty"),
    ]
    if public:
        cases.append(Case("ocr-linn.pdf", (public / "ocr-linn.pdf").read_bytes(),
                          "What is the LinnSequencer?", "LinnSequencer", 1, "public", "ocr",
                          "https://raw.githubusercontent.com/ocrmypdf/OCRmyPDF/main/tests/resources/linn.pdf"))
        cases.append(Case("attention.pdf", (public / "attention.pdf").read_bytes(),
                          "What is the title of the Transformer paper?", "Attention Is All You Need", 11, "public",
                          provenance="https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"))
    return cases


def extracted_text_coverage(blocks, chunks):
    spans = defaultdict(set)
    for chunk in chunks:
        for span in chunk["source_spans"]:
            if span["start_char"] is not None and span["end_char"] is not None:
                spans[span["block_id"]].add((span["start_char"], span["end_char"]))
    expected, covered = 0, 0
    for block in blocks:
        if block["type"] != "TEXT":
            continue
        cursor, text = 0, block["text"]
        expected += len(text)
        for start, end in sorted(spans[block["id"]]):
            if 0 <= start <= end <= len(text):
                covered += max(0, end - max(start, cursor))
                cursor = max(cursor, end)
    return {"expected_text_characters": expected, "covered_text_characters": covered,
            "fraction": covered / expected if expected else None}


def wait_job(client, path, seconds):
    limit = time.monotonic() + seconds
    phases = []
    started = time.perf_counter()
    while time.monotonic() < limit:
        state = client.get(path).raise_for_status().json()
        phase = state.get("phase", state.get("status"))
        if not phases or phases[-1]["phase"] != phase:
            phases.append({"phase": phase, "observed_seconds": round(time.perf_counter() - started, 3)})
        if state.get("worker_status") in {"complete", "completed", "failed"}:
            return state, phases
        time.sleep(.45)
    raise TimeoutError(f"Job did not finish within {seconds}s")


def run_case(client, case, scope, timeout):
    started = time.perf_counter()
    ctx = {**scope, "document_id": case.name, "document_version": 1, "filename": case.name}
    job = client.post("/v1/documents", files={"file": (case.name, case.body, "application/pdf")},
                      data={"context": json.dumps(ctx)}).raise_for_status().json()
    path = job["status_url"]
    state, observations = wait_job(client, path, timeout)
    result = client.get(path + "/result").raise_for_status().json()
    original = client.get(path + "/artifacts/original", params=scope).raise_for_status()
    preview = client.get(f"{path}/page/{case.pages}", params={**scope, "rotation": 90})
    chunks, blocks = result["chunks"], result["parsed_document"]["blocks"]
    diagnostics = state.get("quality_warnings", [])
    warnings = result["parsed_document"]["warnings"]
    pages_with_text = sorted({block["source"]["page"] for block in blocks
                              if block["source"]["page"] is not None and block["text"].strip()})
    ready = state.get("search_available", False)
    rank, seconds = None, None
    if case.expected:
        search_started = time.perf_counter()
        search = client.post("/v1/search", json={**scope, "query": case.query, "top_k": 10})
        hits = search.raise_for_status().json()["results"]
        rank = next((pos for pos, hit in enumerate(hits, 1) if hit["document_id"] == case.name
                     and case.expected.casefold() in hit["text"].casefold() and hit["sources"]), None)
        seconds = round(time.perf_counter() - search_started, 3)
    coverage = extracted_text_coverage(blocks, chunks)
    checks = {"original_preserved": original.content == case.body,
              "rotated_page_preview": (preview.status_code == 200
                                       and int(preview.headers.get("X-Page-Count", 0)) == case.pages)}
    checks["available_expected"] = ready == bool(case.expected)
    checks["control_question"] = bool(rank) if case.expected else not chunks and state["outcome"] == "failed"
    blocks_text = " ".join(block["text"] for block in blocks)
    tables = [block for block in blocks if block["type"] == "TABLE"]
    if case.check == "ocr":
        checks["ocr_warning_retained"] = state.get("outcome") == "completed_with_warnings" and any(
            warn["severity"] == "warning" and not warn["resolved"] for warn in diagnostics)
    if case.check == "order":
        checks["no_business_field_gate"] = all(not warning.startswith("pdf_field_route:") for warning in warnings)
    if case.check == "isolated":
        checks["page_error_does_not_block"] = {1, 3}.issubset(pages_with_text) and any(
            warning.startswith("pdf_page_failed:2:") for warning in warnings)
    if case.check == "columns":
        checks["both_columns_preserved"] = all(text in blocks_text for text in ("Boreal programme", "Crosswind records"))
    if case.check == "merged":
        checks["merged_header"] = any(cell.get("column_span", 1) > 1 for table in tables
                                      for cell in table["metadata"].get("table_cells", []))
        checks["value_paired_with_row"] = any("47" in chunk["text"] and "Morava" in chunk["text"] for chunk in chunks)
    if case.check == "long":
        checks["all_rows_visible"] = all(f"Batch {row:03}" in blocks_text for row in range(1, 57))
        checks["last_row_chunked"] = any("BEAR-9847" in chunk["text"] for chunk in chunks)
    if case.check == "rotated":
        routes = [json.loads(value.split(":", 1)[1]) for value in warnings if value.startswith("pdf_page_route:")]
        checks["rotation_recorded"] = any(route.get("rotation") in (90, 270) for route in routes)
    return {"job_id": job["job_id"], "search_available": ready, "outcome": state.get("outcome"), "page_count": case.pages,
            "pages_with_text": pages_with_text, "pages_without_text": sorted(set(range(1, case.pages+1))
                                                                             - set(pages_with_text)),
            "parsed_blocks": len(blocks), "chunks": len(chunks), "tables": len(tables), "rank_at_10": rank,
            "parsed_text_coverage": coverage, "diagnostics": diagnostics, "raw_parser_warnings": warnings,
            "checks": checks, "passed": all(checks.values()), "observed_phases": observations,
            "elapsed_seconds": round(time.perf_counter() - started, 3), "query_seconds": seconds,
            "preview_http_status": preview.status_code,
            "measured_stage_metrics": {key: value for key, value in result["metrics"].items()
                                      if key.endswith(("_ms", "_seconds"))}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:8002")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--public-corpus", type=Path, help="Existing documented files attention.pdf and ocr-linn.pdf")
    cli.add_argument("--font", default=str(Path(os.getenv("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"))
    cli.add_argument("--job-timeout", type=int, default=240)
    cli.add_argument("--case", action="append", dest="cases", help="Run only the specified filename. Repeatable.")
    args = cli.parse_args()
    scope = {"tenant_id": "stand-verification", "space_id": uuid4().hex}
    selected = [case for case in matrix(args.font, args.public_corpus) if not args.cases or case.name in args.cases]
    if not selected:
        cli.error("No matching PDF case")
    dest = args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(base_url=args.url, trust_env=False, timeout=45) as client:
        ready = client.get("/ready?capability=search").raise_for_status().json()
        formats = client.get("/v1/formats").raise_for_status().json()
        for case in selected:
            result = {"file": case.name, "kind": case.kind, "provenance": case.provenance,
                      "sha256": hashlib.sha256(case.body).hexdigest(), "expected": case.expected, "query": case.query}
            try:
                result.update(run_case(client, case, scope, args.job_timeout))
            except (httpx.HTTPError, TimeoutError, AssertionError, ValueError, KeyError) as exc:
                result.update(passed=False, error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            print(case.name, result.get("outcome"), result.get("rank_at_10"),
                  result.get("checks", result.get("error")), flush=True)
            dest.write_text(json.dumps({"scope": scope, "readiness": ready, "profile": formats,
                                        "completed_cases": len(results), "total_cases": len(selected),
                                        "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(row["passed"] for row in results)
    print(f"PDF acceptance: {passed}/{len(results)} passed", flush=True)
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())

