"""Measured two-page PDF fixture with ground-truth borderless table values."""
import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import DoclingProcess, FallbackPolicy
from bear_fusion.tables import TableProcessor, validate_table_chunks
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import (
    HFTokenCounter,
    UnavailableEmbeddingProvider,
    load_profile,
)
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext, ProcessingProfile
from reportlab.pdfgen import canvas


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "borderless.pdf"
    writer = canvas.Canvas(str(source))
    writer.drawString(40, 740, "Ordinary first page remains unchanged.")
    writer.showPage()
    writer.drawString(40, 740, "Table 1 Revenue report")
    expected = [["Year", "Revenue", "Cost"], ["2024", "150", "80"], ["2025", "180", "90"],
                ["2026", "200", "95"]]
    for row, values in enumerate(expected):
        for col, value in enumerate(values):
            writer.drawString(70 + col * 170, 680 - row * 30, value)
    if args.ruled_control:
        for x in (60, 230, 400, 540):
            writer.line(x, 575, x, 700)
        for y in (700, 670, 640, 610, 575):
            writer.line(60, y, 540, y)
    writer.save()
    context = DocumentContext("pdf-smoke", "local", "borderless", 1, source.name)
    if args.ruled_control:
        # Explicit positive fault-control: native table validation rejects the grid.
        # The real Docling parser still receives the original ruled PDF.
        from bear_fusion import pdf_adapter
        pdf_adapter.accepted_table = lambda table, page: None
    start = time.perf_counter()
    baseline = parse_pdf(source, context, FileLimits(), policy=FallbackPolicy(enabled=False))
    primary_seconds = time.perf_counter() - start
    backend = DoclingProcess(str(args.python), args.artifacts, FallbackPolicy())
    start = time.perf_counter()
    routed = parse_pdf(source, context, FileLimits(), fallback=backend)
    routed_seconds = time.perf_counter() - start
    embedding_profile = load_profile(args.tokenizer_dir)
    counter = HFTokenCounter(args.tokenizer_dir, embedding_profile)
    processor = TableProcessor(counter)
    processor.budget = 320
    chunked = ChunkingService(counter, UnavailableEmbeddingProvider(), embedding_profile,
                              {"TABLE": processor}).process(routed, ProcessingProfile(mode="token"))
    validate_table_chunks(routed, chunked.chunks, counter, 320)
    def records(document):
        return [json.loads(w.split(":", 1)[1]) for w in document.warnings if w.startswith("pdf_page_route:")]
    routes = records(routed)
    assert routes[0]["fallback_attempted"] is False
    assert routes[1]["fallback_attempted"] is True
    assert [b.text for b in baseline.blocks if b.source.page == 1] == [
        b.text for b in routed.blocks if b.source.page == 1]
    expected_values = [value for row in expected for value in row]
    table_values = [cell["text"] for b in routed.blocks if b.source.page == 2 and b.type == "TABLE"
                    and not b.metadata.get("unresolved") for cell in b.metadata.get("table_cells", [])]
    report = {"fixture": "generated two-page PDF, 4x3 table on page 2",
              "ruled_control_with_injected_primary_rejection": args.ruled_control,
              "expected_table": expected, "expected_values_found_in_table": sum(
                  (Counter(expected_values) & Counter(table_values)).values()),
              "expected_values": len(expected_values), "primary_seconds": primary_seconds,
              "routed_seconds": routed_seconds, "added_seconds": routed_seconds - primary_seconds,
              "validated_chunks": len(chunked.chunks), "chunking_completeness": chunked.completeness,
              "routes": routes, "result": asdict(routed),
              "scope": "One synthetic fixture; not a representative document-quality benchmark"}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert routes[1].get("fallback_status") != "failed", routes[1]
    print("REAL_DOCLING_SMOKE_COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ruled-control", action="store_true")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "bear-text-beta" / "models" / "e5")
    main(parser.parse_args())
