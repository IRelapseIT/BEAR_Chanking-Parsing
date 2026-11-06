import json
import subprocess
from dataclasses import replace

import pytest
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import (
    FallbackPolicy,
    map_docling,
    page_reasons,
    route_pages,
)
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import (
    ContentBlock,
    DocumentContext,
    ParsedDocument,
    SourceLocator,
)

CTX = DocumentContext("t", "s", "d", 1, "test.pdf")
LIMITS = FileLimits()


def block(page=1, text="Original text 2025", **metadata):
    return ContentBlock(f"b{page}", page - 1, "TEXT", "paragraph", text, [],
                        SourceLocator("pdf", page=page), metadata=metadata)


class Backend:
    def __init__(self, candidate=None, error=None):
        self.candidate, self.error, self.calls = candidate, error, []

    def parse_page(self, path, page, context, limits):
        self.calls.append(page)
        if self.error:
            raise self.error
        return [replace(b, source=replace(b.source, page=page)) for b in self.candidate]


def test_only_bad_page_replaced_whole_without_duplicates():
    first, second = block(), block(2)
    doc = ParsedDocument(CTX, [first, second], warnings=["pdf_unknown_glyph:page:2"])
    backend = Backend([block(text="Original text 2025 repaired")])
    result = route_pages(doc, "unused", LIMITS, backend)
    assert backend.calls == [2]
    assert result.blocks[0] is first
    assert [b.text for b in result.blocks] == [first.text, "Original text 2025 repaired"]
    assert [b.order for b in result.blocks] == [0, 1]
    assert result.blocks[1].metadata["pdf_route"]["selected_parser"] == "docling"
    assert "pdf_unknown_glyph:page:2" in result.warnings


@pytest.mark.parametrize("text", ["Other text 2025", "Original text 2024", "", "Original text \ufffd 2025"])
def test_candidate_not_assumed_better(text):
    original = block()
    doc = ParsedDocument(CTX, [original], warnings=["pdf_unknown_glyph:page:1"])
    route_pages(doc, "unused", LIMITS, Backend([block(text=text)]))
    assert doc.blocks == [original]
    assert original.metadata["pdf_route"]["status"] == "needs_review"


@pytest.mark.parametrize("before,after", [("Balance -12", "Balance 12"), ("Value 3.14", "Value 3,14"),
                                         ("Rate 12%", "Rate 12")])
def test_numeric_sign_separator_and_percent_preserved(before, after):
    original = block(text=before)
    doc = ParsedDocument(CTX, [original], warnings=["pdf_unknown_glyph:page:1"])
    route_pages(doc, "unused", LIMITS, Backend([block(text=after)]))
    assert doc.blocks[0] is original
    assert original.metadata["pdf_route"]["status"] == "needs_review"


def test_order_disagreement_stays_review_even_if_words_preserved():
    original = block(text="Alpha Beta")
    doc = ParsedDocument(CTX, [original], warnings=["pdf_reading_order_needs_review:page:1"])
    route_pages(doc, "unused", LIMITS, Backend([block(text="Beta Alpha")]))
    assert doc.blocks[0] is original
    assert original.metadata["pdf_route"]["fallback_status"] == "rejected_or_ambiguous"


@pytest.mark.parametrize("error", [RuntimeError("bad"), subprocess.TimeoutExpired("docling", 1),
                                   FileNotFoundError("python"), ValueError("cells")])
def test_worker_failure_preserves_original(error):
    original = block()
    doc = ParsedDocument(CTX, [original], warnings=["pdf_unknown_glyph:page:1"])
    route_pages(doc, "unused", LIMITS, Backend(error=error))
    assert doc.blocks[0] is original
    assert original.metadata["pdf_route"]["fallback_status"] == "failed"


def test_page_budget_disable_and_warning_page_boundary():
    backend = Backend([block()])
    doc = ParsedDocument(CTX, [block(), block(10)], warnings=["pdf_unknown_glyph:page:10"])
    route_pages(doc, "unused", LIMITS, backend)
    assert backend.calls == [10]
    doc = ParsedDocument(CTX, [block(), block(2)], warnings=["pdf_page_failed:1:X", "pdf_page_failed:2:X"])
    backend.calls.clear()
    route_pages(doc, "unused", LIMITS, backend, FallbackPolicy(max_pages=1))
    assert backend.calls == [1]
    assert doc.blocks[1].metadata["pdf_route"]["fallback_status"] == "page_budget_exceeded"
    doc = ParsedDocument(CTX, [block()], warnings=["pdf_page_failed:1:X"])
    backend.calls.clear()
    route_pages(doc, "unused", LIMITS, backend, FallbackPolicy(enabled=False))
    assert backend.calls == []


def test_low_confidence_and_caption_signals():
    assert "low_ocr_confidence" in page_reasons([block(mean_word_confidence=30)], [], 1, FallbackPolicy())
    assert "caption_without_table" in page_reasons([block(text="Таблица 2")], [], 1, FallbackPolicy())


def test_unknown_ocr_confidence_cannot_clear_low_confidence_source():
    original = block(mean_word_confidence=20)
    doc = ParsedDocument(CTX, [original])
    route_pages(doc, "unused", LIMITS, Backend([block()]))
    assert doc.blocks[0] is original
    assert "candidate_ocr_confidence_unknown" in original.metadata["pdf_route"]["candidate_reasons"]


def test_docling_low_confidence_and_missing_cells_rejected():
    assert "low_ocr_confidence" in page_reasons(
        [block(docling_confidence={"ocr_score": .3})], [], 1, FallbackPolicy())
    data = payload()
    data["document"]["tables"][0]["data"]["table_cells"].pop()
    with pytest.raises(ValueError, match="missing grid"):
        map_docling(data, 1, CTX, LIMITS)


def payload():
    box = {"l": 10, "t": 190, "r": 100, "b": 100, "coord_origin": "BOTTOMLEFT"}
    cells = [{"start_row_offset_idx": r, "end_row_offset_idx": r + 1,
              "start_col_offset_idx": c, "end_col_offset_idx": c + 1,
              "text": str(r * 2 + c), "column_header": r == 0}
             for r in range(2) for c in range(2)]
    return {"version": "fixture", "ordered_refs": ["#/texts/0", "#/tables/0"], "document": {
        "pages": {"1": {"size": {"width": 200, "height": 200}}},
        "texts": [{"text": "Table 1", "label": "caption", "prov": [{"page_no": 1, "bbox": box}]}],
        "tables": [{"prov": [{"page_no": 1, "bbox": box}],
                    "data": {"num_rows": 2, "num_cols": 2, "table_cells": cells}}]}}


def test_direct_cells_mapping_coordinates_and_provenance():
    blocks = map_docling(payload(), 1, CTX, LIMITS)
    assert blocks[0].source.bbox == [10, 10, 100, 100]
    assert blocks[1].metadata["table_cells"][3]["text"] == "3"
    assert blocks[1].metadata["header_rows"] == 1
    assert blocks[1].metadata["parser_version"] == "fixture"
    assert len({b.id for b in blocks}) == 2


@pytest.mark.parametrize("fault", ["overlap", "page", "bbox", "text_limit"])
def test_invalid_docling_structure_rejected(fault):
    data = payload()
    limits = LIMITS
    if fault == "overlap":
        data["document"]["tables"][0]["data"]["table_cells"].append(
            data["document"]["tables"][0]["data"]["table_cells"][0])
    elif fault == "page":
        data["document"]["texts"][0]["prov"][0]["page_no"] = 2
    elif fault == "bbox":
        data["document"]["texts"][0]["prov"][0]["bbox"]["l"] = -2
    else:
        limits = replace(limits, max_text_chars=1)
    with pytest.raises(ValueError):
        map_docling(data, 1, CTX, limits)


def test_generated_pdf_routes_only_caption_page(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "test.pdf"
    writer = canvas.Canvas(str(path))
    writer.drawString(40, 700, "Ordinary page remains native.")
    writer.showPage()
    writer.drawString(40, 700, "Table 1 Missing grid")
    writer.save()
    replacement = block(text="Table 1 Missing grid")
    table = replace(block(), type="TABLE", text="Year | Value\n2025 | 12",
                    metadata={"table_rows": [["Year", "Value"], ["2025", "12"]]})
    backend = Backend([replacement, table])
    result = parse_pdf(path, CTX, LIMITS, fallback=backend)
    assert backend.calls == [2]
    assert len([b for b in result.blocks if b.source.page == 1]) == 1
    assert len([b for b in result.blocks if b.source.page == 2]) == 2
    records = [json.loads(w.split(":", 1)[1]) for w in result.warnings if w.startswith("pdf_page_route:")]
    assert [r["selected_parser"] for r in records] == ["bear", "docling"]
