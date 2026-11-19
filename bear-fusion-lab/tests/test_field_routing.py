import json
import subprocess
from dataclasses import replace

import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from bear_fusion import pdf_adapter
from bear_fusion.field_routing import FieldRule
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import ContentBlock, DocumentContext, ParsedDocument, SourceLocator
from bear_fusion.ocr_verification import PaddleVerifier, pdf_box
from bear_fusion.pdf_fallback import FallbackPolicy, route_pages
from bear_fusion.raster import raster_region

CONTEXT = DocumentContext("t", "s", "d", 1, "test.pdf")
LIMITS = FileLimits()


def warnings(document, prefix):
    return [json.loads(line.split(":", 1)[1]) for line in document.warnings if line.startswith(prefix + ":")]


def mixed_fixture(path):
    pdf = Canvas(str(path), pagesize=(400, 400))
    pdf.drawString(30, 360, "Title")
    pdf.showPage()
    pdf.showPage()
    with Image.new("RGB", (350, 240), "white") as img:
        ImageDraw.Draw(img).text((30, 40), "Order number: 48291", fill="black")
        pdf.drawImage(ImageReader(img), 10, 100, 350, 240)
        pdf.showPage()
        pdf.drawString(20, 375, "Total due:")
        pdf.drawImage(ImageReader(img), 10, 60, 350, 240)
    pdf.drawString(20, 340, "Order number:")
    pdf.line(130, 338, 235, 338)
    # Vector outlines emulate visible marks without corresponding text objects.
    for x in (135, 141, 148, 156):
        pdf.rect(x, 374, 3, 7, fill=1)
    pdf.save()


def stub_tesseract(monkeypatch, text=""):
    calls = []
    monkeypatch.setenv("BEAR_TESSERACT", "fake-tesseract")

    def run(command, *args, **kwargs):
        with Image.open(command[1]) as image:
            calls.append(image.size)
        return subprocess.CompletedProcess(command, 0, ((
            "block_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            + (f"1\t1\t1\t100\t200\t120\t35\t92\t{text}\n" if text else "")).encode()), b"")

    monkeypatch.setattr(pdf_adapter.subprocess, "run", run)
    return calls


class Secondary:
    def __init__(self, text="-12.50 USD", failure=None):
        self.calls, self.text, self.failure = [], text, failure

    def verify(self, image, page, words, transform, size):
        self.calls.append((page, transform, size))
        if self.failure:
            raise self.failure
        return {"status": "disagreement", "reasons": [],
                "secondary_lines": [{"text": self.text, "confidence": .97,
                                     "bbox_pdf_points": pdf_box([12, 1, 90, size[1] - 1], transform)}]}


class Backend:
    def __init__(self, blocks=()):
        self.blocks, self.calls = blocks, []

    def parse_page(self, path, page, context, limits):
        self.calls.append(page)
        return self.blocks


def test_clean_blank_scan_mixed_and_field_routes(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    mixed_fixture(path)
    calls = stub_tesseract(monkeypatch, "test")
    secondary = Secondary()
    docling = Backend()
    profile = (
        FieldRule("amount", pattern=r"-\d+\.\d+ USD", page=4, bbox=(130, 16, 225, 34)),
        FieldRule("blank", pattern=r"\d+", page=4, bbox=(130, 50, 235, 70)),
        FieldRule("missing", ("Unknown label",), pattern=r"\d+"),
    )
    result = pdf_adapter.parse_pdf(path, CONTEXT, LIMITS, verifier=secondary,
                                   fallback=docling, field_profile=profile)
    routes = warnings(result, "pdf_page_route")
    assert [r["text_layer"] for r in routes] == ["usable", "absent", "absent", "partial"]
    assert routes[1]["blank"]
    assert calls == [(1200, 1200), (1050, 720)]
    assert docling.calls == []
    assert [page for page, _, _ in secondary.calls] == [4]
    fields = {r["field"]: r for r in warnings(result, "pdf_field_route")}
    assert fields["amount"]["reason"] == "possible_ocr_omission"
    assert fields["amount"]["status"] == "recovered_by_secondary"
    assert fields["amount"]["value"] == "-12.50 USD"
    assert fields["blank"]["status"] == "blank_in_source"
    assert fields["missing"]["reason"] == "field_not_localized"
    assert any(b.text == "-12.50 USD" and b.source.page == 4 and b.source.bbox for b in result.blocks)
    assert result.schema_version == 1


def test_merged_required_crops_take_budget_before_optional(tmp_path):
    path = tmp_path / "marks.pdf"
    pdf = Canvas(str(path), pagesize=(300, 300))
    pdf.drawString(20, 275, "Form")
    for x in (25, 30, 35, 115, 120, 125):
        pdf.rect(x, 243, 2, 6, fill=1)
    pdf.save()

    class OneCrop(PaddleVerifier):
        def __init__(self):
            super().__init__("python", ".", max_regions=1)
            self.calls = 0

        def invoke(self, mode, image, model, budget):
            self.calls += 1
            raise subprocess.TimeoutExpired("paddle", 1)

    secondary = OneCrop()
    rules = (
        FieldRule("optional", page=1, bbox=(100, 44, 160, 63), required=False),
        FieldRule("required", page=1, bbox=(20, 44, 50, 63)),
        FieldRule("overlap", page=1, bbox=(23, 44, 55, 63)),
    )
    result = pdf_adapter.parse_pdf(path, CONTEXT, LIMITS, verifier=secondary, field_profile=rules)
    routes = warnings(result, "pdf_ocr_route")
    assert secondary.calls == 1
    assert [r["bbox"] for r in routes] == [[20, 44, 55, 63], [100, 44, 160, 63]]
    assert [r["reasons"] for r in routes] == [["TimeoutExpired"], ["verification_budget_exhausted"]]
    assert all(r["status"] == "needs_review" for r in warnings(result, "pdf_field_route"))


def test_sign_conflict_preserves_both_observations(tmp_path):
    path = tmp_path / "amount.pdf"
    pdf = Canvas(str(path), pagesize=(300, 300))
    pdf.drawString(25, 250, "-12.50 USD")
    pdf.save()
    profile = (FieldRule("amount", pattern=r"\d+\.\d+ USD", page=1, bbox=(20, 35, 120, 58)),)
    result = pdf_adapter.parse_pdf(path, CONTEXT, LIMITS, verifier=Secondary("12.50 USD"), field_profile=profile)
    field, = warnings(result, "pdf_field_route")
    assert (field["status"], field["decision"]) == ("needs_review", "ocr_conflict")
    assert field["primary"]["text"] == "-12.50 USD"
    assert field["secondary"]["text"] == "12.50 USD"
    assert [b.text for b in result.blocks] == ["-12.50 USD"]


def test_intrinsic_rotation_keeps_raster_and_source_aligned(tmp_path):
    import pdfplumber

    path = tmp_path / "rotated.pdf"
    pdf = Canvas(str(path), pagesize=(200, 300))
    pdf.setPageRotation(90)
    pdf.drawString(40, 130, "Marker")
    pdf.save()
    with pdfplumber.open(path) as document, pdfium.PdfDocument(path) as renderer:
        page = document.pages[0]
        word, = page.extract_words()
        box = pdf_adapter.word_box(word)
        with raster_region(renderer, 0, box) as (image, transform, _, size):
            assert size == (page.width, page.height)
            with image.convert("L") as gray:
                assert gray.getextrema()[0] < 150
            actual = pdf_box([0, 0, *image.size], transform)
            assert max(abs(a - b) for a, b in zip(actual, box)) <= 1 / 3 + 1e-8


def make_block(text, box):
    return ContentBlock(text, 0, "TEXT", "paragraph", text, [],
                        SourceLocator("pdf", page=1, bbox=box))


def page_info(blocks=()):
    return {"width": 400, "height": 400, "rotation": 0, "text_layer": "usable", "ocr": [],
            "ocr_attempts": [], "native": [{"text": b.text, "engine": "pdfplumber",
                                            "bbox_pdf_points": b.source.bbox} for b in blocks]}


def test_column_order_selects_only_structural_repair():
    a = make_block("Alpha", [20, 50, 90, 60])
    b = make_block("Beta", [20, 70, 90, 80])
    c = make_block("Gamma", [230, 50, 300, 60])
    d = make_block("Delta", [230, 70, 300, 80])
    for candidate, expected in (([a, c, b, d], "bear"), ([a, b, c, d], "docling")):
        document = ParsedDocument(CONTEXT, [a, c, b, d], warnings=["pdf_reading_order_needs_review:page:1"])
        backend = Backend(candidate)
        route_pages(document, "unused", LIMITS, backend, page_inputs={1: page_info()})
        assert backend.calls == [1]
        assert warnings(document, "pdf_page_route")[0]["selected_parser"] == expected


def test_table_rejects_swapped_coordinates_and_preserves_header_links():
    from dataclasses import asdict

    from bear_fusion.tables import Cell

    blocks = [make_block(text, [x, y, x + 30, y + 8])
              for text, x, y in (("Year", 10, 30), ("Cost", 100, 30), ("2024", 10, 50), ("-125", 100, 50))]
    caption = make_block("Table 1", [10, 10, 50, 20])
    cells = [asdict(Cell(str(i), i // 2, i % 2, block.text, header=i < 2))
             for i, block in enumerate(blocks)]
    coords = {str(i): block.source.bbox for i, block in enumerate(blocks)}
    for swap in (False, True):
        evidence = coords.copy()
        if swap:
            evidence["2"], evidence["3"] = evidence["3"], evidence["2"]
        table = replace(make_block("Year | Cost\n2024 | -125", [8, 25, 135, 60]), type="TABLE",
                        metadata={"table_cells": cells, "cell_bboxes": evidence})
        original = [caption, *blocks]
        document = ParsedDocument(CONTEXT, original.copy())
        route_pages(document, "unused", LIMITS, Backend([caption, table]), FallbackPolicy(),
                    page_inputs={1: page_info(original)})
        route, = warnings(document, "pdf_page_route")
        assert route["selected_parser"] == ("bear" if swap else "docling")
        if not swap:
            assert document.blocks[-1].metadata["values_verified"]
