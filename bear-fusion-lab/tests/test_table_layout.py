import json
import os
import shutil

import pypdfium2 as pdfium
import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from bear_text_beta.models import DocumentContext
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.parsing import ParsingPipeline
from bear_text_beta.unified_parsing import PDFParser
from bear_fusion.tables import TableProcessor
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import FallbackPolicy
from bear_fusion.raster import raster_region
from bear_fusion.table_layout import assign_words


def fixture(path, *, merged=True):
    pdf = Canvas(str(path), pagesize=(450, 350))
    for y in (290, 260, 230, 200, 170):
        pdf.line(45, y, 400, y)
    for x, top in ((45, 290), (135, 290), (270, 260 if merged else 290), (400, 290)):
        pdf.line(x, 170, x, top)
    for x, y, content in (
        (55, 270, "Record"), (180, 270, "Finances"), (55, 239, "Year"),
        (150, 239, "Revenue USD"), (285, 239, "Cost USD"),
        (55, 209, "2024"), (155, 209, "-125.50"), (285, 209, "80"),
        (55, 179, "2025"), (285, 179, "90"),
    ):
        pdf.drawString(x, y, content)
    pdf.save()


def parse(path):
    return parse_pdf(path, DocumentContext("t", "s", "test", 1, path.name), FileLimits(),
                     policy=FallbackPolicy(enabled=False), ocr_mode="baseline", field_profile=[])


def test_merged_header_and_blank_cell_end_to_end(tmp_path):
    path = tmp_path / "ruled.pdf"
    fixture(path)
    limits = FileLimits()
    parser = ParsingPipeline(tmp_path, extra_parsers={".pdf": PDFParser(limits)}, limits=limits)
    block, = parser.parse(path.read_bytes(), DocumentContext("t", "s", "d", 1, "REPORT.PDF")).blocks
    table = block.metadata
    assert table["table_geometry_source"] == "vector_rules"
    cells = {c["id"]: c for c in table["table_cells"]}
    assert cells["r0c1"]["column_span"] == 2
    assert cells["r2c1"]["text"] == "-125.50"
    assert table["cell_header_links"]["r2c1"] == {
        "column_header_ids": ["r0c1", "r1c1"], "row_header_ids": ["r2c0"]}
    assert table["cell_header_links"]["r2c2"]["column_header_ids"] == ["r0c1", "r1c2"]
    assert cells["r3c1"]["text"] == ""
    assert "r3c1" in table["cell_bboxes"] and "r3c1" not in table["cell_content_bboxes"]
    fragments = TableProcessor().process(block, None)
    assert any("Row 3 (2024); Finances → Revenue USD: -125.50" in f.text for f in fragments)


def test_word_on_divider_is_ambiguous():
    boxes = {"a": [10, 10, 50, 40], "b": [50, 10, 90, 40]}
    words = [{"text": "500", "bbox_pdf_points": [45, 18, 55, 29]}]
    assert assign_words(boxes, words) is None


def scan_from_vector(tmp_path):
    source = tmp_path / "original.pdf"
    fixture(source)
    scan = tmp_path / "scan.pdf"
    with pdfium.PdfDocument(source) as renderer, raster_region(renderer, 0) as (picture, _, _, size):
        writer = Canvas(str(scan), pagesize=size)
        writer.drawImage(ImageReader(picture), 0, 0, width=size[0], height=size[1])
        writer.save()
    return scan


@pytest.mark.skipif(not (os.getenv("BEAR_TESSERACT") or shutil.which("tesseract")),
                    reason="local Tesseract not available")
def test_scanned_rules_with_real_tesseract(tmp_path):
    document = parse(scan_from_vector(tmp_path))
    tables = [b for b in document.blocks if b.metadata.get("table_geometry_source") == "raster_rules"]
    table, = tables
    cells = {c["id"]: c["text"] for c in table.metadata["table_cells"]}
    assert cells["r2c1"] == "-125.50"
    assert table.metadata["cell_header_links"]["r2c1"]["column_header_ids"] == ["r0c1", "r1c1"]
    route = json.loads(next(w for w in document.warnings if w.startswith("pdf_page_route:")).split(":", 1)[1])
    assert route["status"] == "needs_review"
    assert sum("-125.50" in b.text for b in document.blocks) == 1
