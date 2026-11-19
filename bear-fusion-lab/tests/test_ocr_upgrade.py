import json

import pytest
from bear_fusion.ocr_verification import compare_lines, primary_lines
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import FallbackPolicy
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext
from reportlab.pdfgen import canvas


def observation(text, y=10):
    return {"text": text, "bbox_pdf_points": [10, y, 110, y + 10]}


@pytest.mark.parametrize("alternate", ["Баланс 12,5 %", "Баланс −12.5 %", "Баланс −12,5 руб."])
def test_numeric_difference_is_a_conflict(alternate):
    assert compare_lines([observation("Баланс −12,5 %")], [observation(alternate)])["status"] == "disagreement"


def test_reading_order_is_geometric():
    primary = [observation("A 10"), observation("B 20", 30)]
    assert compare_lines(primary, primary[::-1])["status"] == "agreement"
    swapped = [observation("B 20"), observation("A 10", 30)]
    assert compare_lines(primary, swapped)["status"] == "disagreement"
    assert compare_lines(primary, primary[:1])["status"] == "insufficient"


def test_crop_mapping_preserves_pdf_offset():
    words = [{"line_id": (1, 2, 3), "text": "12", "confidence": 80, "bbox_pixels": [9, 15, 6, 3]}]
    assert primary_lines(words, [1 / 3, 0, 0, 1 / 3, 10, 20]) == [
        {"text": "12", "confidence": .8, "bbox_pdf_points": [13, 25, 15, 26]}]


@pytest.mark.parametrize("shift", [0, 20])
def test_pdf_grid_geometry_and_review(tmp_path, shift):
    path = tmp_path / "grid.pdf"
    writer = canvas.Canvas(str(path))
    writer.drawString(40, 740, "Table 1 Revenue report")
    expected = [["Year", "Revenue USD", "Cost USD"], ["2024", "150", "80"],
                ["2025", "180", "90"], ["2026", "200", "95"]]
    for row, values in enumerate(expected):
        for col, value in enumerate(values):
            writer.drawString(70 + col * 170 + (shift if row == 2 and col == 1 else 0), 680 - row * 30, value)
    writer.save()
    result = parse_pdf(path, DocumentContext("t", "s", "d", 1, path.name), FileLimits(),
                       policy=FallbackPolicy(enabled=False))
    tables = [b for b in result.blocks if b.type == "TABLE" and not b.metadata.get("unresolved")]
    route = next(json.loads(w.split(":", 1)[1]) for w in result.warnings if w.startswith("pdf_page_route:"))
    assert route["status"] == "needs_review"
    table, = tables
    cells = table.metadata["table_cells"]
    assert {(c["row"], c["column"]): c["text"] for c in cells} == {
        (row, col): value for row, values in enumerate(expected) for col, value in enumerate(values)}
    assert set(table.metadata["cell_bboxes"]) == {c["id"] for c in cells}
    assert table.metadata["cell_header_links"]["r2c1"] == {
        "column_header_ids": ["r0c1"], "row_header_ids": ["r2c0"]}
    assert table.metadata["cell_bboxes"]["r2c1"] != table.metadata["cell_content_bboxes"]["r2c1"]
    assert [b.text for b in result.blocks if b.type == "TEXT"] == ["Table 1 Revenue report"]
