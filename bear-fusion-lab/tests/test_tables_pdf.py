from pathlib import Path

import pytest
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import ContentBlock, DocumentContext, SourceLocator

from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.tables import TableProcessor, html_rows


def test_merged_cells_expand_without_dropping_values():
    rows = html_rows('<table><tr><th rowspan="2">A</th><th>B</th></tr><tr><td>C</td></tr></table>')
    assert rows == [["A", "B"], ["A", "C"]]


def test_table_values_have_row_name_and_header():
    block = ContentBlock("t", 0, "TABLE", "table", "raw", [], SourceLocator("test"),
                         metadata={"table_rows": [["Metric", "2025, million RUB"], ["Revenue", "150"]]})
    fragments = TableProcessor().process(block, None)
    assert any(all(term in f.text for term in ["Revenue", "2025", "150", "million RUB"]) for f in fragments)


def test_header_only_template_is_searchable():
    block = ContentBlock("t", 0, "TABLE", "table", "raw", [], SourceLocator("test"),
                         metadata={"table_rows": [["Instructions"], [""]]})
    assert TableProcessor().process(block, None)[0].text == "Table headers: Instructions"


def test_no_silent_long_cell_truncation():
    block = ContentBlock("t", 0, "TABLE", "table", "raw", [], SourceLocator("test"),
                         metadata={"table_rows": [["Value"], ["x" * 600 + "END"]]})
    fragments = TableProcessor().process(block, None)[1:]
    assert "".join(f.text[len(f.metadata["prefix"]):] for f in fragments) == "x" * 600 + "END"
    assert all("Value" in f.text for f in fragments)


def test_pdf_page_limit(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "two.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(10, 100, "Page 1"); c.showPage(); c.drawString(10, 100, "Page 2"); c.save()
    with pytest.raises(ValueError, match="page limit"):
        parse_pdf(Path(path), DocumentContext("t", "s", "d", 1, "two.pdf"), FileLimits(max_pages=1))
