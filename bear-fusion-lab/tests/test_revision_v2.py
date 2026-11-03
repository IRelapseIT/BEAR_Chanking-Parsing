from copy import deepcopy
from pathlib import Path

import pytest
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import HFTokenCounter
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import (
    ContentBlock,
    DocumentContext,
    EmbeddingProfile,
    ParsedDocument,
    SourceLocator,
)

from bear_fusion.cli import UnavailableEmbeddings
from bear_fusion.markdown_adapter import StructuredMarkdownParser
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.quality import inspect_result
from bear_fusion.tables import (
    TableProcessor,
    canonical,
    restore_docx_tables,
    validate_table_chunks,
)


@pytest.fixture(scope="module")
def counter():
    return HFTokenCounter(Path(__file__).resolve().parents[2] / "bear-text-beta/models/e5", EmbeddingProfile())


def table_document(text=None):
    raw = text or '<table><thead><tr><th colspan="2">2025</th></tr><tr><th>Revenue</th><th>Costs</th></tr></thead><tr><td>150</td><td>90</td></tr></table>'
    block = ContentBlock("t", 0, "TABLE", "table", "", [], SourceLocator("html"), metadata={"raw_source": raw})
    return ParsedDocument(DocumentContext("t", "s", "d", 1, "x.html"), [block])


def process(document, counter):
    processor = TableProcessor(counter)
    result = ChunkingService(counter, UnavailableEmbeddings(), EmbeddingProfile(), {"TABLE": processor}).process(document)
    validate_table_chunks(document, result.chunks, counter, 320)
    return result


def test_hierarchical_headers_keep_measure_and_year(counter):
    result = process(table_document(), counter)
    assert any("2025 → Revenue: 150" in c.text for c in result.chunks)


@pytest.mark.parametrize("damage", ["value", "header", "drop", "duplicate", "no_proof"])
def test_table_validator_rejects_corruption(counter, damage):
    document = table_document()
    result = process(document, counter)
    chunks = deepcopy(result.chunks)
    if damage == "value":
        chunks[-1].text = "UNRELATED_TEXT"
    elif damage == "header":
        chunks[-1].text = chunks[-1].text.replace("Costs", "Profit")
    elif damage == "drop":
        chunks.pop()
    elif damage == "duplicate":
        chunks.append(deepcopy(chunks[-1]))
    else:
        chunks[-1].metadata = {}
    with pytest.raises(ValueError):
        validate_table_chunks(document, chunks, counter, 320)


def test_long_cell_has_context_in_every_continuation(counter):
    document = table_document('<table><tr><th>Revenue 2025</th></tr><tr><td>' + "abc " * 1400 + '</td></tr></table>')
    result = process(document, counter)
    values = [c for c in result.chunks if c.metadata["cell_id"] == "r1c0"]
    assert len(values) > 2
    assert all("Revenue 2025" in c.text for c in values)
    assert all(c.token_count <= 320 for c in values)


def test_docx_gridspan_and_vmerge(tmp_path):
    from docx import Document
    source = Document()
    table = source.add_table(rows=3, cols=2)
    table.cell(0, 0).merge(table.cell(0, 1)).text = "Year 2025"
    table.cell(1, 0).merge(table.cell(2, 0)).text = "Revenue"
    table.cell(1, 1).text = "150"
    table.cell(2, 1).text = "160"
    path = tmp_path / "merge.docx"
    source.save(path)
    document = table_document()
    restore_docx_tables(path, document)
    cells, _ = canonical(document.blocks[0])
    assert any(c.column_span == 2 and c.text == "Year 2025" for c in cells)
    assert any(c.row_span == 2 and c.text == "Revenue" for c in cells)


def test_markdown_comment_and_container_not_code():
    text = '<!-- hidden -->\n\n<Listing>\n\n```rust\n{{#include x.rs}}\n```\n\n</Listing>\n'
    document = StructuredMarkdownParser().parse(text.encode(), DocumentContext("t", "s", "d", 1, "x.md"))
    assert sum(b.role == "code" for b in document.blocks) == 1
    assert "unresolved_include:1" in document.warnings
    assert not any("hidden" in b.text for b in document.blocks)


def test_blank_middle_page_does_not_abort_document(tmp_path):
    from reportlab.pdfgen import canvas
    path = tmp_path / "blank-middle.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(30, 700, "First page"); c.showPage(); c.showPage()
    c.drawString(30, 700, "Last page"); c.save()
    result = parse_pdf(path, DocumentContext("t", "s", "d", 1, path.name), FileLimits())
    assert "pdf_blank_page:2" in result.warnings
    assert any("Last page" in b.text for b in result.blocks)


def test_pdf_top_to_bottom_order(tmp_path):
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Table, TableStyle
    path = tmp_path / "order.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(30, 780, "Before table")
    t = Table([["Measure", "Value"], ["Revenue", "150"]], colWidths=[100, 100])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    t.wrapOn(c, 300, 300); t.drawOn(c, 30, 650)
    c.drawString(30, 550, "After table"); c.save()
    result = parse_pdf(path, DocumentContext("t", "s", "d", 1, path.name), FileLimits())
    assert result.blocks[0].text == "Before table"
    assert result.blocks[1].type == "TABLE"
    assert result.blocks[2].text == "After table"


def test_complete_blocks_does_not_claim_verified_extraction():
    quality = inspect_result({"chunks": [], "deferred_blocks": [], "warnings": ["ocr_unverified:page:1"], "completeness": "complete"})
    assert quality["block_coverage"] == "complete"
    assert quality["extraction_quality"] == "needs_review"


def test_nested_html_table_not_flattened_into_parent_rows():
    document = table_document('<table><tr><th>A</th></tr><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table>')
    cells, _ = canonical(document.blocks[0])
    assert len(cells) == 2


def test_failed_ocr_page_keeps_following_text_page(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

    import bear_fusion.pdf_adapter as adapter
    path = tmp_path / "retry.pdf"
    c = canvas.Canvas(str(path)); c.showPage(); c.drawString(30, 700, "Surviving page"); c.save()
    def fail(*args):
        raise RuntimeError("test OCR failure")
    monkeypatch.setattr(adapter, "ocr_region", fail)
    result = adapter.parse_pdf(path, DocumentContext("t", "s", "d", 1, path.name), FileLimits())
    assert any(b.metadata.get("page_error") == "test OCR failure" for b in result.blocks)
    assert any("Surviving page" in b.text for b in result.blocks)


def test_mixed_page_native_heading_does_not_disable_region_ocr(tmp_path, monkeypatch):
    from PIL import Image
    from reportlab.pdfgen import canvas

    import bear_fusion.pdf_adapter as adapter
    image = tmp_path / "scan.png"
    Image.new("RGB", (300, 200), "grey").save(image)
    path = tmp_path / "mixed.pdf"
    c = canvas.Canvas(str(path)); c.drawString(30, 750, "Native heading")
    c.drawImage(str(image), 30, 200, width=300, height=200); c.save()
    targets = []
    def read(renderer, index, box=None):
        targets.append(box)
        return "Text in scan", {"extraction": "ocr"}
    monkeypatch.setattr(adapter, "ocr_region", read)
    result = adapter.parse_pdf(path, DocumentContext("t", "s", "d", 1, path.name), FileLimits())
    assert len(targets) == 1 and targets[0] is not None
    assert any("Text in scan" in b.text for b in result.blocks)
    assert any("Native heading" in b.text for b in result.blocks)


def test_row_group_and_row_header_are_both_in_context(counter):
    document = table_document('<table><tr><th>Group</th><th>Product</th><th>Value</th></tr><tr><th scope="rowgroup" rowspan="2">North</th><th scope="row">A</th><td>150</td></tr><tr><th scope="row">B</th><td>90</td></tr></table>')
    result = process(document, counter)
    assert any("North → B" in c.text and "90" in c.text for c in result.chunks)


@pytest.mark.parametrize("mutate", [False, True])
def test_fusion_rejects_bad_external_processor(tmp_path, counter, mutate):
    from bear_text_beta.models import PreparedFragment, SourceSpan
    from bear_text_beta.object_storage import LocalObjectStorage

    from bear_fusion.pipeline import FusionPipeline
    class BadProcessor:
        def process(self, block, context):
            if mutate:
                block.metadata["table_cells"][0]["text"] = "CHANGED"
                return []
            return [PreparedFragment("UNRELATED_TEXT", [SourceSpan(block.id, 0, len(block.text))])]
    source = tmp_path / "input.html"
    source.write_text('<table><tr><th>Year</th></tr><tr><td>2025</td></tr></table>', encoding="utf-8")
    chunker = ChunkingService(counter, UnavailableEmbeddings(), EmbeddingProfile(), {"TABLE": BadProcessor()})
    pipeline = FusionPipeline(None, chunker, LocalObjectStorage(tmp_path / "objects"))
    with pytest.raises(ValueError, match="mutated|proof"):
        pipeline.process_path(source, DocumentContext("t", "s", "d", 1, source.name))
