
import pytest
from bear_text_beta.file_validation import FileLimits, FileValidator
from bear_text_beta.models import DocumentContext
from bear_text_beta.parsing import MDParser

from bear_fusion.bridge import map_tree
from bear_fusion.preflight import reject_known_table_loss


def test_large_markdown_table_retains_last_row():
    text = "| A |\n|---|\n" + "\n".join(f"| {i} |" for i in range(5100)) + "\n| END_MARKER |\n"
    parsed = MDParser().parse(text.encode(), DocumentContext("t", "s", "d", 1, "a.md"))
    assert any("END_MARKER" in b.metadata.get("raw_source", "") for b in parsed.blocks)


def test_repeated_paragraphs_have_distinct_sources():
    child = {"type": "paragraph", "text": "Одинаковый текст 👩‍💻", "children": []}
    parsed = map_tree({"tree": {"type": "document", "text": "", "children": [child, child]}},
                      DocumentContext("t", "s", "d", 1, "a.txt"))
    assert len({b.id for b in parsed.blocks}) == 2
    assert parsed.blocks[0].text == parsed.blocks[1].text


def test_depth_limit():
    root = {"type": "document", "text": "", "children": []}
    for _ in range(130):
        root = {"type": "document", "text": "", "children": [root]}
    with pytest.raises(ValueError, match="depth"):
        map_tree({"tree": root}, DocumentContext("t", "s", "d", 1, "a.txt"))


def test_docx_long_cell_rejected_before_upstream(tmp_path):
    from docx import Document
    doc = Document()
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "x" * 600
    path = tmp_path / "large.docx"
    doc.save(path)
    FileValidator(FileLimits()).validate_docx(path.read_bytes())
    with pytest.raises(ValueError, match="TRUNCATION_RISK"):
        reject_known_table_loss(path)


def test_regular_table_allowed(tmp_path):
    path = tmp_path / "small.html"
    path.write_text("<table><tr><td>small</td></tr></table>", encoding="utf-8")
    reject_known_table_loss(path)
