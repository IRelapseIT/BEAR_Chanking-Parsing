import base64
import codecs
import io
import json
from dataclasses import replace

import httpx
import pytest
from docx import Document

from bear_text_beta.errors import ParseError
from bear_text_beta.parsing import ParsingPipeline, TXTParser, decode_text


@pytest.mark.parametrize("encoding,bom", [
    ("utf-8", b""), ("utf-8", codecs.BOM_UTF8), ("utf-16-le", codecs.BOM_UTF16_LE),
    ("utf-16-be", codecs.BOM_UTF16_BE), ("utf-32-le", codecs.BOM_UTF32_LE),
])
def test_text_encodings_and_newlines(encoding, bom):
    text, warnings = decode_text(bom + "Текст\r\nАбзац".encode(encoding))
    assert text == "Текст\nАбзац"
    assert warnings == []


def test_cp1251_requires_explicit_policy(tmp_path, context):
    data = "Пример".encode("cp1251")
    with pytest.raises(ParseError):
        ParsingPipeline(tmp_path, extra_parsers={".txt": TXTParser()}).parse(data, context)
    result = ParsingPipeline(tmp_path, allow_cp1251=True,
                             extra_parsers={".txt": TXTParser(True)}).parse(data, context)
    assert result.blocks[0].text == "Пример"
    assert result.warnings == ["legacy_encoding:cp1251"]


def test_plain_text_sources(parsed):
    result = parsed("Первый абзац.\r\nПродолжение.\r\n\r\nВторой абзац.")
    assert [block.text for block in result.blocks] == ["Первый абзац.\nПродолжение.", "Второй абзац."]
    assert [(block.source.line_start, block.source.line_end) for block in result.blocks] == [(1, 2), (4, 4)]


def test_markdown_structures_and_table(parsed):
    result = parsed("# Заголовок\n\nТекст [ссылка](https://example.org).\n\n"
                    "1. Первый\n   - Вложенный\n\n```go\n  x := 1\n```\n\n"
                    "| A | B |\n| --- | --- |\n| 1 | 2 |\n", "document.md")
    assert result.blocks[0].role == "heading"
    assert any(block.role == "code" and "  x := 1" in block.text for block in result.blocks)
    assert sum(block.role == "list_item" for block in result.blocks) == 2
    table = next(block for block in result.blocks if block.type == "TABLE")
    assert "| 1 | 2 |" in table.metadata["raw_source"]
    assert all(block.section_path == ["Заголовок"] for block in result.blocks)
    assert result.blocks[1].metadata["links"] == ["https://example.org"]


def test_html_order_tables_images_and_code(parsed):
    result = parsed("<body><h1>Раздел</h1><div>Начало <b>важно</b></div>"
                    "<p>Повтор</p><table><tr><td>1</td></tr></table><p>Повтор</p>"
                    "<pre>  a\n    b</pre><img src='https://example.org/image.png' alt='Схема'>"
                    "<script>secret()</script><style>.x {}</style></body>", "document.html")
    assert [block.type for block in result.blocks] == ["TEXT", "TEXT", "TEXT", "TABLE", "TEXT", "TEXT", "IMAGE"]
    assert result.blocks[1].text == "Начало важно"
    assert result.blocks[5].text == "  a\n    b"
    assert "secret" not in str(result.blocks)
    assert result.attachments[0].metadata["stored"] is False
    assert result.blocks[2].source.element_path != result.blocks[4].source.element_path


def test_docx_dedoc_contract_order_and_embedded_image(parser, context):
    doc = Document()
    doc.add_heading("Раздел", level=1)
    doc.add_paragraph("До таблицы")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "Данные"
    doc.add_paragraph("После таблицы", style="List Bullet")
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBFEAAAAASUVORK5CYII=")
    doc.add_picture(io.BytesIO(png))
    stream = io.BytesIO()
    doc.save(stream)
    payload = {"schema_version": 1, "warnings": [], "blocks": [
        {"type": "TEXT", "role": "heading", "text": "Раздел", "node_id": "0.0",
         "line_id": 0, "page": 1, "heading_level": 1},
        {"type": "TEXT", "role": "paragraph", "text": "До таблицы", "node_id": "0.1",
         "line_id": 1, "page": 1},
        {"type": "TABLE", "role": "table", "text": "", "node_id": "0.1", "line_id": 1,
         "page": 1, "metadata": {"raw_source": json.dumps({"cells": [["Данные"]]}, ensure_ascii=False)}},
        {"type": "TEXT", "role": "list_item", "text": "После таблицы", "node_id": "0.2",
         "line_id": 2, "page": 1},
        {"type": "IMAGE", "role": "image", "text": "", "node_id": "0.3", "line_id": 3,
         "page": 1, "attachment_uids": ["img-1"]},
    ], "attachments": [{"uid": "img-1", "name": "image.png", "mime_type": "image/png",
                        "data_b64": base64.b64encode(png).decode()}]}
    parser.parsers[".docx"].client = httpx.Client(
        base_url="http://dedoc", transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    result = parser.parse(stream.getvalue(), replace(context, filename="document.docx"))
    assert [block.type for block in result.blocks] == ["TEXT", "TEXT", "TABLE", "TEXT", "IMAGE"]
    assert result.blocks[3].role == "list_item"
    assert "Данные" in result.blocks[2].metadata["raw_source"]
    asset = result.attachments[0]
    assert asset.metadata["stored"]
    assert parser.parsers[".docx"].storage._path(asset.object_key).read_bytes() == png


@pytest.mark.parametrize("filename,data", [("file.txt", b"a\x00b"), ("file.pdf", b"%PDF"),
                                           ("file.docx", b"not a zip")])
def test_invalid_formats(parser, context, filename, data):
    with pytest.raises(ParseError):
        parser.parse(data, replace(context, filename=filename))


def test_size_limit_and_checksum(tmp_path, context):
    parser = ParsingPipeline(tmp_path, max_file_bytes=3)
    with pytest.raises(ParseError):
        parser.parse(b"1234", context)
    with pytest.raises(ParseError):
        parser.parse(b"123", replace(context, checksum="wrong"))


def test_whitespace_only_has_no_blocks(parsed):
    assert parsed(" \n\n\t").blocks == []
