import io
from dataclasses import replace
from datetime import date
from zipfile import ZipFile

import pytest

from bear_text_beta.errors import ParseError
from bear_text_beta.file_validation import FileLimits, FileValidator
from bear_text_beta.object_storage import LocalObjectStorage
from bear_text_beta.parsing import ParsingPipeline


def test_csv_keeps_source_rows_and_missing_header_decision(tmp_path, context):
    data = 'Код,Описание,\r\n0004,"north,west\r\nwing",\r\n\r\n,Done,0\r\n'.encode()
    result = ParsingPipeline(tmp_path).parse(data, replace(context, filename="prices.csv"))
    table, = result.blocks
    assert result.context.mime_type == "text/csv"
    assert table.metadata["table_rows"] == [
        ["Код", "Описание", ""], ["0004", "north,west\nwing", ""], [], ["", "Done", "0"],
    ]
    assert table.metadata["header_rows"] == 0
    assert "table_header_unconfirmed" in result.warnings[0]
    assert table.metadata["cell_origins"]["r1c1"] == {"row": 2, "column": 2, "line_start": 2, "line_end": 3}
    assert table.metadata["cell_origins"]["r3c2"]["line_start"] == 5


@pytest.mark.parametrize(("suffix", "value", "delimiter"), [
    (".tsv", '"Код\tобъекта"\t001\n\t\n', "\t"),
    (".csv", 'Код;001\nЦвет;"синий;серый"\n', ";"),
])
def test_delimited_workbook_encodings(tmp_path, context, suffix, value, delimiter):
    pipeline = ParsingPipeline(tmp_path)
    parsed = pipeline.parse(value.encode("utf-16"), replace(context, filename="data" + suffix))
    block, = parsed.blocks
    assert block.metadata["delimiter"] == delimiter
    assert block.metadata["table_rows"][0][1] == "001"
    assert len(block.metadata["table_rows"][1]) == 2


def test_delimited_rejects_bad_encoding_and_malformed_quotes(tmp_path, context):
    ctx = replace(context, filename="entries.csv")
    with pytest.raises(ParseError, match="кодировк"):
        ParsingPipeline(tmp_path).parse(b"\x9f", replace(ctx))
    with pytest.raises(ParseError, match="CSV"):
        ParsingPipeline(tmp_path).parse(b'first,"unterminated\r\n', ctx)


def make_xlsx():
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Rates"
    sheet["A1"] = "Rates sheet"
    sheet.merge_cells("A1:B2")
    sheet["A3"] = "0004"
    sheet["B3"] = 5
    sheet["B3"].number_format = "0000"
    sheet["C3"] = "=B3*2"
    sheet["D3"] = date(2026, 11, 26)
    sheet["E3"] = False
    sheet.row_dimensions[3].hidden = True
    hidden = book.create_sheet("Hidden data")
    hidden.sheet_state = "veryHidden"
    hidden["A20"] = "Hidden datum"
    book.create_sheet("Empty")
    original = io.BytesIO()
    book.save(original)
    cached = io.BytesIO()
    with ZipFile(original) as source, ZipFile(cached, "w") as destination:
        for entry in source.infolist():
            data = source.read(entry)
            if entry.filename == "xl/worksheets/sheet1.xml":
                data = data.replace(b"<f>B3*2</f><v></v>", b"<f>B3*2</f><v>10</v>")
            destination.writestr(entry.filename, data)
    return cached.getvalue()


def test_xlsx_preserves_sheets_merged_cells_hidden_flags_and_typed_values(tmp_path, context):
    parsed = ParsingPipeline(tmp_path).parse(make_xlsx(), replace(context, filename="rates.xlsx"))
    first, hidden, empty = parsed.blocks
    assert [block.metadata["sheet_name"] for block in parsed.blocks] == ["Rates", "Hidden data", "Empty"]
    assert first.metadata["merged_ranges"] == ["A1:B2"]
    cell = first.metadata["table_cells"][0]
    assert (cell["id"], cell["row_span"], cell["column_span"]) == ("r0c0", 2, 2)
    cells = first.metadata["cell_origins"]
    assert first.metadata["hidden_rows"] == [3]
    assert (cells["r2c0"]["value"], cells["r2c0"]["coordinate"]) == ("0004", "A3")
    assert (cells["r2c1"]["value"], cells["r2c1"]["display"]) == (5, "0005")
    assert (cells["r2c2"]["formula"], cells["r2c2"]["cached_value"], cells["r2c2"]["value"]) == (
        "=B3*2", 10, None,
    )
    assert cells["r2c3"]["value"].startswith("2026-11-26")
    assert cells["r2c4"]["value"] is False
    assert hidden.metadata["sheet_state"] == "veryHidden"
    assert hidden.metadata["cell_origins"]["r19c0"]["coordinate"] == "A20"
    assert not empty.metadata["table_cells"]


def test_xlsx_validates_archive_and_actual_text_size(tmp_path, context):
    ctx = replace(context, filename="rates.xlsx")
    with pytest.raises(ParseError, match="XLSX"):
        ParsingPipeline(tmp_path).parse(b"not a workbook", replace(ctx))
    with pytest.raises(ParseError, match="лимит"):
        ParsingPipeline(tmp_path, limits=FileLimits(max_docx_uncompressed_bytes=200)).parse(make_xlsx(), replace(ctx))
    with pytest.raises(ParseError, match="лимит"):
        ParsingPipeline(tmp_path, limits=FileLimits(max_text_chars=8)).parse(make_xlsx(), ctx)


def test_xls_preserves_merges_legacy_values_and_semantic_limitations(tmp_path, context):
    xlwt = pytest.importorskip("xlwt")
    book = xlwt.Workbook()
    sheet = book.add_sheet("Rates")
    sheet.write_merge(0, 1, 0, 1, "Merged rate")
    sheet.write(2, 0, "0009")
    sheet.write(2, 1, 7, xlwt.easyxf(num_format_str="0000"))
    sheet.row(2).hidden = True
    book.add_sheet("Notes")
    output = io.BytesIO()
    book.save(output)
    parsed = ParsingPipeline(tmp_path).parse(output.getvalue(), replace(context, filename="rates.xls"))
    assert [block.metadata["sheet_name"] for block in parsed.blocks] == ["Rates", "Notes"]
    first = parsed.blocks[0]
    assert first.metadata["hidden_rows"] == [3]
    assert first.metadata["merged_ranges"] == ["A1:B2"]
    assert first.metadata["cell_origins"]["r2c0"]["value"] == "0009"
    assert first.metadata["cell_origins"]["r2c1"]["display"] == "0007"
    assert "xls_formula_source_unavailable" in parsed.warnings


def test_new_format_preflight_rejects_container_substitution(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    validator = FileValidator(allowed_extensions={".xlsx", ".xls", ".xml", ".csv"})
    for name, data in [("bad.xlsx", b"not zip"), ("bad.xls", b"not compound"), ("bad.csv", b"abc\x00")]:
        with pytest.raises(ParseError):
            validator.validate_and_store(io.BytesIO(data), name, name, storage)
        assert not (storage.root / name).exists()


def test_xml_preserves_order_attributes_paths_and_textless_elements(tmp_path, context):
    result = ParsingPipeline(tmp_path).parse(
        b"<doc locale='ru'>Before <item id='1'>one</item> after <item id='2'/> End.</doc>",
        replace(context, filename="markup.xml"),
    )
    assert result.context.mime_type == "application/xml"
    assert "".join(block.text for block in result.blocks) == "BeforeoneafterEnd."
    assert all(block.type == "TEXT" for block in result.blocks)
    first = result.blocks[0]
    assert first.metadata["xml_attributes"] == {"locale": "ru"}
    third = result.blocks[2]
    assert third.source.element_path == "/doc[1]/text()[2]"
    assert any(node["path"] == "/doc[1]/item[2]" and node["attributes"] == {"id": "2"}
               for node in first.metadata["xml_nodes"])


@pytest.mark.parametrize("payload", [b"<!DOCTYPE doc><doc>This needs no DTD</doc>", b"<doc><broken></doc>"])
def test_xml_rejects_forbidden_or_bad_document(tmp_path, context, payload):
    with pytest.raises(ParseError, match="XML"):
        ParsingPipeline(tmp_path).parse(payload, replace(context, filename="markup.xml"))
