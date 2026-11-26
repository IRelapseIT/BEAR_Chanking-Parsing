"""Loss-preserving projections of delimited data and Excel workbooks."""

import csv
import io
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta

from lxml import etree

from .errors import ParseError, ServiceUnavailableError
from .file_validation import XLS_SIGNATURE, FileLimits, FileValidator
from .models import SourceLocator
from .parsing import BlockBuilder, decode_text

# This is a process-wide setting. Each document is additionally checked against its own limits.
csv.field_size_limit(max(csv.field_size_limit(), FileLimits().max_text_chars))


def cell_id(row, column):
    return f"r{row}c{column}"


def add_cell(cells, row, column, text, row_span=1, column_span=1):
    cells.append({"id": cell_id(row, column), "row": row, "column": column, "text": text,
                  "row_span": row_span, "column_span": column_span, "header": False, "scope": ""})


def add_table(builder, text, source, cells, origins, metadata):
    builder.add(text, "table", source, kind="TABLE", metadata={
        "table_cells": cells, "cell_origins": origins, "header_rows": 0,
        "header_confidence": "unknown", "cell_header_links": {}, **metadata,
    })


def csv_delimiter(text):
    try:
        return csv.Sniffer().sniff(text[:32768], delimiters=",;").delimiter
    except csv.Error:
        return ","


class DelimitedParser:
    def __init__(self, suffix, limits, allow_cp1251=False):
        self.format = suffix[1:]
        self.limits = limits
        self.allow_cp1251 = allow_cp1251

    def parse(self, data, context):
        text, warnings = decode_text(data, self.allow_cp1251)
        if len(text) > self.limits.max_text_chars:
            raise ParseError("Превышен лимит символов CSV/TSV")
        delimiter = "\t" if self.format == "tsv" else csv_delimiter(text)
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        cells, origins, rows = [], {}, []
        line_start = 1
        try:
            for ri, row in enumerate(reader):
                if ri >= self.limits.max_blocks or len(cells) + len(row) > self.limits.max_table_cells:
                    raise ParseError("Превышен лимит строк или ячеек CSV/TSV")
                rows.append(row)
                for ci, value in enumerate(row):
                    add_cell(cells, ri, ci, value)
                    origins[cell_id(ri, ci)] = {"row": ri + 1, "column": ci + 1, "line_start": line_start,
                                                "line_end": reader.line_num}
                line_start = reader.line_num + 1
        except csv.Error as exc:
            raise ParseError(f"Некорректный CSV/TSV около строки {reader.line_num}") from exc

        builder = BlockBuilder(context)
        builder.warnings.extend(warnings)
        if rows:
            if cells:
                builder.warnings.append("table_header_unconfirmed")
            if len({len(row) for row in rows if row}) > 1:
                builder.warnings.append("delimited_inconsistent_row_width")
            add_table(builder, "\n".join(" | ".join(row) for row in rows),
                      SourceLocator(self.format, 1, reader.line_num), cells, origins,
                      {"table_rows": rows, "delimiter": delimiter,
                       "encoding_policy": "cp1251" if warnings else "UTF-8 or Unicode BOM"})
        document = builder.finish()
        document.parser_profile = f"{self.format}-csv-reader-v1"
        return document


def json_value(value):
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def display_value(value, number_format):
    if value is None:
        return ""
    if (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and float(value).is_integer() and re.fullmatch(r"0{1,100}", number_format)):
        return ("-" if value < 0 else "") + f"{abs(int(value)):0{len(number_format)}d}"
    return str(json_value(value)).replace("\r\n", "\n").replace("\r", "\n")


class SheetProjection:
    def __init__(self, title, sheet_index, state, hidden_rows, merged_ranges, dimensions, maximum_chars):
        self.metadata = {"sheet_name": title, "sheet_index": sheet_index, "sheet_state": state,
                         "hidden_rows": hidden_rows, "merged_ranges": merged_ranges, "dimensions": dimensions}
        self.maximum_chars = maximum_chars
        self.text_chars = 0
        self.cells = []
        self.origins = {}

    def add(self, row, col, text, details, span):
        # The persisted provenance also carries source strings and formula/cache metadata.
        self.text_chars += len(text) + sum(len(value) for value in details.values() if isinstance(value, str))
        if self.text_chars > self.maximum_chars:
            raise ParseError("Книга превышает лимит материала ячеек")
        add_cell(self.cells, row, col, text, *span)
        self.origins[cell_id(row, col)] = {"row": row + 1, "column": col + 1, "display": text, **details}

    def finish(self, builder, source_format):
        rows = defaultdict(list)
        for cell in self.cells:
            rows[cell["row"]].append(cell["text"])
        text = "\n".join(" | ".join(values) for values in rows.values())
        source = SourceLocator(source_format, element_path=f"/sheets/{self.metadata['sheet_index']}")
        add_table(builder, text, source, self.cells, self.origins, self.metadata)


class XlsxParser:
    def __init__(self, limits):
        self.limits = limits

    def parse(self, data, context):
        try:
            from openpyxl import load_workbook
            from openpyxl.cell.cell import MergedCell
            from openpyxl.utils.exceptions import InvalidFileException
        except ImportError as exc:
            raise ServiceUnavailableError("Для чтения XLSX установите openpyxl") from exc
        FileValidator(self.limits).validate_xlsx(data)
        builder = BlockBuilder(context)
        material_budget = self.limits.max_text_chars
        try:
            # Both instances use sparse stored cells. A dense iteration would allocate cells up
            # to the last coordinate, including every blank cell in sparse or heavily formatted sheets.
            workbook = load_workbook(io.BytesIO(data), keep_links=False, rich_text=False)
            cached = load_workbook(io.BytesIO(data), data_only=True, keep_links=False)
            if len(workbook.sheetnames) > self.limits.max_blocks:
                raise ParseError("XLSX превышает лимит листов")
            formulas = False
            for index, name in enumerate(workbook.sheetnames, start=1):
                sheet = workbook[name]
                if not hasattr(sheet, "merged_cells"):
                    builder.warnings.append(f"xlsx_nonworksheet_sheet:{index}")
                    nonworksheet = SheetProjection(name, index, sheet.sheet_state, [], [], [], material_budget)
                    nonworksheet.finish(builder, "xlsx")
                    continue
                merged = sorted(sheet.merged_cells.ranges, key=lambda cell_range: (cell_range.min_row, cell_range.min_col))
                spans = {(r.min_row - 1, r.min_col - 1): (r.max_row - r.min_row + 1, r.max_col - r.min_col + 1)
                         for r in merged}
                projection = SheetProjection(name, index, sheet.sheet_state,
                    sorted(ri for ri, row in sheet.row_dimensions.items() if row.hidden),
                    [str(r) for r in merged], [sheet.max_row, sheet.max_column], material_budget)
                for (row, col), cell in sorted(sheet._cells.items()):
                    if isinstance(cell, MergedCell):
                        continue
                    value = cell.value
                    origin = {"coordinate": cell.coordinate, "number_format": cell.number_format,
                              "value_type": cell.data_type, "value": json_value(value)}
                    text = display_value(value, cell.number_format)
                    if cell.data_type == "f":
                        formulas = True
                        stored_cell = cached[name][cell.coordinate]
                        formula = value if isinstance(value, str) else dict(value)
                        origin.update(value=None, formula=formula, cached_value=json_value(stored_cell.value),
                                      cached_value_type=stored_cell.data_type)
                        text = (display_value(stored_cell.value, cell.number_format) if stored_cell.value is not None
                                else formula if isinstance(formula, str) else json.dumps(formula, ensure_ascii=False))
                    projection.add(row - 1, col - 1, text, origin, spans.get((row - 1, col - 1), (1, 1)))
                projection.finish(builder, "xlsx")
                material_budget -= projection.text_chars
            if formulas:
                builder.warnings.append("xlsx_formula_cache_not_recalculated")
        except (InvalidFileException, OSError, etree.XMLSyntaxError, ValueError, KeyError, TypeError) as exc:
            raise ParseError("Повреждённая или неподдерживаемая структура XLSX") from exc
        if any(block.text for block in builder.blocks):
            builder.warnings.append("table_header_unconfirmed")
        document = builder.finish()
        document.parser_profile = "xlsx-openpyxl-v1"
        return document


class XlsParser:
    def __init__(self, limits):
        self.limits = limits

    def parse(self, data, context):
        if not data.startswith(XLS_SIGNATURE):
            raise ParseError("Некорректная сигнатура контейнера XLS")
        try:
            import xlrd
            from openpyxl.utils.cell import get_column_letter
            from xlrd.compdoc import CompDocError
        except ImportError as exc:
            raise ServiceUnavailableError("Для чтения XLS установите xlrd и openpyxl") from exc
        value_types = {
            xlrd.XL_CELL_EMPTY: "empty", xlrd.XL_CELL_TEXT: "text", xlrd.XL_CELL_NUMBER: "number",
            xlrd.XL_CELL_DATE: "date", xlrd.XL_CELL_BOOLEAN: "boolean", xlrd.XL_CELL_ERROR: "error",
            xlrd.XL_CELL_BLANK: "blank",
        }
        builder = BlockBuilder(context)
        builder.warnings.extend(["xls_formula_source_unavailable", "xls_objects_not_extracted"])
        material_budget = self.limits.max_text_chars
        source_cells = 0
        try:
            book = xlrd.open_workbook(file_contents=data, formatting_info=True, on_demand=True, ragged_rows=True)
            if book.nsheets > self.limits.max_blocks:
                raise ParseError("XLS превышает лимит листов")
            for index in range(book.nsheets):
                sheet = book.sheet_by_index(index)
                spans = {(rlo, clo): (rhi - rlo, chi - clo) for rlo, rhi, clo, chi in sheet.merged_cells}
                merged = [f"{get_column_letter(clo + 1)}{rlo + 1}:{get_column_letter(chi)}{rhi}"
                          for rlo, rhi, clo, chi in sorted(sheet.merged_cells)]
                state = {0: "visible", 1: "hidden", 2: "veryHidden"}.get(sheet.visibility, "unknown")
                projection = SheetProjection(sheet.name, index + 1, state,
                    sorted(ri + 1 for ri, row in sheet.rowinfo_map.items() if row.hidden), merged,
                    [sheet.nrows, sheet.ncols], material_budget)
                source_cells += sum(height * width - 1 for height, width in spans.values())
                for row in range(sheet.nrows):
                    width = sheet.row_len(row)
                    source_cells += width
                    if source_cells > self.limits.max_table_cells:
                        raise ParseError("XLS превышает лимит ячеек")
                    for col in range(width):
                        cell = sheet.cell(row, col)
                        span = spans.get((row, col), (1, 1))
                        if cell.ctype in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK} and span == (1, 1):
                            continue
                        number_format = book.format_map[book.xf_list[cell.xf_index].format_key].format_str
                        value = cell.value
                        if cell.ctype == xlrd.XL_CELL_DATE:
                            value = xlrd.xldate_as_datetime(value, book.datemode)
                        elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                            value = bool(value)
                        elif cell.ctype == xlrd.XL_CELL_ERROR:
                            value = xlrd.error_text_from_code.get(value, str(value))
                        origin = {"coordinate": f"{get_column_letter(col + 1)}{row + 1}",
                                  "number_format": number_format, "value": json_value(value),
                                  "value_type": value_types[cell.ctype]}
                        projection.add(row, col, display_value(value, number_format), origin, span)
                projection.finish(builder, "xls")
                material_budget -= projection.text_chars
                book.unload_sheet(index)
            book.release_resources()
        except (xlrd.XLRDError, CompDocError, ValueError, IndexError, KeyError, TypeError) as exc:
            raise ParseError("Повреждённая или неподдерживаемая структура XLS") from exc
        if any(block.text for block in builder.blocks):
            builder.warnings.append("table_header_unconfirmed")
        document = builder.finish()
        document.parser_profile = "xls-xlrd-v1"
        return document
