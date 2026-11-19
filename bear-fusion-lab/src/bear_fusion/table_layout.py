"""Build and verify table grids in PDF coordinates, independently of the parser."""
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict
from math import floor, isfinite
from statistics import median

from .tables import Cell, grid

ROW_LABEL_HEADERS = {"year", "год", "item", "name", "наименование", "товар", "показатель",
                     "period", "период", "date", "дата", "№", "id"}
NUMBER = re.compile(r"^(?:[₽$€]\s*)?[+−-]?\s*\d+(?:[., ]\d+)*(?:\s*[%‰\w₽$€]+)?$")


def bounds(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def native_word(word):
    return {"text": word["text"], "bbox_pdf_points": [word["x0"], word["top"], word["x1"], word["bottom"]]}


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def valid_box(box):
    return box and len(box) == 4 and all(map(isfinite, box)) and area(box) > 0


def assign_words(boxes, words):
    """A source word belongs to one cell, or the candidate is ambiguous."""
    if not boxes or not all(valid_box(b) for b in boxes.values()):
        return None
    envelope = bounds(list(boxes.values()))
    # Vertical buckets avoid checking every cell of a large table for each word.
    # Scale buckets to the geometry so even very tall PDF pages stay bounded.
    size = max(12, (envelope[3] - envelope[1]) / 128)
    buckets = defaultdict(list)
    for key, box in boxes.items():
        for band in range(floor(box[1] / size), floor(box[3] / size) + 1):
            buckets[band].append(key)
    assigned = {key: [] for key in boxes}
    for word in words:
        box = word["bbox_pdf_points"]
        if not valid_box(box) or not overlap(box, envelope):
            continue
        candidates = {key for band in range(floor(max(box[1], envelope[1]) / size),
                                            floor(min(box[3], envelope[3]) / size) + 1)
                      for key in buckets[band]}
        hits = [(overlap(box, boxes[key]) / area(box), key) for key in candidates]
        hits = sorted((hit for hit in hits if hit[0] > 0), reverse=True)
        if not hits:
            # The candidate claims no cell for a word in its bounding rectangle.
            if overlap(box, envelope) / area(box) >= .5:
                return None
            continue
        if hits[0][0] < .80 or (len(hits) > 1 and hits[1][0] >= .15):
            return None
        assigned[hits[0][1]].append(word)
    return assigned


def word_text(words):
    if not words:
        return ""
    height = median(w["bbox_pdf_points"][3] - w["bbox_pdf_points"][1] for w in words)
    lines = []
    for word in sorted(words, key=lambda w: sum(w["bbox_pdf_points"][1::2])):
        center = sum(word["bbox_pdf_points"][1::2]) / 2
        if not lines or center - lines[-1][0] > height * .5:
            lines.append((center, []))
        lines[-1][1].append(word)
    return "\n".join(" ".join(w["text"] for w in sorted(line, key=lambda w: w["bbox_pdf_points"][0]))
                     for _, line in lines)


def header_links(cells, row_headers=()):
    """Resolve paths per logical column, requiring a common path for a span."""
    explicit_rows = set(row_headers)
    headers = sorted((c for c in cells if c.header and c.text and c.scope != "row"),
                     key=lambda c: (c.row, c.column))
    by_column = defaultdict(list)
    by_row = defaultdict(list)
    for header in headers:
        for column in range(header.column, header.column + header.column_span):
            by_column[column].append(header)
    first_column_data = [c for c in cells if c.column == 0 and c.column_span == 1 and not c.header and c.text]
    infer_rows = not explicit_rows and (
        any(c.column == 0 and c.column_span == 1 and c.text.casefold() in ROW_LABEL_HEADERS for c in headers)
        or bool(first_column_data) and all(any(ch.isalpha() for ch in c.text)
                                          and not NUMBER.fullmatch(c.text) for c in first_column_data))
    inferred_ids = {c.id for c in first_column_data} if infer_rows else set()
    for cell in cells:
        if cell.id in explicit_rows or cell.id in inferred_ids:
            for row in range(cell.row, cell.row + cell.row_span):
                by_row[row].append(cell)
    links, uncertain = {}, []
    for cell in cells:
        if cell.header or cell.id in explicit_rows:
            continue
        paths = {tuple(h.id for h in by_column[column] if h.row + h.row_span <= cell.row)
                 for column in range(cell.column, cell.column + cell.column_span)}
        column_ids = list(next(iter(paths))) if len(paths) == 1 else []
        labels = [r.id for r in by_row[cell.row] if r.id != cell.id and r.text
                  and r.column + r.column_span <= cell.column
                  and r.row + r.row_span >= cell.row + cell.row_span]
        links[cell.id] = {"column_header_ids": column_ids, "row_header_ids": labels}
        if cell.text and not column_ids:
            uncertain.append(cell.id)
    return links, uncertain


def infer_headers(cells):
    by_row = defaultdict(list)
    for cell in cells:
        if cell.text:
            by_row[cell.row].append(cell)
    # Header inference needs a visible transition from labels to data. With
    # all-text rows the structure is still useful, but header roles are unknown.
    for row in range(min(4, max(by_row, default=0) + 1)):
        row_cells = by_row[row]
        if row_cells and any(c.column > 0 and NUMBER.fullmatch(c.text) for c in row_cells):
            if row > 1 and not any(c.row < row - 1 and c.column_span > 1 for c in cells):
                return 0
            return row
        if not row_cells or not all(any(ch.isalpha() for ch in c.text) for c in row_cells):
            return 0
    return 0


def logical_order_matches(cells, boxes):
    for axis, start, span in ((0, "column", "column_span"), (1, "row", "row_span")):
        ending = sorted(cells, key=lambda c: getattr(c, start) + getattr(c, span))
        ends, maximums = [], []
        for cell in ending:
            ends.append(getattr(cell, start) + getattr(cell, span))
            physical_end = boxes[cell.id][axis + 2]
            maximums.append(max(physical_end, maximums[-1]) if maximums else physical_end)
        for cell in cells:
            index = bisect_right(ends, getattr(cell, start)) - 1
            if index >= 0 and maximums[index] > boxes[cell.id][axis] + 1:
                return False
    return True


def layout_from_boxes(boxes, words, method):
    if not 4 <= len(boxes) <= 5000 or not all(valid_box(b) for b in boxes):
        return None
    boxes = [list(box) for box in boxes]
    xs = sorted({x for b in boxes for x in b[::2]})
    ys = sorted({y for b in boxes for y in b[1::2]})
    if min(len(xs), len(ys)) < 3 or (len(xs) - 1) * (len(ys) - 1) > 100_000:
        return None
    if min(b - a for a, b in zip(xs, xs[1:])) < 1 or min(b - a for a, b in zip(ys, ys[1:])) < 1:
        return None
    cells, cell_boxes = [], {}
    for x0, y0, x1, y1 in sorted(boxes, key=lambda box: (box[1], box[0])):
        r, c = bisect_left(ys, y0), bisect_left(xs, x0)
        cell = Cell(f"r{r}c{c}", r, c, "", bisect_left(ys, y1) - r, bisect_left(xs, x1) - c)
        cells.append(cell)
        cell_boxes[cell.id] = [x0, y0, x1, y1]
    try:
        if len(grid(cells)) != (len(xs) - 1) * (len(ys) - 1):
            return None
    except ValueError:
        return None
    assigned = assign_words(cell_boxes, words)
    if assigned is None:
        return None
    ink = {}
    rows = [[""] * (len(xs) - 1) for _ in range(len(ys) - 1)]
    for cell in cells:
        seen = assigned[cell.id]
        if seen:
            ink[cell.id] = bounds([w["bbox_pdf_points"] for w in seen])
            cell.text = word_text(seen)
            rows[cell.row][cell.column] = cell.text
    if not ink:
        return None
    heads = infer_headers(cells)
    for cell in cells:
        if cell.row + cell.row_span <= heads:
            cell.header, cell.scope = True, "col"
    links, uncertain = header_links(cells)
    return {"bbox": bounds(boxes), "table_cells": [asdict(c) for c in cells],
            "table_rows": rows, "cell_bboxes": cell_boxes, "cell_content_bboxes": ink,
            "cell_header_links": links, "unresolved_header_cells": uncertain,
            "header_rows": heads, "header_inference": True, "table_contract": "cells-v1",
            "table_geometry_source": method,
            "table_geometry_inferred": method == "text_gutters",
            "values_verified": method == "vector_rules"}


def table_region(layout):
    box = layout["bbox"]
    text = "\n".join(" | ".join(row).replace("\n", " ") for row in layout["table_rows"])
    return box[1], box[0], "TABLE", "table", text, layout
