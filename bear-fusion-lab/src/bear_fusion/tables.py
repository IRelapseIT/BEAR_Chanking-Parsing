"""Canonical cells, hierarchical headers and checked table fragments."""
import re
from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from bear_text_beta.chunking import ChunkContextBuilder, SafeTextSplitter
from bear_text_beta.models import PreparedFragment, SourceSpan
from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


@dataclass
class Cell:
    id: str
    row: int
    column: int
    text: str
    row_span: int = 1
    column_span: int = 1
    header: bool = False
    scope: str = ""


def place(expanded, cell):
    if min(cell.row_span, cell.column_span) < 1 or cell.row_span > 10000 or cell.column_span > 1000:
        raise ValueError("Invalid table span")
    for row in range(cell.row, cell.row + cell.row_span):
        for col in range(cell.column, cell.column + cell.column_span):
            if (row, col) in expanded:
                raise ValueError("Overlapping table cells")
            expanded[row, col] = cell
            if len(expanded) > 100000:
                raise ValueError("Expanded table exceeds cell budget")


def grid(cells):
    expanded = {}
    for cell in cells:
        place(expanded, cell)
    return expanded


def html_cells(raw):
    table = BeautifulSoup(raw, "lxml").find("table")
    if table is None:
        return [], 0
    cells, occupied, heads = [], {}, 0
    for ri, tr in enumerate(t for t in table.find_all("tr") if t.find_parent("table") is table):
        ci = 0
        tags = tr.find_all(["th", "td"], recursive=False)
        is_head = bool(tr.find_parent("thead")) or bool(tags) and all(t.name == "th" or not t.get_text(strip=True) for t in tags)
        if ri == heads and is_head:
            heads += 1
        for tag in tags:
            while (ri, ci) in occupied:
                ci += 1
            cell = Cell(f"r{ri}c{ci}", ri, ci, tag.get_text(" ", strip=True),
                        int(tag.get("rowspan", 1)), int(tag.get("colspan", 1)), tag.name == "th", tag.get("scope", ""))
            place(occupied, cell)
            cells.append(cell)
            ci += cell.column_span
    return cells, heads or (1 if cells else 0)


def rows_from_cells(cells):
    expanded = grid(cells)
    if not expanded:
        return []
    width = max(c for _, c in expanded) + 1
    height = max(r for r, _ in expanded) + 1
    return [[expanded[r, c].text if (r, c) in expanded else "" for c in range(width)] for r in range(height)]


def html_rows(raw):
    return rows_from_cells(html_cells(raw)[0])


def canonical(block):
    if block.metadata.get("unresolved"):
        return [], 0
    if "table_cells" in block.metadata:
        return [Cell(**cell) for cell in block.metadata["table_cells"]], block.metadata["header_rows"]
    rows = block.metadata.get("table_rows")
    if rows is not None:
        cells = [Cell(f"r{ri}c{ci}", ri, ci, str(value or "")) for ri, row in enumerate(rows) for ci, value in enumerate(row)]
        heads = int(block.metadata.get("header_rows", 1))
    else:
        raw = block.metadata.get("raw_source", block.text)
        if "<table" not in raw.lower():
            raw = MarkdownIt("commonmark").enable("table").render(raw)
        cells, heads = html_cells(raw)
    if sum(len(cell.text) for cell in cells) > 10_000_000:
        raise ValueError("Table text exceeds budget")
    grid(cells)
    block.metadata.update(table_cells=[asdict(cell) for cell in cells], header_rows=heads, table_contract="cells-v1",
                          header_inference=not any(cell.header and cell.row < heads for cell in cells))
    return cells, heads


def restore_docx_tables(path, document):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    nodes = root.findall(".//w:tbl", ns)
    blocks = [b for b in document.blocks if b.type == "TABLE"]
    if len(nodes) != len(blocks):
        raise ValueError("DOCX table mapping ambiguous")
    for block, table in zip(blocks, nodes, strict=True):
        cells, header_rows, occupied = [], 0, {}
        for ri, row in enumerate(table.findall("w:tr", ns)):
            repeat = row.find("w:trPr/w:tblHeader", ns)
            if ri == header_rows and repeat is not None and repeat.get(f"{{{ns['w']}}}val", "1") not in {"0", "false", "off"}:
                header_rows += 1
            before = row.find("w:trPr/w:gridBefore", ns)
            ci = int(before.get(f"{{{ns['w']}}}val", "0")) if before is not None else 0
            for node in row.findall("w:tc", ns):
                text = "\n".join("".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in node.findall("w:p", ns))
                width = node.find("w:tcPr/w:gridSpan", ns)
                span = int(width.get(f"{{{ns['w']}}}val", "1")) if width is not None else 1
                merge = node.find("w:tcPr/w:vMerge", ns)
                above = occupied.get((ri - 1, ci))
                if merge is not None and merge.get(f"{{{ns['w']}}}val", "continue") == "continue":
                    if above is None or above.column != ci or above.column_span != span or text.strip():
                        raise ValueError("Ambiguous DOCX vertical merge; refusing loss")
                    above.row_span += 1
                    for col in range(ci, ci + span):
                        occupied[ri, col] = above
                else:
                    cell = Cell(f"r{ri}c{ci}", ri, ci, text, column_span=span)
                    place(occupied, cell); cells.append(cell)
                ci += span
        block.metadata.update(table_cells=[asdict(cell) for cell in cells], header_rows=header_rows or 1,
                              table_rows=rows_from_cells(cells), table_source="docx-ooxml",
                              merge_geometry="gridSpan-vMerge", header_inference=not bool(header_rows))
        block.text = "\n".join(" | ".join(row) for row in block.metadata["table_rows"])
        if not header_rows:
            document.warnings.append("table_header_inferred")


def context_for(cell, expanded, heads):
    headers = []
    for row in range(heads):
        header = expanded.get((row, cell.column))
        if header and header.id != cell.id and header.text and header.text not in headers:
            headers.append(header.text)
    first = expanded.get((cell.row, 0))
    row_labels = []
    for col in range(cell.column):
        label = expanded.get((cell.row, col))
        if label and label.scope in {"row", "rowgroup"} and label.text and label.text not in row_labels:
            row_labels.append(label.text)
    if not row_labels and first and first.row >= heads and first.id != cell.id and any(c.isalpha() for c in first.text):
        row_labels.append(first.text)
    identity = " → ".join(row_labels)
    return f"Row {cell.row + 1}" + (f" ({identity})" if identity else "") + f"; {' → '.join(headers) or f'Column {cell.column + 1}'}: "


def protected_context(prefix):
    pattern = (r"(?<!\d)(?:19|20)\d{2}(?!\d)|"
               r"(?i:(?<![a-zа-я])(?:RUB|USD|EUR|GBP|руб(?:лей|ля)?|млн|тыс|million|billion|kg|кг|%)(?![a-zа-я]))")
    return list(dict.fromkeys(re.findall(pattern, prefix)))


def bounded_prefix(prefix, fits):
    if fits(prefix + "x" * 32):
        return prefix
    # Keep the leaf qualifier/year/unit at the end, not an arbitrary prefix.
    # The full context remains in the proof metadata and is searchable separately.
    stem = prefix.removesuffix(": ")
    tail = stem.rsplit("; ", 1)[-1]
    protected = protected_context(prefix)
    while tail:
        missing = [item for item in protected if item not in tail]
        label = "[" + "; ".join(missing) + "] " if missing else ""
        shortened = "… " + label + tail + ": "
        if fits(shortened + "x" * 32):
            return shortened
        words = tail.split(" ", 1)
        tail = words[-1] if len(words) > 1 else ""
    raise ValueError("Table context does not fit model budget")


class TableProcessor:
    def __init__(self, counter=None):
        self.counter, self.budget = counter, 320

    def validate(self, document, chunks, counter, maximum):
        validate_table_chunks(document, chunks, counter, maximum)

    def process(self, block, context):
        cells, heads = canonical(block)
        expanded, fragments = grid(cells), []
        for cell in cells:
            if not cell.text.strip():
                continue
            full_context = "Table headers: " if cell.row < heads else context_for(cell, expanded, heads)
            fits = (lambda text: self.counter.full_count(ChunkContextBuilder.embedding_text(text, block.section_path, self.counter)) <= self.budget) if self.counter else lambda text: len(text) <= 300
            prefix = bounded_prefix(full_context, fits)
            truncated = prefix != full_context
            if truncated:
                block.metadata["header_context_truncated"] = True
            for start, end in SafeTextSplitter(lambda value, fits=fits, prefix=prefix: fits(prefix + value)).split(cell.text):
                fragments.append(PreparedFragment(prefix + cell.text[start:end], [SourceSpan(block.id, 0, len(block.text))],
                    metadata={"table_block_id": block.id, "cell_id": cell.id, "value_start": start,
                              "value_end": end, "prefix": prefix, "table_contract": "cells-v1",
                              "header_context_full": full_context,
                              "header_context_truncated": truncated}))
        # Pack a row when it fits; long cells already have bounded, independently proven slices.
        packed, pending, pending_row = [], [], None
        by_id = {cell.id: cell for cell in cells}
        def flush():
            if len(pending) == 1:
                packed.append(pending[0])
            elif pending:
                packed.append(PreparedFragment("\n".join(f.text for f in pending), pending[0].source_spans,
                    metadata={"table_block_id": block.id, "table_contract": "cells-v2",
                              "cell_segments": [f.metadata for f in pending]}))
            pending.clear()
        for fragment in fragments:
            row = by_id[fragment.metadata["cell_id"]].row
            if row < heads:
                flush(); packed.append(fragment)
                continue
            if pending and (row != pending_row or not fits("\n".join([*(f.text for f in pending), fragment.text]))):
                flush()
            pending.append(fragment)
            pending_row = row
        flush()
        return packed


def validate_table_chunks(document, chunks, counter, budget):
    for block in document.blocks:
        if block.type != "TABLE" or block.metadata.get("unresolved"):
            continue
        cells, heads = canonical(block)
        expanded, by_id = grid(cells), {cell.id: cell for cell in cells}
        coverage = {cell.id: [] for cell in cells if cell.text.strip()}
        for chunk in chunks:
            if not any(s.block_id == block.id for s in chunk.source_spans):
                continue
            metas = chunk.metadata.get("cell_segments") if chunk.metadata.get("table_contract") == "cells-v2" else [chunk.metadata]
            if not isinstance(metas, list) or not metas:
                raise ValueError("Table fragment has no cell segments")
            expected_parts = []
            segment_rows = set()
            for meta in metas:
                cell = by_id.get(meta.get("cell_id"))
                if cell is None or meta.get("table_contract") != "cells-v1":
                    raise ValueError("Table fragment has no canonical cell proof")
                segment_rows.add(cell.row)
                start, end = meta.get("value_start"), meta.get("value_end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(cell.text):
                    raise ValueError("Invalid table cell span")
                expected = "Table headers: " if cell.row < heads else context_for(cell, expanded, heads)
                prefix = meta.get("prefix")
                if not isinstance(prefix, str):
                    raise TypeError("Table fragment changes header")
                if (meta.get("header_context_full") != expected
                        or meta.get("header_context_truncated") is not (prefix != expected)):
                    raise ValueError("Table fragment has no full header context proof")
                if prefix != expected:
                    # An independent invariant: terminal semantic qualifier and delimiter
                    # must remain even if the surrounding description is shortened.
                    qualifier = expected.removesuffix(": ").split()[-1]
                    tail = prefix[2:].removesuffix(": ")
                    label = ""
                    if tail.startswith("[") and "] " in tail:
                        label, tail = tail.split("] ", 1)
                        label += "] "
                    missing = [item for item in protected_context(expected) if item not in tail]
                    expected_label = "[" + "; ".join(missing) + "] " if missing else ""
                    if (not prefix.startswith("… ") or not prefix.endswith(qualifier + ": ")
                            or not expected.removesuffix(": ").endswith(tail)
                            or label != expected_label):
                        raise ValueError("Table fragment loses terminal header qualifier")
                expected_parts.append(prefix + cell.text[start:end])
                coverage[cell.id].append((start, end))
            if len(segment_rows) > 1:
                raise ValueError("Packed table fragment mixes different rows")
            if chunk.text != "\n".join(expected_parts):
                raise ValueError("Table fragment changes source value")
            if counter.full_count(chunk.embedding_text) > budget:
                raise ValueError("Table fragment exceeds model budget")
        for cell_id, spans in coverage.items():
            position = 0
            for start, end in sorted(spans):
                if start != position:
                    raise ValueError("Table cell gap or duplicate")
                position = end
            if position != len(by_id[cell_id].text):
                raise ValueError("Table cell not fully covered")
