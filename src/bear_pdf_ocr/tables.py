from dataclasses import dataclass


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


def rows_from_cells(cells):
    expanded = grid(cells)
    if not expanded:
        return []
    width = max(c for _, c in expanded) + 1
    height = max(r for r, _ in expanded) + 1
    return [[expanded[r, c].text if (r, c) in expanded else "" for c in range(width)] for r in range(height)]
