"""Recover simple borderless grids from positioned native PDF words."""
import re
from dataclasses import asdict
from itertools import pairwise
from statistics import median

from .tables import Cell

CAPTION = re.compile(r"^(?:Table|Таблица)\s+\d+\b", re.IGNORECASE)
NUMBER = re.compile(r"^[+−-]?\s*\d+(?:[.,]\d+)*(?:\s*[%‰\w₽$€]+)?$")


def bounds(words):
    return [min(w["x0"] for w in words), min(w["top"] for w in words),
            max(w["x1"] for w in words), max(w["bottom"] for w in words)]


def borderless_table(words):
    lines = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not lines or abs(lines[-1][0]["top"] - word["top"]) > 3:
            lines.append([])
        lines[-1].append(word)
    lines = [sorted(line, key=lambda item: item["x0"]) for line in lines]
    captions = [i for i, line in enumerate(lines) if CAPTION.match(" ".join(w["text"] for w in line))]
    if len(captions) != 1:
        return None
    start = captions[0] + 1
    following = lines[start:]
    if len(following) < 4 or following[0][0]["top"] - lines[start - 1][0]["top"] > 100:
        return None
    step = following[1][0]["top"] - following[0][0]["top"]
    if not 8 <= step <= 52:
        return None
    selected = [following[0]]
    for previous, current in pairwise(following):
        gap = current[0]["top"] - previous[0]["top"]
        if gap > 1.6 * step:
            break
        if abs(gap - step) > max(3, .15 * step):
            return None
        selected.append(current)
    if not 4 <= len(selected) <= 100:
        return None
    rows = []
    for line in selected:
        groups = [[line[0]]]
        for previous, current in pairwise(line):
            if current["x0"] - previous["x1"] > 24:
                groups.append([])
            groups[-1].append(current)
        rows.append([(" ".join(w["text"] for w in group), bounds(group)) for group in groups])
    cols = len(rows[0])
    if not 2 <= cols <= 12 or any(len(row) != cols for row in rows):
        return None
    if any(not any(c.isalpha() for c in value) or NUMBER.fullmatch(value) for value, _ in rows[0]):
        return None
    for index, column in enumerate(zip(*rows, strict=True)):
        left = median(box[0] for _, box in column)
        if any(abs(box[0] - left) > 6 for _, box in column):
            return None
        if index and not all(NUMBER.fullmatch(value) for value, _ in column[1:]):
            return None
    # Inter-column space must remain clear across all rows, not just within each row.
    for index in range(cols - 1):
        if max(row[index][1][2] for row in rows) + 16 >= min(row[index + 1][1][0] for row in rows):
            return None
    cells = []
    for ri, row in enumerate(rows):
        cells.append([{"text": value, "bbox": box,
                       "cell": asdict(Cell(f"r{ri}c{ci}", ri, ci, value, header=ri == 0,
                                           scope="col" if ri == 0 else ""))}
                      for ci, (value, box) in enumerate(row)])
    return bounds([w for line in selected for w in line]), cells
