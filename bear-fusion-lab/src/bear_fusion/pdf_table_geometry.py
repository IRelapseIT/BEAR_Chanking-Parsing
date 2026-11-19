"""Infer a table grid from stable whitespace across the captioned rows."""
import re
from itertools import pairwise
from statistics import median

from .table_layout import NUMBER, layout_from_boxes, native_word

CAPTION = re.compile(r"^(?:Table|Таблица)\s+\d+\b", re.IGNORECASE)


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
    if len(following) < 3 or following[0][0]["top"] - lines[start - 1][0]["top"] > 100:
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
    if not 3 <= len(selected) <= 100:
        return None
    height = median(w["bottom"] - w["top"] for line in selected for w in line)
    header = selected[0]
    gaps = [(p["x1"], q["x0"]) for p, q in pairwise(header) if q["x0"] - p["x1"] >= max(12, 1.4 * height)]
    if not 1 <= len(gaps) <= 11:
        return None
    # Intersect each header gap with whitespace across the body. Keeping the
    # whole common interval allows indentation and left/right aligned values.
    for line in selected[1:]:
        free = [(-float("inf"), line[0]["x0"]),
                *((p["x1"], q["x0"]) for p, q in pairwise(line)),
                (line[-1]["x1"], float("inf"))]
        narrowed = []
        for left, right in gaps:
            options = [(max(left, a), min(right, b)) for a, b in free
                       if min(right, b) - max(left, a) >= max(4, .4 * height)]
            if len(options) != 1:
                return None
            narrowed.append(options[0])
        gaps = narrowed
    flattened = [w for line in selected for w in line]
    xs = [min(w["x0"] for w in flattened), *((a + b) / 2 for a, b in gaps), max(w["x1"] for w in flattened)]
    ys = [min(w["top"] for w in selected[0])]
    for previous, current in pairwise(selected):
        bottom, top = max(w["bottom"] for w in previous), min(w["top"] for w in current)
        if bottom >= top:
            return None
        ys.append((bottom + top) / 2)
    ys.append(max(w["bottom"] for w in selected[-1]))
    boxes = [[x0, y0, x1, y1] for y0, y1 in pairwise(ys) for x0, x1 in pairwise(xs)]
    layout = layout_from_boxes(boxes, [native_word(w) for w in flattened], "text_gutters")
    if layout is None or layout["header_rows"] != 1 or layout["unresolved_header_cells"]:
        return None
    rows = layout["table_rows"]
    # Keep false positives low for the caption-based inference.
    for col in range(1, len(xs) - 1):
        values = [row[col] for row in rows[1:] if row[col]]
        if not values or not all(NUMBER.fullmatch(value) for value in values):
            return None
    return layout
