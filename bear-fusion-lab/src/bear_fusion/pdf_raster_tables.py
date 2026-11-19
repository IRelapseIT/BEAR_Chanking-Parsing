"""Find fully connected ruled tables in an OCR raster."""
import re
from collections import defaultdict
from statistics import median

from pdfplumber.table import cells_to_tables, edges_to_intersections, intersections_to_cells

from .ocr_verification import pdf_box
from .table_layout import layout_from_boxes, overlap


def segments(image, minimum):
    runs = re.compile(rb"\xff{%d,}" % minimum)
    pixels = image.tobytes()
    width, height = image.size
    active, found = {}, []
    for y in range(height):
        current = {}
        for run in runs.finditer(pixels[y * width:(y + 1) * width]):
            start, stop = run.span()
            # Nearby scan lines belonging to the same thick stroke should
            # contribute one boundary, preserving its endpoints.
            key = (round(start / 4), round(stop / 4))
            current[key] = active.pop(key, []) + [(y, start, stop - 1)]
        found.extend(active.values())
        if len(found) > 120:
            return []
        active = current
    found.extend(active.values())
    # Different scan lines of one stroke may quantize to different keys.
    bands = defaultdict(list)
    for lines in found:
        coord, start, end = [median(values) for values in zip(*lines)]
        similar = next((key for key in bands if abs(key[0] - coord) <= 3
                        and abs(key[1] - start) <= 5 and abs(key[2] - end) <= 5), None)
        bands[similar or (coord, start, end)].extend(lines)
    strokes = sorted(tuple(median(values) for values in zip(*lines)) for lines in bands.values())
    aligned = []
    for stroke in strokes:
        if not aligned or stroke[0] - aligned[-1][0][0] > 2:
            aligned.append([])
        aligned[-1].append(stroke)
    merged = []
    for group in aligned:
        coord = median(stroke[0] for stroke in group)
        runs = sorted((start, end) for _, start, end in group)
        start, end = runs[0]
        for left, right in runs[1:]:
            if left > end + 1:
                merged.append((coord, start, end))
                start, end = left, right
            else:
                end = max(end, right)
        merged.append((coord, start, end))
    return merged


def raster_tables(image, transform, words):
    from PIL import Image

    with image.convert("L") as gray:
        width, height = gray.size
        gray.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        with gray.point(lambda v: 255 if v < 145 else 0) as mask, mask.transpose(Image.Transpose.TRANSPOSE) as vertical:
            h = segments(mask, max(24, mask.width // 24))
            v = segments(vertical, max(24, mask.height // 24))
            sx, sy = width / mask.width, height / mask.height
    if not h or not v or len(h) * len(v) > 3600:
        return []
    edges = [{"orientation": "h", "x0": start, "x1": end, "top": y, "bottom": y}
             for y, start, end in h]
    edges.extend({"orientation": "v", "x0": x, "x1": x, "top": start, "bottom": end}
                 for x, start, end in v)
    intersections = edges_to_intersections(edges, 3, 3)
    if len(intersections) > 500:
        return []
    layouts = []
    for cells in cells_to_tables(intersections_to_cells(intersections)):
        boxes = [pdf_box([a * sx, b * sy, c * sx, d * sy], transform) for a, b, c, d in cells]
        layout = layout_from_boxes(boxes, words, "raster_rules")
        if layout and sum(bool(c["text"]) for c in layout["table_cells"]) >= 3:
            if any(overlap(layout["bbox"], existing["bbox"]) for existing in layouts):
                return []
            layouts.append(layout)
    return layouts
