"""Page/region PDF processing; uncertain areas remain explicit and recoverable."""
import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from statistics import median

from bear_text_beta.models import SourceLocator
from bear_text_beta.parsing import BlockBuilder


def inside(obj, box):
    return (box[0] <= obj.get("x0", -1) and obj.get("x1", -1) <= box[2]
            and box[1] <= obj.get("top", -1) and obj.get("bottom", -1) <= box[3])


def accepted_table(table, page):
    rows = [[str(value or "") for value in row] for row in table.extract()]
    if len(rows) < 2 or max(map(len, rows), default=0) < 2:
        return None
    crop = page.crop(table.bbox)
    try:
        original = crop.extract_text() or ""
    finally:
        crop.close()
    # Text can have extra synthetic separators, but extracted characters cannot vanish.
    compact = lambda text: Counter(c for c in text if c.isalnum())
    if compact(original) - compact("".join("".join(row) for row in rows)):
        return None
    return rows


def read_tsv(data):
    lines, confidence, words = {}, [], []
    for row in csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"):
        text = row.get("text", "").strip()
        if not text:
            continue
        key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append(text)
        conf = float(row["conf"])
        confidence.append(conf)
        words.append({"text": text, "confidence": conf,
                      "bbox_pixels": [int(row[k]) for k in ("left", "top", "width", "height")]})
    return "\n".join(" ".join(line) for line in lines.values()), words, confidence


def text_regions(words, page_width):
    """Infer persistent gutters from word gaps; full-width lines delimit bands.

    This is a conservative layout heuristic, not a substitute for a layout model.
    Its inference remains visible in metadata/source-quality reports.
    """
    rows = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if not rows or abs(rows[-1][0]["top"] - word["top"]) > 3:
            rows.append([word])
        else:
            rows[-1].append(word)
    candidates = []
    for row in rows:
        row.sort(key=lambda w: w["x0"])
        for a, b in pairwise(row):
            gap = b["x0"] - a["x1"]
            mid = (a["x1"] + b["x0"]) / 2
            if gap >= page_width * .08 and page_width * .15 < mid < page_width * .85:
                candidates.append((mid, a["x1"], b["x0"]))
    clusters = []
    for candidate in sorted(candidates):
        if not clusters or candidate[0] - median(c[0] for c in clusters[-1]) > page_width * .06:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    gutters = [median(c[0] for c in cluster) for cluster in clusters if len(cluster) >= 2]
    inferred = bool(gutters)
    result = []
    for row in rows:
        parts, part = [], []
        for word in row:
            if (part and word["x0"] - part[-1]["x1"] >= page_width * .04
                    and any(part[-1]["x1"] < x < word["x0"] for x in gutters)):
                parts.append(part); part = []
            part.append(word)
        if part:
            parts.append(part)
        for part in parts:
            box = [min(w["x0"] for w in part), min(w["top"] for w in part),
                   max(w["x1"] for w in part), max(w["bottom"] for w in part)]
            column = sum(box[0] > x for x in gutters)
            full = any(box[0] < x < box[2] for x in gutters)
            result.append((box[1], box[0], "TEXT", "paragraph", " ".join(w["text"] for w in part),
                           {"bbox": box, "layout_column": column, "layout_full_width": full,
                            "reading_order": "column_gutters_inferred" if inferred else "single_column_unverified"}))
    return result, gutters, bool(candidates) and not inferred


def order_regions(regions, gutters):
    if not gutters:
        return sorted(regions, key=lambda r: (r[0], r[1]))
    ordered, pending = [], []
    def flush():
        ordered.extend(sorted(pending, key=lambda r: (sum(r[1] > x for x in gutters), r[0], r[1])))
        pending.clear()
    for region in sorted(regions, key=lambda r: (r[0], r[1])):
        box = region[5].get("bbox")
        full_width = region[5].get("layout_full_width") or (
            box is not None and any(box[0] < x < box[2] for x in gutters))
        if full_width:
            flush(); ordered.append(region)
        else:
            pending.append(region)
    flush()
    return ordered


def ocr_region(renderer, index, box=None):
    from PIL import ImageStat
    executable = os.environ.get("BEAR_TESSERACT") or shutil.which("tesseract")
    page = renderer[index]
    bitmap = picture = None
    try:
        width, height = page.get_size()
        dpi = int(os.environ.get("BEAR_OCR_DPI", "216"))
        if not 72 <= dpi <= 400:
            raise ValueError("OCR DPI must be between 72 and 400")
        scale = dpi / 72
        if width * height * scale**2 > 30_000_000:
            raise ValueError("PDF raster pixel limit exceeded")
        bitmap = page.render(scale=scale)
        picture = bitmap.to_pil().convert("RGB")
        if box:
            cropped = picture.crop(tuple(round(value * scale) for value in box))
            picture.close(); picture = cropped
        grey = picture.convert("L")
        try:
            stat = ImageStat.Stat(grey)
            blank = stat.mean[0] > 254 and stat.stddev[0] < .2
        finally:
            grey.close()
        if blank:
            return "", {"page_state": "blank"}
        if not executable:
            raise RuntimeError("OCR requires Tesseract with eng/rus")
        with tempfile.TemporaryDirectory(prefix="bear-ocr-") as directory:
            image = Path(directory) / "region.png"
            picture.save(image, dpi=(dpi, dpi))
            completed = subprocess.run([executable, str(image), "stdout", "-l", "eng+rus", "--psm", "3", "tsv"],
                                       capture_output=True, timeout=60, check=True)
        text, words, scores = read_tsv(completed.stdout)
        crop_box = list(box) if box else [0, 0, width, height]
        for word in words:
            left, top, word_width, word_height = word["bbox_pixels"]
            word["bbox_pdf_points"] = [crop_box[0] + left / scale, crop_box[1] + top / scale,
                                        crop_box[0] + (left + word_width) / scale,
                                        crop_box[1] + (top + word_height) / scale]
        return text, {"extraction": "ocr", "ocr_words": words, "ocr_scale": scale, "ocr_dpi": dpi,
                      "bbox": crop_box, "coordinate_system": "pdf_points_top_left",
                      "ocr_crop_box": crop_box, "page_size_points": [width, height],
                      "pixel_to_pdf_transform": [1 / scale, 0, 0, 1 / scale, crop_box[0], crop_box[1]],
                      "mean_word_confidence": sum(scores) / len(scores) if scores else None,
                      "numeric_review_required": bool(re.search(r"\d", text)),
                      "page_state": "needs_review" if text else "non_text_or_ocr_failure"}
    finally:
        if picture is not None:
            picture.close()
        if bitmap is not None:
            bitmap.close()
        page.close()


def parse_pdf(path, context, limits):
    import pdfplumber
    import pypdfium2 as pdfium
    builder = BlockBuilder(context)
    builder.warnings.append("pdf_layout_and_table_detection_unverified")
    with pdfplumber.open(path) as pdf, pdfium.PdfDocument(path) as renderer:
        if len(pdf.pages) > limits.max_pages:
            raise ValueError("PDF page limit exceeded")
        for index, page in enumerate(pdf.pages):
            source = SourceLocator("pdf", page=index + 1)
            try:
                words = page.extract_words()
                raw_text = page.extract_text() or ""
                regions, table_boxes = [], []
                for table in page.find_tables():
                    rows = accepted_table(table, page)
                    if rows:
                        regions.append((table.bbox[1], table.bbox[0], "TABLE", "table",
                                        "\n".join(" | ".join(row) for row in rows),
                                        {"table_rows": rows, "bbox": table.bbox, "header_inference": True}))
                        table_boxes.append(table.bbox)
                    else:
                        builder.warnings.append(f"pdf_table_candidate_rejected:page:{index + 1}")
                # Keep a source copy for every ordinary text line, and sort with accepted tables.
                filtered = page.filter(lambda obj, boxes=tuple(table_boxes): not any(inside(obj, box) for box in boxes))
                try:
                    lines, gutters, ambiguous = text_regions(filtered.extract_words(), page.width)
                    regions.extend(lines)
                    if gutters:
                        builder.warnings.append(f"pdf_reading_order_inferred:page:{index + 1}")
                    if ambiguous:
                        builder.warnings.append(f"pdf_reading_order_needs_review:page:{index + 1}")
                finally:
                    filtered.close()
                if len(re.findall(r"(?:Table|Таблица)\s*\d+\s*[:.]?", raw_text, re.IGNORECASE)) > len(table_boxes):
                    regions.append((0, 0, "TABLE", "table", "", {"unresolved": True, "reason": "caption_without_reconstructed_table"}))
                if not raw_text.strip() and not table_boxes:
                    targets = [None]
                else:
                    targets = []
                    for image in page.images:
                        box = [image["x0"], image["top"], image["x1"], image["bottom"]]
                        if (box[2] - box[0]) * (box[3] - box[1]) > 15000 and not any(inside(word, box) for word in words):
                            targets.append(box)
                for box in targets:
                    text, metadata = ocr_region(renderer, index, box)
                    if metadata.get("page_state") == "blank":
                        builder.warnings.append(f"pdf_blank_page:{index + 1}")
                    elif text:
                        regions.append((box[1] if box else 0, box[0] if box else 0, "TEXT", "paragraph", text, metadata))
                        builder.warnings.append(f"ocr_unverified:page:{index + 1}; numeric_and_layout_review_required")
                    else:
                        regions.append((0, 0, "IMAGE", "image", "", {**metadata, "unresolved": True, "bbox": box}))
                        builder.warnings.append(f"ocr_no_text:page:{index + 1}")
                for _, _, kind, role, text, metadata in order_regions(regions, gutters):
                    metadata.setdefault("coordinate_system", "pdf_points_top_left")
                    locator = replace(source, bbox=list(metadata["bbox"]) if metadata.get("bbox") else None)
                    builder.add(text, role, locator, kind=kind, metadata=metadata)
                if page.images and not targets:
                    builder.warnings.append(f"pdf_visual_content_not_interpreted:page:{index + 1}")
                if "(cid:" in raw_text:
                    builder.warnings.append(f"pdf_unknown_glyph:page:{index + 1}")
            except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                builder.add("", "image", source, kind="IMAGE", metadata={"unresolved": True, "page_error": str(exc)})
                builder.warnings.append(f"pdf_page_failed:{index + 1}:{type(exc).__name__}")
            finally:
                page.close()
    result = builder.finish()
    result.parser_profile = "fusion-pdf-cpu-v3"
    return result
