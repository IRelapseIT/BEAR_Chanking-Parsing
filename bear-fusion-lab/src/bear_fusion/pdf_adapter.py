"""Page/region PDF processing; uncertain areas remain explicit and recoverable."""
import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from statistics import median

from .field_routing import check_fields, image_text_gaps, intersects, merge_boxes
from bear_text_beta.models import SourceLocator
from bear_text_beta.parsing import BlockBuilder
from bear_text_beta.visual_context import attach_visual_context, recognize_image

from .ocr_verification import PaddleVerifier, pdf_box, primary_lines
from .raster import raster_region
from .pdf_table_geometry import borderless_table
from .pdf_raster_tables import raster_tables
from .table_layout import layout_from_boxes, native_word, table_region


def inside(obj, box):
    return (box[0] <= obj.get("x0", -1) and obj.get("x1", -1) <= box[2]
            and box[1] <= obj.get("top", -1) and obj.get("bottom", -1) <= box[3])


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
                      "line_id": key,
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


def damaged(text):
    return bool(re.search(r"\(cid:\d+\)|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", text))


def word_box(word):
    return [word["x0"], word["top"], word["x1"], word["bottom"]]


def ocr_region(renderer, index, box=None, *, visual_only=False, native=()):
    executable = os.environ.get("BEAR_TESSERACT") or shutil.which("tesseract")
    with raster_region(renderer, index, box) as (picture, transform, dpi, size):
        if visual_only:
            return recognize_image(picture)
        # Page blankness includes visible rules and graphics. The rule-removing
        # heuristic is used only for field crops.
        with picture.convert("L") as gray:
            if gray.getextrema()[0] >= 250:
                return "", {"page_state": "blank"}
        if not executable:
            raise RuntimeError("OCR requires Tesseract with eng/rus")
        crop_box = pdf_box([0, 0, *picture.size], transform)
        with tempfile.TemporaryDirectory(prefix="bear-ocr-") as directory:
            image = Path(directory) / "region.png"
            picture.save(image, dpi=(dpi, dpi))
            completed = subprocess.run([executable, str(image), "stdout", "-l", "eng+rus", "--psm", "3", "tsv"],
                                       capture_output=True, timeout=60, check=True)
            _, words, _ = read_tsv(completed.stdout)
            for word in words:
                left, top, width, height = word["bbox_pixels"]
                word["bbox_pdf_points"] = pdf_box([left, top, left + width, top + height], transform)
            def duplicates_native(word):
                x0, y0, x1, y1 = word["bbox_pdf_points"]
                return any(intersects(candidate, [x0, y0, x1, y1])
                           and candidate[0] <= (x0 + x1) / 2 <= candidate[2]
                           and candidate[1] <= (y0 + y1) / 2 <= candidate[3] for candidate in native)

            words = [word for word in words if not duplicates_native(word)]
            layouts = raster_tables(picture, transform, words) if words else []
            text = "\n".join(line["text"] for line in primary_lines(words, transform))
            scores = [word["confidence"] for word in words]
            return text, {"extraction": "ocr", "engine": "tesseract", "ocr_words": words,
                          "ocr_scale": dpi / 72, "ocr_dpi": dpi, "bbox": crop_box,
                          "coordinate_system": "pdf_points_top_left", "ocr_crop_box": crop_box,
                          "page_size_points": list(size), "pixel_to_pdf_transform": transform,
                          "ocr_tables": layouts,
                          "mean_word_confidence": sum(scores) / len(scores) if scores else None,
                          "numeric_review_required": bool(re.search(r"\d", text)),
                          "page_state": "needs_review" if text else "non_text_or_ocr_failure"}


def _parse_pdf_primary(path, context, limits, *, verifier=None, page_inputs=None):
    import pdfplumber
    import pypdfium2 as pdfium
    builder = BlockBuilder(context)
    visual_ocr_budget = 8
    builder.warnings.append("pdf_layout_and_table_detection_unverified")
    with pdfplumber.open(path) as pdf, pdfium.PdfDocument(path) as renderer:
        if len(pdf.pages) > limits.max_pages:
            raise ValueError("PDF page limit exceeded")
        for index, page in enumerate(pdf.pages):
            source = SourceLocator("pdf", page=index + 1)
            start = len(builder.blocks)
            raw_text = ""
            info = {"width": float(page.width), "height": float(page.height),
                    "rotation": page.rotation, "native": [], "ocr": [], "ocr_attempts": [], "text_layer": "absent"}
            if page_inputs is not None:
                page_inputs[index + 1] = info
            try:
                words = page.extract_words()
                raw_text = page.extract_text() or ""
                usable = [w for w in words if not damaged(w["text"])]
                broken = [w for w in words if damaged(w["text"])]
                info["native"] = [{"text": w["text"], "bbox_pdf_points": word_box(w),
                                   "engine": "pdfplumber"} for w in usable]
                info["text_layer"] = "damaged" if damaged(raw_text) else "usable" if usable else "absent"
                if len(re.sub(r"\W", "", raw_text)) < 15 and any(
                        (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) > 15000 for im in page.images):
                    builder.warnings.append(f"pdf_sparse_text_visual:page:{index + 1}")
                regions, table_boxes, table_ink = [], [], []
                observations = [native_word(w) for w in words]
                for table in page.find_tables():
                    layout = layout_from_boxes(table.cells, observations, "vector_rules")
                    if layout is None or any(damaged(c["text"]) for c in layout["table_cells"]):
                        builder.warnings.append(f"pdf_table_candidate_rejected:page:{index + 1}")
                        continue
                    regions.append(table_region(layout))
                    table_boxes.append(layout["bbox"])
                    x0, y0, x1, y1 = layout["bbox"]
                    table_ink.extend(box for box in layout["cell_content_bboxes"].values()
                                     if box[0] < x0 or box[1] < y0 or box[2] > x1 or box[3] > y1)
                    if layout["unresolved_header_cells"]:
                        builder.warnings.append(f"pdf_table_headers_unconfirmed:page:{index + 1}")
                if not table_boxes and not page.images:
                    layout = borderless_table(usable)
                    if layout is not None:
                        regions.append(table_region(layout))
                        table_boxes.append(layout["bbox"])
                        builder.warnings.append(f"pdf_table_geometry_unverified:page:{index + 1}")
                # Keep a source copy for every ordinary text line, and sort with accepted tables.
                excluded = (*table_boxes, *table_ink)
                filtered = page.filter(lambda obj: not any(inside(obj, box) for box in excluded))
                try:
                    lines, gutters, ambiguous = text_regions([w for w in filtered.extract_words()
                                                                if not damaged(w["text"])], page.width)
                    regions.extend(lines)
                    if gutters:
                        builder.warnings.append(f"pdf_reading_order_inferred:page:{index + 1}")
                    if ambiguous:
                        builder.warnings.append(f"pdf_reading_order_needs_review:page:{index + 1}")
                finally:
                    filtered.close()
                if not usable and not table_boxes:
                    targets = [None]
                else:
                    targets = [word_box(w) for w in broken]
                    for image in page.images:
                        box = word_box(image)
                        if (box[2] - box[0]) * (box[3] - box[1]) > 15000:
                            targets.extend(image_text_gaps(box, usable))
                    targets = merge_boxes(targets)
                    if len(targets) > limits.max_attachments:
                        targets = targets[:limits.max_attachments]
                        builder.warnings.append(f"ocr_no_text:page:{index + 1}; region_limit")
                    if targets and info["text_layer"] == "usable":
                        info["text_layer"] = "partial"
                image_readings = {}
                for box in targets:
                    attempt = {"page": index + 1, "bbox": box or [0, 0, info["width"], info["height"]],
                               "engine": "tesseract", "reason": "no_usable_text_layer" if box is None else
                               "uncovered_or_damaged_region"}
                    info["ocr_attempts"].append(attempt)
                    try:
                        text, metadata = ocr_region(renderer, index, box, native=[
                            w["bbox_pdf_points"] for w in info["native"]])
                        if metadata.get("extraction") == "ocr":
                            info["ocr"].append(metadata)
                    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                        attempt.update(status="failed", error=type(exc).__name__)
                        if box is None:
                            raise
                        image_readings[tuple(box)] = {"status": "failed", "text": "", "reason": type(exc).__name__}
                        builder.warnings.append(f"ocr_no_text:page:{index + 1}")
                        continue
                    attempt["status"] = ("blank" if metadata.get("page_state") == "blank" else
                                         "recognized" if text else "empty")
                    if metadata.get("page_state") == "blank" and box is None:
                        info["blank"] = True
                        attempt["status"] = "blank_no_ocr"
                        builder.warnings.append(f"pdf_blank_page:{index + 1}")
                    elif text:
                        layouts = metadata.pop("ocr_tables", [])
                        if box:
                            image_readings[tuple(box)] = {"status": "recognized", "text": text,
                                                          "engine": "tesseract", "verified": False}
                        if layouts:
                            # Retain the complete observations for field routing. Emit each
                            # source word once in the parsed blocks.
                            for layout in layouts:
                                layout.update(extraction="ocr", engine="tesseract", page_state="needs_review")
                                regions.append(table_region(layout))
                            outside = [word for word in metadata["ocr_words"] if not any(
                                intersects(word["bbox_pdf_points"], layout["bbox"]) for layout in layouts)]
                            text = "\n".join(line["text"] for line in primary_lines(
                                outside, metadata["pixel_to_pdf_transform"]))
                            metadata = {**metadata, "ocr_words": outside}
                            builder.warnings.append(f"pdf_table_geometry_unverified:page:{index + 1}")
                        if text:
                            regions.append((box[1] if box else 0, box[0] if box else 0, "TEXT", "paragraph", text, metadata))
                        builder.warnings.append(f"ocr_unverified:page:{index + 1}; numeric_and_layout_review_required")
                    else:
                        if not box:
                            regions.append((0, 0, "IMAGE", "image", "", {**metadata, "unresolved": True, "bbox": box}))
                        builder.warnings.append(f"ocr_no_text:page:{index + 1}")
                        if box:
                            image_readings[tuple(box)] = {"status": "no_text_detected", "text": ""}
                if None not in targets:
                    seen_images = set()
                    for image in page.images:
                        box = (image["x0"], image["top"], image["x1"], image["bottom"])
                        if box in seen_images:
                            continue
                        if len(seen_images) >= limits.max_attachments:
                            builder.warnings.append(f"pdf_unresolved_image_limit:page:{index + 1}")
                            break
                        seen_images.add(box)
                        observation = image_readings.get(box)
                        if observation is None:
                            readings = [r for area, r in image_readings.items() if intersects(area, box)]
                            if readings:
                                observation = {"status": "partial_regions", "engine": "tesseract",
                                               "text": "\n".join(r["text"] for r in readings if r["text"]),
                                               "regions": readings, "verified": False}
                        if observation is None:
                            observation = {"status": "budget_exhausted", "text": ""}
                            if visual_ocr_budget:
                                visual_ocr_budget -= 1
                                try:
                                    observation = ocr_region(renderer, index, box, visual_only=True)
                                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                                    observation = {"status": "failed", "text": "", "reason": type(exc).__name__}
                        regions.append((box[1], box[0], "IMAGE", "image", "",
                                        {"bbox": box, "embedded_visual": True, "unresolved": True,
                                         "image_ocr": observation}))
                if len(re.findall(r"(?:Table|Таблица)\s*\d+\s*[:.]?", raw_text, re.IGNORECASE)) > sum(region[2] == "TABLE" for region in regions):
                    metadata = {"unresolved": True, "reason": "caption_without_reconstructed_table"}
                    if verifier is not None:
                        detection = verifier.detect_tables(renderer, index)
                        metadata["table_layout_candidates"] = detection
                        if not table_boxes and len(detection["boxes"]) == 1:
                            metadata["bbox"] = detection["boxes"][0]["bbox_pdf_points"]
                    regions.append((0, 0, "TABLE", "table", "", metadata))
                for _, _, kind, role, text, metadata in order_regions(regions, gutters):
                    metadata.setdefault("coordinate_system", "pdf_points_top_left")
                    locator = replace(source, bbox=list(metadata["bbox"]) if metadata.get("bbox") else None)
                    builder.add(text, role, locator, kind=kind, metadata=metadata)
                if page.images and None not in targets:
                    builder.warnings.append(f"pdf_visual_content_not_interpreted:page:{index + 1}")
                if damaged(raw_text):
                    builder.warnings.append(f"pdf_unknown_glyph:page:{index + 1}")
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                del builder.blocks[start:]
                if raw_text.strip() and not damaged(raw_text):
                    builder.add(raw_text, "paragraph", source, metadata={"partial_page": True})
                builder.add("", "image", source, kind="IMAGE", metadata={"unresolved": True, "page_error": str(exc)})
                builder.warnings.append(f"pdf_page_failed:{index + 1}:{type(exc).__name__}")
            finally:
                page.close()
    result = builder.finish()
    result.parser_profile = "fusion-pdf-cpu-v6-tables"
    return result


def parse_pdf(path, context, limits, *, fallback=None, policy=None, verifier=None,
              field_profile=None, ocr_mode="selective"):
    from .pdf_fallback import route_pages
    if ocr_mode not in {"baseline", "all", "selective"}:
        raise ValueError("Unknown OCR mode")
    if Path(path).stat().st_size > limits.max_file_bytes:
        raise ValueError("PDF file size exceeds the limit")
    if verifier is None:
        verifier = PaddleVerifier.configured()
    pages = {}
    document = _parse_pdf_primary(path, context, limits, verifier=verifier, page_inputs=pages)
    document = route_pages(attach_visual_context(document), path, limits, fallback, policy, page_inputs=pages)
    check_fields(document, path, pages, verifier, limits, policy, field_profile, ocr_mode)
    if len(document.blocks) > limits.max_blocks or sum(len(b.text) for b in document.blocks) > limits.max_text_chars:
        raise ValueError("Parsed document exceeds the output limits")
    return document
