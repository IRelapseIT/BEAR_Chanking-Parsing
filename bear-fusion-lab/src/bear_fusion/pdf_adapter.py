"""Page/region PDF processing; uncertain areas remain explicit and recoverable."""
import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

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
        return text, {"extraction": "ocr", "ocr_words": words, "ocr_scale": scale, "ocr_dpi": dpi,
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
                    for line in filtered.extract_text_lines():
                        regions.append((line["top"], line["x0"], "TEXT", "paragraph", line["text"],
                                        {"bbox": [line["x0"], line["top"], line["x1"], line["bottom"]]}))
                finally:
                    filtered.close()
                if len(re.findall(r"Table\s*\d+\s*:", raw_text, re.IGNORECASE)) > len(table_boxes):
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
                for _, _, kind, role, text, metadata in sorted(regions, key=lambda r: (r[0], r[1])):
                    builder.add(text, role, source, kind=kind, metadata=metadata)
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
    result.parser_profile = "fusion-pdf-cpu-v2"
    return result
