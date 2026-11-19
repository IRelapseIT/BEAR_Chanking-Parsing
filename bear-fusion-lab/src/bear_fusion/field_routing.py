"""Document-wide field checks and bounded secondary OCR routing."""
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import groupby
from math import isfinite
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, uuid5

from bear_text_beta.models import ContentBlock, SourceLocator
from .ocr_verification import normalized, pdf_box
from .raster import raster_region, visual_state


@dataclass(frozen=True)
class FieldRule:
    name: str
    labels: tuple[str, ...] = ()
    pattern: str = r".+"
    required: bool = True
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.required, bool):
            raise ValueError("Invalid field name or required flag")
        if self.page is not None and (type(self.page) is not int or self.page < 1):
            raise ValueError("Invalid field page")
        if self.bbox is not None:
            if not self.page or len(self.bbox) != 4 or not all(map(isfinite, self.bbox)):
                raise ValueError("A template bbox requires four finite coordinates and a page")
            x0, y0, x1, y1 = self.bbox
            if not (0 <= x0 < x1 and 0 <= y0 < y1):
                raise ValueError("Invalid field bbox")
        if not isinstance(self.labels, (list, tuple)) or any(
            not isinstance(label, str) or not label.strip() for label in self.labels
        ):
            raise ValueError("Field labels must be nonempty strings")
        if self.bbox is None and not self.labels:
            raise ValueError("A field requires labels or a template bbox")
        re.compile(self.pattern)


DEFAULT_FIELDS = (
    FieldRule("order_number", ("Номер приказа", "Order number"), r"№?\s*[\w/-]*\d[\w/-]*"),
    FieldRule("amount_due", ("Итого к оплате", "Total due"),
              r"(?:[₽$€]\s*)?[+−-]?(?:\d{1,3}(?:[ \u00a0]\d{3})+|\d+)(?:[.,]\d+)?"
              r"(?:\s*(?:[₽$€%‰]|руб\.?|RUB|USD|EUR|коп\.?|кг|kg))?"),
)


def intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def contains(outer, inner, tolerance=0):
    return (outer[0] - tolerance <= inner[0] < inner[2] <= outer[2] + tolerance
            and outer[1] - tolerance <= inner[1] < inner[3] <= outer[3] + tolerance)


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def merge_group(items, box_of):
    groups = []
    for item in items:
        box = list(box_of(item))
        group = [item]
        while True:
            hit = next((i for i, (other, _) in enumerate(groups) if intersects(box, other)), None)
            if hit is None:
                break
            other, members = groups.pop(hit)
            box, group = union([box, other]), members + group
        groups.append((box, group))
    return groups


def merge_boxes(boxes):
    return [box for box, _ in merge_group(boxes, lambda b: b)]


def image_text_gaps(box, words):
    covered = [word for word in words if intersects(
        box, (word["x0"], word["top"], word["x1"], word["bottom"]))]
    if not covered:
        return [box]
    # Suppress the common full-page image with a dense searchable OCR layer.
    typical = median(w["bottom"] - w["top"] for w in covered)
    cursor, gaps = box[1], []
    for word in sorted(covered, key=lambda w: w["top"]):
        top = min(box[3], word["top"])
        if top - cursor >= max(12, 1.8 * typical):
            gaps.append([box[0], cursor, box[2], top])
        cursor = max(cursor, word["bottom"])
    if box[3] - cursor >= max(12, 1.8 * typical):
        gaps.append([box[0], cursor, box[2], box[3]])
    return gaps


def reading(words, box):
    selected = []
    for word in words:
        x0, y0, x1, y1 = word["bbox_pdf_points"]
        if box[0] <= (x0 + x1) / 2 < box[2] and box[1] <= (y0 + y1) / 2 < box[3]:
            selected.append(word)
    if not selected:
        return None
    selected.sort(key=lambda w: ((w["bbox_pdf_points"][1] + w["bbox_pdf_points"][3]) / 2,
                                 w["bbox_pdf_points"][0]))
    # Sort horizontal words together even when glyph tops differ on one line.
    height = median(w["bbox_pdf_points"][3] - w["bbox_pdf_points"][1] for w in selected)
    rows = []
    for word in selected:
        center = sum(word["bbox_pdf_points"][1::2]) / 2
        if not rows or center - rows[-1][0] > height / 2:
            rows.append((center, []))
        rows[-1][1].append(word)
    ordered = [w for _, row in rows for w in sorted(row, key=lambda w: w["bbox_pdf_points"][0])]
    scores = [w["confidence"] for w in ordered if w.get("confidence") is not None]
    return {"text": " ".join(w["text"] for w in ordered).strip(" :"),
            "bbox": union([w["bbox_pdf_points"] for w in ordered]),
            "confidence": min(scores) if scores else None,
            "engines": sorted({w["engine"] for w in ordered})}


def locate(rule, info, tables):
    if rule.bbox is not None:
        x0, y0, x1, y1 = rule.bbox
        return [list(rule.bbox)] if 0 <= x0 < x1 <= info["width"] and 0 <= y0 < y1 <= info["height"] else []
    labels = {tuple(normalized(label).split()) for label in rule.labels}
    boxes = []
    words = info["words"]
    for word in words:
        start = normalized(word["text"]).rstrip(":")
        matches = [label for label in labels if label and label[0] == start]
        if not matches:
            continue
        first = word["bbox_pdf_points"]
        center = (first[1] + first[3]) / 2
        row = sorted((w for w in words if w["bbox_pdf_points"][0] >= first[0]
                      and w["bbox_pdf_points"][1] <= center <= w["bbox_pdf_points"][3]),
                     key=lambda w: w["bbox_pdf_points"][0])
        for label in matches:
            part = row[:len(label)]
            if tuple(normalized(w["text"]).rstrip(":") for w in part) != label:
                continue
            label_box = union([w["bbox_pdf_points"] for w in part])
            left, top, right, bottom = label_box
            height = bottom - top
            if any(b["bbox_pdf_points"][0] - a["bbox_pdf_points"][2] > 2 * height
                   for a, b in zip(part, part[1:])):
                continue
            limit = min(info["width"], right + max(100, 12 * height))
            box = [right, max(0, top - .45 * height), limit, min(info["height"], bottom + .45 * height)]
            for table in tables:
                cells = table.metadata.get("table_cells", [])
                coordinates = table.metadata.get("cell_bboxes", {})
                for cell in cells:
                    source = coordinates.get(cell["id"])
                    if source and source[0] <= left and source[2] >= right and intersects(source, label_box):
                        neighbor = next((c for c in cells if c["row"] == cell["row"]
                                         and c["column"] == cell["column"] + cell["column_span"]), None)
                        if neighbor and neighbor["id"] in coordinates:
                            box = list(coordinates[neighbor["id"]])
            if box[0] < box[2]:
                boxes.append(box)
    return [box for box, _ in merge_group(boxes, lambda b: b)]


def invoke_crop(verifier, renderer, page, box, words=()):
    if verifier is None:
        return {"status": "not_run", "reasons": ["secondary_ocr_unavailable"], "secondary_lines": []}
    try:
        with raster_region(renderer, page - 1, box) as (picture, transform, dpi, _):
            with tempfile.TemporaryDirectory(prefix="bear-field-") as folder:
                image = Path(folder) / "region.png"
                picture.save(image, dpi=(dpi, dpi))
                scale, _, _, _, ox, oy = transform
                primary = []
                for word in words:
                    x0, y0, x1, y1 = word["bbox_pdf_points"]
                    primary.append({**word, "bbox_pixels": [(x0 - ox) / scale, (y0 - oy) / scale,
                                                           (x1 - x0) / scale, (y1 - y0) / scale]})
                return verifier.verify(image, page, primary, transform, picture.size)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"status": "insufficient", "reasons": [type(exc).__name__], "secondary_lines": []}


def classify(renderer, record, rule, primary, threshold):
    if primary and primary["text"] and not re.fullmatch(r"[_| ]+", primary["text"]):
        if not re.fullmatch(rule.pattern, primary["text"], re.IGNORECASE):
            return "invalid_format"
        if primary["confidence"] is not None and primary["confidence"] < threshold:
            return "low_ocr_confidence"
        return "primary_value"
    try:
        with raster_region(renderer, record["page"] - 1, record["bbox"]) as (image, *rest):
            state = visual_state(image)
    except (OSError, ValueError, RuntimeError) as exc:
        record["visual_error"] = type(exc).__name__
        state = "ambiguous"
    return {"blank": "blank_in_source", "ambiguous": "ambiguous_visual_evidence",
            "marks": "possible_ocr_omission"}[state]


def resolve(record, rule, attempts):
    primary = record["primary"]
    lines = [{**line, "engine": "paddleocr", "confidence": line["confidence"] * 100}
             for attempt in attempts for line in attempt.get("secondary_lines", [])]
    secondary = reading([line for line in lines if contains(record["bbox"], line["bbox_pdf_points"], 1)],
                        record["bbox"])
    record["secondary"] = secondary
    record["secondary_attempts"] = [{key: attempt[key] for key in ("status", "reasons", "elapsed_seconds")
                                    if key in attempt} for attempt in attempts]
    valid = secondary and re.fullmatch(rule.pattern, secondary["text"], re.IGNORECASE)
    if primary and secondary and normalized(primary["text"]) != normalized(secondary["text"]):
        record.update(status="needs_review", decision="ocr_conflict")
    elif record["reason"] == "possible_ocr_omission" and not primary and valid:
        record.update(status="recovered_by_secondary", value=secondary["text"],
                      decision="retain_recovered_value_for_review")
    elif record["reason"] in {"invalid_format", "low_ocr_confidence", "possible_ocr_omission"}:
        record.update(status="needs_review", decision="secondary_did_not_resolve_risk")
    elif record["reason"] == "primary_value":
        record.update(status="observed", value=primary["text"], decision="retain_primary")


def check_fields(document, path, pages, verifier, limits, policy, profile, mode):
    import pypdfium2 as pdfium

    rules = tuple(DEFAULT_FIELDS if profile is None else profile)
    blocks = {}
    for block in document.blocks:
        blocks.setdefault(block.source.page, []).append(block)
    for info in pages.values():
        info["words"] = info["native"] + [
            {**w, "engine": "tesseract"} for region in info["ocr"] for w in region["ocr_words"]]
    records, scheduled, full_reads = [], [], {}
    threshold = policy.ocr_min_confidence if policy else 70
    with pdfium.PdfDocument(path) as renderer:
        for rule in rules:
            found = False
            for page, info in pages.items():
                if rule.page is not None and page != rule.page:
                    continue
                tables = [b for b in blocks.get(page, []) if b.type == "TABLE"
                          and b.metadata.get("values_verified") is not False
                          and not b.metadata.get("unresolved")]
                for box in locate(rule, info, tables):
                    found = True
                    value = reading(info["words"], box)
                    record = {"field": rule.name, "page": page, "bbox": box, "required": rule.required,
                              "primary": value, "engine": "paddleocr", "status": "needs_review"}
                    reason = classify(renderer, record, rule, value, threshold)
                    record.update(reason=reason, decision=reason)
                    if reason == "blank_in_source":
                        record["status"] = reason
                    records.append((record, rule))
                    if reason in {"possible_ocr_omission", "invalid_format", "low_ocr_confidence"}:
                        scheduled.append(record)
            if profile is not None and rule.required and not found:
                records.append(({"field": rule.name, "page": rule.page, "bbox": None,
                                 "required": True, "engine": None, "status": "needs_review",
                                 "reason": "field_not_localized", "decision": "field_not_localized"}, rule))

        if mode == "all":
            for page, info in pages.items():
                for region in info["ocr"]:
                    box = region["ocr_crop_box"]
                    attempt = invoke_crop(verifier, renderer, page, box, region["ocr_words"])
                    region["ocr_verification"] = attempt
                    document.warnings.append("pdf_ocr_route:" + json.dumps(
                        {**attempt, "page": page, "bbox": box, "engine": "paddleocr",
                         "trigger_reasons": ["all_ocr_regions"]}, ensure_ascii=False))
                    full_reads.setdefault(page, []).append((box, attempt))
        if mode == "selective":
            for page, info in pages.items():
                for region in info["ocr"]:
                    box = region["ocr_crop_box"]
                    scores = [word["confidence"] for word in region["ocr_words"]]
                    if not scores or sum(scores) / len(scores) >= threshold:
                        continue
                    if any(r["page"] == page and intersects(r["bbox"], box) for r in scheduled):
                        continue
                    scheduled.append({"page": page, "bbox": box, "required": False,
                                      "reason": "low_region_ocr_confidence"})
            scheduled.sort(key=lambda record: record["page"])
            groups = [(page, box, members) for page, same in groupby(scheduled, key=lambda record: record["page"])
                      for box, members in merge_group(same, lambda record: record["bbox"])]
            groups.sort(key=lambda group: not any(r["required"] for r in group[2]))
            for page, box, members in groups:
                words = [word for region in pages[page]["ocr"] for word in region["ocr_words"]
                         if contains(box, word["bbox_pdf_points"], 1)]
                attempt = invoke_crop(verifier, renderer, page, box, words)
                attempt.update(page=page, bbox=box, engine="paddleocr",
                               trigger_reasons=sorted({r["reason"] for r in members}))
                document.warnings.append("pdf_ocr_route:" + json.dumps(attempt, ensure_ascii=False))
                for record in members:
                    record["_attempt"] = attempt

    recovered = []
    recovered_sources = set()
    text_chars = sum(len(b.text) for b in document.blocks)
    for record, rule in records:
        box = record["bbox"]
        if box is not None:
            attempt = record.pop("_attempt", None)
            attempts = ([attempt] if attempt is not None else
                        [read for area, read in full_reads.get(record["page"], []) if intersects(area, box)])
            if not attempts and record["reason"] in {"possible_ocr_omission", "invalid_format", "low_ocr_confidence"}:
                attempts = [{"status": "not_run", "reasons": [f"ocr_mode_{mode}_no_secondary_read"]}]
            if record["reason"] not in {"blank_in_source", "ambiguous_visual_evidence"}:
                resolve(record, rule, attempts)
        if record["status"] == "recovered_by_secondary":
            value = record["value"]
            location = record["page"], tuple(record["secondary"]["bbox"]), normalized(value)
            if location not in recovered_sources and (
                    len(document.blocks) + len(recovered) >= limits.max_blocks or text_chars + len(value) > limits.max_text_chars):
                record.update(status="needs_review", decision="output_limit")
            elif location not in recovered_sources:
                text_chars += len(value)
                recovered_sources.add(location)
                identity = f"{document.context!r}/{record['page']}/{rule.name}/{box}"
                recovered.append(ContentBlock(
                    str(uuid5(NAMESPACE_URL, identity)), 0, "TEXT", "paragraph", value, [],
                    SourceLocator("pdf", page=record["page"], bbox=record["secondary"]["bbox"]),
                    metadata={"extraction": "ocr", "engine": "paddleocr", "pdf_field": record,
                              "page_state": "needs_review"}))
                document.warnings.append(f"ocr_unverified:page:{record['page']}; recovered_by_secondary")
        document.warnings.append("pdf_field_route:" + json.dumps(record, ensure_ascii=False))
        if record["status"] in {"needs_review", "recovered_by_secondary"} or (
                rule.required and record["status"] == "blank_in_source"):
            document.warnings.append(f"pdf_field_needs_review:{rule.name}:page:{record['page']}")
        if record["page"] in blocks:
            blocks[record["page"]][0].metadata.setdefault("pdf_fields", []).append(record)
    if recovered:
        document.blocks.extend(recovered)
        document.blocks.sort(key=lambda b: b.source.page if b.source.page is not None else 0)
        for index, block in enumerate(document.blocks):
            block.order = index
    document.parser_profile += f":field-routing-v1-{mode}"
