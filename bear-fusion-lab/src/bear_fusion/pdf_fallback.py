"""Conservative page-level fallback; quality heuristics never certify source fidelity."""
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from bear_text_beta.models import ContentBlock, SourceLocator

from .tables import Cell, grid, rows_from_cells


@dataclass(frozen=True)
class FallbackPolicy:
    enabled: bool = True
    max_pages: int = 8
    timeout: float = 120
    max_output_bytes: int = 32 * 1024 * 1024
    ocr_min_confidence: float = 70

    def __post_init__(self):
        if (self.max_pages < 1 or not math.isfinite(self.timeout) or self.timeout <= 0
                or self.max_output_bytes < 1 or not 0 <= self.ocr_min_confidence <= 100):
            raise ValueError("Invalid PDF fallback limits")


class DoclingProcess:
    def __init__(self, python, artifacts, policy):
        self.python, self.artifacts, self.policy = python, Path(artifacts), policy

    def parse_page(self, path, page, context, limits):
        if not self.artifacts.is_dir():
            raise ValueError("Docling local artifacts directory is missing")
        with tempfile.TemporaryDirectory(prefix="bear-docling-") as folder:
            output = Path(folder) / "page.json"
            # Output goes to disk, not an unbounded captured stdout pipe.
            with (Path(folder) / "worker.log").open("wb") as log:
                try:
                    subprocess.run([self.python, str(Path(__file__).with_name("docling_worker.py")),
                                    str(Path(path).resolve()), str(page), str(self.artifacts.resolve()), str(output)],
                                   stdout=log, stderr=log, timeout=self.policy.timeout, check=True,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                        "OMP_NUM_THREADS": "2"})
                except subprocess.CalledProcessError as exc:
                    log.flush()
                    with (Path(folder) / "worker.log").open("rb") as errors:
                        errors.seek(max(0, errors.seek(0, 2) - 2000))
                        raise RuntimeError(errors.read(2000).decode("utf-8", errors="replace")) from exc
            if output.stat().st_size > self.policy.max_output_bytes:
                raise ValueError("Docling output limit exceeded")
            return map_docling(json.loads(output.read_text(encoding="utf-8")), page, context, limits)


def bbox(box, height):
    if str(box.get("coord_origin", "TOPLEFT")).upper() not in {"TOPLEFT", "BOTTOMLEFT"}:
        raise ValueError("Unknown Docling coordinate origin")
    values = [float(box[key]) for key in ("l", "t", "r", "b")]
    if str(box.get("coord_origin", "TOPLEFT")).upper() == "BOTTOMLEFT":
        values = [values[0], height - values[1], values[2], height - values[3]]
    x0, y0, x1, y1 = values
    result = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    if not all(math.isfinite(v) for v in result) or min(result) < 0 or result[3] > height + 1:
        raise ValueError("Invalid Docling coordinates")
    return result


def map_docling(payload, page, context, limits):
    document = payload["document"]
    page_data = document["pages"][str(page)]
    height, width = float(page_data["size"]["height"]), float(page_data["size"]["width"])
    if not all(math.isfinite(v) and v > 0 for v in (height, width)):
        raise ValueError("Invalid Docling page size")
    blocks, seen = [], set()
    text_chars = 0
    for reference in payload["ordered_refs"]:
        if reference in seen:
            continue
        seen.add(reference)
        parts = reference.split("/")
        if len(parts) != 3 or parts[1] not in {"texts", "tables", "pictures"}:
            continue
        item = document[parts[1]][int(parts[2])]
        provenance = item.get("prov", [])
        if not provenance or any(p["page_no"] != page for p in provenance):
            raise ValueError("Missing, foreign or multi-page Docling provenance")
        boxes = [bbox(p["bbox"], height) for p in provenance]
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        if box[2] > width + 1:
            raise ValueError("Docling bbox exceeds page")
        kind = {"texts": "TEXT", "tables": "TABLE", "pictures": "IMAGE"}[parts[1]]
        text = item.get("text", "")
        metadata = {"parser": "docling", "parser_version": payload["version"], "docling_ref": reference,
                    "docling_confidence": payload.get("confidence", {}),
                    "coordinate_system": "pdf_points_top_left", "bbox": box}
        role = "heading" if item.get("label") in {"section_header", "title"} else "paragraph"
        if kind == "TABLE":
            cells, cell_boxes, cell_roles = [], {}, {}
            data = item["data"]
            for index, cell in enumerate(data["table_cells"]):
                row, col = cell["start_row_offset_idx"], cell["start_col_offset_idx"]
                rs, cs = cell["end_row_offset_idx"] - row, cell["end_col_offset_idx"] - col
                if row < 0 or col < 0 or row + rs > data["num_rows"] or col + cs > data["num_cols"]:
                    raise ValueError("Docling cell outside declared grid")
                cells.append(Cell(f"d{index}", row, col, cell["text"], rs, cs,
                                  bool(cell.get("column_header")), "col" if cell.get("column_header") else ""))
                if cell.get("bbox"):
                    cell_boxes[f"d{index}"] = bbox(cell["bbox"], height)
                cell_roles[f"d{index}"] = {key: bool(cell.get(key)) for key in (
                    "column_header", "row_header", "row_section")}
            expanded = grid(cells)
            if len(expanded) != data["num_rows"] * data["num_cols"]:
                raise ValueError("Docling table has missing grid cells")
            if not cells or data["num_rows"] < 2 or data["num_cols"] < 2:
                raise ValueError("Docling table has insufficient structure")
            heads = 0
            while heads < data["num_rows"] and any(c.header and c.row == heads for c in cells):
                heads += 1
            metadata.update(table_cells=[asdict(c) for c in cells], header_rows=heads or 1,
                            cell_bboxes=cell_boxes, docling_cell_roles=cell_roles,
                            header_inference=not bool(heads), table_contract="cells-v1")
            text = "\n".join(" | ".join(row) for row in rows_from_cells(cells))
            role = "table"
        elif kind == "IMAGE":
            metadata["unresolved"] = True
            role = "image"
        if kind == "TEXT" and not text.strip():
            continue
        identity = f"{context.tenant_id}/{context.space_id}/{context.document_id}/{context.document_version}/docling/{page}/{reference}"
        blocks.append(ContentBlock(str(uuid5(NAMESPACE_URL, identity)), len(blocks), kind, role,
                                   text, [], SourceLocator("pdf", page=page, bbox=box), metadata=metadata))
        text_chars += len(text)
        if len(blocks) > limits.max_blocks or text_chars > limits.max_text_chars:
            raise ValueError("Docling content limits exceeded")
    return blocks


def page_reasons(blocks, warnings, page, policy):
    reasons = set()
    for warning in warnings:
        if (f":page:{page}" in warning and re.search(rf":page:{page}(?:\D|$)", warning)
                and any(key in warning for key in (
                    "unknown_glyph", "reading_order_needs_review", "table_candidate_rejected", "sparse_text_visual"))):
            reasons.add(warning.split(":")[0])
        if warning.startswith(f"pdf_page_failed:{page}:"):
            reasons.add("pdf_page_failed")
    text = "\n".join(b.text for b in blocks)
    if "(cid:" in text or "\ufffd" in text or any(ord(c) < 32 and c not in "\n\r\t" for c in text):
        reasons.add("pdf_unknown_glyph")
    tables = [b for b in blocks if b.type == "TABLE" and not b.metadata.get("unresolved")]
    if len(re.findall(r"(?:Table|Таблица)\s*\d+", text, re.IGNORECASE)) > len(tables):
        reasons.add("caption_without_table")
    for block in blocks:
        if block.metadata.get("unresolved"):
            reasons.add("unresolved_content")
        confidence = block.metadata.get("mean_word_confidence")
        docling_scores = block.metadata.get("docling_confidence", {})
        if "ocr_score" in docling_scores:
            confidence = docling_scores["ocr_score"] * 100
        if any(docling_scores.get(key, 1) < .5 for key in ("parse_score", "layout_score", "table_score")):
            reasons.add("low_docling_confidence")
        if confidence is not None and confidence < policy.ocr_min_confidence:
            reasons.add("low_ocr_confidence")
        if block.metadata.get("extraction") == "ocr" and len(block.text.strip()) < 15:
            reasons.add("sparse_ocr_text")
    return sorted(reasons)


def preserved(primary, candidate):
    def tokens(blocks):
        text = " ".join(b.text for b in blocks)
        text = re.sub(r"\(cid:\d+\)", "", text).replace("\ufffd", " ")
        # Preserve signs, decimal separators and percentages as part of a number.
        # A bag of plain \w tokens would incorrectly equate -12 with 12 or 3.14 with 3,14.
        return Counter(re.findall(r"[+-]?\d+(?:[.,]\d+)*(?:[%‰])?|[^\W\d_]+", text.casefold()))
    return not (tokens(primary) - tokens(candidate))


def route_pages(document, path, limits, backend=None, policy=None):
    policy = policy or FallbackPolicy(enabled=os.getenv("BEAR_PDF_DOCLING", "1") == "1")
    if backend is None and os.getenv("BEAR_DOCLING_PYTHON") and os.getenv("BEAR_DOCLING_ARTIFACTS"):
        backend = DoclingProcess(os.environ["BEAR_DOCLING_PYTHON"], os.environ["BEAR_DOCLING_ARTIFACTS"], policy)
    groups = {}
    for block in document.blocks:
        groups.setdefault(block.source.page, []).append(block)
    # Failed/blank pages can have no extracted text. Failed pages still have placeholder blocks.
    attempts, rebuilt = 0, []
    for page, original in groups.items():
        reasons = page_reasons(original, document.warnings, page, policy)
        chosen, selected, remaining = original, "bear", reasons
        record = {"page": page, "primary_reasons": reasons, "selected_parser": selected,
                  "fallback_attempted": False, "source_fidelity": "unverified"}
        started = time.perf_counter()
        if reasons:
            if not policy.enabled:
                record["fallback_status"] = "disabled"
            elif backend is None:
                record["fallback_status"] = "unavailable"
            elif attempts >= policy.max_pages:
                record["fallback_status"] = "page_budget_exceeded"
            else:
                attempts += 1
                record["fallback_attempted"] = True
                try:
                    candidate = backend.parse_page(path, page, document.context, limits)
                    if (not candidate or not any(re.search(r"\w", b.text) for b in candidate)
                            or any(b.source.page != page for b in candidate)
                            or len(candidate) > limits.max_blocks
                            or sum(len(b.text) for b in candidate) > limits.max_text_chars):
                        raise ValueError("Invalid fallback page")
                    issues = page_reasons(candidate, [], page, policy)
                    if "low_ocr_confidence" in reasons and not any(
                            "ocr_score" in b.metadata.get("docling_confidence", {}) for b in candidate):
                        issues.append("candidate_ocr_confidence_unknown")
                    record["candidate_reasons"] = issues
                    record["candidate_parser_versions"] = sorted({
                        b.metadata["parser_version"] for b in candidate if "parser_version" in b.metadata})
                    # Order-only disagreement has no independent oracle: retain the primary result.
                    if not issues and preserved(original, candidate) and "pdf_reading_order_needs_review" not in reasons:
                        chosen, selected, remaining = candidate, "docling", []
                        record["fallback_status"] = "selected"
                    else:
                        record["fallback_status"] = "rejected_or_ambiguous"
                except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    record.update(fallback_status="failed", error_type=type(exc).__name__, fallback_error=str(exc)[:2000])
        record.update(selected_parser=selected, remaining_reasons=remaining,
                      status="needs_review" if remaining else "heuristic_checks_passed",
                      fallback_seconds=round(time.perf_counter() - started, 6))
        document.warnings.append("pdf_page_route:" + json.dumps(record, ensure_ascii=False, sort_keys=True))
        if remaining:
            document.warnings.append(f"pdf_needs_review:page:{page}")
        for block in chosen:
            block.metadata["pdf_route"] = record
            block.order = len(rebuilt)
            rebuilt.append(block)
    document.blocks = rebuilt
    document.parser_profile += ":page-fallback-v1"
    return document
