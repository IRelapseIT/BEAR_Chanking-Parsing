"""Run real OCR and check the borderless PDF by row, column and source geometry."""
import argparse
import json
import time
from pathlib import Path

from bear_fusion.ocr_verification import PaddleVerifier, normalized
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import FallbackPolicy
from bear_fusion.tables import TableProcessor, validate_table_chunks
from bear_text_beta.acceptance import assess
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import (
    CharacterTokenCounter,
    UnavailableEmbeddingProvider,
)
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext, EmbeddingProfile, ProcessingProfile
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from workflow_run import pdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "verification/ocr-upgrade"


def distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(actual, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def metrics(expected, lines):
    actual = normalized(" ".join(line["text"] for line in lines))
    target = normalized(expected)
    return {"text": actual, "cer": distance(target, actual) / len(target),
            "wer": distance(target.split(), actual.split()) / len(target.split())}


def parse(path, verifier=None):
    context = DocumentContext("smoke", "local", path.stem, 1, path.name)
    return parse_pdf(path, context, FileLimits(), policy=FallbackPolicy(enabled=False), verifier=verifier)


def chunk(document):
    profile = EmbeddingProfile()
    counter = CharacterTokenCounter(profile)
    table = TableProcessor(counter)
    result = ChunkingService(counter, UnavailableEmbeddingProvider(), profile, {"TABLE": table}).process(
        document, ProcessingProfile(mode="token"))
    validate_table_chunks(document, result.chunks, counter, 320)
    return result


def main(args):
    OUT.mkdir(exist_ok=True)
    scan = OUT / "scan.pdf"
    scan.write_bytes(pdf(scan=True))
    russian = OUT / "numeric-ru.pdf"
    text_ru = "Сальдо −12,5 %; комиссия 27 руб."
    image = Image.new("RGB", (1600, 360), "white")
    ImageDraw.Draw(image).text((80, 110), text_ru, font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56),
                               fill="black")
    writer = canvas.Canvas(str(russian))
    writer.drawImage(ImageReader(image), 30, 430, width=540, height=121.5)
    writer.save()
    cases = [(scan, "SCAN POLICY: Refund within 27 days.", []), (russian, text_ru, [])]
    if args.comparison_corpus:
        from pypdfium2 import PdfDocument

        root = args.comparison_corpus
        reference = json.loads((root / "references/minzdrav-237.json").read_text(encoding="utf-8"))
        page_ref = next(row for row in reference if row["page"] == 2)
        page_path = OUT / "minzdrav-page-2.pdf"
        with PdfDocument(root / "corpus/minzdrav-237.pdf") as original, PdfDocument.new() as selected:
            selected.import_pages(original, pages=[1])
            selected.save(page_path)
        cases.append((page_path, page_ref["text"], page_ref["anchors"]))
    reports = []
    for path, expected, anchors in cases:
        started = time.monotonic()
        document = parse(path, PaddleVerifier(args.python, args.models))
        result = chunk(document)
        acceptance = assess(result)
        evidence, = [b.metadata["ocr_verification"] for b in document.blocks if "ocr_verification" in b.metadata]
        report = {"file": path.name, "expected": expected, "evidence": evidence, "acceptance": acceptance,
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "primary_metrics": metrics(expected, evidence["primary_lines"]),
                  "secondary_metrics": metrics(expected, evidence["secondary_lines"])}
        report["anchors"] = {anchor: {engine: normalized(anchor) in report[f"{engine}_metrics"]["text"]
                                      for engine in ("primary", "secondary")} for anchor in anchors}
        reports.append(report)
        print(path.name, evidence["status"],
              {engine: {metric: report[f"{engine}_metrics"][metric] for metric in ("cer", "wer")}
               for engine in ("primary", "secondary")}, flush=True)
    borderless = ROOT / "verification/pdf-fallback/borderless"
    expected = json.loads((borderless / "report.json").read_text(encoding="utf-8"))["expected_table"]
    document = parse(borderless / "borderless.pdf")
    result = chunk(document)
    table, = [b for b in document.blocks if b.type == "TABLE" and not b.metadata.get("unresolved")]
    cells = table.metadata["table_cells"]
    correct = sum(cell["text"] == expected[cell["row"]][cell["column"]] for cell in cells)
    table_report = {"exact_cell_matches": correct, "expected_cells": sum(map(len, expected)),
                    "cells": cells, "cell_bboxes": table.metadata["cell_bboxes"], "page": table.source.page,
                    "acceptance": assess(result)}
    layout = []
    if args.comparison_corpus and args.layout_model:
        source = args.comparison_corpus / "corpus/attention-table-sample.pdf"
        document = parse(source, PaddleVerifier(args.python, args.models, layout_model=args.layout_model))
        layout = [{"page": b.source.page, "bbox": b.source.bbox, "unresolved": b.metadata.get("unresolved"),
                   "detector": b.metadata["table_layout_candidates"]}
                  for b in document.blocks if "table_layout_candidates" in b.metadata]
        print("attention-layout", layout, flush=True)
    (OUT / "report.json").write_text(json.dumps({"ocr": reports, "table": table_report, "layout": layout},
                                               ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    assert all(row["evidence"]["secondary_lines"] for row in reports), "Real secondary OCR did not return text"
    assert all(row["acceptance"]["status"] == "needs_review" for row in reports)
    assert correct == len(cells) == 12 and table.source.page == 2
    assert table_report["acceptance"]["status"] == "needs_review"
    if args.comparison_corpus and args.layout_model:
        assert any(item["detector"]["status"] == "detected" and item["unresolved"] and item["bbox"]
                   for item in layout), "Real layout detector did not locate the unresolved table"
    print("OCR_UPGRADE_SMOKE_OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=ROOT / ".ocr-venv/Scripts/python.exe")
    parser.add_argument("--models", type=Path, default=ROOT / "models/paddle-ocr")
    parser.add_argument("--comparison-corpus", type=Path)
    parser.add_argument("--layout-model", type=Path)
    main(parser.parse_args())
