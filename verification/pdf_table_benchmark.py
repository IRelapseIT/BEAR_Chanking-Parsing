"""Compare the three modes on the same generated and explicitly labeled PDF."""
import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from bear_fusion.field_routing import FieldRule
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext
from bear_fusion.ocr_verification import PaddleVerifier
from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import DoclingProcess


def raster(text, font, size, dimensions):
    with Image.new("RGB", dimensions, "white") as picture:
        ImageDraw.Draw(picture).text((12, 8), text, font=ImageFont.truetype(font, size), fill="black")
        stream = io.BytesIO()
        picture.save(stream, format="PNG")
    stream.seek(0)
    return ImageReader(stream)


def corpus(path, font):
    pdf = Canvas(str(path), pagesize=(600, 800))
    pdf.drawString(40, 750, "Clean source, no OCR.")
    pdf.showPage()
    pdf.showPage()
    pdf.drawImage(raster("Order number: 48291", font, 48, (660, 110)), 40, 680, 300, 50)
    pdf.showPage()
    pdf.drawString(40, 700, "Total due:")
    pdf.drawImage(raster("-125.50 USD", font, 44, (385, 75)), 128, 687, 128, 25)
    pdf.drawString(40, 640, "Order number:")
    pdf.line(115, 640, 275, 640)
    pdf.showPage()
    pdf.drawString(40, 740, "Table 1 Annual report")
    # Deliberate gap in the geometry defeats the native borderless heuristic.
    for row, values in enumerate((("Year", "Revenue USD", "Cost USD"), ("2024", "150", "80"),
                                  ("2025", "180", "90"), ("2026", "200", "95"))):
        for col, value in enumerate(values):
            pdf.drawString(70 + col * 155 + (24 if row == 2 and col == 1 else 0), 680 - row * 30, value)
    pdf.showPage()
    pdf.drawString(40, 740, "Table 1 Native geometry control")
    for row, values in enumerate((("Year", "Revenue USD", "Cost USD"), ("2024", "150", "80"),
                                  ("2025", "180", "90"), ("2026", "200", "95"))):
        for col, value in enumerate(values):
            pdf.drawString(70 + col * 155, 680 - row * 30, value)
    pdf.save()


def records(document, prefix):
    return [json.loads(w.split(":", 1)[1]) for w in document.warnings if w.startswith(prefix + ":")]


def benchmark(path, mode):
    calls = Counter()
    import subprocess

    process_run = subprocess.run
    paddle_invoke = PaddleVerifier.invoke
    docling_page = DoclingProcess.parse_page

    def run(command, *args, **kwargs):
        if isinstance(command, list) and "tesseract" in Path(command[0]).stem.lower():
            calls["tesseract"] += 1
        return process_run(command, *args, **kwargs)

    def invoke(self, kind, *args):
        calls["paddle_" + kind] += 1
        return paddle_invoke(self, kind, *args)

    def page(self, *args):
        calls["docling"] += 1
        return docling_page(self, *args)

    profile = (
        FieldRule("scan_order", ("Order number",), r"\d+", page=3),
        FieldRule("amount", ("Total due",), r"-?\d+\.\d{2} USD", page=4),
        FieldRule("blank_order", ("Order number",), r"\d+", page=4),
    )
    with patch.object(subprocess, "run", run), patch.object(PaddleVerifier, "invoke", invoke), patch.object(
        DoclingProcess, "parse_page", page
    ):
        start = perf_counter()
        result = parse_pdf(path, DocumentContext("bench", "bench", "bench", 1, path.name), FileLimits(),
                           field_profile=profile, ocr_mode=mode)
        elapsed = perf_counter() - start
    fields = {r["field"]: r for r in records(result, "pdf_field_route")}
    gold = {"scan_order": "48291", "amount": "-125.50 USD", "blank_order": None}
    correct = sum((fields[k].get("value") == v if v is not None else fields[k]["status"] == "blank_in_source")
                  for k, v in gold.items())
    wrongly_observed = sum(r.get("value") != gold[k] for k, r in fields.items() if r["status"] == "observed")
    tables = [b for b in result.blocks if b.type == "TABLE" and not b.metadata.get("unresolved")]
    relations = set()
    for table in tables:
        cells = {cell["id"]: cell["text"] for cell in table.metadata.get("table_cells", [])}
        for identifier, link in table.metadata.get("cell_header_links", {}).items():
            rows, headings = link["row_header_ids"], link["column_header_ids"]
            if rows and headings and cells[identifier]:
                row = " / ".join(cells[h] for h in rows)
                header = " / ".join(cells[h] for h in headings)
                relations.add((table.source.page, row, header, cells[identifier]))
    expected = {(page, year, header, value) for page in (5, 6) for year, revenue, cost in (
        ("2024", "150", "80"), ("2025", "180", "90"), ("2026", "200", "95"))
                for header, value in (("Revenue USD", revenue), ("Cost USD", cost))}
    calls["pdfplumber_pages"] = len(records(result, "pdf_page_route"))
    return {"mode": mode, "elapsed_seconds": round(elapsed, 3), "correct_field_outcomes": f"{correct}/3",
            "correct_nonblank_field_values": sum(fields[k].get("value") == v for k, v in gold.items() if v is not None),
            "incorrect_observed_fields": wrongly_observed, "correct_table_relations": f"{len(relations & expected)}/12",
            "incorrect_table_relations": len(relations - expected),
            "calls": {k: calls[k] for k in ("pdfplumber_pages", "tesseract", "docling", "paddle_ocr", "paddle_layout")},
            "fields": {k: {key: r.get(key) for key in ("status", "reason", "value", "secondary_attempts")}
                       for k, r in fields.items()},
            "page_routes": [{k: r.get(k) for k in ("page", "selected_parser", "status", "fallback_status",
                                                   "candidate_reasons", "error_type", "fallback_error")}
                            for r in records(result, "pdf_page_route")]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--font", default=os.path.join(os.getenv("WINDIR", "C:/Windows"), "Fonts/arial.ttf"))
    args = cli.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / "annotated.pdf"
    corpus(path, args.font)
    results = []
    for mode in ("baseline", "all", "selective"):
        result = benchmark(path, mode)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    (args.directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
