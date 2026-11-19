"""Paired measurement of visual extraction and context association."""
import argparse
import ast
import io
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from bear_fusion.pdf_adapter import parse_pdf
from bear_fusion.pdf_fallback import FallbackPolicy
from bear_text_beta.acceptance import assess
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import (
    CharacterTokenCounter,
    UnavailableEmbeddingProvider,
)
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext, EmbeddingProfile, ProcessingProfile
from bear_text_beta.unified_parsing import DOCXParser
from docx import Document
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "visual-context"


def picture(text):
    stream = io.BytesIO()
    image = Image.new("RGB", (540, 160), "white")
    if text:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 55)
        ImageDraw.Draw(image).text((25, 50), text, font=font, fill="black")
    else:
        ImageDraw.Draw(image).polygon([(130, 115), (280, 20), (410, 115)], fill="#245684")
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def fixtures():
    for text, name in (("NORD 608", "label"), ("", "symbol")):
        document = Document()
        document.add_heading("Company details", 1)
        document.add_paragraph("This section lists the assets used on official forms.")
        document.add_picture(picture(text))
        document.add_paragraph("Figure 1. Company mark.", style="Caption")
        path = OUT / f"{name}.docx"
        document.save(path)
        yield path, text
    path = OUT / "label.pdf"
    writer = canvas.Canvas(str(path))
    writer.drawString(55, 720, "This page lists the assets used on official forms.")
    writer.drawImage(ImageReader(picture("NORD 608")), 55, 590, width=135, height=40)
    writer.drawString(55, 570, "Figure 1. Company mark.")
    writer.save()
    yield path, "NORD 608"


def upstream_context():
    """Run the pinned upstream context classes without initializing LightRAG or a vision model."""
    repo = ROOT / "rag-anything-source"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "eb7a670f724e59056b6a08f7016d753226a865ee"
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    namespace = {"dataclass": dataclass, "Any": Any, "Dict": dict, "List": list, "Optional": Optional,
                 "logger": logging.getLogger("upstream-context")}
    for filename, names in (("utils.py", {"normalize_caption_list"}),
                            ("modalprocessors.py", {"ContextConfig", "ContextExtractor"})):
        source = repo / "raganything" / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name in names]
        assert {node.name for node in selected} == names
        # Exercise the inspected, pinned upstream definitions without its runtime.
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    caption = "Figure 1. Company mark."
    content = [{"type": "text", "page_idx": 0, "text": "This page lists the assets used on official forms."},
               {"type": "image", "page_idx": 0, "image_caption": [caption]},
               {"type": "text", "page_idx": 1, "text": "Unrelated page about shipping."}]
    result = {}
    for window in (0, 1):
        settings = namespace["ContextConfig"](context_window=window, filter_content_types=["text", "image"])
        actual = namespace["ContextExtractor"](settings).extract_context(content, {"page_idx": 0}, "minerU")
        assert caption in actual
        assert ("Unrelated page" in actual) == bool(window)
        result[str(window)] = actual
    return {"revision": revision, "same_page_vs_neighbor_page": result,
            "scope": "Actual upstream context classes on prestructured content, no visual model"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    report = []
    for path, expected in fixtures():
        started = time.perf_counter()
        context = DocumentContext("visual-test", "local", path.stem, 1, path.name)
        document = (DOCXParser().parse(path.read_bytes(), context) if path.suffix == ".docx"
                    else parse_pdf(path, context, FileLimits(), policy=FallbackPolicy(enabled=False)))
        images = [{"id": b.id, "page": b.source.page, "bbox": b.source.bbox, "source": b.source.element_path,
                   "section": b.section_path, "metadata": b.metadata} for b in document.blocks if b.type == "IMAGE"]
        acceptance = None
        if not args.baseline:
            profile = EmbeddingProfile()
            result = ChunkingService(CharacterTokenCounter(profile), UnavailableEmbeddingProvider(), profile).process(
                document, ProcessingProfile(mode="token"))
            acceptance = assess(result)
            assert acceptance["status"] == "needs_review"
            assert not expected or all(expected not in chunk.text for chunk in result.chunks)
        report.append({"file": path.name, "expected_label": expected, "image_blocks": images, "acceptance": acceptance,
                       "native_text": [b.text for b in document.blocks if b.type == "TEXT"],
                       "warnings": document.warnings, "seconds": round(time.perf_counter() - started, 3)})
    output = OUT / ("baseline.json" if args.baseline else "report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.baseline:
        baseline = json.loads((OUT / "baseline.json").read_text(encoding="utf-8"))
        for old, row in zip(baseline, report, strict=True):
            assert old["file"] == row["file"] and old["native_text"] == row["native_text"]
            image, = row["image_blocks"]
            metadata = image["metadata"]
            assert metadata["unresolved"]
            assert metadata["visual_context"]["caption"]["text"] == "Figure 1. Company mark."
            assert image["bbox"] if row["file"].endswith(".pdf") else image["source"] and image["section"]
            assert metadata["image_ocr"]["text"].strip() == row["expected_label"], row
        (OUT / "rag-anything-context.json").write_text(
            json.dumps(upstream_context(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("VISUAL_CONTEXT_SMOKE_OK")


if __name__ == "__main__":
    main()
