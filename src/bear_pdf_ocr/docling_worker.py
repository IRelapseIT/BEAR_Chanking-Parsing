"""Isolated optional Docling process. No Markdown intermediary or model downloads."""
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path


def main():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    source, page, artifacts, output = sys.argv[1:]
    options = PdfPipelineOptions(artifacts_path=Path(artifacts), enable_remote_services=False)
    options.do_ocr = True
    options.ocr_options = TesseractCliOcrOptions(lang=["eng", "rus"],
                                              tesseract_cmd=os.getenv("BEAR_TESSERACT", "tesseract"))
    options.do_table_structure = True
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(Path(source), page_range=(int(page), int(page)), raises_on_error=True)
    if str(result.status.value) != "success":
        raise ValueError(f"Docling incomplete conversion: {result.status}")
    payload = {"document": result.document.export_to_dict(),
               "ordered_refs": [item.self_ref for item, _ in result.document.iterate_items()],
               "version": importlib.metadata.version("docling"), "confidence": {}}
    scores = result.confidence.pages.get(int(page))
    if scores is not None:
        payload["confidence"] = {key: float(value) for key in (
            "parse_score", "layout_score", "table_score", "ocr_score")
            if math.isfinite(value := getattr(scores, key))}
    Path(output).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
