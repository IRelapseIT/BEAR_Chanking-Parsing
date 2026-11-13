import hashlib
import json

from bear_pdf_ocr.cli import main
from reportlab.pdfgen import canvas


def test_cli_writes_text_and_source_coordinates(tmp_path):
    source = tmp_path / "document.pdf"
    writer = canvas.Canvas(str(source))
    writer.drawString(50, 700, "The reference number is 27.")
    writer.save()

    output = tmp_path / "parsed.json"
    assert main([str(source), "-o", str(output)]) == 0
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["context"]["checksum"] == hashlib.sha256(source.read_bytes()).hexdigest()
    text, = parsed["blocks"]
    assert text["text"] == "The reference number is 27."
    assert text["source"]["page"] == 1
    assert len(text["source"]["bbox"]) == 4
