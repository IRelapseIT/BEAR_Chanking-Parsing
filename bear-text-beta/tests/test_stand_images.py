import io
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from bear_text_beta.errors import ParseError
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.parsing import ParsingPipeline


def make_ocr_image(format, *, orientation=1, phrase="BEAR SCAN CHECK"):
    buffer = io.BytesIO()
    with Image.new("RGB", (1100, 220), "white") as image:
        fonts = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
        path = next((path for path in fonts if path.is_file()), None)
        font = ImageFont.truetype(str(path), 65) if path else ImageFont.load_default(size=65)
        ImageDraw.Draw(image).text((30, 60), phrase, font=font, fill="black")
        exif = image.getexif()
        exif[274] = orientation
        image.save(buffer, format=format, exif=exif)
    return buffer.getvalue()


def test_standalone_images_use_common_parser_registry(tmp_path):
    pipeline = ParsingPipeline(tmp_path)
    assert set(pipeline.parsers) >= {".png", ".jpg", ".jpeg"}


def test_jpeg_exif_geometry_and_paddle_on_primary_failure(tmp_path, context, monkeypatch):
    from bear_fusion import image_adapter

    def fail(path):
        raise subprocess.TimeoutExpired("tesseract", 60)

    class Secondary:
        def verify(self, path, page, words, transform, size):
            assert size == (220, 1100)
            return {"secondary_lines": [{"text": "replacement OCR", "confidence": .8,
                                         "bbox_pixels": [5, 10, 120, 30]}], "status": "insufficient"}

    monkeypatch.setattr(image_adapter, "tesseract_lines", fail)
    monkeypatch.setattr(image_adapter.PaddleVerifier, "configured", lambda: Secondary())
    parsed = ParsingPipeline(tmp_path).parse(
        make_ocr_image("JPEG", orientation=6), replace(context, filename="document.jpeg"),
    )
    assert parsed.context.mime_type == "image/jpeg"
    image, text = parsed.blocks
    assert image.metadata["exif_orientation"] == 6
    assert image.metadata["image_size_pixels"] == [220, 1100]
    assert text.source.bbox == [5, 10, 120, 30]
    assert text.source.page == 1 and text.text == "replacement OCR"
    assert text.metadata["engine"] == "paddleocr"
    assert text.attachment_ids == image.attachment_ids == [parsed.attachments[0].id]
    assert any(w.startswith("image_ocr_primary_failed:") for w in parsed.warnings)


def test_image_no_text_keeps_attachment_without_invented_text(tmp_path, context, monkeypatch):
    from bear_fusion import image_adapter

    monkeypatch.setattr(image_adapter, "tesseract_lines", lambda path: [])
    monkeypatch.setattr(image_adapter.PaddleVerifier, "configured", lambda: None)
    parsed = ParsingPipeline(tmp_path).parse(make_ocr_image("PNG"), replace(context, filename="blank.png"))
    block, = parsed.blocks
    assert block.type == "IMAGE" and block.text == ""
    assert parsed.attachments[0].metadata["stored"] is True
    assert "image_ocr_no_text:page:1" in parsed.warnings


@pytest.mark.parametrize(("suffix", "payload"), [(".jpg", b"not JPG"), (".png", b"not PNG")])
def test_image_rejects_bad_format(tmp_path, context, suffix, payload):
    with pytest.raises(ParseError, match="PNG|JPEG"):
        ParsingPipeline(tmp_path).parse(payload, replace(context, filename="image" + suffix))


def test_image_pixel_budget(tmp_path, context):
    with pytest.raises(ParseError, match="лимит пикселей"):
        ParsingPipeline(tmp_path, limits=FileLimits(max_image_pixels=1000)).parse(
            make_ocr_image("PNG"), replace(context, filename="image.png"),
        )


def test_tesseract_raster_without_mock(tmp_path, context, monkeypatch):
    executable = shutil.which("tesseract")
    if not executable:
        pytest.skip("No local Tesseract")
    languages = subprocess.run([executable, "--list-langs"], capture_output=True, text=True, timeout=10, check=True)
    if not {"eng", "rus"} <= set(languages.stdout.splitlines()):
        pytest.skip("Tesseract needs eng and rus")
    monkeypatch.setenv("BEAR_TESSERACT", executable)
    parsed = ParsingPipeline(tmp_path).parse(make_ocr_image("PNG"), replace(context, filename="scan.png"))
    assert "BEAR SCAN CHECK" in "\n".join(block.text for block in parsed.blocks if block.type == "TEXT")
    assert all(block.source.bbox and block.metadata["engine"] == "tesseract"
               for block in parsed.blocks if block.type == "TEXT")
