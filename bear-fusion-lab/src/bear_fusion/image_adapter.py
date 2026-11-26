"""Standalone raster ingestion using the same OCR engines as the PDF parser."""

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from bear_text_beta.errors import ParseError
from bear_text_beta.file_validation import MIME_TYPES, FileValidator
from bear_text_beta.models import SourceLocator
from bear_text_beta.parsing import BlockBuilder
from PIL import Image, ImageOps, UnidentifiedImageError

from .ocr_verification import PaddleVerifier, primary_lines, run_worker
from .pdf_adapter import read_tsv

PIXEL_TRANSFORM = [1, 0, 0, 1, 0, 0]


def tesseract_lines(image):
    executable = os.getenv("BEAR_TESSERACT") or shutil.which("tesseract")
    if not executable:
        raise FileNotFoundError("Tesseract")
    output = image.with_suffix(".tsv")
    run_worker([executable, str(image), str(output.with_suffix("")), "-l", "eng+rus", "--psm", "11", "tsv"],
               output, 60, {**os.environ, "OMP_THREAD_LIMIT": "2"})
    _, words, _ = read_tsv(output.read_bytes())
    return primary_lines(words, PIXEL_TRANSFORM)


class ImageParser:
    def __init__(self, limits, asset_dir, object_storage=None):
        self.limits = limits
        self.asset_dir = asset_dir
        self.object_storage = object_storage

    def parse(self, data, context):
        suffix = Path(context.filename).suffix.lower()
        image_format = "PNG" if suffix == ".png" else "JPEG"
        if not data.startswith(FileValidator.SIGNATURES[suffix]):
            raise ParseError(f"Содержимое не соответствует сигнатуре {image_format}")
        builder = BlockBuilder(context)
        source = SourceLocator(suffix[1:], page=1, element_path="/image")
        try:
            with Image.open(io.BytesIO(data), formats=[image_format]) as picture:
                if picture.width * picture.height > self.limits.max_image_pixels:
                    raise ParseError("Изображение превышает лимит пикселей OCR")
                orientation = picture.getexif().get(274, 1)
                if getattr(picture, "n_frames", 1) > 1:
                    builder.warnings.append("image_ocr_first_frame_only:page:1")
                with ImageOps.exif_transpose(picture) as oriented:
                    size = oriented.size
                    with tempfile.TemporaryDirectory(prefix="bear-image-ocr-") as folder:
                        image_path = Path(folder) / "image.png"
                        with oriented.convert("RGBA") as alpha, Image.new("RGB", size, "white") as rgb:
                            rgb.paste(alpha, mask=alpha.getchannel("A"))
                            rgb.save(image_path, dpi=(216, 216))
                        lines, engine = self.recognize(image_path, size, builder.warnings)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
            raise ParseError(f"Повреждённый или неподдерживаемый {image_format}") from exc

        mime_type = MIME_TYPES[suffix]
        source.bbox = [0, 0, *size]
        if self.object_storage is None:
            builder.image(context.filename, source, data=data, mime_type=mime_type, asset_dir=self.asset_dir)
        else:
            digest = hashlib.sha256(data).hexdigest()
            key = f"attachments/{digest}"
            self.object_storage.put_stream(key, io.BytesIO(data))
            builder.image(context.filename, source, mime_type=mime_type)
            attachment = builder.attachments[-1]
            attachment.object_key = key
            attachment.metadata.update(stored=True, checksum=digest)
        image_block = builder.blocks[-1]
        geometry = {"coordinate_system": "image_pixels_exif_oriented_top_left",
                    "image_size_pixels": list(size), "exif_orientation": orientation}
        image_block.metadata.update(geometry, ocr_engine=engine, ocr_lines=len(lines))
        builder.attachments[-1].metadata.update(geometry)

        total_chars = 0
        for index, line in enumerate(lines, 1):
            text = line["text"]
            total_chars += len(text)
            if index >= self.limits.max_blocks or total_chars > self.limits.max_text_chars:
                raise ParseError("Результат OCR превышает лимит строк или символов")
            bbox = list(line["bbox_pixels"] if engine == "paddleocr" else line["bbox_pdf_points"])
            if not 0 <= bbox[0] < bbox[2] <= size[0] or not 0 <= bbox[1] < bbox[3] <= size[1]:
                builder.warnings.append("image_ocr_line_geometry_unavailable:page:1")
                bbox = None
            builder.add(text, "ocr_line", replace(source, bbox=bbox, element_path=f"/image/ocr/{index}"),
                        attachments=image_block.attachment_ids,
                        metadata={**geometry, "extraction": "ocr", "engine": engine,
                                  "ocr_confidence": line["confidence"], "verified": False})
        builder.warnings.append("image_ocr_unverified:page:1" if lines else "image_ocr_no_text:page:1")
        result = builder.finish()
        result.parser_profile = "fusion-image-ocr-v1"
        return result

    @staticmethod
    def recognize(path, size, warnings):
        lines = []
        try:
            lines = tesseract_lines(path)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            warnings.append(f"image_ocr_primary_failed:{type(exc).__name__}:page:1")
        if lines:
            return lines, "tesseract"
        paddle = PaddleVerifier.configured()
        if paddle is None:
            return [], None
        observed = paddle.verify(path, 1, [], PIXEL_TRANSFORM, size)
        if observed["secondary_lines"]:
            warnings.append("image_ocr_fallback_used:page:1")
            return observed["secondary_lines"], "paddleocr"
        if observed["status"] != "agreement":
            warnings.append("image_ocr_fallback_no_text:page:1")
        return [], None
