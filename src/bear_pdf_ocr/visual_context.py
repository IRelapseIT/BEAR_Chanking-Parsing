"""Source context and review-only OCR for embedded raster images."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CAPTION = re.compile(r"^(?:fig(?:ure)?\.?|рис(?:унок)?\.?)\s*\d+\b", re.IGNORECASE)


def same_scope(left, right):
    if left.source.format != right.source.format or left.section_path != right.section_path:
        return False
    if left.source.format == "pdf":
        if left.source.page != right.source.page or not left.source.bbox or not right.source.bbox:
            return False
        a, b = left.source.bbox, right.source.bbox
        return min(a[2], b[2]) > max(a[0], b[0]) and max(a[1] - b[3], b[1] - a[3]) <= 80
    left_part = (left.source.element_path or "").split("/", 2)[:2]
    right_part = (right.source.element_path or "").split("/", 2)[:2]
    return left_part[0:1] == right_part[0:1] and (left_part[:1] != ["word"] or left_part == right_part)


def attach_visual_context(document):
    blocks = document.blocks
    pairs = {}
    captions = {}
    for index, block in enumerate(blocks):
        if block.type != "IMAGE" or not block.metadata.get("embedded_visual"):
            continue
        neighbors = [b for b in blocks[max(0, index - 1):index + 2]
                     if b.type == "TEXT" and b.role not in {"header", "footer", "heading"} and same_scope(block, b)]
        explicit = [b for b in neighbors if b.role == "caption" or CAPTION.match(b.text)]
        pairs[block.id] = (block, neighbors, explicit)
        for caption in explicit:
            captions[caption.id] = captions.get(caption.id, 0) + 1
    for block, neighbors, explicit in pairs.values():
        caption = explicit[0] if len(explicit) == 1 and captions[explicit[0].id] == 1 else None
        block.metadata["visual_context"] = {
            "caption": {"text": caption.text, "block_id": caption.id} if caption else None,
            "caption_status": "linked" if caption else "ambiguous" if explicit else "not_found",
            "neighbor_text_block_ids": [b.id for b in neighbors if b not in explicit],
            "section_path": list(block.section_path),
        }
    return document


def recognize_image(picture):
    """Recognize visible text only. Keep the original image for manual verification."""
    executable = os.environ.get("BEAR_TESSERACT") or shutil.which("tesseract")
    if not executable:
        return {"status": "unavailable", "text": ""}
    if picture.width * picture.height > 8_000_000:
        return {"status": "pixel_limit", "text": ""}
    try:
        with tempfile.TemporaryDirectory(prefix="bear-embedded-ocr-") as folder:
            path = Path(folder) / "image.png"
            picture.save(path)
            result = subprocess.run([executable, str(path), "stdout", "-l", "eng+rus",
                                     "--psm", "11", "--dpi", "216"],
                                    capture_output=True, timeout=8, check=True)
        text = result.stdout.decode("utf-8").strip()
        return {"status": "recognized" if text else "no_text_detected", "text": text,
                "engine": "tesseract", "verified": False}
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        return {"status": "failed", "text": "", "reason": type(exc).__name__}
