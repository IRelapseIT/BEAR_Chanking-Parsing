"""Independent OCR evidence in the PDF coordinate system."""
import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import defaultdict
from contextlib import closing
from pathlib import Path


def pdf_box(box, transform):
    x0, y0, x1, y1 = box
    a, b, c, d, ox, oy = transform
    corners = [(a * x + c * y + ox, b * x + d * y + oy)
               for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    xs, ys = zip(*corners)
    return [min(xs), min(ys), max(xs), max(ys)]


def primary_lines(words, transform):
    groups = defaultdict(list)
    for word in words:
        groups[tuple(word["line_id"])].append(word)
    lines = []
    for line in groups.values():
        boxes = [word["bbox_pixels"] for word in line]
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)]
        lines.append({"text": " ".join(word["text"] for word in line),
                      "confidence": sum(word["confidence"] for word in line) / (100 * len(line)),
                      "bbox_pdf_points": pdf_box(box, transform)})
    return lines


def normalized(text):
    # Keep signs, decimal separators, units and order. OCR agreement is evidence only.
    text = unicodedata.normalize("NFC", text).replace("−", "-").casefold()
    text = re.sub(r"(?<=[-+])\s+(?=\d)|(?<=\d)\s+(?=[%‰])", "", text)
    return " ".join(text.split())


def overlap(first, second):
    a, b, c, d = first
    e, f, g, h = second
    intersection = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    # Require coverage of both boxes, avoiding a match to one word of a long line.
    return intersection / max((c - a) * (d - b), (g - e) * (h - f), 1e-9)


def compare_lines(primary, secondary):
    edges = [[j for j, b in enumerate(secondary)
              if overlap(a["bbox_pdf_points"], b["bbox_pdf_points"]) >= .45] for a in primary]
    counts = defaultdict(int)
    for row in edges:
        for j in row:
            counts[j] += 1
    matched = [(i, row[0]) for i, row in enumerate(edges) if len(row) == 1 and counts[row[0]] == 1]
    conflicts = [{"primary": i, "secondary": j, "reason": "text_differs"}
                 for i, j in matched if normalized(primary[i]["text"]) != normalized(secondary[j]["text"])]
    reasons = []
    if conflicts:
        reasons.append("aligned_text_differs")
    if len(matched) != len(primary) or len(matched) != len(secondary):
        reasons.append("missing_or_ambiguous_geometry")
    if not matched:
        reasons.append("no_aligned_lines")
    status = "disagreement" if conflicts else "insufficient" if reasons else "agreement"
    return {"status": status, "matches": [list(pair) for pair in matched],
            "conflicts": conflicts, "reasons": reasons}


def workspace_size(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def run_worker(command, output, budget, env, *, max_output_bytes=2 * 1024 * 1024,
               max_memory_bytes=2 * 1024**3, work_dir=None, max_work_bytes=None):
    import psutil

    deadline = time.monotonic() + budget
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env) as process:
        parent = psutil.Process(process.pid)
        children = []
        try:
            while process.poll() is None:
                try:
                    children = parent.children(recursive=True)
                    rss = sum(item.memory_info().rss for item in [parent, *children])
                except psutil.NoSuchProcess:
                    rss = 0
                if rss > max_memory_bytes:
                    raise ValueError("secondary_memory_limit")
                if max_work_bytes and workspace_size(work_dir) > max_work_bytes:
                    raise ValueError("secondary_disk_limit")
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, budget)
                time.sleep(.1)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except psutil.Error as exc:
            raise ValueError("secondary_process_monitor_failed") from exc
        finally:
            if process.poll() is None:
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                process.kill()
    if output.stat().st_size > max_output_bytes:
        raise ValueError("secondary_output_limit")
    if max_work_bytes and workspace_size(work_dir) > max_work_bytes:
        raise ValueError("secondary_disk_limit")


class PaddleVerifier:
    def __init__(self, python, models, max_regions=8, timeout=60, *, layout_model=None):
        self.python = str(Path(python).resolve())
        self.models = Path(models).resolve()
        self.remaining = max_regions
        self.timeout = timeout
        self.deadline = None
        self.layout_model = Path(layout_model).resolve() if layout_model else None
        self.layout_remaining = 2

    @classmethod
    def configured(cls):
        python, models = os.getenv("BEAR_PADDLE_PYTHON"), os.getenv("BEAR_PADDLE_MODELS")
        return cls(python, models, layout_model=os.getenv("BEAR_PADDLE_LAYOUT_MODEL")) if python and models else None

    def invoke(self, mode, image, model, budget):
        output = image.with_suffix(".paddle.json")
        run_worker([self.python, str(Path(__file__).with_name("paddle_worker.py")), mode,
                    str(image), str(model), str(output)], output, budget,
                   {**os.environ, "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
                    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"})
        return json.loads(output.read_text(encoding="utf-8"))

    def detect_tables(self, renderer, index):
        if self.layout_model is None or self.layout_remaining <= 0:
            return {"status": "not_run", "boxes": []}
        self.layout_remaining -= 1
        try:
            with (tempfile.TemporaryDirectory(prefix="bear-layout-") as folder, closing(renderer[index]) as page):
                width, height = page.get_size()
                if width * height * 4 > 30_000_000:
                    raise ValueError("Layout raster pixel limit exceeded")
                with closing(page.render(scale=2)) as bitmap:
                    image = Path(folder) / "page.png"
                    with bitmap.to_pil() as picture:
                        picture.save(image)
                    payload = self.invoke("layout", image, self.layout_model, 45)
            boxes = []
            for item in payload["boxes"]:
                confidence = item["score"]
                box = pdf_box(item["coordinate"], [.5, 0, 0, .5, 0, 0])
                x0, y0, x1, y1 = box
                if (.5 <= confidence <= 1 and 0 <= x0 < x1 <= width + .5
                        and 0 <= y0 < y1 <= height + .5):
                    boxes.append({"bbox_pdf_points": box, "confidence": confidence})
            return {**payload, "status": "detected" if boxes else "empty", "boxes": boxes}
        except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
            return {"status": "insufficient", "boxes": [], "reason": type(exc).__name__}

    def verify(self, image, page, words, transform, size):
        started = time.monotonic()
        record = {"page": page, "primary_engine": "tesseract", "secondary_engine": "paddleocr",
                  "primary_lines": primary_lines(words, transform), "secondary_lines": [],
                  "raster_size": list(size), "pixel_to_pdf_transform": transform,
                  "secondary_rotation_degrees": 0, "secondary_deskew_applied": False}
        if self.deadline is None:
            self.deadline = started + 120
        budget = min(self.timeout, self.deadline - started)
        if self.remaining <= 0 or budget <= 0:
            return {**record, "status": "not_run", "reasons": ["verification_budget_exhausted"]}
        self.remaining -= 1
        try:
            payload = self.invoke("ocr", image, self.models, budget)
            lines = payload["lines"]
            if len(lines) > 1024:
                raise ValueError("OCR line limit exceeded")
            secondary = []
            for line in lines:
                x0, y0, x1, y1 = box = line["bbox_pixels"]
                confidence = line["confidence"]
                if (not all(math.isfinite(v) for v in box) or not 0 <= x0 < x1 <= size[0]
                        or not 0 <= y0 < y1 <= size[1] or not 0 <= confidence <= 1
                        or not isinstance(line["text"], str) or not line["text"].strip()):
                    raise ValueError("Invalid OCR observation")
                secondary.append({**line, "bbox_pdf_points": pdf_box(box, transform)})
            record.update(secondary_lines=secondary, secondary_profile=payload["profile"],
                          secondary_peak_rss_bytes=payload.get("peak_rss_bytes"))
            record.update(compare_lines(record["primary_lines"], secondary))
        except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            record.update(status="insufficient", reasons=[reason[:200]])
        return {**record, "elapsed_seconds": round(time.monotonic() - started, 3)}
