"""Raster crops in the displayed PDF page's top-left coordinate system."""
import os
from contextlib import closing, contextmanager
from math import ceil, floor


@contextmanager
def raster_region(renderer, index, box=None):
    with closing(renderer[index]) as page:
        width, height = page.get_size()
        dpi = int(os.getenv("BEAR_OCR_DPI", "216"))
        if not 72 <= dpi <= 400:
            raise ValueError("OCR DPI must be between 72 and 400")
        scale = dpi / 72
        if ceil(width * scale) * ceil(height * scale) > 30_000_000:
            raise ValueError("PDF raster pixel limit exceeded")
        # PDFium applies the intrinsic page rotation before rasterization.
        with closing(page.render(scale=scale)) as bitmap, bitmap.to_pil() as full:
            left, top, right, bottom = (0, 0, *full.size)
            if box:
                left, top = max(0, floor(box[0] * scale)), max(0, floor(box[1] * scale))
                right = min(right, ceil(box[2] * scale))
                bottom = min(bottom, ceil(box[3] * scale))
            if left >= right or top >= bottom:
                raise ValueError("Empty OCR crop")
            with full.crop((left, top, right, bottom)) as crop:
                yield crop, [1 / scale, 0, 0, 1 / scale, left / scale, top / scale], dpi, (width, height)


def visual_state(image):
    """Remove long straight rules before testing for text-like ink."""
    with image.convert("L") as gray:
        gray.thumbnail((700, 700))
        width, height = gray.size
        pixels = gray.tobytes()
    ink = {(i % width, i // width) for i, value in enumerate(pixels) if value < 190}
    if not ink:
        return "blank" if min(pixels) > 225 else "ambiguous"
    columns, rows = [0] * width, [0] * height
    for x, y in ink:
        columns[x] += 1
        rows[y] += 1
    marks = [(x, y) for x, y in ink if rows[y] < .60 * width and columns[x] < .82 * height]
    if not marks:
        return "blank"
    if len(marks) < 6 or len(marks) > .40 * width * height:
        return "ambiguous"
    return "marks"
