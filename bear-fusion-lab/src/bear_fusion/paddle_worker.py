"""PaddleOCR 3.x worker. Local model directories are required."""
import json
import os
import sys
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path

DET = "PP-OCRv5_mobile_det"
REC = "eslav_PP-OCRv5_mobile_rec"
LAYOUT = "PP-DocLayout_plus-L"


def main():
    mode = sys.argv[1]
    image, root, output = map(Path, sys.argv[2:])
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(output.resolve().parent / "paddlex-cache")
    if os.name == "nt":
        # Paddle creates its dataset cache at import time under the Windows user profile.
        os.environ["USERPROFILE"] = str(output.resolve().parent)
    if mode == "layout":
        from paddleocr import LayoutDetection

        detector = LayoutDetection(model_name=LAYOUT, model_dir=str(root), device="cpu",
                                   cpu_threads=2, enable_mkldnn=False)
        boxes = [box for result in detector.predict(str(image)) for box in result.json["res"]["boxes"]
                 if box["label"] == "table"]
        with (root / "inference.pdiparams").open("rb") as stream:
            weights = file_digest(stream, "sha256").hexdigest()
        output.write_text(json.dumps({"boxes": boxes, "model": LAYOUT, "weights_sha256": weights,
                                      "paddleocr": version("paddleocr")}), encoding="utf-8")
        return
    if mode != "ocr":
        raise ValueError("Unknown worker mode")
    revisions = {}
    for model in (DET, REC):
        directory = root / model
        manifest = directory / "bear-revision.json"
        revision = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        hashes = {}
        for name in ("inference.yml", "inference.json", "inference.pdiparams"):
            with (directory / name).open("rb") as stream:
                hashes[name] = file_digest(stream, "sha256").hexdigest()
        if revision and revision["sha256"] != hashes:
            raise ValueError(f"Model checksum mismatch for {model}")
        revisions[model] = {"revision": revision.get("revision"), "sha256": hashes}
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        device="cpu", cpu_threads=2, enable_mkldnn=False,
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        text_detection_model_name=DET, text_detection_model_dir=str(root / DET),
        text_recognition_model_name=REC, text_recognition_model_dir=str(root / REC),
    )
    lines = []
    for result in engine.predict(str(image)):
        for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"], strict=True):
            if text.strip():
                lines.append({"text": text, "confidence": float(score), "bbox_pixels": list(map(int, box))})
    profile = {"paddleocr": version("paddleocr"), "paddlepaddle": version("paddlepaddle"),
               "detection_model": DET, "recognition_model": REC, "device": "cpu", "cpu_threads": 2}
    profile.update(revisions)
    import psutil

    memory = psutil.Process().memory_info()
    output.write_text(json.dumps({"lines": lines, "profile": profile,
                                  "peak_rss_bytes": getattr(memory, "peak_wset", None)},
                                 ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
