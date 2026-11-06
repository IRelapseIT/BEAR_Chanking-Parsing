"""One document per disposable process; monitored by the local demo server."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "bear-text-beta/src")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import HFTokenCounter, InfinityEmbeddingProvider, load_profile
from bear_text_beta.models import DocumentContext
from bear_text_beta.object_storage import LocalObjectStorage

from bear_fusion.bridge import ParserProcess
from bear_fusion.cli import UnavailableEmbeddings
from bear_fusion.pipeline import FusionPipeline
from bear_fusion.quality import inspect_result
from bear_fusion.tables import TableProcessor


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main():
    job = Path(sys.argv[1]).resolve()
    (job / "temp").mkdir(exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(job / "temp")
    import tempfile
    tempfile.tempdir = str(job / "temp")
    config = json.loads((job / "request.json").read_text("utf-8"))
    source = job / config["filename"]
    profile = load_profile(Path(config["tokenizer"]))
    counter = HFTokenCounter(Path(config["tokenizer"]), profile)
    url = os.environ.get("BEAR_INFINITY_URL")
    provider = InfinityEmbeddingProvider(url, profile, counter) if url else UnavailableEmbeddings()
    chunker = ChunkingService(counter, provider, profile, external_processors={"TABLE": TableProcessor()})

    def stage(name, data):
        save(job / f"{name}.json", data)

    try:
        pipeline = FusionPipeline(ParserProcess(Path(config["upstream"]), Path(config["parser_python"])),
                                  chunker, LocalObjectStorage(job / "objects"), on_stage=stage)
        result = pipeline.process_path(source, DocumentContext("demo", "local", job.name, 1, source.name))
        result["quality"] = inspect_result(result["result"])
        result["inference_mode"] = "infinity" if url else "offline-fallback"
        save(job / "result.json", result)
    finally:
        if isinstance(provider, InfinityEmbeddingProvider):
            provider.close()


if __name__ == "__main__":
    main()
