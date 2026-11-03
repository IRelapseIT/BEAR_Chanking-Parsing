import argparse
import json
from pathlib import Path

from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import HFTokenCounter, InfinityEmbeddingProvider
from bear_text_beta.errors import EmbeddingError
from bear_text_beta.models import DocumentContext, EmbeddingProfile
from bear_text_beta.object_storage import LocalObjectStorage

from .bridge import ParserProcess
from .pipeline import FusionPipeline


class UnavailableEmbeddings:
    def embed_batch(self, texts, purpose):
        raise EmbeddingError("Explicit offline mode: semantic inference unavailable")


def main():
    parser = argparse.ArgumentParser(description="Experimental BEAR fusion; no indexing")
    parser.add_argument("file", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--parser-python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument("--version", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--infinity-url")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    profile = EmbeddingProfile()
    counter = HFTokenCounter(args.tokenizer, profile)
    embeddings = (UnavailableEmbeddings() if args.offline else
                  InfinityEmbeddingProvider(args.infinity_url, profile, counter))
    try:
        service = FusionPipeline(ParserProcess(args.upstream, args.parser_python),
                                 ChunkingService(counter, embeddings, profile),
                                 LocalObjectStorage(args.state))
        result = service.process_path(args.file, DocumentContext(
            args.tenant, args.space, args.document, args.version, args.file.name))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(f"Saved experimental result: {args.output}; indexing disabled")
    finally:
        if isinstance(embeddings, InfinityEmbeddingProvider):
            embeddings.close()
