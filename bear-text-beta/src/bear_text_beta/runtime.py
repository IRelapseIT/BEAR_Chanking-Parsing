import io
import os
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from .chunking import ChunkingService
from .embeddings import (
    CharacterTokenCounter,
    HFTokenCounter,
    InfinityEmbeddingProvider,
    UnavailableEmbeddingProvider,
    load_profile,
)
from .file_validation import FileValidator, StoredObject
from .indexing import IndexingService, QdrantVectorStore
from .models import DocumentContext, EmbeddingProfile, ProcessingProfile, ProcessingResult
from .object_storage import LocalObjectStorage
from .parsing import ParsingPipeline
from .remote_chunking import RemoteChunkingService
from .storage import SQLiteManifest


class Services:
    def __init__(self, state_dir, model_dir, *, test_counter=False, offline=False, allow_cp1251=False):
        self.state_dir = Path(state_dir)
        if test_counter:
            self.profile = EmbeddingProfile(model_id="test-character", model_revision="test",
                                            tokenizer_revision="test")
            self.counter = CharacterTokenCounter(self.profile)
        else:
            self.profile = load_profile(Path(model_dir))
            self.counter = HFTokenCounter(Path(model_dir), self.profile)
        extra_parsers = None
        if test_counter:
            from .parsing import HTMLParser, TXTParser
            extra_parsers = {".txt": TXTParser(allow_cp1251), ".html": HTMLParser(allow_cp1251),
                             ".htm": HTMLParser(allow_cp1251)}
        self.object_storage = LocalObjectStorage(self.state_dir / "objects")
        self.parser = ParsingPipeline(self.state_dir / "assets", allow_cp1251,
                                      extra_parsers=extra_parsers,
                                      object_storage=self.object_storage)
        self.file_validator = FileValidator(self.parser.limits, set(self.parser.parsers))
        self.manifest = SQLiteManifest(self.state_dir / "manifest.sqlite3")
        self.embeddings = UnavailableEmbeddingProvider() if offline or test_counter else InfinityEmbeddingProvider(
            os.getenv("BEAR_INFINITY_URL", "http://127.0.0.1:7997/v1"), self.profile, self.counter,
            server_model=os.getenv("BEAR_INFINITY_MODEL", "bear-e5-small"),
            api_key=os.getenv("BEAR_INFINITY_API_KEY"))
        chunker_url = os.getenv("BEAR_CHUNKER_URL")
        self.chunker = RemoteChunkingService(chunker_url) if chunker_url else ChunkingService(
            self.counter, self.embeddings, self.profile)
        self.vectors = QdrantVectorStore(os.getenv("BEAR_QDRANT_URL", "http://127.0.0.1:6333"),
                                        os.getenv("BEAR_QDRANT_COLLECTION", "bear_text_e5_small_v1"),
                                        self.profile.vector_dimension,
                                        api_key=os.getenv("BEAR_QDRANT_API_KEY"))
        self.indexer = IndexingService(self.manifest, self.vectors, self.embeddings, self.counter, self.profile)

    def process(self, data: bytes, context: DocumentContext,
                profile: ProcessingProfile) -> ProcessingResult:
        return self.process_stream(io.BytesIO(data), context, profile)

    def process_stream(self, stream: BinaryIO, context: DocumentContext,
                       profile: ProcessingProfile) -> ProcessingResult:
        stored = self.upload_stream(stream, context)
        return self.process_stored(stored, context, profile)

    def upload_stream(self, stream: BinaryIO, context: DocumentContext) -> StoredObject:
        key = f"originals/{uuid4()}"
        return self.file_validator.validate_and_store(
            stream, context.filename, key, self.object_storage, context.checksum)

    def process_stored(self, stored: StoredObject, context: DocumentContext,
                       profile: ProcessingProfile, generation_id: str | None = None) -> ProcessingResult:
        if generation_id:
            from .errors import StorageError
            try:
                return self.manifest.load(generation_id)
            except StorageError:
                pass
        if context.checksum and context.checksum != stored.checksum:
            from .errors import ParseError
            raise ParseError("Checksum не соответствует сохранённому объекту")
        context.checksum = stored.checksum
        with self.object_storage.get_stream(stored.object_key) as source:
            data = source.read(self.parser.max_file_bytes + 1)
        document = self.parser.parse(data, context)
        result = self.chunker.process(document, profile, generation_id)
        result.metrics["original_object_key"] = stored.object_key
        result.metrics["original_size_bytes"] = stored.size_bytes
        import tracemalloc
        if tracemalloc.is_tracing():
            result.metrics["python_peak_allocated_bytes"] = tracemalloc.get_traced_memory()[1]
        self.manifest.save(result)
        return result

    def close(self):
        for component in (self.embeddings, self.vectors, self.chunker,
                          self.parser.parsers[".docx"]):
            if hasattr(component, "close"):
                component.close()


def default_services():
    return Services(os.getenv("BEAR_STATE_DIR", "state"), os.getenv("BEAR_MODEL_DIR", "models/e5"),
                    test_counter=os.getenv("BEAR_TEST_COUNTER") == "1",
                    offline=os.getenv("BEAR_OFFLINE") == "1",
                    allow_cp1251=os.getenv("BEAR_ALLOW_CP1251") == "1")
