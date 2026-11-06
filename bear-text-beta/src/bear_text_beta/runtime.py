import io
import os
from importlib.util import find_spec
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from .chunking import ChunkingService
from .embedding_pipeline import EmbeddingPipeline, MemoryVectorCache
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
        extra_parsers = {}
        external_processors = {}
        if find_spec("bear_fusion") is not None:
            from bear_fusion.markdown_adapter import StructuredMarkdownParser
            from bear_fusion.tables import TableProcessor
            extra_parsers[".md"] = StructuredMarkdownParser()
            external_processors["TABLE"] = TableProcessor(self.counter)
            from .file_validation import FileLimits
            from .unified_parsing import DOCXParser, PDFParser
            extra_parsers.update({".docx": DOCXParser(), ".pdf": PDFParser(FileLimits())})
        self.object_storage = LocalObjectStorage(self.state_dir / "objects")
        self.parser = ParsingPipeline(self.state_dir / "assets", allow_cp1251,
                                      extra_parsers=extra_parsers,
                                      object_storage=self.object_storage)
        self.file_validator = FileValidator(self.parser.limits, set(self.parser.parsers))
        self.manifest = SQLiteManifest(self.state_dir / "manifest.sqlite3")
        if offline or test_counter:
            self.embeddings = UnavailableEmbeddingProvider()
        else:
            server_model = ("bear-e5-small" if self.profile.model_id == "intfloat/multilingual-e5-small"
                            else self.profile.model_id)
            provider = InfinityEmbeddingProvider(
                os.getenv("BEAR_INFINITY_URL", "http://127.0.0.1:7998/v1"), self.profile, self.counter,
                server_model=os.getenv("BEAR_INFINITY_MODEL", server_model),
                api_key=os.getenv("BEAR_INFINITY_API_KEY"))
            self.embeddings = EmbeddingPipeline(provider, self.counter, self.profile,
                                                max_tokens=provider.max_tokens,
                                                cache=MemoryVectorCache())
        chunker_url = os.getenv("BEAR_CHUNKER_URL")
        self.external_processors = external_processors
        validators = {kind: processor for kind, processor in external_processors.items()
                      if callable(getattr(processor, "validate", None))}
        self.chunker = (RemoteChunkingService(chunker_url, counter=self.counter,
                                            embedding_profile=self.profile, external_validators=validators)
                        if chunker_url else ChunkingService(
                            self.counter, self.embeddings, self.profile, external_processors))
        self.vectors = QdrantVectorStore(os.getenv("BEAR_QDRANT_URL", "http://127.0.0.1:6335"),
                                        os.getenv("BEAR_QDRANT_COLLECTION", (
                                            "bear_text_e5_small_v1"
                                            if self.profile.model_id == "intfloat/multilingual-e5-small"
                                            else ("bear_bge_m3_" if self.profile.model_id == "BAAI/bge-m3" else "bear_text_")
                                            + self.profile.identifier())),
                                        self.profile.vector_dimension,
                                        api_key=os.getenv("BEAR_QDRANT_API_KEY"))
        self.indexer = IndexingService(self.manifest, self.vectors, self.embeddings, self.counter,
                                      self.profile, external_validators=validators)

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
                       profile: ProcessingProfile, generation_id: str | None = None, *,
                       processing_claim: tuple[str, str] | None = None) -> ProcessingResult:
        if processing_claim is not None:
            self.manifest.assert_processing_claim(*processing_claim)
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
        if processing_claim:
            self.manifest.set_processing_phase(*processing_claim, "chunking")
        for processor in self.external_processors.values():
            if hasattr(processor, "budget"):
                processor.budget = min(profile.target_tokens, profile.max_tokens,
                                       self.profile.max_input_tokens)
        result = self.chunker.process(document, profile, generation_id)
        from .acceptance import assess
        if find_spec("bear_fusion") is not None:
            from bear_fusion.parents import attach_parents
            result.metrics["parents"] = attach_parents(result)
        result.metrics["acceptance"] = assess(result)
        result.metrics["original_object_key"] = stored.object_key
        result.metrics["original_size_bytes"] = stored.size_bytes
        import tracemalloc
        if tracemalloc.is_tracing():
            result.metrics["python_peak_allocated_bytes"] = tracemalloc.get_traced_memory()[1]
        if processing_claim is None:
            self.manifest.save(result)
        else:
            self.manifest.save_processing_result(*processing_claim, result)
        return result

    def close(self):
        for component in (self.embeddings, self.vectors, self.chunker,
                          self.parser.parsers[".docx"]):
            if hasattr(component, "close"):
                component.close()


def default_services():
    return Services(os.getenv("BEAR_STATE_DIR", "state-bge-m3"), os.getenv("BEAR_MODEL_DIR", "models/bge-m3"),
                    test_counter=os.getenv("BEAR_TEST_COUNTER") == "1",
                    offline=os.getenv("BEAR_OFFLINE") == "1",
                    allow_cp1251=os.getenv("BEAR_ALLOW_CP1251") == "1")
