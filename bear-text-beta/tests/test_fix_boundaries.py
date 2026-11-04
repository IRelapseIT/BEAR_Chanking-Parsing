import copy
import json
from dataclasses import asdict
from uuid import uuid4

import httpx
import pytest
from pydantic import TypeAdapter

from bear_text_beta.chunking import ChunkContextBuilder, ChunkingService, ChunkValidator
from bear_text_beta.errors import ServiceUnavailableError, ValidationError
from bear_text_beta.models import (
    ContentBlock,
    ParsedDocument,
    PreparedFragment,
    ProcessingProfile,
    SourceSpan,
)
from bear_text_beta.remote_chunking import RemoteChunkingService


def alter_text(result, text, counter):
    chunk = result.chunks[0]
    chunk.text = text
    chunk.embedding_text = ChunkContextBuilder.embedding_text(text, chunk.section_path, counter)
    chunk.token_count = counter.full_count(chunk.embedding_text)


@pytest.mark.parametrize("strategy", ["whole", "external"])
@pytest.mark.parametrize("text", ["INVENTED source text", "source text INVENTED", "source INVENTED text"])
def test_text_cannot_gain_unsourced_content(parsed, chunker, counter, strategy, text):
    result = chunker.process(parsed("source text"))
    result.chunks[0].strategy = strategy
    alter_text(result, text, counter)
    with pytest.raises(ValidationError):
        ChunkValidator(counter, 480).validate(result)


def test_exact_external_text_still_requires_and_preserves_its_source(parsed, chunker, counter):
    result = chunker.process(parsed("source text"))
    result.chunks[0].strategy = "external"
    ChunkValidator(counter, 480).validate(result)


def test_external_cannot_mix_table_and_text_to_skip_trailing_check(parsed, chunker, counter):
    result = chunker.process(parsed("source text"))
    text_block = result.parsed_document.blocks[0]
    result.parsed_document.blocks.append(ContentBlock(
        "table", 1, "TABLE", "table", "", [], copy.deepcopy(text_block.source)))
    result.chunks[0].strategy = "external"
    result.chunks[0].source_spans.append(SourceSpan("table", 0, 0))
    alter_text(result, "source text INVENTED", counter)
    with pytest.raises(ValidationError, match="смешивать"):
        ChunkValidator(counter, 480).validate(result)


def test_chunk_cannot_invent_section_path(parsed, chunker, counter):
    result = chunker.process(parsed("source text"))
    result.chunks[0].section_path = ["Invented heading"]
    alter_text(result, "source text", counter)
    with pytest.raises(ValidationError, match="Раздел"):
        ChunkValidator(counter, 480).validate(result)


class VerifiedTable:
    def __init__(self):
        self.validated = False

    def process(self, block, context):
        return [PreparedFragment("A: 1", [SourceSpan(block.id, 0, 0)])]

    def validate(self, document, chunks, counter, maximum):
        self.validated = True
        tables = {b.id for b in document.blocks if b.type == "TABLE"}
        table_chunks = [c for c in chunks if any(s.block_id in tables for s in c.source_spans)]
        if len(table_chunks) != 1 or table_chunks[0].text != "A: 1":
            raise ValidationError("Canonical table value not covered")


def test_external_processor_needs_independent_registered_validation(parsed, counter, embeddings, profile):
    class UncheckedTable:
        process = VerifiedTable.process

    document = parsed("| A |\n| --- |\n| 1 |\n", "table.md")
    with pytest.raises(ValidationError, match="валидатор"):
        ChunkingService(counter, embeddings, profile, {"TABLE": UncheckedTable()}).process(document)
    processor = VerifiedTable()
    service = ChunkingService(counter, embeddings, profile, {"TABLE": processor})
    result = service.process(document)
    assert processor.validated
    ChunkValidator(counter, 480, service.external_validators).validate(result)
    result.chunks[0].text = "A: 999"
    alter_text(result, "A: 999", counter)
    with pytest.raises(ValidationError, match="Canonical"):
        ChunkValidator(counter, 480, service.external_validators).validate(result)
    with pytest.raises(ValidationError, match="валидатор"):
        ChunkValidator(counter, 480).validate(result)


def test_external_value_error_is_a_typed_validation_failure(parsed, counter, embeddings, profile):
    class InvalidTable(VerifiedTable):
        def validate(self, document, chunks, counter, maximum):
            raise ValueError("invalid canonical cells")

    with pytest.raises(ValidationError, match="Внешнее содержимое"):
        ChunkingService(counter, embeddings, profile, {"TABLE": InvalidTable()}).process(
            parsed("| A |\n| --- |\n| 1 |\n", "table.md"))


def remote_client(counter, embeddings, profile, mutate=None, **options):
    def handler(request):
        payload = json.loads(request.content)
        document = TypeAdapter(ParsedDocument).validate_python(payload["document"])
        result = ChunkingService(counter, embeddings, profile).process(
            document, ProcessingProfile(**payload["profile"]), payload["generation_id"])
        body = asdict(result)
        if mutate:
            mutate(body)
        return httpx.Response(200, json=body)

    client = httpx.Client(base_url="http://chunker", transport=httpx.MockTransport(handler))
    return RemoteChunkingService("http://chunker", client, counter=counter, embedding_profile=profile, **options)


@pytest.mark.parametrize("explicit_generation", [False, True])
def test_remote_accepts_original_request_and_keeps_generated_identity(
        parsed, counter, embeddings, profile, explicit_generation):
    document = parsed("Original source text")
    generation = str(uuid4()) if explicit_generation else None
    service = remote_client(counter, embeddings, profile)
    try:
        result = service.process(document, generation_id=generation)
    finally:
        service.close()
    assert result.parsed_document == document
    if generation:
        assert result.generation_id == generation
    assert result.chunks[0].text == "Original source text"


def mutate_context(field):
    def apply(body):
        value = 77 if field == "document_version" else "foreign"
        body["context"][field] = value
        body["parsed_document"]["context"][field] = value
    return apply


MUTATIONS = [
    *[(name, mutate_context(name)) for name in
      ["tenant_id", "space_id", "document_id", "document_version", "checksum", "filename", "mime_type"]],
    ("generation", lambda b: b.update(generation_id=str(uuid4()))),
    ("parser_profile", lambda b: b["parsed_document"].update(parser_profile="other")),
    ("source_text", lambda b: b["parsed_document"]["blocks"][0].update(text="changed source")),
    ("source_metadata", lambda b: b["parsed_document"]["blocks"][0]["metadata"].update(invented=True)),
    ("processing_id", lambda b: b.update(processing_profile_id="foreign")),
    ("processing_profile", lambda b: b["metrics"]["processing_profile"].update(target_tokens=11)),
    ("embedding_profile", lambda b: b["metrics"]["embedding_profile"].update(model_id="foreign")),
    ("embedding_id", lambda b: b["metrics"].update(embedding_profile_id="foreign")),
    ("counter", lambda b: b["metrics"].update(token_counter="ForeignCounter")),
]


@pytest.mark.parametrize("name,mutate", MUTATIONS, ids=[name for name, _ in MUTATIONS])
def test_remote_rejects_response_identity_or_profile_drift(parsed, counter, embeddings, profile, name, mutate):
    service = remote_client(counter, embeddings, profile, mutate)
    try:
        with pytest.raises(ValidationError):
            service.process(parsed("Original source text"))
    finally:
        service.close()


def test_remote_revalidates_chunk_sources_locally(parsed, counter, embeddings, profile):
    def corrupt(body):
        chunk = body["chunks"][0]
        chunk["strategy"] = "external"
        chunk["text"] += " UNSOURCED"
        chunk["embedding_text"] = chunk["text"]
        chunk["token_count"] = counter.full_count(chunk["embedding_text"])

    service = remote_client(counter, embeddings, profile, corrupt)
    try:
        with pytest.raises(ValidationError, match="без источника"):
            service.process(parsed("Original source text"))
    finally:
        service.close()


def test_remote_requires_local_validation_configuration(parsed):
    calls = []
    client = httpx.Client(base_url="http://chunker", transport=httpx.MockTransport(lambda r: calls.append(r)))
    service = RemoteChunkingService("http://chunker", client)
    try:
        with pytest.raises(ValidationError, match="локальные"):
            service.process(parsed("text"))
    finally:
        service.close()
    assert not calls


@pytest.mark.parametrize("declared", [False, True])
def test_remote_response_limit_applies_before_json_decode(parsed, counter, profile, declared):
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{" * 32
            yield b"x" * 32
            raise AssertionError("Reader must stop as soon as the limit is exceeded")

    headers = {"content-length": "64"} if declared else {}
    client = httpx.Client(base_url="http://chunker", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers=headers, stream=Chunks())))
    service = RemoteChunkingService("http://chunker", client, counter=counter,
                                   embedding_profile=profile, max_response_bytes=40)
    try:
        with pytest.raises(ValidationError, match="лимит"):
            service.process(parsed("text"))
    finally:
        service.close()


@pytest.mark.parametrize("status,error", [(429, ServiceUnavailableError), (503, ServiceUnavailableError),
                                         (400, ValidationError)])
def test_remote_status_failure_mapping(parsed, counter, profile, status, error):
    client = httpx.Client(base_url="http://chunker", transport=httpx.MockTransport(
        lambda r: httpx.Response(status, json={"error": "unavailable"})))
    service = RemoteChunkingService("http://chunker", client, counter=counter, embedding_profile=profile)
    try:
        with pytest.raises(error):
            service.process(parsed("text"))
    finally:
        service.close()
