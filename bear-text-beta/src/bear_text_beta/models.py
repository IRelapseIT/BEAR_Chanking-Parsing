from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class DocumentContext:
    tenant_id: str
    space_id: str
    document_id: str
    document_version: int
    filename: str
    mime_type: str = ""
    checksum: str = ""


@dataclass
class SourceLocator:
    format: str
    line_start: int | None = None
    line_end: int | None = None
    element_path: str | None = None
    paragraph_index: int | None = None
    page: int | None = None
    bbox: list[float] | None = None


@dataclass
class SourceSpan:
    block_id: str
    start_char: int
    end_char: int


@dataclass
class Attachment:
    id: str
    type: str
    object_key: str
    mime_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ContentBlock:
    id: str
    order: int
    type: Literal["TEXT", "TABLE", "IMAGE"]
    role: str
    text: str
    section_path: list[str]
    source: SourceLocator
    attachment_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    context: DocumentContext
    blocks: list[ContentBlock]
    schema_version: Literal[1] = 1
    parser_profile: str = "text-v1"
    attachments: list[Attachment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    sequence: int
    text: str
    embedding_text: str
    token_count: int
    source_spans: list[SourceSpan]
    section_path: list[str]
    attachment_ids: list[str]
    strategy: str
    fallback_reason: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    context: DocumentContext
    processing_profile_id: str
    generation_id: str
    parsed_document: ParsedDocument
    chunks: list[Chunk]
    deferred_blocks: list[ContentBlock]
    warnings: list[str]
    completeness: Literal["complete", "partial"]
    schema_version: Literal[1] = 1
    metrics: dict = field(default_factory=dict)


@dataclass
class PreparedFragment:
    text: str
    source_spans: list[SourceSpan]
    attachment_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]
    embedding_profile_id: str


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


@dataclass(frozen=True)
class EmbeddingProfile:
    model_id: str = "intfloat/multilingual-e5-small"
    model_revision: str = ""
    tokenizer_revision: str = ""
    vector_dimension: int = 384
    max_input_tokens: int = 512
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    similarity_prefix: str = "query: "

    def prefix(self, purpose: str) -> str:
        if purpose not in {"query", "document", "similarity"}:
            raise ValueError(f"Unknown embedding purpose: {purpose}")
        return getattr(self, f"{purpose}_prefix")

    def identifier(self) -> str:
        import json
        from hashlib import sha256
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class ProcessingProfile:
    mode: Literal["adaptive", "fixed", "token", "semantic"] = "adaptive"
    target_tokens: int = 320
    max_tokens: int = 480
    semantic_region_tokens: int = 2048
    similarity_window: int = 2
    threshold: float = 0.8
    fixed_characters: int = 1000

    def __post_init__(self):
        if self.mode not in {"adaptive", "fixed", "token", "semantic"}:
            raise ValueError("Unknown chunking mode")
        if not 0 < self.target_tokens <= self.max_tokens or self.semantic_region_tokens < self.max_tokens:
            raise ValueError("Invalid token budgets")
        if self.similarity_window < 1 or not 0 < self.threshold < 1 or self.fixed_characters < 1:
            raise ValueError("Invalid chunking parameters")
