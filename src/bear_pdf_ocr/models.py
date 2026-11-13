from dataclasses import dataclass, field
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
