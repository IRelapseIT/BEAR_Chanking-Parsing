"""Bounded upload and format preflight shared by API, CLI and parser workers."""

from __future__ import annotations

import codecs
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from xml.etree import ElementTree

from .errors import ParseError


@dataclass(frozen=True)
class FileLimits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_docx_entries: int = 10_000
    max_docx_uncompressed_bytes: int = 200 * 1024 * 1024
    max_attachments: int = 100
    max_attachment_bytes: int = 10 * 1024 * 1024
    max_total_attachment_bytes: int = 50 * 1024 * 1024
    max_pages: int = 500
    max_blocks: int = 100_000
    max_text_chars: int = 10_000_000
    max_table_inline_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("File limits must be positive")


@dataclass(frozen=True)
class StoredObject:
    object_key: str
    size_bytes: int
    checksum: str


class ObjectStorage(Protocol):
    def put_stream(self, object_key: str, source: BinaryIO) -> None: ...
    def get_stream(self, object_key: str) -> BinaryIO: ...
    def delete(self, object_key: str) -> None: ...


class BoundedHashingReader:
    def __init__(self, source: BinaryIO, maximum: int):
        self.source = source
        self.maximum = maximum
        self.size = 0
        self.digest = hashlib.sha256()
        self.prefix = bytearray()

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise ValueError("BoundedHashingReader requires bounded reads")
        data = self.source.read(min(size, self.maximum - self.size + 1))
        self.size += len(data)
        if self.size > self.maximum:
            raise ParseError("Файл превышает допустимый размер")
        self.digest.update(data)
        if len(self.prefix) < 16:
            self.prefix.extend(data[:16 - len(self.prefix)])
        return data


class FileValidator:
    """Validate while ObjectStorage performs its already necessary upload read."""

    ALLOWED = {".txt", ".md", ".html", ".htm", ".docx"}
    TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

    def __init__(self, limits: FileLimits | None = None, allowed_extensions: set[str] | None = None):
        self.limits = limits or FileLimits()
        self.allowed_extensions = allowed_extensions if allowed_extensions is not None else self.ALLOWED

    def validate_and_store(
        self, source: BinaryIO, filename: str, object_key: str, storage: ObjectStorage,
        expected_checksum: str = "",
    ) -> StoredObject:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_extensions:
            raise ParseError(f"Нет обработчика формата {suffix}")
        reader = BoundedHashingReader(source, self.limits.max_file_bytes)
        try:
            storage.put_stream(object_key, reader)
            checksum = reader.digest.hexdigest()
            if expected_checksum and expected_checksum != checksum:
                raise ParseError("Checksum не соответствует содержимому файла")
            if suffix == ".docx" and bytes(reader.prefix[:4]) != b"PK\x03\x04":
                raise ParseError("DOCX не является ZIP-архивом")
            unicode_bom = bytes(reader.prefix).startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE,
                                                         codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
            if suffix in self.TEXT_EXTENSIONS and b"\x00" in reader.prefix and not unicode_bom:
                raise ParseError("Бинарное содержимое в текстовом файле")
            return StoredObject(object_key, reader.size, checksum)
        except Exception:
            storage.delete(object_key)
            raise

    def validate_docx(self, data: bytes) -> None:
        if len(data) > self.limits.max_file_bytes:
            raise ParseError("DOCX превышает лимит размера файла")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > self.limits.max_docx_entries:
                    raise ParseError("DOCX содержит слишком много ZIP-элементов")
                if not archive.getinfo("[Content_Types].xml") or not archive.getinfo("word/document.xml"):
                    raise ParseError("DOCX не содержит обязательные Office-элементы")
                total = 0
                attachments = 0
                attachment_total = 0
                for entry in entries:
                    if entry.flag_bits & 0x1:
                        raise ParseError("Зашифрованный DOCX не поддерживается")
                    if entry.filename.startswith("/") or ".." in Path(entry.filename).parts:
                        raise ParseError("Небезопасный путь внутри DOCX")
                    total += entry.file_size
                    if total > self.limits.max_docx_uncompressed_bytes:
                        raise ParseError("DOCX превышает лимит распакованного содержимого")
                    if entry.filename.startswith(("word/media/", "word/embeddings/")) and not entry.is_dir():
                        attachments += 1
                        attachment_total += entry.file_size
                        if entry.file_size > self.limits.max_attachment_bytes:
                            raise ParseError("Вложение DOCX превышает индивидуальный лимит")
                if attachments > self.limits.max_attachments:
                    raise ParseError("DOCX содержит слишком много вложений")
                if attachment_total > self.limits.max_total_attachment_bytes:
                    raise ParseError("Вложения DOCX превышают суммарный лимит")
                # ZIP directory sizes are only a preflight hint; verify actual decompressed bytes.
                actual_total = 0
                for entry in entries:
                    if entry.is_dir():
                        continue
                    actual_entry = 0
                    with archive.open(entry) as source:
                        while chunk := source.read(1024 * 1024):
                            actual_entry += len(chunk)
                            actual_total += len(chunk)
                            if actual_total > self.limits.max_docx_uncompressed_bytes:
                                raise ParseError("DOCX превышает лимит распаковки")
                            if entry.filename.startswith(("word/media/", "word/embeddings/")) and (
                                    actual_entry > self.limits.max_attachment_bytes):
                                raise ParseError("Вложение DOCX превышает лимит распаковки")
                if "docProps/app.xml" in archive.namelist():
                    props = archive.getinfo("docProps/app.xml")
                    if props.file_size <= 1024 * 1024:
                        root = ElementTree.fromstring(archive.read(props))
                        pages = root.find("{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages")
                        if pages is not None and pages.text and pages.text.isdecimal() and (
                                int(pages.text) > self.limits.max_pages):
                            raise ParseError("DOCX сообщает число страниц выше лимита")
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, RuntimeError) as exc:
            raise ParseError("Повреждённый или неподдерживаемый DOCX") from exc
