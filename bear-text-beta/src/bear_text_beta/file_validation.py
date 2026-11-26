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

from lxml import etree

from .errors import ParseError

MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".xml": "application/xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".doc": "application/msword",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".rtf": "application/rtf",
}
XLS_SIGNATURE = bytes.fromhex("D0 CF 11 E0 A1 B1 1A E1")
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
CONTENT_TYPE_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
XLSX_MAIN_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"


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
    max_table_cells: int = 100_000
    max_image_pixels: int = 30_000_000

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

    ALLOWED = set(MIME_TYPES)
    TEXT_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".csv", ".tsv"}
    SIGNATURES = {
        ".docx": b"PK\x03\x04", ".xlsx": b"PK\x03\x04", ".xls": XLS_SIGNATURE,
        ".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff",
        ".doc": XLS_SIGNATURE, ".odt": b"PK\x03\x04", ".rtf": b"{\\rtf",
    }

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
            signature = self.SIGNATURES.get(suffix)
            if signature and not reader.prefix.startswith(signature):
                if suffix == ".docx":
                    raise ParseError("DOCX не является ZIP-архивом")
                raise ParseError(f"Содержимое не соответствует сигнатуре {suffix[1:].upper()}")
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

    def validate_xlsx(self, data: bytes) -> None:
        if len(data) > self.limits.max_file_bytes:
            raise ParseError("XLSX превышает лимит размера файла")
        if not data.startswith(self.SIGNATURES[".xlsx"]):
            raise ParseError("XLSX не является ZIP-архивом")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > self.limits.max_docx_entries:
                    raise ParseError("XLSX превышает лимит ZIP-элементов")
                names = {entry.filename for entry in entries}
                if len(names) != len(entries) or not {"[Content_Types].xml", "xl/workbook.xml"} <= names:
                    raise ParseError("XLSX не содержит однозначного каталога книги")
                total = 0
                expanded_budget = XlsxArchiveBudget(self.limits.max_table_cells)
                for entry in entries:
                    name = entry.filename
                    if (entry.flag_bits & 1 or name.startswith("/") or "\\" in name
                            or ".." in name.split("/") or ":" in name):
                        raise ParseError("Неподдерживаемая запись в XLSX")
                    if entry.is_dir():
                        continue
                    if total + entry.file_size > self.limits.max_docx_uncompressed_bytes:
                        raise ParseError("XLSX превышает лимит распакованного содержимого")
                    is_xml = name.endswith((".xml", ".rels"))
                    expanded_budget.is_manifest = name == "[Content_Types].xml"
                    parser = xml_parser(target=expanded_budget) if is_xml else None
                    with archive.open(entry) as source:
                        while chunk := source.read(1024 * 1024):
                            total += len(chunk)
                            if total > self.limits.max_docx_uncompressed_bytes:
                                raise ParseError("XLSX превышает лимит распаковки")
                            if parser is not None:
                                parser.feed(chunk)
                    if parser is not None:
                        parser.close()
                if not expanded_budget.is_xlsx:
                    raise ParseError("Файл Excel не объявлен обычной книгой XLSX")
        except (zipfile.BadZipFile, OSError, etree.XMLSyntaxError, RuntimeError, ValueError) as exc:
            raise ParseError("Повреждённый или неподдерживаемый XLSX") from exc

    def validate_odt(self, data):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > self.limits.max_docx_entries:
                    raise ParseError("ODT превышает лимит ZIP-элементов")
                names = {entry.filename for entry in entries}
                if len(names) != len(entries) or not {"mimetype", "content.xml", "META-INF/manifest.xml"} <= names:
                    raise ParseError("ODT не содержит однозначного каталога документа")
                mimetype = archive.getinfo("mimetype")
                expected = MIME_TYPES[".odt"].encode("ascii")
                if mimetype.file_size != len(expected) or archive.read(mimetype) != expected:
                    raise ParseError("Контейнер не является текстовым ODT")
                total, attachment_bytes, attachments = 0, 0, 0
                for entry in entries:
                    name = entry.filename
                    if (entry.flag_bits & 1 or name.startswith("/") or "\\" in name
                            or ".." in name.split("/") or ":" in name):
                        raise ParseError("Неподдерживаемая запись в ODT")
                    if entry.is_dir():
                        continue
                    if total + entry.file_size > self.limits.max_docx_uncompressed_bytes:
                        raise ParseError("ODT превышает лимит распакованного содержимого")
                    attached = name.startswith(("Pictures/", "Object ", "ObjectReplacements/"))
                    if attached:
                        attachments += 1
                        if attachments > self.limits.max_attachments:
                            raise ParseError("ODT превышает лимит числа вложений")
                    parser = xml_parser(target=OdtXmlValidation()) if name.endswith(".xml") else None
                    with archive.open(entry) as source:
                        entry_bytes = 0
                        while chunk := source.read(1024 * 1024):
                            total += len(chunk)
                            if total > self.limits.max_docx_uncompressed_bytes:
                                raise ParseError("ODT превышает лимит распаковки")
                            if attached:
                                entry_bytes += len(chunk)
                                attachment_bytes += len(chunk)
                                if (entry_bytes > self.limits.max_attachment_bytes
                                        or attachment_bytes > self.limits.max_total_attachment_bytes):
                                    raise ParseError("ODT превышает лимит распаковки вложений")
                            if parser is not None:
                                parser.feed(chunk)
                    if parser is not None:
                        parser.close()
        except (zipfile.BadZipFile, OSError, etree.XMLSyntaxError, RuntimeError, ValueError) as exc:
            raise ParseError("Повреждённый или неподдерживаемый ODT") from exc

def xml_parser(**kwargs):
    return etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True, huge_tree=False, **kwargs)


class XlsxArchiveBudget:
    """Count real cells and merge expansion before OpenPyXL materializes worksheets."""

    def __init__(self, maximum):
        self.maximum = maximum
        self.cells = 0
        self.merged_cells = 0
        self.is_manifest = False
        self.is_xlsx = False

    def doctype(self, name, public_id, system_id):
        raise ParseError("DTD внутри XLSX запрещена")

    def start(self, tag, attrs):
        if tag == f"{{{SPREADSHEET_NS}}}c":
            self.cells += 1
        elif tag == f"{{{SPREADSHEET_NS}}}mergeCell":
            from openpyxl.utils.cell import range_boundaries

            left, top, right, bottom = range_boundaries(attrs["ref"])
            if not 0 < left <= right <= 16384 or not 0 < top <= bottom <= 1048576:
                raise ParseError("Некорректный диапазон объединения XLSX")
            self.merged_cells += (right - left + 1) * (bottom - top + 1) - 1
        elif (self.is_manifest and tag == f"{{{CONTENT_TYPE_NS}}}Override"
              and attrs.get("PartName") == "/xl/workbook.xml" and attrs.get("ContentType") == XLSX_MAIN_TYPE):
            self.is_xlsx = True
        if self.cells + self.merged_cells > self.maximum:
            raise ParseError("XLSX превышает лимит ячеек с учётом объединений")

    def close(self):
        return None


class OdtXmlValidation:
    def doctype(self, name, public_id, system_id):
        raise ParseError("DTD внутри ODT запрещена")

    def start(self, tag, attrs):
        if tag == "{urn:oasis:names:tc:opendocument:xmlns:manifest:1.0}encryption-data":
            raise ParseError("Зашифрованные записи ODT не поддерживаются")

    def close(self):
        return None
