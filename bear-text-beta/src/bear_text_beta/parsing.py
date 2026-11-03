import codecs
import hashlib
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup, NavigableString, Tag
from markdown_it import MarkdownIt

from .errors import ParseError, ValidationError
from .file_validation import FileLimits, FileValidator, ObjectStorage
from .models import Attachment, ContentBlock, DocumentContext, ParsedDocument, SourceLocator
from .ports import DocumentParser


def decode_text(data: bytes, allow_cp1251: bool = False) -> tuple[str, list[str]]:
    warnings = []
    try:
        if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            text = data.decode("utf-32")
        elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            text = data.decode("utf-16")
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                if not allow_cp1251:
                    raise
                text = data.decode("cp1251")
                warnings.append("legacy_encoding:cp1251")
    except UnicodeError as exc:
        raise ParseError("Неизвестная или повреждённая кодировка; ожидается UTF-8/BOM") from exc
    if "\x00" in text:
        raise ParseError("Бинарное содержимое в текстовом файле")
    return text.replace("\r\n", "\n").replace("\r", "\n"), warnings


class BlockBuilder:
    def __init__(self, context: DocumentContext):
        self.context = context
        self.blocks = []
        self.attachments = []
        self.sections = []
        self.warnings = []

    def add(self, text, role, source, *, kind="TEXT", level=None, metadata=None, attachments=None):
        text = text if role == "code" else text.strip()
        if kind == "TEXT" and not text:
            return
        if role == "heading":
            level = level or 1
            self.sections = self.sections[:level - 1] + [text]
        order = len(self.blocks)
        identity = f"{self.context.tenant_id}/{self.context.space_id}/{self.context.document_id}"
        block_id = str(uuid5(NAMESPACE_URL, f"{identity}/{self.context.document_version}/{order}"))
        self.blocks.append(ContentBlock(
            block_id, order, kind, role, text, list(self.sections), source,
            attachments or [], metadata or {},
        ))

    def image(self, target, source, caption="", data=None, mime_type="application/octet-stream",
              asset_dir: Path | None = None):
        image_id = str(uuid5(NAMESPACE_URL, f"{self.context.checksum}/{len(self.attachments)}/{target}"))
        key = target
        metadata = {"original_target": target, "caption": caption, "stored": False}
        if data is not None:
            if asset_dir is None:
                raise ParseError("Для встроенных изображений нужен asset_dir")
            digest = hashlib.sha256(data).hexdigest()
            destination = asset_dir / digest
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            key = str(destination.resolve())
            metadata.update(stored=True, checksum=digest)
        self.attachments.append(Attachment(image_id, "image", key, mime_type, metadata))
        self.add(caption, "image", source, kind="IMAGE", attachments=[image_id])

    def finish(self):
        return ParsedDocument(self.context, self.blocks, attachments=self.attachments, warnings=self.warnings)


class TXTParser:
    def __init__(self, allow_cp1251=False):
        self.allow_cp1251 = allow_cp1251

    def parse(self, data: bytes, context: DocumentContext) -> ParsedDocument:
        text, warnings = decode_text(data, self.allow_cp1251)
        builder = BlockBuilder(context)
        builder.warnings.extend(warnings)
        for match in re.finditer(r"[^\n]*(?:\n(?![ \t]*\n)[^\n]*)*", text):
            if match.group().strip():
                raw = match.group()
                normalized = raw.strip()
                content_start = match.start() + len(raw) - len(raw.lstrip())
                start = text.count("\n", 0, content_start) + 1
                end = start + normalized.count("\n")
                builder.add(normalized, "paragraph", SourceLocator("txt", start, end))
        return builder.finish()


class MDParser(TXTParser):
    def parse(self, data: bytes, context: DocumentContext) -> ParsedDocument:
        text, warnings = decode_text(data, self.allow_cp1251)
        lines = text.splitlines(keepends=True)
        tokens = MarkdownIt("commonmark").enable("table").parse(text)
        builder = BlockBuilder(context)
        builder.warnings.extend(warnings)
        roles = []
        heading_level = None
        index = 0
        while index < len(tokens):
            token = tokens[index]
            line_map = token.map or [0, 0]
            source = SourceLocator("md", line_map[0] + 1, line_map[1])
            if token.type == "table_open":
                raw = "".join(lines[line_map[0]:line_map[1]])
                builder.add("", "table", source, kind="TABLE", metadata={"raw_source": raw})
                while index < len(tokens) and tokens[index].type != "table_close":
                    index += 1
            elif token.type in {"fence", "code_block"}:
                builder.add(token.content, "code", source, metadata={"language": token.info})
            elif token.type == "heading_open":
                heading_level = int(token.tag[1:])
            elif token.type == "heading_close":
                heading_level = None
            elif token.type in {"list_item_open", "blockquote_open"}:
                roles.append("list_item" if token.type == "list_item_open" else "quote")
            elif token.type in {"list_item_close", "blockquote_close"}:
                roles.pop()
            elif token.type == "inline":
                parts, links, images = [], [], []
                for child in token.children or []:
                    if child.type in {"text", "code_inline", "html_inline"}:
                        parts.append(child.content)
                    elif child.type in {"softbreak", "hardbreak"}:
                        parts.append("\n")
                    elif child.type == "link_open":
                        links.append(child.attrGet("href"))
                    elif child.type == "image":
                        images.append(child)
                role = "heading" if heading_level else (roles[-1] if roles else "paragraph")
                builder.add("".join(parts), role, source, level=heading_level, metadata={"links": links})
                for image in images:
                    builder.image(image.attrGet("src") or "", source, image.content)
            elif token.type == "html_block":
                builder.add(token.content, "code", source, metadata={"language": "html"})
            index += 1
        return builder.finish()


class HTMLParser(TXTParser):
    def parse(self, data: bytes, context: DocumentContext) -> ParsedDocument:
        text, warnings = decode_text(data, self.allow_cp1251)
        soup = BeautifulSoup(text, "lxml")
        builder = BlockBuilder(context)
        builder.warnings.extend(warnings)
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()
        root = soup.body or soup

        def path(node):
            parts = []
            while isinstance(node, Tag) and node.name != "[document]":
                siblings = [n for n in node.parent.children if isinstance(n, Tag) and n.name == node.name]
                index = next(i for i, sibling in enumerate(siblings, 1) if sibling is node)
                parts.append(f"{node.name}[{index}]")
                node = node.parent
            return "/" + "/".join(reversed(parts))

        def visit(node):
            if not isinstance(node, Tag):
                return
            source = SourceLocator("html", element_path=path(node))
            if node.name == "table":
                builder.add("", "table", source, kind="TABLE", metadata={"raw_source": str(node)})
                return
            if node.name == "img":
                builder.image(node.get("src", ""), source, node.get("alt", ""))
                return
            if re.fullmatch(r"h[1-6]", node.name or ""):
                builder.add(node.get_text(), "heading", source, level=int(node.name[1]))
                return
            if node.name in {"pre", "p"}:
                # Remove images from textual extraction but retain their original locators.
                images = list(node.find_all("img"))
                builder.add(node.get_text(), "code" if node.name == "pre" else "paragraph", source,
                            metadata={"links": [a.get("href") for a in node.find_all("a")]})
                for image in images:
                    visit(image)
                return
            pending = []

            def flush():
                builder.add("".join(pending), "list_item" if node.name == "li" else "paragraph", source)
                pending.clear()

            boundaries = {"table", "img", "pre", "p", "ul", "ol", "li", "div", "section", "article",
                          "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer", "main"}
            for child in node.children:
                if isinstance(child, NavigableString):
                    pending.append(str(child))
                elif isinstance(child, Tag) and child.name == "br":
                    pending.append("\n")
                elif isinstance(child, Tag) and child.name in boundaries:
                    flush()
                    visit(child)
                elif isinstance(child, Tag):
                    if child.find(list(boundaries)):
                        flush()
                        visit(child)
                    else:
                        pending.append(child.get_text())
            flush()

        visit(root)
        return builder.finish()


class DocumentNormalizer:
    def normalize(self, document: ParsedDocument,
                  limits: FileLimits | None = None) -> ParsedDocument:
        limits = limits or FileLimits()
        if len(document.blocks) > limits.max_blocks:
            raise ParseError("Превышен лимит блоков документа")
        if sum(len(block.text) for block in document.blocks) > limits.max_text_chars:
            raise ParseError("Превышен лимит символов документа")
        ids = set()
        for order, block in enumerate(document.blocks):
            if block.id in ids:
                raise ValidationError("Duplicate block ID")
            ids.add(block.id)
            block.order = order
            block.text = block.text.replace("\r\n", "\n").replace("\r", "\n")
        return document


class ParsingPipeline:
    def __init__(self, asset_dir: Path, allow_cp1251: bool = False,
                 max_file_bytes: int = 50 * 1024 * 1024,
                 extra_parsers: dict[str, DocumentParser] | None = None,
                 dedoc_client: DocumentParser | None = None, limits: FileLimits | None = None,
                 object_storage: ObjectStorage | None = None):
        self.limits = limits or FileLimits(max_file_bytes=max_file_bytes)
        self.max_file_bytes = self.limits.max_file_bytes
        if dedoc_client is None:
            from .dedoc_adapter import DedocClient
            dedoc_client = DedocClient(asset_dir=asset_dir, limits=self.limits,
                                       object_storage=object_storage)
        self.parsers = {
            ".txt": dedoc_client, ".md": MDParser(allow_cp1251),
            ".html": dedoc_client, ".htm": dedoc_client, ".docx": dedoc_client,
        }
        self.allow_cp1251 = allow_cp1251
        self.parsers.update(extra_parsers or {})

    def parse(self, data: bytes, context: DocumentContext) -> ParsedDocument:
        if len(data) > self.max_file_bytes:
            raise ParseError("Файл превышает лимит 50 MiB")
        if context.document_version < 1 or not all((context.tenant_id, context.space_id, context.document_id)):
            raise ParseError("Некорректный DocumentContext")
        suffix = Path(context.filename).suffix.lower()
        if suffix not in self.parsers:
            raise ParseError(f"Нет обработчика формата {suffix}")
        if suffix == ".docx":
            FileValidator(self.limits).validate_docx(data)
        checksum = hashlib.sha256(data).hexdigest()
        if context.checksum and checksum != context.checksum:
            raise ParseError("Checksum не соответствует содержимому файла")
        context.checksum = checksum
        context.mime_type = {".txt": "text/plain", ".md": "text/markdown", ".html": "text/html",
                             ".htm": "text/html", ".docx":
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(
                                 suffix, context.mime_type)
        parse_data = data
        decode_warnings = []
        if suffix in {".txt", ".html", ".htm"} and self.parsers[suffix] is self.parsers[".docx"]:
            text, decode_warnings = decode_text(data, self.allow_cp1251)
            parse_data = text.encode("utf-8")
        document = self.parsers[suffix].parse(parse_data, context)
        document.warnings.extend(decode_warnings)
        return DocumentNormalizer().normalize(document, self.limits)
