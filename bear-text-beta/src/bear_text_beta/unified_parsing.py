"""Native API adapters sharing the Fusion PDF/canonical-table implementation."""
import io
import posixpath
import re
import tempfile
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .errors import ParseError
from .file_validation import FileLimits
from .models import SourceLocator
from .parsing import BlockBuilder
from .visual_context import attach_visual_context, recognize_image

IMAGE_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
DOC_PROPERTIES = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr"


class PDFParser:
    def __init__(self, limits):
        self.limits = limits

    def parse(self, data, context):
        from bear_fusion.pdf_adapter import parse_pdf
        if not data.startswith(b"%PDF-"):
            raise ParseError("Некорректная сигнатура PDF")
        try:
            with tempfile.TemporaryDirectory(prefix="bear-api-pdf-") as folder:
                path = Path(folder) / "input.pdf"
                path.write_bytes(data)
                return parse_pdf(path, context, self.limits)
        except (ValueError, RuntimeError, OSError) as exc:
            raise ParseError(f"Ошибка PDF: {exc}") from exc


class DOCXParser:
    """OOXML text/body tables; unsupported visual objects remain explicit."""
    def parse(self, data, context):
        from bear_fusion.tables import restore_docx_tables
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        builder = BlockBuilder(context)
        numbered_styles = set()
        observed_media = set()
        relations_by_part = {}
        ocr_budget = 8

        def images(node, source, part):
            nonlocal ocr_budget
            from PIL import Image, UnidentifiedImageError

            drawings = [drawing for drawing in node.iter() if drawing.tag == f"{{{ns['w']}}}drawing"]
            if not drawings:
                return
            rel_name = posixpath.join("word/_rels", posixpath.basename(part) + ".rels")
            if part not in relations_by_part:
                relations_by_part[part] = ({item.get("Id"): item for item in ET.fromstring(archive.read(rel_name))}
                                           if rel_name in names else {})
            relations = relations_by_part[part]
            index = 0
            for drawing in drawings:
                props = next(drawing.iter(DOC_PROPERTIES), None)
                alt = props.get("descr", "")[:1024] if props is not None else ""
                for blip in drawing.iter(DRAWING + "blip"):
                    relation_id = blip.get(IMAGE_REL + "embed")
                    relation = relations.get(relation_id)
                    target = unquote(relation.get("Target", "")) if relation is not None else ""
                    target = posixpath.normpath(target.lstrip("/") if target.startswith("/") else "word/" + target)
                    if (relation is None or relation.get("TargetMode") == "External"
                            or not relation.get("Type", "").endswith("/image")
                            or not target.startswith("word/media/") or target not in names):
                        builder.warnings.append(f"docx_unresolved_image_reference:{source.element_path}")
                        continue
                    if len(builder.attachments) >= limits.max_attachments:
                        builder.warnings.append("docx_unresolved_image_attachment_limit")
                        return
                    location = replace(source, element_path=f"{source.element_path}/image/{index}")
                    index += 1
                    builder.image(target, location)
                    block = builder.blocks[-1]
                    block.metadata.update(embedded_visual=True, unresolved=True, alt_text=alt,
                                          source_part=part, relationship_id=relation_id)
                    attachment = builder.attachments[-1]
                    attachment.metadata.update(container_part=part, relationship_id=relation_id)
                    observed_media.add(target)
                    if target not in image_readings:
                        observation = {"status": "budget_or_byte_limit", "text": ""}
                        if ocr_budget and archive.getinfo(target).file_size <= limits.max_attachment_bytes:
                            ocr_budget -= 1
                            try:
                                with Image.open(io.BytesIO(archive.read(target))) as picture:
                                    attachment.mime_type = Image.MIME.get(picture.format, attachment.mime_type)
                                    observation = recognize_image(picture)
                            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                                observation = {"status": "unreadable", "text": "", "reason": type(exc).__name__}
                        image_readings[target] = observation, attachment.mime_type
                    block.metadata["image_ocr"], attachment.mime_type = image_readings[target]

        def paragraph(node, source, role="paragraph", part="word/document.xml"):
            text = "".join((part.text or "") if part.tag.endswith("}t") else
                           "\t" if part.tag.endswith("}tab") else "\n"
                           for part in node.iter() if part.tag.split("}")[-1] in {"t", "tab", "br"})
            style = node.find("w:pPr/w:pStyle", ns)
            match = re.search(r"(?:heading|заголовок)\s*(\d)",
                              style.get(f"{{{ns['w']}}}val", "") if style is not None else "", re.IGNORECASE)
            number = node.find("w:pPr/w:numPr", ns)
            is_list = number is not None or (style is not None and style.get(f"{{{ns['w']}}}val") in numbered_styles)
            if is_list:
                role = "list_item"
            if style is not None and style.get(f"{{{ns['w']}}}val", "").casefold() == "caption":
                role = "caption"
            builder.add(text, "heading" if match else role, source,
                        level=int(match[1]) if match else None,
                        metadata={"numbering_present": is_list})
            images(node, source, part)

        try:
            with ZipFile(io.BytesIO(data)) as archive:
                limits = FileLimits()
                names = set(archive.namelist())
                image_readings = {}
                if "word/styles.xml" in archive.namelist():
                    styles = ET.fromstring(archive.read("word/styles.xml"))
                    for style in styles.findall("w:style", ns):
                        key = style.get(f"{{{ns['w']}}}styleId", "")
                        if style.find("w:pPr/w:numPr", ns) is not None or re.search(r"list|bullet|number", key, re.IGNORECASE):
                            numbered_styles.add(key)
                    for _ in range(len(styles)):
                        before = len(numbered_styles)
                        for style in styles.findall("w:style", ns):
                            parent = style.find("w:basedOn", ns)
                            if parent is not None and parent.get(f"{{{ns['w']}}}val") in numbered_styles:
                                numbered_styles.add(style.get(f"{{{ns['w']}}}styleId"))
                        if before == len(numbered_styles):
                            break
                root = ET.fromstring(archive.read("word/document.xml"))
                def visit(container, address="body", depth=0):
                    if depth > 64:
                        raise ValueError("DOCX container depth exceeded")
                    for index, node in enumerate(container):
                        tag = node.tag.split("}")[-1]
                        location = f"{address}/{index}"
                        source = SourceLocator("docx", paragraph_index=index if depth == 0 else None, element_path=location)
                        if tag == "p":
                            paragraph(node, source)
                        elif tag == "tbl":
                            builder.add("", "table", source, kind="TABLE")
                            images(node, source, "word/document.xml")
                        elif tag == "sdt":
                            content = node.find("w:sdtContent", ns)
                            if content is not None:
                                visit(content, location + "/sdtContent", depth + 1)
                        elif tag == "customXml":
                            visit(node, location, depth + 1)
                        elif tag not in {"sectPr", "bookmarkStart", "bookmarkEnd", "proofErr", "permStart", "permEnd"}:
                            builder.warnings.append(f"docx_unsupported_body_element:{tag}:{location}")
                visit(root.find("w:body", ns))
                builder.sections = []
                for name in sorted(archive.namelist()):
                    if name in {"word/footnotes.xml", "word/endnotes.xml", "word/comments.xml", "word/vbaProject.bin"}:
                        builder.warnings.append(f"docx_unsupported_part:{name}")
                    if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name):
                        part = ET.fromstring(archive.read(name))
                        if part.find(".//w:tbl", ns) is not None:
                            builder.warnings.append(f"docx_unsupported_header_footer_table:{name}")
                        for index, node in enumerate(part.findall(".//w:p", ns)):
                            paragraph(node, SourceLocator("docx", element_path=f"{name}/{index}"),
                                      "header" if "/header" in name else "footer", name)
                for name in sorted(names):
                    if name.startswith(("word/media/", "word/embeddings/")) and name not in observed_media:
                        builder.warnings.append(f"docx_unsupported_embedded_object:{name}")
                if any(node.tag.split("}")[-1] in {"del", "ins", "altChunk", "oMath", "vanish", "drawing", "object"}
                       for node in root.iter()):
                    builder.warnings.append("docx_complex_markup_needs_review")
                document = builder.finish()
            with tempfile.TemporaryDirectory(prefix="bear-docx-") as folder:
                path = Path(folder) / "input.docx"
                path.write_bytes(data)
                restore_docx_tables(path, document)
            document.parser_profile = "api-docx-ooxml-v3-visual"
            return attach_visual_context(document)
        except (ValueError, KeyError, ET.ParseError) as exc:
            raise ParseError(f"Неоднозначная структура DOCX: {exc}") from exc

    def close(self):
        pass
