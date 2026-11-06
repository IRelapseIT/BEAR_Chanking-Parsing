"""Native API adapters sharing the Fusion PDF/canonical-table implementation."""
import io
import re
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from .errors import ParseError
from .models import SourceLocator
from .parsing import BlockBuilder


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

        def paragraph(node, source, role="paragraph"):
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
            builder.add(text, "heading" if match else role, source,
                        level=int(match[1]) if match else None,
                        metadata={"numbering_present": is_list})

        try:
            with ZipFile(io.BytesIO(data)) as archive:
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
                for index, node in enumerate(root.find("w:body", ns)):
                    source = SourceLocator("docx", paragraph_index=index, element_path=f"body/{index}")
                    if node.tag.endswith("}p"):
                        paragraph(node, source)
                    elif node.tag.endswith("}tbl"):
                        builder.add("", "table", source, kind="TABLE")
                for name in sorted(archive.namelist()):
                    if name in {"word/footnotes.xml", "word/endnotes.xml", "word/comments.xml", "word/vbaProject.bin"}:
                        builder.warnings.append(f"docx_unsupported_part:{name}")
                    if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name):
                        part = ET.fromstring(archive.read(name))
                        if part.find(".//w:tbl", ns) is not None:
                            builder.warnings.append(f"docx_unsupported_header_footer_table:{name}")
                        for index, node in enumerate(part.findall(".//w:p", ns)):
                            paragraph(node, SourceLocator("docx", element_path=f"{name}/{index}"),
                                      "header" if "/header" in name else "footer")
                    if name.startswith(("word/media/", "word/embeddings/")):
                        builder.warnings.append(f"docx_unsupported_embedded_object:{name}")
                if any(node.tag.split("}")[-1] in {"del", "ins", "altChunk", "oMath", "vanish", "drawing", "object"}
                       for node in root.iter()):
                    builder.warnings.append("docx_complex_markup_needs_review")
                document = builder.finish()
            with tempfile.TemporaryDirectory(prefix="bear-docx-") as folder:
                path = Path(folder) / "input.docx"
                path.write_bytes(data)
                restore_docx_tables(path, document)
            document.parser_profile = "api-docx-ooxml-v1"
            return document
        except (ValueError, KeyError, ET.ParseError) as exc:
            raise ParseError(f"Неоднозначная структура DOCX: {exc}") from exc

    def close(self):
        pass
