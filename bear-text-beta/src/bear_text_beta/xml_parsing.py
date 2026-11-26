"""Ordered XML projection with source paths and no external resource resolution."""

import io
from collections import Counter
from uuid import NAMESPACE_URL, uuid5

from lxml import etree

from .errors import ParseError
from .models import ContentBlock, SourceLocator
from .parsing import BlockBuilder


def qualified_name(node):
    local = etree.QName(node).localname
    return f"{node.prefix}:{local}" if node.prefix else local


class XmlParser:
    def __init__(self, limits):
        self.limits = limits

    def parse(self, data, context):
        try:
            events = etree.iterparse(io.BytesIO(data), events=("start",), resolve_entities=False, load_dtd=False,
                                     no_network=True, huge_tree=False, remove_comments=True, remove_pis=True)
            for index, _ in enumerate(events):
                if index >= self.limits.max_blocks:
                    raise ParseError("XML превышает лимит числа элементов")
            root = events.root
        except etree.XMLSyntaxError as exc:
            raise ParseError(f"Некорректный XML около строки {exc.lineno}") from exc
        if root.getroottree().docinfo.doctype:
            raise ParseError("DTD в XML запрещена")

        builder = BlockBuilder(context)
        nodes, text_chars = [], 0

        def emit(value, node, path, ordinal):
            if not value or not value.strip():
                return
            if len(builder.blocks) >= self.limits.max_blocks:
                raise ParseError("XML превышает лимит текстовых блоков")
            builder.add(value, "xml_text", SourceLocator(
                "xml", node.sourceline if ordinal == 1 else None, element_path=f"{path}/text()[{ordinal}]"),
                metadata={"xml_element": node.tag, "xml_attributes": dict(node.attrib)})

        def walk(node, path, parent_path):
            nonlocal text_chars
            attributes = dict(node.attrib)
            children = list(node)
            text_chars += (len(path) + len(node.text or "") + sum(len(child.tail or "") for child in children)
                           + sum(len(key) + len(value) for key, value in attributes.items()))
            if text_chars > self.limits.max_text_chars:
                raise ParseError("XML превышает лимит материала элементов")
            nodes.append({"path": path, "parent_path": parent_path, "tag": node.tag,
                          "attributes": attributes, "line": node.sourceline})
            ordinal = 1
            emit(node.text, node, path, ordinal)
            siblings = Counter()
            for child in children:
                name = qualified_name(child)
                siblings[name] += 1
                walk(child, f"{path}/{name}[{siblings[name]}]", path)
                if child.tail:
                    ordinal += 1
                    emit(child.tail, node, path, ordinal)

        root_path = f"/{qualified_name(root)}[1]"
        walk(root, root_path, None)
        if not builder.blocks:
            # A textless XML tree still exists for inspection, with no synthetic searchable text.
            identity = f"{context.tenant_id}/{context.space_id}/{context.document_id}/{context.document_version}/0"
            builder.blocks.append(ContentBlock(str(uuid5(NAMESPACE_URL, identity)), 0, "TEXT", "xml_tree", "", [],
                                               SourceLocator("xml", root.sourceline, element_path=root_path)))
        builder.blocks[0].metadata["xml_nodes"] = nodes
        document = builder.finish()
        document.parser_profile = "xml-lxml-v1"
        return document
