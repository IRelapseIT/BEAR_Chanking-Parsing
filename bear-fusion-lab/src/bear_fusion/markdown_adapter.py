"""Preserve code semantics and expose unresolved documentation directives."""
import re

from bear_text_beta.parsing import MDParser, decode_text
from bs4 import BeautifulSoup, Comment
from markdown_it import MarkdownIt


class StructuredMarkdownParser(MDParser):
    def parse(self, data, context):
        text, _ = decode_text(data, self.allow_cp1251)
        code_locations = {
            (token.map[0] + 1, token.map[1])
            for token in MarkdownIt("commonmark").enable("table").parse(text)
            if token.type in {"fence", "code_block"} and token.map
        }
        document = super().parse(data, context)
        kept = []
        for block in document.blocks:
            is_literal_code = (block.source.line_start, block.source.line_end) in code_locations
            if block.role == "code" and block.metadata.get("language") == "html" and not is_literal_code:
                soup = BeautifulSoup(block.text, "lxml")
                for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
                    comment.extract()
                visible = soup.get_text()
                if not visible.strip():
                    document.warnings.append("markdown_noncontent_markup_excluded")
                    continue
                block.role = "paragraph"
                block.text = visible
                block.metadata["original_markup"] = True
            kept.append(block)
        document.blocks = kept
        for order, block in enumerate(kept):
            block.order = order
        directives = re.findall(r"\{\{#(?:rustdoc_include|include)\s+([^}]+)\}\}", text)
        if directives:
            document.warnings.append(f"unresolved_include:{len(directives)}")
        document.parser_profile = "fusion-markdown-v3"
        return document
