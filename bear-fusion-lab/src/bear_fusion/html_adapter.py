"""Preserve HTML code and table structure on the measured lossy upstream path."""
from bear_text_beta.models import DocumentContext, ParsedDocument
from bear_text_beta.parsing import HTMLParser, decode_text
from bs4 import BeautifulSoup, Comment


def parse_structured_html(data: bytes, context: DocumentContext) -> ParsedDocument | None:
    """Return native blocks for tables/pre; None delegates ordinary HTML to Dedoc."""
    text, warnings = decode_text(data, False)
    soup = BeautifulSoup(text, "lxml")
    if soup.find(["table", "pre"]) is None:
        return None
    # The base walker treats Comment as NavigableString; comments are not visible content.
    comments = soup.find_all(string=lambda value: isinstance(value, Comment))
    for comment in comments:
        comment.extract()
    document = HTMLParser().parse(str(soup).encode("utf-8"), context)
    document.parser_profile = "fusion-html-structure-v1"
    document.warnings.extend(warnings)
    document.warnings.extend(["source_fidelity_unverified", "html_native_route:preserve_code_and_tables"])
    if comments:
        document.warnings.append(f"excluded_html_comments:{len(comments)}")
    for block in document.blocks:
        block.metadata["source_precision"] = "normalized_dom_path"
    return document
