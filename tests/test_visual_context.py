from bear_pdf_ocr.models import (
    ContentBlock,
    DocumentContext,
    ParsedDocument,
    SourceLocator,
)
from bear_pdf_ocr.visual_context import attach_visual_context


def image(name, page=1, x=0):
    return ContentBlock(name, 0, "IMAGE", "image", "", [],
                        SourceLocator("pdf", page=page, bbox=[x, 10, x + 70, 45]),
                        metadata={"embedded_visual": True, "unresolved": True})


def caption(name, page=1, x=0):
    return ContentBlock(name, 0, "TEXT", "paragraph", "Figure 1. Source caption.", [],
                        SourceLocator("pdf", page=page, bbox=[x, 48, x + 70, 62]))


def attach(*blocks):
    attach_visual_context(ParsedDocument(DocumentContext("t", "s", "d", 1, "x.pdf"), list(blocks)))


def test_visual_context_links_local_caption():
    visual, text = image("a"), caption("c")
    attach(visual, text)
    assert visual.metadata["visual_context"]["caption"] == {"block_id": "c", "text": text.text}
    assert visual.metadata["unresolved"]
    assert visual.text == ""


def test_visual_context_rejects_caption_across_page_or_column():
    for text in (caption("p", page=2), caption("c", x=150)):
        visual = image("a")
        attach(visual, text)
        assert visual.metadata["visual_context"]["caption_status"] == "not_found"


def test_shared_caption_is_ambiguous():
    a, b, text = image("a"), image("b"), caption("c")
    attach(a, text, b)
    assert all(block.metadata["visual_context"]["caption_status"] == "ambiguous" for block in (a, b))
