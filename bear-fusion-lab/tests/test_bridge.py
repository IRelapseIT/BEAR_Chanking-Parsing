from dataclasses import replace

import pytest
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext

from bear_fusion.bridge import map_tree


def node(kind, text, children=()):
    return {"type": kind, "text": text, "children": list(children)}


def context():
    return DocumentContext("tenant", "space", "doc", 1, "test.txt")


def test_nested_sections_do_not_leak_and_root_text_survives():
    tree = node("document", "Root text", [
        node("header", "Outer", [node("header", "Inner", [node("paragraph", "Inner body")]),
                                  node("paragraph", "Outer body")]), node("paragraph", "Root tail")])
    document = map_tree({"tree": tree}, context())
    assert [b.text for b in document.blocks] == ["Root text", "Outer", "Inner", "Inner body", "Outer body", "Root tail"]
    assert document.blocks[-2].section_path == ["Outer"]
    assert document.blocks[-1].section_path == []
    assert all(b.source.page is None for b in document.blocks)


def test_repeatable_ids_and_explicit_tables():
    payload = {"tree": node("document", "", [node("table", "| A |\n|---|\n| 1 |")])}
    first = map_tree(payload, context())
    assert first == map_tree(payload, context())
    assert first.blocks[0].type == "TABLE"


@pytest.mark.parametrize("tree", [None, node("unknown", "x"), node("paragraph", 3),
                                 {"type": "document", "text": "", "children": [1]}])
def test_invalid_contract_rejected(tree):
    with pytest.raises((TypeError, ValueError)):
        map_tree({"tree": tree}, context())


def test_limits():
    with pytest.raises(ValueError):
        map_tree({"tree": node("paragraph", "abc")}, context(),
                 replace(FileLimits(), max_text_chars=2))
