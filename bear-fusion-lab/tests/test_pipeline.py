from dataclasses import replace

import pytest
from bear_text_beta.errors import ParseError
from bear_text_beta.file_validation import FileLimits
from bear_text_beta.models import DocumentContext
from bear_text_beta.object_storage import LocalObjectStorage

from bear_fusion.pipeline import FusionPipeline


class Parser:
    called = False

    def parse(self, path):
        self.called = True
        return {"tree": {"type": "paragraph", "text": path.read_text(encoding="utf-8"), "children": []}}


class Chunker:
    def process(self, document, profile):
        # Only pipeline wiring is tested here, actual chunking is exercised in experiments.
        return document


def run(tmp_path, name, content, limits=None):
    source = tmp_path / name
    source.write_text(content, encoding="utf-8")
    parser = Parser()
    service = FusionPipeline(parser, Chunker(), LocalObjectStorage(tmp_path / "objects"), limits)
    result = service.process_path(source, DocumentContext("t", "s", "d", 1, name))
    return result, parser


def test_txt_preserves_source_without_upstream_startup(tmp_path):
    result, parser = run(tmp_path, "a.txt", "hello")
    assert not parser.called and not result["accepted_for_indexing"]
    assert result["result"]["blocks"][0]["text"] == "hello"


def test_native_md_preserves_code_and_bypasses_upstream(tmp_path):
    result, parser = run(tmp_path, "a.md", "# Heading\n\n```python\n# code\n```\n")
    assert not parser.called
    assert any(b["role"] == "code" for b in result["result"]["blocks"])


def test_long_html_cell_preserved_without_upstream(tmp_path):
    result, parser = run(tmp_path, "a.html", "<table><tr><td>" + "x" * 600 + "END</td></tr></table>")
    assert not parser.called
    assert "x" * 600 + "END" in result["result"]["blocks"][0]["metadata"]["raw_source"]


def test_file_limit(tmp_path):
    with pytest.raises(ParseError):
        run(tmp_path, "a.txt", "x" * 100, replace(FileLimits(), max_file_bytes=10))


def test_txt_source_integrity_fallback(tmp_path):
    class LossyParser:
        def parse(self, path):
            return {"tree": {"type": "paragraph", "text": "Fields", "children": []}}

    source = tmp_path / "index.txt"
    source.write_text("Fields**\n\nРаздел 2", encoding="utf-8")
    service = FusionPipeline(LossyParser(), Chunker(), LocalObjectStorage(tmp_path / "objects"))
    result = service.process_path(source, DocumentContext("t", "s", "d", 1, source.name))
    assert result["parser_runtime"]["route"] == "native-txt"
    assert [b["text"] for b in result["result"]["blocks"]] == ["Fields**", "Раздел 2"]
