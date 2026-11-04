from copy import deepcopy
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
import os

import pytest
from bear_text_beta.chunking import ChunkingService
from bear_text_beta.embeddings import HFTokenCounter
from bear_text_beta.models import DocumentContext, EmbeddingProfile
from bear_text_beta.parsing import MDParser

from bear_fusion.cli import UnavailableEmbeddings
from bear_fusion.markdown_adapter import StructuredMarkdownParser
from bear_fusion.parents import attach_parents, expand_context
from bear_fusion.tables import TableProcessor, validate_table_chunks


@pytest.fixture(scope="module")
def counter():
    path = Path(os.environ.get("BEAR_TEST_TOKENIZER", Path(__file__).resolve().parents[2] / "bear-text-beta/models/e5"))
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip("Set BEAR_TEST_TOKENIZER for real-tokenizer integration tests")
    return HFTokenCounter(path, EmbeddingProfile())


def process(text, counter, document_id="d", version=1):
    document = MDParser().parse(text.encode(), DocumentContext("t", "s", document_id, version, "x.md"))
    result = ChunkingService(counter, UnavailableEmbeddings(), EmbeddingProfile(), {"TABLE": TableProcessor(counter)}).process(document)
    validate_table_chunks(document, result.chunks, counter, 320)
    parents = attach_parents(result)
    return result, {"result": asdict(result), "parents": parents}


def test_small_row_preserves_two_years_in_one_hit(counter):
    result, _ = process('| Metric | 2024 | 2025 |\n|---|---|---|\n| Revenue | 120 | 150 |', counter)
    assert any("120" in c.text and "150" in c.text for c in result.chunks)
    assert any(c.metadata.get("cell_segments") for c in result.chunks)


def test_packed_row_proof_rejects_removed_cell(counter):
    result, _ = process('| Metric | 2024 | 2025 |\n|---|---|---|\n| Revenue | 120 | 150 |', counter)
    chunks = deepcopy(result.chunks)
    packed = next(c for c in chunks if c.metadata.get("cell_segments"))
    packed.metadata["cell_segments"].pop()
    with pytest.raises(ValueError):
        validate_table_chunks(result.parsed_document, chunks, counter, 320)


def test_parent_ids_stable_and_scoped(counter):
    text = '# A\n\nFirst.\n\n## B\n\nSecond.'
    _, a = process(text, counter)
    _, b = process(text, counter)
    _, other = process(text, counter, document_id="other")
    assert a["parents"] == b["parents"]
    assert {p["id"] for p in a["parents"]}.isdisjoint({p["id"] for p in other["parents"]})
    sections = [p for p in a["parents"] if p["title"] in {"A", "B"}]
    assert next(p for p in sections if p["title"] == "B")["parent_id"] == next(p for p in sections if p["title"] == "A")["id"]


def test_repeated_titles_get_distinct_parent_ids(counter):
    _, envelope = process('# Repeat\n\nOne.\n\n# Repeat\n\nTwo.', counter)
    headings = [p for p in envelope["parents"] if p["title"] == "Repeat"]
    assert len(headings) == 2 and headings[0]["id"] != headings[1]["id"]


def test_parent_expansion_stays_in_section_and_budget(counter):
    _, envelope = process('# A\n\n' + 'Alpha sentence. ' * 350 + '\n\n# B\n\nSecret unrelated section.', counter)
    chunk = next(c for c in envelope["result"]["chunks"] if "Alpha" in c["text"])
    expanded = expand_context(envelope, chunk["id"], counter, 480)
    assert expanded["token_count"] <= 480
    assert chunk["text"] in expanded["text"]
    assert "Secret" not in expanded["text"]
    assert expanded["truncated"]
    with pytest.raises(ValueError, match="not found"):
        expand_context(envelope, "another-generation-chunk", counter)


def test_large_code_keeps_language_and_continuation_metadata(counter):
    _, envelope = process('# Code\n\n```python\n' + '    print("hello")\n' * 300 + '```', counter)
    code = [c for c in envelope["result"]["chunks"] if c["metadata"].get("code_languages")]
    assert len(code) > 1
    assert all(c["metadata"]["code_languages"] == ["python"] for c in code)
    assert all(c["metadata"]["code_parts"] for c in code)
    block = next(b for b in envelope['result']['parsed_document']['blocks'] if b['role'] == 'code')
    intervals = sorted((s['start_char'], s['end_char']) for c in code for s in c['source_spans'] if s['block_id'] == block['id'])
    assert intervals[0][0] == 0 and intervals[-1][1] == len(block['text'])
    assert all(a[1] == b[0] for a, b in pairwise(intervals))
    for chunk in code:
        for span in chunk['source_spans']:
            if span['block_id'] == block['id']:
                assert block['text'][span['start_char']:span['end_char']] in chunk['text']


def test_fenced_html_is_code_not_visible_markup():
    code = '<div>Visible</div>\n<!-- retained inside code -->\n<script>run()</script>'
    document = StructuredMarkdownParser().parse(
        ('```html\n' + code + '\n```\n\n<!-- excluded outside code -->').encode(),
        DocumentContext('t', 's', 'd', 1, 'x.md'))
    blocks = [block for block in document.blocks if block.role == 'code']
    assert len(blocks) == 1 and blocks[0].text == code + '\n'
    assert blocks[0].metadata['language'] == 'html'
    assert all('excluded outside' not in block.text for block in document.blocks)
