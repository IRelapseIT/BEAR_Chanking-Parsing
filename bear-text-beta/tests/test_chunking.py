import copy
from dataclasses import asdict

import pytest

from bear_text_beta.chunking import (
    ChunkingService,
    ChunkValidator,
    PreservingSemanticChunker,
    SafeTextSplitter,
)
from bear_text_beta.comparison import boundaries, compare_results
from bear_text_beta.errors import ValidationError
from bear_text_beta.models import PreparedFragment, ProcessingProfile, SourceSpan
from bear_text_beta.serialization import load_result

PROSE = " ".join(f"Предложение номер {i} описывает работу сервера и правила обслуживания." for i in range(18))


@pytest.mark.parametrize("mode", ["adaptive", "fixed", "token", "semantic"])
def test_modes_preserve_content(parsed, chunker, counter, mode):
    document = parsed(PROSE)
    result = chunker.process(document, ProcessingProfile(mode=mode))
    assert "".join(chunk.text for chunk in result.chunks) == PROSE
    assert all(chunk.token_count == counter.full_count(chunk.embedding_text) <= 480 for chunk in result.chunks)
    assert load_result(asdict(result)) == result


def test_semantic_calls_external_provider(parsed, chunker, embeddings):
    result = chunker.process(parsed(PROSE))
    assert embeddings.calls
    assert all(purpose == "similarity" for _, purpose in embeddings.calls)
    assert all(chunk.strategy == "semantic" for chunk in result.chunks)


def test_structural_uses_no_inference(parsed, chunker, embeddings):
    text = "# Раздел\n\n" + "\n\n".join([PROSE[:100], PROSE[100:200], PROSE[200:]])
    result = chunker.process(parsed(text, "document.md"))
    assert not embeddings.calls
    assert all(chunk.strategy == "structural" for chunk in result.chunks)
    assert all(chunk.section_path == ["Раздел"] for chunk in result.chunks)


def test_single_long_paragraph_under_heading_uses_semantic(parsed, chunker, embeddings):
    result = chunker.process(parsed("# Раздел\n\n" + PROSE, "document.md"))
    assert embeddings.calls
    assert all(chunk.strategy == "semantic" for chunk in result.chunks)


def test_semantic_failure_uses_recursive(parsed, chunker, embeddings):
    embeddings.fail = True
    result = chunker.process(parsed(PROSE))
    assert all(chunk.strategy == "recursive" for chunk in result.chunks)
    assert any("semantic_failure" in warning for warning in result.warnings)
    assert "".join(chunk.text for chunk in result.chunks) == PROSE


def test_long_sentence_and_budget(parsed, chunker):
    for text in ("А" * 6000, PROSE * 4):
        result = chunker.process(parsed(text))
        assert "".join(chunk.text for chunk in result.chunks) == text
        assert all(chunk.token_count <= 480 for chunk in result.chunks)


def test_huge_heading_context_is_bounded(parsed, chunker):
    document = parsed("# " + "Заголовок " * 200 + "\n\nТекст.", "document.md")
    result = chunker.process(document)
    assert all(chunk.token_count <= 480 for chunk in result.chunks)
    assert result.chunks[0].section_path == document.blocks[0].section_path


def test_sources_for_repeated_paragraphs(parsed, chunker, counter):
    document = parsed(("Одинаковый абзац.\n\n" * 10).rstrip())
    result = chunker.process(document)
    referenced = {span.block_id for chunk in result.chunks for span in chunk.source_spans}
    assert referenced == {block.id for block in document.blocks}
    ChunkValidator(counter, 480).validate(result)


def test_block_boundaries_stop_at_table(parsed, chunker):
    document = parsed("Перед.\n\n| A |\n| --- |\n| 1 |\n\nПосле.", "document.md")
    result = chunker.process(document)
    assert result.completeness == "partial"
    assert len(result.deferred_blocks) == 1
    assert [chunk.text for chunk in result.chunks] == ["Перед.", "После."]


def test_external_processor_contract(parsed, counter, embeddings, profile):
    class Tables:
        def process(self, block, context):
            return [PreparedFragment("A: 1", [SourceSpan(block.id, 0, 0)])]
        def validate(self, document, chunks, counter, maximum):
            if [chunk.text for chunk in chunks] != ["A: 1"]:
                raise ValidationError("Table value/header contract violated")
    document = parsed("| A |\n| --- |\n| 1 |\n", "document.md")
    result = ChunkingService(counter, embeddings, profile, {"TABLE": Tables()}).process(document)
    assert result.completeness == "complete"
    assert result.chunks[0].strategy == "external"
    assert result.chunks[0].text == "A: 1"


def test_reproducible_boundaries_and_comparison(parsed, chunker):
    document = parsed(PROSE)
    a, b = chunker.process(document), chunker.process(document)
    assert a.generation_id != b.generation_id
    assert boundaries(a) == boundaries(b)
    comparison = compare_results(a, b)
    assert comparison["same_boundaries"]
    assert comparison["left"]["coverage"]["covered_characters"] == len(PROSE)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "bad_count", "bad_source", "extra_text", "wrong_embedding"])
def test_validator_rejects_corrupt_results(parsed, chunker, counter, mutation):
    result = copy.deepcopy(chunker.process(parsed(PROSE)))
    if mutation == "drop":
        result.chunks.pop()
    elif mutation == "duplicate":
        result.chunks[0].source_spans.append(result.chunks[0].source_spans[0])
    elif mutation == "bad_count":
        result.chunks[0].token_count += 1
    elif mutation == "bad_source":
        result.chunks[0].source_spans[0].end_char += 99999
    elif mutation == "extra_text":
        result.chunks[0].text += " invented content"
        result.chunks[0].embedding_text += " invented content"
        result.chunks[0].token_count = counter.full_count(result.chunks[0].embedding_text)
    elif mutation == "wrong_embedding":
        result.chunks[0].embedding_text = "other topic"
        result.chunks[0].token_count = counter.full_count("other topic")
    with pytest.raises(ValidationError):
        ChunkValidator(counter, 480).validate(result)


def test_preserving_sentence_splitter_keeps_whitespace():
    chunker = object.__new__(PreservingSemanticChunker)
    text = "  А.\n\nБ, т.е. пример.  В конце.\n"
    assert "".join(chunker._split_sentences(text)) == text


def test_safe_splitter_progress():
    parts = SafeTextSplitter(lambda text: len(text) <= 7).split("А" * 50)
    assert parts[0][0] == 0 and parts[-1][1] == 50
    assert all(end > start for start, end in parts)
    with pytest.raises(ValidationError):
        SafeTextSplitter(lambda text: False).split("a")
