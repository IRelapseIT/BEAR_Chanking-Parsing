from dataclasses import replace

import pytest
from chonkie.tokenizer import CharacterTokenizer

from bear_text_beta.chunking import ChunkingService
from bear_text_beta.models import DocumentContext, EmbeddingProfile
from bear_text_beta.parsing import HTMLParser, MDParser, ParsingPipeline, TXTParser


class TestCounter:
    __test__ = False

    def __init__(self, profile):
        self.profile = profile
        self.tokenizer = CharacterTokenizer()

    def count(self, text):
        return len(text)

    def full_count(self, text, purpose="document"):
        return len(self.profile.prefix(purpose) + text) + 2


class TestEmbeddings:
    __test__ = False

    def __init__(self, dimension=4):
        self.dimension = dimension
        self.calls = []
        self.fail = False

    def embed_batch(self, texts, purpose):
        from bear_text_beta.errors import EmbeddingError
        self.calls.append((list(texts), purpose))
        if self.fail:
            raise EmbeddingError("simulated failure")
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]

    def health(self):
        return {"ready": not self.fail}


@pytest.fixture
def context():
    return DocumentContext("tenant-a", "space-a", "doc-a", 1, "document.txt")


@pytest.fixture
def profile():
    return EmbeddingProfile(model_id="test", model_revision="fixed", tokenizer_revision="fixed",
                            vector_dimension=4)


@pytest.fixture
def counter(profile):
    return TestCounter(profile)


@pytest.fixture
def embeddings():
    return TestEmbeddings()


@pytest.fixture
def chunker(counter, embeddings, profile):
    return ChunkingService(counter, embeddings, profile)


@pytest.fixture
def parser(tmp_path):
    return ParsingPipeline(tmp_path / "assets", extra_parsers={
        ".txt": TXTParser(), ".html": HTMLParser(), ".htm": HTMLParser(), ".md": MDParser(),
    })


@pytest.fixture
def parsed(parser, context):
    def parse(text, filename="document.txt"):
        return parser.parse(text.encode("utf-8"), replace(context, filename=filename))
    return parse
