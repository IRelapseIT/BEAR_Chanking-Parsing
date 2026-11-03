import httpx
import pytest

from bear_text_beta.embeddings import InfinityEmbeddingProvider
from bear_text_beta.errors import EmbeddingError


def test_batch_prefixes_order_and_limits(profile, counter):
    requests = []
    def handle(request):
        import json
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1.0, 0.0, 0.0, float(i)]}
            for i in reversed(range(len(payload["input"])))
        ]})
    client = httpx.Client(base_url="http://infinity/v1/", transport=httpx.MockTransport(handle))
    provider = InfinityEmbeddingProvider("", profile, counter, client=client)
    vectors = provider.embed_batch(["текст"] * 20, "document")
    assert len(requests) == 2 and len(vectors) == 20
    assert requests[0]["input"][0] == "passage: текст"
    assert vectors[1][-1] == 1.0
    provider.embed_batch(["вопрос"], "query")
    provider.embed_batch(["сравнение"], "similarity")
    assert requests[-2]["input"] == ["query: вопрос"]
    assert requests[-1]["input"] == ["query: сравнение"]
    with pytest.raises(EmbeddingError):
        provider.embed_batch(["a" * 500], "document")


@pytest.mark.parametrize("vector", [[1.0], [0.0] * 4, [float("nan")] * 4, [True] * 4])
def test_invalid_vectors(profile, counter, vector):
    def handle(request):
        # NaN cannot be emitted as valid JSON; test the validator directly for that case.
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector}]})
    from bear_text_beta.embeddings import validate_vectors
    with pytest.raises(EmbeddingError):
        validate_vectors([vector], 1, 4)
    if vector != [float("nan")] * 4 and all(isinstance(v, (bool, int, float)) and v == v for v in vector):
        provider = InfinityEmbeddingProvider("", profile, counter, client=httpx.Client(
            base_url="http://test/", transport=httpx.MockTransport(handle)))
        with pytest.raises(EmbeddingError):
            provider.embed_batch(["text"], "similarity")


@pytest.mark.parametrize("failure", ["timeout", "server"])
def test_retry_once(profile, counter, failure):
    calls = []
    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("simulated", request=request)
        return httpx.Response(503, json={"error": "down"})
    provider = InfinityEmbeddingProvider("", profile, counter, client=httpx.Client(
        base_url="http://test/", transport=httpx.MockTransport(handle)))
    with pytest.raises(EmbeddingError):
        provider.embed_batch(["text"], "similarity")
    assert len(calls) == 2


def test_invalid_response_indices(profile, counter):
    provider = InfinityEmbeddingProvider("", profile, counter, client=httpx.Client(
        base_url="http://test/", transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"data": [{"index": 1, "embedding": [1.0, 0.0, 0.0, 0.0]}]}))))
    with pytest.raises(EmbeddingError):
        provider.embed_batch(["text"], "query")
