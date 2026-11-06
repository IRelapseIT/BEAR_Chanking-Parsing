from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from bear_text_beta.bootstrap import prepare_model
from bear_text_beta.embedding_pipeline import EmbeddingPipeline
from bear_text_beta.embeddings import InfinityEmbeddingProvider, load_profile
from bear_text_beta.errors import ValidationError
from bear_text_beta.model_profiles import model_profile


def test_bge_m3_profile_has_no_e5_or_bge_v15_prefixes(counter):
    import json
    profile = replace(model_profile("bge-m3"), model_revision="fixed", tokenizer_revision="fixed")
    counter = type(counter)(profile)
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [1.0] + [0.0] * 1023} for i, _ in enumerate(body["input"])]})
    provider = InfinityEmbeddingProvider("", profile, counter, client=httpx.Client(
        base_url="http://infinity/v1/", transport=httpx.MockTransport(handle)))
    pipeline = EmbeddingPipeline(provider, counter, profile, max_tokens=480)
    pipeline.embed_batch(["Документ"], "document")
    assert len(pipeline.embed(" Вопрос ")) == 1024
    assert [body["input"] for body in calls] == [["Документ"], ["Вопрос"]]


def test_prepare_bge_pins_revisions_and_rejects_wrong_directory(tmp_path, monkeypatch):
    import huggingface_hub
    revision = "b" * 40
    calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda model, token=False: SimpleNamespace(sha=revision)))
    def download(model, revision, local_dir, **kwargs):
        calls.append((model, revision))
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    model_dir = tmp_path / "model"
    prepare_model(model_dir, model="bge-m3")
    prepared = load_profile(model_dir)
    assert prepared.model_id == "BAAI/bge-m3"
    assert prepared.vector_dimension == 1024 and prepared.query_prefix == ""
    assert prepared.model_revision == prepared.tokenizer_revision == revision
    prepare_model(model_dir, model="bge-m3")
    assert len(calls) == 1
    with pytest.raises(ValidationError, match="another model"):
        prepare_model(model_dir, model="e5-small")
