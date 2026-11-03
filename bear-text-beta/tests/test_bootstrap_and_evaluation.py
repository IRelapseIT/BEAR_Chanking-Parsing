import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bear_text_beta.bootstrap import prepare_model
from bear_text_beta.embeddings import load_profile
from bear_text_beta.errors import ValidationError
from bear_text_beta.evaluation import evaluate
from bear_text_beta.indexing import IndexingService
from bear_text_beta.storage import SQLiteManifest


def test_model_preparation_pins_revision_and_checks_nested_configs(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []
    revision = "a" * 40
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda model, token=False: SimpleNamespace(sha=revision)))
    def download(model, revision, local_dir, **kwargs):
        calls.append(revision)
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "1_Pooling").mkdir()
        (root / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepare_model(tmp_path / "model")
    prepare_model(tmp_path / "model")
    assert calls == [revision]
    assert load_profile(tmp_path / "model").tokenizer_revision == revision
    (tmp_path / "model" / "1_Pooling" / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValidationError, match="Изменён"):
        load_profile(tmp_path / "model")


def test_evaluation_uses_isolated_modes_and_reports_actual_queries(tmp_path, parser, chunker, counter, embeddings, profile):
    from test_indexing import FakeVectors
    manifest = SQLiteManifest(tmp_path / "state.db")
    indexer = IndexingService(manifest, FakeVectors(), embeddings, counter, profile)
    def process(data, context, policy):
        result = chunker.process(parser.parse(data, context), policy)
        manifest.save(result)
        return result
    runtime = SimpleNamespace(profile=profile, parser=parser, indexer=indexer,
                              embeddings=embeddings, process=process)
    (tmp_path / "example.txt").write_text("Пример текста. Обработка документов.", encoding="utf-8")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"documents": [{"id": "doc", "file": "example.txt"}],
                                 "queries": [{"query": "документы", "document_ids": ["doc"]}]}), encoding="utf-8")
    report = evaluate(runtime, corpus)
    assert [entry["mode"] for entry in report["reports"]] == ["adaptive", "token", "semantic"]
    assert all(entry["queries"][0]["retrieved"] == ["doc"] for entry in report["reports"])
    assert all(entry["embedding_profile_id"] == profile.identifier() for entry in report["reports"])
    assert "document" in report["metric_level"]
