import json
import sys

import prepare_stand_models as models
import pytest


def test_bge_manifest_keeps_pinned_model_and_tokenizer(tmp_path, monkeypatch):
    calls = []

    def local_snapshot(repo, revision, target, patterns):
        calls.append((repo, revision))
        for relative in patterns:
            file = target / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(relative, encoding="utf-8")

    monkeypatch.setattr(models, "download", local_snapshot)
    with pytest.raises(ValueError, match="SHA-256"):
        models.prepare_bge(tmp_path)
    assert not (tmp_path / "bear-model.json").exists()
    monkeypatch.setattr(models, "BGE_WEIGHTS", models.sha(tmp_path / "pytorch_model.bin"))
    models.prepare_bge(tmp_path)
    assert calls == [("BAAI/bge-m3", models.BGE_REVISION)] * 2
    manifest = json.loads((tmp_path / "bear-model.json").read_text(encoding="utf-8"))
    assert manifest["embedding_profile"]["model_revision"] == models.BGE_REVISION
    assert manifest["embedding_profile"]["tokenizer_revision"] == models.BGE_REVISION
    assert manifest["tokenizer_checksums"]["tokenizer.json"] == models.sha(tmp_path / "tokenizer.json")


def test_offline_setup_writes_pinned_ocr_manifests(tmp_path, monkeypatch):
    artifacts = ["inference.yml", "inference.json", "inference.pdiparams"]
    snapshot_calls = []

    def fake_snapshot(repo_id, revision, local_dir, allow_patterns, token):
        directory = models.Path(local_dir)
        directory.mkdir(parents=True)
        snapshot_calls.append((repo_id, revision))
        if repo_id.startswith("PaddlePaddle/"):
            for file in allow_patterns:
                (directory / file).write_bytes(b"fixture")

    monkeypatch.setattr(models, "snapshot_download", fake_snapshot)
    bge = []
    monkeypatch.setattr(models, "prepare_bge", bge.append)
    one_model = "PP-OCRv5_mobile_det"
    revision = models.PADDLE[one_model][0]
    digests = ["e1dd0028212961844a6882f91f6b6e28bc4c0a0cf9a2d953cedc176111dc1cfe"] * 3
    # Calculate the fixture digest independently of the model download.
    import hashlib
    digests = [hashlib.sha256(b"fixture").hexdigest()] * 3
    monkeypatch.setattr(models, "PADDLE", {one_model: (revision, tuple(digests))})
    monkeypatch.setattr(sys, "argv", ["setup", "--root", str(tmp_path), "--with-paddle", "--with-docling"])
    models.main()
    assert bge == [tmp_path / "bge-m3"]
    assert (f"PaddlePaddle/{one_model}", revision) in snapshot_calls
    with (tmp_path / "paddle-ocr" / one_model / "bear-revision.json").open(encoding="utf-8") as manifest:
        assert json.load(manifest) == {"revision": revision, "sha256": dict(zip(artifacts, digests, strict=True))}
    with (tmp_path / "docling" / "bear-revisions.json").open(encoding="utf-8") as manifest:
        assert json.load(manifest) == {repo: spec[0] for repo, spec in models.DOCLING.items()}
