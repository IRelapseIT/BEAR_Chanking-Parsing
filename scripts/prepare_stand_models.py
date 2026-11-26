"""Prepare only the model snapshots used by the verified local stand."""
import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from bear_text_beta.model_profiles import model_profile
from huggingface_hub import snapshot_download

BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_WEIGHTS = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
DOCLING = {
    "docling-project/docling-layout-heron": ("8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
                                             ["config.json", "preprocessor_config.json", "model.safetensors"]),
    "docling-project/docling-models": ("fc0f2d45e2218ea24bce5045f58a389aed16dc23", ["model_artifacts/tableformer/**"]),
}
PADDLE = {
    "PP-OCRv5_mobile_det": ("0d63e78e2b680928f6b1747d76a08db6e645efb7", (
        "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
        "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
        "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b")),
    "eslav_PP-OCRv5_mobile_rec": ("7553801264d3379d8d2e854971989e5e26c22e03", (
        "025039bac23eb4a308efcefa4d58eab3af440767815c6ba6938468bf6353ee5a",
        "3fb6e2e658f5139ff16e35260de8f0577f106a9505c902e1dfc1f4f1d03cc9cb",
        "f11057b05d8517868bca505271278973d706600d9dcc184cbcf5c4512091c32b")),
    "PP-DocLayout_plus-L": ("aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d", (
        "d60f782a16f96afb27e8280399899a94c3e9ffc694ffb2f913ea00af1c522f1e",
        "99bd09c0a1b9e88d35c65c305b91711d98055fc94b4a6eecd50da02e04729384",
        "24ca3e2e442164505e250deef59f7ee9a54ea12dd32875c9cd6155d959dc97da")),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(destination, content):
    staging = destination.with_suffix(".tmp")
    staging.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    staging.replace(destination)


def download(repo, revision, target, patterns):
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(target),
                      allow_patterns=patterns, token=False)


def prepare_bge(target):
    tokenizer = ["config.json", "config_sentence_transformers.json", "modules.json",
                 "sentencepiece.bpe.model", "sentence_bert_config.json", "special_tokens_map.json",
                 "tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json"]
    download("BAAI/bge-m3", BGE_REVISION, target, [
        *tokenizer, "pytorch_model.bin", "sparse_linear.pt", "colbert_linear.pt",
    ])
    weights = target / "pytorch_model.bin"
    if sha(weights) != BGE_WEIGHTS:
        raise ValueError("BGE-M3 weights do not match the pinned SHA-256")
    profile = replace(model_profile("bge-m3"), model_revision=BGE_REVISION, tokenizer_revision=BGE_REVISION)
    write_manifest(target / "bear-model.json", {
        "embedding_profile": asdict(profile),
        "tokenizer_checksums": {rel: sha(target / rel) for rel in tokenizer},
        "weights_present": True, "weights_file": weights.name,
        "weights_size_bytes": weights.stat().st_size, "weights_sha256": BGE_WEIGHTS,
    })


def prepare_docling(target):
    for repo, (revision, patterns) in DOCLING.items():
        download(repo, revision, target / repo.replace("/", "--"), patterns)
    write_manifest(target / "bear-revisions.json", {repo: config[0] for repo, config in DOCLING.items()})


def prepare_paddle(target):
    artifacts = ["inference.yml", "inference.json", "inference.pdiparams"]
    for model, (revision, digests) in PADDLE.items():
        directory = target / model
        download("PaddlePaddle/" + model, revision, directory, artifacts)
        checksums = dict(zip(artifacts, digests, strict=True))
        if any(sha(directory / name) != expected for name, expected in checksums.items()):
            raise ValueError(f"Paddle model does not match pinned SHA-256: {model}")
        write_manifest(directory / "bear-revision.json", {"revision": revision, "sha256": checksums})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--with-paddle", action="store_true")
    parser.add_argument("--with-docling", action="store_true")
    args = parser.parse_args()
    prepare_bge(args.root / "bge-m3")
    if args.with_paddle:
        prepare_paddle(args.root / "paddle-ocr")
    if args.with_docling:
        prepare_docling(args.root / "docling")


if __name__ == "__main__":
    main()

