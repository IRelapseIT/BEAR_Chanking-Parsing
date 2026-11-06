import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from .errors import ValidationError
from .model_profiles import model_profile


def prepare_model(destination: Path, *, weights=False, model="bge-m3"):
    from huggingface_hub import HfApi, snapshot_download
    template = model_profile(model)
    manifest = destination / "bear-model.json"
    if manifest.exists():
        from .embeddings import load_profile
        profile = load_profile(destination)
        if profile.model_id != template.model_id:
            raise ValidationError("Model directory belongs to another model; choose a new directory")
        revision = profile.model_revision
        if not weights:
            return {"model_revision": revision, "status": "already_prepared"}
    else:
        profile = template
        revision = HfApi().model_info(profile.model_id, token=False).sha
        profile = replace(profile, model_revision=revision, tokenizer_revision=revision)
    patterns = ["*.json", "*.model", "tokenizer*", "vocab*", "special_tokens_map.json"]
    if weights:
        patterns.extend(["*.safetensors", "pytorch_model*.bin", "*.pt"])
    snapshot_download(profile.model_id, revision=revision, local_dir=str(destination),
                      allow_patterns=patterns, token=False)
    checksums = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in destination.rglob("*") if path.is_file() and ".cache" not in path.parts
                 and path.suffix in {".json", ".model"}
                 and path.name != "bear-model.json"}
    manifest.write_text(json.dumps({"embedding_profile": asdict(profile), "tokenizer_checksums": checksums,
                                    "weights_present": (any(destination.glob("*.safetensors"))
                                                        or any(destination.glob("pytorch_model*.bin")))},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_revision": revision, "status": "prepared", "weights": weights}


def lock_dependencies(destination: Path):
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                      if d.metadata["Name"].lower() not in {"bear-text-beta", "pip"})
    destination.write_text("# Generated from verified Python 3.12 environment\n" + "\n".join(packages) + "\n",
                           encoding="utf-8")


def prepare_images(destination: Path):
    images = {"INFINITY_IMAGE": "michaelf34/infinity:latest-cpu", "QDRANT_IMAGE": "qdrant/qdrant:latest"}
    pins = []
    for name, image in images.items():
        try:
            subprocess.run(["docker", "pull", image], check=True)
            result = subprocess.run(["docker", "image", "inspect", image,
                                     "--format", "{{index .RepoDigests 0}}"],
                                    check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValidationError("Запустите Docker Desktop для подготовки закреплённых образов") from exc
        digest = result.stdout.strip()
        if "@sha256:" not in digest:
            raise ValidationError("Не удалось закрепить Docker digest")
        pins.append(f"{name}={digest}")
    destination.write_text("\n".join(pins) + "\n", encoding="utf-8")
