"""Offline installation and resource diagnostics; never downloads or starts services."""

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path

from . import __version__
from .errors import BearError


def diagnose(model_dir):
    packages = {}
    for name in ("bear-text-beta", "bear-fusion-lab", "chonkie", "numpy", "transformers",
                 "httpx", "fastapi", "starlette", "pdfplumber", "pypdfium2"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    model = Path(model_dir)
    tokenizer_present = (model / "tokenizer.json").is_file() and (model / "tokenizer_config.json").is_file()
    manifest = {}
    error = None
    profile_verified = False
    revision = None
    if (model / "bear-model.json").is_file():
        try:
            manifest = json.loads((model / "bear-model.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
                error = "invalid_manifest"
            else:
                from .embeddings import load_profile
                profile = load_profile(model)
                revision = profile.model_revision
                profile_verified = True
        except (BearError, OSError, ValueError, KeyError, TypeError):
            error = "invalid_manifest"
    return {"version": __version__, "python": platform.python_version(),
            "supported_python": platform.python_version_tuple()[:2] == ("3", "12"),
            "packages": packages, "native_formats": ["txt", "md", "html", "htm"],
            "fusion_installed": importlib.util.find_spec("bear_fusion") is not None,
            "model": {"path": str(model.resolve()), "tokenizer_ready": tokenizer_present and profile_verified,
                      "profile_verified": profile_verified,
                      "manifest_present": bool(manifest), "error": error,
                      "revision": revision},
            "external_services": {"checked": False, "reason": "offline_diagnostics"}}
