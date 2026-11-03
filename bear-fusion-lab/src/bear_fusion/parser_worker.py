"""Standalone worker: run with the upstream Python, not the BEAR environment."""
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo, source, output, expected = sys.argv[1:]
    output = Path(output)
    try:
        git = ["git", "-c", f"safe.directory={Path(repo).as_posix()}", "-C", repo]
        revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True, timeout=10).strip()
        changes = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"],
                                          text=True, timeout=10).strip()
        if revision != expected or changes:
            raise ValueError("Upstream revision differs or tracked files are modified")
        os.environ["DEDOC_OUTPUT_DIR"] = str(output.parent / "debug")
        os.environ["TEMP_DIR"] = str(output.parent / "temp")
        sys.path.insert(0, repo)
        from rag_parser.parsers.dedoc_parser import DedocParser
        result = DedocParser().parse(Path(source))
        payload = result.to_dict()
        payload["runtime"] = {"python": sys.version, "commit": revision,
                              "packages": {d.metadata["Name"]: d.version
                                           for d in importlib.metadata.distributions()}}
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - isolated process serializes all parser failures
        output.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
