import importlib.util
import json
from pathlib import Path


def test_actual_worker_stopped_on_memory_budget(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("demo_test", root / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo.JOBS = tmp_path
    job = tmp_path / "budget"
    job.mkdir()
    (job / "source.md").write_text("# Test\n\nHello", encoding="utf-8")
    config = {"filename": "source.md", "memory_mb": 1, "timeout": 30,
              "tokenizer": str(root.parent / "bear-text-beta/models/e5"),
              "upstream": str(root.parent / "research/RAG_VERSION_1-20261102"),
              "parser_python": str(root.parent / "research/rag-dedoc3-env/Scripts/python.exe")}
    (job / "request.json").write_text(json.dumps(config), encoding="utf-8")
    demo.execute("budget", config)
    assert demo.STATE["budget"]["status"] == "failed"
    assert "resource budget" in demo.STATE["budget"]["error"]
