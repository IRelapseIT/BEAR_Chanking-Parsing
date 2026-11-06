"""Opt-in smoke: local Qwen3 weights -> HTTP provider -> BEAR -> native Qdrant.

No downloads. Qdrant storage, tokenizer manifest and ports are temporary.
The HTTP endpoint is protocol-compatible test infrastructure, not Infinity itself.
"""
import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from bear_text_beta.errors import StorageError, ValidationError
from bear_text_beta.models import DocumentContext, EmbeddingProfile, ProcessingProfile
from bear_text_beta.runtime import Services


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def run(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    source = args.model_dir.resolve()
    revision = source.name
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Use a pinned Hugging Face snapshot directory named by its commit SHA")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                     torch_dtype=torch.float32).eval()
    profile = EmbeddingProfile(
        model_id="Qwen/Qwen3-Embedding-0.6B", model_revision=revision, tokenizer_revision=revision,
        vector_dimension=1024, max_input_tokens=512, document_prefix="", similarity_prefix="",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:")
    calls, model_lock = [], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, status=200):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply({"data": [{"id": profile.model_id}]})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError("Invalid body size")
                body = json.loads(self.rfile.read(size))
                texts = body["input"]
                if body["model"] != profile.model_id or not isinstance(texts, list) or not 1 <= len(texts) <= 16:
                    raise ValueError("Invalid inference request")
                if any(not isinstance(t, str) or not t.strip() or len(tokenizer.encode(t)) > 480 for t in texts):
                    raise ValueError("Invalid input; truncation is forbidden")
                with model_lock, torch.inference_mode():
                    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
                    hidden = model(**inputs, use_cache=False).last_hidden_state[:, -1]
                    vectors = torch.nn.functional.normalize(hidden.float(), p=2, dim=1).tolist()
                    calls.append(len(texts))
                self.reply({"data": [{"index": i, "embedding": vector}
                                     for i, vector in enumerate(vectors)]})
            except (ValueError, TypeError, KeyError) as exc:
                self.reply({"error": str(exc)}, 422)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks = {}
    try:
        with tempfile.TemporaryDirectory(prefix="bear-embedding-live-") as temporary:
            root = Path(temporary)
            model_dir = root / "tokenizer"
            tokenizer.save_pretrained(model_dir)
            checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in model_dir.iterdir() if p.is_file()}
            (model_dir / "bear-model.json").write_text(json.dumps({
                "embedding_profile": asdict(profile), "tokenizer_checksums": checksums}), encoding="utf-8")
            http_port, grpc_port = free_port(), free_port()
            config = root / "qdrant.yaml"
            config.write_text("telemetry_disabled: true\nlog_level: WARN\nstorage:\n  storage_path: "
                              + json.dumps(str(root / "storage")) + "\nservice:\n  host: 127.0.0.1\n"
                              + f"  http_port: {http_port}\n  grpc_port: {grpc_port}\n", encoding="utf-8")
            with (root / "qdrant.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([str(args.qdrant_exe.resolve()), "--config-path", str(config)],
                                           cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                services = None
                env_keys = ("BEAR_INFINITY_URL", "BEAR_INFINITY_MODEL", "BEAR_QDRANT_URL",
                            "BEAR_QDRANT_COLLECTION", "BEAR_CHUNKER_URL")
                saved_env = {key: os.environ.get(key) for key in env_keys}
                try:
                    qdrant_url = f"http://127.0.0.1:{http_port}"
                    with httpx.Client(timeout=1, trust_env=False) as client:
                        for _ in range(100):
                            if process.poll() is not None:
                                raise RuntimeError("Qdrant exited: " + (root / "qdrant.log").read_text())
                            try:
                                if client.get(qdrant_url + "/healthz").is_success:
                                    break
                            except httpx.HTTPError:
                                pass
                            time.sleep(0.1)
                        else:
                            raise RuntimeError("Qdrant did not become ready")
                    os.environ.update(BEAR_INFINITY_URL=f"http://127.0.0.1:{server.server_port}/v1",
                                      BEAR_INFINITY_MODEL=profile.model_id, BEAR_QDRANT_URL=qdrant_url,
                                      BEAR_QDRANT_COLLECTION="bear_embedding_live")
                    os.environ.pop("BEAR_CHUNKER_URL", None)
                    services = Services(root / "state", model_dir)
                    policy = ProcessingProfile(mode="token")
                    results = []
                    for doc_id, text in [
                        ("refund", "Магазин Норд возвращает деньги за товар в течение 12 рабочих дней."),
                        ("ssl", "Веб-сервер NGINX использует сертификат SSL для защищённого соединения HTTPS."),
                        ("stars", "Звёзды образуются из облаков межзвёздного газа и пыли.")]:
                        context = DocumentContext("smoke", "main", doc_id, 1, doc_id + ".txt")
                        result = services.process(text.encode(), context, policy)
                        before = copy.deepcopy(asdict(result))
                        services.indexer.index(result)
                        assert asdict(result) == before
                        results.append(result)
                    hits = services.indexer.search("  За сколько\n дней Норд вернёт деньги? ", "smoke", "main", 3)
                    assert hits and hits[0].payload["document_id"] == "refund"
                    checks["retrieval_top1"] = hits[0].payload["document_id"]
                    assert services.indexer.search("Возврат денег", "other", "main") == []
                    assert services.indexer.search("Возврат денег", "smoke", "other") == []
                    checks["tenant_space_isolation"] = True
                    start_calls = len(calls)
                    one = services.embeddings.embed("Повторный запрос")
                    two = services.embeddings.embed("  Повторный\n запрос  ")
                    assert one == two and len(calls) == start_calls + 1
                    assert abs(sum(v * v for v in one) - 1) < 1e-5
                    checks["query_cache_and_unit_vectors"] = True
                    start_calls = len(calls)
                    try:
                        services.embeddings.embed_batch(["слово " * 2000], "document")
                    except ValidationError:
                        assert len(calls) == start_calls
                    else:
                        raise AssertionError("Oversized input accepted")
                    checks["oversize_rejected_before_http"] = True
                    old = results[0]
                    updated = services.process("Норд возвращает деньги за 90 дней.".encode(),
                                               replace(old.context, document_version=2, checksum=""), policy)
                    original_upsert = services.vectors.upsert

                    def fail_upsert(*_):
                        raise StorageError("Injected write outage")

                    services.vectors.upsert = fail_upsert
                    try:
                        services.indexer.index(updated)
                    except StorageError:
                        assert old.generation_id in services.manifest.active_generations("smoke", "main")
                    else:
                        raise AssertionError("Injected failure did not fail")
                    services.vectors.upsert = original_upsert
                    services.indexer.retry(updated.generation_id)
                    active = services.manifest.active_generations("smoke", "main")
                    assert updated.generation_id in active and old.generation_id not in active
                    checks["failed_reindex_preserves_old_and_retry_activates_new"] = True
                    checks["documents"] = len(results)
                    checks["chunks"] = sum(len(result.chunks) for result in results)
                    checks["http_inference_requests"] = len(calls)
                finally:
                    if services is not None:
                        services.close()
                    for key, value in saved_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                    process.terminate()
                    process.wait(timeout=15)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return {"status": "passed", "profile": asdict(profile), "checks": checks,
            "endpoint": "temporary Infinity-compatible HTTP adapter with real CPU Qwen3 weights"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qdrant-exe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LIVE EMBEDDING INTEGRATION PASSED")
