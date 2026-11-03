"""Loopback-only bounded document queue and stage viewer. Python 3.12."""
import argparse
import base64
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID, uuid4

import psutil

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "bear-text-beta/src")]
from bear_fusion.parents import expand_context
from bear_fusion.quality import search

JOBS = ROOT / "demo-state"
LOCK = threading.Lock()
STATE = {}


def terminate_tree(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for item in reversed(children):
            try:
                item.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs([parent, *children], timeout=5)
    except psutil.NoSuchProcess:
        pass


def execute(job_id, config):
    job = JOBS / job_id
    with LOCK:
        STATE[job_id] = {"status": "running"}
    started = time.monotonic()
    peak = 0
    try:
        with (job / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, str(ROOT / "demo_worker.py"), str(job)],
                                       cwd=job, stdout=log, stderr=log)
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        family = [parent, *parent.children(recursive=True)]
                        rss = sum(p.memory_info().rss for p in family if p.is_running())
                        peak = max(peak, rss)
                        disk = 0
                        for path in job.rglob("*"):
                            try:
                                if path.is_file():
                                    disk += path.stat().st_size
                            except FileNotFoundError:
                                pass  # Worker may finish and remove its temporary files concurrently.
                        if (time.monotonic() - started > config["timeout"] or
                                rss > config["memory_mb"] * 1024**2 or len(family) > 12 or
                                disk > 256 * 1024**2):
                            raise RuntimeError("Job resource budget exceeded (time/RAM/processes/disk)")
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.2)
                if process.returncode or not (job / "result.json").exists():
                    raise RuntimeError((job / "worker.log").read_text("utf-8", errors="replace")[-4000:])
            finally:
                terminate_tree(process)
        status = {"status": "done", "peak_rss_mb": round(peak / 1024**2, 1)}
    except Exception as exc:  # noqa: BLE001 - failed jobs must not stop the queue
        status = {"status": "failed", "error": str(exc)}
    with LOCK:
        STATE[job_id] = status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--upstream", type=Path, default=ROOT.parent / "research/RAG_VERSION_1-20261102")
    parser.add_argument("--parser-python", type=Path, default=ROOT.parent / "research/rag-dedoc3-env/Scripts/python.exe")
    parser.add_argument("--tokenizer", type=Path, default=ROOT.parent / "bear-text-beta/models/e5")
    args = parser.parse_args()
    if min(args.memory_mb, args.timeout) <= 0:
        parser.error("Budgets must be positive")
    JOBS.mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    config = {"upstream": str(args.upstream.resolve()), "parser_python": str(args.parser_python.resolve()),
              "tokenizer": str(args.tokenizer.resolve()), "memory_mb": args.memory_mb, "timeout": args.timeout}
    context_counters = []
    counter_lock = threading.Lock()
    def counter():
        with counter_lock:
            if not context_counters:
                from bear_text_beta.embeddings import HFTokenCounter
                from bear_text_beta.models import EmbeddingProfile
                context_counters.append(HFTokenCounter(args.tokenizer, EmbeddingProfile()))
        return context_counters[0]

    class Handler(BaseHTTPRequestHandler):
        def response(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/jobs":
                with LOCK:
                    ids = list(STATE)
                return self.response([{ "job_id": key, "filename": json.loads(
                    (JOBS / key / "request.json").read_text("utf-8"))["filename"]}
                    for key in ids if (JOBS / key / "request.json").exists()])
            if self.path == "/":
                data = (ROOT / "demo.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers(); self.wfile.write(data)
                return
            try:
                job_id = self.path.removeprefix("/jobs/")
                UUID(job_id)
                with LOCK:
                    state = dict(STATE[job_id])
                stages = {}
                for name in ("original", "parsed", "chunked", "result"):
                    path = JOBS / job_id / f"{name}.json"
                    if path.exists():
                        stages[name] = json.loads(path.read_text("utf-8"))
                self.response({**state, "stages": stages})
            except (ValueError, KeyError):
                self.response({"error": "Unknown job"}, 404)

        def do_POST(self):
            if self.headers.get("X-Bear-Demo") != "local":
                return self.response({"error": "Missing local demo header"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 15 * 1024**2:
                    raise ValueError("Demo upload limit: 10 MiB per file")
                body = json.loads(self.rfile.read(length))
                if self.path in {"/search", "/context"}:
                    job_id = str(UUID(body["job_id"]))
                    data = json.loads((JOBS / job_id / "result.json").read_text("utf-8"))
                    if self.path == "/context":
                        return self.response(expand_context(data, body["chunk_id"], counter(), int(body.get("max_tokens", 480))))
                    hits = search(data["result"]["chunks"], body["query"])
                    if body.get("expand"):
                        for hit in hits:
                            hit["expanded_context"] = expand_context(data, hit["chunk"]["id"], counter())
                    return self.response(hits)
                if self.path != "/jobs":
                    return self.response({"error": "Unknown endpoint"}, 404)
                filename = Path(body["filename"]).name
                if (filename != body["filename"] or any(c in filename for c in ':\\/') or
                        Path(filename).suffix.lower() not in {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}):
                    raise ValueError("Unsupported filename")
                data = base64.b64decode(body["data"], validate=True)
                if len(data) > 10 * 1024**2:
                    raise ValueError("Demo upload limit: 10 MiB")
                with LOCK:
                    if sum(s["status"] in {"queued", "running"} for s in STATE.values()) >= 8:
                        return self.response({"error": "Queue full; wait for running jobs"}, 429)
                    job_id = str(uuid4())
                    STATE[job_id] = {"status": "queued"}
                job = JOBS / job_id
                job.mkdir()
                (job / filename).write_bytes(data)
                (job / "request.json").write_text(json.dumps({**config, "filename": filename}), encoding="utf-8")
                pool.submit(execute, job_id, config)
                self.response({"job_id": job_id}, 202)
            except (ValueError, KeyError, OSError) as exc:
                self.response({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"BEAR demo: http://127.0.0.1:{args.port} — CPU workers={args.workers}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.shutdown(wait=True)


if __name__ == "__main__":
    main()
