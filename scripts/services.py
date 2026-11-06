"""Own persistent local services only; PID identity is checked before stopping."""
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime"
REGISTRY = STATE / "processes.json"


def configuration():
    return json.loads((ROOT / "config.local.json").read_text(encoding="utf-8"))


def stop():
    if not REGISTRY.exists():
        return
    registry = json.loads(REGISTRY.read_text())
    for name, entry in reversed(list(registry.items())):
        try:
            process = psutil.Process(entry["pid"])
            if abs(process.create_time() - entry["created"]) > .01:
                continue
            command = process.cmdline()
            if command != entry["command"]:
                raise RuntimeError(f"PID identity mismatch: {name}")
            children = process.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(children, timeout=15)
            process.terminate()
            process.wait(timeout=20)
        except psutil.NoSuchProcess:
            pass
    REGISTRY.unlink()


def launch():
    config = configuration()
    if REGISTRY.exists():
        raise RuntimeError("Registry exists; inspect status or stop your previous services first")
    STATE.mkdir(exist_ok=True)
    (STATE / "logs").mkdir(exist_ok=True)
    ports = config["ports"]
    for port in ports.values():
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied; no foreign process will be stopped") from exc
    qconfig = STATE / "qdrant.yaml"
    qconfig.write_text("telemetry_disabled: true\nlog_level: WARN\nstorage:\n  storage_path: "
                       + json.dumps(str(STATE / "qdrant-storage")) + "\nservice:\n  host: 127.0.0.1\n"
                       + f"  http_port: {ports['qdrant']}\n  grpc_port: {ports['qdrant_grpc']}\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(BEAR_MODEL_DIR=str(ROOT / "models/bge-m3"), BEAR_STATE_DIR=str(ROOT / "state-bge-m3"),
               BEAR_INFINITY_URL=f"http://127.0.0.1:{ports['embedding']}/v1", BEAR_INFINITY_MODEL="BAAI/bge-m3",
               BEAR_QDRANT_URL=f"http://127.0.0.1:{ports['qdrant']}",
               BEAR_QDRANT_COLLECTION=config["collection"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", INFINITY_ANONYMOUS_USAGE_STATS="0")
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(INFINITY_BETTERTRANSFORMER="false", INFINITY_MODEL_WARMUP="false", DO_NOT_TRACK="1")
    if config.get("docling_python") and config.get("docling_artifacts"):
        if not Path(config["docling_python"]).is_file() or not Path(config["docling_artifacts"]).is_dir():
            raise RuntimeError("Configured Docling runtime/artifacts are missing")
        env.update(BEAR_DOCLING_PYTHON=config["docling_python"], BEAR_DOCLING_ARTIFACTS=config["docling_artifacts"])
    for key in ("BEAR_TEST_COUNTER", "BEAR_OFFLINE", "BEAR_CHUNKER_URL", "PYTHONPATH"):
        env.pop(key, None)
    python = str(ROOT / ".venv/Scripts/python.exe")
    infinity = str(ROOT / ".infinity-venv/Scripts/python.exe")
    commands = {
        "qdrant": [str(ROOT / config["qdrant_exe"]), "--config-path", str(qconfig)],
        "embedding": [infinity, str(ROOT / "scripts/infinity_server.py"), "v2", "--model-id", str(ROOT / "models/bge-m3"),
                      "--served-model-name", "BAAI/bge-m3", "--engine", "torch", "--device", "cpu",
                      "--url-prefix", "/v1", "--host", "127.0.0.1",
                      "--dtype", "float32", "--batch-size", "4", "--port", str(ports["embedding"])],
        "api": [python, "-m", "uvicorn", "bear_text_beta.api:app", "--host", "127.0.0.1",
                "--port", str(ports["api"])],
        "worker": [python, "-m", "bear_text_beta.process_worker"],
    }
    registry = {}
    try:
        for name, command in commands.items():
            with (STATE / "logs" / (name + ".log")).open("ab") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            process = psutil.Process(child.pid)
            # Python console launchers can delegate to another process; register the actual root command.
            registry[name] = {"pid": child.pid, "created": process.create_time(), "command": process.cmdline()}
            REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
            if name in {"qdrant", "embedding", "api"}:
                route = {"qdrant": "healthz", "embedding": "v1/models", "api": "live"}[name]
                url = f"http://127.0.0.1:{ports[name]}/{route}"
                with httpx.Client(timeout=2, trust_env=False) as client:
                    for _ in range(300):
                        if child.poll() is not None:
                            raise RuntimeError(f"{name} exited; inspect runtime/logs/{name}.log")
                        try:
                            if client.get(url).is_success:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(.2)
                    else:
                        raise RuntimeError(f"{name} readiness timeout")
        print(json.dumps(registry, indent=2))
    except Exception:
        stop()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    if args.action == "start":
        launch()
    elif args.action == "stop":
        stop()
    else:
        print(REGISTRY.read_text() if REGISTRY.exists() else "No owned services")
