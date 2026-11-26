"""Start, inspect and stop the local BEAR stack using one runtime directory."""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("qdrant", "embedding", "api", "worker")
READINESS = {"qdrant": "healthz", "embedding": "v1/models", "api": "live"}
PATH_DEFAULTS = {
    "runtime_dir": "runtime",
    "state_dir": "state-bge-m3",
    "api_python": ".venv/Scripts/python.exe",
    "infinity_python": ".infinity-venv/Scripts/python.exe",
    "model_dir": "models/bge-m3",
    "qdrant_exe": "runtime/bin/qdrant.exe",
}


def configuration(path=None):
    defaults = json.loads((ROOT / "config.local.json").read_text(encoding="utf-8-sig"))
    override = path or os.environ.get("BEAR_LOCAL_CONFIG")
    if override:
        custom = json.loads(Path(override).read_text(encoding="utf-8-sig"))
        if not isinstance(custom, dict) or not isinstance(custom.get("ports", {}), dict):
            raise ValueError("Expected a configuration object with a ports object")
        defaults = {**defaults, **custom, "ports": {**defaults["ports"], **custom.get("ports", {})}}
    ports = tuple(defaults["ports"].values())
    if (any(type(port) is not int or not 0 < port <= 65535 for port in ports)
            or len(ports) != len(set(ports))):
        raise ValueError("Expected distinct integer ports between 1 and 65535")
    return {**PATH_DEFAULTS, **defaults}


def runtime_paths(config):
    runtime = ROOT / config["runtime_dir"]
    return {"runtime": runtime, "registry": runtime / "processes.json", "logs": runtime / "logs",
            "qdrant_config": runtime / "qdrant.yaml", "qdrant_storage": runtime / "qdrant-storage"}


def service_environment(config, paths):
    env = dict(os.environ)
    for name in ("BEAR_TEST_COUNTER", "BEAR_OFFLINE", "BEAR_CHUNKER_URL", "BEAR_DOCLING_PYTHON",
                 "BEAR_DOCLING_ARTIFACTS", "BEAR_PADDLE_PYTHON", "BEAR_PADDLE_MODELS",
                 "BEAR_PADDLE_LAYOUT_MODEL", "BEAR_TESSERACT", "BEAR_LIBREOFFICE"):
        env.pop(name, None)
    ports = config["ports"]
    source_dirs = ("bear-text-beta/src", "bear-fusion-lab/src", *config.get("package_dirs", ()))
    env.update(
        BEAR_MODEL_DIR=str(ROOT / config["model_dir"]),
        BEAR_STATE_DIR=str(ROOT / config["state_dir"]),
        BEAR_INSTANCE_REGISTRY=str(paths["registry"]),
        BEAR_INFINITY_URL=f"http://127.0.0.1:{ports['embedding']}/v1",
        BEAR_INFINITY_MODEL="BAAI/bge-m3",
        BEAR_QDRANT_URL=f"http://127.0.0.1:{ports['qdrant']}",
        BEAR_QDRANT_COLLECTION=config["collection"],
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join(str(ROOT / rel) for rel in source_dirs),
        INFINITY_ANONYMOUS_USAGE_STATS="0", INFINITY_BETTERTRANSFORMER="false",
        INFINITY_MODEL_WARMUP="false", DO_NOT_TRACK="1",
    )
    for key, variable in (("docling_python", "BEAR_DOCLING_PYTHON"), ("docling_artifacts", "BEAR_DOCLING_ARTIFACTS"),
                          ("paddle_python", "BEAR_PADDLE_PYTHON"), ("paddle_models", "BEAR_PADDLE_MODELS"),
                          ("paddle_layout_model", "BEAR_PADDLE_LAYOUT_MODEL"),
                          ("tesseract_exe", "BEAR_TESSERACT"), ("libreoffice_exe", "BEAR_LIBREOFFICE")):
        value = config.get(key)
        if value:
            env[variable] = str(ROOT / value)
    return env


def require_python(binary, component, module):
    if not binary.is_file():
        raise RuntimeError(f"{component}: interpreter missing at {binary}")
    probe = (
        "import sys, importlib.util; print(sys.version.split()[0]); "
        "sys.exit(4 if sys.version_info[:2] != (3, 12) else "
        "5 if importlib.util.find_spec(sys.argv[1]) is None else 0)"
    )
    result = subprocess.run([str(binary), "-I", "-c", probe, module], check=False, timeout=10,
                            encoding="utf-8", errors="replace", capture_output=True)
    if result.returncode == 4:
        raise RuntimeError(f"{component}: Python 3.12 required, found {result.stdout.strip()}")
    if result.returncode:
        raise RuntimeError(f"{component}: Python environment cannot resolve {module}: {binary}")
    return result.stdout.strip()


def external_version(executable, label, *, timeout=10):
    if not executable:
        return None
    result = subprocess.run([str(executable), "--version"], check=False, timeout=timeout,
                            encoding="utf-8", errors="replace", capture_output=True)
    version = (result.stdout or result.stderr).strip().splitlines()
    if result.returncode or not version:
        raise RuntimeError(f"{label}: version probe failed at {executable}")
    return version[0]


def inspect_environment(config):
    for path in config.get("package_dirs", ()):
        if not (ROOT / path).is_dir():
            raise RuntimeError(f"Python package directory missing at {ROOT / path}")
    api = require_python(ROOT / config["api_python"], "API", "fastapi")
    infinity = require_python(ROOT / config["infinity_python"], "Infinity", "infinity_emb")
    qdrant = ROOT / config["qdrant_exe"]
    if not qdrant.is_file():
        raise RuntimeError(f"Qdrant: executable missing at {qdrant}")
    model_dir = ROOT / config["model_dir"]
    manifest_path = model_dir / "bear-model.json"
    if not manifest_path.is_file() or not (model_dir / "tokenizer.json").is_file():
        raise RuntimeError(f"BGE-M3: model manifest/tokenizer missing at {model_dir}")
    profile = json.loads(manifest_path.read_text(encoding="utf-8"))["embedding_profile"]
    if profile["model_id"] != "BAAI/bge-m3" or not profile["model_revision"]:
        raise RuntimeError("BGE-M3: model manifest does not identify a pinned BGE-M3 revision")
    weights = (*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin"))
    if not any(path.is_file() and path.stat().st_size > 0 for path in weights):
        raise RuntimeError(f"BGE-M3: model weights missing at {model_dir}")
    optional = {}
    for tool, executable, label in (
        ("tesseract", config.get("tesseract_exe"), "Tesseract"),
        ("libreoffice", config.get("libreoffice_exe"), "LibreOffice"),
    ):
        filename = ROOT / executable if executable else (shutil.which(tool) or (
            shutil.which("soffice") if tool == "libreoffice" else None))
        if tool == "libreoffice" and filename and Path(filename).suffix == ".exe":
            console = Path(filename).with_suffix(".com")
            if console.is_file():
                filename = console
        version = external_version(filename, label)
        optional[tool] = {"available": version is not None, "version": version}
    if optional["tesseract"]["available"]:
        cmd = str(ROOT / config["tesseract_exe"]) if config.get("tesseract_exe") else shutil.which("tesseract")
        langs = subprocess.run([cmd, "--list-langs"], check=True, timeout=10,
                               encoding="utf-8", errors="replace", capture_output=True)
        optional["tesseract"]["languages"] = langs.stdout.splitlines()[1:]
    for label, module, model_key in (("docling", "docling", "docling_artifacts"),
                                     ("paddle", "paddleocr", "paddle_models")):
        binary, model = config.get(f"{label}_python"), config.get(model_key)
        if not binary and not model:
            optional[label] = {"available": False}
            continue
        if not binary or not model or not (ROOT / model).is_dir():
            raise RuntimeError(f"{label}: configure both interpreter and model directory")
        python = require_python(ROOT / binary, label, module)
        artifacts = ROOT / model
        required = (("PP-OCRv5_mobile_det/inference.json", "eslav_PP-OCRv5_mobile_rec/inference.json")
                    if label == "paddle" else ("docling-project--docling-layout-heron/model.safetensors",
                                               "docling-project--docling-models/model_artifacts/tableformer"))
        if any(not (artifacts / rel).exists() for rel in required):
            raise RuntimeError(f"{label}: required artifacts missing at {artifacts}")
        optional[label] = {"available": True, "python": python}
    layout = config.get("paddle_layout_model")
    if layout and (not optional["paddle"]["available"] or not (ROOT / layout / "inference.json").is_file()):
        raise RuntimeError(f"Paddle layout: model missing at {layout}")
    optional["paddle_layout"] = {"available": bool(layout)}
    return {"python": {"api": api, "infinity": infinity}, "model": profile, "qdrant": external_version(qdrant, "Qdrant"),
            "optional": optional}


def owned_process(entry):
    try:
        process = psutil.Process(entry["pid"])
        if abs(process.create_time() - entry["created"]) <= .01 and process.cmdline() == entry["command"]:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def status(config):
    paths = runtime_paths(config)
    registry = paths["registry"]
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    output = {"api": f"http://127.0.0.1:{config['ports']['api']}/", "services": {},
              "runtime": str(paths["runtime"])}
    for name in SERVICES:
        entry = entries.get(name)
        owned = owned_process(entry) if entry else None
        output["services"][name] = {"pid": entry["pid"] if owned else None, "running": owned is not None}
    return output


def stop(config):
    registry = runtime_paths(config)["registry"]
    if not registry.exists():
        return
    entries = json.loads(registry.read_text(encoding="utf-8"))
    for entry in reversed(tuple(entries.values())):
        process = owned_process(entry)
        if process is None:
            continue
        children = process.children(recursive=True)
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=15)
        if alive:
            raise RuntimeError(f"Owned process tree did not stop: {entry['pid']}")
        try:
            process.terminate()
            process.wait(timeout=20)
        except psutil.NoSuchProcess:
            pass
    registry.unlink()


def write_registry(path, entries):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(path)


def wait_http(name, port, child, *, timeout):
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/{READINESS[name]}"
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"{name} exited; inspect logs/{name}.log")
            try:
                if client.get(url).is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.35)
    raise RuntimeError(f"{name} readiness timeout; inspect logs/{name}.log")


def launch(config):
    paths = runtime_paths(config)
    if paths["registry"].exists():
        raise RuntimeError("Registry exists; inspect status, then stop the previous launch first")
    print(json.dumps(inspect_environment(config), indent=2), flush=True)
    ports = config["ports"]
    for port in ports.values():
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied; existing processes were preserved") from exc
    paths["logs"].mkdir(parents=True, exist_ok=True)
    with paths["registry"].open("x", encoding="utf-8") as allocated:
        allocated.write("{}")
    entries = {}
    try:
        qconfig = paths["qdrant_config"]
        qconfig.write_text("telemetry_disabled: true\nlog_level: WARN\nstorage:\n  storage_path: "
                           + json.dumps(str(paths["qdrant_storage"])) + "\nservice:\n  host: 127.0.0.1\n"
                           + f"  http_port: {ports['qdrant']}\n  grpc_port: {ports['qdrant_grpc']}\n",
                           encoding="utf-8")
        env = service_environment(config, paths)
        api_python = str(ROOT / config["api_python"])
        commands = {
            "qdrant": [str(ROOT / config["qdrant_exe"]), "--config-path", str(qconfig)],
            "embedding": [str(ROOT / config["infinity_python"]), str(ROOT / "scripts/infinity_server.py"),
                          "v2", "--model-id", env["BEAR_MODEL_DIR"], "--served-model-name", "BAAI/bge-m3",
                          "--engine", "torch", "--device", "cpu", "--url-prefix", "/v1", "--host", "127.0.0.1",
                          "--dtype", "float32", "--batch-size", "4", "--port", str(ports["embedding"])],
            "api": [api_python, "-m", "uvicorn", "bear_text_beta.api:app", "--host", "127.0.0.1",
                    "--port", str(ports["api"])],
            "worker": [api_python, "-m", "bear_text_beta.process_worker"],
        }
        for name, command in commands.items():
            with (paths["logs"] / f"{name}.log").open("ab") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                process = psutil.Process(child.pid)
                entries[name] = {"pid": child.pid, "created": process.create_time(), "command": process.cmdline()}
            except psutil.NoSuchProcess as exc:
                raise RuntimeError(f"{name} exited; inspect {paths['logs'] / f'{name}.log'}") from exc
            write_registry(paths["registry"], entries)
            if name in READINESS:
                wait_http(name, ports[name], child, timeout=240 if name == "embedding" else 30)
            else:
                time.sleep(.4)
                if child.poll() is not None:
                    raise RuntimeError(f"{name} exited; inspect {paths['logs'] / f'{name}.log'}")
    except (Exception, KeyboardInterrupt):
        write_registry(paths["registry"], entries)
        stop(config)
        raise
    return status(config)


def run_foreground(config):
    print(json.dumps(launch(config), indent=2), flush=True)
    registry = runtime_paths(config)["registry"]
    try:
        while registry.exists():
            time.sleep(1)
            snapshot = status(config)
            if not registry.exists():
                break
            if not all(entry["running"] for entry in snapshot["services"].values()):
                names = ", ".join(name for name, entry in snapshot["services"].items() if not entry["running"])
                raise RuntimeError(f"Stand services stopped unexpectedly: {names}")
    except KeyboardInterrupt:
        pass
    finally:
        stop(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "run", "stop", "status", "doctor"])
    parser.add_argument("--config", type=Path, help="Local JSON overrides. Relative tool paths use the checkout root.")
    args = parser.parse_args()
    try:
        config = configuration(args.config)
        if args.action == "start":
            print(json.dumps(launch(config), indent=2))
        elif args.action == "run":
            run_foreground(config)
        elif args.action == "stop":
            stop(config)
        elif args.action == "doctor":
            print(json.dumps(inspect_environment(config), indent=2))
        else:
            print(json.dumps(status(config), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, psutil.Error) as exc:
        print(f"BEAR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
