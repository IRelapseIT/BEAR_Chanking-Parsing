"""Tests of configuration and ownership before launching real processes."""
import json
from types import SimpleNamespace

import pytest
import services


def test_local_overrides_keep_default_ports_and_select_source_tree(tmp_path):
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"ports": {"api": 8112}, "runtime_dir": str(tmp_path)}), encoding="utf-8")
    config = services.configuration(local)
    assert config["ports"]["api"] == 8112
    assert config["ports"]["embedding"] == 7998
    assert services.runtime_paths(config)["runtime"] == tmp_path
    env = services.service_environment(config, services.runtime_paths(config))
    assert env["PYTHONPATH"].split(services.os.pathsep) == [
        str(services.ROOT / "bear-text-beta/src"), str(services.ROOT / "bear-fusion-lab/src"),
    ]


@pytest.mark.parametrize("ports", [{"api": 7998}, {"api": True}, {"api": 0}, {"api": 65536}])
def test_malformed_or_conflicting_port_is_rejected(tmp_path, ports):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"ports": ports}), encoding="utf-8")
    with pytest.raises(ValueError, match="port"):
        services.configuration(config)


def test_ownership_requires_pid_creation_time_and_command(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 900.0, cmdline=lambda: ["actual"])
    monkeypatch.setattr(services.psutil, "Process", lambda pid: process)
    assert services.owned_process({"pid": 12, "created": 900.0, "command": ["actual"]}) is process
    assert services.owned_process({"pid": 12, "created": 880.0, "command": ["actual"]}) is None
    assert services.owned_process({"pid": 12, "created": 900.0, "command": ["foreign"]}) is None


def test_interpreter_preflight_rejects_incompatible_venv(tmp_path, monkeypatch):
    binary = tmp_path / "python.exe"
    binary.touch()
    monkeypatch.setattr(services.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=4, stdout="3.11.12", stderr=""))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        services.require_python(binary, "API", "fastapi")


def test_doctor_checks_model_weights_and_all_profiles(tmp_path, monkeypatch):
    paths = {
        "api_python": tmp_path / "api.exe", "infinity_python": tmp_path / "infinity.exe",
        "qdrant_exe": tmp_path / "qdrant.exe", "model_dir": tmp_path / "model",
        "paddle_python": tmp_path / "paddle.exe", "paddle_models": tmp_path / "paddle-model",
        "docling_python": tmp_path / "docling.exe", "docling_artifacts": tmp_path / "docling-model",
        "paddle_layout_model": tmp_path / "layout", "tesseract_exe": tmp_path / "tesseract.exe",
    }
    models = paths["model_dir"]
    model_manifest = {"model_id": "BAAI/bge-m3", "model_revision": "pinned"}
    for file in (models / "tokenizer.json",
                 paths["docling_artifacts"] / "docling-project--docling-layout-heron/model.safetensors",
                 *(paths["paddle_models"] / name / "inference.json"
                   for name in ("PP-OCRv5_mobile_det", "eslav_PP-OCRv5_mobile_rec")),
                 paths["paddle_layout_model"] / "inference.json"):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    (paths["docling_artifacts"] / "docling-project--docling-models/model_artifacts/tableformer").mkdir(parents=True)
    (models / "bear-model.json").write_text(json.dumps({"embedding_profile": model_manifest}), encoding="utf-8")
    for file in (paths["api_python"], paths["infinity_python"], paths["qdrant_exe"],
                 paths["paddle_python"], paths["docling_python"], paths["tesseract_exe"]):
        file.touch()
    monkeypatch.setattr(services.shutil, "which", lambda name: None)

    def probe(command, *args, **kwargs):
        stdout = ("List of languages:\neng\nrus\n" if command[-1] == "--list-langs" else
                  "test version" if command[-1] == "--version" else "3.12.14")
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(services.subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="weights missing"):
        services.inspect_environment(paths)
    (models / "pytorch_model.bin").write_bytes(b"weights")
    report = services.inspect_environment(paths)
    assert report["optional"]["paddle_layout"]["available"]
    assert report["optional"]["docling"]["python"] == "3.12.14"
    assert report["optional"]["tesseract"]["languages"] == ["eng", "rus"]
    assert report["optional"]["libreoffice"] == {"available": False, "version": None}
    assert report["model"] == model_manifest


def test_launch_records_processes_and_stop_ignores_unowned_pids(tmp_path, monkeypatch):
    config = {**services.configuration(), "runtime_dir": tmp_path}
    events = []
    known = {}
    monkeypatch.setattr(services, "inspect_environment", lambda config: {})
    # Inject a free port reservation without binding a real stand's ports.
    class FreePort:
        def __enter__(self):
            return self

        def __exit__(self, *error):
            return None

        def bind(self, address):
            assert address[0] == "127.0.0.1"

    class Process:
        def __init__(self, command, pid):
            self.command = command
            self.pid = pid

        def poll(self):
            return None

        def create_time(self):
            return 900 + self.pid

        def cmdline(self):
            return self.command

        def children(self, recursive):
            return []

        def terminate(self):
            events.append(("stopped", self.pid))

        def wait(self, timeout):
            events.append(("wait", self.pid))

    def start(command, **kwargs):
        process = Process(command, len(known) + 1)
        known[process.pid] = process
        return process

    monkeypatch.setattr(services.socket, "socket", FreePort)
    monkeypatch.setattr(services.subprocess, "Popen", start)
    monkeypatch.setattr(services.psutil, "Process", known.get)
    monkeypatch.setattr(services, "wait_http", lambda name, port, child, **kwargs: events.append(("ready", name)))
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    output = services.launch(config)
    assert all(entry["running"] for entry in output["services"].values())
    assert {event[1] for event in events} == {"api", "embedding", "qdrant"}
    assert "host: 127.0.0.1" in (tmp_path / "qdrant.yaml").read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="Registry exists"):
        services.launch(config)
    known[2].command = ["foreign command"]
    services.stop(config)
    assert [pid for action, pid in events if action == "stopped"] == [4, 3, 1]
    assert not (tmp_path / "processes.json").exists()
    assert not any(entry["running"] for entry in services.status(config)["services"].values())


def test_foreground_stops_owned_processes_after_failure_or_control_c(tmp_path, monkeypatch):
    config = {"runtime_dir": tmp_path}
    (tmp_path / "processes.json").touch()
    stopped = []
    monkeypatch.setattr(services, "launch", lambda config: {})
    monkeypatch.setattr(services, "status", lambda config: {"services": {"worker": {"running": False}}})
    monkeypatch.setattr(services, "stop", stopped.append)
    monkeypatch.setattr(services.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="worker"):
        services.run_foreground(config)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(services.time, "sleep", interrupt)
    services.run_foreground(config)
    assert stopped == [config, config]
