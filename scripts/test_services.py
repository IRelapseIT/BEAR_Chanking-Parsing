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
