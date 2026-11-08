"""Runtime planning, protected-path safety and diagnostic failure semantics; no installs."""

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest


def load_script(name: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"aem_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load_script("bootstrap")
checker = load_script("check_runtime")


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [("Linux", "x86_64", "linux-amd64"), ("Darwin", "arm64", "macos-arm64")],
)
def test_platform_lock_selection(system: str, machine: str, expected: str) -> None:
    assert bootstrap.platform_name(system, machine) == expected
    with pytest.raises(ValueError, match="Unsupported platform"):
        bootstrap.platform_name("Linux", "aarch64")


@pytest.mark.parametrize("name", [".venv", ".venv-eye-pose", "outputs", "data", "weights"])
def test_bootstrap_refuses_protected_roots(tmp_path: Path, monkeypatch, name: str) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="protected directory"):
        bootstrap.validate_root(tmp_path / name / "nested", ["app", "pose"])
    assert not list(tmp_path.iterdir())


def test_bootstrap_refuses_existing_or_linked_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root = tmp_path / ".runtime"
    (root / "pose").mkdir(parents=True)
    with pytest.raises(ValueError, match="existing environment"):
        bootstrap.validate_root(root, ["app", "pose"])
    assert not (root / "app").exists()
    legacy = tmp_path / ".venv"
    legacy.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="protected directory"):
        bootstrap.validate_root(linked, ["app"])


def test_plan_keeps_cpu_sources_build_order_and_interpreters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root = tmp_path / ".runtime"
    plan = bootstrap.build_plan(root, ["app", "pose"], "linux-amd64", "/tools/uv")
    app_commands = plan["plans"]["app"]["commands"]
    pose_commands = plan["plans"]["pose"]["commands"]
    assert app_commands[0][-1] == str(root / "app")
    assert all(
        command[:5] == ["/tools/uv", "pip", "install", "--python", str(root / "app/bin/python")]
        for command in app_commands[1:]
    )
    assert all(bootstrap.CPU_INDEX in command for command in app_commands[1:])
    assert any(bootstrap.SAM_COMMIT in argument for argument in app_commands[-2])
    assert "numpy==1.26.4" in pose_commands[2]
    assert "--requirement" in pose_commands[3]
    assert "mmcv==2.1.0" in pose_commands[4] and "--no-build-isolation" in pose_commands[4]
    assert any(bootstrap.POSE_COMMIT in argument for argument in pose_commands[5])
    assert not root.exists()


def test_pip_fallback_targets_only_new_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root = tmp_path / ".runtime"
    plan = bootstrap.build_plan(root, ["app"], "macos-arm64", None)
    for command in plan["plans"]["app"]["commands"][1:]:
        assert command[:4] == [str(root / "app/bin/python"), "-m", "pip", "install"]
        assert "--extra-index-url" not in command


def test_dry_run_has_no_environment_or_manifest_writes(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "platform_name", lambda: "macos-arm64")
    requirements = tmp_path / "requirements"
    requirements.mkdir()
    for name in ("app", "pose"):
        (requirements / f"{name}-macos-arm64.txt").write_text("numpy==1.26.4\n")
    root = tmp_path / ".runtime"
    assert bootstrap.main(["--all", "--root", str(root), "--dry-run"]) == 0
    assert set(json.loads(capsys.readouterr().out)["plans"]) == {"app", "pose"}
    assert not root.exists()


def test_failed_install_preserves_exact_command_and_failure_record(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root = tmp_path / ".runtime"
    lock = tmp_path / "requirements/app-macos-arm64.txt"
    lock.parent.mkdir()
    lock.write_text("numpy==2.2.6\n")
    plan = bootstrap.build_plan(root, ["app"], "macos-arm64", None)
    seen = []

    def failed(command, **kwargs):
        seen.append((command, kwargs))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(bootstrap.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="exit code 7"):
        bootstrap.execute(plan, root / "cache")
    record = json.loads((root / "environment_app.json").read_text())
    assert record["status"] == "failed" and record["commands"][0]["returncode"] == 7
    assert record["commands"][0]["argv"] == seen[0][0]
    assert seen[0][1]["env"]["SAM2_BUILD_CUDA"] == "0"
    assert seen[0][1]["env"]["MMCV_WITH_OPS"] == "1"
    assert seen[0][1]["env"]["MAX_JOBS"] == "2"


def test_check_keeps_interpreter_symlink_and_reports_missing_runtime(tmp_path: Path) -> None:
    executable = tmp_path / "base-python"
    executable.write_text("synthetic executable")
    interpreter = tmp_path / "app/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(executable)
    config = tmp_path / "configs/inference.toml"
    assert checker.configured_path(config, {"python": "../app/bin/python"}, "python") == interpreter
    missing = tmp_path / "missing-python"
    cache = tmp_path / "cache"
    result = checker.run_probe("app", missing, config, "", [], cache, False, 1)
    assert result["status"] == "failed" and "Missing executable" in result["error"]
    assert not cache.exists()


def test_probe_programs_compile_without_importing_optional_models() -> None:
    compile(checker.APP_PROBE, "app_probe", "exec")
    compile(checker.POSE_PROBE, "pose_probe", "exec")


def incomplete_runtime(tmp_path: Path, status: str = "failed") -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / ".runtime"
    environment = root / "pose"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("synthetic python executable")
    (environment / "pyvenv.cfg").write_text("version = 3.11.14\n")
    lock = tmp_path / "requirements/pose-macos-arm64.txt"
    lock.parent.mkdir()
    lock.write_text("numpy==1.26.4\n")
    import hashlib

    record = {
        "status": status,
        "environment": "pose",
        "directory": str(environment),
        "platform": "macos-arm64",
        "lock": str(lock),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "commands": [{"argv": ["previous", "attempt"], "returncode": 7}],
    }
    (root / "environment_pose.json").write_text(json.dumps(record))
    return root, lock, record


def test_resume_keeps_failed_attempt_and_never_recreates_venv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root, _, original = incomplete_runtime(tmp_path)
    bootstrap.validate_root(root, ["pose"], resume=True)
    plan = bootstrap.build_plan(root, ["pose"], "macos-arm64", None, resume=True)
    assert not any("venv" in command for command in plan["plans"]["pose"]["commands"])
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9),
    )
    with pytest.raises(RuntimeError, match="exit code 9"):
        bootstrap.execute(plan, root / "cache")
    new = json.loads((root / "environment_pose.json").read_text())
    assert new["status"] == "failed"
    assert new["previous_attempts"] == [original]
    assert new["commands"][0]["returncode"] == 9


def test_resume_refuses_changed_lock_before_install(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "platform_name", lambda: "macos-arm64")
    root, lock, _ = incomplete_runtime(tmp_path)
    lock.write_text("numpy==1.26.3\n")
    assert bootstrap.main(["--pose", "--root", str(root), "--resume", "--dry-run"]) == 1
    assert "lock hash changed" in capsys.readouterr().err


def test_resume_refuses_completed_or_running_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    root, _, record = incomplete_runtime(tmp_path, "installed_not_runtime_validated")
    with pytest.raises(ValueError, match="incomplete matching runtime"):
        bootstrap.validate_root(root, ["pose"], resume=True)
    import os

    record.update(status="building", process_id=os.getpid())
    (root / "environment_pose.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="still active"):
        bootstrap.validate_root(root, ["pose"], resume=True)


def test_cache_overrides_are_explicit_and_keep_legacy_paths_protected(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    pip_cache = tmp_path / "existing-pip-cache"
    uv_cache = tmp_path / "existing-uv-cache"
    monkeypatch.setenv("PIP_CACHE_DIR", str(pip_cache))
    monkeypatch.setenv("UV_CACHE_DIR", str(uv_cache))
    locations = bootstrap.cache_locations(
        tmp_path / "default-cache", tmp_path / ".runtime", ["pose"]
    )
    assert locations == {"UV_CACHE_DIR": str(uv_cache), "PIP_CACHE_DIR": str(pip_cache)}
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / ".venv" / "cache"))
    with pytest.raises(ValueError, match="protected directory"):
        bootstrap.cache_locations(tmp_path / "default-cache", tmp_path / ".runtime", ["pose"])
    assert not list(tmp_path.iterdir())
