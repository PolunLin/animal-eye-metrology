#!/usr/bin/env python3
"""Build fresh, pinned Python 3.11 runtimes without changing research environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
POSE_COMMIT = "5408bc76f5b848cf925a0d1857899011d8c5b497"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
INVENTORY_CODE = """
import importlib.metadata as metadata
import json
import platform
import sys
packages = {}
sources = {}
for package in metadata.distributions():
    name = package.metadata['Name']
    packages[name] = package.version
    if name.lower() in {'sam-2', 'mmpose', 'animal-eye-metrology'}:
        direct = package.read_text('direct_url.json')
        if direct:
            sources[name] = json.loads(direct)
print(json.dumps({'python': sys.version, 'platform': platform.platform(),
                  'packages': dict(sorted(packages.items())), 'sources': sources}))
"""


def platform_name(system: str | None = None, machine: str | None = None) -> str:
    """Select a supported build target explicitly, rather than guessing another lock."""
    identity = (system or platform.system(), (machine or platform.machine()).lower())
    if identity[0] == "Linux" and identity[1] in {"x86_64", "amd64"}:
        return "linux-amd64"
    if identity[0] == "Darwin" and identity[1] in {"arm64", "aarch64"}:
        return "macos-arm64"
    raise ValueError(f"Unsupported platform {identity}; use Linux amd64 or macOS arm64.")


def validate_root(root: Path, environments: list[str], *, resume: bool = False) -> None:
    """Refuse overwrites and paths containing protected code, data or legacy runtimes."""
    root = root.resolve()
    protected = (".venv", ".venv-eye-pose", ".git", "data", "outputs", "weights", "src", "app")
    for name in protected:
        path = (PROJECT_ROOT / name).resolve()
        if root == path or root.is_relative_to(path):
            raise ValueError(f"Runtime root must be separate from protected directory: {name}")
    for name in environments:
        target = root / name
        if resume:
            read_resume_record(root, name)
            continue
        if target.exists() or target.is_symlink():
            raise ValueError(f"Refusing to replace existing environment: {target}")
        if (root / f"environment_{name}.json").exists():
            raise ValueError(f"A previous runtime record exists for {name}; choose a new --root.")


def read_resume_record(root: Path, name: str) -> dict[str, Any]:
    """Continue only our incomplete, separate Python 3.11 runtime, without replacing it."""
    target = root / name
    path = root / f"environment_{name}.json"
    if target.is_symlink() or not target.is_dir() or not path.is_file():
        raise ValueError(f"--resume requires the existing runtime and its saved record: {target}")
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("environment") != name
        or record.get("directory") != str(target)
        or record.get("status") not in {"failed", "building"}
    ):
        raise ValueError(f"--resume only accepts an incomplete matching runtime: {target}")
    if record.get("status") == "building" and (pid := record.get("process_id")):
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError(
                f"Runtime bootstrap process {pid} is still active; do not run concurrently."
            )
    pyvenv = target / "pyvenv.cfg"
    if (
        not pyvenv.is_file()
        or not any(
            line.strip().startswith("version = 3.11.") for line in pyvenv.read_text().splitlines()
        )
        or not (target / "bin/python").is_file()
    ):
        raise ValueError(f"Cannot resume an incomplete or non-Python-3.11 venv: {target}")
    return record


def model_source(name: str, commit: str, repository: str) -> dict[str, str]:
    """Record the official immutable VCS source for the selected runtime."""
    return {
        "commit": commit,
        "repository": repository,
        "requirement": f"{name} @ git+{repository}@{commit}",
    }


def build_plan(
    root: Path,
    environments: list[str],
    target: str,
    uv: str | None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Return exact commands; constructing a plan does not write files or use the network."""
    root = root.resolve()
    plans: dict[str, Any] = {}
    for name in environments:
        source = model_source(
            "SAM-2" if name == "app" else "mmpose",
            SAM_COMMIT if name == "app" else POSE_COMMIT,
            "https://github.com/facebookresearch/sam2.git"
            if name == "app"
            else "https://github.com/open-mmlab/mmpose.git",
        )
        environment = root / name
        interpreter = environment / "bin/python"
        lock = PROJECT_ROOT / "requirements" / f"{name}-{target}.txt"
        if uv:
            installer = [uv, "pip", "install", "--python", str(interpreter)]
        else:
            installer = [str(interpreter), "-m", "pip", "install"]
        index = ["--extra-index-url", CPU_INDEX] if target == "linux-amd64" else []
        if uv and target == "linux-amd64":
            # Every requirement is pinned; allow PyPI pins when the CPU index has
            # only an older release of a shared dependency such as NumPy.
            index += ["--index-strategy", "unsafe-best-match"]
        install = installer + index
        commands = [] if resume else [[sys.executable, "-m", "venv", str(environment)]]
        if name == "app":
            buildtools = ["setuptools==80.9.0", "wheel==0.45.1"]
        else:
            buildtools = [
                "pip==24.0",
                "setuptools==69.5.1",
                "wheel==0.48.0",
                "Cython==3.3.1",
                "packaging==24.2",
            ]
        commands.append(install + ["--no-deps", *buildtools])
        if name == "pose":
            commands.append(install + ["--no-deps", "numpy==1.26.4"])
        commands.append(install + ["--no-deps", "--no-build-isolation", "--requirement", str(lock)])
        if name == "pose":
            commands.append(
                install
                + ["--no-deps", "--no-build-isolation", "--no-binary", "mmcv", "mmcv==2.1.0"]
            )
            commands.append(
                install
                + [
                    "--no-deps",
                    "--no-build-isolation",
                    source["requirement"],
                ]
            )
        else:
            commands.append(
                install
                + [
                    "--no-deps",
                    "--no-build-isolation",
                    source["requirement"],
                ]
            )
            commands.append(
                install + ["--no-deps", "--no-build-isolation", "--editable", str(PROJECT_ROOT)]
            )
        plans[name] = {
            "directory": str(environment),
            "lock": str(lock),
            "commands": commands,
            "resume": resume,
            "source": source,
        }
    return {
        "root": str(root),
        "platform": target,
        "installer": "uv" if uv else "pip",
        "plans": plans,
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_locations(cache: Path, root: Path, environments: list[str]) -> dict[str, str]:
    """Honor tool-specific cache paths while keeping legacy runtimes protected."""
    locations = {
        "UV_CACHE_DIR": Path(os.environ.get("UV_CACHE_DIR", cache / "uv")).resolve(),
        "PIP_CACHE_DIR": Path(os.environ.get("PIP_CACHE_DIR", cache / "pip")).resolve(),
    }
    for path in locations.values():
        validate_root(path, [])
        if any(path.is_relative_to(root / name) for name in environments):
            raise ValueError("Package caches must be separate from runtime environments.")
    return {key: str(path) for key, path in locations.items()}


def execute(plan: dict[str, Any], cache: Path) -> None:
    root = Path(plan["root"])
    locations = plan.get("cache") or cache_locations(cache, root, list(plan["plans"]))
    root.mkdir(parents=True, exist_ok=True)
    for path in locations.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    overrides = {
        **locations,
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "120"),
        "PIP_DEFAULT_TIMEOUT": os.environ.get("PIP_DEFAULT_TIMEOUT", "120"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "SAM2_BUILD_CUDA": "0",
        "MMCV_WITH_OPS": "1",
        "MAX_JOBS": "2",
        "FORCE_CUDA": "0",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for name, item in plan["plans"].items():
        runtime_overrides = overrides | {
            "PATH": str(Path(item["directory"]) / "bin") + os.pathsep + os.environ.get("PATH", "")
        }
        environment = os.environ | runtime_overrides
        lock = Path(item["lock"])
        record_path = root / f"environment_{name}.json"
        previous = read_resume_record(root, name) if item.get("resume") else None
        record: dict[str, Any] = {
            "status": "building",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "platform": plan["platform"],
            "installer": plan["installer"],
            "environment": name,
            "process_id": os.getpid(),
            "directory": item["directory"],
            "model_source": item["source"],
            "lock": str(lock),
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "build_environment": runtime_overrides,
            "commands": [],
        }
        if previous:
            history = previous.get("previous_attempts", [])
            record["previous_attempts"] = [
                *history,
                {key: value for key, value in previous.items() if key != "previous_attempts"},
            ]
        write_record(record_path, record)
        try:
            for command in item["commands"]:
                event: dict[str, Any] = {
                    "argv": command,
                    "started_at_utc": datetime.now(UTC).isoformat(),
                }
                record["commands"].append(event)
                write_record(record_path, record)
                print(json.dumps({"environment": name, "command": command}), flush=True)
                process = subprocess.run(command, env=environment, cwd=PROJECT_ROOT, check=False)
                event["returncode"] = process.returncode
                event["finished_at_utc"] = datetime.now(UTC).isoformat()
                write_record(record_path, record)
                if process.returncode:
                    raise RuntimeError(
                        f"{name} bootstrap failed with exit code {process.returncode}."
                    )
            inventory_command = [str(Path(item["directory"]) / "bin/python"), "-c", INVENTORY_CODE]
            inventory = subprocess.run(
                inventory_command,
                env=environment,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            record["inventory"] = json.loads(inventory.stdout)
            record["status"] = "installed_not_runtime_validated"
            record["next_step"] = (
                "Run scripts/check_runtime.py, then --load-models and a real image."
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
            record["status"] = "failed"
            record["error"] = str(error)
            raise
        finally:
            record["finished_at_utc"] = datetime.now(UTC).isoformat()
            write_record(record_path, record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Create both isolated runtimes.")
    selection.add_argument("--app", action="store_true", help="Create the UI/YOLO/SAM runtime.")
    selection.add_argument("--pose", action="store_true", help="Create the HRNet/MMCV runtime.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / ".runtime")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory; defaults to <root>/cache.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without installing.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a stopped incomplete runtime with the same lock, preserving prior attempts.",
    )
    parser.add_argument(
        "--installer",
        choices=("auto", "uv", "pip"),
        default="auto",
        help="Use pip explicitly to reuse an existing pip cache, or prefer available uv.",
    )
    args = parser.parse_args(argv)
    try:
        if sys.version_info[:2] != (3, 11):
            raise ValueError(
                "Run bootstrap with Python 3.11: python3.11 scripts/bootstrap.py --all"
            )
        target = platform_name()
        environments = ["app", "pose"] if args.all else ["app" if args.app else "pose"]
        root = args.root.resolve()
        validate_root(root, environments, resume=args.resume)
        cache = (args.cache_dir or root / "cache").resolve()
        validate_root(cache, [])
        if any(cache.is_relative_to(root / name) for name in environments):
            raise ValueError("The package cache must be separate from runtime environments.")
        uv = shutil.which("uv") if args.installer != "pip" else None
        if args.installer == "uv" and not uv:
            raise ValueError("--installer uv was requested but uv is not installed.")
        plan = build_plan(
            root,
            environments,
            target,
            uv,
            resume=args.resume,
        )
        plan["cache"] = cache_locations(cache, root, environments)
        for item in plan["plans"].values():
            if not Path(item["lock"]).is_file():
                raise ValueError(f"Missing committed lock file: {item['lock']}")
        if args.resume:
            for name, item in plan["plans"].items():
                previous = read_resume_record(root, name)
                digest = hashlib.sha256(Path(item["lock"]).read_bytes()).hexdigest()
                if (
                    previous.get("platform") != target
                    or previous.get("lock") != item["lock"]
                    or previous.get("lock_sha256") != digest
                ):
                    raise ValueError(
                        f"Cannot resume {name}: platform, lock path or lock hash changed."
                    )
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        execute(plan, cache)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
