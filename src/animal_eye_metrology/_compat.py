"""Lazy aliases for historical imports and fresh ``python -m`` commands.

New code uses responsibility packages. Old names share the canonical module object,
so exception identities, monkeypatches and execution locks stay consistent.
"""

import importlib
import importlib.abc
import importlib.util
import sys
from types import CodeType, ModuleType
from typing import Any

from animal_eye_metrology.paths import MODULE_LOCATIONS, module_source


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, name: str, target: str) -> None:
        self.name = name
        self.target = target

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        sys.modules[self.name] = importlib.import_module(self.target)

    def get_filename(self, fullname: str) -> str:
        return str(module_source(self.name.rsplit(".", 1)[1]))

    def get_code(self, fullname: str) -> CodeType:
        # Do not import/execute the target before runpy executes it as __main__.
        source = module_source(self.name.rsplit(".", 1)[1])
        return compile(source.read_bytes(), str(source), "exec")

    def is_package(self, fullname: str) -> bool:
        return False


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> Any:
        prefix = "animal_eye_metrology."
        if not fullname.startswith(prefix):
            return None
        name = fullname.removeprefix(prefix)
        location = MODULE_LOCATIONS.get(name)
        if location is None:
            return None
        canonical = prefix + location.removesuffix(".py").replace("/", ".")
        loader = _LegacyLoader(fullname, canonical)
        return importlib.util.spec_from_loader(
            fullname, loader, origin=str(module_source(name)), is_package=False
        )


def install_legacy_imports() -> None:
    """Install one bounded, standard-library-only finder; no application modules load."""
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
