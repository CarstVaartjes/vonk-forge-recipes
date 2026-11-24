"""Post-import hooks: run a callback on a module right after it executes.

Patching at import time (instead of importing sglang from sitecustomize) keeps the
interpreter start cheap and works in every spawned scheduler process. A callback that
raises makes the triggering ``import`` fail, so a missing or changed patch target
stops the server at startup.
"""

from __future__ import annotations

import importlib.abc
import sys
from typing import Callable, Dict

Hook = Callable[[object], None]


class _HookedLoader(importlib.abc.Loader):
    def __init__(self, loader, hook: Hook) -> None:
        self._loader = loader
        self._hook = hook

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        self._hook(module)

    def __getattr__(self, name):
        # get_source / get_code / is_package / get_filename ... (inspect, linecache).
        return getattr(self._loader, name)


class PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks: Dict[str, Hook]) -> None:
        self.hooks = dict(hooks)
        self._busy: set[str] = set()

    def find_spec(self, fullname, path, target=None):
        hook = self.hooks.get(fullname)
        if hook is None or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _HookedLoader(spec.loader, hook)
        return spec


def install(hooks: Dict[str, Hook]) -> PostImportFinder:
    """Install the finder first on sys.meta_path; modules already imported are
    patched immediately."""
    finder = PostImportFinder(hooks)
    sys.meta_path.insert(0, finder)
    for name, hook in hooks.items():
        module = sys.modules.get(name)
        if module is not None:
            hook(module)
    return finder
