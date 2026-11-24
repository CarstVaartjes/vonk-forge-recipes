"""Post-import hooks: run a callback right after a module finishes executing.

Patching at the import of each target module (instead of importing sglang from
sitecustomize) keeps interpreter start cheap, avoids import cycles, and patches
the module before any code can take references to the stock functions.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import threading
from typing import Callable, Dict

_lock = threading.Lock()
_callbacks: Dict[str, Callable] = {}
_finder = None


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, callback):
        self._inner = inner
        self._callback = callback

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._callback(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        with _lock:
            callback = _callbacks.get(fullname)
        if callback is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        with _lock:
            _callbacks.pop(fullname, None)
        spec.loader = _Loader(spec.loader, callback)
        return spec


def when_imported(fullname: str, callback: Callable) -> None:
    """Run callback(module) now if imported, else right after its import."""
    global _finder
    module = sys.modules.get(fullname)
    if module is not None:
        callback(module)
        return
    with _lock:
        _callbacks[fullname] = callback
        if _finder is None:
            _finder = _Finder()
            sys.meta_path.insert(0, _finder)


def pending() -> list:
    with _lock:
        return sorted(_callbacks)
