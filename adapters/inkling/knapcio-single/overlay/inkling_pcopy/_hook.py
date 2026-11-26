"""Post-import hooks: run a callback right after a module finishes executing.

Patches attach lazily so that processes which never import the model pay nothing
and never import sglang or torch. If the target module is already imported, the
callback runs at once. A callback that raises makes the import of its target fail.
Several overlays may each install a finder of this kind; they nest (each wraps the
loader the next one returned), so every package's callbacks run.
"""

import importlib.abc
import sys
import threading

_lock = threading.RLock()
_callbacks: dict = {}
_finder = None


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        _run(self._name, module)

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path, target=None):
        if fullname not in _callbacks or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, fullname)
        return spec


def _run(name, module):
    with _lock:
        callbacks = _callbacks.pop(name, [])
    for callback in callbacks:
        callback(module)


def when_imported(name, callback):
    """Call callback(module) once module `name` has been imported."""
    global _finder
    with _lock:
        module = sys.modules.get(name)
        if module is None:
            _callbacks.setdefault(name, []).append(callback)
            if _finder is None:
                _finder = _Finder()
                sys.meta_path.insert(0, _finder)
            return
    callback(module)


def pending():
    with _lock:
        return sorted(_callbacks)
