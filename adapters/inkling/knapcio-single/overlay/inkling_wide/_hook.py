"""Post-import hooks: run a callback right after a module finishes executing.

The overlay must not import sglang (and torch) in every Python process that the
container starts, so patches are attached to their target modules lazily. If the
target module is already imported when the hook is registered, the callback runs
immediately. A callback that raises makes the import of its target fail.
"""

import importlib.abc
import sys
import threading

_lock = threading.RLock()
_callbacks: dict[str, list] = {}
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
        # get_source, get_filename, is_package, ... for inspect / linecache.
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
