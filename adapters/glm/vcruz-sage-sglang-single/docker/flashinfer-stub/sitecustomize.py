# Runs at interpreter start for every python process with /opt/flashinfer-stub on PYTHONPATH.
# Pins `flashinfer` to the fail-closed stub and never lets the real package load.
import importlib.abc
import importlib.machinery
import os
import sys

_ROOT = "/opt/flashinfer-stub"
_STUB = _ROOT + "/flashinfer"


class _PinFlashinfer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "flashinfer":
            spec = importlib.machinery.PathFinder.find_spec("flashinfer", [_ROOT])
            if spec is None:
                raise ImportError("flashinfer stub missing at " + _STUB)
            return spec
        return None


if not any(type(f).__name__ == "_PinFlashinfer" for f in sys.meta_path):
    sys.meta_path.insert(0, _PinFlashinfer())

import flashinfer as _fi  # noqa: E402  (executes the stub __init__ only)

if not str(getattr(_fi, "__file__", "")).startswith(_STUB + "/"):
    raise ImportError("flashinfer resolved outside the stub: %r" % (getattr(_fi, "__file__", None),))
print("[flashinfer-stub] pinned: %s pid=%d" % (_fi.__file__, os.getpid()), file=sys.stderr, flush=True)
