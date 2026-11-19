# Fail-closed flashinfer stub. Lives in /opt/flashinfer-stub, first on PYTHONPATH in the image.
# No real flashinfer code is imported or executed. Every attribute and every submodule resolves to a
# plain function that raises RuntimeError when CALLED, so SGLang's module-level imports succeed.
# Names in _ABSENT raise AttributeError (-> ImportError on `from flashinfer import X`), so SGLang's
# optional-import probes choose their torch/triton fallback instead of a flashinfer kernel.
import importlib.abc
import importlib.machinery
import os
import sys

__version__ = "0.0.0+disabled-stub"
__path__ = []  # the real package directory is never searched for submodules
STUB_DIR = os.path.dirname(os.path.abspath(__file__))

# Names SGLang probes inside try/except ImportError (only these are absent; everything else raises when called).
_ABSENT = frozenset({
    "top_k",                   # layers/dflash -> _flashinfer_top_k=None -> torch.topk
    "mm_fp4",                  # quantization/modelopt_quant try-block -> enable_flashinfer_fp4_gemm=False
    "fp4_quantize",            # quantization/fp4_utils try-block -> fp4_quantize=None
    "nvfp4_quantize",
    "mm_bf16_fp4",
    "prepare_bf16_fp4_weights",
})
_MSG = "flashinfer is disabled in this image (fail-closed stub): {}"


def _disabled(qualname):
    def _raise(*args, **kwargs):
        raise RuntimeError(_MSG.format(qualname))
    _raise.__name__ = qualname.rsplit(".", 1)[-1]
    _raise.__qualname__ = qualname
    _raise.__module__ = qualname.rpartition(".")[0] or "flashinfer"
    return _raise


def _module_getattr(modname, is_top):
    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if is_top and name in _ABSENT:
            raise AttributeError(f"module 'flashinfer' has no attribute {name!r} (fail-closed stub)")
        obj = _disabled(f"{modname}.{name}")
        setattr(sys.modules[modname], name, obj)  # cache so identity is stable
        return obj
    return __getattr__


__getattr__ = _module_getattr("flashinfer", True)


class _StubSubmoduleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves every flashinfer.<sub> import as an empty stub module (never a real file)."""

    def find_spec(self, fullname, path, target=None):
        # flashinfer.norm must NOT import. sgl_kernel/elementwise.py and sglang layernorm.py
        # treat a successful import as "flashinfer available" and then call the stubbed norm kernel
        # (RuntimeError at first decode forward). An ImportError here sends them to the native
        # torch.ops.sgl_kernel.rmsnorm / sglang forward_native fallbacks. No flashinfer code runs.
        if fullname == "flashinfer.norm" or fullname.startswith("flashinfer.norm."):
            raise ImportError(f"{fullname} disabled by the flashinfer stub (use native rmsnorm)")
        if fullname.startswith("flashinfer.") and fullname.split(".")[1] in _ABSENT:
            # Absent names must not become submodules, or `from flashinfer import top_k` gets an empty module.
            # Returning None -> ModuleNotFoundError -> `from flashinfer import top_k` raises ImportError.
            return None
        if fullname.startswith("flashinfer."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None  # default module object

    def exec_module(self, module):
        module.__path__ = []
        module.__getattr__ = _module_getattr(module.__name__, False)


if not any(type(f).__name__ == "_StubSubmoduleFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _StubSubmoduleFinder())
