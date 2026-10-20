"""Run the shipped TabbyAPI vision load call against a deterministic loader fault."""

from __future__ import annotations

import ast
import runpy
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/minicpm/vcruz305-exl3-single"


def _source(archive: str, suffix: str) -> str:
    with tarfile.open(ADAPTER / "vendor" / archive) as bundle:
        member = next(
            item for item in bundle.getmembers() if item.name.endswith(suffix)
        )
        stream = bundle.extractfile(member)
        assert stream is not None
        return stream.read().decode()


def _load_method(source: str, name: str, namespace: dict):
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - execute methods from the shipped, pinned source archives
        compile(ast.fix_missing_locations(module), "<vendored-loader>", "exec"),
        namespace,
    )
    return namespace[name]


class VisionLoader:
    """Fail at the reported boundary if Tabby selects the text measuring path.

    The actual pinned Model.load_gen below chooses _load_single or _load_autosplit.
    Only weight I/O, CUDA operations and the failing measuring forward are doubles.
    """

    def __init__(self, device_count: int):
        self.component = "vision"
        self.config = SimpleNamespace(infer_params=SimpleNamespace())
        self.modules = [SimpleNamespace(device="cuda:0")]
        self.cache_weakrefs = {}
        self.single = Mock()
        self.split = Mock()
        self.cuda = SimpleNamespace(
            device_count=lambda: device_count, memory_allocated=lambda _: 0
        )
        self.load_gen = _load_method(
            _source("exllamav3.tar.gz", "/exllamav3/model/model.py"),
            "load_gen",
            {
                "torch": SimpleNamespace(cuda=self.cuda),
                "gc": SimpleNamespace(get_freeze_count=lambda: 1),
                "free_mem": lambda: None,
                "g_tensor_cache": SimpleNamespace(drop_all=lambda: None),
            },
        ).__get__(self)

    def _load_single(self, *args) -> None:
        self.single(*args)

    def _load_autosplit(self, *args) -> Iterator[tuple[int, int]]:
        self.split(*args)
        raise RuntimeError("conv2d received text-shaped input [1, 2048]")


@pytest.mark.parametrize("patched", [False, True])
def test_single_gpu_vision_uses_native_loader(tmp_path: Path, patched: bool) -> None:
    source = _source("tabby.tar.gz", "/backends/exllamav3/model.py")
    if patched:
        target = tmp_path / "backends/exllamav3/model.py"
        target.parent.mkdir(parents=True)
        target.write_text(source)
        subprocess.run(
            [sys.executable, str(ADAPTER / "patch-tabby-vision.py"), str(tmp_path)],
            check=True,
        )
        source = target.read_text()

    vision = VisionLoader(device_count=1)
    text = Mock()
    text.load_gen.return_value = iter([(1, 2)])
    draft = Mock()
    draft.load_gen.return_value = iter([(1, 1)])
    backend = SimpleNamespace(
        use_vision=True,
        vision_model=vision,
        use_draft_model=True,
        draft_model=draft,
        draft_gpu_split=None,
        gpu_split=None,
        autosplit_reserve=[96 / 1024],
        use_tp=False,
        tp_backend="native",
        model_dir="/models/target",
        model=text,
        chunk_size=512,
        max_batch_size=1,
    )
    load = _load_method(
        source,
        "load_model_sync",
        {"torch": SimpleNamespace(cuda=vision.cuda, device=str), "xlogger": Mock()},
    )
    callback = Mock()
    if not patched:
        with pytest.raises(RuntimeError, match=r"conv2d.*\[1, 2048\]"):
            list(load(backend, callback))
        vision.split.assert_called_once()
        vision.single.assert_not_called()
        text.load_gen.assert_not_called()
        return

    assert list(load(backend, callback)) == [(1, 1), (1, 2)]
    vision.single.assert_called_once()
    assert vision.single.call_args.args[1] == "cuda:0"
    vision.split.assert_not_called()
    draft.load_gen.assert_called_once_with(
        reserve_per_device=backend.autosplit_reserve,
        use_per_device=None,
        callback=callback,
    )
    text.load_gen.assert_called_once_with(
        tensor_p=False,
        tp_backend="native",
        reserve_per_device=backend.autosplit_reserve,
        use_per_device=None,
        callback=callback,
        max_chunk_size=512,
        max_batch_size=1,
    )


def test_build_applies_patch_and_rejects_source_drift() -> None:
    patch_text = runpy.run_path(str(ADAPTER / "patch-tabby-vision.py"))["patch_text"]
    with pytest.raises(ValueError, match="Pinned TabbyAPI"):
        patch_text("unknown upstream revision")
    source = _source("tabby.tar.gz", "/backends/exllamav3/model.py")
    with pytest.raises(ValueError, match="Pinned TabbyAPI"):
        patch_text(patch_text(source))
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "COPY patch-tabby-vision.py /tmp/patch-tabby-vision.py" in dockerfile
    assert dockerfile.index(
        "python /tmp/patch-tabby-vision.py /opt/tabbyAPI"
    ) < dockerfile.index("--no-build-isolation /opt/tabbyAPI")
