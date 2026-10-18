"""Runtime job caches and their lock files must survive a read-only image."""

from __future__ import annotations

import ast
import importlib.util
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

import pytest
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
_WRITABLE = (PurePosixPath("/tmp"), PurePosixPath("/state"), PurePosixPath("/outputs"))
_MUTABLE = re.compile(r"cache|lock|temp|tmp", re.IGNORECASE)


def _bad_path(name: str, value: str) -> bool:
    path = PurePosixPath(value)
    return (
        path.is_absolute()
        and bool(_MUTABLE.search(name) or _MUTABLE.search(path.name))
        and not any(path.is_relative_to(root) for root in _WRITABLE)
    )


def _violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    found: list[str] = []
    if path.name.startswith("Dockerfile") or path.suffix == ".sh":
        for line in source.replace("\\\n", " ").splitlines():
            if path.name.startswith("Dockerfile") and not line.startswith("ENV "):
                continue  # Build-time downloads and installed source are image inputs.
            for token in shlex.split(line, comments=True):
                name, _, value = token.partition("=")
                if _bad_path(name, value):
                    found.append(f"{name}={value}")
        return found
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Cover env defaults/assignments, named cache paths, and literal paths
        # passed to file APIs. Ignore imports and docstrings.
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Call)):
            continue
        name = ast.unparse(node).split("=", 1)[0]
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and _bad_path(name, child.value)
            ):
                found.append(f"{child.lineno}: {child.value}")
    return found


def test_job_adapters_use_only_writable_cache_lock_and_temp_paths() -> None:
    paths: set[Path] = set()
    for recipe_path in (ROOT / "recipes").glob("*.json"):
        recipe = RecipeDefinition.model_validate_json(recipe_path.read_text())
        if not recipe.validation.serving.interface.endswith("-job"):
            continue
        build = recipe.execution.build
        paths.add(ROOT / build.dockerfile)
        context = ROOT / build.context.path
        paths.update(context.rglob("*.py"))
        paths.update(context.rglob("*.sh"))
        paths.update(context.rglob("Dockerfile*"))
    violations = {
        str(path.relative_to(ROOT)): bad
        for path in sorted(paths)
        if (bad := _violations(path))
    }
    assert not violations, violations


@pytest.mark.parametrize(
    "root", ["/opt/vonk", "/usr/local", "/app", "/root", "/inputs"]
)
@pytest.mark.parametrize("kind", ["docker", "env", "assignment", "path"])
def test_guard_rejects_image_paths(tmp_path: Path, root: str, kind: str) -> None:
    value = f"{root}/autotune_cache.json"
    snippets = {
        "docker": f'ENV FLEX_GEMM_AUTOTUNE_CACHE_PATH="{value}"\n',
        "env": f'import os\nos.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", "{value}")\n',
        "assignment": f'cache_path = "{value}"\n',
        "path": f'open("{value}.lock", "w")\n',
    }
    path = tmp_path / ("Dockerfile" if kind == "docker" else "job.py")
    path.write_text(snippets[kind])
    assert _violations(path)


@pytest.mark.parametrize("root", ["/tmp", "/state", "/outputs"])
def test_guard_accepts_job_writable_paths(tmp_path: Path, root: str) -> None:
    path = tmp_path / "Dockerfile"
    path.write_text(f'ENV CACHE_PATH="{root}/autotune_cache.json"\n')
    assert not _violations(path)


@pytest.mark.parametrize("override", [False, True])
def test_pixal_sets_writable_cache_before_loading_flexgemm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: bool
) -> None:
    adapter = ROOT / "adapters/three-d/trellis2-native"
    monkeypatch.syspath_prepend(str(adapter))
    spec = importlib.util.spec_from_file_location(
        "pixal_cache_test", adapter / "pixal3d_job.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "one_input", lambda path: tmp_path / "input.png")
    monkeypatch.setattr(module, "validated_input", lambda path: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pixal3d_job.py",
            "--output-mime",
            "model/gltf-binary",
            "--output-dir",
            str(tmp_path),
        ],
    )
    key = "FLEX_GEMM_AUTOTUNE_CACHE_PATH"
    monkeypatch.delenv(key, raising=False)
    if override:
        monkeypatch.setenv(key, str(tmp_path / "custom-cache.json"))

    # Stop exactly where the model stack starts, after the real env defaults run.
    import builtins
    import os

    original_import = builtins.__import__

    def stop_model_import(name: str, *args, **kwargs):
        if name == "numpy":
            raise RuntimeError("model import boundary")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_model_import)
    with pytest.raises(RuntimeError, match="model import boundary"):
        module.main()
    expected = (
        str(tmp_path / "custom-cache.json")
        if override
        else "/tmp/flexgemm-autotune-cache.json"
    )
    assert os.environ[key] == expected
