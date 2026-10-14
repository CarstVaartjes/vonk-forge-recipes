from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_model_families", str(ROOT / "tools/check-model-families")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
families = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = families
LOADER.exec_module(families)


def _write(root: Path, name: str, publisher: str, family: dict[str, str]) -> None:
    (root / "models").mkdir(exist_ok=True)
    document = {
        "identity": {
            "publisher": publisher,
            "slug": f"{name}-aaaaaaaa",
            "family": family,
            "model": {"publisher": publisher, "slug": name, "title": name},
            "version": "1",
            "variant": "x",
        }
    }
    (root / "models" / f"{name}.json").write_text(json.dumps(document))


def test_the_library_is_consistent() -> None:
    assert families.check(ROOT) == []


def test_republished_checkpoints_share_one_family(tmp_path: Path) -> None:
    lineage = {"publisher": "zai-org", "slug": "glm", "title": "GLM"}
    _write(tmp_path, "glm-5-3-nvfp4", "nvidia", lineage)
    _write(tmp_path, "glm-5-3-exl3", "turboderp", lineage)
    assert families.check(tmp_path) == []


def test_one_slug_cannot_mean_two_families(tmp_path: Path) -> None:
    _write(
        tmp_path, "a", "qwen", {"publisher": "qwen", "slug": "qwen", "title": "Qwen"}
    )
    _write(
        tmp_path,
        "b",
        "nvidia",
        {"publisher": "nvidia", "slug": "qwen", "title": "NVIDIA Qwen"},
    )
    problems = families.check(tmp_path)
    assert len(problems) == 1
    assert "one family slug means one publisher and one title" in problems[0]


def test_family_never_names_the_republisher_or_the_format(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "glm-5-3-flash-exl3",
        "0xsero",
        {"publisher": "0xsero", "slug": "glm-5-3-0xsero", "title": "GLM 5.3 0xSero"},
    )
    _write(
        tmp_path,
        "qwen3-8-27b-exl3",
        "turboderp",
        {
            "publisher": "turboderp",
            "slug": "qwen3-8-exl3",
            "title": "turboderp Qwen3.8 EXL3 quants",
        },
    )
    problems = families.check(tmp_path)
    assert any("ends in the republisher '0xsero'" in problem for problem in problems)
    assert any(
        "family slug 'qwen3-8-exl3' names a weight format" in problem
        for problem in problems
    )
    assert any(
        "family title 'turboderp Qwen3.8 EXL3 quants' names a weight format" in problem
        for problem in problems
    )
