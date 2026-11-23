from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_model_families", str(ROOT / "tools/check-model-families")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
families = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = families
LOADER.exec_module(families)

EXAMPLES = ROOT / "contracts/src/vonk_forge_contracts/examples"


def _example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _lineage(**update: str) -> dict[str, str]:
    return {
        "family": "glm",
        "line": "glm-flash",
        "model": "glm-5-3-flash",
        "variant_kind": "quantization",
        **update,
    }


def _family(
    title: str = "GLM", line_title: str = "GLM Flash", slug: str = "glm"
) -> dict[str, Any]:
    return {
        "kind": "family-display",
        "slug": slug,
        "title": title,
        "lines": [
            {
                "slug": "glm-flash",
                "title": line_title,
                "models": [{"slug": "glm-5-3-flash", "title": "GLM 5.3 Flash"}],
            }
        ],
    }


def _library(
    tmp_path: Path,
    lineages: dict[str, dict[str, str] | None],
    family: dict[str, Any] | None = None,
) -> Path:
    """One recipe serving the example Model first, plus a display per Model."""

    for folder in ("display/models", "display/families", "recipes"):
        (tmp_path / folder).mkdir(parents=True)
    recipe = _example("recipe-source-build.json")
    (tmp_path / "recipes" / "demo.json").write_text(json.dumps(recipe))
    for slug, lineage in lineages.items():
        display = _example("model-display.json")
        display.pop("lineage")
        display["identity"]["slug"] = slug
        if lineage is not None:
            display["lineage"] = lineage
        (tmp_path / "display/models" / f"{slug}.json").write_text(json.dumps(display))
    family = family or _family()
    (tmp_path / "display/families" / f"{family['slug']}.json").write_text(
        json.dumps(family)
    )
    return tmp_path


PRIMARY = _example("recipe-source-build.json")["models"][0]["model"]["slug"]


def test_the_library_is_consistent() -> None:
    assert families.check(ROOT) == []


def test_republished_checkpoints_share_one_family(tmp_path: Path) -> None:
    root = _library(tmp_path, {PRIMARY: _lineage(), "glm-5-3-flash-exl3": _lineage()})
    assert families.check(root) == []


def test_a_lineage_outside_the_families_is_refused(tmp_path: Path) -> None:
    root = _library(tmp_path, {PRIMARY: _lineage(line="glm-other")})
    assert len(families.check(root)) == 1


def test_a_served_model_needs_a_lineage_and_a_helper_does_not(tmp_path: Path) -> None:
    root = _library(tmp_path, {PRIMARY: None, "helper": _lineage()})
    (problem,) = families.check(root)
    assert PRIMARY in problem
    helper_only = _library(tmp_path / "ok", {PRIMARY: _lineage(), "helper": None})
    assert families.check(helper_only) == []


def test_a_listed_release_no_model_names_is_refused(tmp_path: Path) -> None:
    family = _family()
    family["lines"][0]["models"].append({"slug": "glm-6-flash", "title": "GLM 6"})
    root = _library(tmp_path, {PRIMARY: _lineage()}, family)
    (problem,) = families.check(root)
    assert "glm-6-flash" in problem


def test_a_family_or_line_never_names_a_weight_format(tmp_path: Path) -> None:
    root = _library(
        tmp_path,
        {PRIMARY: _lineage(family="glm-exl3")},
        _family(title="GLM EXL3 quants", line_title="GLM Flash NVFP4", slug="glm-exl3"),
    )
    # The family slug, the family title and the line title.
    assert len(families.check(root)) == 3
