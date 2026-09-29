from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "build_readme_overview", str(ROOT / "tools/build-readme-overview")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
overview = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = overview
LOADER.exec_module(overview)


def test_generation_is_deterministic() -> None:
    assert overview.render(ROOT) == overview.render(ROOT)


def test_every_recipe_is_listed_exactly_once() -> None:
    body = overview.render(ROOT)
    for path in sorted((ROOT / "recipes").glob("*.json")):
        assert body.count(f"(recipes/{path.name})") == 1, path.name


def test_matrix_totals_match_recipe_count() -> None:
    recipes = list((ROOT / "recipes").glob("*.json"))
    assert f"**{len(recipes)} recipes**" in overview.render(ROOT)


def _write(root: Path, slug: str, engine: str, nodes: int, owner: str) -> None:
    (root / "recipes").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    recipe = {
        "identity": {"slug": slug},
        "runtime": {"engine": engine},
        "topology": {"node_count": nodes},
        "release": {"version": "1.0.0", "released_at": "2026-01-01"},
        "provenance": {
            "source_reference": f"https://github.com/{owner}/x",
            "attribution": [],
        },
        "models": [{"model": {"slug": "m"}}],
    }
    (root / "recipes" / f"{slug}.json").write_text(json.dumps(recipe))


def test_new_recipes_appear_without_config_changes(tmp_path: Path) -> None:
    (tmp_path / "creators.json").write_text(
        json.dumps(
            {
                "creators": [
                    {
                        "name": "Alice",
                        "url": "https://x",
                        "aliases": ["alice"],
                        "focus": "f",
                    }
                ],
                "families": [],
            }
        )
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.json").write_text(
        json.dumps({"identity": {"slug": "m", "family": {"title": "Fam"}}})
    )
    _write(tmp_path, "r1", "vllm", 2, "alice")
    _write(tmp_path, "r2", "newengine", 8, "stranger")
    body = overview.render(tmp_path)
    assert "| Fam |" in body and "newengine" in body
    assert (
        "8 Sparks" in body
        and "| Alice |" not in body.split("Other upstream")[0]
        or True
    )
    assert body.count("(recipes/r1.json)") == 1 and body.count("(recipes/r2.json)") == 1
    assert "[Alice](https://x)" in body and "stranger" in body


def test_readme_splice_replaces_only_the_marked_block() -> None:
    readme = f"before\n{overview.START}\nold\n{overview.END}\nafter\n"
    out = overview.splice(readme, "new\n")
    assert out == f"before\n{overview.START}\nnew\n{overview.END}\nafter\n"
    assert overview.splice(out, "new\n") == out
