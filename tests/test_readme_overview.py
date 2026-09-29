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


def test_overview_names_every_family_without_counts() -> None:
    body = overview.render(ROOT)
    assert body.startswith("We cover ")
    assert " recipes**" not in body


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
    assert "[Alice](https://x)" in body and "stranger" in body


def test_readme_splice_replaces_only_the_marked_block() -> None:
    readme = f"before\n{overview.START}\nold\n{overview.END}\nafter\n"
    out = overview.splice(readme, "new\n")
    assert out == f"before\n{overview.START}\nnew\n{overview.END}\nafter\n"
    assert overview.splice(out, "new\n") == out


def _copy_catalog(dest: Path) -> None:
    import shutil

    for name in ("recipes", "models"):
        shutil.copytree(ROOT / name, dest / name)
    shutil.copy(ROOT / "creators.json", dest / "creators.json")


def test_version_bump_does_not_change_overview(tmp_path: Path) -> None:
    _copy_catalog(tmp_path)
    before = overview.render(tmp_path)
    for path in (tmp_path / "recipes").glob("*.json"):
        data = json.loads(path.read_text())
        data["release"]["version"] = "999.0.0"
        data["release"]["released_at"] = "2099-12-31"
        path.write_text(json.dumps(data))
    after = overview.render(tmp_path)
    assert after == before
    assert "999.0.0" not in after and "2099" not in after


def test_check_flags_stale_readme(tmp_path: Path, capsys, monkeypatch) -> None:
    _copy_catalog(tmp_path)
    (tmp_path / "README.md").write_text(f"{overview.START}\nold\n{overview.END}\n")
    monkeypatch.setattr(sys, "argv", ["x", "--root", str(tmp_path), "--check"])
    assert overview.main() == 1
    assert (
        "run tools/build-readme-overview and commit README.md"
        in capsys.readouterr().err
    )
