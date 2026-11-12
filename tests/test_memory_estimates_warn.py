"""The recipes are estimates until tried: a memory mismatch warns, a contract fault fails."""

import ast
import json
import runpy
from pathlib import Path

import pytest
from documents import recipe
from pydantic import ValidationError
from vonk_forge_contracts.recipe import RecipeMemoryResources

from qualification.memory_need import (
    EstimateFindingKind,
    MemoryInputs,
    estimate_problems,
)

ROOT = Path(__file__).resolve().parents[1]
LLAMA = "llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single"


def mismatching_root(tmp_path: Path, **memory: int) -> Path:
    """A catalog holding one new recipe whose declared memory understates its need."""
    document = json.loads(recipe(LLAMA).model_dump_json())
    document["identity"]["slug"] = "synthetic-memory-mismatch"
    for role in document["topology"]["roles"]:
        role["resources"]["memory"].update({"peak_bytes": 1_000_000_000, **memory})
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/synthetic-memory-mismatch.json").write_text(
        json.dumps(document)
    )
    (tmp_path / "models").symlink_to(ROOT / "models", target_is_directory=True)
    (tmp_path / "display/recipes").mkdir(parents=True)
    (tmp_path / "display/recipes/synthetic-memory-mismatch.json").write_text(
        json.dumps(
            {
                "kind": "recipe-display",
                "identity": {
                    "publisher": document["identity"]["publisher"],
                    "slug": "synthetic-memory-mismatch",
                },
                "title": "Synthetic",
                "description": "A synthetic recipe.",
                "tags": [],
            }
        )
    )
    (tmp_path / "display/models").symlink_to(
        ROOT / "display/models", target_is_directory=True
    )
    return tmp_path


class PackagingReached(Exception):
    pass


def test_estimate_mismatch_does_not_fail_the_catalog_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["build"]
    monkeypatch.setitem(build.__globals__, "ROOT", mismatching_root(tmp_path))

    def reached_packaging(*args: object, **kwargs: object) -> None:
        raise PackagingReached

    monkeypatch.setitem(build.__globals__, "recipe_package", reached_packaging)
    with pytest.raises(PackagingReached):
        build(
            package_dir=tmp_path / "packages",
            source_commit="0" * 40,
            source_date_epoch=0,
        )
    assert "peak understates need" in capsys.readouterr().err


def test_high_but_valid_declaration_warns_and_fails_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A peak of 125e9 on a copy of a real recipe is a valid contract, not admitted."""
    root = mismatching_root(tmp_path, peak_bytes=125_000_000_000)
    build = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["build"]
    monkeypatch.setitem(build.__globals__, "ROOT", root)

    def reached_packaging(*args: object, **kwargs: object) -> None:
        raise PackagingReached

    monkeypatch.setitem(build.__globals__, "recipe_package", reached_packaging)
    with pytest.raises(PackagingReached):
        build(
            package_dir=tmp_path / "packages",
            source_commit="0" * 40,
            source_date_epoch=0,
        )
    assert "not admissible on an idle Spark" in capsys.readouterr().err
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", root)
    monkeypatch.setattr(
        "sys.argv", ["check-recipe-memory-need", "synthetic-memory-mismatch"]
    )
    assert main() == 0


def test_estimate_mismatch_does_not_fail_check_recipe_memory_need(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", mismatching_root(tmp_path))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(
        "sys.argv", ["check-recipe-memory-need", "synthetic-memory-mismatch"]
    )
    assert main() == 0
    assert "peak understates need" in capsys.readouterr().out
    assert "peak understates need" in summary.read_text()


def run_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: MemoryInputs | None = None,
) -> str:
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setitem(main.__globals__, "ROOT", mismatching_root(tmp_path))
    if inputs is not None:
        monkeypatch.setitem(main.__globals__, "load_inputs", lambda: inputs)
    report = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check-recipe-memory-need",
            "--report",
            str(report),
            "synthetic-memory-mismatch",
        ],
    )
    assert main() == 0
    return report.read_text()


def test_report_leaves_disagreement_advisories_out_of_the_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_report(tmp_path, monkeypatch) == ""


def test_creator_flag_request_is_an_advisory_not_a_missing_input() -> None:
    def rule() -> list[str]:
        raise ValueError("x: declare swa-full-tokens-ratio")

    (finding,) = estimate_problems("x", rule)
    assert finding.kind is EstimateFindingKind.DISAGREEMENT


def test_report_lists_only_the_missing_estimator_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualification.memory_need import load_inputs

    inputs = load_inputs().model_copy(update={"geometries": {}})
    body = run_report(tmp_path, monkeypatch, inputs)
    items = [line for line in body.splitlines() if line.startswith("- ")]
    assert len(items) == 1
    assert "add source-backed cache geometry" in items[0]
    assert "peak understates" not in body


@pytest.mark.parametrize(
    "memory",
    [
        {"peak_bytes": 0, "reserve_bytes": 0},
        {"peak_bytes": 10, "reserve_bytes": 0, "load_peak_bytes": 10},
        {"peak_bytes": 10, "reserve_bytes": 0, "minimum_swap_bytes": 1},
    ],
    ids=["peak-below-one", "load-peak-not-above-peak", "swap-without-load-peak"],
)
def test_contract_shape_violations_still_fail(memory: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RecipeMemoryResources(**memory)


def test_contract_shape_violation_fails_the_catalog_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = runpy.run_path(str(ROOT / "tools/build-catalog-index"))["build"]
    root = mismatching_root(tmp_path, load_peak_bytes=1_000_000_000)
    monkeypatch.setitem(build.__globals__, "ROOT", root)
    with pytest.raises(SystemExit, match="load_peak_bytes must exceed peak_bytes"):
        build(
            package_dir=tmp_path / "packages",
            source_commit="0" * 40,
            source_date_epoch=0,
        )


def test_no_estimate_rule_can_raise_or_gate() -> None:
    """The class guard: an estimate rule returns warnings; only the contract raises."""
    tree = ast.parse((ROOT / "qualification/memory_need.py").read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert not [f.name for f in functions if f.name.startswith("require_")]
    rules = [f for f in functions if f.name.endswith("_warnings")]
    for rule in rules:
        assert not [n for n in ast.walk(rule) if isinstance(n, ast.Raise)], rule.name
    for tool in ("build-catalog-index", "check-recipe-memory-need"):
        assert "require_memory_need" not in (ROOT / "tools" / tool).read_text()
