"""Placement uses the existing fleet installation evidence, without pinning lanes."""

from dataclasses import replace
from pathlib import Path

import pytest
from sweep_fakes import make_sweep
from test_sweep_policy import CAP, RESERVE, alias, recipe

from spark_sweep import policy
from spark_sweep.catalog import Recipe, parse_fleet


def observed_bins(
    tmp_path: Path, catalog: list[Recipe], installed: list[list[str]]
) -> list[policy.SparkBin]:
    sweep, fleet, _ = make_sweep(tmp_path, [], [])
    sweep.recipes = {r.key: r for r in catalog}
    document = fleet._fleet()
    for node, revisions in zip(document["nodes"], installed, strict=True):
        node["installed"] = [
            {
                "recipe_revision_id": revision,
                "complete": True,
                "expected_rank_count": 1,
            }
            for revision in revisions
        ]
    sweep.fleet = parse_fleet(document, CAP)
    before = len(fleet.calls)
    bins = sweep._bins()
    assert len(fleet.calls) == before  # locality needs no further Controller read
    return bins


def test_variant_uses_the_spark_with_all_its_models(tmp_path: Path) -> None:
    installed = recipe("installed", ("a", "b"), revision_id="installed")
    variant = recipe("variant", ("a", "b"), revision_id="variant")
    bins = observed_bins(tmp_path, [installed, variant], [[], ["installed"]])
    placed = policy.place_singles([variant], bins, RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("spk_b",)]


def test_partial_model_set_loses_to_the_complete_set(tmp_path: Path) -> None:
    partial = recipe("partial", ("a",), revision_id="partial")
    complete = recipe("complete", ("a", "b"), revision_id="complete")
    variant = recipe("variant", ("a", "b"), revision_id="variant")
    bins = observed_bins(
        tmp_path, [partial, complete, variant], [["partial"], ["complete"]]
    )
    placed = policy.place_singles([variant], bins, RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("spk_b",)]


def test_exact_installed_runtime_breaks_a_model_locality_tie(tmp_path: Path) -> None:
    sibling = recipe("sibling", revision_id="sibling")
    target = recipe("target", revision_id="target")
    bins = observed_bins(tmp_path, [sibling, target], [["sibling"], ["target"]])
    placed = policy.place_singles([target], bins, RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("spk_b",)]


def test_model_locality_outweighs_image_hint(tmp_path: Path) -> None:
    target = recipe("target", revision_id="target")
    bins = observed_bins(tmp_path, [target], [[], ["target"]])
    # Model and image hints can disagree after an observation changes.
    bins[0] = replace(bins[0], installed_revisions=frozenset({"target"}))
    placed = policy.place_singles([target], bins, RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("spk_b",)]


@pytest.mark.parametrize("next_model", ["other", "download"])
def test_other_lane_takes_next_local_model_or_download(
    tmp_path: Path, next_model: str
) -> None:
    head = recipe("head", ("a",), revision_id="head")
    variant = recipe("variant", ("a",), revision_id="variant")
    other = recipe("other", ("other",), revision_id="other")
    next_recipe = recipe("next", (next_model,), revision_id="next")
    bins = observed_bins(
        tmp_path, [head, variant, other, next_recipe], [["other"], ["head"]]
    )
    placed = policy.place_singles([head, variant, next_recipe], bins, RESERVE, alias)
    assert [(p.recipe.slug, p.spark_ids) for p in placed] == [
        ("head", ("spk_b",)),
        ("next", ("spk_a",)),
    ]
    # On the next round the skipped variant is head, even if its owner stays busy.
    bins[1] = replace(bins[1], aliases=frozenset({"head"}))
    placed = policy.place_singles([variant, next_recipe], bins, RESERVE, alias)
    assert [(p.recipe.slug, p.spark_ids) for p in placed] == [("variant", ("spk_a",))]


def test_no_idle_lane_when_only_variants_remain(tmp_path: Path) -> None:
    head = recipe("head", revision_id="head")
    variant = recipe("variant", revision_id="variant")
    bins = observed_bins(tmp_path, [head, variant], [[], ["head"]])
    placed = policy.place_singles([head, variant], bins, RESERVE, alias)
    assert [(p.recipe.slug, p.spark_ids) for p in placed] == [
        ("head", ("spk_b",)),
        ("variant", ("spk_a",)),
    ]


def test_unknown_inventory_preserves_queue_and_id_order(tmp_path: Path) -> None:
    candidates = [recipe(slug, revision_id=slug) for slug in ("head", "next")]
    bins = observed_bins(tmp_path, candidates, [[], ["unknown-revision"]])
    placed = policy.place_singles(candidates, list(reversed(bins)), RESERVE, alias)
    assert [(p.recipe.slug, p.spark_ids) for p in placed] == [
        ("head", ("spk_a",)),
        ("next", ("spk_b",)),
    ]


@pytest.mark.parametrize("complete,width", [(False, 1), (None, 1), (True, 2)])
def test_incomplete_or_dual_installations_do_not_claim_a_whole_model(
    tmp_path: Path, complete: bool | None, width: int
) -> None:
    target = recipe("target", revision_id="target")
    sweep, fleet, _ = make_sweep(tmp_path, [], [])
    sweep.recipes = {target.key: target}
    document = fleet._fleet()
    document["nodes"][1]["installed"] = [
        {
            "recipe_revision_id": "target",
            "complete": complete,
            "expected_rank_count": width,
        }
    ]
    sweep.fleet = parse_fleet(document, CAP)
    placed = policy.place_singles([target], sweep._bins(), RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("spk_a",)]


def test_affinity_never_overrides_fit_or_ports(tmp_path: Path) -> None:
    target = recipe("target", revision_id="target")
    bins = observed_bins(tmp_path, [target], [[], ["target"]])
    for blocked in (
        replace(bins[1], capacity=target.memory_bytes),
        replace(bins[1], ports=frozenset(target.ports)),
    ):
        placed = policy.place_singles([target], [bins[0], blocked], RESERVE, alias)
        assert [p.spark_ids for p in placed] == [("spk_a",)]
