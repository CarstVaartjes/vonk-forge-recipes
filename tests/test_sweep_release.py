"""A platform fix requeues the failures it may have cured; --retry-failed requeues on request."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep import policy
from spark_sweep.cli import main

OLD, NEW = "1" * 40, "2" * 40
IMAGE_BUG = "runtime image no longer matches the approved recipe"


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def _failing_set() -> tuple[list[FakeRecipe], list[FakeModel]]:
    recipes = [
        FakeRecipe(
            "image-bug", ("m1",), fail_load=IMAGE_BUG, fail_phase="runtime-install"
        ),
        FakeRecipe(
            "corrupt",
            ("m2",),
            fail_download="sha256 mismatch for model-00001.safetensors",
            fail_download_code="model.digest_mismatch",
        ),
        FakeRecipe("policy", ("m3",), fail_download="policy"),
        FakeRecipe(
            "oom",
            ("m4",),
            fail_load="CUDA error: out of memory",
            fail_phase="start",
            fail_code="recipe.runtime_oom",
        ),
        FakeRecipe("fine", ("m5",)),
    ]
    return recipes, [FakeModel(f"m{i}") for i in range(1, 6)]


def test_platform_side_classes_are_requeued_and_recipe_side_ones_are_not() -> None:
    for klass in (
        "install",
        "start",
        "readiness",
        "fit",
        "capacity",
        "build",
        "download",
        "review",
    ):
        assert policy.platform_side(klass), klass
    for klass in (
        "model-integrity",
        "build-policy",
        "oom",
        "network",
        "timeout",
        "smoke-assertion",
        "smoke-request",
    ):
        assert not policy.platform_side(klass), klass
    assert policy.platform_side(
        None
    )  # a failure recorded before classes existed: maybe the platform's


def test_a_new_controller_release_requeues_platform_failures_only_and_keeps_the_history(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _failing_set()
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    fleet.release_sha = OLD
    sweep.run()
    assert {k.split("/")[1]: e["status"] for k, e in sweep.state.recipes.items()} == {
        "image-bug": "deferred",
        "corrupt": "failed",
        "policy": "failed",
        "oom": "failed",
        "fine": "passed",
    }
    assert all(e["release"] == OLD for e in sweep.state.recipes.values())
    old_lines = (tmp_path / "results.jsonl").read_text()
    assert all(
        line["controller_release"] == OLD for line in _lines(tmp_path / "results.jsonl")
    )

    # The fix is deployed: the accepted release changes, and the recipe no longer fails.
    fleet.release_sha = NEW
    fleet.recipes["vonk-forge/image-bug"].fail_load = None
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
    assert again.run() == 0
    assert (
        _entry(again, "image-bug")["status"] == "passed"
        and _entry(again, "image-bug")["release"] == NEW
    )
    for slug in (
        "corrupt",
        "policy",
        "oom",
    ):  # recipe-side: a platform change cannot have cured them
        assert (
            _entry(again, slug)["status"] == "failed"
            and _entry(again, slug)["release"] == OLD
        )
    assert _entry(again, "fine")["release"] == OLD  # untouched, not retested

    # results.jsonl only grew: the old failure, a requeue line naming why, then the new pass.
    lines = _lines(tmp_path / "results.jsonl")
    assert (tmp_path / "results.jsonl").read_text().startswith(old_lines)
    mine = [
        (x["step"], x["status"], x.get("controller_release"))
        for x in lines
        if x["recipe"] == "vonk-forge/image-bug"
    ]
    assert mine == [
        ("smoke", "deferred", OLD),
        ("requeue", "requeued", NEW),
        ("smoke", "passed", NEW),
    ]
    requeue = next(x for x in lines if x["step"] == "requeue")
    assert (
        requeue["reason"].startswith("controller-release-changed")
        and requeue["previous"]["release"] == OLD
    )
    assert requeue["previous"]["error"] and IMAGE_BUG in requeue["previous"]["error"]


def test_evidence_of_a_requeued_failure_is_kept(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG, fail_phase="runtime-install")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    fleet.release_sha = OLD
    sweep.run()
    first = Path(_entry(sweep, "image-bug")["evidence_bundle"])
    assert first.exists()
    fleet.release_sha = (
        NEW  # a release that does not fix it: it fails again, under the new release
    )
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    again.run()
    second = Path(_entry(again, "image-bug")["evidence_bundle"])
    assert second != first and first.exists() and second.exists()
    assert _entry(again, "image-bug")["release"] == NEW


def test_the_same_release_does_not_requeue_again_so_there_is_no_loop(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG, fail_phase="runtime-install")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    fleet.release_sha = OLD
    sweep.run()
    for _ in range(2):
        again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
        again.run()
    assert not [x for x in _lines(tmp_path / "results.jsonl") if x["step"] == "requeue"]
    assert _entry(sweep, "image-bug")["status"] == "deferred"


def test_failures_recorded_before_releases_were_known_are_retried_once(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG, fail_phase="runtime-install")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    for entry in sweep.state.recipes.values():  # as an older sweep wrote them
        entry.pop("release", None)
    sweep.state.data["release"] = {}
    sweep.state.save()
    fleet.recipes["vonk-forge/image-bug"].fail_load = None
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    again.run()
    assert _entry(again, "image-bug")["status"] == "passed"


def test_a_release_change_while_the_sweep_runs_requeues_without_a_restart(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG, fail_phase="runtime-install")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("m1")],
        gateway=gateway,
        release_seconds=20,
        watch_seconds=30,
    )
    fleet.release_sha = OLD

    def deploy_fix(_now: float) -> None:
        if (
            _entry_or_none(sweep).get("status") == "deferred"
            and fleet.release_sha == OLD
        ):
            fleet.release_sha = NEW
            fleet.recipes["vonk-forge/image-bug"].fail_load = None
        if _entry_or_none(sweep).get("status") == "passed":
            raise KeyboardInterrupt  # the watching sweep has done its job

    def _entry_or_none(s) -> dict:
        return s.state.recipes.get("vonk-forge/image-bug", {})

    clock.hooks.append(deploy_fix)
    assert sweep.run() == 130
    assert _entry(sweep, "image-bug")["status"] == "passed"
    assert any(
        "Controller release changed" in e["message"] for e in sweep.state.data["events"]
    )


# -- --retry-failed ------------------------------------------------------------------------------


def test_retry_failed_requeues_everything_that_failed_whatever_the_cause(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _failing_set()
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    for recipe in recipes:  # all causes are fixed
        recipe.fail_load = recipe.fail_download = None
        fleet.recipes[recipe.key].fail_load = fleet.recipes[
            recipe.key
        ].fail_download = None
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet, retry_failed=True)
    assert again.run() == 0
    assert {e["status"] for e in again.state.recipes.values()} == {"passed"}
    requeues = [x for x in _lines(tmp_path / "results.jsonl") if x["step"] == "requeue"]
    assert len(requeues) == 4 and {x["reason"] for x in requeues} == {
        "operator: --retry-failed"
    }


def test_retry_failed_can_be_limited_to_a_selector(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _failing_set()
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    fleet.recipes["vonk-forge/oom"].fail_load = None
    again, _, _ = make_sweep(
        tmp_path, recipes, models, fleet=fleet, retry_failed=True, only=("oom",)
    )
    again.run()
    assert _entry(again, "oom")["status"] == "passed"
    assert (
        _entry(again, "corrupt")["status"] == "failed"
        and _entry(again, "image-bug")["status"] == "deferred"
    )
    assert [
        x["recipe"]
        for x in _lines(tmp_path / "results.jsonl")
        if x["step"] == "requeue"
    ] == ["vonk-forge/oom"]


def test_the_cli_takes_retry_failed_and_only(tmp_path: Path, gateway: Gateway) -> None:
    from sweep_fakes import make_definitions

    from spark_sweep.vonkctl import Vonkctl

    recipes, models = _failing_set()
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    fleet.recipes["vonk-forge/oom"].fail_load = None
    code = main(
        [
            "run",
            "--yes",
            "--state-dir",
            str(tmp_path),
            "--retry-failed",
            "--only",
            "oom",
        ],
        vonkctl_factory=lambda exe: Vonkctl(exe, runner=fleet.runner),
        clock=clock,
        definitions=make_definitions(recipes),
    )
    assert code == 0
    assert (
        json.loads((tmp_path / "state.json").read_text())["recipes"]["vonk-forge/oom"][
            "status"
        ]
        == "passed"
    )


# -- status --------------------------------------------------------------------------------------


def test_failures_are_grouped_by_controller_release_on_the_status_page(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _failing_set()
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    fleet.recipes["vonk-forge/image-bug"].fail_code = "recipe.runtime_exit"
    fleet.release_sha = OLD
    sweep.run()
    fleet.release_sha = NEW
    fleet.accepted_version = "1.0"
    for recipe in ("image-bug",):
        fleet.recipes[
            f"vonk-forge/{recipe}"
        ].fail_load = IMAGE_BUG  # still broken on the new release
    again, _, _ = make_sweep(tmp_path, recipes, models, fleet=fleet)
    again.run()
    status = json.loads((tmp_path / "status.json").read_text())
    groups = {g["release"]: g for g in status["failures_by_release"]}
    assert status["release"] == NEW
    assert groups[NEW]["current"] and groups[NEW]["failed"] == 1
    assert groups[OLD]["failed"] == 3 and not groups[OLD]["current"]
    assert status["failures_by_release"][0]["release"] == NEW  # newest release first
    page = (tmp_path / "status.md").read_text()
    assert (
        "## Failures by Controller release" in page
        and f"release {NEW[:12]}" in page
        and "(current)" in page
    )


def test_watch_retests_after_bounded_cooldown_without_new_release_or_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("recovering", fail_load=IMAGE_BUG)
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1")], gateway=gateway, watch_seconds=3600
    )
    fleet.release_sha = OLD
    first_finished = []

    def repair_and_stop(now: float) -> None:
        assert clock.sleeps < 100, "watch recovery must converge after cooldown"
        entry = sweep.state.recipes.get(recipe.key, {})
        if entry.get("status") == "deferred" and not first_finished:
            first_finished.append(entry["finished_at"])
            fleet.recipes[recipe.key].fail_load = None
        if entry.get("status") == "passed":
            assert now >= first_finished[0] + 24 * 3600
            raise KeyboardInterrupt

    clock.hooks.append(repair_and_stop)
    assert sweep.run() == 130
    assert _entry(sweep, "recovering")["status"] == "passed"
    assert fleet.release_sha == OLD
    requeues = [
        row for row in _lines(tmp_path / "results.jsonl") if row["step"] == "requeue"
    ]
    assert len(requeues) == 1
    assert requeues[0]["reason"] == "scheduled-retest"
    assert requeues[0]["previous"]["status"] == "deferred"
