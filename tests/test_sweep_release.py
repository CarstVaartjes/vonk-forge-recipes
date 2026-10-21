"""Observed owner changes schedule scoped recovery; publication owns no runtime fact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

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
            "image-bug",
            ("1" * 64,),
            content="c" * 64,
            fail_load=IMAGE_BUG,
            fail_phase="runtime-install",
        ),
        FakeRecipe(
            "corrupt",
            ("2" * 64,),
            content="c" * 64,
            fail_download="sha256 mismatch for model-00001.safetensors",
            fail_download_code="model.digest_mismatch",
        ),
        FakeRecipe("policy", ("3" * 64,), content="c" * 64, fail_download="policy"),
        FakeRecipe(
            "oom",
            ("4" * 64,),
            content="c" * 64,
            fail_load="CUDA error: out of memory",
            fail_phase="start",
            fail_code="recipe.runtime_oom",
        ),
        FakeRecipe("fine", ("5" * 64,), content="c" * 64),
    ]
    return recipes, [FakeModel(str(i) * 64) for i in range(1, 6)]


def _deployed_api(fleet, source: str) -> None:
    fleet.api_source_sha = source


def test_accepted_publication_is_not_a_deployed_fix_or_a_retest(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG)
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    _deployed_api(fleet, OLD)
    fleet.release_sha = OLD
    sweep.run()
    previous = dict(_entry(sweep, "image-bug"))
    fleet.release_sha = NEW
    fleet.recipes[recipe.key].fail_load = None
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    again.run()
    current = _entry(again, "image-bug")
    assert current["status"] == "deferred"
    assert current["attempts"] == previous["attempts"]
    assert current["evidence_bundle"] == previous["evidence_bundle"]
    assert current["release"] == OLD
    assert again.release_sha == OLD
    assert again.state.data["publication"]["source_sha"] == NEW
    assert not [
        row for row in _lines(tmp_path / "results.jsonl") if row["step"] == "retest"
    ]


def test_a_new_api_process_does_not_attribute_a_generic_application_fault(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("image-bug", fail_load=IMAGE_BUG)
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], gateway=gateway)
    _deployed_api(fleet, OLD)
    sweep.run()
    previous = dict(_entry(sweep, "image-bug"))
    fleet.api_source_sha = NEW
    fleet.recipes[recipe.key].fail_load = None
    again, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    again.run()
    assert again.release_sha == NEW
    assert _entry(again, "image-bug")["status"] == "deferred"
    assert _entry(again, "image-bug")["attempts"] == previous["attempts"]
    assert _entry(again, "image-bug")["evidence_bundle"] == previous["evidence_bundle"]
    assert _entry(again, "image-bug").get("recovery_basis") is None


def _record_local_argument_fault(sweep, fleet, recipe):
    # This fault is owned by the local CLI argument parser, not an application.
    fleet.api_source_sha = OLD
    fleet.client_build["source_sha"] = OLD
    sweep.check_client()
    basis = sweep._component_observations()["client"]
    assert basis is not None
    entry = sweep.state.entry(recipe.key)
    entry.update(
        status="deferred",
        code="arguments",
        error="local argument parse rejected",
        attempts=3,
        not_before=sweep.clock.now() + 120,
        defer_until=sweep.clock.now() + 240,
        finished_at=sweep.clock.now(),
        recovery_basis=basis.record(),
        evidence={"producer": "local-cli"},
    )
    return entry


def test_only_observed_fault_owner_change_schedules_once_without_resetting_history(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("argument-recovery")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    entry = _record_local_argument_fault(sweep, fleet, recipe)
    previous = dict(entry)
    fleet.release_sha = NEW
    fleet.api_source_sha = NEW
    sweep.check_client()
    assert entry["status"] == "deferred"  # API/publication is not this fault's owner.
    fleet.client_build["source_sha"] = NEW
    sweep.check_client()
    assert entry["status"] == "pending"
    for key in (
        "attempts",
        "not_before",
        "defer_until",
        "finished_at",
        "evidence",
        "recovery_basis",
    ):
        assert entry[key] == previous[key]
    assert len(entry["recovery_history"]) == 1
    assert entry["recovery_history"][0]["reason"] == "observed-fault-owner-changed"
    entry["status"] = "deferred"
    sweep.check_client()
    assert entry["status"] == "deferred"
    assert len(entry["recovery_history"]) == 1


def test_stale_or_contract_mismatched_observation_cannot_trigger_a_retest(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("argument-recovery")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    entry = _record_local_argument_fault(sweep, fleet, recipe)
    fleet.client_build["source_sha"] = NEW
    fleet.platform_observation_age = 90
    sweep.check_client()
    assert entry["status"] == "deferred"
    fleet.platform_observation_age = 0
    fleet.api_contract_sha256 = "f" * 64
    sweep.check_client()
    assert entry["status"] == "deferred"
    fleet.api_contract_sha256 = fleet.client_build["control_contract_sha256"]
    sweep.check_client()
    assert entry["status"] == "pending" and entry["attempts"] == 3


def test_historical_publication_only_failure_is_not_retried_as_a_known_owner(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe("historic")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    entry = sweep.state.entry(recipe.key)
    entry.update(
        status="deferred", code="arguments", attempts=7, release=OLD, finished_at=0
    )
    fleet.api_source_sha = fleet.release_sha = NEW
    fleet.client_build["source_sha"] = NEW
    sweep.check_client()
    assert entry["status"] == "deferred" and entry["attempts"] == 7
    assert not entry.get("recovery_history")


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
            "--no-publish-evidence",
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
    fleet.api_source_sha = OLD
    sweep.run()
    fleet.release_sha = NEW
    fleet.api_source_sha = NEW
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
    assert NEW not in groups  # A new process does not rewrite old failure provenance.
    assert groups[OLD]["failed"] == 4 and not groups[OLD]["current"]
    page = (tmp_path / "status.md").read_text()
    assert "## Failures by Controller release" in page and f"release {OLD[:12]}" in page


def test_watch_retests_platform_fault_only_after_its_observed_owner_fix(
    tmp_path: Path, gateway: Gateway
) -> None:
    from spark_sweep.identity import content_identity
    from spark_sweep.lifecycle import RetestReason
    from spark_sweep.run import RETEST_SECONDS

    recipe = FakeRecipe("recovering")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1")], gateway=gateway, watch_seconds=3600
    )
    sweep.refresh_catalog()
    entry = _record_local_argument_fault(sweep, fleet, recipe)
    entry.update(
        content_sha256="c1",
        verdict_identity=content_identity(sweep.recipes[recipe.key]).model_dump(
            mode="json"
        ),
    )
    since = clock.now()
    fixed = []

    def repair_and_stop(now: float) -> None:
        assert clock.sleeps < 100, "watch recovery must converge after an observed fix"
        if not fixed:
            assert entry["status"] == "deferred"
            assert not sweep.state.slots
            if now >= since + RETEST_SECONDS:
                fleet.client_build["source_sha"] = NEW
                fixed.append(now)
        if entry.get("status") == "passed":
            assert fixed and now >= fixed[0]
            clock.hooks.remove(repair_and_stop)
            raise KeyboardInterrupt

    clock.hooks.append(repair_and_stop)
    assert sweep.run() == 130
    assert entry["status"] == "passed"
    requeues = [
        row for row in _lines(tmp_path / "results.jsonl") if row["step"] == "retest"
    ]
    assert len(requeues) == 1
    assert requeues[0]["reason"] == RetestReason.OWNER_CHANGED
    assert requeues[0]["previous"]["status"] == "deferred"


@pytest.mark.parametrize("phase", ["loading", "smoking"])
@pytest.mark.parametrize("restart", [False, True])
def test_catalog_change_during_lane_keeps_the_verdict_bound_to_accepted_content(
    tmp_path: Path, phase: str, restart: bool
) -> None:
    from dataclasses import replace

    from spark_sweep import policy

    fake = FakeRecipe(
        "changing",
        digests=("1" * 64,),
        content="c" * 64,
        local="cached",
        adapter="readiness",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [fake], [FakeModel("1" * 64, local="cached")]
    )
    sweep.refresh_catalog()
    accepted = sweep.recipes[fake.key]
    sweep._open_slot(
        policy.Placement(accepted, ("spk_a",)), "accepted-request", clock.now()
    )
    slot = sweep.state.slots[fake.key]
    slot.update(phase=phase, ready_at=clock.now(), run_id="accepted-run")
    sweep.state.save()
    if restart:
        sweep, _, _ = make_sweep(tmp_path, [fake], [], fleet=fleet)
        sweep.refresh_catalog()
        slot = sweep.state.slots[fake.key]
    sweep.recipes[fake.key] = replace(
        accepted,
        content_sha256="d" * 64,
        revision_id="rev-c2",
        model_digests=("2" * 64,),
    )
    sweep._submit_smoke(fake.key, slot)
    sweep._advance_smoking(fake.key, slot)
    assert sweep.state.entry(fake.key)["content_sha256"] == accepted.content_sha256
    row = next(
        row for row in _lines(tmp_path / "results.jsonl") if row["status"] == "passed"
    )
    assert row["content_sha256"] == accepted.content_sha256
    assert row["recipe_revision_id"] == accepted.revision_id
    assert row["run_id"] == "accepted-run"
    assert row["library_commit"] == "lib-commit-1"
    sweep._reconcile_entry(sweep.recipes[fake.key], 2)
    entry = sweep.state.entry(fake.key)
    assert entry["status"] == "pending"
    assert entry["previous"]["content_sha256"] == accepted.content_sha256


@pytest.mark.parametrize("bound", [False, True])
def test_saved_lane_fixture_never_certifies_the_catalog_at_restart(
    tmp_path: Path, bound: bool
) -> None:
    from spark_sweep.identity import IdentityReason

    fixture = Path(__file__).parent / "fixtures/sweep-accepted-content-state.json"
    saved = json.loads(fixture.read_text())
    key = "vonk-forge/changing"
    if not bound:
        saved["slots"][key].pop("accepted_content")
    (tmp_path / "state.json").write_text(json.dumps(saved))
    fake = FakeRecipe(
        "changing",
        digests=("1" * 64,),
        content="d" * 64,
        local="cached",
        adapter="readiness",
    )
    sweep, fleet, _ = make_sweep(
        tmp_path, [fake], [FakeModel("1" * 64, local="cached")]
    )
    sweep.refresh_catalog()
    slot = sweep.state.slots[key]
    sweep._submit_smoke(key, slot)
    if bound:
        sweep._advance_smoking(key, slot)
    rows = _lines(tmp_path / "results.jsonl")
    passed = [row for row in rows if row["status"] == "passed"]
    assert [row["content_sha256"] for row in passed] == (["c" * 64] if bound else [])
    assert sweep.state.entry(key)["status"] == "pending"
    assert slot["phase"] == "finished"
    if bound:
        assert passed[0]["library_commit"] == "accepted-library"
        assert passed[0]["run_id"] == "saved-run"
    else:
        assert rows[-1]["reason"] == IdentityReason.UNBOUND
    assert not [call for _, call in fleet.calls if call[:2] == ("fleet", "cancel")]


def test_lane_failure_and_its_evidence_bind_to_the_accepted_content(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from spark_sweep import policy

    fake = FakeRecipe("changing", digests=("1" * 64,), content="c" * 64, local="cached")
    sweep, _, clock = make_sweep(
        tmp_path, [fake], [FakeModel("1" * 64, local="cached")]
    )
    sweep.refresh_catalog()
    accepted = sweep.recipes[fake.key]
    sweep._open_slot(
        policy.Placement(accepted, ("spk_a",)), "accepted-request", clock.now()
    )
    sweep.recipes[fake.key] = replace(
        accepted, content_sha256="d" * 64, revision_id="rev-c2"
    )
    sweep._record_failure(
        fake.key,
        policy.classify("start", fake.fail_code, "injected failure"),
        None,
    )
    row = next(
        row for row in _lines(tmp_path / "results.jsonl") if row["step"] == "smoke"
    )
    assert row["content_sha256"] == accepted.content_sha256
    assert row["recipe_revision_id"] == accepted.revision_id
    assert row["request_key"] == "accepted-request"
    assert row["model_digests"] == ["1" * 64]
    entry = sweep.state.entry(fake.key)
    assert entry["status"] == "pending"
    assert entry["previous"]["content_sha256"] == accepted.content_sha256


def test_smoke_submission_uses_the_accepted_adapter_and_checks(
    tmp_path: Path, monkeypatch
) -> None:
    from concurrent.futures import Future
    from dataclasses import replace

    from spark_sweep import policy

    fake = FakeRecipe("changing", local="cached")
    sweep, _, clock = make_sweep(tmp_path, [fake], [FakeModel("m1", local="cached")])
    sweep.refresh_catalog()
    accepted = sweep.recipes[fake.key]
    sweep._open_slot(
        policy.Placement(accepted, ("spk_a",)), "accepted-request", clock.now()
    )
    slot = sweep.state.slots[fake.key]
    sweep.recipes[fake.key] = replace(
        accepted, content_sha256="c2", adapter="readiness"
    )
    submitted = []

    class Executor:
        def submit(self, fn, *args):
            submitted.append(args[0])
            return Future()

    sweep.executor = Executor()
    monkeypatch.setattr(
        sweep.vk,
        "call",
        lambda *args, **kwargs: {
            "assignments": [
                {"alias": slot["alias"], "endpoint": {"api_base": "http://fake"}}
            ]
        },
    )
    sweep._submit_smoke(fake.key, slot)
    assert submitted == [accepted]
