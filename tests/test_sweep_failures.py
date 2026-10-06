"""Failure handling: classes, retry, timeouts, evidence, clusters, model-level causes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

from spark_sweep.policy import TimeoutPolicy


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _entry(sweep, slug: str) -> dict:
    return sweep.state.recipes[f"vonk-forge/{slug}"]


def test_failed_load_is_recorded_with_phase_class_and_evidence(
    tmp_path: Path, gateway: Gateway
) -> None:
    good = FakeRecipe("good", ("m1",))
    bad = FakeRecipe(
        "bad",
        ("m2",),
        fail_load="container exited with code 1: unsupported kernel",
        fail_phase="start",
        fail_code="recipe.runtime_exit",
    )
    sweep, fleet, _ = make_sweep(
        tmp_path, [good, bad], [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    sweep.run()
    assert _entry(sweep, "good")["status"] == "passed"  # the other lane is unaffected
    failed = _entry(sweep, "bad")
    assert failed["status"] == "failed"
    assert (failed["phase"], failed["failure_class"]) == ("start", "start")
    assert "unsupported kernel" in failed["error"]
    assert Path(failed["evidence_bundle"]).exists()  # `fleet evidence` was downloaded
    line = next(
        json.loads(x)
        for x in (tmp_path / "results.jsonl").read_text().splitlines()
        if '"failed"' in x
    )
    assert (
        line["recipe"] == "vonk-forge/bad"
        and line["phase"] == "start"
        and line["evidence"]
    )
    # A failed recipe is stopped and removed from the lane profile.
    assert fleet.profiles[10]["assignments"] == []


def test_transient_failure_is_retried_once_then_final(
    tmp_path: Path, gateway: Gateway
) -> None:
    flaky = FakeRecipe(
        "flaky",
        fail_load="connection reset by peer while pulling",
        fail_phase="container-download",
        fail_code="controller.transport_timeout",
    )
    sweep, _, _ = make_sweep(tmp_path, [flaky], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    entry = _entry(sweep, "flaky")
    assert entry["status"] == "deferred" and entry["attempts"] == 2
    assert entry["failure_class"] == "network"


def test_non_transient_failure_is_not_retried(tmp_path: Path, gateway: Gateway) -> None:
    oom = FakeRecipe(
        "oom",
        fail_load="CUDA error: out of memory",
        fail_phase="start",
        fail_code="recipe.runtime_oom",
    )
    sweep, _, _ = make_sweep(tmp_path, [oom], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    assert (
        _entry(sweep, "oom")["attempts"] == 1
        and _entry(sweep, "oom")["failure_class"] == "oom"
    )


def test_load_timeout_cancels_the_application(tmp_path: Path, gateway: Gateway) -> None:
    slow = FakeRecipe("slow", load_seconds=10**6)
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [slow],
        [FakeModel("m1")],
        gateway=gateway,
        timeouts=TimeoutPolicy(first_start_seconds=300),
    )
    sweep.run()
    entry = _entry(sweep, "slow")
    assert (entry["status"], entry["phase"]) == ("deferred", "timeout")
    assert any(c[:2] == ("profile", "cancel") for p, c in fleet.calls if p == 10)


def test_smoke_failure_is_a_smoke_phase_failure(
    tmp_path: Path, gateway: Gateway
) -> None:
    gateway.broken.add("broken")
    sweep, _, _ = make_sweep(
        tmp_path, [FakeRecipe("broken")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    entry = _entry(sweep, "broken")
    assert (entry["status"], entry["phase"], entry["failure_class"]) == (
        "deferred",
        "smoke",
        "smoke-request",
    )


def test_declared_demand_beyond_a_spark_fails_at_review(
    tmp_path: Path, gateway: Gateway
) -> None:
    huge = FakeRecipe("huge", memory=128 * 10**9)
    sweep, _, _ = make_sweep(tmp_path, [huge], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    entry = _entry(sweep, "huge")
    assert (entry["status"], entry["phase"]) == ("deferred", "review")
    assert "insufficient_capacity" in entry["code"]


def test_recipes_needing_more_sparks_are_skipped_not_tested(
    tmp_path: Path, gateway: Gateway
) -> None:
    big = FakeRecipe("big", node_count=4)
    sweep, fleet, _ = make_sweep(
        tmp_path, [big, FakeRecipe("ok")], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    assert _entry(sweep, "big")["status"] == "skipped"
    assert "needs 4 Sparks" in _entry(sweep, "big")["reason"]
    assert not any(
        "big" in " ".join(c) for _, c in fleet.calls if c[:2] == ("recipe", "download")
    )


def test_model_level_download_failure_still_tests_siblings_independently(
    tmp_path: Path, gateway: Gateway
) -> None:
    lead = FakeRecipe(
        "lead",
        ("m1",),
        fail_download="sha256 mismatch for model-00001.safetensors",
        fail_download_code="model.digest_mismatch",
    )
    sibling = FakeRecipe("sibling", ("m1",), engine="sglang")
    other = FakeRecipe("other", ("m2",))
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [lead, sibling, other],
        [FakeModel("m1"), FakeModel("m2")],
        gateway=gateway,
    )
    sweep.run()
    assert _entry(sweep, "lead")["failure_class"] == "model-integrity"
    assert _entry(sweep, "sibling")["status"] == "passed"
    assert "inherited_from" not in _entry(sweep, "sibling")
    assert _entry(sweep, "other")["status"] == "passed"
    downloads = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert "vonk-forge/sibling" in downloads


def test_build_policy_failure_does_not_condemn_siblings(
    tmp_path: Path, gateway: Gateway
) -> None:
    lead = FakeRecipe("lead", ("m1",), fail_download="policy")
    sibling = FakeRecipe("sibling", ("m1",), engine="sglang")
    sweep, _, _ = make_sweep(
        tmp_path, [lead, sibling], [FakeModel("m1")], gateway=gateway
    )
    sweep.run()
    assert _entry(sweep, "lead")["failure_class"] == "build-policy"
    assert _entry(sweep, "sibling")["status"] == "passed"


def test_same_root_cause_forms_one_cluster(tmp_path: Path, gateway: Gateway) -> None:
    recipes = [
        FakeRecipe(
            "a",
            ("m1",),
            fail_load="kernel image 123 not found for device 7",
            fail_phase="start",
            fail_code="recipe.runtime_exit",
        ),
        FakeRecipe(
            "b",
            ("m2",),
            fail_load="kernel image 456 not found for device 9",
            fail_phase="start",
            fail_code="recipe.runtime_exit",
        ),
        FakeRecipe(
            "c",
            ("m3",),
            fail_load="driver mismatch",
            fail_phase="start",
            fail_code="recipe.runtime_exit",
        ),
    ]
    sweep, _, _ = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1"), FakeModel("m2"), FakeModel("m3")],
        gateway=gateway,
    )
    sweep.run()
    report = json.loads((tmp_path / "report.json").read_text())
    sizes = sorted(len(c["recipes"]) for c in report["failure_clusters"])
    assert sizes == [1, 2]


def test_a_bad_checkpoint_found_at_start_pushes_its_siblings_to_the_back_but_still_tests_them(
    tmp_path: Path, gateway: Gateway
) -> None:
    lead = FakeRecipe(
        "lead",
        ("m1",),
        fail_load="invalid model: corrupt safetensors header",
        fail_phase="start",
        fail_code="model.digest_mismatch",
    )
    sibling = FakeRecipe("sibling", ("m1",), engine="sglang")
    others = [FakeRecipe(f"other{i}", (f"m{i + 2}",)) for i in range(3)]
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [lead, sibling, *others],
        [FakeModel(f"m{i + 1}") for i in range(4)],
        gateway=gateway,
        sparks=("spark-a",),
    )
    sweep.run()
    order = [c[2] for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")]
    assert (
        order[0] == "vonk-forge/lead" and order[-1] == "vonk-forge/sibling"
    )  # siblings last
    assert sweep.state.recipes["vonk-forge/lead"]["failure_class"] == "model-integrity"
    assert (
        sweep.state.recipes["vonk-forge/sibling"]["status"] == "passed"
    )  # still tested, not condemned


def test_a_transport_unknown_download_recovers_without_recipe_blame(
    tmp_path: Path, gateway: Gateway
) -> None:
    flaky = FakeRecipe(
        "flaky",
        fail_download="connection reset by peer",
        fail_download_code="controller.transport_timeout",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [flaky], [FakeModel("m1")], gateway=gateway
    )

    def repair(now: float) -> None:
        if now >= 1_000_000 + 200:
            fleet.recipes[flaky.key].fail_download = None

    clock.hooks.append(repair)
    assert sweep.run() == 0
    entry = _entry(sweep, "flaky")
    assert entry["status"] == "passed"
    assert not sweep.state.data["model_failures"]
    assert not [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
        if json.loads(line)["status"] == "failed"
    ]


def test_a_controller_that_stops_answering_costs_a_pass_not_the_sweep(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    real = fleet.runner
    outage = {"until": clock.now() + 100}

    def runner(argv, timeout):
        if clock.now() < outage["until"] and "library" in argv:
            from spark_sweep.vonkctl import VonkctlTimeout

            raise VonkctlTimeout(argv, None, "vonkctl timed out")
        return real(argv, timeout)

    sweep.vk.runner = runner
    assert sweep.run() == 0
    assert _entry(sweep, "a")["status"] == "passed"
    assert any(
        "library page failed" in e["message"] for e in sweep.state.data["events"]
    )


@pytest.mark.parametrize(
    "detail",
    ["CUDA error: out of memory", "sha256 mismatch for model-00001.safetensors"],
)
def test_generic_platform_fault_never_blames_recipe_or_siblings(
    tmp_path: Path, gateway: Gateway, detail: str
) -> None:
    lead = FakeRecipe("lead", fail_download=detail)
    sibling = FakeRecipe("sibling", engine="sglang")
    sweep, fleet, clock = make_sweep(
        tmp_path, [lead, sibling], [FakeModel("m1")], gateway=gateway
    )

    def bounded(_now: float) -> None:
        assert clock.sleeps < 200, (
            "platform fault must finish this pass without a terminal loop"
        )

    clock.hooks.append(bounded)
    assert sweep.run() == 0
    entry = _entry(sweep, "lead")
    assert entry["status"] == "deferred"
    assert entry["failure_class"] == "download"
    assert detail in entry["error"]
    assert _entry(sweep, "sibling")["status"] == "passed"
    assert not sweep.state.data["model_failures"]
    assert "vonk-forge/sibling" in [
        c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")
    ]
