"""Failure handling: classes, retry, timeouts, evidence, clusters, model-level causes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, Gateway, make_sweep

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
    assert Path(failed["evidence"]).exists()  # `fleet evidence` was downloaded
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
    )
    sweep, _, _ = make_sweep(tmp_path, [flaky], [FakeModel("m1")], gateway=gateway)
    sweep.run()
    entry = _entry(sweep, "flaky")
    assert entry["status"] == "failed" and entry["attempts"] == 2
    assert entry["failure_class"] == "network"


def test_non_transient_failure_is_not_retried(tmp_path: Path, gateway: Gateway) -> None:
    oom = FakeRecipe("oom", fail_load="CUDA error: out of memory", fail_phase="start")
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
    assert (entry["status"], entry["phase"]) == ("failed", "timeout")
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
        "failed",
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
    assert (entry["status"], entry["phase"]) == ("failed", "review")
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


def test_model_level_download_failure_fails_siblings_without_trying_them(
    tmp_path: Path, gateway: Gateway
) -> None:
    lead = FakeRecipe(
        "lead", ("m1",), fail_download="sha256 mismatch for model-00001.safetensors"
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
    assert _entry(sweep, "sibling")["inherited_from"] == "vonk-forge/lead"
    assert _entry(sweep, "other")["status"] == "passed"
    downloads = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert "vonk-forge/sibling" not in downloads


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
        ),
        FakeRecipe(
            "b",
            ("m2",),
            fail_load="kernel image 456 not found for device 9",
            fail_phase="start",
        ),
        FakeRecipe("c", ("m3",), fail_load="driver mismatch", fail_phase="start"),
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
