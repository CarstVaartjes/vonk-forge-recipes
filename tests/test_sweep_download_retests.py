"""Retests preserve owned download history and wait for actual terminal evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.prefetch import PrefetchConfig
from spark_sweep.state import State
from spark_sweep.vonkctl import request_key


def _failed_download(tmp_path: Path):
    fake = FakeRecipe(
        "retest",
        fail_download="invalid recipe input",
        fail_download_code="recipe.invalid_definition",
    )
    sweep, fleet, clock = make_sweep(
        tmp_path, [fake], [FakeModel("m1")], prefetch=PrefetchConfig(pin_profile=None)
    )
    sweep.refresh_catalog()
    recipe = sweep.recipes[fake.key]
    sweep.prefetcher._request(recipe, "model")
    clock.sleep(fleet.download_seconds + 1)
    sweep.prefetcher._poll()
    entry = sweep.state.entry(recipe.key)
    entry.update(
        status="failed",
        finished_at=clock.now(),
        content_sha256=recipe.content_sha256,
        revision_id=recipe.revision_id,
    )
    assert sweep.state.downloads[recipe.key]["state"] == "failed"
    return sweep, fleet, clock, recipe


def test_terminal_download_retest_waits_then_keeps_complete_old_receipt(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    record = copy.deepcopy(sweep.state.downloads[recipe.key])
    previous = copy.deepcopy(sweep.state.entry(recipe.key))
    sweep._schedule_retest(recipe.key, "scheduled-retest")
    assert sweep.state.downloads[recipe.key] == record
    assert not sweep.prefetcher._can_request(recipe, clock.now())
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    assert sweep.prefetcher._can_request(recipe, clock.now())
    verified = copy.deepcopy(sweep.state.downloads[recipe.key])
    assert verified["terminal_receipt"]["id"] == record["operation_id"]
    assert verified["terminal_receipt"]["state"] == "failed"
    fleet.recipes[recipe.key].fail_download = None
    sweep.prefetcher._request(recipe, "model")
    new = sweep.state.downloads[recipe.key]
    assert new["attempt"] == record["attempt"] + 1
    assert new["request_key"] == request_key(
        "download", sweep.state.nonce, recipe.key, recipe.content_sha256, new["attempt"]
    )
    assert new["operation_id"] != record["operation_id"]
    assert new["request_key"] != record["request_key"]
    assert new["retry_of_operation_id"] == record["operation_id"]
    assert new["accepted_intent"] == {
        "kind": "retry",
        "operation_id": record["operation_id"],
    }
    retry_calls = [call for _, call in fleet.calls if call[:2] == ("recipe", "retry")]
    assert retry_calls == [
        (
            "recipe",
            "retry",
            record["operation_id"],
            "--yes",
            "--detach",
            "--request-key",
            new["request_key"],
        )
    ]
    assert sweep.state.data["download_history"][recipe.key] == [verified]
    event = sweep.state.entry(recipe.key)["recovery_history"][-1]
    assert all(
        event["previous"][key] == value
        for key, value in previous.items()
        if key in event["previous"]
    )
    sweep.state.save()
    resumed = State.load(sweep.state.path, clock.now)
    assert resumed.downloads[recipe.key] == new
    assert resumed.data["download_history"][recipe.key] == [verified]


def test_active_and_unknown_receipts_keep_exact_download_identity(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    sweep._schedule_retest(recipe.key, "scheduled-retest")
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    sweep.state.downloads[recipe.key]["retry_at"] = clock.now() - 1
    original = copy.deepcopy(sweep.state.downloads[recipe.key])
    canonical = fleet._op_doc(fleet.ops[original["operation_id"]])
    for saved_state in ("failed", "retired"):
        sweep.state.downloads[recipe.key]["state"] = saved_state
        unchanged = copy.deepcopy(sweep.state.downloads[recipe.key])
        for state in ("queued", "running", "observing", "unexpected-state"):
            fleet.observations.append(
                (("recipe", "progress"), 0, {**canonical, "state": state})
            )
            assert not sweep.prefetcher._can_request(recipe, clock.now())
            assert sweep.state.downloads[recipe.key] == unchanged
    assert (
        len([call for _, call in fleet.calls if call[:2] == ("recipe", "download")])
        == 1
    )


def test_bounded_transient_retry_still_requires_exact_terminal_receipt(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    entry = sweep.state.entry(recipe.key)
    entry.clear()
    entry["status"] = "pending"
    record = sweep.state.downloads[recipe.key]
    record["retry_at"] = clock.now() + sweep.prefetcher.config.retry_cooldown
    assert not sweep.prefetcher._can_request(recipe, clock.now())
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    assert sweep.prefetcher._can_request(recipe, clock.now())
    assert record["terminal_receipt"] == fleet._op_doc(
        fleet.ops[record["operation_id"]]
    )
    assert not entry.get("retest")


def test_foreign_or_unreadable_terminal_receipt_cannot_authorize_new_attempt(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    sweep._schedule_retest(recipe.key, "scheduled-retest")
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    sweep.state.downloads[recipe.key]["retry_at"] = clock.now() - 1
    original = copy.deepcopy(sweep.state.downloads[recipe.key])
    for status, document in (
        (0, {"id": "another-operation", "state": "failed"}),
        (2, {"error_type": "control_api", "code": "http.404", "status": 404}),
        (2, {"error_type": "control_api", "code": "controller.transport_timeout"}),
    ):
        fleet.observations.append((("recipe", "progress"), status, document))
        assert not sweep.prefetcher._can_request(recipe, clock.now())
        assert sweep.state.downloads[recipe.key] == original
    canonical = fleet._op_doc(fleet.ops[original["operation_id"]])
    for field, wrong in (
        ("request_id", "another-request"),
        ("recipe_revision_id", "another-revision"),
        ("recipe_content_sha256", "another-content"),
        ("failure", None),
        (
            "request",
            {"kind": "selector", "selector": "vonk-forge/another", "force": False},
        ),
    ):
        document = {**canonical, field: wrong}
        fleet.observations.append((("recipe", "progress"), 0, document))
        assert not sweep.prefetcher._can_request(recipe, clock.now())
        assert sweep.state.downloads[recipe.key] == original
    assert recipe.key not in sweep.state.data["download_history"]


def test_lost_new_reply_persists_same_request_before_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    sweep._schedule_retest(recipe.key, "scheduled-retest")
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    assert sweep.prefetcher._can_request(recipe, clock.now())
    old = copy.deepcopy(sweep.state.downloads[recipe.key])
    original_call = sweep.vk.call

    def interrupted(*args, **kwargs):
        saved = json.loads(sweep.state.path.read_text())
        persisted = saved["downloads"][recipe.key]
        assert persisted["state"] == "observing"
        assert persisted["attempt"] == old["attempt"] + 1
        assert persisted["request_key"] != old["request_key"]
        assert args[:3] == ("recipe", "retry", old["operation_id"])
        assert persisted["retry_of_operation_id"] == old["operation_id"]
        assert persisted["requested_intent"] == {
            "kind": "retry",
            "operation_id": old["operation_id"],
        }
        fleet.observations.append(
            (
                ("recipe", "retry"),
                2,
                {"error_type": "control_api", "code": "controller.transport_timeout"},
            )
        )
        return original_call(*args, **kwargs)

    monkeypatch.setattr(sweep.vk, "call", interrupted)
    sweep.prefetcher._request(recipe, "model")
    new = copy.deepcopy(sweep.state.downloads[recipe.key])
    assert new["state"] == "observing"
    assert new["request_intent"] == old["request_intent"]
    resumed = State.load(sweep.state.path, clock.now)
    assert resumed.downloads[recipe.key] == new
    assert resumed.data["download_history"][recipe.key] == [old]
    assert not sweep.prefetcher._can_request(recipe, clock.now())


def test_publication_does_not_retest_until_actual_fault_owner_changes(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    entry = sweep.state.entry(recipe.key)
    entry.update(
        code="arguments",
        recovery_basis={
            "owner": "client",
            "source_sha": "a" * 40,
            "contract_sha256": "c" * 64,
        },
    )
    original = copy.deepcopy(sweep.state.downloads[recipe.key])
    sweep.state.data["client"] = dict(fleet.client_build)
    sweep.observe_platform()
    sweep.observe_publication(
        {"accepted_source_sha": "b" * 40, "accepted_version": "new"}
    )
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "failed"
    assert not entry.get("recovery_history")
    fleet.client_build["source_sha"] = "b" * 40
    sweep.state.data["client"] = dict(fleet.client_build)
    sweep.observe_platform()
    assert entry["status"] == "pending"
    assert entry["retest"]["reason"] == "observed-fault-owner-changed"
    assert entry["retest"]["basis"]["source_sha"] == "b" * 40
    assert sweep.state.downloads[recipe.key] == original
    assert not sweep.prefetcher._can_request(recipe, clock.now())


def test_cache_arriving_during_retest_reuses_assets_without_download(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, recipe = _failed_download(tmp_path)
    sweep._schedule_retest(recipe.key, "scheduled-retest")
    clock.sleep(sweep.prefetcher.config.retry_cooldown)
    original = copy.deepcopy(sweep.state.downloads[recipe.key])
    fleet.assessments = (
        True  # publish canonical exact-assets readiness, not only local cache state
    )
    fleet.recipes[recipe.key].local = "cached"
    fleet.models["m1"].local = "cached"
    sweep.refresh_catalog()
    current = sweep.recipes[recipe.key]
    assert current.cache_ready
    assert not sweep.prefetcher._can_request(current, clock.now())
    sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        [current],
        sweep.sizes,
        sweep._present(),
        sweep.boost,
        sweep._cached_models(),
    )
    assert sweep.state.downloads[recipe.key] == original
    assert (
        len([call for _, call in fleet.calls if call[:2] == ("recipe", "download")])
        == 1
    )
    assert recipe.key not in sweep.state.data["download_history"]
