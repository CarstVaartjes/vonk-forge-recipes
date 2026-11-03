"""A deferred (blocked-platform) recipe has one exit: the Controller release it failed on changed."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy, recovery, verdict


def _deferred(tmp_path: Path, _monkeypatch, release: str):
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64)],
        [FakeModel("1" * 64)],
    )
    fleet.api_source_sha = release
    release_box[:] = [fleet]
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.observe_platform()
    key = next(iter(sweep.recipes))
    entry = sweep.state.entry(key)
    entry.update(
        status="deferred",
        failure_class="start",
        code="application.failed",
        release=release,
    )
    return sweep, key, entry


release_box: list = []


def _deploy(sweep, release: str) -> None:
    release_box[0].api_source_sha = release
    sweep.observe_platform()


def test_a_deferred_recipe_is_not_retried_on_the_same_release(tmp_path, monkeypatch):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "1" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred" and not entry.get("recovery_history")


def test_a_deferred_recipe_is_requeued_once_when_the_release_changes(
    tmp_path, monkeypatch
):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "1" * 40)
    _deploy(sweep, "2" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "pending"
    assert entry["recovery_history"][-1]["basis"] == {"controller_release": "2" * 40}
    # Failing again on rel-2 records rel-2; the same release never reopens it twice.
    entry.update(status="deferred", release="2" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred"


def test_a_failed_recipe_is_unaffected_by_a_release_change(tmp_path, monkeypatch):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "1" * 40)
    entry["status"] = "failed"
    _deploy(sweep, "2" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "failed" and not entry.get("recovery_history")


@pytest.mark.parametrize(
    "klass", ["start", "timeout", "smoke", "download", *sorted(verdict.OBSERVE_CLASSES)]
)
def test_every_deferred_state_has_a_named_exit_when_the_platform_changes(klass):
    deferred = {"status": "deferred", "failure_class": klass, "release": "1" * 40}
    assert recovery.changed_release(deferred, "1" * 40) is None
    assert recovery.changed_release(deferred, None) is None
    expected = {"controller_release": "2" * 40}
    assert recovery.changed_release(deferred, "2" * 40) == expected
    # A failure that recorded no release (platform unreadable then, or an older entry).
    assert recovery.changed_release({**deferred, "release": None}, "2" * 40) == expected
    absent = {k: v for k, v in deferred.items() if k != "release"}
    assert recovery.changed_release(absent, "2" * 40) == expected
    assert recovery.changed_release({**deferred, "status": "failed"}, "2" * 40) is None


def test_an_unclassified_class_still_defers_without_recipe_blame():
    failure = policy.classify("start", "application.failed", "unknown")
    assert verdict.disposition(failure) is verdict.Disposition.FAIL


def test_a_named_owner_fault_without_an_observed_basis_still_has_an_exit():
    fault = {"status": "deferred", "code": "arguments", "release": "1" * 40}
    assert recovery.changed_release(fault, "2" * 40) == {"controller_release": "2" * 40}
    owned = {**fault, "recovery_basis": {"owner": "client", "source_sha": "x"}}
    assert recovery.changed_release(owned, "2" * 40) is None


def test_a_download_that_fails_after_a_release_flip_is_retested_once_on_the_new_release(
    tmp_path, monkeypatch
):
    from spark_sweep.lifecycle import RetestReason
    from spark_sweep.prefetch import Prefetcher
    from spark_sweep.run import AcceptedRequest

    sweep, key, entry = _deferred(tmp_path, monkeypatch, "1" * 40)
    recipe = sweep.recipes[key]
    sweep.prefetcher.release = lambda: sweep.release_sha
    # A download record is created by the prefetcher with the release it started on.
    sweep.state.downloads[key] = {
        "state": "failed",
        "operation_id": "op-1",
        "request_key": "rk-1",
        "request_intent": {
            "recipe_key": key,
            "content_sha256": recipe.content_sha256,
            "revision_id": recipe.revision_id,
        },
        "release": "1" * 40,
        "accepted_content": AcceptedRequest(
            request_key="rk-1", recipe=recipe
        ).model_dump(mode="json"),
    }
    entry["status"] = "pending"
    _deploy(sweep, "2" * 40)  # the release flips while the download runs
    sweep._download_failed(
        key, policy.classify("download", "download.failed", "unknown"), "op-1", False
    )
    assert entry["status"] == "deferred" and entry["release"] == "1" * 40
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "pending"
    assert entry["retest"]["reason"] == RetestReason.RELEASE_CHANGED
    # The download retest path accepts the release retest: it asks for the request's receipt.
    before = len(
        [c for _, c in release_box[0].calls if c[:2] == ("recipe", "progress")]
    )
    Prefetcher._terminal_retest(
        sweep.prefetcher, recipe, sweep.state.downloads[key], sweep.clock.now() + 10**6
    )
    after = len([c for _, c in release_box[0].calls if c[:2] == ("recipe", "progress")])
    assert after == before + 1
    # One retest per release.
    entry.update(status="deferred", release="2" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred"


def test_a_load_started_while_the_platform_was_unreadable_is_retested_once(tmp_path):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64, local="cached")],
        [FakeModel("1" * 64, local="cached")],
    )
    fleet.api_source_sha = None  # the platform reply names no release
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.observe_platform()
    sweep.schedule(clock.now())
    key = next(iter(sweep.state.slots))
    assert sweep.state.slots[key]["release"] is None
    fleet.api_source_sha = "2" * 40
    sweep.observe_platform()
    sweep._record_failure(
        key, policy.classify("start", "application.failed", "unknown"), None
    )
    entry = sweep.state.entry(key)
    assert entry["status"] == "deferred" and entry["release"] is None
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "pending"
    assert entry["recovery_history"][-1]["basis"] == {"controller_release": "2" * 40}
    entry.update(status="deferred", release="2" * 40)
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred"


def test_a_release_that_flips_mid_load_is_recorded_against_the_one_the_load_started_on(
    tmp_path,
):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64, local="cached")],
        [FakeModel("1" * 64, local="cached")],
    )
    fleet.api_source_sha = "1" * 40
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.observe_platform()
    sweep.schedule(clock.now())
    key = next(iter(sweep.state.slots))
    fleet.api_source_sha = "2" * 40
    sweep.observe_platform()
    assert sweep.release_sha == "2" * 40
    sweep._record_failure(
        key, policy.classify("start", "application.failed", "unknown"), None
    )
    entry = sweep.state.entry(key)
    assert entry["status"] == "deferred" and entry["release"] == "1" * 40
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "pending"
    assert entry["recovery_history"][-1]["basis"] == {"controller_release": "2" * 40}


def test_a_download_request_records_the_release_it_started_on(tmp_path):
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64)],
        [FakeModel("1" * 64)],
    )
    fleet.api_source_sha = "1" * 40
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.observe_platform()
    sweep.prefetcher.release = lambda: sweep.release_sha
    key = next(iter(sweep.recipes))
    sweep.prefetcher._request(sweep.recipes[key], "model")
    assert sweep.state.downloads[key]["release"] == "1" * 40
    # A request the Controller finally refuses replaces the record: the release it
    # started on is carried, not the one deployed by the time the refusal is read.
    sweep.state.downloads.pop(key)
    sweep.prefetcher._request(sweep.recipes[key], "model")
    assert sweep.state.downloads[key]["release"] == "1" * 40

    def refuse_after_a_flip(a):
        fleet.api_source_sha = (
            "2" * 40
        )  # the release flips while the request is in flight
        sweep.observe_platform()
        return fleet._error(next(iter(policy.RECIPE_FAILURE_CODES)), "refused")

    fleet._download = refuse_after_a_flip
    sweep.state.downloads.pop(key)
    sweep.prefetcher._request(sweep.recipes[key], "model")
    assert sweep.state.downloads[key]["state"] == "failed"
    assert sweep.state.downloads[key]["release"] == "1" * 40
    assert sweep.state.entry(key)["release"] == "1" * 40
