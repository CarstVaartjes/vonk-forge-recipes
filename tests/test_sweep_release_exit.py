"""A deferred (blocked-platform) recipe has one exit: the Controller release it failed on changed."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy, recovery, verdict
from spark_sweep.run import Sweep


def _deferred(tmp_path: Path, monkeypatch, release: str):
    monkeypatch.setattr(Sweep, "release_sha", property(lambda self: release_box[0]))
    release_box[0] = release
    sweep, _, _ = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64)],
        [FakeModel("1" * 64)],
    )
    sweep.preflight()
    sweep.refresh_catalog()
    key = next(iter(sweep.recipes))
    entry = sweep.state.entry(key)
    entry.update(
        status="deferred",
        failure_class="start",
        code="application.failed",
        release=release,
    )
    return sweep, key, entry


release_box = ["rel-1"]


def test_a_deferred_recipe_is_not_retried_on_the_same_release(tmp_path, monkeypatch):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "rel-1")
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred" and not entry.get("recovery_history")


def test_a_deferred_recipe_is_requeued_once_when_the_release_changes(
    tmp_path, monkeypatch
):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "rel-1")
    release_box[0] = "rel-2"
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "pending"
    assert entry["recovery_history"][-1]["basis"] == {"controller_release": "rel-2"}
    # Failing again on rel-2 records rel-2; the same release never reopens it twice.
    entry.update(status="deferred", release="rel-2")
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "deferred"


def test_a_failed_recipe_is_unaffected_by_a_release_change(tmp_path, monkeypatch):
    sweep, _key, entry = _deferred(tmp_path, monkeypatch, "rel-1")
    entry["status"] = "failed"
    release_box[0] = "rel-2"
    sweep.requeue_for_changed_owner()
    assert entry["status"] == "failed" and not entry.get("recovery_history")


@pytest.mark.parametrize(
    "klass", ["start", "timeout", "smoke", "download", *sorted(verdict.OBSERVE_CLASSES)]
)
def test_every_deferred_state_has_a_named_exit_when_the_platform_changes(klass):
    deferred = {"status": "deferred", "failure_class": klass, "release": "rel-1"}
    assert recovery.changed_release(deferred, "rel-1") is None
    assert recovery.changed_release(deferred, None) is None
    assert recovery.changed_release(deferred, "rel-2") == {
        "controller_release": "rel-2"
    }
    assert recovery.changed_release({**deferred, "status": "failed"}, "rel-2") is None


def test_an_unclassified_class_still_defers_without_recipe_blame():
    failure = policy.classify("start", "application.failed", "unknown")
    assert verdict.disposition(failure) is verdict.Disposition.FAIL
