"""A receipt's final word is its failure, whatever the lifecycle word.

Recorded incident (2026-11-05, glm-5-3-flash-exl3-k2-0xsero-vllm-single): the Controller ends
an exhausted image preparation as ``cancelled`` (to keep the model download) with
``failure: {code: recipe_image.preparation_exhausted, retryable: false}``. The sweep read only
the lifecycle word, retired the record and asked again about every 76 minutes: 182 requests,
383 such receipts across 12 recipes. The receipt is the real terminal one from the live state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.prefetch import PIN_RETRY_MAX_SECONDS, PrefetchConfig

RECEIPT = (
    Path(__file__).parent
    / "fixtures"
    / "sweep-verdicts"
    / "k2-preparation-exhausted-cancel.json"
)


def _setup(tmp_path: Path, failure: dict[str, Any] | None | bool):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", ("1" * 64,), content="a" * 64)],
        [FakeModel("1" * 64)],
        prefetch=PrefetchConfig(pin_profile=None),
    )
    fleet.api_source_sha = "1" * 40
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.observe_platform()
    sweep.prefetcher.release = lambda: sweep.release_sha
    key = next(iter(sweep.recipes))
    recipe = sweep.recipes[key]
    real = sweep.vk.runner

    def runner(argv, timeout):
        args = list(argv)
        record = sweep.state.downloads.get(key, {})
        request = record.get("request_key") or record.get("previous_request_key")
        if (
            "progress" in args
            and record.get("operation_id")
            and (record["operation_id"] in args or request in args)
        ):
            receipt = json.loads(RECEIPT.read_text())
            receipt["request"]["selector"] = key
            receipt.update(
                id=record["operation_id"],
                request_id=request,
                recipe_content_sha256=recipe.content_sha256,
                recipe_revision_id=f"rev-{recipe.content_sha256}",
            )
            if failure is False:
                receipt.pop("failure")
                receipt["children"] = []
            elif failure is not True:
                receipt["failure"] = failure
            return 0, json.dumps(receipt), ""
        return real(argv, timeout)

    sweep.vk.runner = runner
    sweep.prefetcher._request(recipe, "model")
    record = sweep.state.downloads[key]
    op = next(iter(fleet.ops.values()))
    record["operation_id"] = op["id"]
    return sweep, fleet, clock, key, recipe


def _requests(fleet) -> list[tuple]:
    return [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]


def test_a_cancel_with_a_final_failure_is_recorded_and_never_requested_again(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, key, _recipe = _setup(tmp_path, True)
    assert len(_requests(fleet)) == 1
    sweep.prefetcher._poll()
    entry = sweep.state.entry(key)
    assert entry["status"] == "deferred"  # (a) the platform's, not the recipe's
    assert entry["failure_class"] == "download"
    assert entry["code"] == "recipe_image.preparation_exhausted"
    assert key not in [r.key for r in sweep.pending()]
    for wait in (
        sweep.prefetcher.config.retry_cooldown + 1,
        PIN_RETRY_MAX_SECONDS + 1,
        10 * PIN_RETRY_MAX_SECONDS,
    ):  # (b) not after the cooldown, not after the cap
        clock.sleep(wait)
        sweep.prefetcher._poll()
        assert key not in [r.key for r in sweep.pending()]
        assert len(_requests(fleet)) == 1


def test_a_release_change_reopens_the_final_failure_once(tmp_path: Path) -> None:
    sweep, fleet, _clock, key, recipe = _setup(tmp_path, True)
    sweep.prefetcher._poll()
    entry = sweep.state.entry(key)
    assert entry["status"] == "deferred" and entry["release"] == "1" * 40
    fleet.api_source_sha = "2" * 40
    sweep.observe_platform()
    sweep.requeue_for_changed_owner()  # (c) reopened exactly once
    assert entry["status"] == "pending"
    assert len(entry["recovery_history"]) == 1
    sweep.requeue_for_changed_owner()
    assert len(entry["recovery_history"]) == 1
    assert sweep.prefetcher._can_request(recipe, sweep.clock.now() + 10**6)


def test_a_cancel_without_a_failure_is_asked_again_as_before(tmp_path: Path) -> None:
    sweep, _fleet, clock, key, recipe = _setup(tmp_path, False)
    sweep.prefetcher._poll()
    entry = sweep.state.entry(key)
    assert entry.get("status") != "deferred"
    record = sweep.state.downloads[key]
    assert record["state"] == "retired"
    clock.sleep(PIN_RETRY_MAX_SECONDS + 1)
    assert sweep.prefetcher._can_request(recipe, clock.now())


def test_a_cancel_with_a_retryable_failure_is_asked_again_as_before(
    tmp_path: Path,
) -> None:
    retryable = {"code": "model_cache.unavailable", "retryable": True}
    sweep, _fleet, clock, key, recipe = _setup(tmp_path, retryable)
    sweep.prefetcher._poll()
    assert sweep.state.entry(key).get("status") != "deferred"
    clock.sleep(PIN_RETRY_MAX_SECONDS + 1)
    assert sweep.prefetcher._can_request(recipe, clock.now())
