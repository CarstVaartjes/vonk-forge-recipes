"""A cancelled receipt that carries a failure is an observed outcome, bounded by repetition.

Recorded incident (2026-11-05, glm-5-3-flash-exl3-k2-0xsero-vllm-single): the Controller ends
an exhausted image preparation (a 15-minute no-progress timeout) as ``cancelled`` with
``failure: {code: recipe_image.preparation_exhausted, retryable: false}``. The sweep read only
the lifecycle word and asked again about every 76 minutes: 182 requests, 383 such receipts
across 12 recipes. The receipt is the real terminal one from the live state.

Owner rule (identical-exit retry bound): three identical outcomes on the same content identity
end deferred with evidence; a content change retries. A supersession cancel names a recovery
action and is never an outcome.
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
SUPERSEDED = {
    "code": "recipe_image.superseded_by_newer_revision",
    "retryable": False,
    "recovery_actions": ["download_again"],
}


class Mode:
    """What the Controller answers: the recorded receipt shape, or the fake's own."""

    failure: dict[str, Any] | None | str = "recorded"  # None: no failure at all
    real = False


def _setup(tmp_path: Path):
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
    key = next(iter(sweep.recipes))
    mode = Mode()
    real = sweep.vk.runner

    def runner(argv, timeout):
        args = list(argv)
        if "progress" in args and not mode.real:
            op = next(
                (
                    o
                    for o in fleet.ops.values()
                    if o["id"] in args or o["request"] in args
                ),
                None,
            )
            if op is not None:
                receipt = json.loads(RECEIPT.read_text())
                receipt["request"]["selector"] = key
                receipt.update(
                    id=op["id"],
                    request_id=op["request"],
                    recipe_content_sha256=op["recipe_content_sha256"],
                    recipe_revision_id=op["recipe_revision_id"],
                )
                if mode.failure is None:
                    receipt.pop("failure")
                    receipt["children"] = []
                elif mode.failure != "recorded":
                    receipt["failure"] = mode.failure
                return 0, json.dumps(receipt), ""
        return real(argv, timeout)

    sweep.vk.runner = runner
    return sweep, fleet, clock, key, mode


def _requests(fleet) -> list[tuple]:
    return [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]


def _cycle(sweep, fleet, clock, key) -> None:
    """Ask for the download when the sweep would, then read its terminal receipt."""
    clock.sleep(PIN_RETRY_MAX_SECONDS + 1)
    recipe = sweep.recipes[key]
    assert sweep.prefetcher._can_request(recipe, clock.now())
    sweep.prefetcher._request(recipe, "model")
    sweep.prefetcher._poll()  # accepts the parent
    clock.sleep(1)
    sweep.prefetcher._poll()  # reads its terminal receipt


def test_two_exhausted_cancels_ask_again_and_the_third_defers(tmp_path: Path) -> None:
    sweep, fleet, clock, key, _mode = _setup(tmp_path)
    entry = sweep.state.entry(key)
    for expected in (1, 2):
        _cycle(sweep, fleet, clock, key)
        assert len(_requests(fleet)) == expected
        assert entry["status"] == "pending"
        assert entry["exit_repeat"]["count"] == expected
    _cycle(sweep, fleet, clock, key)
    assert len(_requests(fleet)) == 3
    assert entry["status"] == "deferred"
    assert entry["code"] == "recipe_image.preparation_exhausted"
    assert entry["evidence"]["identical_exits"] == 3
    assert key not in [r.key for r in sweep.pending()]
    for _ in range(3):  # nothing asks again, however long it waits
        clock.sleep(10 * PIN_RETRY_MAX_SECONDS)
        sweep.prefetcher._poll()
        assert key not in [r.key for r in sweep.pending()]
    assert len(_requests(fleet)) == 3


def test_a_content_change_reopens_the_deferral_and_a_new_request_is_admitted(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, key, mode = _setup(tmp_path)
    for _ in range(3):
        _cycle(sweep, fleet, clock, key)
    assert sweep.state.entry(key)["status"] == "deferred"
    ops = len(fleet.ops)
    fleet.recipes[key].content = "b" * 64
    mode.real = True  # the new document is answered by the Controller's own rules
    sweep.refresh_catalog()
    assert sweep.state.entry(key)["status"] == "pending"
    sweep.queue = list(sweep.recipes)
    clock.sleep(PIN_RETRY_MAX_SECONDS + 1)
    for _ in range(5):
        sweep.tick()
        clock.sleep(30)
    assert len(fleet.ops) == ops + 1  # sent and admitted by the fake
    new = list(fleet.ops.values())[-1]
    assert new["recipe_content_sha256"] == "b" * 64


def test_exhausted_then_a_landed_download_resets_the_count(tmp_path: Path) -> None:
    sweep, fleet, clock, key, mode = _setup(tmp_path)
    entry = sweep.state.entry(key)
    for _ in range(2):
        _cycle(sweep, fleet, clock, key)
    assert entry["exit_repeat"]["count"] == 2
    mode.real = True
    clock.sleep(PIN_RETRY_MAX_SECONDS + 1)
    sweep.prefetcher._request(sweep.recipes[key], "model")
    clock.sleep(fleet.download_seconds + 1)
    sweep.prefetcher._poll()
    assert sweep.state.downloads[key]["state"] == "succeeded"
    assert "exit_repeat" not in entry


def test_a_supersession_cancel_is_never_an_outcome(tmp_path: Path) -> None:
    sweep, fleet, clock, key, mode = _setup(tmp_path)
    mode.failure = SUPERSEDED
    entry = sweep.state.entry(key)
    for expected in (1, 2, 3, 4):
        _cycle(sweep, fleet, clock, key)
        assert len(_requests(fleet)) == expected
        assert entry["status"] == "pending"
        assert "exit_repeat" not in entry


def test_a_cancel_without_a_failure_is_asked_again_as_before(tmp_path: Path) -> None:
    sweep, fleet, clock, key, mode = _setup(tmp_path)
    mode.failure = None
    for expected in (1, 2, 3, 4):
        _cycle(sweep, fleet, clock, key)
        assert len(_requests(fleet)) == expected
        assert sweep.state.entry(key)["status"] == "pending"
        assert "exit_repeat" not in sweep.state.entry(key)


def test_a_cancel_with_a_retryable_failure_is_asked_again_as_before(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, key, mode = _setup(tmp_path)
    mode.failure = {"code": "model_cache.unavailable", "retryable": True}
    for expected in (1, 2, 3, 4):
        _cycle(sweep, fleet, clock, key)
        assert len(_requests(fleet)) == expected
        assert sweep.state.entry(key)["status"] == "pending"
