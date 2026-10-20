"""Replay the D3 saved renewal wait without touching a production sweep."""

from __future__ import annotations

import io
import json
import urllib.error
from concurrent.futures import Future
from email.message import Message
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy
from vonk_forge_contracts.sweep import SweepWaitCode, SweepWaitReason

from spark_sweep.auth_contract import SmokeAuthClass
from spark_sweep.removal_contract import RecipeVerdict
from spark_sweep.report import build_status
from spark_sweep.smoke import SmokeResult, _transport_failure
from spark_sweep.verdict import Disposition, disposition

FIXTURE = Path(__file__).parent / "fixtures/sweep-d3-route-refusal.json"


def _failure(status: int, body: object):
    error = urllib.error.HTTPError(
        "https://gateway/v1/systemone",
        status,
        "refused",
        Message(),
        io.BytesIO(json.dumps(body).encode()),
    )
    return _transport_failure(error, "D3_TEXT")


def _resumed(tmp_path: Path):
    recipe = FakeRecipe("d3-edge-vllm-single", local="cached")
    models = [FakeModel("m1", local="cached")]
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], models)
    sweep.refresh_catalog()
    sweep._open_slot(
        policy.Placement(sweep.recipes[recipe.key], ("spk_a",)), "original", clock.now()
    )
    saved = json.loads(FIXTURE.read_text())
    saved.pop("response")
    sweep.state.slots[recipe.key].update(saved)
    sweep.state.save()
    resumed, _, _ = make_sweep(tmp_path, [recipe], models, fleet=fleet)
    resumed.refresh_catalog()
    return resumed, recipe, clock, fleet


def test_saved_d3_refusal_fails_immediately_with_case_body_and_releases_lane(
    tmp_path: Path,
):
    sweep, recipe, _, _ = _resumed(tmp_path)
    sweep.note_infra("client-key", "saved renewal incident")
    failure = _failure(403, json.loads(FIXTURE.read_text())["response"])
    future: Future[SmokeResult] = Future()
    future.set_result(SmokeResult(ok=False, kind="service", failure=failure))
    sweep.futures[recipe.key] = future
    sweep._advance_smoking(recipe.key, sweep.state.slots[recipe.key])
    sweep.enqueue_finished_cleanup()
    assert recipe.key not in sweep.state.slots
    entry = sweep.state.entry(recipe.key)
    assert entry["status"] == RecipeVerdict.FAILED
    assert entry["attempts"] == 1
    assert entry["evidence"]["case"] == "D3_TEXT"
    assert entry["evidence"]["http_status"] == 403
    assert "/vllm/systemone" in entry["evidence"]["body"]
    assert "client-key" not in sweep.state.data["infra"]


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "body",
    [
        {"detail": "Forbidden"},
        {"error": {"type": "auth_error", "message": "policy denied"}},
        "expired_key",
    ],
)
def test_untyped_or_policy_auth_denials_cannot_authorize_renewal(
    status: int, body: object
):
    assert disposition(_failure(status, body)) == Disposition.FAIL


@pytest.mark.parametrize("kind", ["expired_key", "token_not_found_in_db"])
def test_typed_key_failure_wait_is_visible_durable_and_releases_lane_at_deadline(
    tmp_path: Path, kind: str
):
    sweep, recipe, clock, fleet = _resumed(tmp_path)
    slot = sweep.state.slots[recipe.key]
    # A fresh auth failure gets one full lane window, not a fresh window on restart.
    slot.pop("smoke_watch")
    slot["ready_at"] = clock.now()
    failure = _failure(401, {"error": {"type": kind, "message": "key unusable"}})
    assert failure.klass == SmokeAuthClass.CREDENTIAL
    sweep._smoke_wait(recipe.key, slot, failure)
    deadline = slot["smoke_watch"]["wait_deadline"]
    (lane,) = [
        item for items in build_status(sweep)["lanes"].values() for item in items
    ]
    assert lane["waiting_on"].startswith(SweepWaitReason.CLIENT_KEY)
    assert lane["elapsed_s"] == 0 and lane["limit_s"] == 1200
    sweep.state.save()
    resumed, _, _ = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("m1", local="cached")],
        fleet=fleet,
    )
    # Use the saved state, and expire at its persisted boundary without another HTTP call.
    clock.sleep(deadline - clock.now() + 1)
    resumed._advance_smoking(recipe.key, resumed.state.slots[recipe.key])
    resumed.enqueue_finished_cleanup()
    assert recipe.key not in resumed.state.slots
    assert resumed.state.entry(recipe.key).get("attempts", 0) == 0
    assert resumed.state.status(recipe.key) == RecipeVerdict.PENDING
    assert resumed.log.entries()[-1]["code"] == SweepWaitCode.BUDGET_EXHAUSTED


def test_saved_legacy_renewal_wait_cannot_gain_another_window(tmp_path: Path):
    sweep, recipe, _, _ = _resumed(tmp_path)
    slot = sweep.state.slots[recipe.key]
    failure = _failure(401, {"error": {"type": "expired_key", "message": "expired"}})
    sweep._smoke_wait(recipe.key, slot, failure)
    sweep._advance_smoking(recipe.key, slot)
    sweep.enqueue_finished_cleanup()
    assert recipe.key not in sweep.state.slots
    assert sweep.state.entry(recipe.key).get("attempts", 0) == 0
