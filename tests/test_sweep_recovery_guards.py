"""Platform observations release lanes without blame and admit fresh requests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.lifecycle import ACTIVE
from spark_sweep.prefetch import DOWNLOAD_STALL_SECONDS, DOWNLOAD_WALL_SECONDS
from spark_sweep.state import State
from spark_sweep.vonkctl import RECIPE_ERROR_CODES, Reply, is_infrastructure


@pytest.mark.parametrize("status", range(100, 600))
def test_every_http_error_preserves_the_request_until_canonical_recovery(
    tmp_path: Path, status: int
) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.refresh_catalog()
    recipe = next(iter(sweep.recipes.values()))
    document = {
        "error_type": "control_api",
        "code": f"http.{status}",
        "status": status,
        "detail": "Controller response",
    }
    fleet.observations.append((("recipe", "download"), 2, document))
    sweep.prefetcher._request(recipe, "model")
    assert sweep.state.status(recipe.key) == "pending"
    assert not sweep.state.entry(recipe.key).get("attempts")
    original = dict(sweep.state.downloads[recipe.key])
    assert original["state"] == "observing"
    assert not original.get("operation_id")
    fleet.observations.append((("recipe", "progress"), 2, document))
    sweep.prefetcher._poll()
    assert sweep.state.downloads[recipe.key]["request_key"] == original["request_key"]
    assert not [call for _, call in fleet.calls if call[:2] == ("recipe", "download")][
        1:
    ]
    clock.sleep(60)
    sweep.prefetcher._poll()  # actual native absence authorizes only same-key replay
    record = sweep.state.downloads[recipe.key]
    assert record["state"] == "running"
    assert record["request_key"] == original["request_key"]
    assert record["attempt"] == original["attempt"] == 1
    assert record["started_at"] == original["started_at"]
    assert not sweep.state.entry(recipe.key).get("attempts")
    assert not sweep.state.data["download_history"]
    requests = [
        call[-1] for _, call in fleet.calls if call[:2] == ("recipe", "download")
    ]
    assert requests == [original["request_key"], original["request_key"]]


@pytest.mark.parametrize("field", ["status", "status_code", "http_status"])
@pytest.mark.parametrize("status", [401, 403, 404, 409, 429, 500, 502, 503, 504])
def test_http_status_overrides_recipe_code(field: str, status: int) -> None:
    for code in RECIPE_ERROR_CODES:
        assert is_infrastructure(Reply((), 2, {"code": code, field: str(status)}, ""))


@pytest.mark.parametrize(
    "code",
    [
        "controller.unavailable",
        "controller.invalid_request",
        "controller.protocol_invalid",
        "controller.transport_timeout",
        "profile.busy",
        "unknown",
    ],
)
def test_platform_codes_do_not_become_recipe_faults(code: str) -> None:
    assert is_infrastructure(
        Reply(
            (), 2, {"code": code, "detail": "dockerfile.heredoc_forbidden: prose"}, ""
        )
    )


def test_legacy_load_adopted_without_changing_request_identity(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    original = State(path)
    original.data["load"] = {
        "request_key": "old",
        "app_id": "app-old",
        "state": "observing",
    }
    original.save()
    adopted = State.load(path)
    assert adopted.data["loads"] == {"old": original.data["load"]}
    adopted.save()
    assert "load" not in json.loads(path.read_text())


@pytest.mark.parametrize("word", sorted(ACTIVE | {"unexpected-state"}))
def test_all_active_download_states_keep_polling(tmp_path: Path, word: str) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.refresh_catalog()
    recipe = next(iter(sweep.recipes.values()))
    sweep.prefetcher._request(recipe, "model")
    record = sweep.state.downloads[recipe.key]
    record["state"] = word
    op = record["operation_id"]
    fleet.observations.append(
        (("recipe", "progress"), 0, {"id": op, "state": "succeeded", "progress": {}})
    )
    sweep.prefetcher._poll()
    assert record["state"] == "succeeded"
    assert record["observed_at"] == clock.now()


@pytest.mark.parametrize("reason", ["needs-operator", "unreadable"])
def test_download_wait_preserves_owned_request_without_cancellation(
    tmp_path: Path, reason: str
) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.refresh_catalog()
    recipe = next(iter(sweep.recipes.values()))
    sweep.prefetcher._request(recipe, "model")
    record = sweep.state.downloads[recipe.key]
    old_key = record["request_key"]
    if reason == "needs-operator":
        fleet.observations.append(
            (
                ("recipe", "progress"),
                0,
                {"id": record["operation_id"], "state": reason, "progress": {}},
            )
        )
        sweep.prefetcher._poll()
    else:
        record["progress_at"] = clock.now() - DOWNLOAD_STALL_SECONDS
        sweep.prefetcher._poll()
    assert record["state"] == "observing"
    assert sweep.state.status(recipe.key) == "pending"
    clock.t += sweep.prefetcher.config.retry_cooldown
    sweep.prefetcher.pause_until = 0
    assert not sweep.prefetcher._can_request(recipe, clock.now())
    fleet.observations.append(
        (("recipe", "progress"), 0, {"id": record["operation_id"], "state": "running"})
    )
    sweep.prefetcher._poll()
    assert record["state"] == "running"
    assert record["request_key"] == old_key
    assert not any(call[:2] == ("recipe", "cancel") for _, call in fleet.calls)


def _loading(tmp_path: Path):
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [FakeRecipe("a", local="cached", copy_seconds=9000)],
        [FakeModel("m1", local="cached")],
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    return sweep, fleet, clock


@pytest.mark.parametrize("reason", ["needs-operator", "wall-budget", "gate"])
def test_application_wait_ends_without_blame_and_fresh_load_is_admitted(
    tmp_path: Path, reason: str
) -> None:
    sweep, fleet, clock = _loading(tmp_path)
    load = next(iter(sweep.state.data["loads"].values()))
    old_key = load["request_key"]
    if reason == "needs-operator":
        fleet.observations.append(
            (("profile", "progress"), 0, {"id": load["app_id"], "state": reason})
        )
    elif reason == "gate":
        fleet.observations.append(
            (
                ("profile", "progress"),
                0,
                {
                    "id": load["app_id"],
                    "state": "backoff",
                    "blockers": [
                        {
                            "code": "model.deletion_in_progress",
                            "severity": "error",
                            "detail": "gate held",
                        }
                    ],
                },
            )
        )
    else:
        sweep.state.slots["vonk-forge/a"]["wall_deadline"] = clock.now()
    sweep.advance_slots(clock.now())
    assert sweep.state.data["loads"] == {}
    assert all(e["status"] == "pending" for e in sweep.state.recipes.values())
    sweep.cleanup_finished()
    assert not sweep.state.slots
    clock.t += sweep.cfg.retry_delay
    sweep.backoff_until = 0
    sweep.schedule(clock.now())
    assert next(iter(sweep.state.data["loads"])) != old_key


def test_pending_work_does_not_exit_after_thirty_idle_ticks(tmp_path: Path) -> None:
    sweep, _, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    start = clock.now()

    def tick() -> None:
        if clock.now() - start > 400:
            raise KeyboardInterrupt

    sweep.tick = tick
    sweep.done = lambda: False
    sweep._nothing_moves = lambda: True
    assert sweep.run() == 130
    assert any(
        "IDLE: pending work" in event["message"] for event in sweep.state.data["events"]
    )


def test_external_download_wait_is_bounded_without_cancelling_foreign_work(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.refresh_catalog()
    recipe = next(iter(sweep.recipes.values()))
    from dataclasses import replace

    recipes = {recipe.key: replace(recipe, local="preparing")}
    sweep.prefetcher._track_external(recipes, clock.now())
    clock.t += DOWNLOAD_WALL_SECONDS
    sweep.prefetcher._track_external(recipes, clock.now())
    assert sweep.state.downloads[recipe.key]["state"] == "retired"
    assert not [c for _, c in fleet.calls if c[:2] == ("recipe", "cancel")]
    clock.t += sweep.prefetcher.config.retry_cooldown
    assert sweep.prefetcher._can_request(recipe, clock.now())


def test_cleanup_error_never_retains_busy_lane_and_is_retried(tmp_path: Path) -> None:
    sweep, fleet, clock = _loading(tmp_path)
    load = next(iter(sweep.state.data["loads"].values()))
    sweep._end_load(load, "application.needs-operator")
    fleet.observations.append(
        (("profile", "remove"), 2, {"code": "http.503", "error_type": "control_api"})
    )
    sweep.cleanup_finished()
    assert not sweep.state.slots
    assert sweep.state.data["cleanup"]
    assert sweep.state.status("vonk-forge/a") == "pending"
    clock.t += sweep.cfg.retry_delay * 2
    sweep.cleanup_finished()
    assert not sweep.state.data["cleanup"]
    sweep.backoff_until = 0
    sweep.schedule(clock.now())
    assert sweep.state.data["loads"]


def test_application_history_is_bounded_and_keeps_live_requests(tmp_path: Path) -> None:
    state = State(tmp_path / "state.json")
    state.data["apps"] = {str(i): {"state": "succeeded"} for i in range(10000)}
    state.slots["live"] = {"request_key": "7", "phase": "loading"}
    state.data["release_history"] = list(range(10000))
    state.save()
    assert state.data["apps"] == {"7": {"state": "succeeded"}}
    assert len(state.data["release_history"]) == 60


def test_dual_does_not_use_a_busy_spark_and_singles_get_distinct_sparks(
    tmp_path: Path,
) -> None:
    from spark_sweep import policy

    recipes = [
        FakeRecipe("a", local="cached", port=8000),
        FakeRecipe("b", local="cached", port=8001),
        FakeRecipe("dual", local="cached", node_count=2),
    ]
    sweep, _, clock = make_sweep(tmp_path, recipes, [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    singles = [recipe for recipe in sweep.recipes.values() if recipe.node_count == 1]
    sweep._apply(
        policy.place_singles(
            singles, sweep._bins(), sweep.cfg.reserve_bytes, lambda r: r.slug
        ),
        clock.now(),
    )
    assert len(sweep.state.slots) == 2
    assert len({slot["node_ids"][0] for slot in sweep.state.slots.values()}) == 2
    assert sweep._bins() == []
    assert (
        policy.place_dual(
            list(sweep.recipes.values()), sweep._bins(), sweep.cfg.reserve_bytes
        )
        is None
    )


def test_smoke_future_has_total_budget_and_does_not_block_a_fresh_load(
    tmp_path: Path,
) -> None:
    from concurrent.futures import Future

    from spark_sweep.run import SMOKE_WALL_SECONDS

    sweep, _, clock = _loading(tmp_path)
    key = next(iter(sweep.state.slots))
    slot = sweep.state.slots[key]
    old_key = slot["request_key"]
    slot.update(phase="smoking", ready_at=clock.now())
    sweep.futures[key] = Future()
    clock.t += SMOKE_WALL_SECONDS
    sweep._advance_smoking(key, slot)
    assert slot["phase"] == "finished"
    assert sweep.state.status(key) == "pending"
    assert not sweep.futures
    sweep.cleanup_finished()
    clock.t += sweep.cfg.retry_delay
    sweep.schedule(clock.now())
    assert next(iter(sweep.state.data["loads"])) != old_key


def test_request_map_observes_each_lane_independently(tmp_path: Path) -> None:
    sweep, fleet, _ = make_sweep(tmp_path, [FakeRecipe("a")], [FakeModel("m1")])
    sweep.state.data["loads"] = {
        "slow": {"request_key": "slow", "app_id": "slow-app"},
        "fast": {"request_key": "fast", "app_id": "fast-app"},
    }
    fleet.observations.extend(
        [
            (("profile", "progress"), 0, {"id": "slow-app", "state": "observing"}),
            (("profile", "progress"), 0, {"id": "fast-app", "state": "succeeded"}),
        ]
    )
    assert sweep.poll_application()
    assert list(sweep.state.data["loads"]) == ["slow"]
    assert sweep.state.data["apps"]["fast"]["state"] == "succeeded"


def test_unknown_load_receipt_is_observed_without_rolling_back_accepted_work(
    tmp_path: Path,
) -> None:
    from sweep_fakes import Gateway

    gateway = Gateway()
    gateway.start()
    try:
        sweep, fleet, _ = make_sweep(
            tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
        )
        real = fleet.runner
        lost = False

        def runner(argv, timeout):
            nonlocal lost
            reply = real(argv, timeout)
            if not lost and "load" in argv and "--yes" in argv:
                lost = True
                return (
                    2,
                    json.dumps({"error_type": "control_api", "code": "http.503"}),
                    "",
                )
            return reply

        sweep.vk.runner = runner
        assert sweep.run() == 0
        assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
        assert sweep.state.recipes["vonk-forge/a"]["attempts"] == 1
        assert not sweep.state.data["loads"] and not sweep.state.data["dirty"]
        placements = [
            item
            for item in sweep.state.data["own_loads"]
            if item["kind"] == "placement"
        ]
        assert len(placements) == 1
    finally:
        gateway.stop()


@pytest.mark.parametrize(
    "contents",
    [
        "{broken",
        "[]",
        '{"schema": 1, "slots": []}',
        '{"schema": 1, "slots": {"bad": "broken"}}',
        '{"schema": 1, "load": "broken"}',
        '{"schema": 1, "owner": {"baseline": []}}',
    ],
)
def test_unreadable_state_is_preserved_and_fresh_request_can_be_recorded(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "state.json"
    path.write_text(contents)
    state = State.load(path)
    assert next(tmp_path.glob("state.json.unreadable-*")).read_text() == contents
    state.data["loads"]["fresh"] = {"request_key": "fresh", "state": "queued"}
    state.save()
    assert State.load(path).data["loads"]["fresh"]["state"] == "queued"


def test_unknown_platform_failure_ends_without_recipe_blame_and_releases_lane(
    tmp_path: Path,
) -> None:
    from spark_sweep import policy

    sweep, _, clock = _loading(tmp_path)
    key = next(iter(sweep.state.slots))
    sweep._record_failure(
        key,
        policy.classify("start", "application.failed", "network timeout out of memory"),
        None,
    )
    assert sweep.state.status(key) == "deferred"
    assert sweep.state.entry(key)["failure_class"] == "start"
    sweep.cleanup_finished()
    sweep.release_orphan_load()
    assert not sweep.state.slots and not sweep.state.data["loads"]
    sweep._requeue_failed(key, "fresh-request")
    sweep.backoff_until = 0
    sweep.schedule(clock.now())
    assert sweep.state.data["loads"]


def test_watch_retests_deferred_outcome_with_a_fresh_request(tmp_path: Path) -> None:
    from spark_sweep import policy
    from spark_sweep.run import RETEST_SECONDS

    sweep, _, clock = _loading(tmp_path)
    key = next(iter(sweep.state.slots))
    original = sweep.state.slots[key]["request_key"]
    sweep._record_failure(
        key, policy.classify("start", "application.failed", "unknown"), None
    )
    sweep.cleanup_finished()
    sweep.release_orphan_load()
    sweep.cfg.watch_seconds = 60
    clock.t += RETEST_SECONDS
    sweep.tick()
    assert sweep.state.status(key) == "pending"
    assert next(iter(sweep.state.data["loads"])) != original


def test_unreadable_owner_never_looks_like_absent_owner(tmp_path: Path) -> None:
    sweep, fleet, clock = _loading(tmp_path)
    fleet.observations.append((("profile", "progress"), 2, {"code": "http.503"}))
    status = sweep.guard.check(clock.now())
    assert status.paused and "unreadable" in status.reason


def test_stuck_owner_ends_with_owned_holds_released(tmp_path: Path) -> None:
    from spark_sweep.owner import OwnerStatus
    from spark_sweep.run import OWNER_WAIT_SECONDS

    sweep, _, clock = _loading(tmp_path)
    sweep.owner_status = OwnerStatus(True, "owner intent is running")
    sweep.guard.check = lambda now: sweep.owner_status
    sweep.state.data["owner"]["paused_at"] = clock.now() - OWNER_WAIT_SECONDS
    assert sweep.run() == 0
    assert sweep.state.data["end"]["code"] == "sweep.owner_intent_superseded"
    assert not sweep.state.slots and not sweep.state.data["loads"]
    old_requests = {item["request_key"] for item in sweep.state.data["own_loads"]}
    sweep.owner_status = OwnerStatus()
    assert sweep.run() == 0
    assert "end" not in sweep.state.data
    assert any(
        item["request_key"] not in old_requests
        for item in sweep.state.data["own_loads"]
    )


def test_removed_catalog_recipe_releases_lane_and_fresh_recipe_is_admitted(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock = _loading(tmp_path)
    key = next(iter(sweep.state.slots))
    fleet.recipes.clear()
    sweep.refresh_catalog()
    assert sweep.state.status(key) == "deferred"
    sweep.cleanup_finished()
    sweep.release_orphan_load()
    assert not sweep.state.slots and not sweep.state.data["loads"]
    fresh = FakeRecipe("fresh", local="cached")
    fleet.recipes[fresh.key] = fresh
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.backoff_until = 0
    sweep.schedule(clock.now())
    assert sweep.state.slots and sweep.state.data["loads"]


def test_results_growth_is_bounded_without_losing_latest_outcome(
    tmp_path: Path,
) -> None:
    from spark_sweep.state import ResultsLog

    path = tmp_path / "results.jsonl"
    line = json.dumps({"authority_id": "auth", "recipe": "old", "padding": "x" * 5000})
    path.write_text((line + "\n") * 1000)
    log = ResultsLog(path, "auth")
    log.append(recipe="fresh", status="passed")
    assert path.stat().st_size <= 4 * 1024 * 1024
    assert log.entries()[-1]["recipe"] == "fresh"
