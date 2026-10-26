"""Prefetch: bounded concurrency, in-flight downloads, eviction pins, measured throughput."""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep
from vonk_forge_contracts.sweep import SweepWaitReason

from spark_sweep import prefetch
from spark_sweep.prefetch import DOWNLOAD_STALL_SECONDS, PrefetchConfig
from spark_sweep.report import build_status

TIB = 1024**4


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _active_ops(fleet) -> int:
    return sum(1 for op in fleet.ops.values() if op["state"] == "running")


def _many(n: int) -> tuple[list[FakeRecipe], list[FakeModel]]:
    return [FakeRecipe(f"r{i}", (f"m{i}",)) for i in range(n)], [
        FakeModel(f"m{i}") for i in range(n)
    ]


def test_model_downloads_never_exceed_the_slot_budget(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(6)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=2, pin_profile=13),
    )
    peak = []
    clock.hooks.append(lambda _t: peak.append(_active_ops(fleet)))
    sweep.run()
    assert max(peak) == 2  # saturated, never above the budget
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_preparing_siblings_share_one_model_download_slot(tmp_path: Path) -> None:
    """One active model with many preparing recipe views cannot exhaust the pool."""
    recipes = [FakeRecipe(f"shared-{i}", ("shared",)) for i in range(8)]
    recipes += [FakeRecipe(f"cold-{i}", (f"cold-{i}",)) for i in range(3)]
    models = [FakeModel("shared"), *(FakeModel(f"cold-{i}") for i in range(3))]
    sweep, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        models,
        prefetch=PrefetchConfig(max_model_downloads=3, pin_profile=13),
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[recipes[0].key], "model")
    # All recipes sharing that model report preparing; only one parent exists.
    for recipe in recipes[:8]:
        recipe.local = "preparing"
    sweep.refresh_catalog()
    result = sweep.prefetcher.tick(
        sweep.recipes,
        sweep.models,
        list(sweep.recipes.values()),
        sweep.sizes,
        sweep._present(),
        sweep.boost,
    )
    assert result["model_ops"] == 3
    assert _active_ops(fleet) == 3
    assert sum(args[:2] == ("recipe", "download") for _, args in fleet.calls) == 3


def test_a_controller_space_hold_longer_than_the_window_pauses_no_other_work(
    tmp_path: Path, gateway: Gateway
) -> None:
    """The Controller alone owns NAS space: a download it cannot fit yet is a queued
    ``download_blocked`` wait until eviction can free a set, here once the sweep has
    tested and unpinned it. Past the sweep's own observation window it stays a typed
    NAS-space wait, not an infrastructure problem: other downloads and image pulls go on
    (the hold is released only by them)."""
    # The hold lifts once m3's recipes are tested; m3 is the last set to download
    # and two recipes share it, so once it lands its second recipe pulls only an image.
    blocked = FakeRecipe("blocked", ("big",), download_blocked_behind="m3")
    others = [FakeRecipe(f"o{i}", (f"m{i}",)) for i in (1, 2, 4)]
    o3, o3img = FakeRecipe("o3", ("m3",)), FakeRecipe("o3img", ("m3",), engine="sglang")
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [blocked, *others, o3, o3img],
        [
            FakeModel("big", bytes=TIB),
            *(FakeModel(f"m{i}", bytes=TIB) for i in (1, 2, 4)),
            FakeModel("m3", bytes=3 * TIB),
        ],
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=2, pin_profile=13),
    )
    fleet.download_seconds = 1500.0
    fleet.page_size = 100  # the first catalog view is complete: plan order is fixed
    start = clock.now()
    during: list[tuple[float, dict, set[str], set[str], list[dict]]] = []
    runaway: list[float] = []

    def observe(now: float) -> None:
        op = next((o for o in fleet.ops.values() if o["recipe"] == blocked.key), None)
        record = sweep.state.downloads.get(blocked.key)
        if op is None or op["state"] != "backoff" or record is None:
            return
        if now - start > 40000:  # the hold is released only by the work behind it
            runaway.append(now)
            op.update(state="running", started=now)
            return
        during.append(
            (
                now,
                dict(record.get("watch") or {}),
                {o["recipe"] for o in fleet.ops.values()},
                set(sweep.state.data["infra"]),
                build_status(sweep)["downloads"]["in_flight"],
            )
        )

    clock.hooks.append(observe)
    sweep.run()

    assert not runaway, "other work stopped behind the Controller's space hold"
    assert during[-1][0] - during[0][0] > DOWNLOAD_STALL_SECONDS
    # Past its window the wait is no infrastructure problem and pauses nothing.
    assert all("download" not in infra for _, _, _, infra, _ in during)
    _, _, requested, _, _ = during[-1]
    assert {o3.key, o3img.key} <= requested
    kinds = {sweep.state.downloads[k]["kind"] for k in (o3.key, o3img.key)}
    assert kinds == {"model", "image"}
    # A typed, visible NAS-space wait on the original request, never a failure.
    watches = [watch for _, watch, _, _, _ in during if watch]  # once observed
    assert len(watches) > 1
    assert all(w["wait"]["reason"] == SweepWaitReason.NAS_SPACE for w in watches)
    assert all(
        w["wait_deadline"] <= w["wait_started_at"] + DOWNLOAD_STALL_SECONDS
        for w in watches
        if "wait_deadline" in w
    )
    # Its window restarts: it is looked at again once per window, not once per poll.
    assert any(
        w.get("wait_started_at", start) > start + DOWNLOAD_STALL_SECONDS for w in watches
    )
    renewals = [e for e in sweep.state.data["events"] if "waits for NAS space" in e["message"]]
    assert 1 <= len(renewals) <= (during[-1][0] - start) // DOWNLOAD_STALL_SECONDS
    shown = [
        item["waiting_on"]
        for *_, in_flight in during
        for item in in_flight
        if item["recipe"] == blocked.key
    ]
    while shown and shown[0] is None:
        shown.pop(0)  # requested, not yet observed
    assert shown and all(
        w is not None and w.startswith(SweepWaitReason.NAS_SPACE) for w in shown
    )
    downloads = [c for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert [c[2] for c in downloads].count(blocked.key) == 1
    # Eviction freed the space: the download ran and the recipe was tested.
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert not sweep.state.recipes[blocked.key].get("failure_class")
    assert not [c for p, c in fleet.calls if p == 13 and c[:2] == ("profile", "load")]


def test_new_model_downloads_wait_for_pin_capacity_and_resume_as_pins_drain(
    tmp_path: Path, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A downloaded, untested set left out of the full pin profile would be evicted and
    downloaded again. While the sets to protect fill the profile, no new model download
    starts; the hold is visible and lifts as tested sets leave the profile."""
    limit = 3  # the contract's 64, scaled down
    monkeypatch.setattr(prefetch, "PIN_PROFILE_MAX_ASSIGNMENTS", limit)
    recipes, models = _many(6)
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=6, pin_profile=13),
    )
    fleet.max_assignments = limit
    unprotected: list[set[str]] = []
    holds: list[dict] = []

    def observe(_now: float) -> None:
        pinned = {
            a["recipe_selector"]
            for a in fleet.profiles.get(13, {}).get("assignments", [])
        }
        untested = {
            r.key
            for r in recipes
            if sweep.state.recipes.get(r.key, {}).get("status") == "pending"
            and (
                all(fleet.models[d].local == "cached" for d in r.digests)
                or any(
                    o["recipe"] == r.key and o["state"] == "running"
                    for o in fleet.ops.values()
                )
            )
        }
        unprotected.append(untested - pinned)
        if (hold := build_status(sweep)["downloads"]["pin_capacity"]) is not None:
            holds.append(hold)

    clock.hooks.append(observe)
    sweep.run()

    assert not any(unprotected), [u for u in unprotected if u][:3]
    assert holds and {h["reason"] for h in holds} == {SweepWaitReason.PIN_CAPACITY}
    assert sweep.prefetcher.pin_hold is None  # released as the pins drained
    downloads = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert sorted(downloads) == sorted(r.key for r in recipes)  # each fetched once
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}


def test_a_download_someone_else_started_is_not_requested_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    # The operator already requested r0 (in flight); the library reports it as preparing.
    fleet.ops["op-ext"] = {
        "id": "op-ext",
        "recipe": "vonk-forge/r0",
        "state": "running",
        "started": clock.now(),
        "duration": 100.0,
        "request": "ext",
        "bytes": 20 * GB,
    }
    sweep.run()
    requested = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert "vonk-forge/r0" not in requested and "vonk-forge/r1" in requested
    assert (
        sweep.state.recipes["vonk-forge/r0"]["status"] == "passed"
    )  # tested once it landed


def test_cached_recipes_are_tested_before_ones_that_need_a_download(
    tmp_path: Path, gateway: Gateway
) -> None:
    cold, warm = (
        FakeRecipe("cold", ("m1",)),
        FakeRecipe("warm", ("m2",), local="cached"),
    )
    sweep, fleet, _ = make_sweep(
        tmp_path,
        [cold, warm],
        [FakeModel("m1"), FakeModel("m2", local="cached")],
        gateway=gateway,
        sparks=("spark-a",),
    )
    sweep.run()
    adds = [c[2] for p, c in fleet.calls if p == 10 and c[:2] == ("profile", "add")]
    assert adds == ["vonk-forge/warm", "vonk-forge/cold"]


def test_seed_list_recipes_go_first(tmp_path: Path, gateway: Gateway) -> None:
    recipes, models = _many(4)
    sweep, fleet, _ = make_sweep(
        tmp_path,
        recipes,
        models,
        gateway=gateway,
        seed=frozenset({"r3"}),
        prefetch=PrefetchConfig(max_model_downloads=1, pin_profile=None),
    )
    sweep.run()
    downloads = [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]
    assert downloads[0] == "vonk-forge/r3"


def test_downloaded_recipes_are_pinned_until_tested_and_the_pin_profile_is_never_loaded(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(3)
    sweep, fleet, clock = make_sweep(tmp_path, recipes, models, gateway=gateway)
    pinned_during = []
    clock.hooks.append(
        lambda _t: pinned_during.append(
            list(fleet.profiles.get(13, {}).get("assignments", []))
        )
    )
    sweep.run()
    assert max(len(p) for p in pinned_during) >= 2
    assert {a["desired_state"] for p in pinned_during for a in p} == {
        "installed"
    }  # named, never started
    assert (
        fleet.profiles[13]["assignments"] == []
    )  # released when everything was tested
    assert not [c for p, c in fleet.calls if p == 13 and c[:2] == ("profile", "load")]
    assert sweep.state.data["pins"] == []


def test_pin_profile_trouble_does_not_stop_the_sweep(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, fleet, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    fleet.reject_pins = True
    sweep.run()
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert "assignment limit" in (sweep.prefetcher.pin_error or "")
    attempts = [c for p, c in fleet.calls if p == 13 and c[:2] == ("profile", "add")]
    assert 1 <= len(attempts) <= 6  # backs off, never hammers, never gives up for good


def test_nas_full_pauses_new_model_downloads_and_is_a_wait(
    tmp_path: Path, gateway: Gateway
) -> None:
    full = FakeRecipe(
        "full",
        ("m1",),
        fail_download="insufficient free space on the NAS",
        fail_download_code="storage.insufficient_capacity",
    )
    sweep, _, clock = make_sweep(
        tmp_path,
        [full, FakeRecipe("next", ("m2",))],
        [FakeModel("m1"), FakeModel("m2")],
        gateway=gateway,
        prefetch=PrefetchConfig(max_model_downloads=1, pin_profile=None),
    )
    start = clock.now()

    def collector_frees_space(now: float) -> None:
        if now - start > 3600:
            full.fail_download = None

    clock.hooks.append(collector_frees_space)
    sweep.run()
    assert any("NAS pressure" in e["message"] for e in sweep.state.data["events"])
    # A full NAS is a wait, never a verdict: once there is room the recipe is tested.
    assert sweep.state.recipes["vonk-forge/full"]["status"] == "passed"
    assert not sweep.state.recipes["vonk-forge/full"].get("failure_class")
    assert sweep.state.recipes["vonk-forge/next"]["status"] == "passed"


def test_throughput_is_measured_from_the_operations(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes, models = _many(2)
    sweep, _, _ = make_sweep(tmp_path, recipes, models, gateway=gateway)
    sweep.run()
    assert sweep.rate.rate == pytest.approx(
        2 * 20 * GB / 60, rel=0.01
    )  # two operations moving at once
    assert sweep.state.data["rate"]["samples"] > 0


def test_one_pin_protects_a_model_shared_by_several_recipes(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("a1", ("m1",)),
        FakeRecipe("a2", ("m1",), engine="sglang"),
        FakeRecipe("a3", ("m1",), engine="trtllm"),
        FakeRecipe("b", ("m2",)),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path, recipes, [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    peak: list[int] = []
    clock.hooks.append(
        lambda _t: peak.append(len(fleet.profiles.get(13, {}).get("assignments", [])))
    )
    sweep.run()
    assert max(peak) == 2  # one per model set, not one per recipe
