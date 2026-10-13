"""An enrolled Spark that goes offline keeps its lane: a wait, never a verdict.

Recorded on the hardware sweep (2026-10-13): the Controller failed the load of
qwen3-8-27b-nvfp4-dflash2-eugr-vllm-single (run-switch.final-verification-timeout) while
its run kept starting on spark-3542, and the Spark stopped reporting (offline, stale,
last seen 21:53:10Z). Two minutes later the sweep counted online Sparks, found 1, and
skipped 79 multi-Spark recipes as "not testable on this fleet: needs 2 Sparks, 1 usable";
the spark-3542 lane vanished from the status. Testable is the enrolled topology; fewer
lanes online is a wait, and the lane stays, with the Controller's reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep.report import build_status, render_status

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "sweep-offline-lane-2026-10-13.json"
    ).read_text(encoding="utf-8")
)
OFFLINE, ONLINE = FIXTURE["nodes"]  # spark-3542 (offline, stale), spark-2297
ORPHAN = OFFLINE["loaded"][0]  # the failed application's run, still starting
SINGLES = ("skintokens-pytorch-single", "gemma-4-e2b-it-q8-0-ggml-org-llamacpp-single")
DUALS = tuple(
    e["recipe"].partition("/")[2]
    for e in FIXTURE["skipped_examples"]
    if e["node_count"] == 2
)
EIGHT = next(
    e["recipe"].partition("/")[2]
    for e in FIXTURE["skipped_examples"]
    if e["node_count"] == 8
)


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _results(tmp_path: Path) -> list[dict]:
    path = tmp_path / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _recorded_world(tmp_path: Path, gateway: Gateway, *, enrolled=(OFFLINE, ONLINE)):
    recipes = [
        *(FakeRecipe(slug, local="cached", memory=20 * GB) for slug in SINGLES),
        *(FakeRecipe(slug, node_count=2, local="cached") for slug in DUALS),
        FakeRecipe(EIGHT, node_count=8, local="cached"),
    ]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        gateway=gateway,
        only=(*SINGLES, *DUALS, EIGHT),
    )
    fleet.sparks = [(n["id"], n["display_name"]) for n in enrolled]
    if OFFLINE in enrolled:
        fleet.offline_nodes = {OFFLINE["id"]}
        fleet.offline_facts = {OFFLINE["id"]: dict(OFFLINE["connection"])}
        selector = FIXTURE["deferred"]["recipe"]
        fleet.recipes[selector] = FakeRecipe(
            selector.partition("/")[2], alias=ORPHAN["alias"]
        )
        fleet.runs.append(
            {
                "alias": ORPHAN["alias"],
                "recipe": selector,
                "run_id": ORPHAN["run_id"],
                "node_ids": [OFFLINE["id"]],
                "ready": False,  # starting: the application failed, the run did not end
            }
        )
        sweep.state.data["own_aliases"] = [ORPHAN["alias"]]
    # The recorded verdicts, as the sweep wrote them at 21:55:45Z.
    for example in FIXTURE["skipped_examples"]:
        sweep.state.entry(example["recipe"]).update(
            status="skipped",
            reason=f"not testable on this fleet: needs {example['node_count']} Sparks, 1 usable",
            reason_kind="fleet",
        )
    sweep.state.save()
    return sweep, fleet, clock


def test_the_recorded_offline_spark_is_a_wait_and_its_lane_returns(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _recorded_world(tmp_path, gateway)
    seen: list[dict] = []

    def watch(_now: float) -> None:
        if not sweep.catalog_loaded:
            return
        statuses = {k: e.get("status") for k, e in sweep.state.recipes.items()}
        offline = OFFLINE["id"] in fleet.offline_nodes
        if offline:
            seen.append({"statuses": statuses, "status": build_status(sweep)})
        singles_done = all(statuses.get(f"vonk-forge/{s}") == "passed" for s in SINGLES)
        if offline and singles_done:
            fleet.offline_nodes.clear()  # the Spark answers again

    clock.hooks.append(watch)
    assert sweep.run() == 0

    entries = sweep.state.recipes
    # The online lane carried on while the duals waited; the duals ran once the lane was back.
    assert all(
        entries[f"vonk-forge/{s}"]["status"] == "passed" for s in SINGLES + DUALS
    )
    assert not any(run["run_id"] == ORPHAN["run_id"] for run in fleet.runs)
    # The topology verdict stays a verdict, now about the enrolled fleet.
    eight = entries[f"vonk-forge/{EIGHT}"]
    assert eight["status"] == "skipped"
    assert eight["reason"] == "not testable on this fleet: needs 8 Sparks, 2 enrolled"

    # While spark-3542 was offline: never skipped, its lane shown with the Controller's reason.
    assert seen
    for observed in seen:
        assert all(observed["statuses"][f"vonk-forge/{d}"] != "skipped" for d in DUALS)
        assert set(observed["status"]["lanes"]) == {"spark-3542", "spark-2297"}
    page = render_status(seen[-1]["status"])
    assert "spark-3542: offline (stale, last seen 2026-10-13T21:53:10" in page

    rows = _results(tmp_path)
    healed = [
        r
        for r in rows
        if r.get("step") == "reconcile"
        and r.get("status") == "pending"
        and r["recipe"].endswith(DUALS[0])
    ]
    assert healed and "waiting for 1 more lane" in healed[0]["reason"]
    assert "controller_document" in healed[0]
    assert not [
        r
        for r in rows
        if r.get("status") == "skipped" and r["recipe"].partition("/")[2] in DUALS
    ]
    lane = [r for r in rows if r.get("step") == "lane"]
    assert [(r["spark"], r["status"]) for r in lane] == [
        ("spark-3542", "waiting"),
        ("spark-3542", "available"),
    ]
    assert "stale" in json.dumps(lane[0]["controller_document"])


def test_the_status_says_how_many_lanes_a_queued_recipe_waits_for(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = _recorded_world(tmp_path, gateway)
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.heal_fleet_skips()
    sweep.cached_models = sweep._cached_models()
    sweep.queue = [f"vonk-forge/{k}" for k in (*DUALS, *SINGLES)]
    page = render_status(build_status(sweep))
    assert f"vonk-forge/{DUALS[0]} (waiting for 1 more lane)" in page
    assert sweep.state.recipes[f"vonk-forge/{DUALS[0]}"]["status"] == "pending"
    # The online lane is offered the singles; the duals wait for theirs.
    assert [r.key for r in sweep.ready_candidates()] == [
        f"vonk-forge/{k}" for k in SINGLES
    ]


def test_a_spark_that_is_not_enrolled_is_a_topology_verdict(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, _, _ = _recorded_world(tmp_path, gateway, enrolled=(ONLINE,))
    assert sweep.run() == 0
    entries = sweep.state.recipes
    assert all(entries[f"vonk-forge/{s}"]["status"] == "passed" for s in SINGLES)
    for slug in DUALS:
        entry = entries[f"vonk-forge/{slug}"]
        assert entry["status"] == "skipped"
        assert (
            entry["reason"] == "not testable on this fleet: needs 2 Sparks, 1 enrolled"
        )
    skips = [
        r
        for r in _results(tmp_path)
        if r.get("status") == "skipped" and r.get("step") == "reconcile"
    ]
    assert skips and all("controller_document" in r for r in skips)
