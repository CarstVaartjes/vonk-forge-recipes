"""A clearing the sweep decides on is a load it submits, or a refusal it records.

Recorded on the hardware sweep (2026-10-13): the clearing budget had been spent days
earlier and was never given back, so every later clearing decision returned at once
without a load. The sweep profile still held three assignments no lane accounted for
(one of them running: a 115 GB TensorRT-LLM workload the previous stop load had started).
Each review was blocked by that workload, the sweep said it had "outlived a clearing"
that was never submitted, and both lanes idled with ready work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

from spark_sweep import policy

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "sweep-exhausted-clearing-2026-10-13.json"
    ).read_text(encoding="utf-8")
)
SWEPT = ("fastcontext", "hunyuanocr")
ORPHANS = {row["assignment_name"] for row in FIXTURE["profile_assignments"]}
LEFTOVER = FIXTURE["runs"][0]


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _recorded_world(tmp_path: Path, gateway: Gateway):
    recipes = [FakeRecipe(slug, local="cached", memory=20 * GB) for slug in SWEPT]
    sweep, fleet, clock = make_sweep(
        tmp_path,
        recipes,
        [FakeModel("m1", local="cached")],
        gateway=gateway,
        only=SWEPT,
    )
    for row in FIXTURE["profile_assignments"]:
        selector = row["recipe_selector"]
        fleet.recipes[selector] = FakeRecipe(
            selector.partition("/")[2],
            memory=FIXTURE["declared_memory_bytes"].get(selector, 20 * GB),
            alias=row["assignment_name"],
        )
    fleet.profiles[10] = {
        "revision": 1,
        "latest": None,
        "labels": {"purpose": "hardware-sweep"},
        "assignments": [dict(row) for row in FIXTURE["profile_assignments"]],
    }
    fleet.runs.extend(dict(run) for run in FIXTURE["runs"])
    sweep.state.data.update(json.loads(json.dumps(FIXTURE["state"])))
    sweep.state.save()
    return sweep, fleet, clock


def _results(tmp_path: Path) -> list[dict]:
    path = tmp_path / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _blocked_by_leftover(sweep, clock) -> None:
    recipe = sweep.recipes[f"vonk-forge/{SWEPT[0]}"]
    sweep._blocked({}, {}, [policy.Placement(recipe, ("spk_a",))], clock.now())


def test_the_recorded_state_sweeps_on_without_restarting_what_no_lane_needs(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _recorded_world(tmp_path, gateway)
    started: set[str] = set()
    alerts: list[str] = []

    def watch(_now: float) -> None:
        started.update(run["alias"] for run in fleet.runs)
        alerts.extend(sweep.state.data.get("alerts", []))

    clock.hooks.append(watch)
    assert sweep.run() == 0

    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert not any(run["alias"] in ORPHANS for run in fleet.runs)
    assert not ((ORPHANS - {LEFTOVER["alias"]}) & started)
    assert (
        not {a["assignment_name"] for a in fleet.profiles[10]["assignments"]} & ORPHANS
    )
    assert not [a for a in alerts if a.startswith("IDLE WITH READY WORK")]
    assert not any("outlived" in str(r.get("reason")) for r in _results(tmp_path))


def test_a_clearing_decision_submits_a_cleanup_after_an_old_budget_was_spent(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _recorded_world(tmp_path, gateway)
    sweep.preflight()
    sweep.refresh_catalog()
    # The recorded decision (cleared_for == the holders) was never submitted.
    _blocked_by_leftover(sweep, clock)
    waits = [r for r in _results(tmp_path) if r.get("status") == "waiting"]
    assert waits and "was not submitted" in waits[-1]["reason"]
    assert not any("outlived" in str(r.get("reason")) for r in _results(tmp_path))
    assert not sweep.state.data.get("takeover_fallback")

    sweep._take_over(clock.now())

    takeovers = [i for i in sweep.state.data["own_loads"] if i["kind"] == "takeover"]
    assert len(takeovers) == 1
    app = fleet.apps[takeovers[0]["application_id"]]
    assert [row["target_id"] for row in app["effects"]] == [LEFTOVER["run_id"]]
    assert not app["assign"]  # nothing the profile no longer holds is started
    assert sweep.state.data["cleared_by"] == takeovers[0]["request_key"]


def test_a_refused_clearing_is_the_recorded_wait_and_is_decided_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = _recorded_world(tmp_path, gateway)
    sweep.preflight()
    sweep.refresh_catalog()
    original = fleet._load
    blocker = FIXTURE["review_blocker"]

    def refusing(n, data, a):
        if "--review" in a and not data["assignments"]:
            return 2, {
                "allowed": False,
                "effects": {"runs": [], "adopted": []},
                "admission_decisions": [],
                "reasons": [{"severity": "error", **blocker}],
            }
        return original(n, data, a)

    fleet._load = refusing
    _blocked_by_leftover(sweep, clock)
    assert sweep._take_over(clock.now())
    assert sweep.state.data["takeover_fallback"]
    assert not any(i["kind"] == "takeover" for i in sweep.state.data["own_loads"])
    reason = sweep.state.data["clearing_reason"]
    assert reason.startswith("clearing refused") and blocker["code"] in reason
    events = [e["message"] for e in sweep.state.data["events"]]
    assert sum(blocker["code"] in m for m in events) == 1

    # The next blocked review names the refusal and decides the clearing again.
    _blocked_by_leftover(sweep, clock)
    assert blocker["code"] in _results(tmp_path)[-1]["reason"]
    assert sweep.state.data["takeover_budget"]["attempts"] == 0
    fleet._load = original
    sweep._take_over(clock.now())
    assert any(i["kind"] == "takeover" for i in sweep.state.data["own_loads"])


# Same-key re-entry of a clearing whose reviewed stops were already recorded: it
# submits only when the fresh review equals the original intent, never a new one.
REVIEW_WITHOUT_PRUNE = {"observe_cleanup"}


def test_every_review_of_a_new_intent_reconciles_the_profile_first() -> None:
    """Guard for the class: no load of the sweep profile carries what no lane needs."""
    import ast

    source = (Path(__file__).parents[1] / "spark_sweep" / "run.py").read_text(
        encoding="utf-8"
    )
    missing = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or node.name in REVIEW_WITHOUT_PRUNE:
            continue
        reviews = [
            call.lineno
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and any(
                isinstance(arg, ast.Constant) and arg.value == "--review"
                for arg in call.args
            )
        ]
        prunes = [
            call.lineno
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_prune_profile"
        ]
        if reviews and not (prunes and min(prunes) < min(reviews)):
            missing.append(node.name)
    assert missing == []
