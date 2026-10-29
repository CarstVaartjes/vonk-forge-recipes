"""The production deadlock: assets ready, fleet fit blocked by someone else's workload.

The sweep once treated the library's cache state, fit or readiness as "download finished". While
the owner's workload filled the Sparks nothing could ever look ready, downloads were requested
again and again, and both lanes stayed idle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, Gateway, make_sweep

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

OWNER = {
    "alias": "owner-glm",
    "recipe": "x",
    "run_id": "run-owner",
    "node_ids": ["spk_a", "spk_b"],
    "ready": True,
}


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def _downloads(fleet) -> list[str]:
    return [c[2] for _, c in fleet.calls if c[:2] == ("recipe", "download")]


def test_a_finished_download_stands_even_when_the_library_still_says_not_cached(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", ("m1",)), FakeRecipe("b", ("m2",))]
    sweep, fleet, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1"), FakeModel("m2")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.library_lag = True  # the listing never catches up
    fleet.assessments = True  # and its advisory fit/readiness say blocked while the owner's workload runs
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert sorted(_downloads(fleet)) == [
        "vonk-forge/a",
        "vonk-forge/b",
    ]  # each asked for exactly once
    assert all(
        r["alias"] != "owner-glm" for r in fleet.runs
    )  # the sweep took the fleet over to test


def test_assets_the_library_assessment_calls_ready_need_no_download(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a", ("m1",), local="cached")]
    sweep, fleet, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1", local="cached")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    fleet.library_lag = True
    fleet.assessments = True
    assert sweep.run() == 0
    assert sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    assert _downloads(fleet) == []


def test_the_sweep_takes_over_at_the_start_not_at_the_first_placement(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a")], [FakeModel("m1")], gateway=gateway
    )
    fleet.runs.append(dict(OWNER))
    start = clock.now()
    sweep.run()
    first = next(
        t
        for t, (p, c) in zip(fleet.call_times, fleet.calls, strict=True)
        if p == 10 and c[:3] == ("profile", "load", "--yes")
    )
    assert first == start  # before the first download had even finished


def test_a_review_blocked_by_a_workload_that_is_back_clears_the_fleet_instead_of_failing_the_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, clock = make_sweep(
        tmp_path, [FakeRecipe("a", memory=100 * GB)], [FakeModel("m1")], gateway=gateway
    )
    brought_back = []

    def owner_returns(
        now: float,
    ) -> None:  # a foreign workload reappears right after the takeover
        if sweep.state.data.get("took_over") and not brought_back and not fleet.runs:
            fleet.runs.append(dict(OWNER, memory=120 * GB))
            brought_back.append(now)

    clock.hooks.append(owner_returns)
    real = fleet._load

    def blocking_load(n, data, a):  # the review sees the foreign 120 GB and refuses
        if (
            "--review" in a
            and any(i["desired_state"] == "running" for i in data["assignments"])
            and any(r["alias"] == "owner-glm" for r in fleet.runs)
        ):
            return 2, {
                "allowed": False,
                "effects": {"runs": []},
                "admission_decisions": [
                    {
                        "alias": "a",
                        "allowed": False,
                        "blockers": [
                            {
                                "code": "run-switch.resource.insufficient_capacity",
                                "detail": "memory",
                            }
                        ],
                    }
                ],
            }
        return real(n, data, a)

    fleet._load = blocking_load
    assert sweep.run() == 0
    assert (
        sweep.state.recipes["vonk-forge/a"]["status"] == "passed"
    )  # not a `review` failure


def test_a_state_file_with_finished_downloads_resumes_without_asking_again(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [FakeRecipe("a1", ("m1",)), FakeRecipe("a2", ("m1",), engine="sglang")]
    first, fleet, _ = make_sweep(
        tmp_path, recipes, [FakeModel("m1")], gateway=gateway, limit=0
    )
    from spark_sweep.identity import AcceptedRequest

    first.refresh_catalog()
    first.state.downloads["vonk-forge/a1"] = {
        "accepted_content": AcceptedRequest(
            request_key="old-request", recipe=first.recipes["vonk-forge/a1"]
        ).model_dump(mode="json"),
        "state": "succeeded",
        "kind": "model",
        "attempt": 6,
        "done_at": fleet.clock.now(),
        "operation_id": "old",
    }
    fleet.library_lag = True  # and the library still says not cached, as in production
    first.state.save()
    sweep, _, _ = make_sweep(tmp_path, recipes, [FakeModel("m1")], fleet=fleet)
    assert sweep.run() == 0
    assert {e["status"] for e in sweep.state.recipes.values()} == {"passed"}
    assert _downloads(fleet) == [
        "vonk-forge/a2"
    ]  # the sibling pulled its image; a1 was not asked for again


SWAP_REFUSAL = {
    "allowed": False,
    "effects": {"runs": []},
    "admission_decisions": [
        {
            "alias": "a",
            "allowed": False,
            "blockers": [
                {
                    "code": "run-switch.resource.insufficient_swap",
                    "detail": "swap",
                },
                {"code": "run-switch.run_admission_blocked", "detail": "blocked"},
                {"code": "run.insufficient_swap", "detail": "swap"},
            ],
        }
    ],
}


def test_a_swap_shortfall_in_a_review_waits_instead_of_failing_the_recipe(
    tmp_path: Path, gateway: Gateway
) -> None:
    sweep, fleet, _clock = make_sweep(
        tmp_path, [FakeRecipe("a", memory=100 * GB)], [FakeModel("m1")], gateway=gateway
    )
    real = fleet._load
    refusals: list[tuple[str, ...]] = []

    def short_of_swap(n, data, a):
        if (
            "--review" in a
            and any(i["desired_state"] == "running" for i in data["assignments"])
            and len(refusals) < 2
        ):
            refusals.append(a)
            return 2, SWAP_REFUSAL
        return real(n, data, a)

    fleet._load = short_of_swap
    assert sweep.run() == 0
    entry = sweep.state.recipes["vonk-forge/a"]
    assert entry["status"] == "passed"
    assert entry.get("attempts", 0) == 1  # only the real attempt
    assert len(refusals) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    waits = [r for r in rows if r.get("status") == "waiting"]
    assert len(waits) >= 2 and {r.get("disposition") for r in waits} == {"wait"}
    assert not any(r.get("status") == "failed" for r in rows)
    # no fleet clearing was requested for it
    assert not entry.get("review_freeing")
    assert not any("clearing the fleet" in str(r.get("reason")) for r in rows)


def test_a_pass_records_the_swap_each_spark_reported_and_the_declared_values(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipe = FakeRecipe("a", memory=100 * GB, load_peak=110 * GB, minimum_swap=8 * GB)
    sweep, fleet, _clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1")], gateway=gateway
    )
    fleet.swap = {node_id: (64 * GB, 62 * GB) for node_id, _ in fleet.sparks[:1]}
    assert sweep.run() == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    memory = next(r["memory"] for r in rows if r.get("status") == "passed")
    assert memory["load_peak_bytes"] == 110 * GB
    assert memory["minimum_swap_bytes"] == 8 * GB
    reported = [n for n in memory["nodes"] if n["swap_total_bytes"] is not None]
    assert [(n["swap_total_bytes"], n["swap_used_bytes"]) for n in reported] == [
        (64 * GB, 2 * GB)
    ]
    assert all(
        n["swap_total_bytes"] is None and n["swap_used_bytes"] is None
        for n in memory["nodes"]
        if n not in reported
    )
