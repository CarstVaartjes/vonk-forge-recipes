"""A cleanup the Controller ended with unconfirmed stops retires once its targets are gone.

Recorded on the hardware sweep (2026-10-16): 8 old cleanup records whose Controller
receipt was terminal with stops unconfirmed were never observed again, so they kept
spark-2297 from placement for as long as a workload the sweep did not own ran there.
A terminal receipt is final; current fleet facts retire the record, and only fresh,
complete, online facts that show the stop targets gone do.
"""

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from vonk_forge_contracts.sweep import CleanupEnd, CleanupEndReason, ReviewedCleanupStop

from spark_sweep import policy
from spark_sweep.run import fetch_fleet

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

OLD_RUN = {
    "alias": "old",
    "recipe": "external",
    "run_id": "old-run",
    "node_ids": ["spk_a"],
    "ready": True,
}


def scenario(tmp_path: Path):
    recipe = FakeRecipe("ready", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.data["cleanup_loads"]["old-request"] = {
        "request_key": "old-request",
        "app_id": "old-load",
        "state": "failed",
        "terminal_receipt": {
            "id": "old-load",
            "request_key": "old-request",
            "state": "failed",
        },
        "reviewed_stops": [
            ReviewedCleanupStop(run_id="old-run", node_ids=["spk_a"]).model_dump_json()
        ],
    }
    sweep.state.data["took_over"] = True
    fleet.runs.append(dict(OLD_RUN))
    return sweep, fleet, clock, recipe, sweep.state.data["cleanup_loads"]["old-request"]


@pytest.mark.parametrize("projection_known", ["unset", None, False])
def test_a_terminal_cleanup_retires_once_its_targets_are_gone(
    tmp_path: Path, projection_known
) -> None:
    sweep, fleet, clock, recipe, record = scenario(tmp_path)
    if projection_known != "unset":
        record["projection_known"] = projection_known
    sweep.observe_cleanup()
    assert "end" not in record
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    assert "spk_a" in sweep._cleanup_nodes()

    fleet.runs.clear()
    sweep.observe_cleanup()  # within its backoff: no look yet
    assert "end" not in record
    clock.sleep(record["observation_budget"]["next_check"] - clock.now())
    sweep.observe_cleanup()
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        == CleanupEndReason.TARGET_ABSENT
    )
    assert record["terminal_receipt"]["state"] == "failed"  # the receipt is kept
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    assert sweep._cleanup_nodes() == set()
    assert sweep._take_over(clock.now())
    sweep._apply([policy.Placement(sweep.recipes[recipe.key], ("spk_a",))], clock.now())
    assert recipe.key in sweep.state.slots


def test_an_offline_target_node_never_retires_a_terminal_cleanup(
    tmp_path: Path,
) -> None:
    sweep, fleet, clock, _recipe, record = scenario(tmp_path)
    fleet.runs.clear()
    fleet.offline_nodes = {"spk_a"}
    for _ in range(3):
        sweep.observe_cleanup()
        assert "end" not in record
        clock.sleep(record["observation_budget"]["next_check"] - clock.now())
    fleet.offline_nodes = set()
    sweep.observe_cleanup()
    assert "end" in record


@pytest.mark.parametrize(
    "no_targets", [{"reviewed_stops": []}, {"reviewed_stops": None}, {}]
)
def test_a_terminal_cleanup_without_targets_retires_beside_a_foreign_workload(
    tmp_path: Path, no_targets
) -> None:
    """Recorded on the live sweep (2026-10-16): 13 of 52 unresolved terminal records
    named no stop target and waited for a fleet with no runs, which never comes."""

    sweep, fleet, clock, _recipe, record = scenario(tmp_path)
    record.pop("reviewed_stops")
    record.update(no_targets)
    # A workload that is not the sweep's runs on spk_a, and spk_b is offline.
    fleet.offline_nodes = {"spk_b"}
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    assert "spk_a" in sweep._cleanup_nodes()
    sweep.observe_cleanup()
    assert "end" not in record  # incomplete facts never invent absence

    fleet.offline_nodes = set()
    clock.sleep(record["observation_budget"]["next_check"] - clock.now())
    sweep.observe_cleanup()
    assert (
        CleanupEnd.model_validate(record["end"]).reason
        == CleanupEndReason.TARGET_ABSENT
    )
    assert any(run["run_id"] == "old-run" for run in fleet.runs)
    sweep.fleet = fetch_fleet(sweep.vk, sweep.cfg.default_spark_memory)
    assert sweep._cleanup_nodes() == set()
