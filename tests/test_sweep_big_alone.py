"""A large recipe runs alone: the sweep never loads two recipes past one Spark's memory.

Application fad7d2ec held FLUX.2 Klein (36e9 declared) and Gemma 4 26B Mia (103.4e9)
on spark-2297; with the reserve that is beyond one Spark, so the load looped for hours.
"""

from pathlib import Path

import pytest
from sweep_fakes import GB, FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

GEMMA = FakeRecipe("gemma", memory=103_400_000_000, local="cached")


def scenario(tmp_path: Path, beside: FakeRecipe):
    sweep, fleet, clock = make_sweep(
        tmp_path, [beside, GEMMA], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    assert sweep._take_over(clock.now())
    # An assignment already in the sweep profile on spark-a that no lane accounts for.
    fleet.profiles[sweep.cfg.sweep_profile]["assignments"].append(
        {
            "recipe_selector": f"vonk-forge/{beside.slug}",
            "spark_ids": ["spark-a"],
            "assignment_name": f"{beside.slug}-stray",
            "desired_state": "running",
        }
    )
    gemma = sweep.recipes["vonk-forge/gemma"]
    before = len(fleet.calls)
    sweep._apply([policy.Placement(gemma, ("spk_a",))], clock.now())
    return sweep, fleet, gemma, fleet.calls[before:]


def loads(sweep, calls, *, review: bool = False) -> list:
    return [
        command
        for profile, command in calls
        if profile == sweep.cfg.sweep_profile
        and command[:2] == ("profile", "load")
        and ("--review" in command) is review
    ]


def test_flux_and_gemma_never_share_a_spark(tmp_path: Path) -> None:
    flux = FakeRecipe("flux", memory=36 * GB, local="cached")
    sweep, fleet, gemma, calls = scenario(tmp_path, flux)
    assert gemma.key not in sweep.state.slots  # waits for the next free slot
    assert not loads(sweep, calls, review=True)  # never even proposed
    assert not loads(sweep, calls)
    names = {
        a["assignment_name"]
        for a in fleet.profiles[sweep.cfg.sweep_profile]["assignments"]
    }
    assert sweep._alias(gemma) not in names  # the edit is rolled back
    assert sweep._alias(gemma) in sweep.state.data["own_aliases"]
    assert sweep.state.status(gemma.key) != "failed"


def test_two_recipes_that_fit_together_still_share_a_spark(tmp_path: Path) -> None:
    small = FakeRecipe("small", memory=10 * GB, local="cached")
    sweep, _, gemma, calls = scenario(tmp_path, small)
    assert gemma.key in sweep.state.slots
    assert len(loads(sweep, calls)) == 1
