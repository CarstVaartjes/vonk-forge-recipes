"""A workload under a name the sweep derives is the sweep's, whatever its ledger kept.

On hardware a FLUX lane ran under the recipe-keyed name
``flux-2-klein-4b-nvfp4-comfyui-single-9c28c`` that the sweep gives a second
recipe sharing one served alias. The ledger of remembered names lacked it, so
the sweep read its own leftover workload as someone else's: the Spark stayed
occupied behind an old cleanup record and idled for hours with ready work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.profile_alias import unique_profile_alias
from spark_sweep.run import alias_for

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")


def test_a_leftover_under_the_sweeps_suffixed_name_is_its_own(tmp_path: Path) -> None:
    recipe = FakeRecipe("flux", local="cached")
    sweep, fleet, _clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    suffixed = unique_profile_alias(
        alias_for(sweep.recipes[recipe.key], sweep.defs), recipe.key
    )
    fleet.runs.extend(
        [
            {
                "alias": suffixed,
                "recipe": recipe.key,
                "run_id": "run-leftover",
                "node_ids": ["spk_b"],
                "ready": True,
            },
            {
                "alias": "owners-model",
                "recipe": "external",
                "run_id": "run-owner",
                "node_ids": ["spk_a"],
                "ready": True,
            },
        ]
    )
    assert suffixed not in sweep.state.data["own_aliases"]

    foreign, ours = sweep._fleet_runs()

    assert ours == [suffixed]
    assert foreign == ["owners-model"]
