"""The producer job's scope: member PRs run lean, everything else runs it all."""

import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "select_producer_checks", str(ROOT / "tools/select-producer-checks")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
selector = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(selector)

REFRESH_LOADER = importlib.machinery.SourceFileLoader(
    "refresh_upstream_branch_prefix", str(ROOT / "tools/refresh-upstream")
)
REFRESH_SPEC = importlib.util.spec_from_loader(REFRESH_LOADER.name, REFRESH_LOADER)
assert REFRESH_SPEC is not None
refresher = importlib.util.module_from_spec(REFRESH_SPEC)
sys.modules[REFRESH_LOADER.name] = refresher
REFRESH_LOADER.exec_module(refresher)

# Import and reference lines copied from real tests.
TESTS = {
    "tests/test_sweep_blame.py": "from spark_sweep import policy, verdict\n",
    "tests/test_sweep_tick_fanout.py": "from spark_sweep.fanout import gather\n",
    "tests/test_kimodo_adapter.py": 'ADAPTER = ROOT / "adapters/three-d/kimodo"\n',
    "tests/test_qualification_status.py": (
        '    for path in (ROOT / "qualification/evidence").rglob("*.json"):\n'
    ),
    "tests/test_deepseek_ds4_recipe.py": (
        'RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"\n'
    ),
}


def test_refresher_branches_run_the_full_suite() -> None:
    assert refresher.BRANCH_PREFIX in selector.FULL_SUITE_HEADS


def chosen(*changed: str) -> set[str]:
    return set(selector.targeted(changed, TESTS)) - set(selector.CORE)


@pytest.mark.parametrize(
    ("event", "head_ref"),
    [
        ("pull_request", "integration/recipes-batch-8"),
        ("pull_request", "refresh/qwen3-6-35b-a3b-fp8-eugr-vllm-single"),
        ("push", ""),
    ],
)
def test_batches_refresher_prs_and_pushes_run_the_full_suite(event, head_ref) -> None:
    scope = selector.select(event, head_ref, ["spark_sweep/fanout.py"])
    assert scope["tests"] == selector.FULL_SUITE


def test_a_member_pr_with_an_unknown_change_runs_the_full_suite() -> None:
    assert selector.select("pull_request", "fix-x", None) == {
        "gates": "full",
        "tests": selector.FULL_SUITE,
    }


def test_hardware_evidence_lands_without_a_batch_so_it_runs_the_full_suite() -> None:
    changed = [
        "qualification/evidence/vonk-forge/qwen3-6-35b-a3b-nvfp4-eugr-vllm-dual.json"
    ]
    scope = selector.select("pull_request", "qualification/evidence", changed)
    assert scope["tests"] == selector.FULL_SUITE


def test_a_member_pr_runs_the_core_plus_the_tests_its_paths_name() -> None:
    scope = selector.select("pull_request", "fix-x", ["tests/test_sweep_cli.py"])
    tests = scope["tests"].split()
    assert set(selector.CORE) <= set(tests)
    assert "tests/test_sweep_cli.py" in tests
    assert len(tests) == len(selector.CORE) + 1
    assert all((ROOT / test).is_file() for test in tests)


def test_changed_paths_select_the_tests_that_name_them() -> None:
    assert chosen("spark_sweep/verdict.py") == {"tests/test_sweep_blame.py"}
    assert chosen("spark_sweep/fanout.py") == {"tests/test_sweep_tick_fanout.py"}
    assert chosen("adapters/three-d/kimodo/run.py") == {"tests/test_kimodo_adapter.py"}
    assert chosen("qualification/evidence/mia/x-vllm-single.json") == {
        "tests/test_qualification_status.py"
    }
    assert chosen("recipes/deepseek-v4-flash-0731-ds4-single.json") == {
        "tests/test_deepseek_ds4_recipe.py"
    }
    assert chosen("spark_sweep/state.py", "recipes/other.json") == set()


def test_only_recipe_data_skips_the_python_gates() -> None:
    assert selector.gates(["recipes/a.json", "adapters/x/y/run.py"]) == "light"
    assert selector.gates(["recipes/a.json", "tools/kit-pins"]) == "full"


def test_the_producer_job_reads_the_selector_for_its_scope() -> None:
    """One owner: no step re-derives the scope from the branch or the paths."""

    workflow = (ROOT / ".github/workflows/validate.yml").read_text(encoding="utf-8")
    assert "python3 tools/select-producer-checks" in workflow
    assert "TESTS: ${{ steps.scope.outputs.tests }}" in workflow
    assert re.search(r"-m pytest .* \$TESTS$", workflow, re.MULTILINE)
    assert "integration/" not in workflow
