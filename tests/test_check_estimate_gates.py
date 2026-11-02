"""The guard that keeps tests from gating on a real recipe's declared memory."""

import importlib.machinery
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.machinery.SourceFileLoader(
    "check_estimate_gates", str(ROOT / "tools/check-estimate-gates")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
gates = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(gates)

LOADS = """
def load():
    return documents.recipe("x")

def test_a():
    recipe = load()
    {body}
"""


def hits(body: str) -> int:
    return len(gates.scan(LOADS.format(body=body), None))


def test_a_gate_on_a_real_recipe_is_rejected() -> None:
    assert hits("assert_not_warns(memory_estimate_warnings(recipe), 'x')") == 1
    assert hits("assert memory_estimate_warnings(recipe) == []") == 1


def test_assigning_the_peak_first_is_accepted() -> None:
    body = (
        "recipe.topology.roles[0].resources.memory.peak_bytes = 1\n"
        "    assert memory_estimate_warnings(recipe) == []"
    )
    assert hits(body) == 0
    assert (
        hits("declare_peak(recipe)\n    assert_not_warns(f_warnings(recipe), 'x')") == 0
    )


def test_a_warning_assertion_is_not_a_gate() -> None:
    assert hits("assert_warns(memory_estimate_warnings(recipe), 'x')") == 0


def test_only_added_lines_are_judged() -> None:
    source = LOADS.format(body="assert memory_estimate_warnings(recipe) == []")
    assert gates.scan(source, {1}) == []


def test_the_whole_suite_is_clean() -> None:
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        assert gates.scan(path.read_text(encoding="utf-8"), None) == [], path.name
