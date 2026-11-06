"""The DeepSeek 1017 TensorFold adapters carry identical copies of their shared helpers.

A build context cannot share files, so each adapter holds its own copy; these tests keep the copies equal.
Functions are compared by AST (docstrings included), whole files by bytes. Only what is truly identical is
asserted: the dual-Spark Mia adapter differs in `prepare_rank` (no parity layout), and bertholomus differs in the rest."""

import ast
from pathlib import Path

import pytest

ADAPTERS = Path(__file__).resolve().parents[1] / "adapters/deepseek"
ALL = [
    "bertholomus-1017-tensorfold-tp4",
    "mia-1017-tensorfold-dual",
    "mia-1017-tensorfold-four",
    "tonyd2wild-1017-tensorfold-tp2",
]
PARITY_RANK = [name for name in ALL if name != "mia-1017-tensorfold-dual"]


def function(adapter: str, filename: str, name: str) -> str:
    tree = ast.parse((ADAPTERS / adapter / filename).read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(found) == 1, f"{adapter}/{filename} must define {name} once"
    return ast.dump(found[0])


@pytest.mark.parametrize(
    ("filename", "name", "adapters"),
    [
        ("launcher.py", "fabric", ALL),
        ("prepare_cache.py", "sync_dir", ALL),
        ("prepare_cache.py", "atomic_bytes", ALL),
        ("prepare_cache.py", "rank_cache_valid", ALL),
        ("prepare_cache.py", "prepare_rank", PARITY_RANK),
    ],
)
def test_shared_function_is_identical(
    filename: str, name: str, adapters: list[str]
) -> None:
    dumps = {adapter: function(adapter, filename, name) for adapter in adapters}
    assert len(set(dumps.values())) == 1, f"{name} differs between {sorted(dumps)}"


def test_tensorfold_serve_is_byte_identical() -> None:
    reference = (ADAPTERS / ALL[0] / "tensorfold-serve").read_bytes()
    for adapter in ALL:
        assert (ADAPTERS / adapter / "tensorfold-serve").read_bytes() == reference, (
            adapter
        )
