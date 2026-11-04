"""The Bolt image's sm_121 kernel scan accepts exactly the code a GB10 loads."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "adapters/nemotron/mia-bolt-tensorfold-single/scan-sm121.py"
)
spec = importlib.util.spec_from_file_location("scan_sm121", SCRIPT)
assert spec and spec.loader
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def elf(*names: str) -> str:
    return "\n".join(f"ELF file    {i + 1}: x.{n}.cubin" for i, n in enumerate(names))


def ptx(*names: str) -> str:
    return "\n".join(f"PTX file    {i + 1}: x.{n}.ptx" for i, n in enumerate(names))


@pytest.mark.parametrize(
    ("listing_elf", "listing_ptx", "passes"),
    [
        (elf("sm_121a"), "", True),
        (elf("sm_121"), "", True),
        (elf("sm_121f"), "", True),
        (elf("sm_120f"), "", True),
        (elf("sm_120"), "", True),
        (elf("sm_120a"), "", False),
        ("", ptx("sm_120a"), False),
        (elf("sm_120a"), ptx("sm_120a"), False),
        (elf("sm_120a"), ptx("compute_121"), True),
        ("", ptx("sm_121a"), True),
        (elf("sm_90"), "", False),
        (elf("sm_80", "sm_86"), "", False),
        (elf("sm_100a"), "", False),
        ("", ptx("compute_90"), True),
        (elf("sm_120a", "sm_121a"), "", True),
        ("", ptx("compute_122"), False),
        (elf("sm_122"), "", False),
        (elf("sm_125a"), "", False),
    ],
)
def test_verdicts(listing_elf: str, listing_ptx: str, passes: bool) -> None:
    ok, _ = scan.verdict(listing_elf, listing_ptx)
    assert ok is passes


def test_other_targets_are_not_loadable_even_when_required() -> None:
    ok, why = scan.verdict(elf("sm_90"), "", required=True)
    assert not ok and "no image loads" in why


def test_a_file_without_device_code_is_reported_and_required_ones_fail() -> None:
    assert scan.verdict("", "") == (True, "no device code")
    assert scan.verdict("", "", required=True) == (False, "no device code")


def test_cuobjdump_errors_are_failures_but_missing_device_code_is_not() -> None:
    message = "cuobjdump info    : File 'x' does not contain device code"
    assert scan.interpret("", message, 0) == ("", None)
    assert scan.interpret("", message, 1) == ("", None)
    assert scan.interpret("ELF file 1: x.sm_121.cubin", "", 0)[1] is None
    text, error = scan.interpret("", "cuobjdump fatal : Unrecognized file", 1)
    assert text == "" and error and "exited 1" in error


def test_every_adapter_copy_of_the_scan_is_this_one() -> None:
    copies = sorted(SCRIPT.parents[2].glob("*/*/scan-sm121.py"))
    assert SCRIPT in copies
    for copy in copies:
        assert copy.read_bytes() == SCRIPT.read_bytes(), copy


def test_the_two_mia_tensorfold_dual_adapters_are_byte_identical() -> None:
    root = SCRIPT.parents[2] / "adapters/qwen"
    first = root / "mia-flash-next-tensorfold-dual-int4-autoround"
    second = root / "mia-flash-next-tensorfold-dual-nvfp4"
    for name in ("Dockerfile", "kit-final-stage.txt", "scan-sm121.py"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
