#!/usr/bin/env python3
"""Build-time scan: the CUDA code a file carries must load on a GB10 (sm_121).

Usage: scan-sm121.py [--require FILE]... [FILE]...
Each file's `cuobjdump --list-elf` and `--list-ptx` output (stdout and stderr) is echoed. Rules
(docs/recipe-authoring.md):
  * cubins load when they are sm_121, sm_121a or sm_121f, or plain/family sm_120 or sm_120f; an arch-specific
    sm_120a cubin does not load on sm_121, and no other target loads;
  * PTX loads only when it is not arch-specific and targets 121 or lower (compute_XY or sm_XY with XY <= 121), or is
    sm_121a / compute_121a PTX;
  * any file with device code needs at least one loadable image;
  * a file with no device code (cuobjdump says so) passes unless it is a --require file;
  * any other cuobjdump failure fails the scan.
"""

from __future__ import annotations

import re
import subprocess
import sys

ARCH = re.compile(r"\b(?:sm|compute)_(\d+)([af]?)\b")
NO_DEVICE_CODE = "does not contain device code"


def images(listing: str) -> list[tuple[int, str]]:
    return [(int(m[1]), m[2]) for m in ARCH.finditer(listing)]


def loadable_cubin(number: int, suffix: str) -> bool:
    return (number == 121 and suffix in {"", "a", "f"}) or (
        number == 120 and suffix in {"", "f"}
    )


def loadable_ptx(number: int, suffix: str) -> bool:
    return (suffix == "" and number <= 121) or (number == 121 and suffix == "a")


def verdict(elf: str, ptx: str, *, required: bool = False) -> tuple[bool, str]:
    cubins, ptxs = images(elf), images(ptx)
    if not cubins and not ptxs:
        return (not required), "no device code"
    if any(loadable_cubin(*i) for i in cubins) or any(loadable_ptx(*i) for i in ptxs):
        return True, "loadable image(s)"
    return False, "device code but no image loads on sm_121"


def interpret(stdout: str, stderr: str, code: int) -> tuple[str, str | None]:
    """The listing, or the error when cuobjdump failed for any reason but missing device code."""
    if NO_DEVICE_CODE in stdout + stderr:
        return "", None
    if code != 0:
        return "", f"cuobjdump exited {code}: {stderr.strip() or stdout.strip()}"
    return stdout, None


def listing(command: str, path: str) -> tuple[str, str, str | None]:
    done = subprocess.run(
        ["cuobjdump", command, path], capture_output=True, text=True, check=False
    )
    text, error = interpret(done.stdout, done.stderr, done.returncode)
    return text, done.stderr, error


def main(arguments: list[str]) -> int:
    files: list[tuple[str, bool]] = []
    it = iter(arguments)
    for argument in it:
        if argument == "--require":
            files.append((next(it), True))
        else:
            files.append((argument, False))
    failed = False
    for path, required in files:
        elf, elf_err, elf_fail = listing("--list-elf", path)
        ptx, ptx_err, ptx_fail = listing("--list-ptx", path)
        error = elf_fail or ptx_fail
        ok, why = (False, error) if error else verdict(elf, ptx, required=required)
        print(f"== {path}: {'ok' if ok else 'FAIL'} ({why})")
        for line in (elf + ptx + elf_err + ptx_err).splitlines():
            if line.strip():
                print(f"   {line.strip()}")
        failed = failed or not ok
    print("sm_121 kernel scan " + ("FAILED" if failed else "passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
