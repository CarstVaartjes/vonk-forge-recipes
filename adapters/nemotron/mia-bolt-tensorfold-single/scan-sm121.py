#!/usr/bin/env python3
"""Build-time scan: the CUDA code a file carries must load on a GB10 (sm_121).

Usage: scan-sm121.py [--require FILE]... [FILE]...
Each file's `cuobjdump --list-elf` and `--list-ptx` listing is echoed. Rules (docs/recipe-authoring.md):
  * cubins load when they are sm_121, sm_121a or sm_121f, or plain/family sm_120 or sm_120f; an arch-specific
    sm_120a cubin does not load on sm_121;
  * PTX loads only when it is not arch-specific and targets 121 or lower (compute_XY or sm_XY with XY <= 121), or is
    sm_121a / compute_121a PTX;
  * a file with any SM12x code needs at least one loadable image, or the scan fails;
  * a --require file must carry a loadable image at all (a file with no device code fails);
  * other files without device code are reported and pass.
"""

from __future__ import annotations

import re
import subprocess
import sys

ARCH = re.compile(r"\b(?:sm|compute)_(\d+)([af]?)\b")


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
    sm12x = [i for i in cubins + ptxs if 120 <= i[0] <= 129]
    ok = [i for i in cubins if loadable_cubin(*i)] + [
        i for i in ptxs if loadable_ptx(*i)
    ]
    if ok:
        return True, f"{len(ok)} loadable image(s)"
    if sm12x:
        return False, "SM12x code but no image loads on sm_121"
    return True, "no SM12x code (other targets only)"


def listing(command: str, path: str) -> str:
    done = subprocess.run(
        ["cuobjdump", command, path], capture_output=True, text=True, check=False
    )
    return done.stdout


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
        elf, ptx = listing("--list-elf", path), listing("--list-ptx", path)
        ok, why = verdict(elf, ptx, required=required)
        print(f"== {path}: {'ok' if ok else 'FAIL'} ({why})")
        for line in (elf + ptx).splitlines():
            if line.strip():
                print(f"   {line.strip()}")
        failed = failed or not ok
    print("sm_121 kernel scan " + ("FAILED" if failed else "passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
