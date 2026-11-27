"""An added Model display licence id must be SPDX-shaped; old lines are not judged."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_display_licences", str(ROOT / "tools/check-display-licences")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)


@pytest.mark.parametrize(
    "value",
    [
        "Apache-2.0",
        "LicenseRef-OpenMDW-1.1",
        "NOASSERTION",
        "LicenseRef-A AND LicenseRef-B",
    ],
)
def test_spdx_ids_licence_refs_and_expressions_pass(value: str) -> None:
    assert guard.valid(value)


@pytest.mark.parametrize(
    "value",
    ["other", "apache-2.0", "LTX-2 Community License Agreement", "LicenseRef-a_b"],
)
def test_unknown_miscased_and_spaced_ids_fail(value: str) -> None:
    assert not guard.valid(value)


def test_only_added_display_model_lines_are_judged(tmp_path: Path) -> None:
    (tmp_path / "display/models").mkdir(parents=True)
    (tmp_path / "display/models/a.json").write_text(
        '{\n  "license": {\n    "spdx": "other",\n    "spdx": "lfm1.0"\n  }\n}\n'
    )
    (tmp_path / "display/recipes").mkdir(parents=True)
    (tmp_path / "display/recipes/b.json").write_text('    "spdx": "other"\n')
    diff = (
        '+++ b/display/models/a.json\n@@ -4,0 +4 @@\n+    "spdx": "lfm1.0"\n'
        '+++ b/display/recipes/b.json\n@@ -0,0 +1 @@\n+    "spdx": "other"\n'
    )
    problems = guard.check(tmp_path, diff)
    assert [p.split(":")[:2] for p in problems] == [["display/models/a.json", "4"]]
