"""The shared added-lines parser, and the workflow step that writes its input."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from added_diff import added_diff

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools._added_lines import ADDED_PATHSPEC, parse_added

DIFF = """\
diff --git a/tools/a.py b/tools/a.py
--- a/tools/a.py
+++ b/tools/a.py
@@ -3 +3,2 @@
-old
+new
+newer
@@ -10,2 +11,0 @@
-gone
-gone
diff --git a/tests/b.py b/tests/b.py
new file mode 100644
--- /dev/null
+++ b/tests/b.py
@@ -0,0 +1 @@
+x
diff --git a/tools/deleted.py b/tools/deleted.py
--- a/tools/deleted.py
+++ /dev/null
@@ -1 +0,0 @@
-x
"""


def test_added_line_numbers_per_file() -> None:
    assert parse_added(DIFF) == {"tools/a.py": {3, 4}, "tests/b.py": {1}}


def test_an_empty_diff_adds_nothing() -> None:
    assert parse_added("") == {}


def test_the_test_helper_round_trips() -> None:
    diff = added_diff({"a.py": "1\n2\n3\n"}, old_lines={"a.py": 1})
    assert parse_added(diff) == {"a.py": {2, 3}}


def test_the_workflow_step_uses_the_shared_pathspec() -> None:
    workflow = (ROOT / ".github/workflows/validate.yml").read_text()
    step = workflow.split("- name: Write the added-lines diff", 1)[1]
    command = re.search(
        r"git diff[^>]*>\s*\"\$RUNNER_TEMP/added\.diff\"", step, re.DOTALL
    )
    assert command is not None
    pathspec = command[0].split(" -- ", 1)[1].split(">")[0].replace("\\", " ")
    assert tuple(pathspec.replace("'", "").split()) == ADDED_PATHSPEC
