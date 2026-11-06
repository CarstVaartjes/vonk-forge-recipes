"""The shared added-lines parser."""

from __future__ import annotations

import sys
from pathlib import Path

from added_diff import added_diff

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools._added_lines import parse_added

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
