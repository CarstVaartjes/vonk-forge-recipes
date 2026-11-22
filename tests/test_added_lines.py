"""The shared added-lines parser."""

from __future__ import annotations

from pathlib import Path

from added_diff import added_diff

ROOT = Path(__file__).resolve().parents[1]
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


BLOCK = ["def load(path):", "    text = path.read_text()", "    return parse(text)", ""]


def _move(
    old: str, new: str, block: list[str], *, deleted: bool = False, indent: str = ""
) -> str:
    """A diff that removes ``block`` from ``old`` and adds ``block`` to ``new``."""
    gone = "".join(f"-{row}\n" for row in block)
    came = "".join(f"+{indent}{row}\n" if row else "+\n" for row in block)
    target = "/dev/null" if deleted else f"b/{old}"
    return (
        f"diff --git a/{old} b/{old}\n--- a/{old}\n+++ {target}\n"
        f"@@ -5,{len(block)} +4,0 @@\n{gone}"
        f"diff --git a/{new} b/{new}\n--- /dev/null\n+++ b/{new}\n"
        f"@@ -0,0 +1,{len(block)} @@\n{came}"
    )


def test_a_verbatim_move_adds_nothing() -> None:
    assert parse_added(_move("tools/a.py", "tools/b.py", BLOCK))["tools/b.py"] == set()


def test_moving_out_of_a_deleted_file_adds_nothing() -> None:
    diff = _move("tools/a.py", "tools/b.py", BLOCK, deleted=True)
    assert parse_added(diff) == {"tools/b.py": set()}


def test_an_indented_move_adds_nothing() -> None:
    diff = _move("tools/a.py", "tools/b.py", BLOCK, indent="    ")
    assert parse_added(diff)["tools/b.py"] == set()


def test_an_edited_line_in_a_moved_block_is_still_added() -> None:
    block = [*BLOCK[:3], "    log(text)", "    return text", "    done()", ""]
    diff = _move("tools/a.py", "tools/b.py", block).replace(
        "+    done()", "+    json.loads(text)"
    )
    added = parse_added(diff)["tools/b.py"]
    assert 6 in added
    assert not added & {1, 2, 3, 4, 5}


def test_a_copy_spends_the_deleted_block_once() -> None:
    first = _move("tools/a.py", "tools/b.py", BLOCK)
    second = first.split("diff --git a/tools/b.py")[1]
    diff = first + "diff --git a/tools/b.py" + second.replace("b.py", "c.py")
    added = parse_added(diff)
    assert added["tools/b.py"] == set()
    assert added["tools/c.py"] == {1, 2, 3, 4}


def test_a_move_across_file_types_is_still_added() -> None:
    assert parse_added(_move("tools/a.py", "tools/b.txt", BLOCK))["tools/b.txt"]


def test_an_isolated_common_line_is_still_added() -> None:
    diff = _move("tools/a.py", "tools/b.py", ["return None"])
    assert parse_added(diff)["tools/b.py"] == {1}
