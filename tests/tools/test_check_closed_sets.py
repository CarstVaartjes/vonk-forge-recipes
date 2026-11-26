"""Every closed set lives in executed.py or shown.py, and shown sets are kept as words."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from repo_root import ROOT

PACKAGE = "contracts/src/vonk_forge_contracts"
SHOWN = 'from enum import StrEnum\n\n\nclass Color(StrEnum):\n    RED = "red"\n'
WORD = (
    "from typing import Annotated\n\n"
    "ColorWord = Annotated[Color | UnrecognisedWord, KeepUnrecognised]\n\n\n"
)
WRAPPED = WORD + "class Paint(Base):\n    color: ColorWord\n"


def run(tmp_path: Path, **modules: str) -> subprocess.CompletedProcess[str]:
    package = tmp_path / PACKAGE
    package.mkdir(parents=True)
    (package / "shown.py").write_text(SHOWN)
    for name, source in modules.items():
        (package / f"{name}.py").parent.mkdir(exist_ok=True)
        (package / f"{name}.py").write_text(source)
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/check-closed-sets"),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_repository_keeps_its_sets_in_the_two_modules() -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/check-closed-sets")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout


def test_an_enum_outside_the_two_modules_exits_one(tmp_path: Path) -> None:
    done = run(tmp_path, display=SHOWN.replace("Color", "Shade"))
    assert done.returncode == 1
    assert "display.py" in done.stdout


def test_an_alias_or_an_inline_multi_value_literal_exits_one(tmp_path: Path) -> None:
    alias = run(tmp_path / "a", recipe='Kind = Literal["a", "b"]\n')
    inline = run(
        tmp_path / "b", recipe='class Row(Base):\n    kind: Literal["a", "b"]\n'
    )
    single = run(tmp_path / "c", recipe='class Row(Base):\n    kind: Literal["a"]\n')
    assert (alias.returncode, inline.returncode, single.returncode) == (1, 1, 0)


def test_a_shown_set_must_be_kept_as_an_unrecognised_word(tmp_path: Path) -> None:
    bare = run(tmp_path / "a", recipe="class Paint(Base):\n    color: Color\n")
    kept = run(tmp_path / "b", recipe=WRAPPED)
    twin = run(
        tmp_path / "c",
        recipe=WORD
        + "class Paint(Base):\n    color: Color\n\n\n"
        + "class PublishedPaint(Base):\n    color: ColorWord\n",
    )
    assert (bare.returncode, kept.returncode, twin.returncode) == (1, 0, 0)


def test_the_less_obvious_shapes_exit_one(tmp_path: Path) -> None:
    shapes = {
        "annotated-literal": 'Kind = Annotated[Literal["a", "b"], Field()]\n',
        "optional-literal": 'Kind = Literal["a", "b"] | None\n',
        "flag-base": "class Bits(IntFlag):\n    A = 1\n",
        "nested-enum": "def make():\n    class Inner(Enum):\n        A = 1\n",
        "name-alias": "Shade = Color\n",
        "attribute-field": "class Paint(Base):\n    color: shown.Color\n",
    }
    codes = {
        name: run(tmp_path / name, recipe=source).returncode
        for name, source in shapes.items()
    }
    assert codes == dict.fromkeys(shapes, 1)


def test_a_file_named_like_a_home_is_not_a_home(tmp_path: Path) -> None:
    done = run(tmp_path, **{"sub/shown": SHOWN.replace("Color", "Shade")})
    assert done.returncode == 1
    assert "sub/shown.py" in done.stdout
