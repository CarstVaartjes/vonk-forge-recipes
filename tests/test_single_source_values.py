"""One guard for the registered classes, with no permitted duplicate owners."""

from pathlib import Path

import pytest

from tools.single_source_values import Registration, ValueKind, check, scan_source

ROOT = Path(__file__).resolve().parents[1]


def test_registered_values_have_one_owner() -> None:
    assert not (issues := check(ROOT)), "\n".join(map(str, issues))


def test_guard_rejects_old_and_new_contract_versions() -> None:
    registration = Registration(
        semantic_id="test.version",
        owner="pyproject.toml",
        selector="project.version",
        kind=ValueKind.CONTRACT_VERSION,
    )
    for version in ("1.0.0", "99.2.0"):
        issues = scan_source(
            registration, "consumer.py", f'CONTRACT_VERSION = "{version}"', "4.0.0"
        )
        assert len(issues) == 1
        assert issues[0].line == 1
        assert "pyproject.toml" in str(issues[0])


def test_guard_rejects_duplicate_and_missing_owner_selectors(tmp_path: Path) -> None:
    (tmp_path / "owner.py").write_text("FLOOR = 23\n")
    registry = tmp_path / "registry/single-source-values.json"
    registry.parent.mkdir()
    registration = Registration(
        semantic_id="test.memory",
        owner="owner.py",
        selector="FLOOR",
        kind=ValueKind.MEMORY_BYTES,
    )
    registry.write_text(
        "["
        + registration.model_dump_json()
        + ","
        + registration.model_dump_json()
        + "]"
    )
    with pytest.raises(ValueError, match="duplicate/overlapping"):
        check(tmp_path, ["owner.py", "registry/single-source-values.json"])
    registry.write_text(
        "["
        + registration.model_copy(update={"selector": "MISSING"}).model_dump_json()
        + "]"
    )
    with pytest.raises(ValueError, match="exactly one"):
        check(tmp_path, ["owner.py", "registry/single-source-values.json"])
