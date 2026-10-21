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


def test_rejects_protocol_wheel_family() -> None:
    registration = Registration(
        semantic_id="test.wheel",
        owner="selection.json",
        selector="artifact",
        kind=ValueKind.PROTOCOL_WHEEL,
    )
    for name in (
        "vonk_agent_protocol-0.1.0-py3-none-any.whl",
        "vonk_agent_protocol-99.0.0-py3-none-any.whl",
    ):
        for path, source in (
            (".github/workflows/new.yml", f"run: pip install {name}\n"),
            ("adapters/new/Dockerfile", f"COPY {name} /tmp/{name}\n"),
            ("tools/new", f"#!/bin/sh\npip install {name}\n"),
            ("docs/recipe-authoring.md", f"```sh\npip install {name}\n```\n"),
        ):
            issues = scan_source(registration, path, source, "selected.whl")
            assert issues and all(issue.semantic_id == "test.wheel" for issue in issues)
            assert all("selection.json" in str(issue) for issue in issues)


def test_unrelated_package_version_is_not_recipe_contract_version() -> None:
    registration = Registration(
        semantic_id="test.version",
        owner="pyproject.toml",
        selector="project.version",
        kind=ValueKind.CONTRACT_VERSION,
    )
    assert not scan_source(
        registration, "adapters/upstream/__init__.py", '__version__ = "3.0.0"', "4.0.0"
    )


@pytest.mark.parametrize(
    "symbol", ["OS_FLOOR_BYTES", "SPARK_MEMORY_BYTES", "GUARD_BYTES"]
)
def test_memory_guard_normalizes_expressions_and_preserves_other_domains(
    symbol: str,
) -> None:
    from qualification import memory_need

    value = getattr(memory_need, symbol)
    registration = Registration(
        semantic_id="test.memory",
        owner="qualification/memory_need.py",
        selector=symbol,
        kind=ValueKind.MEMORY_BYTES,
    )
    forms = [str(value), f"{value:_}", f"{value} / 2 * 2"]
    if symbol == "OS_FLOOR_BYTES":
        forms += [f"{value // 1000**3} * 1000**3", f'"{value // 1000**3}GB"']
    if symbol == "GUARD_BYTES":
        forms += [f"{value // 1024**3} * 1024**3", f'"{value // 1024**3}GiB"']
    for expression in forms:
        assert scan_source(
            registration, "new_memory.py", f"memory_floor = {expression}", value
        )
        assert not scan_source(
            registration, "new_memory.py", f"population = {expression}", value
        )
    assert not scan_source(
        registration,
        "consumer.py",
        f"from qualification.memory_need import {symbol}",
        value,
    )


def test_stale_memory_default_is_rejected_after_owner_changes() -> None:
    registration = Registration(
        semantic_id="test.memory",
        owner="owner.py",
        selector="SPARK_MEMORY_BYTES",
        kind=ValueKind.MEMORY_BYTES,
        consumer_symbols=("default_spark_memory",),
    )
    assert scan_source(registration, "consumer.py", "default_spark_memory = 123", 456)


def test_registry_rejects_duplicate_allowances_and_broad_exceptions() -> None:
    from pydantic import ValidationError

    registration = Registration(
        semantic_id="test.memory",
        owner="owner.py",
        selector="FLOOR",
        kind=ValueKind.MEMORY_BYTES,
    )
    with pytest.raises(ValidationError):
        Registration.model_validate(registration.model_dump() | {"max_count": 3})
    with pytest.raises(ValidationError):
        Registration.model_validate(
            registration.model_dump()
            | {
                "adversarial_fixtures": [
                    {"path": "tests/*.py", "function": "*", "reason": "all fixtures"}
                ]
            }
        )
