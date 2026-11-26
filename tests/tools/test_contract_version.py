"""The contract version is computed from the change class, never hand-set (G3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipe_tooling.contract_version import (
    ChangeClass,
    ContractVersionError,
    Surface,
    change_class,
    committed_major,
    findings,
    main,
    next_version,
    shared_functions,
)

PROPERTIES: dict[str, object] = {
    "name": {"type": "string", "maxLength": 64},
    "kind": {"enum": ["chat", "job"], "type": "string"},
    "requires_token": {"type": "boolean"},
}
REQUIRED = ["name", "kind", "requires_token"]


def model(
    properties: dict[str, object] | None = None,
    required: list[str] | None = None,
    defs: dict[str, object] | None = None,
) -> dict[str, object]:
    document = {
        "type": "object",
        "additionalProperties": False,
        "properties": PROPERTIES if properties is None else properties,
        "required": REQUIRED if required is None else required,
    }
    return {"$defs": {"Model": document, **(defs or {})}, "$ref": "#/$defs/Model"}


def classify(
    head: dict[str, object], vectors: dict[str, bytes] | None = None
) -> ChangeClass:
    base = Surface(schemas={"model.json": model()}, vectors={"a.json": b'{"x": 1}'})
    after = Surface(
        schemas={"model.json": head},
        vectors=dict(base.vectors) if vectors is None else vectors,
    )
    return change_class(findings(base, after), src_changed=True)


def write_surface(root: Path, schema: dict[str, object]) -> tuple[Path, Path]:
    schemas, vectors = root / "schema", root / "examples"
    schemas.mkdir(parents=True)
    vectors.mkdir()
    (schemas / "model.json").write_text(json.dumps(schema))
    (vectors / "a.json").write_text('{"x": 1}')
    return schemas, vectors


def test_a_new_required_field_on_an_existing_model_needs_a_major(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    head = model({**PROPERTIES, "licence": {"type": "string"}}, [*REQUIRED, "licence"])
    assert classify(head) is ChangeClass.BREAKING

    base_schemas, base_vectors = write_surface(tmp_path / "base", model())
    head_schemas, head_vectors = write_surface(tmp_path / "head", head)
    arguments = [
        "next",
        f"--base-schemas={base_schemas}",
        f"--base-vectors={base_vectors}",
        f"--head-schemas={head_schemas}",
        f"--head-vectors={head_vectors}",
        "--base-version=5.8.0",
        "--src-changed=yes",
    ]
    assert main([*arguments, "--major=5"]) == 1
    assert main([*arguments, "--major=6"]) == 0
    assert capsys.readouterr().out.split() == ["6.0.0"]


def test_born_optional_and_relaxed_changes_are_minor() -> None:
    kind = {"enum": ["chat", "job", "decision"], "type": "string"}
    assert classify(model({**PROPERTIES, "kind": kind})) is ChangeClass.MINOR
    note = {"type": "string", "default": ""}
    assert classify(model({**PROPERTIES, "note": note})) is ChangeClass.MINOR
    # The dead-field rule's second step: a required field becomes optional.
    nullable = {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": None}
    relaxed = model({**PROPERTIES, "requires_token": nullable}, ["name", "kind"])
    assert classify(relaxed) is ChangeClass.MINOR
    # A model-typed field widened to a union keeps its ``$ref`` as one alternative.
    union = {"anyOf": [{"$ref": "#/$defs/Model"}, {"type": "null"}], "default": None}
    nested = model({**PROPERTIES, "parent": {"$ref": "#/$defs/Model"}})
    widened = model({**PROPERTIES, "parent": union})
    base = Surface(schemas={"model.json": nested})
    head = Surface(schemas={"model.json": widened})
    assert change_class(findings(base, head), src_changed=True) is ChangeClass.MINOR
    # A union whose members moved into a referenced union keeps every member.
    flat = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
    value = {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}]}
    moved = {"anyOf": [{"$ref": "#/$defs/Value"}, {"type": "null"}]}
    base = Surface(schemas={"model.json": model({**PROPERTIES, "value": flat})})
    head = Surface(
        schemas={
            "model.json": model({**PROPERTIES, "value": moved}, defs={"Value": value})
        }
    )
    assert change_class(findings(base, head), src_changed=True) is ChangeClass.MINOR
    # An alternative that was relaxed while another was added is kept, not removed.
    short = {"anyOf": [{"type": "string", "maxLength": 5}]}
    longer = {"anyOf": [{"type": "string", "maxLength": 9}, {"type": "null"}]}
    base = Surface(schemas={"model.json": model({**PROPERTIES, "note": short})})
    head = Surface(schemas={"model.json": model({**PROPERTIES, "note": longer})})
    assert change_class(findings(base, head), src_changed=True) is ChangeClass.MINOR
    # A const is a one-member enum, so growing it into an enum widens it.
    single = {"const": "chat", "type": "string"}
    base = Surface(schemas={"model.json": model({**PROPERTIES, "kind": single})})
    assert (
        change_class(
            findings(base, Surface(schemas={"model.json": model()})), src_changed=True
        )
        is ChangeClass.MINOR
    )
    assert next_version("5.8.0", 5, ChangeClass.MINOR) == "5.9.0"


def test_narrowing_or_removing_is_breaking() -> None:
    shrunk = {"enum": ["chat"], "type": "string"}
    assert classify(model({**PROPERTIES, "kind": shrunk})) is ChangeClass.BREAKING
    bounded = {"type": "string", "maxLength": 32}
    assert classify(model({**PROPERTIES, "name": bounded})) is ChangeClass.BREAKING
    kept = {key: value for key, value in PROPERTIES.items() if key != "requires_token"}
    assert classify(model(kept, ["name", "kind"])) is ChangeClass.BREAKING


def test_vectors_may_only_grow() -> None:
    assert classify(model(), {"a.json": b'{"x": 1, "y": null}'}) is ChangeClass.MINOR
    assert (
        classify(model(), {"a.json": b'{"x": 1}', "b.json": b"{}"}) is ChangeClass.MINOR
    )
    assert classify(model(), {"a.json": b'{"x": 2}'}) is ChangeClass.BREAKING
    assert classify(model(), {}) is ChangeClass.BREAKING


def test_identical_surface_is_a_patch_and_an_unchanged_source_cuts_nothing() -> None:
    surface = Surface(schemas={"model.json": model()})
    unchanged = findings(surface, surface)
    assert change_class(unchanged, src_changed=True) is ChangeClass.PATCH
    assert next_version("5.8.0", 5, ChangeClass.PATCH) == "5.8.1"
    assert change_class(unchanged, src_changed=False) is ChangeClass.NONE
    assert next_version("5.8.0", 5, ChangeClass.NONE) is None


def test_a_pull_request_never_sets_the_minor_or_patch() -> None:
    placeholder = 'CONTRACT_MAJOR = 5\n__version__ = "5.0.0"\n'
    assert committed_major(placeholder, "5.8.0") == 5
    for text in (
        'CONTRACT_MAJOR = 5\n__version__ = "5.9.0"\n',
        'CONTRACT_MAJOR = 7\n__version__ = "7.0.0"\n',
        '__version__ = "5.0.0"\n',
    ):
        with pytest.raises(ContractVersionError):
            committed_major(text, "5.8.0")
    assert committed_major('CONTRACT_MAJOR = 6\n__version__ = "6.0.0"\n', "5.8.0") == 6


def test_shared_functions_may_only_grow() -> None:
    def surface(*names: str) -> Surface:
        listed = "".join(f'    "{name}",\n' for name in names)
        return Surface(
            shared_functions=shared_functions(
                f"SHARED_FUNCTIONS: Final = (\n{listed})\n"
            )
        )

    base = surface("read_model", "wav_facts")
    removed = findings(base, surface("read_model"))
    assert change_class(removed, src_changed=True) is ChangeClass.BREAKING
    added = findings(base, surface("read_model", "wav_facts", "build_key"))
    assert change_class(added, src_changed=True) is ChangeClass.MINOR
    # A tree without the list (released before it existed) is not compared.
    assert shared_functions("CONTRACT_VERSION = __version__\n") is None
    assert not findings(Surface(), surface("read_model"))
