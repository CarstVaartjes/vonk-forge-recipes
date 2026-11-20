"""Exact generated projections bind the reviewed wheel's bytes, not its filename."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.adapter_sources import CACHE_ENVIRONMENT, load
from tools.adapter_sources import materialize as place
from tools.protocol_wheel import ROOT, SELECTION, materialize, projections, selection
from tools.single_source_values import check

MIRROR = "https://github.com/owner/repo/releases/download/adapter-sources/"


class FakeGh:
    """A mirror release that accepts every upload."""

    def __init__(self) -> None:
        self.assets: dict[str, str] = {}

    def __call__(self, arguments: list[str]) -> str:
        if arguments[1] == "upload":
            name = Path(arguments[3]).name
            self.assets[name] = f"sha256:{name}"
            return ""
        return json.dumps(
            {
                "assets": [
                    {"name": n, "digest": d, "state": "uploaded"}
                    for n, d in self.assets.items()
                ]
            }
        )


def fixture_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(CACHE_ENVIRONMENT, str(tmp_path / "cache"))
    tmp_path.joinpath("policy").mkdir(exist_ok=True)
    (tmp_path / "policy/adapter-sources.json").write_text(
        json.dumps({"mirror": MIRROR, "sources": {}})
    )
    shutil.copytree(ROOT / "tools/protocol-wheel", tmp_path / "tools/protocol-wheel")
    registry = tmp_path / "tools/protocol-wheel/single-source-values.json"
    registrations = json.loads(registry.read_text())
    registrations[0]["adversarial_fixtures"] = []
    registry.write_text(json.dumps(registrations))
    selected = selection(tmp_path)
    target = tmp_path / selected.artifact
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / selected.artifact).read_bytes())
    materialize(tmp_path, gh=FakeGh())
    return tmp_path


def files(root: Path) -> list[str]:
    return [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]


def test_every_protocol_consumer_is_reproduced_byte_for_byte() -> None:
    materialize(check=True)
    selected = selection()
    holders = {
        path.parent.as_posix().removesuffix("/wheels")
        for path in ROOT.glob("adapters/**/vonk_agent_protocol-*.whl")
    }
    assert {item.context for item in selected.consumers} == {
        str(Path(holder).relative_to(ROOT)) for holder in holders
    }
    wheels = [
        payload for path, payload in projections().items() if path.endswith(".whl")
    ]
    assert all(
        hashlib.sha256(payload).hexdigest() == selected.artifact_sha256
        for payload in wheels
    )


def test_guard_rejects_hand_edited_generated_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture_root(tmp_path, monkeypatch)
    assert not check(root, files(root))
    selected = selection(root)
    target = root / selected.consumers[0].context / "Dockerfile"
    target.write_text(target.read_text() + "\nRUN echo drift\n")
    issues = check(root, files(root))
    assert any(
        issue.observed == "stale generated projection"
        and issue.path.endswith("Dockerfile")
        for issue in issues
    )


def test_changed_owner_invalidates_stale_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture_root(tmp_path, monkeypatch)
    selected = selection(root)
    path = root / SELECTION
    path.write_text(
        selected.model_copy(
            update={"platform_source_commit": "f" * 40}
        ).model_dump_json()
    )
    assert any(
        issue.observed == "stale generated projection"
        for issue in check(root, files(root))
    )


def test_same_filename_with_different_content_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture_root(tmp_path, monkeypatch)
    selected = selection(root)
    artifact = root / selected.artifact
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="expected sha256"):
        projections(root)


def test_new_reviewed_artifact_drives_filename_hash_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = fixture_root(tmp_path, monkeypatch)
    selected = selection(root)
    old = root / selected.artifact
    version = "92.3.1"
    name = f"vonk_agent_protocol-{version}-py3-none-any.whl"
    output = io.BytesIO()
    with ZipFile(output, "w") as wheel:
        wheel.writestr(
            f"vonk_agent_protocol-{version}.dist-info/METADATA",
            f"Name: vonk-agent-protocol\nVersion: {version}\n",
        )
    payload = output.getvalue()
    new = old.with_name(name)
    new.write_bytes(payload)
    old.unlink()
    updated = selected.model_copy(
        update={
            "artifact": str(new.relative_to(root)),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    (root / SELECTION).write_text(updated.model_dump_json())
    projected = projections(root)
    for consumer in updated.consumers:
        context = Path(consumer.context)
        assert projected[str(context / consumer.wheel_directory / name)] == payload
        assert name.encode() in projected[str(context / "Dockerfile")]
        if consumer.wheel_directory:
            receipt = json.loads(
                projected[str(context / consumer.wheel_directory / "provenance.json")]
            )
            assert receipt["version"] == version
            assert receipt["artifact_sha256"] == updated.artifact_sha256
    assert any(
        issue.observed == "stale generated projection"
        for issue in check(root, files(root))
    )

    materialize(root, gh=FakeGh())
    assert not check(root, files(root))
    # one wheel on disk; the manifest declares the one blob for every consumer path
    assert len(list(root.glob("adapters/**/*.whl"))) == 1
    place(root)  # a tree materialized from the mirror cache holds every path
    materialize(root, check=True)
    declared = load(root).sources[updated.artifact_sha256].paths
    assert sorted(declared) == sorted(
        str(Path(c.context) / c.wheel_directory / name) for c in updated.consumers
    )
    assert (root / ".gitignore").read_text().count(name) == len(updated.consumers)


def test_check_needs_a_materialized_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fixture_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing or stale protocol wheel"):
        materialize(root, check=True)
    place(root)
    materialize(root, check=True)
