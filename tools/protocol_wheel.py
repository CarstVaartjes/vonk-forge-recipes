"""Materialize adapter inputs from one reviewed, content-identified protocol wheel."""

from __future__ import annotations

import hashlib
import io
import json
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SELECTION = "tools/protocol-wheel/selection.json"


class ProtocolConsumer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context: str
    wheel_directory: str = ""


class ProtocolSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    platform_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_path: str
    consumers: tuple[ProtocolConsumer, ...]


def selection(root: Path = ROOT) -> ProtocolSelection:
    selected = ProtocolSelection.model_validate_json((root / SELECTION).read_bytes())
    paths = [selected.artifact, *(item.context for item in selected.consumers)]
    if any(
        PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
        for path in paths
    ):
        raise ValueError("protocol selection paths must stay inside the repository")
    if len({item.context for item in selected.consumers}) != len(selected.consumers):
        raise ValueError("duplicate protocol consumer")
    if any(item.wheel_directory not in ("", "wheels") for item in selected.consumers):
        raise ValueError("unsupported protocol wheel directory")
    return selected


def verified_payload(root: Path, selected: ProtocolSelection) -> tuple[bytes, str]:
    payload = (root / selected.artifact).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != selected.artifact_sha256:
        raise ValueError(
            f"{SELECTION}: expected sha256:{selected.artifact_sha256}, got sha256:{actual}"
        )
    with ZipFile(io.BytesIO(payload)) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(
                "protocol wheel must have exactly one package metadata record"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
    if metadata["Name"] != "vonk-agent-protocol":
        raise ValueError("selected artifact is not the agent protocol package")
    version = metadata["Version"]
    if (
        not version
        or Path(selected.artifact).name
        != f"vonk_agent_protocol-{version}-py3-none-any.whl"
    ):
        raise ValueError("protocol wheel filename disagrees with package metadata")
    return payload, version


def projections(root: Path = ROOT) -> dict[str, bytes]:
    selected = selection(root)
    payload, version = verified_payload(root, selected)
    name = Path(selected.artifact).name
    substitutions = {
        "@protocol.wheel@": name,
        "@protocol.sha256@": selected.artifact_sha256,
        "@protocol.source_commit@": selected.platform_source_commit,
    }
    receipt = {
        "artifact": name,
        "artifact_sha256": selected.artifact_sha256,
        "platform_source_commit": selected.platform_source_commit,
        "source_path": selected.source_path,
        "version": version,
    }
    outputs: dict[str, bytes] = {}
    for consumer in selected.consumers:
        context = Path(consumer.context)
        outputs[str(context / consumer.wheel_directory / name)] = payload
        templates = root / "tools/protocol-wheel"
        for target in ("Dockerfile", "vendored-upstream.json"):
            template = templates / f"{context.name}.{target}.in"
            if not template.exists() and target == "vendored-upstream.json":
                continue
            text = template.read_text()
            for token, value in substitutions.items():
                text = text.replace(token, value)
            if "@protocol." in text:
                raise ValueError(f"unknown protocol template token: {template}")
            outputs[str(context / target)] = text.encode()
        if consumer.wheel_directory:
            outputs[str(context / consumer.wheel_directory / "provenance.json")] = (
                json.dumps(receipt, indent=2) + "\n"
            ).encode()
    if selected.artifact not in outputs:
        raise ValueError("selected wheel must belong to a registered consumer")
    return outputs


def materialize(root: Path = ROOT, *, check: bool = False) -> None:
    outputs = projections(root)
    for path, expected in outputs.items():
        target = root / path
        if check:
            if not target.is_file() or target.read_bytes() != expected:
                raise ValueError(
                    f"{path}: stale protocol projection; run tools/materialize-protocol-wheel"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(expected)
