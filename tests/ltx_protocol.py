from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from tools.protocol_wheel import selection, verified_payload

SELECTED_PROTOCOL = selection(ROOT)
PROTOCOL_WHEEL = Path(SELECTED_PROTOCOL.artifact).name
PROTOCOL_WHEEL_SHA256 = SELECTED_PROTOCOL.artifact_sha256
PROTOCOL_SOURCE_COMMIT = SELECTED_PROTOCOL.platform_source_commit
verified_payload(ROOT, SELECTED_PROTOCOL)
PROTOCOL_CONTEXTS = (
    ROOT / "adapters/video/ltx2-pytorch",
    ROOT / "adapters/video/ltx2-sync-native",
    ROOT / "adapters/video/ltx23-sync-native-disk",
)

wheel_path = PROTOCOL_CONTEXTS[0] / "wheels" / PROTOCOL_WHEEL
sys.path.insert(0, str(wheel_path))

# Load the exact binary fixture dynamically; type checking does not resolve
# ZIP imports, and a handwritten contract stub would create another owner.
RecipeJobInputFile = importlib.import_module("vonk_agent_protocol").RecipeJobInputFile
RecipeJobInputManifest = importlib.import_module(
    "vonk_agent_protocol.job_inputs"
).RecipeJobInputManifest
MAX_INPUT_FILE_BYTES = importlib.import_module(
    "vonk_agent_protocol.recipe_jobs"
).MAX_INPUT_FILE_BYTES


def manifest_bytes(payload: bytes, *, name: str = "prompt.txt") -> bytes:
    file = RecipeJobInputFile.parse(
        {
            "slot": "prompt",
            "name": name,
            "media_type": "text/plain",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        maximum_bytes=MAX_INPUT_FILE_BYTES,
    )
    manifest = RecipeJobInputManifest(
        schema_version=1, total_bytes=len(payload), files=[file]
    )
    return manifest.model_dump_json().encode("utf-8")


def assert_streaming_quantization_supported(command: list[str]) -> None:
    """Pinned LTX streaming supports no quantization policy or fp8-cast only."""
    offload = (
        command[command.index("--offload") + 1] if "--offload" in command else "none"
    )
    quantization = (
        command[command.index("--quantization") + 1]
        if "--quantization" in command
        else None
    )
    checkpoint = (
        command[command.index("--checkpoint-path") + 1]
        if "--checkpoint-path" in command
        else ""
    )
    if "-fp4" in Path(checkpoint).stem:
        assert offload == "none" and quantization == "nvfp4-prequant", (
            "NVFP4 checkpoints require the native nvfp4-prequant loader without block streaming"
        )
    assert offload == "none" or quantization in (None, "fp8-cast"), (
        f"{offload} block streaming does not support {quantization} quantization"
    )


def verify_bundled_wheels() -> None:
    artifacts = []
    for context in PROTOCOL_CONTEXTS:
        path = context / "wheels" / PROTOCOL_WHEEL
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == PROTOCOL_WHEEL_SHA256
        receipt = json.loads((path.parent / "provenance.json").read_text())
        assert receipt["platform_source_commit"] == PROTOCOL_SOURCE_COMMIT
        assert receipt["artifact_sha256"] == PROTOCOL_WHEEL_SHA256
        artifacts.append(payload)
    assert artifacts[0] == artifacts[1] == artifacts[2]
