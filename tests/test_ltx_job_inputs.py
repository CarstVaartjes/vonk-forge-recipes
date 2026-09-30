"""The actual Controller protocol producer feeds all three native LTX adapters."""

import json
import runpy
from pathlib import Path

import pytest
from ltx_protocol import (
    RecipeJobInputFile,
    RecipeJobInputManifest,
    manifest_bytes,
    verify_bundled_wheels,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = (
    "ltx2-pytorch/pipelines/run.py",
    "ltx2-sync-native/run.py",
    "ltx23-sync-native-disk/run.py",
)


def stage_prompt(root: Path, content: bytes, *, name: str = "Prompt.TXT") -> None:
    (root / name).write_bytes(content)
    (root / "manifest.json").write_bytes(manifest_bytes(content, name=name))


def test_all_native_contexts_bundle_the_same_current_protocol_wheel() -> None:
    verify_bundled_wheels()


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_real_protocol_manifest_round_trips_to_prompt_slot(
    adapter: str, tmp_path: Path
) -> None:
    module = runpy.run_path(str(ROOT / "adapters/video" / adapter))
    read = module["_load_prompt"]
    read.__globals__["INPUT_ROOT"] = tmp_path
    content = "A quiet forest at sunrise — wind through the pines".encode()
    stage_prompt(tmp_path, content)
    assert read() == content.decode("utf-8")

    # The real model binds each metadata field; a content swap cannot hide behind a valid size.
    (tmp_path / "Prompt.TXT").write_bytes(b"x" * len(content))
    with pytest.raises(SystemExit, match="digest"):
        read()

    (tmp_path / "Prompt.TXT").write_bytes(content)
    (tmp_path / "undeclared.txt").write_text("unlisted")
    with pytest.raises(SystemExit, match="manifest"):
        read()
    (tmp_path / "undeclared.txt").unlink()
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest"):
        read()


def test_manifest_model_rejects_wrong_totals_before_adapter_reads_files(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(str(ROOT / "adapters/video/ltx2-sync-native/run.py"))
    read = module["_load_prompt"]
    read.__globals__["INPUT_ROOT"] = tmp_path
    payload = b"one prompt"
    (tmp_path / "prompt.txt").write_bytes(payload)
    manifest = json.loads(manifest_bytes(payload))
    manifest["total_bytes"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid Vonk job input manifest"):
        read()


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_prompt_only_adapter_rejects_an_extra_declared_slot(
    adapter: str, tmp_path: Path
) -> None:
    module = runpy.run_path(str(ROOT / "adapters/video" / adapter))
    read = module["_load_prompt"]
    read.__globals__["INPUT_ROOT"] = tmp_path
    content = b"one prompt"
    stage_prompt(tmp_path, content)
    manifest = RecipeJobInputManifest.model_validate_json(
        (tmp_path / "manifest.json").read_bytes()
    )
    extra = RecipeJobInputFile.parse(
        {
            "slot": "auxiliary",
            "name": "extra.txt",
            "media_type": "text/plain",
            "size_bytes": len(content),
            "sha256": manifest.files[0].sha256,
        },
        maximum_bytes=16384,
    )
    manifest = RecipeJobInputManifest(
        schema_version=1, total_bytes=2 * len(content), files=[*manifest.files, extra]
    )
    (tmp_path / "extra.txt").write_bytes(content)
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    with pytest.raises(SystemExit, match="exactly one declared"):
        read()


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_prompt_size_bound_is_checked_before_loading_contents(
    adapter: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runpy.run_path(str(ROOT / "adapters/video" / adapter))
    read = module["_load_prompt"]
    read.__globals__["INPUT_ROOT"] = tmp_path
    stage_prompt(tmp_path, b"x" * 16385)
    original = Path.read_bytes

    def bounded_read(path: Path) -> bytes:
        if path.name == "Prompt.TXT":
            raise AssertionError(
                "Oversized prompt was loaded before enforcing its bound"
            )
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    with pytest.raises(SystemExit, match="1..16384"):
        read()
