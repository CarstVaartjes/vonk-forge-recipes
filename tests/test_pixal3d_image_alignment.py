"""Publish the pinned Trimesh export without weakening texture validation."""

import ast
import json
import runpy
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/trellis2-native"
# Trimesh 4.10.1 + Pillow 12.0.0, box with two PBR textures. The first WebP
# contains 78 encoded bytes but its exported view includes two alignment spaces.
FIXTURE = ROOT / "tests/fixtures/pixal3d-trimesh-4.10.1-padded.glb"


@pytest.mark.parametrize("damage", [None, "signature", "padding", "encoded"])
def test_pinned_export_publishes_only_intact_textures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    monkeypatch.syspath_prepend(str(ADAPTER))
    monkeypatch.delitem(sys.modules, "glb_validation", raising=False)
    adapter = runpy.run_path(str(ADAPTER / "pixal3d_job.py"))
    data = FIXTURE.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20 : 20 + json_length])
    binary_start = 28 + json_length
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    start = binary_start + view["byteOffset"]
    encoded_length = struct.unpack_from("<I", data, start + 4)[0] + 8
    assert view["byteLength"] == encoded_length + 2
    assert data[start + encoded_length : start + view["byteLength"]] == b"  "
    changed = bytearray(data)
    if damage == "signature":
        changed[start] = 0
    elif damage == "padding":
        changed[start + encoded_length] = ord("x")
    elif damage == "encoded":
        # Keep RIFF size and alignment valid; destroy the encoded VP8 frame.
        changed[start + 20 : start + encoded_length] = b"\0" * (encoded_length - 20)
    temporary, output = tmp_path / ".output.tmp.glb", tmp_path / "output.glb"
    temporary.write_bytes(changed)
    # The original artifact reproduces the hardware failure with the real
    # validator, even though Pillow can decode the padded image view.
    with pytest.raises(ValueError, match="image payload"):
        adapter["validate_mesh_glb"](temporary, profile="textured-pbr")
    # Execute main's actual export/publication block with only GPU mesh
    # generation replaced. On main the intact export exits with the exact
    # hardware error; the repaired adapter publishes it.
    tree = ast.parse((ADAPTER / "pixal3d_job.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    start_index = next(
        index
        for index, node in enumerate(main.body)
        if ast.unparse(node).startswith("args.output_dir.mkdir(")
    )
    publication = ast.Module(body=main.body[start_index:], type_ignores=[])

    def publish() -> None:
        namespace = {
            **adapter,
            "args": SimpleNamespace(output_dir=tmp_path),
            "glb": SimpleNamespace(export=lambda path, **_: path.write_bytes(changed)),
        }
        exec(compile(publication, str(ADAPTER / "pixal3d_job.py"), "exec"), namespace)  # noqa: S102

    if damage:
        with pytest.raises(SystemExit, match="image payload"):
            publish()
        assert not output.exists()
        assert not temporary.exists()
    else:
        publish()
        adapter["validate_mesh_glb"](output, profile="textured-pbr")
        published = output.read_bytes()
        length = struct.unpack_from("<I", published, 12)[0]
        normalized = json.loads(published[20 : 20 + length])
        updated_view = normalized["bufferViews"][document["images"][0]["bufferView"]]
        assert updated_view["byteLength"] == encoded_length
        # Texture bytes, geometry, offsets and BIN alignment remain identical.
        assert published[28 + length :] == data[binary_start:]
        adapter["_exclude_webp_alignment"](output)
        assert output.read_bytes() == published
