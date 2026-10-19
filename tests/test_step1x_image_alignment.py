"""Trimesh's padded image views must publish without relaxing GLB validation."""

import json
import runpy
import struct
import sys
from pathlib import Path

import pytest
from test_three_d_glb_validation import Glb

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/step1x-3d"


@pytest.mark.parametrize("padding", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("corrupt", [False, True])
def test_exported_image_view_excludes_only_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, padding: int, corrupt: bool
) -> None:
    monkeypatch.syspath_prepend(str(ADAPTER))
    monkeypatch.delitem(sys.modules, "glb_validation", raising=False)
    adapter = runpy.run_path(str(ADAPTER / "run.py"))
    builder = Glb()
    document = json.loads(json.dumps(builder.document("textured")))
    image = builder.views[document["images"][0]["bufferView"]]
    image["byteLength"] = int(str(image["byteLength"])) + padding
    if corrupt:
        # Preserve IEND; break the PNG signature, which normalization cannot fix.
        builder.blob[int(str(image["byteOffset"]))] = 0
    builder.blob.extend(b" " * padding)
    source, output = tmp_path / "temporary.glb", tmp_path / "mesh.glb"
    source.write_bytes(builder.bytes(document))
    if corrupt or padding == 4:
        with pytest.raises(SystemExit, match="image payload"):
            adapter["_publish_glb"](source, output, profile="textured")
        assert not output.exists()
    else:
        adapter["_publish_glb"](source, output, profile="textured")
        content = output.read_bytes()
        length = struct.unpack_from("<I", content, 12)[0]
        published = json.loads(content[20 : 20 + length])
        view = published["bufferViews"][document["images"][0]["bufferView"]]
        assert view["byteLength"] == int(str(image["byteLength"])) - padding
