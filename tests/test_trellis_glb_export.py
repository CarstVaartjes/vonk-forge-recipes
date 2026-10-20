"""Exercise the actual pinned CPU exporter on a WebP requiring alignment."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/three-d/trellis2-native"


def test_trimesh_webp_padding_is_outside_the_embedded_image(tmp_path: Path) -> None:
    with patch.object(sys, "path", [str(ADAPTER), *sys.path]):
        export = runpy.run_path(str(ADAPTER / "glb_export.py"))["export_webp_glb"]
        validator = runpy.run_path(str(ADAPTER / "glb_validation.py"))
    # These are image-build dependencies; the gate environment need not import
    # their full type trees to check our callback regression.
    np = importlib.import_module("numpy")
    trimesh = importlib.import_module("trimesh")
    Image = importlib.import_module("PIL.Image")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.new("RGB", (16, 16), (50, 10, 1)),
        metallicRoughnessTexture=Image.new("RGB", (16, 16), (90, 90, 90)),
    )
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        vertex_normals=[[0, 0, 1]] * 3,
        visual=trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [0, 1]]), material=material
        ),
    )
    path = tmp_path / "output.glb"
    # Reproduce the production error with the real pinned exporter first.
    mesh.export(path, extension_webp=True)
    validator["normalize_glb_json_padding"](path)
    with pytest.raises(ValueError, match="image payload does not match"):
        validator["validate_mesh_glb"](path, profile="textured-pbr")
    export(mesh, path)
    validator["normalize_glb_json_padding"](path)
    validator["validate_mesh_glb"](path, profile="textured-pbr")
