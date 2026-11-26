"""Apply fail-closed ARM64 offline patches to the pinned Hunyuan3D 2.1 source."""

from __future__ import annotations

import os
from pathlib import Path


def replace_exact(path: Path, old: str, new: str, expected: int = 1) -> None:
    value = path.read_text(encoding="utf-8")
    if value.count(old) != expected:
        raise SystemExit(f"unexpected Hunyuan3D 2.1 source layout: {path.name}: {old}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def patch_bpy_import(root: Path) -> None:
    """Drop the top-level bpy import, which has no aarch64 wheel.

    bpy is used only by the Blender helpers (``convert_obj_to_glb`` and its callees), which
    ``save_glb=False`` never calls; ``export_pbr_glb.py`` writes the GLB instead.
    """

    replace_exact(root / "hy3dpaint/DifferentiableRenderer/mesh_utils.py", "import bpy\n", "")


def patch_multiview_download(root: Path) -> None:
    """Read the mounted paint checkpoint instead of calling snapshot_download."""

    replace_exact(
        root / "hy3dpaint/utils/multiview_utils.py",
        "        model_path = huggingface_hub.snapshot_download(\n"
        "            repo_id=config.multiview_pretrained_path,\n"
        '            allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],\n'
        "        )\n",
        "        model_path = config.multiview_pretrained_path\n",
    )


def prepare(root: Path) -> None:
    patch_bpy_import(root)
    patch_multiview_download(root)


def main() -> None:
    prepare(Path(os.environ.get("HUNYUAN3D_SOURCE_ROOT", "/opt/hunyuan3d")))


if __name__ == "__main__":
    main()
