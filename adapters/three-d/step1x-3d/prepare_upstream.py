"""Apply fail-closed ARM64 inference patches to pinned Step1X source."""

from __future__ import annotations

import os
from pathlib import Path


def replace_exact(path: Path, old: str, new: str, expected: int = 1) -> None:
    value = path.read_text(encoding="utf-8")
    if value.count(old) != expected:
        raise SystemExit(f"unexpected Step1X authority layout: {path.name}: {old}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def patch_pipeline_utils(root: Path) -> None:
    """Make the optional pymeshlab dependency lazy without breaking annotations."""

    path = root / "step1x3d_geometry/models/pipelines/pipeline_utils.py"
    source = path.read_text(encoding="utf-8")
    global_import = "import pymeshlab\n"
    if source.count(global_import) != 1:
        raise SystemExit("unexpected Step1X pipeline_utils pymeshlab import")
    if source.startswith("from __future__ import annotations\n"):
        raise SystemExit("Step1X pipeline_utils is already patched")
    source = "from __future__ import annotations\n" + source.replace(
        global_import, "", 1
    )
    for function in (
        "load_mesh",
        "trimesh2pymeshlab",
        "pymeshlab2trimesh",
        "import_mesh",
        "remove_degenerate_face",
    ):
        marker = f"def {function}("
        if source.count(marker) != 1:
            raise SystemExit(f"unexpected Step1X function layout: {function}")
        line_end = source.index("\n", source.index(marker)) + 1
        source = source[:line_end] + "    import pymeshlab\n" + source[line_end:]
    path.write_text(source, encoding="utf-8")


def patch_label_encoder(root: Path) -> None:
    """Repair the upstream string lookup used by controlled geometry inference."""

    replace_exact(
        root
        / "step1x3d_geometry/models/conditional_encoders/label_encoder.py",
        'GEOMETRY_QUALITY_MAPPING[label["geometry_type"][0]]',
        'GEOMETRY_QUALITY_MAPPING[label["geometry_type"]]',
    )


def patch_model_authorities(root: Path) -> None:
    conditional_root = root / "step1x3d_geometry/models/conditional_encoders"
    replace_exact(
        conditional_root / "dinov2_encoder.py",
        'self.cfg.dino_type is not None\n                    ), "The dino_type should be provided"',
        'self.cfg.dino_type is not None\n                    ), "The dino_type should be provided"',
    )
    replace_exact(
        conditional_root / "dinov2_encoder.py",
        'print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
        'if self.cfg.dino_type == "facebook/dinov2-with-registers-large":\n'
        '                        self.cfg.dino_type = "/models/dinov2-registers"\n'
        '                    print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
    )
    replace_exact(
        conditional_root / "dinov2_clip_encoder.py",
        'print("Loading CLIP model from openai/clip-vit-large-patch14")',
        'print("Loading CLIP model from /models/clip")',
    )
    replace_exact(
        conditional_root / "dinov2_clip_encoder.py",
        '"openai/clip-vit-large-patch14"',
        '"/models/clip"',
        2,
    )
    replace_exact(
        conditional_root / "dinov2_clip_encoder.py",
        'print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
        'if self.cfg.dino_type == "facebook/dinov2-with-registers-large":\n'
        '                        self.cfg.dino_type = "/models/dinov2-registers"\n'
        '                    print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
    )


def prepare(root: Path) -> None:
    patch_pipeline_utils(root)
    patch_label_encoder(root)
    patch_model_authorities(root)


def main() -> None:
    root = Path(os.environ.get("STEP1X_SOURCE_ROOT", "/opt/step1x"))
    prepare(root)


if __name__ == "__main__":
    main()
