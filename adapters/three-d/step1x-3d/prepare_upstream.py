"""Apply the minimal fail-closed ARM64 inference patch to pinned Step1X source."""

from __future__ import annotations

import os
from pathlib import Path


root = Path(os.environ.get("STEP1X_SOURCE_ROOT", "/opt/step1x"))
path = root / "step1x3d_geometry/models/pipelines/pipeline_utils.py"
source = path.read_text()
needle = "import pymeshlab\n"
if source.count(needle) != 1:
    raise SystemExit("unexpected Step1X pipeline_utils pymeshlab import")
# The mesh cleanup functions retain their local references and will import the
# optional package only if a future adapter elects to call them.
source = source.replace(needle, "", 1)
for function in (
    "import_mesh",
    "remove_floater",
    "remove_degenerate_face",
    "reduce_face",
):
    marker = f"def {function}("
    if source.count(marker) != 1:
        raise SystemExit(f"unexpected Step1X function layout: {function}")
source = source.replace(
    "def import_mesh(mesh):\n",
    "def import_mesh(mesh):\n    import pymeshlab\n",
    1,
)
path.write_text(source)


def replace_exact(path: Path, old: str, new: str, expected: int) -> None:
    value = path.read_text()
    if value.count(old) != expected:
        raise SystemExit(f"unexpected Step1X authority layout: {path.name}: {old}")
    path.write_text(value.replace(old, new))


conditional_root = root / "step1x3d_geometry/models/conditional_encoders"
replace_exact(
    conditional_root / "dinov2_encoder.py",
    'self.cfg.dino_type is not None\n                    ), "The dino_type should be provided"',
    'self.cfg.dino_type is not None\n                    ), "The dino_type should be provided"',
    1,
)
replace_exact(
    conditional_root / "dinov2_encoder.py",
    'print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
    'if self.cfg.dino_type == "facebook/dinov2-with-registers-large":\n'
    '                        self.cfg.dino_type = "/models/dinov2-registers"\n'
    '                    print(f"Loading Dinov2 model from {self.cfg.dino_type}")',
    1,
)
replace_exact(
    conditional_root / "dinov2_clip_encoder.py",
    'print("Loading CLIP model from openai/clip-vit-large-patch14")',
    'print("Loading CLIP model from /models/clip")',
    1,
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
    1,
)
