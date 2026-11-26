"""Apply fail-closed offline patches to the pinned MMAudio source."""

from __future__ import annotations

import os
from pathlib import Path


def replace_exact(path: Path, old: str, new: str, expected: int = 1) -> None:
    value = path.read_text(encoding="utf-8")
    if value.count(old) != expected:
        raise SystemExit(f"unexpected MMAudio source layout: {path.name}: {old}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def prepare(root: Path) -> None:
    """Read the two checkpoints the creator fetches from the hub from the recipe's mounts."""

    # The creator's DFN5B-CLIP id now redirects to apple/DFN5B-CLIP-ViT-H-14-378: open_clip's
    # local-dir loader reads open_clip_config.json and open_clip_pytorch_model.bin from the mount.
    replace_exact(
        root / "mmaudio/model/utils/features_utils.py",
        "'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'",
        "'local-dir:/models/dfn5b-clip'",
    )
    # BigVGAN's from_pretrained reads config.json and bigvgan_generator.pt from a directory.
    replace_exact(
        root / "mmaudio/ext/autoencoder/autoencoder.py",
        "'nvidia/bigvgan_v2_44khz_128band_512x'",
        "'/models/bigvgan'",
    )


def main() -> None:
    prepare(Path(os.environ.get("MMAUDIO_SOURCE_ROOT", "/opt/mmaudio")))


if __name__ == "__main__":
    main()
