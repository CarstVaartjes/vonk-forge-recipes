"""Make the creator's inference import closure installable on aarch64.

``torch_cluster`` has no aarch64 wheel (source only), and the creator's autoencoder imports it at
module level for ``fps``, which only its encoder uses (``_sample_features``). PartCrafter issues 45
and 47 confirm inference never calls it, and TripoSG upstream comments out the same line. Each
edit requires exactly one match, so a moved upstream pin fails the image build instead of running
unpatched code.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath


ENCODER_FILE = PurePosixPath("src/models/autoencoders/autoencoder_kl_triposg.py")


def replace_exact(path: Path, old: str, new: str, expected: int = 1) -> None:
    value = path.read_text(encoding="utf-8")
    if value.count(old) != expected:
        raise SystemExit(f"unexpected PartCrafter source layout: {path.name}: {old}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def patch_encoder_only_import(root: Path) -> None:
    replace_exact(
        root.joinpath(*ENCODER_FILE.parts),
        "from torch_cluster import fps\n",
        "# from torch_cluster import fps  # encoder-only (_sample_features); no aarch64 wheel\n",
    )


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: prepare_upstream.py <partcrafter-root>")
    patch_encoder_only_import(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
