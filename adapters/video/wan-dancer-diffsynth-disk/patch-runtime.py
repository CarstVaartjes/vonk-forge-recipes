"""Fail-closed offline adaptation of pinned DiffSynth-Studio source."""

from __future__ import annotations

import sys
from pathlib import Path


EXPECTED_VERSION = "__version__ = '2.1.5'"


def replace_once(path: Path, old: str, new: str) -> None:
    value = path.read_text(encoding="utf-8")
    count = value.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch target, found {count}: {old!r}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    required = (
        root / "LICENSE",
        root / "diffsynth/version.py",
        root / "diffsynth/core/loader/config.py",
        root / "diffsynth/core/data/__init__.py",
        root / "diffsynth/pipelines/wan_video.py",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"DiffSynth-Studio source is incomplete: {missing}")
    if EXPECTED_VERSION not in required[1].read_text(encoding="utf-8"):
        raise SystemExit("unexpected DiffSynth-Studio version")

    # Inference imports `diffsynth.core`, whose training-only data export eagerly
    # imports pandas. Remove that optional surface from this inference runtime.
    data_init = required[3]
    original_data_init = data_init.read_text(encoding="utf-8")
    if original_data_init != "from .unified_dataset import UnifiedDataset\n":
        raise SystemExit("unexpected DiffSynth core data exports")
    data_init.write_text(
        '"""Training data helpers are excluded from the offline inference image."""\n',
        encoding="utf-8",
    )

    # ModelConfig.path is mandatory in this image. Keep the upstream loader API
    # but make every accidental downloader path fail before network access.
    config = required[2]
    replace_once(
        config,
        "from modelscope import snapshot_download\n"
        "from huggingface_hub import snapshot_download as hf_snapshot_download",
        "def snapshot_download(*args, **kwargs):\n"
        "    raise RuntimeError('network model loading is disabled')\n\n"
        "def hf_snapshot_download(*args, **kwargs):\n"
        "    raise RuntimeError('network model loading is disabled')",
    )

    value = config.read_text(encoding="utf-8")
    if "from modelscope" in value or "from huggingface_hub" in value:
        raise SystemExit("runtime downloader imports remain")


if __name__ == "__main__":
    main()
