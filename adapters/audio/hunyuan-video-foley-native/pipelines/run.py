"""Bounded offline adapter for the official HunyuanVideo-Foley inference code."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SOURCE = Path("/opt/hunyuan-video-foley/infer.py")
_ENTRYPOINTS = {
    "/opt/vonk/source/pipelines/hunyuan_foley_xl.py": (
        "xl",
        "hunyuanvideo_foley_xl.pth",
    ),
    "/opt/vonk/source/pipelines/hunyuan_foley_xxl.py": (
        "xxl",
        "hunyuanvideo_foley.pth",
    ),
}
_VIDEO_SUFFIXES = frozenset({".mkv", ".mov", ".mp4", ".webm"})


def _one_video(path: Path) -> Path:
    candidates = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in _VIDEO_SUFFIXES
    )
    if len(candidates) != 1:
        raise SystemExit(
            "Foley inference requires exactly one supported video in /inputs"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True, choices=tuple(_ENTRYPOINTS))
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    del args.timeout_seconds

    if args.output_mime != "audio/wav":
        raise SystemExit(
            "HunyuanVideo-Foley only emits audio/wav as its primary artifact"
        )
    model_size, checkpoint = _ENTRYPOINTS[args.entrypoint]
    model_root = Path("/models/target")
    required = [checkpoint, "synchformer_state_dict.pth", "vae_128d_48k.pth"]
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing:
        raise SystemExit(
            f"pinned Foley snapshot is missing required files: {', '.join(missing)}"
        )
    auxiliary = [Path("/models/siglip2/config.json"), Path("/models/clap/config.json")]
    missing_auxiliary = [str(path) for path in auxiliary if not path.is_file()]
    if missing_auxiliary:
        raise SystemExit(
            f"pinned Foley auxiliary snapshot is missing: {', '.join(missing_auxiliary)}"
        )
    video = _one_video(Path("/inputs"))
    if not _SOURCE.is_file():
        raise SystemExit("pinned HunyuanVideo-Foley source is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(args.seed)
    command = [
        sys.executable,
        str(_SOURCE),
        "--model_path",
        str(model_root),
        "--model_size",
        model_size,
        "--single_video",
        str(video),
        "--single_prompt",
        os.environ.get("VONK_PROMPT", "Natural synchronized ambient sound"),
        "--output_dir",
        str(args.output_dir),
        "--guidance_scale",
        "4.5",
        "--num_inference_steps",
        "50",
        "--device",
        "cuda",
        "--gpu_id",
        "0",
        "--enable_offload",
    ]
    subprocess.run(command, cwd=_SOURCE.parent, env=env, check=True)
    outputs = list(args.output_dir.glob("*_generated.wav"))
    if len(outputs) != 1 or outputs[0].stat().st_size == 0:
        raise SystemExit("upstream Foley inference did not produce one WAV artifact")


if __name__ == "__main__":
    main()
