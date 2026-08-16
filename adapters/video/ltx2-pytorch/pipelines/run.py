from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--output-mime", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.entrypoint != "/opt/vonk/source/pipelines/run.py":
        raise SystemExit("unexpected pipeline entrypoint")
    if args.output_mime != "video/mp4":
        raise SystemExit("this candidate currently emits video/mp4")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python3",
        "-m",
        "ltx_pipelines.ti2vid_two_stages",
        "--checkpoint-path",
        "/models/ltx-2-19b-dev-fp4.safetensors",
        "--distilled-lora",
        "/models/ltx-2-19b-distilled-lora-384.safetensors",
        "0.8",
        "--spatial-upsampler-path",
        "/models/ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "--gemma-root",
        "/models/text_encoder",
        "--prompt",
        os.environ.get("VONK_PROMPT", "A small red fox running through a snowy forest"),
        "--output-path",
        str(args.output_dir / "ltx2.mp4"),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True, timeout=args.timeout_seconds)


if __name__ == "__main__":
    main()
