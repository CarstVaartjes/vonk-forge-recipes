"""The creator's text-to-motion runtime, run once for one prompt, seed and duration.

``run.py`` starts this module in its own process group with the scratch workspace as the
working directory (the creator's code reads ``ckpts/``, ``stats/`` and
``scripts/gradio/static/assets/dump_wooden/`` relative to it) and ``PYTHONPATH`` set to the
creator's source tree. The prompt rewriter and the duration estimator are off: that is the
creator's ``disable_prompt_engineering`` path, which never contacts a prompter host.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hymotion.utils.t2m_runtime import T2MRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    runtime = T2MRuntime(
        config_path=str(args.config),
        ckpt_name=str(args.checkpoint),
        disable_prompt_engineering=True,
    )
    # output_format="dict": the creator's own NPZ (poses, trans, Rh, betas, gender) without the
    # FBX conversion, which needs the x86-only Autodesk SDK.
    runtime.generate_motion(
        text=prompt,
        seeds_csv=str(args.seed),
        duration=args.duration_seconds,
        cfg_scale=args.cfg_scale,
        output_format="dict",
        original_text=prompt,
        output_dir=str(args.output_dir),
        output_filename=args.output_name,
    )


if __name__ == "__main__":
    main()
