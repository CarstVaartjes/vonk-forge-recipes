"""The creator's preprocessing and sampling, run once for one rig.

``run.py`` starts this module in its own process group with the scratch workspace as the
working directory (the creator names the T5 encoder ``t5-base`` and reads it relative to it)
and ``PYTHONPATH`` set to the creator's source tree and the Motion library. It calls the
creator's own steps in the creator's order:

1. ``process_skeleton`` (what ``python -m utils.process_new_skeleton`` calls) turns the rig's
   BVH clips into ``cond.npy``: the rig's topology, joint-name list, offsets and the
   mean/std of its motion features.
2. ``sample.generate.main`` (``python -m sample.generate``) samples motions for that rig from
   the pretrained checkpoint and writes one BVH per repetition.

The creator renders a stick-figure mp4 for every processed clip and every generated motion
(a visual sanity check, minutes of matplotlib per clip); ``plot_general_skeleton_3d_motion``
is replaced by a no-op in the two modules that call it, and nothing else is changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import BVH
import data_loaders.truebones.truebones_utils.motion_process as motion_process
import sample.generate as generate
from data_loaders.truebones.truebones_utils.param_utils import MAX_JOINTS

OBJECT = "Rig"


def _skip_plot(*_arguments: object, **_keywords: object) -> None:
    return None


def rig_joint_names(clips: list[Path]) -> list[str]:
    """The creator's own reading of each clip's joints; every clip must name the same ones."""

    found: list[str] | None = None
    for clip in clips:
        _animation, names, _frame_time = BVH.load(str(clip))
        if found is None:
            found = list(names)
        elif list(names) != found:
            raise SystemExit(f"{clip.name} names different joints than {clips[0].name}")
    assert found is not None
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--rest-pose", default="")
    parser.add_argument("--face-joints", nargs=4, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--motion-length-seconds", type=float, required=True)
    parser.add_argument("--num-repetitions", type=int, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    clips = sorted(args.bvh_dir.glob("*.bvh"))
    names = rig_joint_names(clips)
    if len(names) > MAX_JOINTS:
        raise SystemExit(f"the rig has {len(names)} joints; AnyTop takes at most {MAX_JOINTS}")
    missing = [joint for joint in args.face_joints if joint not in names]
    if missing:
        raise SystemExit(f"face joints {missing} are not joints of the rig: {names}")

    # Without a rest pose the empty value is the creator's own default: its find_tpos_path decides.
    motion_process.plot_general_skeleton_3d_motion = _skip_plot
    generate.plot_general_skeleton_3d_motion = _skip_plot
    motion_process.process_skeleton(
        OBJECT, str(args.bvh_dir), list(args.face_joints), str(args.save_dir), args.rest_pose
    )
    sys.argv = [
        "generate",
        "--model_path",
        str(args.checkpoint),
        "--object_type",
        OBJECT,
        "--cond_path",
        str(args.save_dir / "cond.npy"),
        "--num_repetitions",
        str(args.num_repetitions),
        "--motion_length",
        str(args.motion_length_seconds),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(args.output_dir),
        "--device",
        "0",
    ]
    generate.main()


if __name__ == "__main__":
    main()
