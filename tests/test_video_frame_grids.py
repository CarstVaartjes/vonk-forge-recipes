"""Frame counts a video adapter accepts are exactly what its engine renders.

An off-grid request is snapped up by the engine and then fails the adapter's
frame-count check after a full render, so every video-job case must ask for a
frame count on its adapter's grid, and the grid constants must match upstream.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
VIDEO = ROOT / "adapters/video"
FASTH3 = VIDEO / "drowzeys-fasth3-tensorfold/comfyui_job.py"
MINIMAX = VIDEO / "minimax-h3-modular-diffusers/minimax_h3.py"
MINIMAX_FL2VA = VIDEO / "minimax-h3-fl2va-modular-diffusers/minimax_h3.py"
GRID_NAMES = (
    "FRAME_GRID_STEP",
    "FRAME_GRID_OFFSET",
    "FRAME_GRID_MIN",
    "FRAME_GRID_MAX",
)

# Adapters that export their grid as module constants.
EXPORTED_GRID = {
    "fasth3-8step-v2-drowzeys-tensorfold-single": FASTH3,
    "minimax-h3-diffusers-single": MINIMAX,
    "minimax-h3-fl2va-diffusers-single": MINIMAX_FL2VA,
}
# Proven adapters keep their own validation (changing them would move a proven
# content identity). Their bounds and step are read from the adapter source:
# `1 <= args.num_frames <= BOUND or args.num_frames % STEP != OFFSET`.
HAND_COPIED_ADAPTERS = {
    "ltx2-pytorch": VIDEO / "ltx2-pytorch/pipelines/run.py",
    "hunyuan-video-15-native": VIDEO / "hunyuan-video-15-native/run.py",
    "mova-pytorch": VIDEO / "mova-pytorch/run.py",
}
# Adapters that render one fixed `FRAME_COUNT`; the case must ask for exactly it.
FIXED_COUNT_ADAPTERS = {
    "ltx2-sync-native": VIDEO / "ltx2-sync-native/run.py",
    "ltx23-sync-native-disk": VIDEO / "ltx23-sync-native-disk/run.py",
}
# Fixed ComfyUI/request-controlled workflows: no adapter frame-count knob.
NO_FRAME_KNOB = {
    "wan-2-2-ti2v-5b-comfyui-single",
    "wan-2-2-t2v-14b-comfyui-single",
    "wan-2-2-i2v-14b-comfyui-single",
    "wan-dancer-14b-disk-offload-pytorch-single",
    "wan-dancer-14b-pytorch-single",
}
CHECK = re.compile(
    r"1 <= args\.num_frames <= (\w+) or args\.num_frames % (\d+) != (\d+)"
)


def _grid(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in GRID_NAMES
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(values) == set(GRID_NAMES), f"{path} must export {GRID_NAMES}"
    return values


def _on_grid(grid: dict[str, int], frames: int) -> bool:
    return (
        grid["FRAME_GRID_MIN"] <= frames <= grid["FRAME_GRID_MAX"]
        and frames % grid["FRAME_GRID_STEP"] == grid["FRAME_GRID_OFFSET"]
    )


def _accepted(grid: dict[str, int]) -> list[int]:
    return [n for n in range(1, 500) if _on_grid(grid, n)]


def test_comfyui_grid_matches_pinned_upstream():
    source = (FIXTURES / "comfyui-minimax-h3-align.py.txt").read_text()
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - vendored upstream function text
    align = namespace["align_frame_count"]
    # ComfyUI 5a665890 nodes_minimax_h3.py: tooltip "trained range is ~124-362".
    assert "124-362" in source
    expected = [n for n in range(124, 363) if align(n) == n]
    assert _accepted(_grid(FASTH3)) == expected
    assert expected[0] == 124 and expected[-1] == 362


@pytest.mark.parametrize("path", [MINIMAX, MINIMAX_FL2VA])
def test_diffusers_grid_matches_pinned_upstream(path):
    namespace: dict[str, Any] = {}
    exec((FIXTURES / "diffusers-minimax-h3-align.py.txt").read_text(), namespace)  # noqa: S102
    align = namespace["align_num_frames"]
    duration: dict[str, Any] = {}
    body = (FIXTURES / "diffusers-minimax-h3-duration.py.txt").read_text()
    exec("class Components:\n" + body, duration)  # noqa: S102
    components = duration["Components"]()
    fps = 24
    # diffusers efabd60d: frames_per_chunk 17, latents_per_chunk 5, 5..15 s.
    expected = [
        n
        for n in range(1, 500)
        if align(n, 17, 5) == n
        and components.min_duration <= n / fps <= components.max_duration
    ]
    assert _accepted(_grid(path)) == expected
    assert expected[0] == 124 and expected[-1] == 345


def _video_cases():
    for path in sorted((ROOT / "qualification/recipes").glob("*.json")):
        spec = json.loads(path.read_text()).get("recipes", {})
        if spec.get("interface") != "video-job":
            continue
        for case in [spec, *spec.get("cases", [])]:
            video = next(
                (
                    a
                    for a in case.get("assertions", spec["assertions"])
                    if a["kind"] == "video-metadata"
                ),
                None,
            )
            if video is not None and "frame_count" in video:
                yield path.stem, video["frame_count"]


def _int_constant(path: Path, name: str) -> int:
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path} has no module constant {name}")


def _hand_copied(path: Path) -> tuple[int, int, int]:
    """(maximum, step, offset) as the adapter's own check states them."""
    match = CHECK.search(path.read_text())
    assert match, f"{path}: frame-count check not in the expected form"
    bound = match.group(1)
    maximum = int(bound) if bound.isdigit() else _int_constant(path, bound)
    return maximum, int(match.group(2)), int(match.group(3))


def _adapter_dir(slug: str) -> str | None:
    text = (ROOT / "recipes" / f"{slug}.json").read_text()
    names = set(re.findall(r"adapters/video/([A-Za-z0-9._-]+)", text))
    assert len(names) <= 1, (slug, names)
    return next(iter(names), None)


def test_every_video_case_asks_for_a_frame_count_on_its_adapter_grid():
    seen = set()
    for slug, frames in _video_cases():
        seen.add(slug)
        adapter = _adapter_dir(slug)
        if slug in EXPORTED_GRID:
            assert _on_grid(_grid(EXPORTED_GRID[slug]), frames), (slug, frames)
        elif adapter in HAND_COPIED_ADAPTERS:
            maximum, step, offset = _hand_copied(HAND_COPIED_ADAPTERS[adapter])
            assert 1 <= frames <= maximum and frames % step == offset, (slug, frames)
        elif adapter in FIXED_COUNT_ADAPTERS:
            fixed = _int_constant(FIXED_COUNT_ADAPTERS[adapter], "FRAME_COUNT")
            assert frames == fixed, (slug, frames, fixed)
        else:
            assert slug in NO_FRAME_KNOB, f"{slug}: declare its frame grid owner"
    assert set(EXPORTED_GRID) <= seen
    assert {"ltx2-sync-native", "ltx23-sync-native-disk"} <= {
        _adapter_dir(slug) for slug in seen
    }
