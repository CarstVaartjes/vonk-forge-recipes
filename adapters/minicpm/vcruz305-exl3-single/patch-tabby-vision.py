#!/usr/bin/env python3
"""Use ExLlamaV3's native single-device vision loader in the pinned TabbyAPI.

MiniCPM's vision modules are not a sequential text-style measuring graph.
See VISION_LOAD.md for the pinned source and failure evidence.
"""

import sys
from pathlib import Path

OLD = """        if self.use_vision:
            for value in self.vision_model.load_gen(
                reserve_per_device=None if self.gpu_split else self.autosplit_reserve,
                use_per_device=self.gpu_split or None,
                callback=progress_callback,
            ):
"""
NEW = """        if self.use_vision:
            # The single-device loader loads weights without text-shaped measuring forwards.
            vision_device = torch.device("cuda:0") if torch.cuda.device_count() == 1 else None
            for value in self.vision_model.load_gen(
                device=vision_device,
                reserve_per_device=(
                    None if self.gpu_split or vision_device is not None else self.autosplit_reserve
                ),
                use_per_device=None if vision_device is not None else self.gpu_split or None,
                callback=progress_callback,
            ):
"""


def patch_text(source: str) -> str:
    if source.count(OLD) != 1:
        raise ValueError("Pinned TabbyAPI vision load block changed; review the patch")
    return source.replace(OLD, NEW, 1)


if __name__ == "__main__":
    target = Path(sys.argv[1]) / "backends/exllamav3/model.py"
    target.write_text(patch_text(target.read_text()))
