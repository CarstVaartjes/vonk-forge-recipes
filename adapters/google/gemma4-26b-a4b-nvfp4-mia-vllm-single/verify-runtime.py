"""Install and verify the patched Gemma 4 module at image build time.

Upstream bind-mounts its gemma4.py over the image's vLLM copy. This script
copies the same file over that module, wherever the base image's vLLM lives.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
from pathlib import Path

PATCH = Path("/opt/vonk/build/gemma4.py")


def main() -> None:
    wrapper = Path("/opt/vonk/bin/vllm")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise SystemExit("the vLLM wrapper is missing or not executable")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is missing from the base image")
    target = (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor"
        / "models"
        / "gemma4.py"
    )
    if not target.is_file():
        raise SystemExit(f"{target} is missing from the base image")
    shutil.copyfile(PATCH, target)
    target.chmod(0o644)
    digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise SystemExit("the patched gemma4.py was not installed")
    print(f"installed {PATCH.name} sha256={digest} over {target}")


if __name__ == "__main__":
    main()
