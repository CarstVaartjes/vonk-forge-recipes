import json
import os
from pathlib import Path


source = Path(
    os.environ.get(
        "DSPARK_MODEL_DIR", "/var/tmp/inkling-small-dspark-preview"
    )
).resolve()
overlay = Path(
    os.environ.get(
        "DSPARK_OVERLAY_DIR",
        "/var/tmp/inkling-small-dspark-preview-vllm",
    )
).resolve()

if source == overlay:
    raise SystemExit("overlay must differ from the source snapshot")

config = json.loads((source / "config.json").read_text())
config["architectures"] = ["Qwen3DSparkModel"]
config["n_predict"] = 7
config["dspark_block_size"] = 7

overlay.mkdir(parents=True, exist_ok=True)
(overlay / "config.json").write_text(json.dumps(config, indent=2) + "\n")

for name in ("README.md", "dflash.py", "dspark.py", "model.safetensors"):
    target = overlay / name
    if target.is_symlink() or target.exists():
        target.unlink()
    # Keep the overlay portable when both directories are bind-mounted as
    # siblings under a different container path (for example, /models).
    target.symlink_to(os.path.relpath(source / name, start=overlay))

print(overlay)
