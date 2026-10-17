from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from pydantic import ValidationError
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest
INPUT_ROOT = Path("/inputs")
MAX_PROMPT_BYTES = 16384

def _load_prompt() -> str:
    if not INPUT_ROOT.is_dir() or INPUT_ROOT.is_symlink():
        raise SystemExit("/inputs must be a directory containing the declared job inputs")
    manifest_path = INPUT_ROOT / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SystemExit("a regular Vonk job input manifest is required")
    try:
        manifest = RecipeJobInputManifest.model_validate_json(
            manifest_path.read_bytes()
        )
    except (OSError, ValidationError) as error:
        raise SystemExit(f"invalid Vonk job input manifest: {error}") from error

    prompts = [item for item in manifest.files if item.slot == "prompt"]
    if len(manifest.files) != 1 or len(prompts) != 1 or prompts[0].media_type != "text/plain":
        raise SystemExit("exactly one declared UTF-8 prompt file is required")
    prompt_file = prompts[0]
    if Path(prompt_file.name).suffix.lower() != ".txt":
        raise SystemExit("exactly one regular UTF-8 .txt prompt file is required")

    declared_names = {item.name for item in manifest.files} | {"manifest.json"}
    entries = list(INPUT_ROOT.iterdir())
    if {entry.name for entry in entries} != declared_names:
        raise SystemExit("job input files do not match the manifest")
    prompt_path = INPUT_ROOT / prompt_file.name
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise SystemExit("the declared prompt must be a regular file")
    try:
        size = prompt_path.stat().st_size
    except OSError as error:
        raise SystemExit("the declared prompt file is unavailable") from error
    if size != prompt_file.size_bytes:
        raise SystemExit("prompt file size does not match the job manifest")
    if not 1 <= size <= MAX_PROMPT_BYTES:
        raise SystemExit("prompt.txt must contain 1..16384 UTF-8 bytes")
    try:
        payload = prompt_path.read_bytes()
    except OSError as error:
        raise SystemExit("the declared prompt file is unavailable") from error
    if len(payload) != prompt_file.size_bytes:
        raise SystemExit("prompt file size does not match the job manifest")
    if hashlib.sha256(payload).hexdigest() != prompt_file.sha256:
        raise SystemExit("prompt file digest does not match the job manifest")
    if not 1 <= len(payload) <= MAX_PROMPT_BYTES:
        raise SystemExit("prompt.txt must contain 1..16384 UTF-8 bytes")
    try:
        prompt = payload.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SystemExit("prompt.txt must contain valid UTF-8") from error
    if not prompt or "\x00" in prompt:
        raise SystemExit("prompt.txt must contain nonblank, non-NUL text")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["text-to-image"], required=True)
    parser.add_argument("--output-mime", choices=["image/png"], required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not all(64 <= n <= 2048 and n % 32 == 0 for n in (args.width, args.height)):
        raise SystemExit("canvas dimensions must be multiples of 32 in 64..2048")
    prompt = _load_prompt()
    from flux3.generate import Models, generate
    models = Models({"distilled": "/models/flux-3-image-klein"}, "/models/pup", 0.35)
    image, _ = generate(models, "distilled", prompt, [], width=args.width,
                        height=args.height, steps=16, seed=args.seed, pup=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "image.png", format="PNG")


if __name__ == "__main__":
    main()
