"""Run the pinned native Music3 engine on declared, immutable job inputs."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import wave

from vonk_agent_protocol.job_inputs import RecipeJobInputManifest


def inputs(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("job inputs must be a regular directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("a regular input manifest is required")
    manifest = RecipeJobInputManifest.model_validate_json(manifest_path.read_bytes())
    found = {}
    for item in manifest.files:
        if item.slot not in ("caption", "lyrics") or item.slot in found:
            raise ValueError("exactly one caption and at most one lyrics file are supported")
        path = root / item.name
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".txt" or item.media_type != "text/plain":
            raise ValueError("music inputs must be regular UTF-8 .txt files")
        if not 1 <= item.size_bytes <= 16384 or path.stat().st_size != item.size_bytes:
            raise ValueError("music input exceeds its declared byte bound")
        data = path.read_bytes()
        if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
            raise ValueError("music input digest or size differs from its manifest")
        text = data.decode("utf-8")
        if not text.strip() or "\x00" in text:
            raise ValueError("music input must contain nonblank, non-NUL text")
        found[item.slot] = path
    if "caption" not in found:
        raise ValueError("a music caption is required")
    if {p.name for p in root.iterdir()} != {x.name for x in manifest.files} | {"manifest.json"}:
        raise ValueError("undeclared music job input")
    return found


def run(root, output, seconds, seed, binary="/opt/tensorfold/music3-generate"):
    if not math.isfinite(seconds) or not 1 <= seconds <= 60 or not 0 <= seed < 2**64:
        raise ValueError("duration must be 1..60 seconds and seed an unsigned 64-bit integer")
    selected = inputs(root)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "song.wav"
    command = [binary, "--model", "/models/music3", "--caption-file", str(selected["caption"]),
               "--out", str(destination), "--seconds", str(seconds), "--seed", str(seed),
               "--weights", "int8", "--steps", "30", "--ar-cfg", "1.5", "--dit-cfg", "1.7"]
    if "lyrics" in selected:
        command += ["--lyrics-file", str(selected["lyrics"])]
    subprocess.run(command, check=True)
    with wave.open(str(destination), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (2, 2, 44100):
            raise ValueError("Music3 output is not PCM16 stereo at 44100 Hz")
        if not 0 < audio.getnframes() <= 60 * 44100:
            raise ValueError("Music3 output is empty or exceeds the recipe duration")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["text-to-audio"], required=True)
    parser.add_argument("--output-mime", choices=["audio/wav"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(Path("/inputs"), args.output_dir, args.seconds, args.seed)


if __name__ == "__main__":
    main()
