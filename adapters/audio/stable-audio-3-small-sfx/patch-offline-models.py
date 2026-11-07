"""Replace every upstream inference-time model download with local-only loading."""

from __future__ import annotations

import argparse
from pathlib import Path

PATCHES = {
    "stable_audio_3/model_configs.py": {
        "        local_config = hf_hub_download(repo_id=self.repo_id, filename=self.config_path)\n"
        "        local_ckpt = hf_hub_download(repo_id=self.repo_id, filename=self.ckpt_path)\n": (
            '        local_config = "/models/" + self.config_path\n'
            '        local_ckpt = "/models/" + self.ckpt_path\n',
            1,
        ),
        "        local_config = hf_hub_download(\n"
        "            repo_id=self.ae_repo_id, filename=self.ae_config_path\n"
        "        )\n"
        "        local_ckpt = hf_hub_download(\n"
        "            repo_id=self.ae_repo_id, filename=self.ae_ckpt_path\n"
        "        )\n": (
            '        raise RuntimeError("runtime autoencoder downloads are disabled")\n',
            1,
        ),
    },
    "stable_audio_3/models/conditioners.py": {
        "load_from = model_path or repo_id or model_name": (
            'load_from = "/models/t5gemma-b-b-ul2"',
            1,
        ),
        'hf_kwargs = {"subfolder": subfolder} if subfolder else {}': (
            'hf_kwargs = {"local_files_only": True}',
            1,
        ),
    },
}


def patch_file(target: Path, replacements: dict[str, tuple[str, int]]) -> None:
    text = target.read_text(encoding="utf-8")
    for old, (new, expected) in replacements.items():
        actual = text.count(old)
        if actual != expected:
            raise SystemExit(f"expected {expected} copies of {old!r}, found {actual}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")
    verified = target.read_text(encoding="utf-8")
    for old, (new, expected) in replacements.items():
        if old in verified or verified.count(new) != expected:
            raise SystemExit(f"offline replacement failed for {old!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    for relative, replacements in PATCHES.items():
        patch_file(args.source / relative, replacements)


if __name__ == "__main__":
    main()
