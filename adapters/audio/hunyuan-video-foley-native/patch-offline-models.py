"""Replace every upstream inference-time model download with local-only loading."""

from __future__ import annotations

import argparse
from pathlib import Path

PATCHES = {
    "hunyuanvideo_foley/utils/model_utils.py": {
        'AutoModel.from_pretrained("google/siglip2-base-patch16-512")': (
            'AutoModel.from_pretrained("/models/siglip2", local_files_only=True)',
            2,
        ),
        'AutoTokenizer.from_pretrained("laion/larger_clap_general")': (
            'AutoTokenizer.from_pretrained("/models/clap", local_files_only=True)',
            2,
        ),
        'ClapTextModelWithProjection.from_pretrained("laion/larger_clap_general")': (
            "ClapTextModelWithProjection.from_pretrained("
            + '"/models/clap", local_files_only=True)',
            2,
        ),
    },
    "hunyuanvideo_foley/models/dac_vae/utils/__init__.py": {
        "response = requests.get(download_link)": (
            'raise RuntimeError("runtime DAC downloads are disabled")',
            1,
        ),
    },
    "hunyuanvideo_foley/models/synchformer/vit_helper.py": {
        "state_dict = torch.hub.load_state_dict_from_url("
        "url=default_cfgs[cfg.VIT.PRETRAINED_WEIGHTS])": (
            'raise RuntimeError("runtime Vision Transformer downloads are disabled")',
            1,
        ),
    },
    "hunyuanvideo_foley/models/synchformer/utils.py": {
        "from tqdm import tqdm\n\nPARENT_LINK": (
            "from tqdm import tqdm\n\n\n"
            + "def _runtime_download_disabled(_url):\n"
            + '    raise RuntimeError("runtime Synchformer downloads are disabled")\n\n\n'
            + "PARENT_LINK",
            1,
        ),
        "with requests.get(fname2link[path.name], stream=True) as r:": (
            "with _runtime_download_disabled(fname2link[path.name]) as r:",
            1,
        ),
    },
    "hunyuanvideo_foley/models/synchformer/ast_model.py": {
        "self.config = ASTConfig.from_pretrained(ckpt_path, revision=revision)": (
            'raise RuntimeError("runtime AST config downloads are disabled")',
            1,
        ),
        "full_model = ASTForAudioClassification.from_pretrained("
        "ckpt_path, revision=revision)": (
            'raise RuntimeError("runtime AST model downloads are disabled")',
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
