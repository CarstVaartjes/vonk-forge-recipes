"""Keep pinned TRELLIS/Pixal pipeline sub-model loading local and fail fast."""

from __future__ import annotations

import argparse
from pathlib import Path

OLD = '''            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                _models[k] = models.from_pretrained(v)
'''
NEW = '''            model_path = os.path.abspath(os.path.join(path, v))
            for suffix in (".json", ".safetensors"):
                if not os.path.isfile(model_path + suffix):
                    raise FileNotFoundError(f"missing local sub-model file: {model_path}{suffix}")
            _models[k] = models.from_pretrained(model_path)
'''


def patch_tree(package_root: Path) -> None:
    path = package_root / "pipelines" / "base.py"
    source = path.read_text()
    count = source.count(OLD)
    if count != 1:
        raise SystemExit(f"expected one local model loader in {path}, found {count}")
    path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package_roots", nargs="+", type=Path)
    for root in parser.parse_args().package_roots:
        patch_tree(root)
