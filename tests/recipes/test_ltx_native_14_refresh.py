from __future__ import annotations

import json
import re
import runpy
import unittest
from pathlib import Path
from typing import Any

from documents import recipe_display
from generated_catalog import GENERATED
from repo_root import ROOT

from recipe_tooling.protocol_wheel import selection

RECIPES = {
    "ltx-2-19b-dev-fp4-pytorch-single": "adapters/video/ltx2-pytorch",
    "ltx-2-19b-dev-bf16-diffusers-single": "adapters/video/ltx23-sync-native-disk",
    "ltx-2-19b-distilled-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-19b-distilled-fp8-diffusers-single": "adapters/video/ltx2-sync-native",
    "ltx-2-3-22b-distilled-1-1-diffusers-single": "adapters/video/ltx23-sync-native-disk",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class LtxNative14RefreshTests(unittest.TestCase):
    def test_all_native_recipes_bind_source_bundle_and_package(self) -> None:
        tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        for slug, adapter in RECIPES.items():
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            context = recipe["execution"]["build"]["context"]
            assert context["path"] == adapter
            _, _, bundle_digest = tool["source_bundle"](ROOT / adapter)
            assert bundle_digest
            index = load(GENERATED / "catalog-index.json")
            entry = next(
                item
                for item in index["recipes"]
                if item["source_path"] == f"recipes/{slug}.json"
            )
            assert len(entry["package"]["recipe_content_sha256"]) == 64

    def test_historical_recipes_keep_explicit_candidate_metadata(self) -> None:
        for slug in RECIPES:
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            assert {"executable"} <= set(
                recipe_display(recipe["identity"]["slug"]).tags
            )
            assert recipe["models"]
            assert recipe["topology"]["node_count"] == 1

    def test_runtime_sources_are_pinned_and_pipeline_runners_verify_shape(self) -> None:
        for slug, adapter in RECIPES.items():
            root = ROOT / adapter
            dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
            runner = (
                root
                / ("pipelines/run.py" if adapter.endswith("ltx2-pytorch") else "run.py")
            ).read_text(encoding="utf-8")
            # The archive fetched is the commit the labels claim, and the digest the
            # label records is the one the download is checked against.
            commit = re.search(r"LTX-2/tar\.gz/([0-9a-f]{40})", dockerfile)
            digest = re.search(
                r'ltx2\.source-archive-sha256="([0-9a-f]{64})"', dockerfile
            )
            assert commit is not None, slug
            assert digest is not None, slug
            assert commit is not None
            assert digest is not None
            assert dockerfile.count(commit.group(1)) == 2, slug
            assert dockerfile.count(digest.group(1)) == 2, slug
            assert "sha256sum --check --strict" in dockerfile
            assert selection(ROOT).platform_source_commit in dockerfile
            assert selection(ROOT).artifact_sha256 in dockerfile
            assert "colour-science==0.4.7" in (root / "requirements.lock").read_text()
            assert "_verify_ltx_runtime_contract()" in runner, slug
            assert '"tiling_config",\n)' in runner, slug
            assert '"video_latent"' not in runner, slug


if __name__ == "__main__":
    unittest.main()


def test_native_images_resolve_one_compatible_pytorch_release() -> None:
    """torchaudio's native extension must use the torch ABI it was built for."""
    for adapter in set(RECIPES.values()):
        root = ROOT / adapter
        dockerfile = (root / "Dockerfile").read_text()
        torch = re.search(r"\btorch==([\d.]+)", dockerfile)
        audio = re.search(r"\btorchaudio==([\d.]+)", dockerfile)
        assert torch is not None
        assert audio is not None
        assert torch.group(1) == audio.group(1), adapter
        assert "--no-deps" not in dockerfile.split("COPY requirements.lock")[0]
        lock = (root / "requirements.lock").read_text()
        # Constraints from the selected cu130 torch wheel's METADATA. Keep the
        # subsequent requirements install from replacing its binary dependencies.
        assert "nvidia-cudnn-cu13==9.19.0.56" in lock
        assert "setuptools==81.0.0" in lock
        assert (
            'RUN python3 -c "import torchaudio; from transformers import PreTrainedModel; from ltx_pipelines.utils.types import PipelineOutput"'
            in dockerfile
        )
