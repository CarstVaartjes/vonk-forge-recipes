"""Actual Controller manifests must work with both container adapters."""
import hashlib
import runpy
from pathlib import Path

import pytest
from vonk_agent_protocol.job_inputs import RecipeJobInputManifest
from vonk_agent_protocol.recipe_jobs import RecipeJobInputFile, manifest_document

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("adapter", ["ltx2-sync-native", "ltx23-sync-native-disk"])
def test_declared_prompt_and_platform_metadata(adapter, tmp_path):
    read = runpy.run_path(str(ROOT / f"adapters/video/{adapter}/run.py"))["_load_prompt"]
    read.__globals__["INPUT_ROOT"] = tmp_path
    prompt = b"A quiet forest at sunrise"
    (tmp_path / "Prompt.TXT").write_bytes(prompt)
    item = RecipeJobInputFile(slot="prompt", name="Prompt.TXT", media_type="text/plain",
                              size_bytes=len(prompt), sha256=hashlib.sha256(prompt).hexdigest())
    manifest = RecipeJobInputManifest.model_validate(manifest_document((item,)))
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    assert read() == prompt.decode()
    (tmp_path / "undeclared.txt").write_text("not in the manifest")
    with pytest.raises(SystemExit, match="manifest"):
        read()
    (tmp_path / "undeclared.txt").unlink()
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(SystemExit, match="manifest"):
        read()
