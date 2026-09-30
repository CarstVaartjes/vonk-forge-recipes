from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/qwen3-8-flash-next-nvfp4-vllm-dual.json"
ADAPTER = ROOT / "adapters/qwen/flash-next-vllm-dual"


def test_dual_qwen_language_vocab_choices_resolve_to_bundled_extensions() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    vocab_option = next(
        option for option in recipe["options"] if option["name"] == "draft_vocab"
    )
    choices = vocab_option["choices"]
    default = next(choice for choice in choices if choice.get("default") is True)

    assert default["value"] == "reduced-47k"
    assert recipe["runtime"]["environment"]
    runtime_vocab = next(
        variable["value"]
        for variable in recipe["runtime"]["environment"]
        if variable["name"] == "VLLM_MTP_DRAFT_VOCAB"
    )
    assert runtime_vocab == "/etc/vllm-draft-vocab.txt"
    assert "COPY draft-vocab/ /opt/vonk/draft-vocab/" in (
        ADAPTER / "Dockerfile"
    ).read_text(encoding="utf-8")

    assets = {path.name: path for path in (ADAPTER / "draft-vocab").glob("*.txt")}
    language_choices = [
        choice
        for choice in choices
        if choice.get("env", {})
        .get("VLLM_MTP_DRAFT_VOCAB", "")
        .startswith("/opt/vonk/draft-vocab/")
    ]
    assert len(language_choices) == 7
    assert {
        Path(choice["env"]["VLLM_MTP_DRAFT_VOCAB"]).name for choice in language_choices
    } == set(assets)

    english_floor = {
        int(line)
        for line in (ADAPTER / "draft_vocab_en_code_47k.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    for asset in assets.values():
        ids = [int(line) for line in asset.read_text(encoding="utf-8").splitlines()]
        assert len(ids) == 65_536
        assert len(set(ids)) == len(ids)
        assert english_floor <= set(ids)
