"""Keep the plain OCR path independent of speculative-decoding support."""

from pathlib import Path

from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]


def test_plain_hunyuanocr_resolves_only_target_files() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/hunyuanocr-1-5-vllm-single.json").read_text()
    )
    assert recipe.topology.node_count == 1
    assert len(recipe.models) == 1
    selection = recipe.models[0]
    model = catalog_models()[selection.model.content_sha256]
    files = {file.id: file for file in model.files}
    selected = [files[file.file_id].path for file in selection.files]
    assert "model.safetensors" in selected
    assert "config.json" in selected
    assert all("/" not in path for path in selected)
    assert all(arg.name != "speculative-config" for arg in recipe.runtime.arguments)
    assert recipe.runtime.entrypoint == ["/opt/vonk/bin/vllm", "serve", "/models"]


def test_plain_hunyuanocr_exercises_image_input() -> None:
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/hunyuanocr-1-5-vllm-single.json").read_text()
    )
    # Use the authored serving request: a health-only addition cannot pass.
    checks = recipe.validation.serving.checks
    chat = next(check for check in checks if check.kind == "openai.chat")
    request = chat.request.model_dump(mode="json")
    content = request["body"]["messages"][0]["content"]
    assert any(part["type"] == "image_url" for part in content)
    assert any(part["type"] == "text" for part in content)
    assert request["body"]["model"] == recipe.identity.slug
