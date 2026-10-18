"""Chat recipes must declare a template or select a chat-capable Model.

A tokenizer can contain a template even for base weights (DeepSeek-V2-Lite
does). That alone does not make the checkpoint trained for conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, read_recipe
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    read_recipe(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((ROOT / "recipes").glob("*.json"))
]
CHAT_RECIPES = [
    recipe
    for recipe in RECIPES
    if any(check.kind == "openai.chat" for check in recipe.validation.serving.checks)
]


def _has_chat_configuration(
    recipe: RecipeDefinition, models: dict[str, ModelDefinition]
) -> bool:
    if any(
        argument.name in {"chat-template", "chat_template"} and argument.value
        for argument in recipe.runtime.arguments
    ):
        return True
    # The catalog selects the served Model first, before companion/draft Models.
    selection = recipe.models[0]
    model = models[selection.model.content_sha256]
    if "chat" in model.capabilities:
        return True
    selected = {item.file_id for item in selection.files}
    return any(
        item.id in selected
        and Path(item.path).name in {"chat_template.jinja", "chat_template.json"}
        for item in model.files
    )


@pytest.mark.parametrize("recipe", CHAT_RECIPES, ids=lambda r: r.identity.slug)
def test_chat_recipe_declares_template_or_chat_model(recipe: RecipeDefinition) -> None:
    assert _has_chat_configuration(recipe, catalog_models()), (
        "chat serving requires an explicit chat template, a selected template file, "
        "or a Model with the chat capability; base weights are not chat weights"
    )


def test_chat_guard_rejects_a_base_model() -> None:
    recipe = next(
        r
        for r in CHAT_RECIPES
        if r.identity.slug == "deepseek-v2-lite-nvidia-sglang-single"
    )
    models = dict(catalog_models())
    digest = recipe.models[0].model.content_sha256
    models[digest] = models[digest].model_copy(
        update={"capabilities": ["text-generation"]}
    )
    assert not _has_chat_configuration(recipe, models)

    for name in ("chat-template", "chat_template"):
        explicit = recipe.model_copy(deep=True)
        explicit.runtime.arguments.append(
            RecipeRuntimeArgument(name=name, value="deepseek-chat")
        )
        assert _has_chat_configuration(explicit, models)
