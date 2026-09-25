from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import RecipeBuildDefinition

from qualification.coverage_identity import execution_stack_identity

ROOT = Path(__file__).resolve().parents[1]


def _recipe_for_root(root: Path) -> RecipeDefinition:
    document = json.loads(
        (
            ROOT
            / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
        ).read_text(encoding="utf-8")
    )
    document["execution"]["build"]["context"]["path"] = "context"
    document["execution"]["build"]["dockerfile"] = "Dockerfile"
    document["execution"]["build"]["patches"] = [{"path": "patches/runtime.patch"}]
    return RecipeDefinition.model_validate(document)


def _build_definition(recipe: RecipeDefinition) -> RecipeBuildDefinition:
    if recipe.execution.mode != "build":
        raise AssertionError("source-build example must retain build execution")
    return recipe.execution.build


def test_source_bytes_and_file_modes_invalidate_the_execution_stack() -> None:
    """Path-only grouping must not reuse recovery after any selected source edit."""

    # A private tree keeps each mutation independent from the checked-in recipe.
    with TemporaryDirectory() as temporary:
        repository = Path(temporary)
        context = repository / "context"
        patch_directory = repository / "patches"
        context.mkdir()
        patch_directory.mkdir()
        adapter = context / "adapter.py"
        dockerfile = repository / "Dockerfile"
        patch = patch_directory / "runtime.patch"
        adapter.write_bytes(b"adapter = 'first'\n")
        dockerfile.write_bytes(b"FROM pinned@sha256:abc\n")
        patch.write_bytes(b"patch one\n")

        recipe = _recipe_for_root(repository)
        build = _build_definition(recipe)
        initial = execution_stack_identity(repository, recipe, build)

        adapter.write_bytes(b"adapter = 'second'\n")
        after_adapter_change = execution_stack_identity(repository, recipe, build)
        assert (
            after_adapter_change["runtime_stack_sha256"]
            != initial["runtime_stack_sha256"]
        )
        assert (
            after_adapter_change["build_source_sha256"]
            != initial["build_source_sha256"]
        )

        dockerfile.write_bytes(b"FROM another-pinned@sha256:def\n")
        after_dockerfile_change = execution_stack_identity(repository, recipe, build)
        assert (
            after_dockerfile_change["runtime_stack_sha256"]
            != after_adapter_change["runtime_stack_sha256"]
        )

        patch.write_bytes(b"patch two\n")
        after_patch_change = execution_stack_identity(repository, recipe, build)
        assert (
            after_patch_change["runtime_stack_sha256"]
            != after_dockerfile_change["runtime_stack_sha256"]
        )

        previous_mode_digest = after_patch_change["build_source_sha256"]
        os.chmod(adapter, 0o755)
        after_mode_change = execution_stack_identity(repository, recipe, build)
        assert after_mode_change["build_source_sha256"] != previous_mode_digest


def test_symlinks_in_the_build_closure_fail_closed() -> None:
    """A silently omitted symlink must not produce an incomplete stack identity."""

    with TemporaryDirectory() as temporary:
        repository = Path(temporary)
        context = repository / "context"
        context.mkdir()
        (context / "adapter.py").write_text("pass\n", encoding="utf-8")
        (repository / "Dockerfile").write_text("FROM pinned\n", encoding="utf-8")
        (repository / "patches").mkdir()
        (repository / "patches/runtime.patch").write_text("patch\n", encoding="utf-8")
        (context / "linked.py").symlink_to(context / "adapter.py")
        recipe = _recipe_for_root(repository)
        build = _build_definition(recipe)

        with pytest.raises(ValueError, match="symlinks"):
            execution_stack_identity(repository, recipe, build)
