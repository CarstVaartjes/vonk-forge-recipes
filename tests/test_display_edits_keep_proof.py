"""Entry points treat a display edit as the same recipe and an execution edit as new.

Imports nothing introduced with the qualification fingerprint, so each test runs
unchanged against the code before it and fails there by behaviour.
"""

import runpy
import subprocess
from copy import deepcopy

import pytest
from catalog_documents import catalog_models
from fingerprint_support import (
    IMAGE,
    LATENT,
    PROVEN,
    ROOT,
    SOURCE_BUNDLE,
    checkout,
    load,
    resize,
    retitle,
    sweep_recipe,
    write,
)
from vonk_forge_contracts import (
    QualificationStatus,
    RecipeDefinition,
    RecipeQualificationEvidence,
    derive_status,
)

import qualification.memory_need as memory
import spark_sweep.identity as sweep_identity
from spark_sweep.identity import content_identity, current_identity, verdict_matches
from spark_sweep.qualification import read_evidence

TOOL = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))
KEY = f"vonk-forge/{PROVEN}"


def proven_on(recipe) -> RecipeQualificationEvidence:
    """A real passing record, re-bound to the identity of ``recipe`` as the sweep sees it."""
    template = read_evidence(ROOT, KEY).records[-1]
    record = template.model_copy(update={"identity": current_identity(recipe)})
    return RecipeQualificationEvidence(recipe_key=KEY, records=(record,))


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = checkout(tmp_path, PROVEN)
    monkeypatch.setattr(sweep_identity, "ROOT", root, raising=False)
    document = load(PROVEN)

    def recipe(doc: dict, **bindings):
        return sweep_recipe(doc, write(root, PROVEN, doc), **bindings)

    return document, recipe


def status(evidence, recipe) -> QualificationStatus:
    return derive_status(evidence, current_identity(recipe), 1, False).status


BOUND = {"source_bundle": SOURCE_BUNDLE, "image": IMAGE}


def test_display_edit_keeps_proof_when_source_and_image_are_bound(world):
    document, recipe = world
    evidence = proven_on(recipe(document, **BOUND))
    assert status(evidence, recipe(document, **BOUND)) == QualificationStatus.WORKING
    assert (
        status(evidence, recipe(retitle(document), **BOUND))
        == QualificationStatus.WORKING
    )
    assert (
        status(evidence, recipe(resize(document), **BOUND))
        != QualificationStatus.WORKING
    )


def test_changed_source_or_image_behind_a_display_bump_needs_a_recheck(world):
    document, recipe = world
    evidence = proven_on(recipe(document, **BOUND))
    other_image = IMAGE.model_copy(update={"image_digest": "sha256:" + "9" * 64})
    for changed in (
        {"source_bundle": "4" * 64, "image": IMAGE},
        {"source_bundle": SOURCE_BUNDLE, "image": other_image},
    ):
        assert (
            status(evidence, recipe(retitle(document), **changed))
            == QualificationStatus.RECHECK
        )


def test_proof_without_bound_source_and_image_matches_only_its_exact_document(world):
    document, recipe = world
    evidence = proven_on(recipe(document))
    assert status(evidence, recipe(document)) == QualificationStatus.WORKING
    assert status(evidence, recipe(retitle(document))) == QualificationStatus.RECHECK


def test_a_recorded_verdict_follows_the_same_rule_as_the_status(world):
    """Removal frees a recipe's Models only while its verdict matches (removal.py)."""
    document, recipe = world
    entry = {
        "verdict_identity": content_identity(recipe(document, **BOUND)).model_dump(
            mode="json"
        )
    }
    assert verdict_matches(entry, recipe(document, **BOUND))
    assert verdict_matches(entry, recipe(retitle(document), **BOUND))
    assert not verdict_matches(entry, recipe(resize(document), **BOUND))
    assert not verdict_matches(
        entry, recipe(retitle(document), source_bundle="4" * 64, image=IMAGE)
    )
    unbound = {
        "verdict_identity": content_identity(recipe(document)).model_dump(mode="json")
    }
    assert not verdict_matches(unbound, recipe(retitle(document)))


def test_memory_gate_ignores_display_edits_but_not_execution_edits(
    tmp_path, monkeypatch
):
    root = checkout(tmp_path, LATENT)
    monkeypatch.setattr(memory, "ROOT", root)
    inputs = memory.load_inputs()
    document = load(LATENT)

    def gate(doc: dict) -> None:
        digest = write(root, LATENT, doc)
        memory.require_memory_need(
            RecipeDefinition.model_validate(doc),
            catalog_models(),
            inputs,
            document_digest=digest,
        )

    gate(document)  # recorded content: judged by hardware, not by this check
    gate(retitle(document))  # a title edit is still that recorded content
    with pytest.raises(ValueError, match="not admissible"):
        gate(resize(document))


def git(root, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_change_gate_touches_only_execution_edits(tmp_path):
    root = checkout(tmp_path, LATENT)
    for command in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "t@example.invalid"),
        ("config", "user.name", "t"),
        ("add", "."),
        ("commit", "-q", "-m", "base", "--no-gpg-sign"),
    ):
        git(root, *command)
    document = load(LATENT)
    touched = TOOL["touched_recipes"]

    assert touched(root, "HEAD") == []
    write(root, LATENT, retitle(document))
    assert touched(root, "HEAD") == []  # display-only: no execution guard applies
    write(root, LATENT, resize(retitle(document)))
    assert touched(root, "HEAD") == [root / "recipes" / f"{LATENT}.json"]
    # A new recipe has no base to compare against.
    write(root, LATENT, document)
    copy = deepcopy(document)
    copy["identity"]["slug"] = "a-new-recipe"
    write(root, "a-new-recipe", copy)
    assert touched(root, "HEAD") == [root / "recipes" / "a-new-recipe.json"]
