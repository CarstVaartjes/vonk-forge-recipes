"""Proof follows what a recipe executes; authored display edits do not unprove it.

The fingerprint (``vonk_forge_contracts.execution_sha256``) is consumed by the
status derivation, the sweep identity, the memory gate and the change gate; each
consumer is exercised here through its own entry point.
"""

import json
import runpy
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import (
    QualificationStatus,
    RecipeDefinition,
    derive_status,
    document_sha256,
    execution_sha256,
)

import qualification.memory_need as memory
import qualification.model_index as model_index_module
import spark_sweep.identity as sweep_identity
from qualification.model_index import ModelIndex, model_index
from spark_sweep.catalog import Recipe
from spark_sweep.qualification import read_evidence

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))
LATENT = "deepseek-v4-1-flash-1017-mia-sglang-triple"  # declared peak 124e9
PROVEN = "d3-mini-vllm-single"


def load(slug: str) -> dict:
    return json.loads((ROOT / "recipes" / f"{slug}.json").read_text())


def checkout(tmp_path: Path, slug: str) -> Path:
    """A repository root holding one recipe, every Model and its evidence."""
    (tmp_path / "recipes").mkdir()
    shutil.copy(ROOT / "recipes" / f"{slug}.json", tmp_path / "recipes")
    shutil.copytree(ROOT / "models", tmp_path / "models")
    shutil.copytree(
        ROOT / "qualification/evidence", tmp_path / "qualification/evidence"
    )
    return tmp_path


def write(root: Path, slug: str, document: dict) -> str:
    (root / "recipes" / f"{slug}.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    return document_sha256(document)


def retitle(document: dict) -> dict:
    edited = deepcopy(document)
    edited["metadata"]["title"] = "A readable title"
    edited["metadata"]["description"] = "A clearer description"
    edited["metadata"]["tags"] = ["renamed"]
    edited["release"]["version"] = "9.9"
    edited["provenance"]["attribution"] = ["Someone else"]
    return edited


def sweep_recipe(document: dict, digest: str) -> Recipe:
    return Recipe(
        key=f"{document['identity']['publisher']}/{document['identity']['slug']}",
        title="t",
        node_count=document["topology"]["node_count"],
        usage=(),
        engine="vllm",
        creator="",
        model_digests=tuple(
            sorted({m["model"]["content_sha256"] for m in document["models"]})
        ),
        model_selectors=(),
        memory_bytes=0,
        image_bytes=0,
        artifact_bytes=0,
        ports=(),
        adapter="openai",
        aliases=(),
        content_sha256=digest,
        revision_id="",
        local="",
        updated_at="",
        base_image="",
    )


# --- the fingerprint itself ---------------------------------------------------


def test_display_edits_keep_the_fingerprint_and_execution_edits_change_it():
    recipe = load(LATENT)
    models = model_index(ROOT).documents
    fingerprint = execution_sha256(recipe, models)
    assert execution_sha256(retitle(recipe), models) == fingerprint
    resized = deepcopy(recipe)
    resized["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    assert execution_sha256(resized, models) != fingerprint
    for path, value in (
        (("metadata", "alignment"), "abliterated"),
        (("provenance", "source_reference"), "https://example.invalid/other"),
        (("metadata", "a_future_field"), "x"),  # unknown fields stay covered
    ):
        edited = deepcopy(recipe)
        target = edited
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value
        assert execution_sha256(edited, models) != fingerprint, path


def test_model_labels_are_display_but_variant_version_and_files_are_not():
    recipe = load(LATENT)
    models = deepcopy(dict(model_index(ROOT).documents))
    fingerprint = execution_sha256(recipe, models)
    reference = recipe["models"][0]["model"]
    original = models[reference["content_sha256"]]

    def with_model(change) -> str:
        edited = deepcopy(original)
        change(edited)
        digest = document_sha256(edited)
        models[digest] = edited
        moved = deepcopy(recipe)
        moved["models"][0]["model"]["content_sha256"] = digest
        return execution_sha256(moved, models)

    def labels(model: dict) -> None:
        model["identity"]["family"]["title"] = "Readable family"
        model["identity"]["model"]["title"] = "Readable model"
        model["metadata"]["description"] = "Readable"
        model["metadata"]["tags"] = ["readable"]

    assert with_model(labels) == fingerprint
    # The Controller selects Model files by variant and lineage uses version.
    assert with_model(lambda m: m["identity"].update(variant="other")) != fingerprint
    assert with_model(lambda m: m["identity"].update(version="other")) != fingerprint
    assert with_model(lambda m: m["files"][0].update(sha256="0" * 64)) != fingerprint


def test_substituted_model_bytes_are_refused():
    recipe = load(LATENT)
    models = deepcopy(dict(model_index(ROOT).documents))
    models[recipe["models"][0]["model"]["content_sha256"]]["source"]["revision"] = (
        "0" * 40
    )
    with pytest.raises(ValueError, match="different digest"):
        execution_sha256(recipe, models)


# --- stored evidence ------------------------------------------------------------


def test_stored_evidence_fingerprints_recompute():
    """A fingerprint is derived from the recipe bytes its record names, never invented."""
    index = model_index(ROOT)
    documents = {}
    for path in (ROOT / "recipes").glob("*.json"):
        document = json.loads(path.read_text())
        documents[
            f"{document['identity']['publisher']}/{document['identity']['slug']}"
        ] = document
    stored = 0
    for path in sorted((ROOT / "qualification/evidence").glob("*/*.json")):
        evidence = read_evidence(ROOT, json.loads(path.read_text())["recipe_key"])
        document = documents.get(evidence.recipe_key)
        for record in evidence.records:
            if record.identity.execution_sha256 is None:
                continue
            stored += 1
            assert document is not None, evidence.recipe_key
            assert record.identity.content_sha256 == document_sha256(document)
            assert record.identity.execution_sha256 == index.fingerprint(document)
    assert stored


# --- status derivation and the sweep identity --------------------------------------


def test_sweep_identity_keeps_proof_through_a_display_edit(tmp_path, monkeypatch):
    root = checkout(tmp_path, PROVEN)
    document = load(PROVEN)
    key = f"vonk-forge/{PROVEN}"
    monkeypatch.setattr(sweep_identity, "ROOT", root)

    def status(doc: dict) -> QualificationStatus:
        digest = write(root, PROVEN, doc)
        identity = sweep_identity.current_identity(sweep_recipe(doc, digest))
        return derive_status(read_evidence(ROOT, key), identity, 1, False).status

    assert status(document) == QualificationStatus.WORKING
    assert status(retitle(document)) == QualificationStatus.WORKING
    changed = deepcopy(document)
    changed["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    assert status(changed) != QualificationStatus.WORKING


def test_unreadable_model_is_a_miss_not_an_error(tmp_path):
    root = checkout(tmp_path, PROVEN)
    document = load(PROVEN)
    digest = document["models"][0]["model"]["content_sha256"]
    victim = next(
        p
        for p in (root / "models").glob("*.json")
        if document_sha256(json.loads(p.read_text())) == digest
    )
    victim.write_text("{ not json")
    index = model_index(root)
    assert victim.name in index.unreadable
    assert index.fingerprint(document) is None


def test_catalog_pass_reads_and_hashes_each_model_once(monkeypatch):
    """Per-call work stays flat: one read of each Model, nothing recomputed after."""
    names = sorted(p.name for p in (ROOT / "models").glob("*.json"))
    documents = [load(p.stem) for p in sorted((ROOT / "recipes").glob("*.json"))]
    recipes = [sweep_recipe(d, document_sha256(d)) for d in documents]
    model_index_module._cache.clear()
    reads: list[str] = []
    hashes: list[int] = []
    real_read = Path.read_text
    real_hash = document_sha256

    def read_text(self, *args, **kwargs):
        reads.append(self.parent.name + "/" + self.name)
        return real_read(self, *args, **kwargs)

    def counting_hash(document):
        hashes.append(1)
        return real_hash(document)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr("vonk_forge_contracts.canonical.document_sha256", counting_hash)
    monkeypatch.setattr(model_index_module, "document_sha256", counting_hash)
    monkeypatch.setattr(sweep_identity, "document_sha256", counting_hash)

    first = [sweep_identity.content_identity(r) for r in recipes]
    assert sorted(n for n in reads if n.startswith("models/")) == [
        f"models/{n}" for n in names
    ]
    assert sum(1 for n in reads if n.startswith("recipes/")) == len(recipes)
    assert all(i.execution_sha256 for i in first)

    reads.clear()
    hashes.clear()
    second = [sweep_identity.content_identity(r) for r in recipes]
    assert second == first
    assert reads == [] and hashes == []


# --- the memory gate ------------------------------------------------------------


def test_memory_gate_ignores_display_edits_but_not_execution_edits(
    tmp_path, monkeypatch
):
    root = checkout(tmp_path, LATENT)
    monkeypatch.setattr(memory, "ROOT", root)
    inputs = memory.load_inputs()
    document = load(LATENT)

    def gate(doc: dict) -> None:
        digest = write(root, LATENT, doc)
        recipe = RecipeDefinition.model_validate(doc)
        memory.require_memory_need(
            recipe, catalog_models(), inputs, document_digest=digest
        )

    gate(document)  # recorded content: judged by hardware, not by this check
    gate(retitle(document))  # a title edit is still that recorded content
    changed = deepcopy(document)
    changed["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    with pytest.raises(ValueError, match="not admissible"):
        gate(changed)


# --- the change gate (--base) ---------------------------------------------------


def git(root: Path, *args: str) -> None:
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
    changed = retitle(document)
    changed["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    write(root, LATENT, changed)
    assert touched(root, "HEAD") == [root / "recipes" / f"{LATENT}.json"]
    # A new recipe has no base to compare against.
    write(root, LATENT, document)
    copy = deepcopy(document)
    copy["identity"]["slug"] = "a-new-recipe"
    write(root, "a-new-recipe", copy)
    assert touched(root, "HEAD") == [root / "recipes" / "a-new-recipe.json"]


def test_index_is_content_addressed():
    index = model_index(ROOT)
    assert isinstance(index, ModelIndex)
    assert all(document_sha256(d) == k for k, d in index.documents.items())


def test_one_owner_reads_models_and_computes_fingerprints():
    """Class guard: consumers go through qualification.model_index, never a private loader."""
    sources = [
        *(ROOT / "spark_sweep").glob("*.py"),
        *(ROOT / "qualification").glob("*.py"),
        ROOT / "tools/build-catalog-index",
        ROOT / "tools/check-recipe-memory-need",
    ]
    owner = ROOT / "qualification/model_index.py"
    for path in sources:
        text = path.read_text()
        if path != owner:
            assert '"models").glob(' not in text, f"{path.name} reads Models itself"
        assert path == owner or "execution_sha256(" not in text, path.name
