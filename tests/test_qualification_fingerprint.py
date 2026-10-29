"""Proof follows what a recipe executes; authored display edits do not unprove it.

The fingerprint (``vonk_forge_contracts.qualification_fingerprint_sha256``) is consumed by the
status derivation, the sweep identity, the memory gate and the change gate; each
consumer is exercised here through its own entry point.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from fingerprint_support import (
    LATENT,
    PROVEN,
    ROOT,
    checkout,
    load,
    retitle,
    sweep_recipe,
)
from vonk_forge_contracts import document_sha256, qualification_fingerprint_sha256

import qualification.model_index as model_index_module
import spark_sweep.identity as sweep_identity
from qualification.model_index import ModelIndex, model_index
from spark_sweep.qualification import read_evidence

# --- the fingerprint itself ---------------------------------------------------


def test_display_edits_keep_the_fingerprint_and_execution_edits_change_it():
    recipe = load(LATENT)
    models = model_index(ROOT).documents
    fingerprint = qualification_fingerprint_sha256(recipe, models)
    assert qualification_fingerprint_sha256(retitle(recipe), models) == fingerprint
    resized = deepcopy(recipe)
    resized["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    assert qualification_fingerprint_sha256(resized, models) != fingerprint
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
        assert qualification_fingerprint_sha256(edited, models) != fingerprint, path


def test_model_labels_are_display_but_variant_version_and_files_are_not():
    recipe = load(LATENT)
    models = deepcopy(dict(model_index(ROOT).documents))
    fingerprint = qualification_fingerprint_sha256(recipe, models)
    reference = recipe["models"][0]["model"]
    original = models[reference["content_sha256"]]

    def with_model(change) -> str:
        edited = deepcopy(original)
        change(edited)
        digest = document_sha256(edited)
        models[digest] = edited
        moved = deepcopy(recipe)
        moved["models"][0]["model"]["content_sha256"] = digest
        return qualification_fingerprint_sha256(moved, models)

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
        qualification_fingerprint_sha256(recipe, models)


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
            if record.identity.qualification_fingerprint_sha256 is None:
                continue
            if document is None or record.identity.content_sha256 != document_sha256(
                document
            ):
                continue  # a superseded document: derived when it was current
            stored += 1
            assert (
                record.identity.qualification_fingerprint_sha256
                == index.fingerprint(document)
            )
    assert stored


# --- status derivation and the sweep identity --------------------------------------


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
    assert "Expecting" in index.unreadable[victim.name] or index.unreadable[victim.name]
    assert index.fingerprint(document) is None


def test_a_miss_is_remembered_until_the_recipe_changes(tmp_path, monkeypatch):
    root = checkout(tmp_path, PROVEN)
    monkeypatch.setattr(sweep_identity, "ROOT", root)
    document = load(PROVEN)
    path = root / "recipes" / f"{PROVEN}.json"
    recipe = sweep_recipe(document, "0" * 64)  # not the digest of the file
    reads: list[str] = []
    real_read = Path.read_text

    def read_text(self, *args, **kwargs):
        reads.append(self.name)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert (
        sweep_identity.content_identity(recipe).qualification_fingerprint_sha256 is None
    )
    first = len(reads)
    assert first
    sweep_identity.content_identity(recipe)
    assert len(reads) == first  # remembered: a stat, no read
    path.write_text(path.read_text() + "\n")  # the file changed: look again
    sweep_identity.content_identity(recipe)
    assert len(reads) > first


def test_catalog_pass_reads_and_hashes_each_model_once(monkeypatch):
    """Per-call work stays flat: one read of each Model, nothing recomputed after."""
    names = sorted(p.name for p in (ROOT / "models").glob("*.json"))
    documents = [load(p.stem) for p in sorted((ROOT / "recipes").glob("*.json"))]
    recipes = [sweep_recipe(d, document_sha256(d)) for d in documents]
    model_index_module._cache.clear()
    reads: list[str] = []
    loaded: list[str] = []  # documents hashed to key the index
    stripped: list[str] = []  # documents hashed to fingerprint
    real_read = Path.read_text
    real_hash = document_sha256

    def read_text(self, *args, **kwargs):
        reads.append(self.parent.name + "/" + self.name)
        return real_read(self, *args, **kwargs)

    def counting(into: list[str]):
        def count(document):
            into.append(document["kind"])
            return real_hash(document)

        return count

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(
        "vonk_forge_contracts.canonical.document_sha256", counting(stripped)
    )
    monkeypatch.setattr(model_index_module, "document_sha256", counting(loaded))
    monkeypatch.setattr(sweep_identity, "document_sha256", counting(loaded))

    first = [sweep_identity.content_identity(r) for r in recipes]
    assert sorted(n for n in reads if n.startswith("models/")) == [
        f"models/{n}" for n in names
    ]
    assert sum(1 for n in reads if n.startswith("recipes/")) == len(recipes)
    assert all(i.qualification_fingerprint_sha256 for i in first)
    # Each Model is hashed once to key the index and once stripped, never per recipe.
    assert loaded.count("model") == len(names)
    referenced = {m["model"]["content_sha256"] for d in documents for m in d["models"]}
    assert stripped.count("model") == len(referenced)
    assert stripped.count("recipe") == len(recipes)

    reads.clear()
    loaded.clear()
    stripped.clear()
    second = [sweep_identity.content_identity(r) for r in recipes]
    assert second == first
    assert reads == [] and loaded == [] and stripped == []


def test_index_is_content_addressed():
    index = model_index(ROOT)
    assert isinstance(index, ModelIndex)
    assert all(document_sha256(d) == k for k, d in index.documents.items())


# Library code and the two catalog consumers read Models only through the index.
# Other tools author or inspect single Model files by slug (refresh-upstream,
# kit-pins, check-model-families, check-recipe-model-references,
# check-upstream-drift, check-model-revision-holds, build-qualification-authority,
# generate-contract-examples, _catalog_model.py, build-family-aware-coverage,
# prepare-remote-code-smoke): they are not consumers of the catalog's fingerprints.
MODEL_PATH = re.compile(
    r"""/\s*["']models["']|["']models["']\s*\)\s*\.(?:glob|rglob|iterdir)"""
    r"""|joinpath\(\s*["']models["']"""
)


def test_one_owner_reads_models_and_computes_fingerprints():
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
            assert not MODEL_PATH.search(text), f"{path.name} reads Models itself"
        assert path == owner or "qualification_fingerprint_sha256(" not in text, (
            path.name
        )


# Identities in sweep state predate the fingerprint, and the fingerprint follows
# from content and Models: compare them with ``same_identity``, never ``==``.
RAW_IDENTITY_COMPARISON = re.compile(
    r"(?<![\w.])(?:\w+\.)*(?:identity|content_identity\(.*?\)|current_identity\(.*?\))"
    r"\s*[!=]="
)


def test_sweep_compares_identities_only_through_same_identity():
    for path in sorted((ROOT / "spark_sweep").glob("*.py")):
        if path.name == "identity.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "controller.identity" in line or "target_identity" in line:
                continue  # a Controller operation identity, not a recipe identity
            assert not RAW_IDENTITY_COMPARISON.search(line), (
                f"{path.name}:{number}: use same_identity"
            )
