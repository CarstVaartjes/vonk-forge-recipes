"""Shared fixtures for the display-versus-execution tests; imports only long-standing APIs."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

from vonk_forge_contracts import document_sha256

from spark_sweep.catalog import Recipe
from spark_sweep.catalog_contract import RuntimeImageIdentity

ROOT = Path(__file__).resolve().parents[1]
LATENT = "deepseek-v4-1-flash-1017-mia-sglang-triple"  # declared peak 124e9
PROVEN = "d3-mini-vllm-single"
SOURCE_BUNDLE = "3" * 64
IMAGE = RuntimeImageIdentity(build_key="1" * 64, image_digest="sha256:" + "2" * 64)


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
    """Only authored display fields differ."""
    edited = deepcopy(document)
    edited["metadata"]["title"] = "A readable title"
    edited["metadata"]["description"] = "A clearer description"
    edited["metadata"]["tags"] = ["renamed"]
    edited["release"]["version"] = "9.9"
    edited["provenance"]["attribution"] = ["Someone else"]
    return edited


def resize(document: dict) -> dict:
    """An execution change."""
    edited = deepcopy(document)
    edited["topology"]["roles"][0]["resources"]["memory"]["peak_bytes"] += 1
    return edited


def sweep_recipe(
    document: dict,
    digest: str,
    *,
    source_bundle: str | None = None,
    image: RuntimeImageIdentity | None = None,
) -> Recipe:
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
        source_bundle_sha256=source_bundle,
        runtime_image=image,
    )
