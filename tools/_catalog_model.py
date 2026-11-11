"""The Model document shared by the catalog authoring tools.

One owner for the identity, licence and format flags, the file roles and ids,
and the validated write of ``models/<version-slug>.json`` and the
``display/models/<version-slug>.json`` that carries its titles, description and tags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_MODALITIES = {"language": "text", "multimodal": "text"}


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--publisher",
        required=True,
        help="who publishes this checkpoint (the identity and display publisher)",
    )
    parser.add_argument(
        "--family-publisher",
        help="the original publisher of the model lineage (default: --publisher); "
        "set it when --publisher republished or quantized another creator's model",
    )
    parser.add_argument("--model-group-slug", required=True)
    parser.add_argument("--model-group-title", required=True)
    parser.add_argument("--model-group-description", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-title", required=True)
    parser.add_argument("--model-description", required=True)
    parser.add_argument(
        "--modality",
        action="append",
        help="a modality the model handles (repeatable, default: language)",
    )
    parser.add_argument("--version-slug", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--license-spdx", required=True)
    parser.add_argument("--license-url", required=True)
    parser.add_argument("--attribution", action="append", required=True)
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="a capability name the model supports (repeatable)",
    )


def slug(value: str) -> str:
    if not value:
        return "metadata"
    # Every character a file id cannot hold becomes "-" (e.g. "model-step=2000000").
    text = re.sub(r"[^a-z0-9-]", "-", value.lower())
    return text if text[:1].isalpha() else f"file-{text}"


def roles_for(path: str) -> list[str]:
    lowered = path.lower()
    if lowered.endswith(".safetensors") or "model-" in lowered:
        return ["weights"]
    if any(k in lowered for k in ("token", "vocab", "merges")) or lowered.endswith(
        ".jinja"
    ):
        return ["tokenizer"]
    if "config" in lowered or "generation" in lowered or lowered.endswith(".json"):
        return ["configuration"]
    if lowered.endswith(".py"):
        return ["runtime"]
    return ["metadata"]


def file_id(path: str, sha256: str) -> str:
    """Readable id bound to both content and full-path digests.

    Basenames and even basename/content pairs are not unique in a tree, so the
    id carries the content digest and the digest of the full path.
    """

    stem = slug(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    return f"{stem[:39]}-{sha256[:12]}-{sha256_hex(path.encode())[:10]}"


def write_model(
    args: argparse.Namespace,
    *,
    source: dict[str, str],
    files: list[dict[str, Any]],
    requires_token: bool,
) -> str:
    """Validate and write the Model; return its ``document_sha256``."""

    # Imported here so file_id stays usable without the contracts dependencies
    # (tools/kit-pins runs on a bare interpreter).
    from vonk_forge_contracts import (
        ModelDefinition,
        ModelDisplay,
        document_sha256,
        published_json,
    )

    modalities = list(
        dict.fromkeys(
            _MODALITIES.get(name, name) for name in args.modality or ["language"]
        )
    )
    for modality in modalities:
        if modality not in {"text", "image", "audio", "video", "3d", "embeddings"}:
            raise SystemExit(f"unsupported model modality: {modality}")
    lineage_publisher = args.family_publisher or args.publisher
    document = {
        "kind": "model",
        "identity": {
            "publisher": args.publisher,
            "slug": args.version_slug,
            "family": {
                "publisher": lineage_publisher,
                "slug": args.model_group_slug,
            },
            "model": {
                "publisher": lineage_publisher,
                "slug": args.model_slug,
            },
            "version": args.version,
            "variant": args.version,
        },
        "requires_token": requires_token,
        "dependencies": [],
        "source": source,
        "format": {
            "precision": args.precision,
            "quantization": args.quantization,
        },
        "files": files,
    }
    ModelDefinition.model_validate(document)
    display = ModelDisplay.model_validate(
        {
            "kind": "model-display",
            "identity": {"publisher": args.publisher, "slug": args.version_slug},
            "family_title": args.model_group_title,
            "model_title": args.model_title,
            "description": args.model_description,
            "tags": list(dict.fromkeys([lineage_publisher, args.model_group_slug])),
            "modalities": modalities,
            "capabilities": sorted(set(args.capability)),
            "license": {
                "spdx": args.license_spdx,
                "url": args.license_url,
                "attribution": args.attribution,
            },
        }
    )
    root = args.root.resolve()
    for path, payload in (
        (root / "models" / f"{args.version_slug}.json", document),
        (
            root / "display" / "models" / f"{args.version_slug}.json",
            published_json(display),
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    # Recipes reference this Model by the digest of the document as written.
    return document_sha256(document)
