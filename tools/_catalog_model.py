"""The Model document shared by the catalog authoring tools.

One owner for the identity, licence and format flags, the file roles and ids,
and the validated write of ``models/<version-slug>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_MODALITIES = {"language": "text", "multimodal": "text"}


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--model-group-slug", required=True)
    parser.add_argument("--model-group-title", required=True)
    parser.add_argument("--model-group-description", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-title", required=True)
    parser.add_argument("--model-description", required=True)
    parser.add_argument("--modality", default="language")
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
    text = value.lower().replace("_", "-").replace(".", "-").replace("/", "-")
    return text if text[:1].isalpha() else f"file-{text}"


def roles_for(path: str) -> list[str]:
    lowered = path.lower()
    if lowered.endswith(".safetensors") or "model-" in lowered:
        return ["weights"]
    if "token" in lowered or "vocab" in lowered or lowered.endswith(".jinja"):
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
    from vonk_forge_contracts import ModelDefinition, document_sha256

    modality = _MODALITIES.get(args.modality, args.modality)
    if modality not in {"text", "image", "audio", "video", "3d", "embeddings"}:
        raise SystemExit(f"unsupported model modality: {args.modality}")
    document = {
        "kind": "model",
        "identity": {
            "publisher": args.publisher,
            "slug": args.version_slug,
            "family": {
                "publisher": args.publisher,
                "slug": args.model_group_slug,
                "title": args.model_group_title,
            },
            "model": {
                "publisher": args.publisher,
                "slug": args.model_slug,
                "title": args.model_title,
            },
            "version": args.version,
            "variant": args.version,
        },
        "metadata": {
            "description": args.model_description,
            "tags": list(
                dict.fromkeys(
                    [args.publisher, args.model_group_slug, args.modality, "candidate"]
                )
            ),
        },
        "requires_token": requires_token,
        "dependencies": [],
        "modalities": [modality],
        "source": source,
        "format": {
            "precision": args.precision,
            "quantization": args.quantization,
        },
        "license": {
            "spdx": args.license_spdx,
            "url": args.license_url,
            "attribution": args.attribution,
        },
        "files": files,
        "capabilities": sorted(set(args.capability)),
    }
    ModelDefinition.model_validate(document)
    model_path = args.root.resolve() / "models" / f"{args.version_slug}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    # Recipes reference this Model by the digest of the document as written.
    return document_sha256(document)
