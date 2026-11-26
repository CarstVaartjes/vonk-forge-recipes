"""Plan one recipe image per distinct build key, from recipe inputs only.

Runs on a built library (``tools/build-catalog-index`` wrote ``catalog-index.json``
and ``packages/``). For every distinct key it writes ``<output>/<key>/context/``,
the recipe's build context exactly as its package carries it, and returns the
``podman build`` arguments and the recipes that share the image.

The key names the inputs that produce the image bytes: the source bundle, the
pinned base images, the base image and Dockerfile the recipe declares, and this
repository's build policy (``policy/build-policy.json``). It has no platform input:
a Vonk Forge release never changes it, and no consumer computes it. The Controller
runs the digest the signed catalog pins for a recipe's content; the key is only
this repository's handle for finding or skipping a build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from recipe_tooling.paths import REPO_ROOT

KEY_SCHEMA_VERSION = 2
POLICY = REPO_ROOT / "policy" / "build-policy.json"


def _canonical(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def build_key(
    policy: Mapping[str, Any],
    *,
    source_bundle_sha256: str,
    base_images: list[Mapping[str, Any]],
    base_image: Mapping[str, Any],
    dockerfile: str,
) -> str:
    """The recipe-owned build key (v2): recipe inputs and the build policy only."""

    return hashlib.sha256(
        _canonical(
            {
                "schema_version": KEY_SCHEMA_VERSION,
                "artifact_format": policy["artifact_format"],
                "source_bundle_sha256": source_bundle_sha256,
                "base_images": base_images,
                "base_image": base_image,
                "dockerfile": dockerfile,
                "options": policy["build_options"],
                "security": policy["build_security"],
            }
        )
    ).hexdigest()


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and "\0" not in value:
        return value
    raise ValueError("build environment value is not a scalar")


def podman_build_arguments(
    policy: Mapping[str, Any], *, dockerfile: str, context: str
) -> list[str]:
    """The ``podman build`` arguments of the policy (the Spark builder's flags)."""

    options = policy["build_options"]
    processes = int(policy["build_resources"]["processes"])
    arguments = [
        "--no-cache",
        "--pull=never",
        "--platform",
        "linux/arm64",
        "--file",
        f"{context}/{dockerfile}",
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        f"--ulimit=nproc={processes}:{processes}",
        f"--format={options['format']}",
        f"--identity-label={_scalar(options['identity_label'])}",
        f"--jobs={options['jobs']}",
        "--disable-compression=" + _scalar(options["layer_compression"] == "disabled"),
        f"--layers={_scalar(options['layers'])}",
        f"--no-hostname={_scalar(options['no_hostname'])}",
        f"--no-hosts={_scalar(options['no_hosts'])}",
        f"--omit-history={_scalar(options['omit_history'])}",
        f"--shm-size={options['shm_bytes']}",
        f"--skip-unused-stages={_scalar(options['skip_unused_stages'])}",
    ]
    for item in options["additional_contexts"]:
        arguments += ["--build-context", f"{item['name']}={context}/{item['path']}"]
    for flag, entries in (
        ("--annotation", options["annotations"]),
        ("--label", options["labels"]),
        ("--layer-label", options["layer_labels"]),
    ):
        for entry in entries:
            arguments += [flag, f"{entry['name']}={entry['value']}"]
    for variable in options["environment"]:
        arguments += ["--env", f"{variable['name']}={_scalar(variable['value'])}"]
    if options["ignorefile"] is not None:
        arguments += ["--ignorefile", f"{context}/{options['ignorefile']}"]
    for feature in options["os_features"]:
        arguments += ["--os-feature", str(feature)]
    if options["os_version"] is not None:
        arguments += ["--os-version", str(options["os_version"])]
    if options["squash"] == "new":
        arguments.append("--squash")
    elif options["squash"] == "all":
        arguments.append("--squash-all")
    if options["timestamp"] is not None:
        arguments.append(f"--timestamp={options['timestamp']}")
    for name in options["unset_environment"]:
        arguments += ["--unsetenv", str(name)]
    for name in options["unset_labels"]:
        arguments += ["--unsetlabel", str(name)]
    arguments += [
        f"--cap-add={capability}"
        for capability in policy["build_security"]["capabilities"]
    ]
    return arguments


def _package(path: Path, context: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The package manifest and its build-context files, relative to the context."""

    prefix = context.rstrip("/") + "/"
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.extractfile("manifest.json")
        if manifest_member is None:
            raise ValueError("package has no manifest")
        manifest = json.load(manifest_member)
        for member in archive.getmembers():
            if not (member.isfile() and member.name.startswith(prefix)):
                continue
            relative = PurePosixPath(member.name.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe build context path {member.name}")
            payload = archive.extractfile(member)
            assert payload is not None
            files[relative.as_posix()] = payload.read()
    if not files:
        raise ValueError("package has no build context")
    return manifest, files


def write_library_image_plan(
    library_root: Path, output: Path, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Plan one image per distinct build key for a built recipe library.

    A recipe whose package cannot be planned is listed under ``skipped`` with
    the reason; one recipe never blocks the others.
    """

    index = json.loads((library_root / "catalog-index.json").read_text("utf-8"))
    images: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for entry in index.get("recipes", []):
        document = entry.get("document", {})
        slug = str(document.get("identity", {}).get("slug", ""))
        try:
            build = document["execution"]["build"]
            context = str(build["context"]["path"])
            manifest, files = _package(
                library_root / str(entry["package"]["path"]), context
            )
            dockerfile = str(build["dockerfile"]).removeprefix(
                context.rstrip("/") + "/"
            )
            if dockerfile not in files:
                raise ValueError(f"build context has no {dockerfile}")
            base_images = list(manifest["build_inputs"])
            key = build_key(
                policy,
                source_bundle_sha256=str(entry["source_bundle_sha256"]),
                base_images=base_images,
                base_image=build["base_image"],
                dockerfile=dockerfile,
            )
            release = document.get("release") or {}
            recipe = {
                "publisher": document["identity"]["publisher"],
                "slug": slug,
                "version": release.get("version"),
                "content_sha256": entry["content_sha256"],
                "source_bundle_sha256": entry["source_bundle_sha256"],
                "source_path": entry["source_path"],
                "context": context,
            }
        except (KeyError, TypeError, ValueError, OSError, tarfile.TarError) as error:
            skipped.append({"slug": slug, "reason": str(error)[:600]})
            continue
        if key in images:
            images[key]["recipes"].append(recipe)
            continue
        directory = output / key / "context"
        for relative, payload in files.items():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            # The Spark build context is the canonical tar: every file 0644.
            target.chmod(0o644)
        images[key] = {
            "build_key": key,
            "recipes": [recipe],
            "base_images": [str(image["reference"]) for image in base_images],
            "recipe_build_arguments": podman_build_arguments(
                policy, dockerfile=dockerfile, context="context"
            ),
        }
    return {
        "schema_version": KEY_SCHEMA_VERSION,
        "source_commit": index.get("source_commit"),
        "images": sorted(images.values(), key=lambda value: value["build_key"]),
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    plan = write_library_image_plan(
        arguments.library_root.resolve(),
        arguments.output_dir.resolve(),
        json.loads(POLICY.read_text(encoding="utf-8")),
    )
    json.dump(plan, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0
