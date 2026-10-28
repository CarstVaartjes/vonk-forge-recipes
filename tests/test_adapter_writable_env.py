"""Adapter images run read-only: every writable-state ENV and the cwd must be writable."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path, PurePosixPath

from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
# Containers run with a read-only root, a tmpfs /tmp and the /outputs bind (its
# cache/ included) for every role, services as well as jobs.
_WRITABLE = (PurePosixPath("/tmp"), PurePosixPath("/outputs"))
_STATE_ENV = re.compile(r"(_PATH|_DIR|_ROOT)$|CACHE|^HOME$|^XDG_", re.IGNORECASE)
# Names that point at image or model files that are only read, never written.
_READ_ONLY_ENV = frozenset(
    {
        "MODEL_PATH",
        "LD_LIBRARY_PATH",
        "VLLM_NCCL_SO_PATH",
        "TRITON_PTXAS_PATH",
        "VLLM_BASE_DIR",
        "TIKTOKEN_CACHE_DIR",
        "TIKTOKEN_RS_CACHE_DIR",
    }
)
# WORKDIR under the image's own code roots is a read-only cwd: nothing writes it.
_CODE_ROOTS = (
    PurePosixPath("/opt"),
    PurePosixPath("/workspace"),
    PurePosixPath("/app"),
)
_PROVEN = "proven on hardware; path not written"
_UNPROVEN_ALLOWLIST = "unproven_writable_paths.json"


def _writable(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        or "$" in value
        or any(path.is_relative_to(root) for root in _WRITABLE)
    )


def _violations(dockerfile: Path) -> list[str]:
    """Writable-state ENV values and the final-stage WORKDIR outside writable roots."""
    lines = [
        line.strip()
        for line in dockerfile.read_text(encoding="utf-8")
        .replace("\\\n", " ")
        .splitlines()
    ]
    stage = max(
        (i for i, line in enumerate(lines) if line.upper().startswith("FROM ")),
        default=0,
    )
    found: list[str] = []
    for line in lines[stage:]:
        keyword, _, rest = line.partition(" ")
        if keyword.upper() == "WORKDIR":
            value = rest.strip()
            path = PurePosixPath(value)
            if not _writable(value) and not any(
                path.is_relative_to(root) for root in _CODE_ROOTS
            ):
                found.append(f"WORKDIR {value}")
        elif keyword.upper() == "ENV":
            tokens = shlex.split(rest, comments=True)
            if tokens and "=" not in tokens[0]:  # legacy `ENV NAME value` form
                tokens = [f"{tokens[0]}={' '.join(tokens[1:])}"]
            for token in tokens:
                name, _, value = token.partition("=")
                if (
                    _STATE_ENV.search(name)
                    and name not in _READ_ONLY_ENV
                    and not _writable(value)
                ):
                    found.append(f"{name}={value}")
    return found


def _recipes_by_dockerfile() -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for recipe_path in (ROOT / "recipes").glob("*.json"):
        recipe = RecipeDefinition.model_validate_json(recipe_path.read_text())
        build = recipe.execution.build
        context = ROOT / build.context.path
        for dockerfile in [ROOT / build.dockerfile, *context.rglob("Dockerfile*")]:
            owners.setdefault(str(dockerfile.relative_to(ROOT)), set()).add(
                recipe_path.stem
            )
    return owners


def _is_proven(recipe: str) -> bool:
    evidence = ROOT / "qualification/evidence/vonk-forge" / f"{recipe}.json"
    if not evidence.exists():
        return False
    records = json.loads(evidence.read_text())["records"]
    return any(record["outcome"] == "passed" for record in records)


def test_adapter_images_keep_state_and_cwd_on_writable_paths() -> None:
    owners = _recipes_by_dockerfile()
    unproven = set(
        json.loads((Path(__file__).parent / _UNPROVEN_ALLOWLIST).read_text())
    )
    flagged = {
        str(path.relative_to(ROOT)): bad
        for path in sorted((ROOT / "adapters").rglob("Dockerfile*"))
        if (bad := _violations(path))
    }
    # A proven recipe is never changed pre-emptively (reason: _PROVEN); only
    # images with an unproven recipe must be fixed or listed as unproven.
    offending = {
        path: bad
        for path, bad in flagged.items()
        if path not in unproven
        and not (owners.get(path) and all(_is_proven(r) for r in owners[path]))
    }
    assert not offending, offending
    # The allowlist shrinks by itself: fixed or proven entries must be dropped.
    stale = sorted(
        path
        for path in unproven
        if path not in flagged
        or (owners.get(path) and all(_is_proven(r) for r in owners[path]))
    )
    assert not stale, stale


def test_guard_flags_state_env_and_cwd(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM base AS build\nENV BUILD_DIR=/src\n"
        "FROM base\nENV STATE_PATH=/state HOME=/state \\\n    XDG_CACHE_HOME=/outputs/cache\n"
        "ENV TORCHINDUCTOR_CACHE_DIR /state/torch\nENV MODEL_PATH=/models/target\n"
        "WORKDIR /state\n"
    )
    assert _violations(dockerfile) == [
        "STATE_PATH=/state",
        "HOME=/state",
        "TORCHINDUCTOR_CACHE_DIR=/state/torch",
        "WORKDIR /state",
    ]


def test_guard_accepts_writable_paths_and_code_cwd(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM base\nENV STATE_PATH=/outputs/cache/state HOME=/outputs/cache/home "
        "TRITON_CACHE_DIR=/tmp/triton\nWORKDIR /outputs/cache\n"
    )
    assert not _violations(dockerfile)
    dockerfile.write_text("FROM base\nWORKDIR /opt/vonk\n")
    assert not _violations(dockerfile)
