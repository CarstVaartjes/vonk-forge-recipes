"""Every literal adapter model root must be selected by its consuming recipes.

Scan Dockerfiles, Python, shell and extensionless executable sources in each build
context, including build patches. Upstream tests and documentation are not
runtime code. Shared-context branches and upstream launchers whose defaults are
replaced at runtime need a recipe/file/path exception with an explicit reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from vonk_forge_contracts import RecipeDefinition

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = re.compile(r"(?<![\w/}~])(/models/[A-Za-z0-9][A-Za-z0-9_./-]*)")
# Each exception is limited to a recipe, source file and literal path. Stale
# exceptions fail below, so broad or unexplained exclusions cannot hide drift.
EXCEPTION_GROUPS: tuple[tuple[tuple[str, ...], str, tuple[str, ...], str], ...] = (
    (
        ("glm-5-3-flash-nvfp4-vllm-four",),
        "upstream-launch-tp4-24g.sh",
        (
            "/models/glm-5.3-flash-nvfp4",
            "/models/dflash2-draft",
            "/models/glm-5.3-flash-nvfp4/chat_template_mm.jinja",
        ),
        "Retained upstream host launcher; the recipe invokes /opt/vonk/bin/vllm with target/drafter and the packaged template instead.",
    ),
    (
        ("glm-5-3-flash-nvfp4-vllm-four",),
        "upstream-launch-tp4-dflash2-topkfix.sh",
        ("/models/glm-5.3-flash-nvfp4", "/models/dflash2-draft"),
        "Retained upstream host launcher; the recipe invokes /opt/vonk/bin/vllm with target/drafter instead.",
    ),
    (
        ("inkling-small-nvfp4-kv-1m-drowzeys-sglang-dual",),
        "scripts/inkling-sglang-launch.sh",
        ("/models/dspark-draft", "/models/inkling-small-nvfp4"),
        "Upstream host launcher is not the recipe entrypoint; sglang-serve receives target/drafter arguments from the recipe.",
    ),
    (
        ("laguna-s-2-1-nvfp4-r0b0tlab-vllm-single",),
        "scripts/audit_runtime.py",
        ("/models/Laguna-S-2.1-NVFP4",),
        "Retained upstream audit defaults; the recipe invokes vllm directly with /models/target and does not run this audit.",
    ),
    (
        ("laguna-s-2-1-nvfp4-r0b0tlab-vllm-single",),
        "scripts/entrypoint.sh",
        ("/models/Laguna-S-2.1-NVFP4", "/models/Laguna-S-2.1-DFlash-NVFP4"),
        "Upstream launcher is replaced by /opt/vonk/bin/vllm and explicit target/drafter arguments.",
    ),
    (
        ("minimax-m3-w4a16-gptq-eagle3-tonyd2wild-vllm-dual",),
        "vendor/b12x/verify_moe_split.py",
        ("/models/GLM-5.1-NVFP4-MTP-NVFP4",),
        "Vendored GLM arithmetic diagnostic, not called by the MiniMax recipe's build or runtime.",
    ),
    (
        ("muse-glimmer-30b-nvfp4-vllm-mia-single",),
        "verify-speculative-config.py",
        ("/models/parser-only",),
        "Build-time CLI parser sentinel; the verification never loads model files.",
    ),
    (
        ("pixal3d-pytorch-single",),
        "trellis2_job.py",
        ("/models/sparse-decoder-config", "/models/sparse-decoder-weights"),
        "Shared image's TRELLIS.2 entrypoint is not executed by Pixal3D, whose primary snapshot already includes the decoder.",
    ),
    (
        ("trellis-2-4b-pytorch-single",),
        "pixal3d_job.py",
        ("/models/naf/naf_release.pth",),
        "Shared image's Pixal3D entrypoint is not executed by TRELLIS.2; only Pixal3D uses NAF.",
    ),
    (
        (
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
        ),
        "run.py",
        ("/models/sdxl", "/models/sdxl-vae"),
        "Shared runner's texture-only branch; these recipes call main(geometry) or main(label-geometry).",
    ),
    (
        ("step1x-3d-geometry-pytorch-single",),
        "prepare_upstream.py",
        ("/models/clip",),
        "Patch for the alternate CLIP encoder; this recipe selects the DINOv2-only checkpoint configuration.",
    ),
    (
        ("step1x-3d-texture-pytorch-single",),
        "prepare_upstream.py",
        ("/models/clip", "/models/dinov2-registers"),
        "Shared build patches geometry encoders; texture inference uses SDXL and its VAE instead.",
    ),
)
EXCEPTIONS = {
    (recipe, adapter, path): reason
    for recipes, adapter, paths, reason in EXCEPTION_GROUPS
    for recipe in recipes
    for path in paths
}


@dataclass(frozen=True)
class MissingMount:
    recipe: str
    adapter: str
    model_path: str
    line: int


def missing_mounts(root: Path) -> list[MissingMount]:
    missing: list[MissingMount] = []
    for path in sorted((root / "recipes").glob("*.json")):
        recipe = RecipeDefinition.model_validate_json(path.read_text())
        context = root / recipe.execution.build.context.path
        mounts = {
            file.mount.target.rstrip("/")
            for selection in recipe.models
            for file in selection.files
        }
        for source in sorted(context.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(context)
            if "tests" in relative.parts or source.name.startswith("test_"):
                continue
            if source.suffix not in {".py", ".sh", ""}:
                continue
            text = source.read_text(encoding="utf-8")
            if (
                not source.suffix
                and source.name != "Dockerfile"
                and not text.startswith("#!")
            ):
                continue
            seen: set[str] = set()
            for line, value in enumerate(text.splitlines(), 1):
                for match in MODEL_PATH.finditer(value):
                    model_path = match[1].rstrip("/.")
                    if (
                        any(
                            model_path == mount or model_path.startswith(mount + "/")
                            for mount in mounts
                        )
                        or model_path in seen
                    ):
                        continue
                    seen.add(model_path)
                    missing.append(
                        MissingMount(path.stem, str(relative), model_path, line)
                    )
    return missing


def test_catalog_adapter_model_mounts() -> None:
    missing = missing_mounts(ROOT)
    keys = {(item.recipe, item.adapter, item.model_path) for item in missing}
    assert all(reason.strip() for reason in EXCEPTIONS.values())
    assert not EXCEPTIONS.keys() - keys, "Remove stale model mount exceptions"
    problems = [
        f"{item.recipe}: {item.adapter}:{item.line}: undeclared {item.model_path}"
        for item in missing
        if (item.recipe, item.adapter, item.model_path) not in EXCEPTIONS
    ]
    assert not problems, "\n".join(problems)


def test_scan_covers_python_shell_patches_and_exact_path_boundaries(
    tmp_path: Path,
) -> None:
    import json

    recipe = json.loads((ROOT / "recipes/trellis-2-4b-pytorch-single.json").read_text())
    recipe["models"] = [recipe["models"][0]]
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "fixture.json").write_text(json.dumps(recipe))
    context = tmp_path / recipe["execution"]["build"]["context"]["path"]
    context.mkdir(parents=True)
    (context / "job.py").write_text(
        'load("/models/dino")\nload("/models/target-extra/config.json")\n'
        'load("/models/target/config.json")\n'
        'source = "/usr/lib/models/not-a-mount"\n'
    )
    (context / "launch.sh").write_text("load /models/naf/naf_release.pth\n")
    (context / "Dockerfile").write_text(
        "FROM scratch\nRUN check /models/build-companion\n"
    )
    (context / "patch.py").write_text('replacement = "/models/decoder"\n')
    (context / "launcher").write_text("#!/bin/sh\nload /models/encoder\n")
    assert {item.model_path for item in missing_mounts(tmp_path)} == {
        "/models/dino",
        "/models/target-extra/config.json",
        "/models/naf/naf_release.pth",
        "/models/decoder",
        "/models/encoder",
        "/models/build-companion",
    }
    selection = recipe["models"][0]
    selection["files"][0]["mount"]["target"] = "/models"
    (recipes / "fixture.json").write_text(json.dumps(recipe))
    assert missing_mounts(tmp_path) == []


def test_removing_each_declared_companion_breaks_the_guard(tmp_path: Path) -> None:
    """Fault injection: each repaired mount remains mandatory."""
    import shutil

    cases = (
        (
            "trellis-2-4b",
            "sparse-decoder",
            {"/models/sparse-decoder-config", "/models/sparse-decoder-weights"},
        ),
        ("trellis-2-4b", "dino", {"/models/dino"}),
        ("pixal3d", "dino", {"/models/dino"}),
        ("pixal3d", "naf", {"/models/naf/naf_release.pth"}),
    )
    for name, selection_id, expected in cases:
        slug = f"{name}-pytorch-single"
        root = tmp_path / f"{slug}-{selection_id}"
        recipe = RecipeDefinition.model_validate_json(
            (ROOT / "recipes" / f"{slug}.json").read_text()
        )
        assert selection_id in {selection.id for selection in recipe.models}
        recipe = recipe.model_copy(
            update={
                "models": [
                    selection
                    for selection in recipe.models
                    if selection.id != selection_id
                ]
            }
        )
        (root / "recipes").mkdir(parents=True)
        (root / "recipes" / f"{slug}.json").write_text(recipe.model_dump_json())
        context = recipe.execution.build.context.path
        shutil.copytree(ROOT / context, root / context)
        failures = {
            item.model_path
            for item in missing_mounts(root)
            if (item.recipe, item.adapter, item.model_path) not in EXCEPTIONS
        }
        assert expected <= failures, (slug, selection_id, failures)
