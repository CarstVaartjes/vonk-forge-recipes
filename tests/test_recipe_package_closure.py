"""Format checks for the generated recipe-package archives.

These pin what this repository produces: exactly one ``recipe.json`` entrypoint,
the full model/source/fixture closure, and the bounded recipe-owned tensors the
export path allows through.

Hostile-input cases deliberately do not live here. ``validate_recipe_archive`` is
called by ``recipe_package`` on the archive it assembled a line earlier, so it
never meets an adversary and every rejection branch it holds is unreachable from
its only production call site. The copy that does parse untrusted bytes is the
control plane's ``RecipePackageClient._decode_package``, and its adversarial
archive cases live with it in that repository's
``control/tests/test_recipe_package_archive_safety.py``.
"""

from __future__ import annotations

import io
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))


def _job_row(built_catalog: tuple[dict, Path]) -> tuple[dict, bytes]:
    catalog, package_dir = built_catalog
    row = next(
        row
        for row in catalog["recipes"]
        if row["document"]["interfaces"][0]["adapter"] != "openai"
    )
    return row, (package_dir / Path(row["package"]["path"]).name).read_bytes()


def _row_for_slug(built_catalog: tuple[dict, Path], slug: str) -> tuple[dict, bytes]:
    catalog, package_dir = built_catalog
    row = next(
        row for row in catalog["recipes"] if row["document"]["identity"]["slug"] == slug
    )
    return row, (package_dir / Path(row["package"]["path"]).name).read_bytes()


def test_archive_has_one_entrypoint_and_real_closure(
    built_catalog: tuple[dict, Path],
) -> None:
    row, payload = _job_row(built_catalog)
    TOOL["validate_recipe_archive"](payload, row["document"])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = [
            (member.name, archive.extractfile(member).read())
            for member in archive.getmembers()
            if member.isfile()
        ]
    assert [name for name, _ in names].count("recipe.json") == 1


def test_archive_allows_bounded_recipe_owned_direction_tensor(
    built_catalog: tuple[dict, Path],
) -> None:
    row, payload = _row_for_slug(
        built_catalog,
        "glm-5-3-flash-exl3-dflash2-vllm-dual",
    )
    TOOL["validate_recipe_archive"](payload, row["document"])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("refusal_direction_glm53_bf_oproj.pt") for name in names)
    assert any(
        name.endswith("refusal_direction_glm53_dealign_late.pt") for name in names
    )
