"""A recipe that runs its source's published image follows the image, not the commits.

#1297 moved eugr's nemotron recipes to a newer eugr/spark-vllm-docker commit while the
Dockerfile kept building FROM the same eugr/spark-vllm digest: a new build and a claim of
a revision the image does not contain. The refresher now moves such a recipe only when
the published tag names another digest, to the build commit that image records; a pin
move that changes no file the recipe uses is never a refresh on its own; and
``tools/check-image-provenance`` holds a changed recipe to its image's recorded commit.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, module: str) -> Any:
    loader = importlib.machinery.SourceFileLoader(module, str(ROOT / "tools" / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = loaded
    loader.exec_module(loaded)
    return loaded


kit_pins = load("kit-pins", "kit_pins_for_image_follow")
refresh = load("refresh-upstream", "refresh_upstream_for_image_follow")
guard = load("check-image-provenance", "check_image_provenance_under_test")

SOURCE = "eugr/spark-vllm-docker"
IMAGE = "docker.io/eugr/spark-vllm"
OLD = "a" * 40
BUILT = "b" * 40
PINNED = "sha256:" + "1" * 64


def sha(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def metadata_layer(text: str) -> bytes:
    data = text.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("workspace/build-metadata.yaml")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    return gzip.compress(buffer.getvalue())


class FakeRegistry:
    """Anonymous registry: ``latest`` and every manifest and blob by digest."""

    def __init__(self, labels: dict[str, str], layer: bytes | None) -> None:
        config = json.dumps({"config": {"Labels": labels}}).encode()
        layers = [] if layer is None else [{"digest": sha(layer), "size": len(layer)}]
        manifest = json.dumps(
            {"schemaVersion": 2, "config": {"digest": sha(config)}, "layers": layers}
        ).encode()
        self.digest = sha(manifest)
        self.objects = {
            "latest": manifest,
            self.digest: manifest,
            sha(config): config,
        }
        if layer is not None:
            self.objects[sha(layer)] = layer

    def __call__(self, url: str, _headers: dict[str, str]):
        body = self.objects.get(url.rsplit("/", 1)[1])
        if body is None:
            return 404, {}, b""
        return 200, {"docker-content-digest": sha(body)}, body


def eugr_registry(commit: str = BUILT) -> FakeRegistry:
    return FakeRegistry(
        {"org.opencontainers.image.version": "24.04"},
        metadata_layer(
            f"build_date: 2026-10-13T00:40:45Z\nbuild_script_commit: {commit}\n"
        ),
    )


class ProvenanceTests(unittest.TestCase):
    def test_build_metadata_of_the_last_layer_names_the_commit(self) -> None:
        fake = eugr_registry()
        found = kit_pins.Registry(fake).provenance(IMAGE, fake.digest, SOURCE)
        self.assertEqual(found, (BUILT, "2026-10-13"))

    def test_an_oci_revision_label_of_the_source_names_the_commit(self) -> None:
        fake = FakeRegistry(
            {
                "org.opencontainers.image.revision": BUILT,
                "org.opencontainers.image.source": f"https://github.com/{SOURCE}.git",
                "org.opencontainers.image.created": "2026-10-01T00:00:00Z",
            },
            None,
        )
        found = kit_pins.Registry(fake).provenance(IMAGE, fake.digest, SOURCE)
        self.assertEqual(found, (BUILT, "2026-10-01"))

    def test_another_repository_or_no_record_is_never_guessed(self) -> None:
        other = FakeRegistry(
            {
                "org.opencontainers.image.revision": BUILT,
                "org.opencontainers.image.source": "https://github.com/vllm-project/vllm",
            },
            None,
        )
        registry = kit_pins.Registry(other)
        self.assertIsNone(registry.provenance(IMAGE, other.digest, SOURCE))
        bare = FakeRegistry({}, metadata_layer("build_date: 2026-10-13\n"))
        self.assertIsNone(
            kit_pins.Registry(bare).provenance(IMAGE, bare.digest, SOURCE)
        )


def write_recipe(root: Path, source: str, digest: str = PINNED) -> dict[str, Any]:
    (root / "published-images.json").write_text(
        json.dumps({"images": {IMAGE: {"source": SOURCE, "tag": "latest"}}})
    )
    adapter = root / "adapters/eugr/x"
    adapter.mkdir(parents=True)
    (adapter / "Dockerfile").write_text(
        f"FROM {IMAGE}@{digest}\n\n"
        f'LABEL org.opencontainers.image.source="https://github.com/{source}" \\\n'
        f'      org.opencontainers.image.revision="{OLD}"\n'
    )
    recipe = {
        "identity": {"slug": "recipe"},
        "provenance": {
            "attribution": ["eugr"],
            "source_reference": f"https://github.com/{source}/tree/{OLD}",
        },
        "release": {"version": "prebuilt-vllm-current", "released_at": "2026-10-01"},
        "models": [],
        "execution": {
            "build": {
                "base_image": {
                    "repository": IMAGE,
                    "digest": digest.removeprefix("sha256:"),
                },
                "context": {"path": "adapters/eugr/x"},
            }
        },
    }
    (root / "recipes").mkdir()
    (root / "recipes/recipe.json").write_text(json.dumps(recipe))
    return recipe


class RefreshTests(unittest.TestCase):
    def assess(self, fake: FakeRegistry, compare: Any) -> Any:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, SOURCE)
            assessor = refresh.Assessor(
                refresh.Catalog.load(root), object(), None, {"recipes": []}
            )
            source = kit_pins.Upstream(None, None, kit_pins.Registry(fake))  # type: ignore[arg-type]
            empty = refresh.vendored.Tree({}, True)
            with (
                patch.object(refresh, "kit_upstream", lambda _http: source),
                patch.object(refresh, "compare_commits", compare),
                patch.object(refresh.vendored, "fetch_tree", lambda *_: empty),
            ):
                result = assessor.assess("recipe")
            result.edits = {
                k: v for k, v in result.edits.items() if v != (root / k).read_bytes()
            }
            return result

    @staticmethod
    def identical(*_: Any) -> Any:
        return refresh.Comparison("ahead", [], [], False, "u")

    def test_an_unchanged_image_proposes_nothing_however_far_the_source_moved(
        self,
    ) -> None:
        fake = eugr_registry()
        fake.objects["latest"] = b"{}"
        with tempfile.TemporaryDirectory() as tmp:
            write_recipe(Path(tmp), SOURCE, sha(b"{}"))
            catalog = refresh.Catalog.load(Path(tmp))
            source = kit_pins.Upstream(None, None, kit_pins.Registry(fake))  # type: ignore[arg-type]
            newest = refresh.Target("github", SOURCE, "c" * 40, "d", "tag", "1", "u")
            result = refresh.collect_items(
                catalog, "recipe", object(), lambda *_: newest, source
            )
        self.assertFalse(result.drifted)
        self.assertIn("unchanged", result.evidence[0])

    def test_a_new_image_moves_from_labels_and_reference_to_its_build_commit(
        self,
    ) -> None:
        fake = eugr_registry()
        result = self.assess(fake, self.identical)
        self.assertTrue(result.mechanical, result.reasons)
        dockerfile = result.edits["adapters/eugr/x/Dockerfile"].decode()
        self.assertIn(f"FROM {IMAGE}@{fake.digest}", dockerfile)
        self.assertIn(f'revision="{BUILT}"', dockerfile)
        recipe = json.loads(result.edits["recipes/recipe.json"])
        self.assertTrue(recipe["provenance"]["source_reference"].endswith(BUILT))
        self.assertEqual(
            recipe["execution"]["build"]["base_image"]["digest"],
            fake.digest.removeprefix("sha256:"),
        )
        self.assertEqual(recipe["release"]["version"], "prebuilt-vllm-current")

    def test_an_image_built_from_an_unpublished_commit_waits(self) -> None:
        def missing(*_: Any) -> Any:
            raise RuntimeError("HTTP 422: No commit found")

        result = self.assess(eugr_registry(), missing)
        self.assertFalse(result.mechanical)
        self.assertFalse(result.reasons)
        self.assertTrue(result.unreachable)
        self.assertEqual(result.edits, {})

    def test_an_image_without_a_build_record_is_a_review(self) -> None:
        result = self.assess(FakeRegistry({}, None), self.identical)
        self.assertFalse(result.mechanical)
        self.assertIn("records no build commit", result.reasons[0])

    def test_a_commit_move_that_changes_no_used_file_is_not_a_refresh(self) -> None:
        # Not a followed image (another source): the classic source pin, whose newest
        # commit changes nothing the recipe uses.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, "someone/recipe")
            assessor = refresh.Assessor(
                refresh.Catalog.load(root), object(), None, {"recipes": []}
            )
            newest = refresh.Target(
                "github", "someone/recipe", BUILT, "d", None, None, "u"
            )
            assessor.target = lambda *_: newest
            empty = refresh.vendored.Tree({}, True)
            with (
                patch.object(refresh, "compare_commits", self.identical),
                patch.object(refresh.vendored, "fetch_tree", lambda *_: empty),
            ):
                result = assessor.assess("recipe")
        self.assertFalse(result.drifted)
        self.assertEqual(result.edits, {})


class GuardTests(unittest.TestCase):
    def check(self, changed: set[str], fake: Any) -> tuple[list[str], list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, SOURCE, fake.digest)
            return guard.check(root, changed, kit_pins.Registry(fake))

    def test_a_changed_recipe_must_claim_its_images_build_commit(self) -> None:
        problems, _ = self.check({"adapters/eugr/x/Dockerfile"}, eugr_registry())
        self.assertEqual(len(problems), 1)
        self.assertIn(BUILT, problems[0])
        self.assertIn("revision label", problems[0])
        self.assertIn("source_reference", problems[0])
        self.assertEqual(
            self.check({"recipes/recipe.json"}, eugr_registry(OLD)), ([], [])
        )

    def test_unchanged_recipes_are_not_checked_and_unreadable_images_are_noted(
        self,
    ) -> None:
        self.assertEqual(self.check({"README.md"}, eugr_registry()), ([], []))
        problems, notes = self.check({"recipes/recipe.json"}, FakeRegistry({}, None))
        self.assertEqual(problems, [])
        self.assertIn("cannot be checked", notes[0])


if __name__ == "__main__":
    unittest.main()
