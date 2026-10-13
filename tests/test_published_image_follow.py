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
import os
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
        self.calls = 0
        self.objects = {
            "latest": manifest,
            self.digest: manifest,
            sha(config): config,
        }
        if layer is not None:
            self.objects[sha(layer)] = layer

    def __call__(self, url: str, _headers: dict[str, str]):
        self.calls += 1
        body = self.objects.get(url.rsplit("/", 1)[1])
        if body is None:
            return 404, {}, b""
        return 200, {"docker-content-digest": sha(body)}, body


def eugr_registry(commit: str = BUILT) -> FakeRegistry:
    return FakeRegistry(
        {"org.opencontainers.image.version": "24.04"},
        metadata_layer(
            f"build_date: 2026-10-13T00:40:45Z\nbuild_script_commit: {commit}\n"
            "vllm_version: 0.31.1rc1.dev380+g6202d0fb1.d20261013\n"
            "vllm_commit: 6202d0fb148b1f639c0d3e809d0012ed42af54bf\n"
        ),
    )


class ProvenanceTests(unittest.TestCase):
    def test_build_metadata_of_the_last_layer_names_the_commit(self) -> None:
        fake = eugr_registry()
        registry = kit_pins.Registry(fake)
        found = registry.provenance(IMAGE, fake.digest, SOURCE)
        self.assertEqual(found, (BUILT, "2026-10-13"))
        # Recipes sharing an image read it once a run (anonymous pulls are limited).
        calls = fake.calls
        registry.provenance(IMAGE, fake.digest, SOURCE)
        registry.resolve(IMAGE, fake.digest)
        self.assertEqual(fake.calls, calls)

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


def write_recipe(
    root: Path,
    source: str,
    digest: str = PINNED,
    pinned: str = OLD,
    mark: str | None = None,
) -> dict[str, Any]:
    (root / "published-images.json").write_text(
        json.dumps({"images": {IMAGE: {"source": SOURCE, "tag": "latest"}}})
    )
    adapter = root / "adapters/eugr/x"
    adapter.mkdir(parents=True)
    (adapter / "Dockerfile").write_text(
        f"FROM {IMAGE}@{digest}\n\n"
        f'LABEL org.opencontainers.image.source="https://github.com/{source}" \\\n'
        f'      org.opencontainers.image.revision="{pinned}"\n'
    )
    manifest_source = {"commit": pinned, "repo": source}
    if mark is not None:
        manifest_source["unverified"] = mark
    (adapter / "vendored-upstream.json").write_text(
        json.dumps({"ours": [], "source": manifest_source}, indent=2) + "\n"
    )
    recipe = {
        "identity": {"slug": "recipe"},
        "provenance": {
            "attribution": ["eugr"],
            "source_reference": f"https://github.com/{source}/tree/{pinned}",
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

    @staticmethod
    def missing(*_: Any) -> Any:
        raise RuntimeError("HTTP 422: No commit found")

    def test_github_unreadable_for_an_image_commit_waits(self) -> None:
        with patch.object(refresh, "commit_on_github", lambda *_: None):
            result = self.assess(eugr_registry(), self.missing)
        self.assertFalse(result.mechanical)
        self.assertFalse(result.reasons)
        self.assertTrue(result.unreachable)
        self.assertEqual(result.edits, {})

    def test_an_image_built_from_an_unpublished_commit_moves_for_review(self) -> None:
        fake = eugr_registry()
        with patch.object(refresh, "commit_on_github", lambda *_: False):
            result = self.assess(fake, self.missing)
        # Followed by digest, provenance marked, held for a person (never armed).
        self.assertTrue(result.for_review, result.reasons)
        self.assertFalse(result.mechanical)
        dockerfile = result.edits["adapters/eugr/x/Dockerfile"].decode()
        self.assertIn(f"FROM {IMAGE}@{fake.digest}", dockerfile)
        self.assertIn(f'revision="{BUILT}"', dockerfile)
        recipe = json.loads(result.edits["recipes/recipe.json"])
        self.assertTrue(recipe["provenance"]["source_reference"].endswith(BUILT))
        manifest = json.loads(result.edits["adapters/eugr/x/vendored-upstream.json"])
        self.assertEqual(manifest["source"]["commit"], BUILT)
        self.assertEqual(
            manifest["source"]["unverified"],
            f"commit {BUILT} not on github.com/{SOURCE}",
        )
        result.pull_request = 7
        body = refresh.issue_body(result)
        self.assertIn("#7", body)
        self.assertIn("not on GitHub", body)
        self.assertIn("vllm_version: 0.31.1rc1", body)  # what the image itself shows
        self.assertIn("Needs review", refresh.pr_body(result))

    def test_a_move_off_an_unpublished_pin_is_held_and_drops_the_mark(self) -> None:
        fake = eugr_registry("c" * 40)  # the new image's commit is published
        mark = f"commit {BUILT} not on github.com/{SOURCE}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, SOURCE, pinned=BUILT, mark=mark)
            assessor = refresh.Assessor(
                refresh.Catalog.load(root), object(), None, {"recipes": []}
            )
            source = kit_pins.Upstream(None, None, kit_pins.Registry(fake))  # type: ignore[arg-type]
            with (
                patch.object(refresh, "kit_upstream", lambda _http: source),
                patch.object(refresh, "compare_commits", self.missing),
                patch.object(refresh, "commit_on_github", lambda *_: True),
            ):
                result = assessor.assess("recipe")
            result.edits = {
                k: v for k, v in result.edits.items() if v != (root / k).read_bytes()
            }
        self.assertTrue(result.for_review, result.reasons)
        self.assertIn("never published", result.held[0])
        dockerfile = result.edits["adapters/eugr/x/Dockerfile"].decode()
        self.assertIn(f"FROM {IMAGE}@{fake.digest}", dockerfile)
        manifest = json.loads(result.edits["adapters/eugr/x/vendored-upstream.json"])
        self.assertEqual(manifest["source"], {"commit": "c" * 40, "repo": SOURCE})

    def test_the_mark_goes_once_github_serves_the_commit(self) -> None:
        fake = eugr_registry()
        mark = f"commit {BUILT} not on github.com/{SOURCE}"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, SOURCE, fake.digest, pinned=BUILT, mark=mark)
            assessor = refresh.Assessor(
                refresh.Catalog.load(root), object(), None, {"recipes": []}
            )
            source = kit_pins.Upstream(None, None, kit_pins.Registry(fake))  # type: ignore[arg-type]
            for served in (False, True):
                with (
                    patch.object(refresh, "kit_upstream", lambda _http: source),
                    patch.object(refresh, "commit_on_github", lambda *_, s=served: s),
                ):
                    result = assessor.assess("recipe")
                if not served:
                    self.assertFalse(result.drifted)  # still unpublished: nothing to do
                    continue
                self.assertTrue(result.mechanical, result.reasons)
                self.assertEqual(
                    sorted(result.edits), ["adapters/eugr/x/vendored-upstream.json"]
                )
                manifest = json.loads(
                    result.edits["adapters/eugr/x/vendored-upstream.json"]
                )
                self.assertEqual(manifest["source"], {"commit": BUILT, "repo": SOURCE})

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


class SharedContextTests(unittest.TestCase):
    def test_single_and_dual_held_for_review_are_prepared_together(self) -> None:
        target = refresh.Target("github", SOURCE, BUILT, "2026-10-13", None, None, "u")
        parts = {}
        for name in ("x-dual", "x-single"):
            item = refresh.Item("source", "github", SOURCE, OLD, target, unverified="m")
            parts[name] = refresh.Assessment(
                name, items=[item], held=["not on GitHub"], edits={"a": b"1"}
            )
        combined = refresh.combine_group(["x-dual", "x-single"], parts)
        assert not isinstance(combined, str), combined
        self.assertTrue(combined.for_review)
        self.assertFalse(combined.mechanical)  # never armed
        self.assertEqual(combined.held, ["not on GitHub"])


class GuardTests(unittest.TestCase):
    def check(
        self,
        changed: set[str],
        fake: Any,
        served: bool | None = True,
        pinned: str = OLD,
        mark: str | None = None,
    ) -> tuple[list[str], list[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recipe(root, SOURCE, fake.digest, pinned=pinned, mark=mark)
            return guard.check(
                root, changed, kit_pins.Registry(fake), lambda *_: served
            )

    def test_an_unpublished_build_commit_is_accepted_only_when_marked(self) -> None:
        mark = f"commit {BUILT} not on github.com/{SOURCE}"
        changed = {"recipes/recipe.json"}
        problems, _ = self.check(changed, eugr_registry(), False, BUILT)
        self.assertIn("source.unverified", problems[0])
        self.assertEqual(
            self.check(changed, eugr_registry(), False, BUILT, mark), ([], [])
        )
        problems, _ = self.check(changed, eugr_registry(), True, BUILT, mark)
        self.assertIn("remove source.unverified", problems[0])  # stale mark
        problems, _ = self.check(changed, eugr_registry(), False, BUILT, "unverified")
        self.assertIn("must read", problems[0])
        problems, notes = self.check(changed, eugr_registry(), None, BUILT)
        self.assertEqual(problems, [])
        self.assertIn("could not tell", notes[0])

    def test_the_vendored_check_skips_only_a_correctly_marked_pin(self) -> None:
        vendored = guard.vendored
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mark = f"commit {BUILT} not on github.com/{SOURCE}"
            write_recipe(root, SOURCE, pinned=BUILT, mark=mark)

            def unreadable(*_: Any) -> Any:
                raise RuntimeError("No commit found")

            adapter = root / "adapters/eugr/x"
            report = vendored.check_adapter(adapter, root, unreadable)
            self.assertEqual(report.problems, [])
            self.assertIn("unverified", report.note)
            manifest = adapter / "vendored-upstream.json"
            document = json.loads(manifest.read_text())
            document["source"]["unverified"] = "trust me"
            manifest.write_text(json.dumps(document))
            report = vendored.check_adapter(adapter, root, unreadable)
            self.assertIn("must read", report.problems[0])

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


PLAYBOOKS = "NVIDIA/dgx-spark-playbooks"


class PlaybookMoveTests(unittest.TestCase):
    """#1326-#1328: a playbook commit moved; the recipe uses none of its files."""

    def test_a_playbook_commit_move_with_a_semver_release_is_not_a_refresh(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapters/nvidia/x"
            adapter.mkdir(parents=True)
            (adapter / "Dockerfile").write_text(
                f"FROM docker.io/lmsysorg/sglang@{PINNED}\n"
                'LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang"\n'
            )
            (root / "recipes").mkdir()
            reference = (
                f"https://github.com/{PLAYBOOKS}/blob/{OLD}/nvidia/llama-cpp/README.md"
            )
            (root / "recipes/recipe.json").write_text(
                json.dumps(
                    {
                        "identity": {"slug": "recipe"},
                        "provenance": {"source_reference": reference},
                        "release": {"version": "0.6.2", "released_at": "2026-10-08"},
                        "models": [],
                        "execution": {
                            "build": {"context": {"path": "adapters/nvidia/x"}}
                        },
                    }
                )
            )
            assessor = refresh.Assessor(
                refresh.Catalog.load(root), object(), None, {"recipes": []}
            )
            newest = refresh.Target(
                "github", PLAYBOOKS, BUILT, "2026-10-12", None, None, "u"
            )
            assessor.target = lambda *_: newest
            readme = refresh.Comparison(
                "ahead",
                [],
                [{"filename": "nvidia/playbook-llms/README.md", "status": "modified"}],
                False,
                "u",
            )

            def offline(*_: Any) -> Any:
                raise RuntimeError("no upstream tree is needed for a recipe using none")

            with (
                patch.object(refresh, "compare_commits", lambda *_: readme),
                patch.object(refresh.vendored, "fetch_tree", offline),
            ):
                result = assessor.assess("recipe")
        self.assertFalse(result.drifted, result.reasons)
        self.assertEqual(result.edits, {})  # no source_reference, date or 0.6.3 bump


class StaleRunTests(unittest.TestCase):
    """#1325/#1326: a run started before the rule changed kept preparing for an hour."""

    def test_a_refresh_whose_files_or_rules_changed_on_main_is_deferred(self) -> None:
        env = {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.test",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.test",
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env):
            root = Path(tmp)
            remote = root / "remote"
            work = root / "work"
            refresh.git("init", "-q", "--initial-branch=main", str(remote), cwd=root)
            (remote / "recipe.json").write_text("old pin\n")
            (remote / "tools").mkdir()
            (remote / "tools/refresh-upstream").write_text("rules v1\n")
            refresh.git("add", "-A", cwd=remote)
            refresh.git("commit", "-qm", "Base", cwd=remote)
            refresh.git("clone", "-q", str(remote), str(work), cwd=root)
            started = refresh.git("rev-parse", "HEAD", cwd=work)
            assessment = refresh.Assessment(
                "recipe", edits={"recipe.json": b"new pin\n"}
            )
            calls: list[tuple[str, ...]] = []

            def github(*arguments: str, check: bool = True) -> str:
                calls.append(arguments)
                return "[]"

            for changed in ("tools/refresh-upstream", "recipe.json"):
                (remote / changed).write_text(f"changed on main: {changed}\n")
                refresh.git("commit", "-qam", f"Change {changed}", cwd=remote)
                summary = refresh.Summary(started=started)
                with patch.object(refresh, "gh", github):
                    self.assertIsNone(
                        refresh.publish_mechanical(work, assessment, summary)
                    )
                self.assertIn(changed, summary.deferred[0])
                self.assertEqual(summary.prepared, 0)
                refresh.git("reset", "-q", "--hard", f"{started}", cwd=remote)
            self.assertFalse([c for c in calls if c[:2] == ("pr", "create")])
