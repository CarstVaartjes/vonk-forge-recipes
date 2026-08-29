from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/check-upstream-drift"
LOADER = importlib.machinery.SourceFileLoader("check_upstream_drift", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
drift = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = drift
LOADER.exec_module(drift)

PINNED = "a" * 40
OBSERVED = "b" * 40


class FakeClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str]] = []

    def json(self, url: str, provider: str) -> object:
        self.requests.append((url, provider))
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


class SourceParsingTests(unittest.TestCase):
    def test_normalizes_supported_github_forms(self) -> None:
        for source in (
            "https://github.com/example/project",
            "https://github.com/example/project.git",
            f"https://github.com/example/project/tree/{PINNED}",
            f"https://github.com/example/project/commit/{PINNED}",
            f"https://github.com/example/project/blob/{PINNED}/README.md",
            f"https://github.com/example/project@{PINNED}",
            f"https://github.com/example/project.git@{PINNED}",
        ):
            with self.subTest(source=source):
                location = drift.parse_source(source)
                self.assertEqual(location.provider, "github")
                self.assertEqual(location.repository, "example/project")
        self.assertEqual(
            drift.parse_source(
                f"https://github.com/example/project/tree/{PINNED}"
            ).embedded_revision,
            PINNED,
        )
        self.assertEqual(
            drift.parse_source(
                f"https://github.com/example/project/blob/{PINNED}/README.md"
            ).embedded_revision,
            PINNED,
        )

    def test_normalizes_supported_hugging_face_forms(self) -> None:
        for source in (
            "example/project",
            "https://huggingface.co/example/project",
            f"https://huggingface.co/example/project/tree/{PINNED}",
            f"https://huggingface.co/example/project/resolve/{PINNED}/weights/model.safetensors",
        ):
            with self.subTest(source=source):
                location = drift.parse_source(source)
                self.assertEqual(location.provider, "huggingface")
                self.assertEqual(location.repository, "example/project")

    def test_rejects_credentials_and_non_https_sources(self) -> None:
        for source in (
            "http://github.com/example/project",
            "https://user@example.com/example/project",
            "https://github.com/example/project/blob/main/README.md",
            "https://huggingface.co/example/project/blob/main/README.md",
        ):
            with self.assertRaises(drift.DriftInputError):
                drift.parse_source(source)


class DiscoveryTests(unittest.TestCase):
    def test_catalog_sources_are_all_discoverable(self) -> None:
        defaults, overrides = drift.load_manifest(ROOT / "upstream-watch.json")

        watches = drift.discover_watches(ROOT, defaults, overrides)

        self.assertGreater(len(watches), 100)
        self.assertTrue(
            any(
                watch.entity
                == "recipe/vonk-forge/inkling-small-nvfp4-sglang-dual"
                and watch.pinned_revision
                == "a74222ef6e690f851e2e4ff1c0be7dc1357be313"
                and watch.policy == "manual"
                for watch in watches
            )
        )

    def test_discovers_entity_and_recipe_sources_with_manifest_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model-versions").mkdir()
            (root / "runtime-distributions").mkdir()
            (root / "patch-bundles").mkdir()
            (root / "recipes").mkdir()
            model = {
                "kind": "model-version",
                "identity": {"publisher": "example", "slug": "model"},
                "version": "1.0",
                "source": {
                    "repository": "https://huggingface.co/example/model",
                    "revision": PINNED,
                },
            }
            recipe = {
                "kind": "recipe",
                "identity": {"publisher": "example", "slug": "recipe"},
                "provenance": {
                    "source_reference": f"https://github.com/example/recipe/tree/{PINNED}"
                },
            }
            (root / "model-versions/model.json").write_text(json.dumps(model))
            (root / "recipes/recipe.json").write_text(json.dumps(recipe))
            defaults = {"github": "default-branch", "huggingface": "default-branch"}
            overrides = {
                "recipe/example/recipe": {"policy": "default-branch", "ref": "stable"}
            }

            watches = drift.discover_watches(root, defaults, overrides)

            self.assertEqual(
                [watch.entity for watch in watches],
                [
                    "model-version/example/model",
                    "recipe/example/recipe",
                ],
            )
            self.assertEqual(watches[0].pinned_version, "1.0")
            self.assertEqual(watches[1].ref, "stable")

    def test_rejects_mismatched_embedded_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (*drift.ENTITY_DIRECTORIES, "recipes"):
                (root / directory).mkdir()
            document = {
                "kind": "model-version",
                "identity": {"publisher": "example", "slug": "model"},
                "source": {
                    "repository": f"https://huggingface.co/example/model/tree/{OBSERVED}",
                    "revision": PINNED,
                },
            }
            (root / "model-versions/model.json").write_text(json.dumps(document))
            with self.assertRaises(drift.DriftInputError):
                drift.discover_watches(root, {"huggingface": "default-branch"}, {})


class ObservationTests(unittest.TestCase):
    def watch(self, *, provider: str, policy: str) -> object:
        return drift.Watch(
            entity="runtime-distribution/example/runtime",
            source_path="runtime-distributions/runtime.json",
            provider=provider,
            repository="example/runtime",
            pinned_revision=PINNED,
            pinned_version="1.0",
            policy=policy,
            ref=None,
        )

    def test_github_latest_release_reports_advanced(self) -> None:
        release_url = "https://api.github.com/repos/example/runtime/releases/latest"
        tag_url = "https://api.github.com/repos/example/runtime/commits/v2.0.0"
        pinned_url = f"https://api.github.com/repos/example/runtime/commits/{PINNED}"
        client = FakeClient(
            {
                release_url: {"tag_name": "v2.0.0"},
                tag_url: {
                    "sha": OBSERVED,
                    "html_url": "https://github.com/example/runtime/commit/observed",
                    "commit": {"committer": {"date": "2026-08-23T00:00:00Z"}},
                },
                pinned_url: {
                    "sha": PINNED,
                    "html_url": "https://github.com/example/runtime/commit/pinned",
                    "commit": {"committer": {"date": "2026-08-01T00:00:00Z"}},
                },
            }
        )

        observation = drift.observe(
            self.watch(provider="github", policy="latest-release"), client
        )

        self.assertEqual(observation.status, "advanced")
        self.assertEqual(observation.observed_version, "v2.0.0")
        self.assertEqual(observation.observed_revision, OBSERVED)

    def test_hugging_face_default_branch_reports_current(self) -> None:
        url = "https://huggingface.co/api/models/example/runtime"
        client = FakeClient(
            {url: {"sha": PINNED, "lastModified": "2026-08-23T00:00:00Z"}}
        )

        observation = drift.observe(
            self.watch(provider="huggingface", policy="default-branch"), client
        )

        self.assertEqual(observation.status, "current")
        self.assertEqual(observation.observed_revision, PINNED)
        self.assertEqual(client.requests, [(url, "huggingface")])

    def test_missing_pinned_revision_is_distinct_from_drift(self) -> None:
        repository_url = "https://api.github.com/repos/example/runtime"
        head_url = "https://api.github.com/repos/example/runtime/commits/main"
        pinned_url = f"https://api.github.com/repos/example/runtime/commits/{PINNED}"
        client = FakeClient(
            {
                repository_url: {"default_branch": "main"},
                head_url: {
                    "sha": OBSERVED,
                    "html_url": "https://github.com/example/runtime/commit/observed",
                    "commit": {"committer": {"date": "2026-08-23T00:00:00Z"}},
                },
                pinned_url: drift.RemoteNotFound(pinned_url),
            }
        )

        observation = drift.observe(
            self.watch(provider="github", policy="default-branch"), client
        )

        self.assertEqual(observation.status, "missing")
        self.assertIn("no longer resolvable", observation.detail)


if __name__ == "__main__":
    unittest.main()
