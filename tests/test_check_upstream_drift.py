from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime
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
        expected = set()
        for directory, kind in (("models", "model"), ("recipes", "recipe")):
            for path in (ROOT / directory).glob("*.json"):
                document = json.loads(path.read_text())
                identity = document["identity"]
                source = (
                    document.get("source")
                    if kind == "model"
                    else document.get("provenance", {}).get("source_reference")
                )
                if source:
                    expected.add(f"{kind}/{identity['publisher']}/{identity['slug']}")
        self.assertEqual({watch.entity for watch in watches}, expected)
        self.assertEqual(len(watches), len(expected))
        self.assertTrue(all(watch.policy != "manual" for watch in watches))

    def test_rejects_watch_policy_for_an_absent_document(self) -> None:
        defaults, _ = drift.load_manifest(ROOT / "upstream-watch.json")
        with self.assertRaisesRegex(drift.DriftInputError, "unknown entity"):
            drift.discover_watches(
                ROOT, defaults, {"recipe/example/absent": {"policy": "default-branch"}}
            )

    def test_discovers_entity_and_recipe_sources_with_manifest_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "recipes").mkdir()
            model = json.loads(
                (
                    ROOT
                    / "contracts/src/vonk_forge_contracts/examples/model-definition.json"
                ).read_text()
            )
            model["identity"]["publisher"] = "example"
            model["identity"]["slug"] = "model"
            model["identity"]["family"]["publisher"] = "example"
            model["identity"]["family"]["slug"] = "model-family"
            model["identity"]["model"]["publisher"] = "example"
            model["identity"]["model"]["slug"] = "model"
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": PINNED,
            }
            recipe = json.loads(
                (
                    ROOT
                    / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
                ).read_text()
            )
            recipe["identity"] = {"publisher": "example", "slug": "recipe"}
            recipe["provenance"] = {
                "attribution": ["Example"],
                "source_reference": f"https://github.com/example/recipe/tree/{PINNED}",
            }
            (root / "models/model.json").write_text(json.dumps(model))
            (root / "recipes/recipe.json").write_text(json.dumps(recipe))
            defaults = {"github": "default-branch", "huggingface": "default-branch"}
            overrides = {
                "recipe/example/recipe": {"policy": "default-branch", "ref": "stable"}
            }

            watches = drift.discover_watches(root, defaults, overrides)

            self.assertEqual(
                [watch.entity for watch in watches],
                [
                    "model/example/model",
                    "recipe/example/recipe",
                ],
            )
            self.assertIsNone(watches[0].pinned_version)
            self.assertEqual(watches[1].ref, "stable")

    def test_github_lfs_model_is_a_git_commit_watch_on_its_repository(self) -> None:
        example = (
            ROOT
            / "contracts/src/vonk_forge_contracts/examples/model-definition-github-lfs.json"
        )
        source = json.loads(example.read_text())["source"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            (root / "recipes").mkdir()
            (root / "models/lfs.json").write_text(example.read_text())
            defaults = {"github": "default-branch", "huggingface": "default-branch"}

            (watch,) = drift.discover_watches(root, defaults, {})

        self.assertEqual(watch.provider, "github")
        self.assertEqual(
            watch.repository, source["repository"].removeprefix("https://github.com/")
        )
        self.assertEqual(watch.pinned_revision, source["revision"])
        self.assertIsNone(watch.release_source)

    def test_rejects_mismatched_embedded_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (*drift.ENTITY_DIRECTORIES, "recipes"):
                (root / directory).mkdir()
            document = {
                "kind": "model",
                "identity": {"publisher": "example", "slug": "model"},
                "source": {
                    "repository": f"https://huggingface.co/example/model/tree/{OBSERVED}",
                    "revision": PINNED,
                },
            }
            (root / "models/model.json").write_text(json.dumps(document))
            with self.assertRaises(drift.DriftInputError):
                drift.discover_watches(root, {"huggingface": "default-branch"}, {})

    def test_embedded_dockerfile_pin_change_is_visible_without_provenance_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("recipes", "models", "context"):
                (root / directory).mkdir()
            recipe = json.loads(
                (
                    ROOT
                    / "contracts/src/vonk_forge_contracts/examples/recipe-source-build.json"
                ).read_text()
            )
            recipe["identity"] = {"publisher": "example", "slug": "embedded"}
            recipe["execution"]["build"]["dockerfile"] = "context/Dockerfile"
            recipe["execution"]["build"]["context"]["path"] = "context"
            recipe["runtime"]["engine"] = "vllm"
            (root / "recipes/embedded.json").write_text(json.dumps(recipe))
            model = json.loads(
                (
                    ROOT
                    / "contracts/src/vonk_forge_contracts/examples/model-definition.json"
                ).read_text()
            )
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": PINNED,
            }
            (root / "models/synthetic-tiny-fp16.json").write_text(json.dumps(model))
            dockerfile = root / "context/Dockerfile"
            dockerfile.write_text(
                "ARG BASE_IMAGE=registry.example/runtime:stable\nFROM ${BASE_IMAGE}\n"
            )

            first = drift.embedded_inventory(root, [])
            first_input = next(
                item
                for item in first["recipes"][0]["inputs"]
                if item["kind"] == "docker-image"
            )
            self.assertEqual(first_input["status"], "mutable")

            dockerfile.write_text(
                "ARG BASE_IMAGE=registry.example/runtime@sha256:"
                + "a" * 64
                + "\nFROM ${BASE_IMAGE}\n"
            )
            second = drift.embedded_inventory(root, [])
            second_input = next(
                item
                for item in second["recipes"][0]["inputs"]
                if item["kind"] == "docker-image"
            )
            self.assertEqual(second_input["status"], "pin_only")
            self.assertIn("no configured moving channel", second_input["reason"])


class ObservationTests(unittest.TestCase):
    def watch(self, *, provider: str, policy: str) -> object:
        return drift.Watch(
            entity="recipe/example/runtime",
            source_path="recipes/runtime.json",
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


class ReleaseAssetWatchTests(unittest.TestCase):
    def watch(self):
        defaults, overrides = drift.load_manifest(ROOT / "upstream-watch.json")
        return next(
            watch
            for watch in drift.discover_watches(ROOT, defaults, overrides)
            if watch.entity == "model/valeoai/naf"
        )

    def test_release_source_keeps_release_and_asset_identity(self) -> None:
        watch = self.watch()
        self.assertEqual(watch.pinned_revision, "264676230")
        self.assertEqual(watch.policy, "latest-release")
        self.assertEqual(watch.release_source.assets[0].asset_id, 320107386)

    def test_release_observation_uses_ids_without_git_commit_lookups(self) -> None:
        watch = self.watch()
        base = "https://api.github.com/repos/valeoai/NAF/releases"
        pinned = {"id": 264676230, "assets": [{"id": 320107386}]}
        latest = {
            **pinned,
            "tag_name": "model",
            "html_url": "https://github.com/valeoai/NAF/releases/tag/model",
            "published_at": "2025-11-24T12:00:00Z",
        }
        client = FakeClient({f"{base}/264676230": pinned, f"{base}/latest": latest})
        self.assertEqual(drift.observe(watch, client).status, "current")
        self.assertEqual(len(client.requests), 2)
        client.responses[f"{base}/latest"] = {**latest, "id": 264676231}
        self.assertEqual(drift.observe(watch, client).status, "advanced")
        # A release tag still existing cannot hide a deleted/replaced asset.
        client.responses[f"{base}/264676230"] = {**pinned, "assets": []}
        self.assertEqual(drift.observe(watch, client).status, "missing")
        client.responses[f"{base}/264676230"] = drift.RemoteNotFound("deleted release")
        self.assertEqual(drift.observe(watch, client).status, "missing")


MIA_BANNER = """# 🚨 NEW, BETTER RECIPE AVAILABLE — EVERYONE SHOULD USE IT 🚨

> [!IMPORTANT]
> ## This recipe is superseded. **Everything should be using the new one.**
>
> # 👉 [github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) 👈

<h1 align="center">GLM-5.3-Flash EXL3 on DGX Sparks with TensorFold</h1>

<a href="https://github.com/sponsors/MiaAI-Lab">Sponsor</a>
"""
MIA_OLD = "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
MIA_NEW_README = """<h1>GLM-5.3-Flash on DGX Sparks with TensorFold</h1>

> [!IMPORTANT]
> **This is a brand-new recipe, version 1104.** It is not an update of
> [old](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
"""
NEWHEAD = "c" * 40
NOW = datetime(2026, 11, 6, tzinfo=UTC)


class CreatorSignalTests(unittest.TestCase):
    def test_mia_banner_names_the_successor(self) -> None:
        notice = drift.find_supersession(MIA_BANNER, MIA_OLD, PINNED)
        assert notice is not None
        self.assertEqual(notice.kind, "supersession")
        self.assertEqual(
            notice.dedupe_key,
            f"supersession:{MIA_OLD}:miaai-lab/glm-5.3-flash-dgx-sparks-1104".lower(),
        )
        self.assertEqual(notice.linked_repo, "MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104")
        self.assertEqual(notice.source_repo, MIA_OLD)
        self.assertEqual(notice.source_commit, PINNED)

    def test_ordinary_readmes_and_back_links_are_not_notices(self) -> None:
        self.assertIsNone(
            drift.find_supersession(
                MIA_NEW_README, "MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104", PINNED
            )
        )
        self.assertIsNone(
            drift.find_supersession("# Tool\nSee https://github.com/a/b", "a/c", PINNED)
        )
        self_link = "> Deprecated, see https://github.com/a/c"
        self.assertIsNone(drift.find_supersession(self_link, "a/c", PINNED))
        deep = "\n" * 80 + "superseded https://github.com/a/b"
        self.assertIsNone(drift.find_supersession(deep, "a/c", PINNED))

    def test_new_repos_match_carried_families_inside_the_window(self) -> None:
        listing = [
            {
                "name": "GLM-5.3-Flash-DGX-Sparks-1104",
                "created_at": "2026-11-04T11:54:11Z",
                "description": "GLM-5.3-Flash on DGX Sparks",
            },
            {
                "name": "Hy4-100-HTML-Files",
                "created_at": "2026-11-04T09:03:42Z",
                "description": "art",
            },
            {
                "name": "old-glm",
                "created_at": "2026-08-01T00:00:00Z",
                "description": "glm",
            },
            {
                "name": "glm-fork",
                "created_at": "2026-11-05T00:00:00Z",
                "description": None,
                "fork": True,
            },
            {
                "name": "deepseek-pinned",
                "created_at": "2026-11-05T00:00:00Z",
                "description": None,
            },
        ]
        found = drift.match_new_repos(
            "MiaAI-Lab",
            listing,
            NOW,
            {"glm", "deepseek"},
            {"miaai-lab/deepseek-pinned"},
        )
        self.assertEqual(
            [(r.linked_repo, r.matched_family) for r in found],
            [("MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104", "glm")],
        )
        with self.assertRaises(TypeError):
            drift.match_new_repos("x", {"message": "rate limited"}, NOW, set(), set())

    def test_scan_collects_both_signals_and_isolates_failures(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "models").mkdir()
        shutil.copy(
            ROOT / "models/glm-5-3-dflash2-425aa615.json", root / "models" / "m.json"
        )
        (root / "creators.json").write_text(
            json.dumps(
                {
                    "creators": [
                        {"url": "https://github.com/MiaAI-Lab"},
                        {"url": "https://huggingface.co/x"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        watches = [
            drift.Watch(
                "model/incoai/glm-5-3-dflash2-425aa615",
                "m",
                "github",
                MIA_OLD,
                PINNED,
                None,
                "default-branch",
                None,
            ),
            drift.Watch(
                "model/p/b",
                "m",
                "github",
                "gone/repo",
                PINNED,
                None,
                "default-branch",
                None,
            ),
        ]
        api = "https://api.github.com"
        readme = base64.b64encode(MIA_BANNER.encode()).decode()
        client = FakeClient(
            {
                f"{api}/repos/{MIA_OLD}/commits/HEAD": {
                    "sha": OBSERVED,
                    "html_url": "u",
                },
                f"{api}/repos/{MIA_OLD}/readme?ref={OBSERVED}": {"content": readme},
                f"{api}/repos/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/commits/HEAD": {
                    "sha": NEWHEAD,
                    "html_url": "u",
                },
                f"{api}/repos/gone/repo/commits/HEAD": drift.RemoteNotFound("gone"),
                f"{api}/users/MiaAI-Lab/repos?sort=created&direction=desc&per_page=30&type=owner": [
                    {
                        "name": "GLM-5.3-Flash-DGX-Sparks-1104",
                        "created_at": "2026-11-04T11:54:11Z",
                        "description": "GLM-5.3-Flash",
                    }
                ],
                f"{api}/users/gone/repos?sort=created&direction=desc&per_page=30&type=owner": drift.RemoteNotFound(
                    "gone"
                ),
            }
        )
        signals = drift.scan_creator_signals(root, watches, client, NOW)
        by_kind = {f["kind"]: f for f in signals["findings"]}
        self.assertEqual(set(by_kind), {"supersession", "new-repo"})
        old = by_kind["supersession"]
        self.assertEqual(old["linked_repo"], "MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104")
        self.assertEqual(
            (old["source_commit"], old["linked_commit"]), (OBSERVED, NEWHEAD)
        )
        self.assertEqual(old["recipe_slugs"], ["model/incoai/glm-5-3-dflash2-425aa615"])
        new = by_kind["new-repo"]
        self.assertEqual(new["matched_family"], "glm")
        self.assertEqual(new["linked_commit"], NEWHEAD)
        self.assertEqual(new["recipe_slugs"], ["model/incoai/glm-5-3-dflash2-425aa615"])
        self.assertEqual(
            sorted(e["subject"] for e in signals["signal_errors"]),
            ["gone", "gone/repo"],
        )
        self.assertEqual(
            drift.report([], "t", None, signals)["creator_signals"], signals
        )


if __name__ == "__main__":
    unittest.main()
