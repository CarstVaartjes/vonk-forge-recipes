from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from email.message import Message
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.error import HTTPError, URLError
from urllib.request import Request

from repo_root import ROOT

from recipe_tooling.upstream_drift import embedded as drift_embedded
from recipe_tooling.upstream_drift import inventory as drift_inventory
from recipe_tooling.upstream_drift import issues as drift_issues
from recipe_tooling.upstream_drift import model as drift_model
from recipe_tooling.upstream_drift import observe as drift_observe
from recipe_tooling.upstream_drift import signals as drift_signals
from recipe_tooling.upstream_http import UpstreamBusy, fetch, forget_busy

PINNED = "a" * 40
OBSERVED = "b" * 40


class FakeClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str]] = []

    def json(self, url: str, provider: str) -> Any:
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
            f"https://github.com/example/project/tree/{PINNED}/runs/launch",
            f"https://github.com/example/project@{PINNED}",
            f"https://github.com/example/project.git@{PINNED}",
        ):
            with self.subTest(source=source):
                location = drift_inventory.parse_source(source)
                self.assertEqual(location.provider, "github")
                self.assertEqual(location.repository, "example/project")
        self.assertEqual(
            drift_inventory.parse_source(
                f"https://github.com/example/project/tree/{PINNED}"
            ).embedded_revision,
            PINNED,
        )
        self.assertEqual(
            drift_inventory.parse_source(
                f"https://github.com/example/project/blob/{PINNED}/README.md"
            ).embedded_revision,
            PINNED,
        )
        self.assertEqual(
            drift_inventory.parse_source(
                f"https://github.com/example/project/tree/{PINNED}/runs/launch"
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
                location = drift_inventory.parse_source(source)
                self.assertEqual(location.provider, "huggingface")
                self.assertEqual(location.repository, "example/project")

    def test_rejects_credentials_and_non_https_sources(self) -> None:
        for source in (
            "http://github.com/example/project",
            "https://user@example.com/example/project",
            "https://github.com/example/project/blob/main/README.md",
            "https://github.com/example/project/tree/main/runs/launch",
            "https://huggingface.co/example/project/blob/main/README.md",
        ):
            with self.assertRaises(drift_model.DriftInputError):
                drift_inventory.parse_source(source)


class DiscoveryTests(unittest.TestCase):
    def test_catalog_sources_are_all_discoverable(self) -> None:
        defaults, overrides = drift_inventory.load_manifest(
            ROOT / "policy/upstream-watch.json"
        )
        watches = drift_inventory.discover_watches(ROOT, defaults, overrides)
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
        defaults, _ = drift_inventory.load_manifest(ROOT / "policy/upstream-watch.json")
        with self.assertRaisesRegex(drift_model.DriftInputError, "unknown entity"):
            drift_inventory.discover_watches(
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

            watches = drift_inventory.discover_watches(root, defaults, overrides)

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

            (watch,) = drift_inventory.discover_watches(root, defaults, {})

        self.assertEqual(watch.provider, "github")
        self.assertEqual(
            watch.repository, source["repository"].removeprefix("https://github.com/")
        )
        self.assertEqual(watch.pinned_revision, source["revision"])
        self.assertIsNone(watch.release_source)

    def test_rejects_mismatched_embedded_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (*drift_model.ENTITY_DIRECTORIES, "recipes"):
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
            with self.assertRaises(drift_model.DriftInputError):
                drift_inventory.discover_watches(
                    root, {"huggingface": "default-branch"}, {}
                )

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

            first = drift_embedded.embedded_inventory(root, [])
            first_input = next(
                item
                for item in first["recipes"][0]["inputs"]  # pyright: ignore[reportIndexIssue]
                if item["kind"] == "docker-image"
            )
            self.assertEqual(first_input["status"], "mutable")

            dockerfile.write_text(
                "ARG BASE_IMAGE=registry.example/runtime@sha256:"
                + "a" * 64
                + "\nFROM ${BASE_IMAGE}\n"
            )
            second = drift_embedded.embedded_inventory(root, [])
            second_input = next(
                item
                for item in second["recipes"][0]["inputs"]  # pyright: ignore[reportIndexIssue]
                if item["kind"] == "docker-image"
            )
            self.assertEqual(second_input["status"], "pin_only")
            self.assertIn("no configured moving channel", second_input["reason"])


class ObservationTests(unittest.TestCase):
    def watch(self, *, provider: str, policy: str) -> Any:
        return drift_model.Watch(
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

        observation = drift_observe.observe(
            self.watch(provider="github", policy="latest-release"),
            client,  # pyright: ignore[reportArgumentType]
        )

        self.assertEqual(observation.status, "advanced")
        self.assertEqual(observation.observed_version, "v2.0.0")
        self.assertEqual(observation.observed_revision, OBSERVED)

    def test_hugging_face_default_branch_reports_current(self) -> None:
        url = "https://huggingface.co/api/models/example/runtime"
        client = FakeClient(
            {url: {"sha": PINNED, "lastModified": "2026-08-23T00:00:00Z"}}
        )

        observation = drift_observe.observe(
            self.watch(provider="huggingface", policy="default-branch"),
            client,  # pyright: ignore[reportArgumentType]
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
                pinned_url: drift_model.RemoteNotFound(pinned_url),
            }
        )

        observation = drift_observe.observe(
            self.watch(provider="github", policy="default-branch"),
            client,  # pyright: ignore[reportArgumentType]
        )

        self.assertEqual(observation.status, "missing")
        self.assertIn("no longer resolvable", observation.detail)  # pyright: ignore[reportArgumentType]

    def test_only_an_anonymous_hugging_face_401_means_missing(self) -> None:
        for provider, token, expected in (
            ("huggingface", None, "missing"),
            ("huggingface", "set", "error"),
            ("github", None, "error"),
        ):

            def unauthorized(request: Any, timeout: float) -> Any:
                error = HTTPError(
                    request.full_url, 401, "Unauthorized", Message(), None
                )
                error.close()
                raise error

            client = drift_observe.ProviderClient(
                github_token=token, huggingface_token=token, opener=unauthorized
            )
            with self.subTest(provider=provider, token=token):
                observation = drift_observe.observe(
                    self.watch(provider=provider, policy="default-branch"), client
                )
                self.assertEqual(observation.status, expected)


class RateLimitTests(unittest.TestCase):
    """A rate limit is retried within a bound; if it persists the item is not checked."""

    class Responses:
        def __init__(self, codes: list[Any], retry_after: str | None = None) -> None:
            self.codes = codes
            self.retry_after = retry_after
            self.calls = 0

        def __call__(self, request: Any, timeout: float) -> Any:
            code = self.codes[min(self.calls, len(self.codes) - 1)]
            self.calls += 1
            if code == 200:
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = b'{"sha": "x"}'
                return response
            if code == "net":
                raise URLError("down")
            headers = Message()
            if self.retry_after is not None:
                headers["Retry-After"] = self.retry_after
            error = HTTPError(request.full_url, code, "busy", headers, None)
            error.close()
            raise error

    def setUp(self) -> None:
        # unittest (the daily workflow) does not run conftest's autouse fixture
        forget_busy()
        self.addCleanup(forget_busy)
        self.naps: list[float] = []

    def get(self, opener: Any, url: str = "https://huggingface.co/api/x") -> Any:
        client = drift_observe.ProviderClient(opener=opener, sleep=self.naps.append)
        return client.json(url, "huggingface")

    def observe(self, opener: Any) -> Any:
        watch = drift_model.Watch(
            entity="recipe/example/runtime",
            source_path="recipes/runtime.json",
            provider="huggingface",
            repository="example/runtime",
            pinned_revision=PINNED,
            pinned_version=None,
            policy="default-branch",
            ref=None,
        )
        client = drift_observe.ProviderClient(opener=opener, sleep=self.naps.append)
        return drift_observe.observe(watch, client)

    def test_recovers_after_honouring_retry_after(self) -> None:
        opener = self.Responses([429, 503, 200], retry_after="2")
        self.assertEqual(self.get(opener), {"sha": "x"})
        self.assertEqual(self.naps, [2.0, 2.0])

    def test_retry_after_may_be_an_http_date(self) -> None:
        epoch = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
        when = format_datetime(epoch + timedelta(seconds=10), usegmt=True)
        fetch(
            Request("https://huggingface.co/api/x"),
            lambda r: r,
            opener=self.Responses([429, 200], retry_after=when),
            sleep=self.naps.append,
            clock=lambda: epoch.timestamp(),
        )
        self.assertEqual(self.naps, [10.0])

    def test_persistent_limit_is_not_checked_and_not_an_error(self) -> None:
        opener = self.Responses([429])
        observation = self.observe(opener)
        self.assertEqual(observation.status, "unchecked")
        self.assertEqual(opener.calls, len(self.naps) + 1)
        counts = drift_observe.report([observation], "now")["counts"]
        self.assertEqual((counts["unchecked"], counts["error"]), (1, 0))  # pyright: ignore[reportIndexIssue]

    def test_our_own_faults_stay_errors(self) -> None:
        for codes in ([500], ["net"]):
            with self.subTest(codes=codes):
                self.assertEqual(self.observe(self.Responses(codes)).status, "error")

    def test_a_long_retry_after_marks_the_host_busy_without_waiting(self) -> None:
        now = [100.0]
        opener = self.Responses([429], retry_after="3600")
        request = Request("https://huggingface.co/api/x")

        def call() -> Any:
            return fetch(
                request,
                lambda r: r,
                opener=opener,
                sleep=self.naps.append,
                clock=lambda: now[0],
            )

        with self.assertRaises(UpstreamBusy):
            call()
        with self.assertRaises(UpstreamBusy):
            call()  # remembered: the provider is not asked again
        self.assertEqual((opener.calls, self.naps), (1, []))
        now[0] += 3601
        with self.assertRaises(UpstreamBusy):
            call()  # the memory expired: asked again
        self.assertEqual(opener.calls, 2)

    def test_an_anonymous_limit_does_not_block_an_authenticated_read(self) -> None:
        url = "https://huggingface.co/api/x"
        with self.assertRaises(UpstreamBusy):
            fetch(
                Request(url),
                lambda r: r,
                opener=self.Responses([429], retry_after="3600"),
                sleep=self.naps.append,
            )
        authenticated = Request(url, headers={"Authorization": "Bearer t"})
        opener = self.Responses([200])
        fetch(authenticated, lambda r: r, opener=opener, sleep=self.naps.append)
        self.assertEqual(opener.calls, 1)


class ReleaseAssetWatchTests(unittest.TestCase):
    def watch(self):
        defaults, overrides = drift_inventory.load_manifest(
            ROOT / "policy/upstream-watch.json"
        )
        return next(
            watch
            for watch in drift_inventory.discover_watches(ROOT, defaults, overrides)
            if watch.entity == "model/valeoai/naf"
        )

    def test_release_source_keeps_release_and_asset_identity(self) -> None:
        watch = self.watch()
        self.assertEqual(watch.pinned_revision, "264676230")
        self.assertEqual(watch.policy, "latest-release")
        self.assertEqual(watch.release_source.assets[0].asset_id, 320107386)  # pyright: ignore[reportOptionalMemberAccess]

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
        self.assertEqual(drift_observe.observe(watch, client).status, "current")  # pyright: ignore[reportArgumentType]
        self.assertEqual(len(client.requests), 2)
        client.responses[f"{base}/latest"] = {**latest, "id": 264676231}
        self.assertEqual(drift_observe.observe(watch, client).status, "advanced")  # pyright: ignore[reportArgumentType]
        # A release tag still existing cannot hide a deleted/replaced asset.
        client.responses[f"{base}/264676230"] = {**pinned, "assets": []}
        self.assertEqual(drift_observe.observe(watch, client).status, "missing")  # pyright: ignore[reportArgumentType]
        client.responses[f"{base}/264676230"] = drift_model.RemoteNotFound(
            "deleted release"
        )
        self.assertEqual(drift_observe.observe(watch, client).status, "missing")  # pyright: ignore[reportArgumentType]


MIA_BANNER = """# 🚨 NEW, BETTER RECIPE AVAILABLE — EVERYONE SHOULD USE IT 🚨

> [!IMPORTANT]
> ## This recipe is superseded. **Everything should be using the new one.**
>
> # 👉 [github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) 👈

<h1 align="center">GLM-5.3-Flash EXL3 on DGX Sparks with TensorFold</h1>

<a href="https://github.com/sponsors/MiaAI-Lab">Sponsor</a>
"""
MIA_OLD = "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
MIA_NEW = "MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104"
MIA_NEW_README = """<h1>GLM-5.3-Flash on DGX Sparks with TensorFold</h1>

> [!IMPORTANT]
> **This is a brand-new recipe, version 1104.** It is not an update of
> [old](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
"""
MIA_LISTING = {
    "name": "GLM-5.3-Flash-DGX-Sparks-1104",
    "created_at": "2026-11-04T11:54:11Z",
    "description": "GLM-5.3-Flash on DGX Sparks (ver 1104) with TensorFold",
}
# Real 2026-11 listings that the first matcher flagged wrongly.
NOISE_LISTINGS = [
    ("microsoft", "amplifier-bundle-ui-use", "2026-11-02T00:00:00Z",
     "Shared browser and computer UI-use behavior with optional decision models"),
    ("microsoft", "amplifier-constellation-instance", "2026-11-03T01:13:49Z",
     "Independent installed agents with setup, profiles, authorized context, and management UI/API"),
    ("microsoft", "amplifier-lib-decisions", "2026-11-02T00:00:00Z",
     "Optional typed decision-model contracts for the Amplifier ecosystem"),
    ("microsoft", "amplifier-bundle-alexa-tester", "2026-11-02T00:00:00Z",
     "Test and validate Alexa custom skills against Amazon's real NLU"),
    ("huggingface", "safeagents", "2026-11-03T13:52:42Z",
     "Safely run your agents using any coding harness"),
    ("Tencent-Hunyuan", "FlashTopK", "2026-11-05T10:38:09Z",
     "Fast, exact fused top-k selection for learned sparse attention training and inference, from HyScale (Tencent Hunyuan)."),
]  # fmt: skip
NEWHEAD = "c" * 40
NOW = datetime(2026, 11, 6, tzinfo=UTC)
API = "https://api.github.com"


class QuietClient(FakeClient):
    """Unknown URLs are absent upstream, like a real 404."""

    def json(self, url: str, provider: str) -> Any:
        if url not in self.responses:
            raise drift_model.RemoteNotFound(url)
        return super().json(url, provider)


def _listing_url(owner: str, page: int = 1) -> str:
    return (
        f"{API}/users/{owner}/repos?sort=created&direction=desc"
        f"&per_page=100&page={page}&type=owner"
    )


def _watch(entity: str, repository: str, **fields: Any) -> Any:
    provider = "github" if entity.startswith("recipe/") else "huggingface"
    return drift_model.Watch(
        entity, "x.json", provider, repository, PINNED, None, "default-branch", None,
        **fields,
    )  # fmt: skip


# A fixed, recipe-shaped catalog: the signals must not depend on live data.
FIXTURE_WATCHES = [
    _watch("model/example/glm-a", "example/glm-a", family="glm"),
    _watch("model/example/qwen-a", "example/qwen-a", family="qwen"),
    _watch("recipe/vonk-forge/glm-old-a", MIA_OLD, models=("model/example/glm-a",)),
    _watch("recipe/vonk-forge/glm-old-b", MIA_OLD, models=("model/example/glm-a",)),
    _watch("recipe/vonk-forge/glm-other", "other/glm-kit", models=("model/example/glm-a",)),
    _watch("recipe/vonk-forge/qwen-x", "other/qwen-kit", models=("model/example/qwen-a",)),
]  # fmt: skip
FIXTURE_FAMILIES = {
    "glm", "qwen", "agents-a1", "decision-3", "ui-mate", "real-esrgan",
    "hunyuan-video", "deepseek-flash", "gem-x",
}  # fmt: skip


class CreatorSignalTests(unittest.TestCase):
    watches = FIXTURE_WATCHES
    families = FIXTURE_FAMILIES

    def test_live_catalog_recipe_watches_are_structurally_sound(self) -> None:
        defaults, overrides = drift_inventory.load_manifest(
            ROOT / "policy/upstream-watch.json"
        )
        live = drift_inventory.discover_watches(ROOT, defaults, overrides)
        recipes = [w for w in live if w.entity.startswith("recipe/")]
        self.assertTrue(recipes)
        for watch in recipes:
            slug = watch.entity.split("/")[2]
            self.assertTrue((ROOT / "recipes" / f"{slug}.json").is_file(), slug)
            self.assertTrue(watch.models, slug)
        self.assertTrue(any(w.family for w in live if w.entity.startswith("model/")))

    def test_mia_banner_names_the_successor(self) -> None:
        notice = drift_signals.find_supersession(MIA_BANNER, MIA_OLD, PINNED)
        assert notice is not None
        self.assertEqual(notice.kind, "supersession")
        self.assertEqual(notice.label, "upstream-superseded")
        self.assertEqual(notice.linked_repo, MIA_NEW)
        self.assertEqual(notice.source_commit, PINNED)
        self.assertEqual(notice.dedupe_key, f"supersession:{MIA_OLD}:{MIA_NEW}".lower())

    def test_ordinary_readmes_and_non_repo_links_are_not_notices(self) -> None:
        for readme in (
            MIA_NEW_README,
            "# Tool\nSee https://github.com/a/b",
            "> Deprecated, see https://github.com/a/c",
            "\n" * 80 + "superseded https://github.com/a/b",
            # real false positives: an attachment and a pull request link
            "> This is superseded, see https://github.com/user-attachments/assets/1f2e",
            "> Superseded by the work in https://github.com/a/b/pull/12",
            "> Deprecated, see https://github.com/a/b/issues/3 and https://github.com/orgs/x/y",
        ):
            with self.subTest(readme=readme):
                self.assertIsNone(
                    drift_signals.find_supersession(readme, "a/c", PINNED)
                )

    def test_new_repos_match_whole_family_slugs_inside_the_window(self) -> None:
        for owner, name, created, description in NOISE_LISTINGS:
            with self.subTest(repo=name):
                listing = [
                    {"name": name, "created_at": created, "description": description}
                ]
                self.assertEqual(
                    drift_signals.match_new_repos(
                        owner, listing, NOW, self.families, set()
                    ),
                    [],
                )
        listing = [
            MIA_LISTING,
            {
                "name": "old-glm",
                "created_at": "2026-08-01T00:00:00Z",
                "description": "glm",
            },
            {"name": "glm-fork", "created_at": "2026-11-05T00:00:00Z", "fork": True},
            {"name": "glm-pinned", "created_at": "2026-11-05T00:00:00Z"},
        ]
        found = drift_signals.match_new_repos(
            "MiaAI-Lab", listing, NOW, self.families, {"miaai-lab/glm-pinned"}
        )
        self.assertEqual(
            [(f.linked_repo, f.matched_family) for f in found], [(MIA_NEW, "glm")]
        )
        self.assertEqual(found[0].label, "upstream-new-repo")
        with self.assertRaises(TypeError):
            drift_signals.match_new_repos(
                "x", {"message": "rate limited"}, NOW, set(), set()
            )

    def test_listing_pages_until_the_window_is_left(self) -> None:
        recent = [
            {"name": f"r{i}", "created_at": "2026-11-05T00:00:00Z"} for i in range(100)
        ]
        old = [{"name": "o", "created_at": "2026-01-01T00:00:00Z"}]
        client = FakeClient(
            {
                _listing_url("o", 1): recent,
                _listing_url("o", 2): old,
                _listing_url("o", 3): [],
            }
        )
        self.assertEqual(len(drift_signals._recent_repos(client, "o", NOW)[0]), 101)  # pyright: ignore[reportArgumentType]
        self.assertEqual(
            [u for u, _ in client.requests],
            [_listing_url("o", 1), _listing_url("o", 2)],
        )

    def test_paging_cap_is_a_signal_error(self) -> None:
        full = [
            {"name": f"r{i}", "created_at": "2026-11-05T00:00:00Z"} for i in range(100)
        ]
        client = FakeClient({_listing_url("o", n): full for n in range(1, 6)})
        self.assertTrue(drift_signals._recent_repos(client, "o", NOW)[1])  # pyright: ignore[reportArgumentType]
        root = self._scan_root(
            json.dumps(
                {
                    "_comment": "c",
                    "creators": [
                        {
                            "name": "o",
                            "url": "https://github.com/o",
                            "aliases": [],
                            "focus": "f",
                        }
                    ],
                }
            )
        )
        signals = drift_signals.scan_creator_signals(root, [], client, NOW)  # pyright: ignore[reportArgumentType]
        self.assertEqual([e.subject for e in signals.signal_errors], ["o"])
        self.assertIn("5 pages", signals.signal_errors[0].detail)

    def test_label_is_derived_from_kind(self) -> None:
        finding = drift_signals._finding(
            "new-repo", "o", "o/r", source_repo=None, source_commit=None,
            linked_commit=None, recipe_slugs=[], matched_family="glm", detail="d",
        )  # fmt: skip
        self.assertEqual(finding.label, "upstream-new-repo")
        record = {**finding.model_dump(), "label": "upstream-superseded"}
        with self.assertRaises(ValueError):
            drift_model.CreatorFinding.model_validate(record)

    def _scan_root(self, creators: str) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        root.joinpath("policy").mkdir(exist_ok=True)
        (root / "policy/creators.json").write_text(creators, encoding="utf-8")
        return root

    def test_scan_names_real_recipes_for_both_signals(self) -> None:
        root = self._scan_root(
            json.dumps(
                {
                    "_comment": "c",
                    "creators": [
                        {
                            "name": "Mia",
                            "url": "https://github.com/MiaAI-Lab",
                            "aliases": [],
                            "focus": "f",
                        }
                    ],
                }
            )
        )
        readme = base64.b64encode(MIA_BANNER.encode()).decode()
        client = QuietClient(
            {
                f"{API}/repos/{MIA_OLD}/commits/HEAD": {
                    "sha": OBSERVED,
                    "html_url": "u",
                },
                f"{API}/repos/{MIA_OLD}/readme?ref={OBSERVED}": {"content": readme},
                f"{API}/repos/{MIA_NEW}/commits/HEAD": {
                    "sha": NEWHEAD,
                    "html_url": "u",
                },
                _listing_url("MiaAI-Lab"): [MIA_LISTING],
            }
        )
        signals = drift_signals.scan_creator_signals(root, self.watches, client, NOW)  # pyright: ignore[reportArgumentType]
        by_kind = {f.kind: f for f in signals.findings}
        self.assertEqual(set(by_kind), {"supersession", "new-repo"})
        old, new = by_kind["supersession"], by_kind["new-repo"]
        self.assertEqual((old.source_commit, old.linked_commit), (OBSERVED, NEWHEAD))
        self.assertEqual(new.linked_commit, NEWHEAD)
        self.assertEqual(old.recipe_slugs, ["glm-old-a", "glm-old-b"])
        self.assertEqual(new.recipe_slugs, ["glm-old-a", "glm-old-b", "glm-other"])
        self.assertEqual(
            drift_observe.report([], "t", None, signals)["creator_signals"],
            signals.model_dump(mode="json"),
        )

    def test_bad_creators_file_is_a_signal_error_not_a_lost_scan(self) -> None:
        root = self._scan_root(
            '{"_comment": "c", "creators": [{"name": "x", "extra": 1}]}'
        )
        readme = base64.b64encode(MIA_BANNER.encode()).decode()
        client = QuietClient(
            {
                f"{API}/repos/{MIA_OLD}/commits/HEAD": {
                    "sha": OBSERVED,
                    "html_url": "u",
                },
                f"{API}/repos/{MIA_OLD}/readme?ref={OBSERVED}": {"content": readme},
            }
        )
        signals = drift_signals.scan_creator_signals(root, self.watches, client, NOW)  # pyright: ignore[reportArgumentType]
        self.assertEqual([f.kind for f in signals.findings], ["supersession"])
        self.assertIn(
            "policy/creators.json", [e.subject for e in signals.signal_errors]
        )

    def test_detail_cannot_close_the_issue_fence(self) -> None:
        finding = drift_signals._finding(
            "supersession", "a/b", "a/c", source_repo="a/b", source_commit=PINNED,
            linked_commit=None, recipe_slugs=[], matched_family=None, detail="x ``` y",
        )  # fmt: skip
        self.assertNotIn("```", finding.detail)


FAKE_GH = """#!{python}
import json, sys
state_path = {state!r}
state = json.load(open(state_path))
args = sys.argv[1:]
state["log"].append(args)

def flag(name):
    return args[args.index(name) + 1]

def save():
    json.dump(state, open(state_path, "w"))

if args[:2] == ["issue", "list"]:
    rows = [i for i in state["issues"] if flag("--label") in i["labels"]]
    print(json.dumps([{{k: i[k] for k in ("number", "state", "title")}} for i in rows]))
elif args[:2] == ["issue", "create"]:
    if "failme" in flag("--title"):
        save()
        sys.exit("refused")
    state["issues"].append({{"number": len(state["issues"]) + 1, "state": "OPEN",
        "title": flag("--title"), "body": flag("--body"), "labels": [flag("--label")]}})
elif args[:2] == ["issue", "edit"]:
    for i in state["issues"]:
        if str(i["number"]) == args[2]:
            i["body"] = flag("--body")
save()
"""


class IssueWritingTests(unittest.TestCase):
    def _finding(self, kind: str, source: str, linked: str) -> Any:
        return drift_signals._finding(
            kind, source, linked, source_repo=source if kind == "supersession" else None,  # pyright: ignore[reportArgumentType]
            source_commit=None, linked_commit=None, recipe_slugs=["r"], matched_family=None,
            detail="d",
        )  # fmt: skip

    def _issue(
        self, number: int, finding: Any, state: str = "OPEN"
    ) -> dict[str, object]:
        return {
            "number": number,
            "state": state,
            "title": f"[{finding.label}] {finding.dedupe_key}",
            "body": f"```json\n{finding.model_dump_json()}\n```",
            "labels": [finding.label],
        }

    def test_create_refresh_skip_closed_isolate_failure_and_never_close(self) -> None:
        fresh = self._finding("supersession", "o/fresh", "o/next")
        open_one = self._finding("supersession", "o/open", "o/next")
        closed = self._finding("supersession", "o/closed", "o/next")
        refused = self._finding("new-repo", "o", "o/failme")
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        state_path = directory / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "log": [],
                    "issues": [
                        self._issue(1, open_one),
                        self._issue(2, closed, "CLOSED"),
                    ],
                }
            )
        )
        gh = directory / "gh"
        gh.write_text(FAKE_GH.format(python=sys.executable, state=str(state_path)))
        gh.chmod(0o755)
        signals = drift_model.CreatorSignals(
            findings=[fresh, open_one, closed, refused],
        )
        with mock.patch.dict(
            "os.environ", {"PATH": f"{directory}:{os.environ['PATH']}"}
        ):
            failures = drift_issues.file_issues(signals)
        state = json.loads(state_path.read_text())
        self.assertEqual(failures, 1)
        by_title = {i["title"]: i for i in state["issues"]}
        created = by_title[f"[upstream-superseded] {fresh.dedupe_key}"]
        self.assertIn(f"<!-- dedupe-key: {fresh.dedupe_key} -->", created["body"])
        self.assertEqual(drift_issues.parse_issue_body(created["body"]), fresh)
        self.assertEqual(
            by_title[f"[upstream-superseded] {open_one.dedupe_key}"]["state"], "OPEN"
        )
        self.assertIn(
            "dedupe-key",
            by_title[f"[upstream-superseded] {open_one.dedupe_key}"]["body"],
        )
        self.assertEqual(
            by_title[f"[upstream-superseded] {closed.dedupe_key}"]["state"], "CLOSED"
        )
        self.assertNotIn(f"[upstream-new-repo] {refused.dedupe_key}", by_title)
        self.assertNotIn(
            "close", [call[1] for call in state["log"] if call[0] == "issue"]
        )
        self.assertEqual(
            sum(1 for call in state["log"] if call[:2] == ["issue", "create"]), 2
        )  # the refused write did not stop the fresh one


if __name__ == "__main__":
    unittest.main()
