"""Refresh-upstream tests: classification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from refresh_helpers import (
    NEW,
    OLD,
    classify,
    comparison,
    model_with,
    single_recipe_catalog,
)
from repo_root import ROOT

from recipe_tooling import vendored_upstream
from recipe_tooling.catalog_model import file_id
from recipe_tooling.qualification import documents
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import render as refresh_render


class ClassificationTests(unittest.TestCase):
    def test_docs_and_code_only_is_mechanical(self) -> None:
        cmp = comparison(
            files=("src/model.py", "docs/guide.md", ".github/ci.yml", "logo.png")
        )
        assert classify(cmp) == []

    def test_launch_script_config_readme_and_patches_are_not(self) -> None:
        for name in (
            "run.sh",
            "Dockerfile",
            "serve.py",
            "configs/a.yaml",
            "x.patch",
        ):
            with self.subTest(name=name):
                assert classify(comparison(files=(name,)))

    def test_downgrade_divergence_and_truncation_are_not(self) -> None:
        for cmp in (
            comparison("behind"),
            comparison("diverged"),
            comparison(truncated=True),
        ):
            assert classify(cmp)

    def test_patch_target_change_and_vendor_problems_are_not(self) -> None:
        assert classify(comparison(files=("lib/a.py",)), patched={"lib/a.py"})
        assert classify(comparison(), problems=["x differs"])
        assert classify(comparison(files=("lib/b.py",)), patched={"lib/a.py"}) == []

    def test_renamed_away_blocking_file_counts(self) -> None:
        cmp = refresh_model.Comparison(
            "ahead",
            [],
            [
                {
                    "filename": "old.txt",
                    "previous_filename": "run.sh",
                    "status": "renamed",
                }
            ],
            False,
            "u",
        )
        assert classify(cmp)

    def test_patch_targets_are_read_from_diff_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.patch").write_text("--- a/lib/x.py\n+++ b/lib/x.py\n@@\n")
            assert refresh_classification.patch_targets(Path(tmp)) == {"lib/x.py"}


class NothingRunsTests(unittest.TestCase):
    """A move that changes nothing a recipe runs is not a refresh (#1085, #848, #1282)."""

    TARGET = refresh_model.Target(
        "github", "example/recipe", NEW, "2026-09-30", None, None, "u"
    )

    def test_docs_benchmarks_tests_and_licences_do_not_run(self) -> None:
        for name in (
            "README.md",
            "VALIDATION_2026-10-08.md",
            "bench/requirements-sdk.txt",
            "bench/results/2026-10-08/run/result.json",
            "benchmarks/x.sh",
            "tests/test_a.py",
            "docs/guide.rst",
            "LICENSE",
            "NOTICE.txt",
            ".gitignore",
            "assets/chart.svg",
        ):
            with self.subTest(name=name):
                assert not refresh_model.runs(name)
        for name in ("src/model.py", "requirements.txt", "run.sh", "Makefile", "a.c"):
            with self.subTest(name=name):
                assert refresh_model.runs(name)

    def test_only_complete_docs_comparisons_change_nothing(self) -> None:
        docs = comparison(files=("README.md", "bench/results/a.json", "LICENSE"))
        assert refresh_model.nothing_runs(docs, frozenset())
        assert not refresh_model.nothing_runs(docs, {"README.md"})  # adapter copies it
        assert not refresh_model.nothing_runs(
            comparison(files=("README.md", "src/a.py")), set()
        )
        assert not refresh_model.nothing_runs(
            comparison(files=("README.md",), truncated=True), set()
        )

    def assess(
        self, comparison_, *, used=None, report=None, engine=None, document=None
    ):
        catalog = single_recipe_catalog(f"https://github.com/example/recipe/tree/{OLD}")
        if document is not None:
            catalog.recipes["recipe"] = document
        elif engine is not None:
            recipe = catalog.recipes["recipe"]
            recipe["runtime"] = {"engine": engine}
            recipe["topology"] = {"roles": []}
        assessor = refresh_assessment.Assessor(catalog, object(), None, {})  # pyright: ignore[reportArgumentType]
        assessor.target = lambda _provider, _repo: self.TARGET  # pyright: ignore[reportAttributeAccessIssue]
        patches = [
            patch(
                "recipe_tooling.refresh_upstream.items.compare_commits",
                return_value=comparison_,
            ),
            patch.object(
                refresh_assessment.Assessor, "_used_changes", return_value=used
            ),
        ]
        if report is not None:
            pin = refresh_catalog_view.AdapterPin(
                "example/recipe", OLD, Path("missing")
            )
            patches += [
                patch(
                    "recipe_tooling.refresh_upstream.catalog_view.adapter_pin",
                    return_value=pin,
                ),
                patch(
                    "recipe_tooling.refresh_upstream.classification.patch_targets",
                    return_value=set(),
                ),
                patch.object(
                    vendored_upstream,
                    "fetch_tree",
                    return_value=vendored_upstream.Tree({"README.md": "b" * 40}, True),
                ),
                patch.object(vendored_upstream, "check_adapter", return_value=report),
                patch.object(refresh_assessment.Assessor, "_retarget_adapter"),
                patch.object(
                    refresh_assessment.Assessor,
                    "_review_vendored",
                    return_value="changed",
                ),
            ]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)
        return assessor.assess("recipe")

    def test_whole_repository_move_of_docs_and_benchmarks_is_current(self) -> None:
        result = self.assess(
            comparison(files=("README.md", "bench/results/a.json", "AGENTS.md"))
        )
        assert not result.drifted
        assert not result.reasons

    def test_a_moved_vllm_document_meets_the_budget_envelope(self) -> None:
        moved = comparison(files=("src/engine.py",))
        with (
            patch("recipe_tooling.refresh_upstream.assessment.RecipeDefinition"),
            patch(
                "recipe_tooling.refresh_upstream.evidence.raise_role_peaks"
            ) as raised,
        ):
            self.assess(moved, engine="vllm")
            raised.assert_called_once()
            roles, _recipe, replaced, digest = raised.call_args.args
            assert roles == []
            assert len(digest) == 64  # the replaced document, in hand
            assert replaced is not None
        with patch(
            "recipe_tooling.refresh_upstream.evidence.raise_role_peaks"
        ) as raised:
            self.assess(moved, engine="sglang")
            raised.assert_not_called()
        with (
            patch("recipe_tooling.refresh_upstream.assessment.RecipeDefinition"),
            patch(
                "recipe_tooling.refresh_upstream.evidence.raise_role_peaks",
                side_effect=ValueError("bad"),
            ),
        ):
            result = self.assess(moved, engine="vllm")
        assert not result.mechanical
        assert "memory envelope" in " ".join(result.reasons)

    def body(self, result) -> str:
        assert result.mechanical
        return refresh_render.pr_body(result)

    def test_body_of_a_non_vllm_move_claims_no_peaks(self) -> None:
        body = self.body(
            self.assess(comparison(files=("src/engine.py",)), engine="sglang")
        )
        assert "## Principles" in body
        assert "peaks" not in body
        assert "re-checks the new digest" in body
        assert "The sweep proves the recipe runs." in body

    def test_body_of_an_unmark_only_drift_claims_no_move_and_no_new_digest(
        self,
    ) -> None:
        item = refresh_model.Item(
            "source", "github", "example/recipe", OLD, self.TARGET, unmark=True
        )
        found = refresh_model.Assessment("recipe", items=[item])

        def stub(_self, _item, _recipe, _new, _result, edits):
            return edits.update({"adapters/x/vendored-upstream.json": b"{}"})

        with (
            patch(
                "recipe_tooling.refresh_upstream.items.collect_items",
                return_value=found,
            ),
            patch.object(refresh_assessment.Assessor, "_mark", stub),
        ):
            result = self.assess(comparison(files=("src/engine.py",)))
        body = self.body(result)
        assert "## Principles" in body
        assert "pins move" not in body
        assert "new digest" not in body
        assert "sweep proves" not in body
        assert "peaks" not in body

    def test_body_of_a_poolless_vllm_move_states_the_raised_peaks(self) -> None:
        recipe = documents.recipe("d3-lite-vllm-single", ROOT).model_copy(deep=True)
        recipe.provenance.source_reference = (
            f"https://github.com/example/recipe/tree/{OLD}"
        )
        recipe.execution.build.context.path = "missing"
        for role in recipe.topology.roles:
            role.resources.memory.peak_bytes = 1 << 30
            role.resources.memory.load_peak_bytes = None
        document = recipe.model_dump(mode="json", exclude_none=True)
        result = self.assess(comparison(files=("src/engine.py",)), document=document)
        assert result.peaks_raised
        body = self.body(result)
        assert "raised to the vLLM budget envelope" in body

    def test_body_names_held_pins_and_does_not_claim_they_moved(self) -> None:
        result = refresh_model.Assessment(
            "recipe",
            items=[
                refresh_model.Item("source", "github", "o/r", OLD, self.TARGET),
            ],
            declined=["image o/base"],
            edits={"recipes/recipe.json": b"x"},
        )
        body = refresh_render.pr_body(result)
        assert "pins that moved go" in body
        assert "image o/base" in body

    def test_whole_repository_code_move_is_still_a_refresh(self) -> None:
        result = self.assess(comparison(files=("README.md", "src/engine.py")))
        assert result.mechanical

    def test_stale_vendored_readme_copy_alone_is_current(self) -> None:
        message = "adapters/x: upstream-README.md: differs from upstream README.md at f"
        report = vendored_upstream.Report("adapters/x")
        report.problems.append(message)
        report.stale.append((message, "upstream-README.md", "README.md"))
        result = self.assess(comparison(files=("README.md",)), used=[], report=report)
        assert not result.drifted

    def test_huggingface_source_reference_alone_is_current(self) -> None:
        catalog = single_recipe_catalog(
            f"https://huggingface.co/example/model/tree/{OLD}"
        )
        assessor = refresh_assessment.Assessor(catalog, object(), None, {})  # pyright: ignore[reportArgumentType]
        assessor.target = lambda _provider, _repo: refresh_model.Target(  # pyright: ignore[reportAttributeAccessIssue]
            "huggingface", "example/model", NEW, "2026-09-30", None, None, "u"
        )
        with patch(
            "recipe_tooling.refresh_upstream.pinning.follow_huggingface_branch",
            return_value=None,
        ):
            result = assessor.assess("recipe")
        assert not result.drifted
        assert not result.edits


class ReadmeAndScopeTests(unittest.TestCase):
    def test_readme_is_not_a_blocking_file(self) -> None:
        assert not refresh_model.is_blocking("README.md")
        assert not refresh_model.is_blocking("models/readme.rst")
        assert refresh_model.is_blocking("run.sh")

    def test_ui_mate_unrelated_upstream_changes_are_mechanical(self) -> None:
        # #491: 3 commits, 300+ files (truncated), none of them used by the recipe.
        cmp = refresh_model.Comparison(
            "ahead",
            [],
            [
                {"filename": "README.md", "status": "modified"},
                {
                    "filename": "osworker_bench/CUA-Gym-Hub/deploy-all.sh",
                    "status": "added",
                },
            ],
            True,
            "u",
        )
        assert classify(cmp, used_changes=[]) == []
        # Without a verified used set the same change is judged file by file.
        assert classify(cmp)

    def test_used_file_change_is_a_reason(self) -> None:
        assert classify(comparison(), used_changes=["agents/ui_mate_agent.py"])

    def test_diverged_is_not_a_reason_once_used_files_are_verified(self) -> None:
        assert classify(comparison("diverged"), used_changes=[]) == []
        assert classify(comparison("diverged"))
        assert classify(comparison("diverged"), used_changes=[], problems=["x differs"])

    def test_licence_removal_is_informational(self) -> None:
        cmp = refresh_model.Comparison(
            "ahead", [], [{"filename": "LICENSE", "status": "removed"}], False, "u"
        )
        item = refresh_model.Item(
            "source",
            "github",
            "o/r",
            OLD,
            refresh_model.Target("github", "o/r", NEW, "2026-09-30", None, None, "u"),
        )
        assert classify(cmp, used_changes=[]) == []
        assert "never gate" in refresh_evidence.github_evidence(item, cmp, set())


class AdapterUsageTests(unittest.TestCase):
    def usage(self, files: dict[str, str]):
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in files.items():
                (Path(tmp) / name).write_text(text)
            return refresh_catalog_view.adapter_usage(Path(tmp), "o/Repo", OLD)

    def test_wrapper_without_references_uses_nothing(self) -> None:
        assert self.usage({"Dockerfile": "FROM x\nCOPY a /a\n"}) == (set(), False)

    def test_explicit_urls_name_files(self) -> None:
        text = (
            "curl https://raw.githubusercontent.com/o/repo/main/cfg/a.yaml\n"
            "# https://github.com/o/Repo/blob/v1/docs/b.md\n"
        )
        assert self.usage({"setup.sh": text}) == ({"cfg/a.yaml", "docs/b.md"}, False)

    def test_clone_or_commit_named_archive_uses_the_whole_repository(self) -> None:
        clone = "RUN git clone https://github.com/o/Repo /src\n"
        assert self.usage({"Dockerfile": clone})[1]
        assert self.usage({f"tensorfold-{OLD}.tar.gz": ""})[1]
        codeload = f"ADD https://codeload.github.com/o/Repo/tar.gz/{OLD} /src.tgz\n"
        assert self.usage({"Dockerfile": codeload})[1]

    def test_another_repositorys_download_does_not_make_this_one_whole(self) -> None:
        dockerfile = (
            'LABEL org.opencontainers.image.source="https://github.com/o/Repo"\n'
            "ADD https://codeload.github.com/other/Tensor/tar.gz/abc /t.tgz\n"
            "RUN git clone https://github.com/comfy/Comfy /c\n"
            "RUN pip install https://github.com/other/lib/archive/v1.zip\n"
        )
        assert self.usage({"Dockerfile": dockerfile}) == (set(), False)


class ArchiveLabelTests(unittest.TestCase):
    OLD_DIGEST = "1" * 64
    NEW_DIGEST = "2" * 64
    OTHER = "3" * 64

    def item(self):
        target = refresh_model.Target(
            "github", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        return refresh_model.Item("source", "github", "o/r", OLD, target)

    def digest(self, _repo: str, commit: str) -> str:
        return self.OLD_DIGEST if commit == OLD else self.NEW_DIGEST

    def test_archive_label_of_the_pin_moves_with_it(self) -> None:
        text = (
            f'LABEL a.archive-sha256="{self.OLD_DIGEST}" \\\n'
            f'      b.mia-source-archive-sha256="{self.OTHER}"\n'
        )
        with patch(
            "recipe_tooling.refresh_upstream.classification.archive_sha256", self.digest
        ):
            moved, changed = refresh_classification.retarget_archive_labels(
                text, self.item()
            )
        assert changed
        assert f'a.archive-sha256="{self.NEW_DIGEST}"' in moved
        assert f'mia-source-archive-sha256="{self.OTHER}"' in moved

    def test_the_download_check_moves_with_the_label(self) -> None:
        # The LTX adapters state the digest twice; moving only the label left the
        # sha256sum check on the old archive and the build would fail.
        text = (
            f'LABEL a.archive-sha256="{self.OLD_DIGEST}"\n'
            f"RUN curl -o /tmp/s.tar.gz https://codeload.github.com/o/r/tar.gz/{OLD} \\\n"
            f"    && printf '%s  %s\\n' {self.OLD_DIGEST} /tmp/s.tar.gz | sha256sum -c\n"
            f"    && printf '%s  %s\\n' {self.OTHER} /tmp/wheel.whl | sha256sum -c\n"
        )
        with patch(
            "recipe_tooling.refresh_upstream.classification.archive_sha256", self.digest
        ):
            moved, changed = refresh_classification.retarget_archive_labels(
                text, self.item()
            )
        assert changed
        assert self.OLD_DIGEST not in moved
        assert moved.count(self.NEW_DIGEST) == 2
        assert self.OTHER in moved

    def test_unrelated_archive_labels_are_left_alone(self) -> None:
        text = f'LABEL b.archive-sha256="{self.OTHER}"\n'
        with patch(
            "recipe_tooling.refresh_upstream.classification.archive_sha256", self.digest
        ):
            assert refresh_classification.retarget_archive_labels(
                text, self.item()
            ) == (text, False)

    def test_fetch_failure_is_a_review_reason_not_a_stale_label(self) -> None:
        def broken(_repo: str, _commit: str) -> str:
            raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "adapter"
            directory.mkdir()
            (directory / "Dockerfile").write_text(
                f'LABEL x.archive-sha256="{self.OLD_DIGEST}" y="{OLD}"\n'
            )
            assessor = refresh_assessment.Assessor(
                refresh_catalog_view.Catalog(root),
                object(),  # pyright: ignore[reportArgumentType]
                None,
                {"recipes": []},
            )
            pin = refresh_catalog_view.AdapterPin("o/r", OLD, directory)
            result = refresh_model.Assessment("recipe")
            edits: dict[str, Any] = {}
            with patch(
                "recipe_tooling.refresh_upstream.classification.archive_sha256", broken
            ):
                assessor._retarget_adapter(pin, self.item(), edits, result)
            assert result.reasons
            with patch(
                "recipe_tooling.refresh_upstream.classification.archive_sha256",
                self.digest,
            ):
                result = refresh_model.Assessment("recipe")
                assessor._retarget_adapter(pin, self.item(), edits, result)
            new = edits["adapter/Dockerfile"].decode()
            assert self.NEW_DIGEST in new
            assert NEW in new
            assert not result.reasons


class ReleaseAssetTests(unittest.TestCase):
    """An adapter installing a creator's release asset moves its version and sha with the pin.

    TensorFold 1.x adapters fetch ``releases/download/v<version>/<asset>`` pinned by
    sha256; a mechanical move that rewrote only the commit kept installing the old
    binary while the recipe claimed the new release.
    """

    REPO = "ashhart/TensorFold"
    OLD_SHA256 = "1" * 64
    NEW_SHA256 = "2" * 64
    OTHER = "3" * 64  # another download's digest, never touched
    TARGET = refresh_model.Target(
        "github", REPO, NEW, "2026-11-07", "v1.2.4", "1.2.4", "u"
    )

    def asset(self, version: str) -> str:
        return f"tensorfold-{version}-linux-aarch64.tar.gz"

    def dockerfile(self) -> str:
        return (
            "# TensorFold v1.2.3 (Apache-2.0): the native engine as released.\n"
            "FROM base@sha256:" + "4" * 64 + "\n"
            f'LABEL org.opencontainers.image.source="https://github.com/{self.REPO}" \\\n'
            f'      org.opencontainers.image.revision="{OLD}"\n'
            "RUN curl -fsSL -o /tmp/tf.tar.gz "
            f"https://github.com/{self.REPO}/releases/download/v1.2.3/{self.asset('1.2.3')} \\\n"
            f'    && echo "{self.OLD_SHA256}  /tmp/tf.tar.gz" | sha256sum -c - \\\n'
            '    && test "$(cat /opt/tensorfold/VERSION)" = 1.2.3\n'
            "RUN curl -fsSL -o /tmp/j.tar.gz "
            "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/j.tar.gz \\\n"
            f'    && echo "{self.OTHER}  /tmp/j.tar.gz" | sha256sum -c -\n'
        )

    def release(self, tag: str, digest: str, *, with_sha256: bool = True):
        version = tag.removeprefix("v")
        name = self.asset(version)
        assets = [
            {
                "name": name,
                "digest": f"sha256:{digest}",
                "browser_download_url": f"https://dl/{tag}/{name}",
            }
        ]
        if with_sha256:
            assets.append(
                {
                    "name": f"{name}.sha256",
                    "browser_download_url": f"https://dl/{tag}/{name}.sha256",
                }
            )
        return {"tag_name": tag, "assets": assets}

    def assess(self, *, new_has_sha256: bool = True):
        test = self
        releases = {
            "v1.2.3": self.release("v1.2.3", self.OLD_SHA256),
            "v1.2.4": self.release(
                "v1.2.4", self.NEW_SHA256, with_sha256=new_has_sha256
            ),
        }
        published = {
            f"https://dl/v1.2.3/{self.asset('1.2.3')}.sha256": self.OLD_SHA256,
            f"https://dl/v1.2.4/{self.asset('1.2.4')}.sha256": self.NEW_SHA256,
        }

        class FakeHttp:
            def github(self, path):
                prefix = f"repos/{test.REPO}/releases/tags/"
                if path.startswith(prefix):
                    return releases[path.removeprefix(prefix)]
                raise refresh_http.NotFound(path)

        def download(url: str) -> str:
            name = url.rsplit("/", 1)[1].removesuffix(".sha256")
            return f"{published[url]}  {name}\n"

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        adapter = root / "adapters/tf"
        adapter.mkdir(parents=True)
        (adapter / "Dockerfile").write_text(self.dockerfile())
        (adapter / "tensorfold-serve").write_text("#!/bin/sh\n")
        (adapter / "vendored-upstream.json").write_text(
            json.dumps(
                {
                    "ours": ["tensorfold-serve"],
                    "source": {"commit": OLD, "line": "1", "repo": self.REPO},
                }
            )
        )
        catalog = refresh_catalog_view.Catalog(
            root,
            recipes={
                "recipe": {
                    "identity": {"slug": "recipe"},
                    "provenance": {
                        "attribution": ["Ash Hart (TensorFold)"],
                        "source_reference": f"https://github.com/{self.REPO}/tree/{OLD}",
                    },
                    "release": {"version": "1.2.3", "released_at": "2026-11-05"},
                    "models": [],
                    "execution": {"build": {"context": {"path": "adapters/tf"}}},
                }
            },
        )
        assessor = refresh_assessment.Assessor(catalog, FakeHttp(), None, {})  # pyright: ignore[reportArgumentType]
        assessor.target = lambda _provider, _repo: self.TARGET  # pyright: ignore[reportAttributeAccessIssue]
        zig_only = comparison(files=("zig/src/families/qwen3_5/cuda_engine.zig",))
        for active in (
            patch(
                "recipe_tooling.refresh_upstream.items.compare_commits",
                return_value=zig_only,
            ),
            patch(
                "recipe_tooling.refresh_upstream.pinning.resolve_github_line",
                return_value=self.TARGET,
            ),
            patch(
                "recipe_tooling.refresh_upstream.classification.download_text",
                side_effect=download,
                create=True,
            ),
            patch.object(
                vendored_upstream,
                "fetch_tree",
                return_value=vendored_upstream.Tree({}, True),
            ),
            patch.object(
                vendored_upstream,
                "check_adapter",
                return_value=vendored_upstream.Report("adapters/tf"),
            ),
        ):
            active.start()
            self.addCleanup(active.stop)
        return assessor.assess("recipe")

    def test_a_code_only_release_re_pins_url_version_and_sha(self) -> None:
        result = self.assess()
        assert result.mechanical, result.reasons
        moved = result.edits["adapters/tf/Dockerfile"].decode()  # pyright: ignore[reportOptionalMemberAccess]
        assert (
            f"https://github.com/{self.REPO}/releases/download/v1.2.4/{self.asset('1.2.4')}"
            in moved
        )
        assert f'echo "{self.NEW_SHA256}  /tmp/tf.tar.gz"' in moved
        assert 'test "$(cat /opt/tensorfold/VERSION)" = 1.2.4' in moved
        assert "# TensorFold v1.2.4 (Apache-2.0)" in moved
        assert f'revision="{NEW}"' in moved
        assert "1.2.3" not in moved
        assert self.OLD_SHA256 not in moved
        # Another project's download and digest are not this release's.
        assert "libjpeg-turbo/releases/download/3.1.4.1/j.tar.gz" in moved
        assert self.OTHER in moved
        assert "recipes/recipe.json" in result.edits
        assert result.version == ("1.2.4", "2026-11-07")

    def test_a_release_without_its_sha256_file_is_a_review(self) -> None:
        result = self.assess(new_has_sha256=False)
        assert not result.mechanical
        assert any(
            "release asset pinned by version and sha; re-pin by hand" in reason
            for reason in result.reasons
        ), result.reasons


class EnvelopeTests(unittest.TestCase):
    def retarget(
        self, artifact_bytes: int, old_size: int, new_size: int, prefix: str = "sel-"
    ):
        old = model_with(("model.safetensors",))
        selection = prefix + old["files"][0]["id"]
        old["files"][0]["size_bytes"] = old_size
        fresh = json.loads(json.dumps(old))
        fresh["identity"]["slug"] = "demo-" + NEW[:8]
        fresh["files"][0]["sha256"] = "d" * 64
        fresh["files"][0]["size_bytes"] = new_size
        fresh["files"][0]["id"] = file_id("model.safetensors", "d" * 64)
        recipe = {
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": selection,
                            "file_id": old["files"][0]["id"],
                        }
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": artifact_bytes}}}]
            },
        }
        result = refresh_model.Assessment("recipe")
        assessor = refresh_assessment.Assessor(
            refresh_catalog_view.Catalog(Path()),
            object(),  # pyright: ignore[reportArgumentType]
            None,
            {"recipes": []},
        )
        assessor._retarget_model(recipe, old, fresh, result)
        self.selected = recipe["models"][0]["files"][0]
        return recipe["topology"]["roles"][0]["resources"]["disk"], result

    def test_a_recipe_mounting_every_file_mounts_the_new_ones(self) -> None:
        # #531: the official checkpoint gained chat_template.jinja; an SGLang recipe
        # that mounts the whole Model must mount it too, and a dropped README leaves.
        old = model_with(("model.safetensors", "README.md"))
        fresh = json.loads(json.dumps(old))
        fresh["identity"]["slug"] = "demo-" + NEW[:8]
        fresh["files"] = [
            dict(old["files"][0]),
            {
                "id": file_id("chat_template.jinja", "e" * 64),
                "path": "chat_template.jinja",
                "roles": ["tokenizer"],
                "sha256": "e" * 64,
                "size_bytes": 7,
            },
        ]
        mount = {"target": "/models"}
        recipe = {
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": "primary-" + f["id"],
                            "file_id": f["id"],
                            "mount": mount,
                            "roles": ["entrypoint"],
                        }
                        for f in old["files"]
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": 1_000_000}}}]
            },
        }
        result = refresh_model.Assessment("recipe")
        assessor = refresh_assessment.Assessor(
            refresh_catalog_view.Catalog(Path()),
            object(),  # pyright: ignore[reportArgumentType]
            None,
            {"recipes": []},
        )
        assessor._retarget_model(recipe, old, fresh, result)
        assert not result.reasons
        selected = recipe["models"][0]["files"]
        assert sorted(s["file_id"] for s in selected) == sorted(
            f["id"] for f in fresh["files"]
        )
        assert all(s["mount"] == mount for s in selected)
        disk = recipe["topology"]["roles"][0]["resources"]["disk"]
        assert disk["artifact_bytes"] == 1000000 - 10 + 7

    def test_selected_file_id_follows_the_new_file_within_the_contract_bound(
        self,
    ) -> None:
        # #1290: a long model-reference prefix plus the new file id overflowed 64.
        prefix = "dependency-nemotron-3-5-lightning-30b-a3b-nvfp4-dspark-"
        _disk, result = self.retarget(10_000_000, 10_000_000, 10_000_000, prefix)
        assert not result.reasons
        new_id = file_id("model.safetensors", "d" * 64)
        assert self.selected["file_id"] == new_id
        assert len(self.selected["id"]) <= 64
        assert self.selected["id"].startswith(prefix + new_id[:5])
        _disk, _result = self.retarget(10_000_000, 10_000_000, 10_000_000)
        assert self.selected["id"] == "sel-" + new_id

    def test_small_size_change_shifts_a_derived_envelope(self) -> None:
        disk, result = self.retarget(10_000_000, 10_000_000, 9_999_600)
        assert disk["artifact_bytes"] == 9999600
        assert not result.reasons

    def test_small_size_change_keeps_a_declared_allowance(self) -> None:
        disk, result = self.retarget(10_500_000, 10_000_000, 9_999_600)
        assert disk["artifact_bytes"] == 10499600
        assert not result.reasons

    def test_material_size_change_needs_review(self) -> None:
        _disk, result = self.retarget(10_000_000, 10_000_000, 10_200_000)
        assert result.reasons
        assert "more than 1%" in result.reasons[0]
