"""Refresh-upstream tests: assessment."""

from __future__ import annotations

import copy
import http.server
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

import documents
import pytest
from catalog_documents import catalog_models
from refresh_helpers import NEW, OLD, OLD_YAML, comparison, model_with
from repo_root import ROOT
from warning_asserts import assert_not_warns, assert_warns

from recipe_tooling import catalog_hf_model, upstream_http, vendored_upstream
from recipe_tooling.catalog_model import roles_for
from recipe_tooling.kit_pins import consumers as kit_consumers
from recipe_tooling.qualification import documents as qualification_documents
from recipe_tooling.qualification.memory_need.evidence import load_inputs
from recipe_tooling.qualification.memory_need.warnings import (
    memory_estimate_warnings,
)
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import items as refresh_items
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.upstream_drift import heads as drift_heads
from vonk_forge_contracts import RecipeDefinition


class EmbeddedSourceReviewTests(unittest.TestCase):
    def test_unrelated_default_head_advance_does_not_block_safe_refresh(
        self,
    ) -> None:
        class FakeProvider:
            def json(self, url: str, _provider: str):
                if url.endswith("/repos/example/dependency"):
                    return {"default_branch": "main"}
                if url.endswith("/commits/main"):
                    return {
                        "sha": NEW,
                        "html_url": "https://github.com/example/dependency/commit/new",
                        "commit": {"committer": {"date": "2026-09-30T00:00:00Z"}},
                    }
                raise AssertionError(url)

        embedded = {
            "coverage": {},
            "recipes": [
                {
                    "recipe_id": "recipe",
                    "inputs": [
                        {
                            "kind": "download-source",
                            "repository": "example/dependency",
                            "revision": OLD,
                        }
                    ],
                }
            ],
        }
        drift_heads.observe_embedded_heads(embedded, FakeProvider())
        catalog = refresh_catalog_view.Catalog(
            Path(tempfile.gettempdir()),
            recipes={
                "recipe": {
                    "identity": {"slug": "recipe"},
                    "provenance": {
                        "attribution": ["Example"],
                        "source_reference": f"https://github.com/example/recipe/tree/{OLD}",
                    },
                    "release": {
                        "version": "1.0.0",
                        "released_at": "2026-01-01",
                    },
                    "models": [],
                    "execution": {"build": {"context": {"path": "missing"}}},
                }
            },
        )
        assessor = refresh_assessment.Assessor(catalog, object(), None, embedded)  # pyright: ignore[reportArgumentType]
        target = refresh_model.Target(
            "github",
            "example/recipe",
            NEW,
            "2026-09-30",
            "v2.0.0",
            "2.0.0",
            "https://github.com/example/recipe/commit/new",
        )
        assessor.target = lambda _provider, _repo: target  # pyright: ignore[reportAttributeAccessIssue]
        # The recipe uses the whole repository and code changed, so the move is a
        # refresh (a move that changes nothing that runs is not one; see
        # NothingRunsTests and test_published_image_follow).
        with (
            patch(
                "recipe_tooling.refresh_upstream.items.compare_commits",
                return_value=refresh_model.Comparison(
                    "ahead",
                    [],
                    [{"filename": "src/model.py", "status": "modified"}],
                    False,
                    "https://github.com/example/recipe/compare",
                ),
            ),
            patch.object(
                refresh_assessment.Assessor, "_used_changes", return_value=None
            ),
        ):
            result = assessor.assess("recipe")

        assert embedded["recipes"][0]["inputs"][0]["candidate_status"] == "advanced"
        assert not result.reasons
        assert result.mechanical
        assert "review evidence, not a configured upgrade channel" in result.evidence[0]
        assert NEW in result.edits["recipes/recipe.json"].decode()  # pyright: ignore[reportOptionalMemberAccess]


class ImageHeaderMoveTests(unittest.TestCase):
    OLD = "sha256:" + "a" * 64
    NEW = "sha256:" + "b" * 64
    UPSTREAM = "# Upstream recipes start it with `vllm serve`; the wrapper is the entrypoint.\n"

    def move(self, header: str, result: Any = None) -> str:
        result = result or refresh_model.Assessment("recipe")
        text = (
            f"FROM docker.io/eugr/spark-vllm@{self.OLD}\n\n"
            'LABEL org.opencontainers.image.source="https://example.test/r" \\\n'
            '      org.opencontainers.image.revision="42f62e34"\n\n'
            + header
            + "COPY vllm-wrapper.py /opt/vonk/bin/vllm\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Dockerfile").write_text(text)
            edits: dict[str, bytes | None] = {}
            refresh_assessment.Assessor(
                refresh_catalog_view.Catalog(root),
                object(),  # pyright: ignore[reportArgumentType]
                None,
                {},
            )._move_image(
                root,
                "docker.io/eugr/spark-vllm",
                self.NEW,
                lambda written: f"docker.io/eugr/spark-vllm@{self.NEW}",
                {"execution": {"build": {}}},
                edits,
                published=True,
                result=result,
            )
        written = edits["Dockerfile"]
        assert written is not None
        return written.decode()

    def test_published_move_replaces_the_versioned_header_prose(self) -> None:
        # Verbatim from adapters/eugr/qwen3-6-35b-a3b-fp8/Dockerfile.
        header = (
            "# eugr's published vllm-node image (nightly-20260929). Its /workspace/build-metadata.yaml records\n"
            "# build_script_commit 42f62e34 (the revision above), vLLM af5b4857e and FlashInfer 4a638133, built\n"
            "# from the repository's default Dockerfile (upstream's local `vllm-node` build) with the stock\n"
            "# build arguments.\n" + self.UPSTREAM
        )
        moved = self.move(header)
        assert (
            "# The published docker.io/eugr/spark-vllm image, built from the revision "
            "above; the digest is what runs.\n" + self.UPSTREAM in moved
        )
        assert "nightly" not in moved
        assert "42f62e34 (" not in moved

    def test_header_without_the_upstream_line_is_kept(self) -> None:
        header = "# build arguments. Upstream recipes start it with `vllm serve`; the wrapper.\n"
        assert header in self.move(header)

    def test_header_the_note_cannot_replace_is_named_for_review(self) -> None:
        # Verbatim from adapters/eugr/qwen3-6-35b-a3b-nvfp4/Dockerfile.
        mid_line = (
            "# eugr's published vllm-node image (nightly-20260929). Its /workspace/build-metadata.yaml records\n"
            "# build_script_commit 42f62e34 (the revision above), vLLM af5b4857e and FlashInfer 4a638133, built\n"
            "# from the repository's default Dockerfile (upstream's local `vllm-node` build) with the stock\n"
            "# build arguments. Upstream recipes start it with `vllm serve`; the wrapper is the Vonk entrypoint.\n"
        )
        for header, reviewed in (
            (mid_line, True),
            ("# Old prose.\n" + self.UPSTREAM, False),
        ):
            result = refresh_model.Assessment("recipe")
            self.move(header, result)
            assert (
                "Review the Dockerfile's header comment.\n" in result.evidence
            ) == reviewed


class ImageLabelClaimTests(unittest.TestCase):
    """The revision label of an adapter built FROM a digest-pinned image is a claim about
    that image, never a git pin: only a new image digest (which records its own build
    commit) may move it."""

    DOCKERFILE = (
        "FROM ghcr.io/ggml-org/llama.cpp@sha256:" + "e" * 64 + "\n"
        'LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \\\n'
        f'      org.opencontainers.image.revision="{OLD}"\n'
        "COPY verify-runtime.sh /opt/vonk/verify-runtime.sh\n"
    )

    def collect(self, directory: Path, asked: list[str]):
        recipe = {
            "identity": {"slug": "recipe"},
            "release": {"version": "1.0.0", "released_at": "2026-01-01"},
            "models": [],
            "execution": {"build": {"context": {"path": "adapter"}}},
        }
        catalog = refresh_catalog_view.Catalog(directory, recipes={"recipe": recipe})

        def targets(provider: str, repo: str):
            asked.append(repo)
            return refresh_model.Target(
                provider, repo, NEW, "2026-11-08", "b11200", "b11200", "u"
            )

        with patch(
            "recipe_tooling.refresh_upstream.items.compare_commits",
            return_value=refresh_model.Comparison("ahead", [], [], False, "u"),
        ):
            return refresh_items.collect_items(catalog, "recipe", object(), targets)  # pyright: ignore[reportArgumentType]

    def adapter(self, root: Path, dockerfile: str, manifest: bool = False) -> None:
        directory = root / "adapter"
        directory.mkdir()
        (directory / "Dockerfile").write_text(dockerfile)
        (directory / "verify-runtime.sh").write_text("#!/bin/sh\n")
        if manifest:
            (directory / "vendored-upstream.json").write_text(
                json.dumps({"ours": ["verify-runtime.sh"]})
            )

    def test_newer_release_with_unchanged_digest_proposes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.adapter(root, self.DOCKERFILE)
            asked: list[str] = []
            result = self.collect(root, asked)
            assert result.items == []
            assert not result.drifted
            assert asked == []
            assert "describes the digest-pinned image" in result.evidence[0]

    def test_pin_the_build_uses_is_still_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            used = self.DOCKERFILE + f"RUN git checkout {OLD}\n"
            for name, dockerfile, manifest in (
                ("commit used by the build", used, False),
                ("manifest declares the pin", self.DOCKERFILE, True),
                (
                    "not an image of the repository",
                    self.DOCKERFILE.replace("ggml-org/llama.cpp@", "nvcr.io/cuda@"),
                    False,
                ),
                (
                    "image not pinned by digest",
                    self.DOCKERFILE.replace("@sha256:" + "e" * 64, ":latest"),
                    False,
                ),
            ):
                with self.subTest(name):
                    shutil.rmtree(root / "adapter", ignore_errors=True)
                    self.adapter(root, dockerfile, manifest)
                    result = self.collect(root, [])
                    assert [i.kind for i in result.items] == ["adapter"]

    def test_twin_label_and_notice_do_not_make_the_pin_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.adapter(
                root,
                self.DOCKERFILE.replace(
                    f'revision="{OLD}"',
                    f'revision="{OLD}" \\\n      io.vonk.vllm.build-commit="{OLD}"',
                ),
            )
            (root / "adapter" / "NOTICE").write_text(
                f"Runtime https://github.com/ggml-org/llama.cpp/tree/{OLD} (MIT).\n"
            )
            asked: list[str] = []
            result = self.collect(root, asked)
            assert result.items == []
            assert asked == []

    def test_no_catalog_adapter_of_the_shape_moves_with_a_release(self) -> None:
        # Selected by shape alone, independently of image_label_claim.
        catalog = refresh_catalog_view.Catalog.load(ROOT)
        covered = 0
        for recipe_id, recipe in catalog.recipes.items():
            directory = ROOT / recipe["execution"]["build"]["context"]["path"]
            pin = refresh_catalog_view.adapter_pin(ROOT, recipe)
            if (
                pin is None
                or pin.derived
                or (directory / "vendored-upstream.json").is_file()
            ):
                continue
            name = pin.repo.rsplit("/", 1)[-1].lower()
            images = [
                m["image"]
                for m in kit_consumers.FROM_LINE.finditer(
                    (directory / "Dockerfile").read_text(encoding="utf-8")
                )
            ]
            if not any(
                kit_consumers.split_image(i)[2] and name in i.lower() for i in images
            ):
                continue
            if refresh_catalog_view.adapter_usage(directory, pin.repo, pin.commit) != (
                set(),
                False,
            ):
                continue
            source = refresh_catalog_view.source_pin(recipe)
            if source is not None and source[1].lower() == pin.repo.lower():
                continue  # the source path owns the pin (a separate class)
            covered += 1
            with self.subTest(recipe_id):
                alone = {
                    **recipe,
                    "models": [],
                    "provenance": {
                        k: v
                        for k, v in recipe["provenance"].items()
                        if k != "source_reference"
                    },
                }
                asked: list[str] = []

                def targets(provider: str, repo: str, asked=asked):
                    asked.append(repo)
                    return refresh_model.Target(
                        provider, repo, NEW, "2026-11-08", None, None, "u"
                    )

                result = refresh_items.collect_items(
                    refresh_catalog_view.Catalog(ROOT, recipes={recipe_id: alone}),
                    recipe_id,
                    object(),  # pyright: ignore[reportArgumentType]
                    targets,
                )
                assert result.items == []
                assert asked == []
        assert covered > 0


class VendoredCreatorMoveTests(unittest.TestCase):
    """A vendored creator file that changed upstream is always a review: no byte of it
    is judged harmless, and an unchanged file moves with the pin without a rewrite."""

    def assess(
        self,
        new_yaml: str,
        *,
        old_yaml: str = OLD_YAML,
        dockerfile: str = "",
        ours=(),
        extra=None,
        engine=None,
        name: str = "recipes/r.yaml",
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        adapter = root / "adapters/x"
        (adapter / name).parent.mkdir(parents=True)
        (adapter / name).write_text(old_yaml)
        (adapter / "vllm-wrapper.py").write_text("print('wrapper')\n")
        (adapter / "Dockerfile").write_text(
            "FROM example/engine@sha256:" + "a" * 64 + "\nENTRYPOINT []\n" + dockerfile
        )
        for extra_name, text in (extra or {}).items():
            (adapter / extra_name).write_text(text)
        (adapter / "vendored-upstream.json").write_text(
            json.dumps(
                {
                    "ours": ["vllm-wrapper.py", *ours],
                    "source": {"repo": "example/recipe", "commit": OLD},
                }
            )
        )
        recipe = {
            "identity": {"slug": "recipe"},
            "provenance": {
                "attribution": ["Example"],
                "source_reference": f"https://github.com/example/recipe/tree/{OLD}",
            },
            "release": {"version": "1.0.0", "released_at": "2026-01-01"},
            "models": [],
            "execution": {"build": {"context": {"path": "adapters/x"}}},
            "runtime": {
                "arguments": [
                    {"name": "trust-remote-code", "setting": None, "value": True},
                    {"name": "served-model-name", "setting": None, "value": "m"},
                ],
                "environment": [{"name": "VLLM_MARLIN_USE_ATOMIC_ADD", "value": "1"}],
                "entrypoint": ["/opt/vonk/bin/vllm", "serve", "/models"],
            },
        }
        if engine is not None:
            recipe["runtime"]["engine"] = engine
            recipe["topology"] = {"roles": []}
        catalog = refresh_catalog_view.Catalog(root, recipes={"recipe": recipe})
        assessor = refresh_assessment.Assessor(catalog, object(), None, {})  # pyright: ignore[reportArgumentType]
        assessor.target = lambda _provider, _repo: refresh_model.Target(  # pyright: ignore[reportAttributeAccessIssue]
            "github", "example/recipe", NEW, "2026-10-13", None, None, "u"
        )
        data = new_yaml.encode()
        tree = vendored_upstream.Tree(
            {name: vendored_upstream.git_blob_sha(data)}, True
        )
        for active in (
            patch(
                "recipe_tooling.refresh_upstream.items.compare_commits",
                return_value=comparison(files=(name,)),
            ),
            patch.object(vendored_upstream, "fetch_tree", return_value=tree),
            patch(
                "recipe_tooling.refresh_upstream.classification.upstream_blob",
                return_value=data,
            ),
        ):
            active.start()
            self.addCleanup(active.stop)
        return assessor.assess("recipe")

    def test_real_unproven_dual_is_raised_and_then_passes_the_touched_check(
        self,
    ) -> None:

        slug = "muse-spark-open-nvfp4-vllm-eight"
        before = documents.recipe(slug)
        for role in before.topology.roles:
            role.resources.memory.peak_bytes = 100_000_000_000
        payload = before.model_dump(mode="json", by_alias=True)
        models, inputs = catalog_models(), load_inputs()
        assert_warns(
            memory_estimate_warnings(
                before, models, inputs, document_digest="", touched=True
            ),
            "vLLM budget plus overhead",
        )
        replaced = documents.recipe(slug)
        refresh_evidence.raise_role_peaks(
            payload["topology"]["roles"],
            before,
            replaced,
            qualification_documents.recipe_digest(slug),
        )
        after = RecipeDefinition.model_validate(payload)
        assert {r.resources.memory.peak_bytes for r in after.topology.roles} == {
            114930585191
        }
        assert_not_warns(
            memory_estimate_warnings(
                after, models, inputs, document_digest="", touched=True
            ),
            "vLLM budget plus overhead",
        )

    NOTE = "# creator note\n\n"

    def test_an_unchanged_vendored_file_moves_with_the_pin(self) -> None:
        # The eugr group that moves the image and pin only (no yaml change): nothing a
        # reviewer must judge, and the file itself is never rewritten.
        result = self.assess(OLD_YAML)
        assert result.reasons == []
        assert "adapters/x/recipes/r.yaml" not in result.edits

    def test_any_other_change_to_a_vendored_creator_file_needs_review(self) -> None:
        quoted = (
            "model: org/m\ncontainer: vllm-node\ndefaults:\n  gpu_memory_utilization: 0.8\n"
            'command: "vllm serve org/m --gpu-memory-utilization 0.8"\n'
        )
        inline = "mods: [mods/a]\nenv: {A: 1}\ncommand: x\n"
        shell = "#!/bin/sh\nexec vllm serve m --max-model-len 4096\n"
        env = "# profile\nGPU_MEMORY_UTILIZATION=0.8\nMODEL=org/m\n"
        yaml = "container: vllm-node\nbuild_args:\n  - --exp-b12x\nsolo_only: true\n"
        cases = {
            "utilization": (
                OLD_YAML,
                OLD_YAML.replace("8192", "4096"),
                "recipes/r.yaml",
            ),
            "model line": (
                "model: org/m\n" + OLD_YAML,
                "model: org/n\n" + OLD_YAML,
                "recipes/r.yaml",
            ),
            "quoted command": (quoted, quoted.replace("0.8", "0.9"), "recipes/r.yaml"),
            "inline mods": (
                inline,
                inline.replace("mods/a", "mods/b"),
                "recipes/r.yaml",
            ),
            "inline env": (inline, inline.replace("A: 1", "A: 2"), "recipes/r.yaml"),
            "shell flag": (shell, shell.replace("4096", "8192"), "launch/serve.sh"),
            "env file": (env, env.replace("0.8", "0.9"), "profiles/p.env"),
            "container": (
                yaml,
                yaml.replace("vllm-node", "vllm-node-b12x"),
                "recipes/r.yaml",
            ),
            "build args": (
                yaml,
                yaml.replace("--exp-b12x", "--other"),
                "recipes/r.yaml",
            ),
            "solo only": (yaml, yaml.replace("true", "false"), "recipes/r.yaml"),
            "shebang": (
                shell,
                shell.replace("/bin/sh", "/bin/bash"),
                "launch/serve.sh",
            ),
            "inline comment": (
                OLD_YAML,
                OLD_YAML.replace("8192", "8192  # raised"),
                "recipes/r.yaml",
            ),
            "unknown format": ("a b\n", "a  b\n", "launch/run.cfgx"),
            "json": ('{"a": 1}', '{"a": 1} ', "launch/profile.json"),
            "comment only": (OLD_YAML, self.NOTE + OLD_YAML, "recipes/r.yaml"),
            "trailing space": (
                shell,
                shell.replace("4096", "4096 "),
                "launch/serve.sh",
            ),
            "kernel macro in a raw string": (
                'SRC = r"""\n#define TILE 16\n"""\n',
                'SRC = r"""\n#define TILE 32\n"""\n',
                "mods/k/kernel.py",
            ),
            "continuation ends the command": (
                "vllm serve m \\\n  --a 1\n",
                "vllm serve m \\ \n  --a 1\n",
                "launch/serve.sh",
            ),
            "continuation in a yaml command": (
                "command: |\n  vllm serve m \\\n    --a 1\n",
                "command: |\n  vllm serve m \\ \n    --a 1\n",
                "recipes/r.yaml",
            ),
            "heredoc line": (
                "cat <<EOF\n# not a comment\nEOF\n",
                "cat <<EOF\nEOF\n",
                "launch/serve.sh",
            ),
        }
        for label, (old, new, name) in cases.items():
            with self.subTest(label):
                result = self.assess(new, old_yaml=old, name=name)
                assert not result.mechanical
                assert "changed" in " ".join(result.reasons)
                assert f"adapters/x/{name}" not in result.edits
                assert f"b/{name}" in " ".join(result.evidence)

    def test_a_patched_or_edited_file_stays_a_review(self) -> None:
        new_yaml = self.NOTE + OLD_YAML
        cases = {
            "sed": {"dockerfile": "RUN sed -i 's/a/b/' \\\n    /app/recipes/r.yaml\n"},
            "patch": {
                "ours": ("local.patch",),
                "extra": {
                    "local.patch": "--- a/recipes/r.yaml\n+++ b/recipes/r.yaml\n"
                },
            },
        }
        for name, case in cases.items():
            with self.subTest(name=name):
                result = self.assess(new_yaml, **case)
                assert not result.mechanical
                assert "not vendored verbatim" in " ".join(result.reasons)
                assert "adapters/x/recipes/r.yaml" not in result.edits

    def test_running_the_file_is_not_editing_it(self) -> None:
        result = self.assess(
            self.NOTE + OLD_YAML,
            dockerfile="RUN cd /app && bash ./recipes/r.yaml 2>/dev/null\n",
        )
        reasons = " ".join(result.reasons)
        assert "changed" in reasons
        assert "not vendored verbatim" not in reasons


class RewrittenHistoryTests(unittest.TestCase):
    class Http:
        def __init__(self, message: str) -> None:
            self.message = message

        def github(self, _path: str):
            raise refresh_http.NotFound("u", self.message)

    def test_no_common_ancestor_is_a_diverged_pin_move(self) -> None:
        compared = refresh_items.compare_commits(
            self.Http("No common ancestor between a and b."),  # pyright: ignore[reportArgumentType]
            "o/r",
            OLD,
            NEW,
        )
        assert compared.status == "diverged"
        assert compared.rewritten == NEW
        reasons = refresh_classification.classify_changes(
            compared, patched=set(), vendor_problems=[], used_changes=None
        )
        assert len(reasons) == 1, reasons
        assert "history was rewritten" in reasons[0]
        assert NEW in reasons[0]
        # Used files are compared by content: a rewrite alone is no reason.
        for used, expected in (([], 0), (["a.yaml"], 1)):
            reasons = refresh_classification.classify_changes(
                compared, patched=set(), vendor_problems=[], used_changes=used
            )
            assert len(reasons) == expected, reasons
            assert "history was rewritten" not in "".join(reasons)

    def test_any_other_missing_comparison_is_still_unassessable(self) -> None:
        with pytest.raises(refresh_http.NotFound):
            refresh_items.compare_commits(self.Http("Not Found"), "o/r", OLD, NEW)  # pyright: ignore[reportArgumentType]

    def test_the_provider_message_is_read_from_the_error_body(self) -> None:

        error = urllib.error.HTTPError(
            "u",
            404,
            "Not Found",
            Message(),
            io.BytesIO(b'{"message": "No common ancestor"}'),
        )
        assert refresh_http._error_message(error) == "No common ancestor"
        error = urllib.error.HTTPError(
            "u", 404, "Not Found", Message(), io.BytesIO(b"<html>")
        )
        assert refresh_http._error_message(error) == ""


class GatedRepositoryTests(unittest.TestCase):
    """#1517: a gated repository masks LFS digests and refuses non-LFS bytes (401).

    The trees are recorded from the Hugging Face API for black-forest-labs/flux-3-image-klein
    at 337337621b85 and 09561231f44e (four files kept): only README.md differs.
    """

    trees = json.loads((ROOT / "tests/fixtures/hf-gated-klein-trees.json").read_text())

    def setUp(self) -> None:
        # Nothing here may reach the network.
        blocker = patch(
            "socket.socket.connect", side_effect=AssertionError("network used")
        )
        blocker.start()
        self.addCleanup(blocker.stop)

    def old(self):
        return model_with(("README.md", "config.json", "model.safetensors"))

    def gated(self, old, after):
        return refresh_classification.gated_payload_identical(
            old, self.trees["before"], after
        )

    def test_same_blob_ids_and_sizes_are_unchanged_weights(self) -> None:
        assert self.gated(self.old(), self.trees["after"])

    def test_a_changed_weights_pointer_is_a_change(self) -> None:
        after = json.loads(json.dumps(self.trees["after"]))
        for entry in after:
            if entry["path"] == "model.safetensors":
                entry["oid"] = "0" * 40
        assert not self.gated(self.old(), after)

    def test_a_new_non_metadata_file_is_a_change(self) -> None:
        extra = {
            "type": "file",
            "path": "extra.safetensors",
            "oid": "1" * 40,
            "size": 5,
        }
        assert not self.gated(self.old(), [*self.trees["after"], extra])

    def test_roles_decide_exactly_as_without_access(self) -> None:
        # The document stores config.json as metadata, roles_for says configuration:
        # either saying payload makes it payload, with or without access.
        old = self.old()
        for entry in old["files"]:
            entry["sha256"] = "c" * 64
            if entry["path"] == "config.json":
                entry["roles"] = ["metadata"]
        for changed in (False, True):
            with self.subTest(config_changed=changed):
                after = copy.deepcopy(self.trees["after"])
                ungated = []
                for entry in after:
                    if changed and entry["path"] == "config.json":
                        entry["oid"] = "2" * 40
                    differs = changed and entry["path"] == "config.json"
                    ungated.append(
                        {
                            "path": entry["path"],
                            "roles": roles_for(entry["path"]),
                            "sha256": "d" * 64 if differs else "c" * 64,
                        }
                    )
                # The real ungated path curates the new snapshot the same way.
                before = {e["path"] for e in self.trees["before"]}
                curated = refresh_classification.curate(old, before, ungated)
                assert self.gated(
                    old, after
                ) == refresh_classification.payload_identical(old, curated)
                assert self.gated(old, after) == (not changed)

    def test_masked_lfs_digest_is_refused_not_catalogued(self) -> None:
        lfs_only = [e for e in self.trees["after"] if "lfs" in e]
        with pytest.raises(catalog_hf_model.AccessDenied):
            catalog_hf_model.inventory("o/r", NEW, lfs_only)

    def test_only_401_and_403_are_access_denied(self) -> None:
        entries = [{"type": "file", "path": "config.json", "size": 2}]
        for code, expected in (
            (401, "AccessDenied"),
            (403, "AccessDenied"),
            (503, "HTTPError"),
            (429, "HTTPError"),
        ):

            def fail(_url: str, code: int = code) -> bytes:
                error = HTTPError(_url, code, "x", Message(), None)
                error.close()
                raise error

            with (
                self.subTest(code=code),
                patch.object(catalog_hf_model, "_get_bytes", fail),
                pytest.raises(
                    catalog_hf_model.AccessDenied
                    if expected == "AccessDenied"
                    else HTTPError
                ),
            ):
                catalog_hf_model.inventory("o/r", NEW, entries)  # pyright: ignore[reportArgumentType]

    def test_the_token_is_sent_to_hugging_face_when_set(self) -> None:
        seen = {}

        class Opener:
            def open(self, request, timeout):
                seen.update(request.header_items())
                raise OSError("stop")

        for url, expected in (
            ("https://huggingface.co/api/models/o/r", "Bearer secret"),
            ("https://example.test/x", None),
        ):
            seen.clear()
            with (
                patch.dict(os.environ, {"HF_TOKEN": "secret"}),
                patch("urllib.request.build_opener", return_value=Opener()),
                pytest.raises(OSError),
            ):
                catalog_hf_model._get_json(url)
            assert seen.get("Authorization") == expected

    def test_a_gated_metadata_only_revision_is_not_a_move(self) -> None:
        old = self.old()
        old["source"]["repository"] = "https://huggingface.co/o/r"
        recipe = {"identity": {"slug": "recipe"}, "models": []}
        assessor = refresh_assessment.Assessor(
            refresh_catalog_view.Catalog(
                Path(),
                recipes={"recipe": recipe},
                models={old["identity"]["slug"]: old},
            ),
            object(),  # pyright: ignore[reportArgumentType]
            None,
            {},
        )
        item = refresh_model.Item(
            "model",
            "huggingface",
            "o/r",
            OLD,
            refresh_model.Target(
                "huggingface", "o/r", NEW, "2026-11-02", None, None, "u"
            ),
            old["identity"]["slug"],
        )
        result = refresh_model.Assessment("recipe", items=[item])
        trees = {OLD: self.trees["before"], NEW: self.trees["after"]}
        fetched: list[str] = []

        def tree(url: str) -> list[Any]:
            fetched.append(url)
            return trees[next(r for r in trees if f"/tree/{r}?" in url)]

        def denied(url: str) -> bytes:
            error = HTTPError(url, 401, "Unauthorized", Message(), None)
            error.close()
            raise error

        with (
            patch.object(catalog_hf_model, "_get_tree", side_effect=tree),
            patch.object(catalog_hf_model, "_get_bytes", denied),
        ):
            assessor._model(item, recipe, copy.deepcopy(recipe), result, {})
        assert result.reasons == []
        assert item.noop
        assert len(fetched) == len(set(fetched))  # each tree once


class UpstreamRedirectTests(unittest.TestCase):
    def test_a_redirect_to_another_origin_drops_the_credential(self) -> None:
        seen: dict[tuple[int, str], str | None] = {}

        def serve(redirects: dict[str, str]) -> http.server.HTTPServer:
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self) -> None:
                    seen[(int(self.headers["Host"].split(":")[1]), self.path)] = (
                        self.headers.get("Authorization")
                    )
                    target = redirects.get(self.path)
                    self.send_response(307 if target else 200)
                    if target:
                        self.send_header("Location", target)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")

                def log_message(self, format: str, *args: Any) -> None:
                    pass

            server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            return server

        redirects: dict[str, str] = {}
        a = serve(redirects)
        b = serve({})
        redirects["/cross"] = f"http://127.0.0.1:{b.server_port}/landed"
        redirects["/same"] = f"http://127.0.0.1:{a.server_port}/landed"
        for path in ("/cross", "/same"):
            request = Request(
                f"http://127.0.0.1:{a.server_port}{path}",
                headers={"Authorization": "Bearer t"},
            )
            assert upstream_http.fetch(request, lambda r: r.read()) == b"ok"
        assert seen[(b.server_port, "/landed")] is None
        assert seen[(a.server_port, "/landed")] == "Bearer t"
