"""Refresh-upstream tests: model."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from documents import json_document
from refresh_helpers import NEW, OLD, classify, comparison, files, model_with

from recipe_tooling import catalog_hf_model, vendored_upstream
from recipe_tooling.catalog_model import file_id
from recipe_tooling.refresh_upstream import assessment as refresh_assessment
from recipe_tooling.refresh_upstream import catalog_view as refresh_catalog_view
from recipe_tooling.refresh_upstream import classification as refresh_classification
from recipe_tooling.refresh_upstream import evidence as refresh_evidence
from recipe_tooling.refresh_upstream import http as refresh_http
from recipe_tooling.refresh_upstream import items as refresh_items
from recipe_tooling.refresh_upstream import model as refresh_model
from recipe_tooling.refresh_upstream import pinning as refresh_pinning
from recipe_tooling.upstream_http import UpstreamBusy
from vonk_forge_contracts import ModelDisplay


class ModelTests(unittest.TestCase):
    def test_same_paths_with_new_weight_digests_is_compatible(self) -> None:
        old = model_with(("model.safetensors", "README.md"))
        assert (
            refresh_classification.model_shape_problems(
                old, files("model.safetensors", "README.md", "NOTES.md")
            )
            == []
        )

    def test_changed_configuration_tokenizer_or_runtime_file_is_not(self) -> None:
        for path in ("config.json", "tokenizer.json", "dspark.py"):
            with self.subTest(path=path):
                old = model_with(("model.safetensors", path))
                assert refresh_classification.model_shape_problems(
                    old, files("model.safetensors", path)
                )

    def test_removed_and_new_weight_files_are_not(self) -> None:
        old = model_with(("model.safetensors", "config.json"))
        assert refresh_classification.model_shape_problems(
            old, files("model.safetensors")
        )
        assert refresh_classification.model_shape_problems(
            old, files("model.safetensors", "config.json", "model-2.safetensors")
        )

    def curated(self, old_paths, before, after, history=None):
        old = model_with(old_paths)
        item = refresh_model.Item(
            "model",
            "huggingface",
            "o/r",
            OLD,
            refresh_model.Target(
                "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
            ),
        )
        assessor = refresh_assessment.Assessor(
            refresh_catalog_view.Catalog(Path()),
            object(),  # pyright: ignore[reportArgumentType]
            None,
            {},
        )
        with patch.object(
            refresh_assessment.Assessor,
            "_inventory",
            return_value=files(*before),
            side_effect=history,
        ):
            kept = assessor._curated(old, item, files(*after))
        return old, kept

    def test_files_the_document_left_out_are_not_new(self) -> None:
        # #620: LTX-2.3's Model catalogues one checkpoint of eight; a README-only
        # revision is not "six new weights files".
        repository = ("a.safetensors", "b.safetensors", "README.md")
        old, kept = self.curated(("a.safetensors",), repository, repository)
        assert [f["path"] for f in kept] == ["a.safetensors"]
        assert refresh_classification.model_shape_problems(old, kept) == []

    def test_a_busy_history_read_is_not_judged_as_every_file_new(
        self,
    ) -> None:
        repository = ("a.safetensors", "b.safetensors")
        with pytest.raises(UpstreamBusy):
            self.curated(
                ("a.safetensors",),
                repository,
                repository,
                history=UpstreamBusy(),
            )
        # Any other unreadable history stays "unknown": every file is judged.
        _, kept = self.curated(
            ("a.safetensors",), repository, repository, history=OSError("gone")
        )
        assert len(kept) == 2

    def test_files_added_upstream_are_still_new(self) -> None:
        old, kept = self.curated(
            ("a.safetensors",),
            ("a.safetensors", "b.safetensors"),
            ("a.safetensors", "b.safetensors", "c.safetensors"),
        )
        assert [f["path"] for f in kept] == ["a.safetensors", "c.safetensors"]
        assert "new weights file" in " ".join(
            refresh_classification.model_shape_problems(old, kept)
        )

    def test_a_recipe_is_moved_only_by_files_it_mounts(self) -> None:
        # #530: the EXL3 recipe mounts config and two shards of the official
        # checkpoint; a new chat template and reference code concern other recipes.
        old = model_with(("config.json", "model.safetensors", "inference/model.py"))
        recipe = {
            "models": [
                {
                    "model": {"slug": old["identity"]["slug"]},
                    "files": [{"file_id": old["files"][0]["id"]}],
                }
            ]
        }
        assert refresh_classification.mounted_paths(recipe, old) == {"config.json"}
        new = files("config.json", "model.safetensors", "inference/model.py")
        for entry in new:
            entry["sha256"] = "c" * 64 if entry["path"] == "config.json" else "e" * 64
        new += files("chat_template.jinja")
        assert refresh_classification.mounted_unchanged(old, new, {"config.json"})
        new[0]["sha256"] = "f" * 64
        assert not refresh_classification.mounted_unchanged(old, new, {"config.json"})
        everything = [{"file_id": f["id"]} for f in old["files"]]
        recipe["models"][0]["files"] = everything
        assert refresh_classification.mounted_paths(recipe, old) is None

    def test_refreshed_model_renames_and_remaps_ids(self) -> None:
        old = model_with(("model.safetensors",))
        target = refresh_model.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        new = refresh_classification.refreshed_model(
            old, files("model.safetensors"), target
        )
        assert new["identity"]["slug"] == "demo-ffffffff"
        assert new["source"]["revision"] == NEW
        assert new["files"][0]["id"] == file_id("model.safetensors", "a" * 64)

    def test_refreshed_model_follows_the_source_between_split_and_whole(self) -> None:
        old = model_with(("model.safetensors",))
        target = refresh_model.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        split = files("model.safetensors")
        split[0]["installed_bytes"] = 20
        split[0]["parts"] = [
            {
                "path": f"model.safetensors.part0{i}",
                "sha256": "b" * 64,
                "size_bytes": 10,
            }
            for i in range(2)
        ]
        moved = refresh_classification.refreshed_model(old, split, target)
        assert len(moved["files"][0]["parts"]) == 2
        back = refresh_classification.refreshed_model(
            moved, files("model.safetensors"), target
        )
        assert "parts" not in back["files"][0]

    def test_provider_gate_change_creates_review_without_retargeting_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = model_with(("model.safetensors",))
            model["identity"]["slug"] = "gated-model"
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": OLD,
            }
            model["requires_token"] = False
            recipe = {
                "identity": {"slug": "recipe"},
                "models": [{"model": {"slug": "gated-model"}}],
                "execution": {"build": {"context": {"path": "missing-context"}}},
            }
            catalog = refresh_catalog_view.Catalog(
                Path(temporary),
                recipes={"recipe": recipe},
                models={"gated-model": model},
            )
            target = refresh_model.Target(
                "huggingface",
                "example/model",
                OLD,
                "2026-09-30",
                None,
                None,
                "https://huggingface.co/example/model/tree/" + OLD,
                access_status="gated",
            )

            result = refresh_items.collect_items(
                catalog,
                "recipe",
                object(),  # pyright: ignore[reportArgumentType]
                lambda _provider, _repo: target,
            )

            assert not result.items
            assert result.drifted
            assert "provider access changed to gated" in result.reasons[0]

    def test_restricted_token_required_or_unspecified_model_is_evidence(self) -> None:
        for requires_token in (True, None):
            with (
                self.subTest(requires_token=requires_token),
                tempfile.TemporaryDirectory() as temporary,
            ):
                model = model_with(("model.safetensors",))
                model["identity"]["slug"] = "restricted-model"
                model["source"] = {
                    "repository": "https://huggingface.co/example/model",
                    "revision": OLD,
                }
                if requires_token is not None:
                    model["requires_token"] = requires_token
                else:
                    model.pop("requires_token", None)
                recipe = {
                    "identity": {"slug": "recipe"},
                    "models": [{"model": {"slug": "restricted-model"}}],
                    "execution": {"build": {"context": {"path": "missing-context"}}},
                }
                catalog = refresh_catalog_view.Catalog(
                    Path(temporary),
                    recipes={"recipe": recipe},
                    models={"restricted-model": model},
                )

                def restricted(_provider: str, _repo: str):
                    raise refresh_http.AccessRestricted("anonymous metadata denied")

                def missing(_provider: str, _repo: str):
                    raise refresh_http.NotFound("gone")

                for resolver in (restricted, missing):
                    result = refresh_items.collect_items(
                        catalog,
                        "recipe",
                        object(),  # pyright: ignore[reportArgumentType]
                        resolver,
                    )
                    assert not result.items
                    assert not result.reasons
                    assert not result.drifted
                    assert not result.unreachable
                    assert len(result.evidence) == 1

    def test_unknown_anonymous_access_status_is_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = model_with(("model.safetensors",))
            model["identity"]["slug"] = "unknown-access-model"
            model["source"] = {
                "repository": "https://huggingface.co/example/model",
                "revision": OLD,
            }
            model["requires_token"] = False
            recipe = {
                "identity": {"slug": "recipe"},
                "models": [{"model": {"slug": "unknown-access-model"}}],
                "execution": {"build": {"context": {"path": "missing-context"}}},
            }
            catalog = refresh_catalog_view.Catalog(
                Path(temporary),
                recipes={"recipe": recipe},
                models={"unknown-access-model": model},
            )
            target = refresh_model.Target(
                "huggingface",
                "example/model",
                OLD,
                "2026-09-30",
                None,
                None,
                "https://huggingface.co/example/model/tree/" + OLD,
                access_status="unknown",
            )

            result = refresh_items.collect_items(
                catalog,
                "recipe",
                object(),  # pyright: ignore[reportArgumentType]
                lambda _provider, _repo: target,
            )

            assert not result.items
            assert not result.reasons
            assert not result.drifted
            assert not result.unreachable
            assert len(result.evidence) == 1

    def test_huggingface_resolver_records_explicit_access_status(self) -> None:
        class FakeHttp:
            def json(self, _url, _provider, *, anonymous=False):
                self.anonymous = anonymous
                return {"sha": OLD, "gated": "auto", "private": False}

        http = FakeHttp()
        target = refresh_pinning.resolve_huggingface(http, "example/model")  # pyright: ignore[reportArgumentType]
        assert target.access_status == "gated"
        assert http.anonymous


class ModelScopeTests(unittest.TestCase):
    def old_model(self, paths: tuple[str, ...]):
        document = model_with(paths)
        for entry in document["files"]:  # ids as older catalogs wrote them
            entry["id"] = f"{entry['path'].split('.')[0].lower()}-{'c' * 12}"
        return document

    def test_old_format_ids_do_not_block(self) -> None:
        # #476: only README.md changed; every id predates the path-digest suffix.
        old = self.old_model(
            (".gitattributes", "README.md", "config.json", "model.safetensors")
        )
        new = files(".gitattributes", "config.json", "model.safetensors", "README.md")
        for entry in new:
            entry["sha256"] = "c" * 64 if entry["path"] != "README.md" else "d" * 64
        assert refresh_classification.model_shape_problems(old, new) == []

    def test_same_fixture_with_config_digest_change_needs_review(self) -> None:
        old = self.old_model(("README.md", "config.json", "model.safetensors"))
        new = files("README.md", "config.json", "model.safetensors")
        for entry in new:
            entry["sha256"] = "d" * 64 if entry["path"] == "config.json" else "c" * 64
        problems = refresh_classification.model_shape_problems(old, new)
        assert len(problems) == 1
        assert "config.json" in problems[0]

    def test_removed_licence_and_readme_are_informational(self) -> None:
        old = self.old_model(("LICENSE", "README.md", "model.safetensors"))
        new = files("model.safetensors")
        for entry in new:
            entry["sha256"] = "c" * 64
        assert refresh_classification.model_shape_problems(old, new) == []
        target = refresh_model.Target(
            "huggingface", "o/r", NEW, "2026-09-30", None, None, "u"
        )
        refreshed = refresh_classification.refreshed_model(old, new, target)
        assert [f["path"] for f in refreshed["files"]] == ["model.safetensors"]
        item = refresh_model.Item("model", "huggingface", "o/r", OLD, target)
        assert "never gate" in refresh_evidence.model_evidence(item, old, new)

    def test_restructured_repository_needs_review(self) -> None:
        # #473: turboderp moved its files; config.json and shards are gone, new weights.
        old = self.old_model(
            ("config.json", "model-00001-of-00002.safetensors", "tokenizer.json")
        )
        new = files("cal_trace.safetensors", "measurement.json")
        joined = " ".join(refresh_classification.model_shape_problems(old, new))
        assert "config.json" in joined
        assert "model-00001-of-00002.safetensors" in joined
        assert "new weights file" in joined

    def test_tensorfold_whole_repo_adapter_keeps_judging_every_file(self) -> None:
        cmp = comparison(files=("src/tensorfold/cuda/server.py", "pyproject.toml"))
        reasons = classify(cmp)  # used_changes=None: the adapter uses the whole repo
        assert "server.py" in " ".join(reasons)

    def test_file_id_matches_a_fresh_catalog(self) -> None:
        tree = [
            {
                "type": "file",
                "path": "sub/model-1.safetensors",
                "size": 5,
                "lfs": {"oid": "e" * 64},
            }
        ]
        (entry,) = catalog_hf_model.inventory("o/r", NEW, tree)  # pyright: ignore[reportArgumentType]
        assert entry["id"] == file_id(entry["path"], "e" * 64)


class ModelReExportTests(unittest.TestCase):
    """A re-export of the same Model repository is its newest revision (#566)."""

    def shards(self, count: int) -> tuple[str, ...]:
        return tuple(
            f"model-{i:05d}-of-{count:05d}.safetensors" for i in range(1, count + 1)
        )

    def move(
        self,
        mounted: tuple[str, ...] | None,
        *,
        old_paths: tuple[str, ...] | None = None,
        new_inventory: list[dict[str, Any]] | None = None,
        kit_pin: bool = False,
    ):
        old = model_with(
            old_paths or ("config.json", "tokenizer.json", *self.shards(36))
        )
        old["source"]["repository"] = "https://huggingface.co/o/r"
        new = files("config.json", "HYBRID.json", *self.shards(41), sha="e" * 64)
        new += files("tokenizer.json")
        if new_inventory is not None:
            new = new_inventory
        selected = [f for f in old["files"] if mounted is None or f["path"] in mounted]
        mount = {"target": "/models"}
        recipe = {
            "identity": {"slug": "recipe"},
            "models": [
                {
                    "id": "primary",
                    "model": {"slug": old["identity"]["slug"], "content_sha256": "x"},
                    "files": [
                        {
                            "id": "primary-" + f["id"][:40],
                            "file_id": f["id"],
                            "mount": mount,
                            "roles": ["entrypoint"],
                        }
                        for f in selected
                    ],
                }
            ],
            "topology": {
                "roles": [{"resources": {"disk": {"artifact_bytes": 1_000_000}}}]
            },
        }
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "display/models").mkdir(parents=True)
        (root / "display/models" / f"{old['identity']['slug']}.json").write_text(
            json.dumps(
                {
                    "kind": "model-display",
                    "identity": {
                        "publisher": old["identity"]["publisher"],
                        "slug": old["identity"]["slug"],
                    },
                    "family_title": "Demo family",
                    "model_title": "Demo",
                    "description": "A demo model.",
                    "tags": ["demo"],
                    "modalities": ["text"],
                    "capabilities": [],
                    "license": {
                        "spdx": "MIT",
                        "url": "https://example.com/license",
                        "attribution": [],
                    },
                }
            )
        )
        catalog = refresh_catalog_view.Catalog(
            root,
            recipes={"recipe": recipe},
            models={old["identity"]["slug"]: old},
        )
        assessor = refresh_assessment.Assessor(catalog, object(), None, {})  # pyright: ignore[reportArgumentType]
        item = refresh_model.Item(
            "model",
            "huggingface",
            "o/r",
            OLD,
            refresh_model.Target(
                "huggingface", "o/r", NEW, "2026-10-06", None, None, "u"
            ),
            old["identity"]["slug"],
        )
        if kit_pin:
            item.kit_pin = object()
        inventories = {
            OLD: files(*(f["path"] for f in old["files"])),
            NEW: new,
        }
        result = refresh_model.Assessment("recipe", items=[item])
        new_recipe = json.loads(json.dumps(recipe))
        edits: dict[str, bytes | None] = {}
        with patch.object(
            refresh_assessment.Assessor,
            "_inventory",
            side_effect=lambda _repo, revision: inventories[revision],
        ):
            assessor._model(item, recipe, new_recipe, result, edits)
        return result, new_recipe, edits

    def test_resharded_re_export_moves_a_recipe_mounting_everything(self) -> None:
        result, new_recipe, edits = self.move(None)
        assert result.reasons == []
        assert not result.items[0].noop
        written = edits[f"models/demo-{NEW[:8]}.json"]
        assert written is not None
        moved = json_document(written)
        paths = {f["path"] for f in moved["files"]}
        assert "HYBRID.json" in paths
        assert "model-00041-of-00041.safetensors" in paths
        assert "model-00036-of-00036.safetensors" not in paths
        mounted = {s["file_id"] for s in new_recipe["models"][0]["files"]}
        assert mounted == {f["id"] for f in moved["files"]}

    def test_a_sibling_checkpoint_beside_identical_files_is_not_a_move(self) -> None:
        # #2749: the repository gained another text encoder; every catalogued file
        # is the same bytes.
        old_paths = (
            "split_files/diffusion_models/a.safetensors",
            "split_files/text_encoders/te_fp8.safetensors",
            "split_files/vae/v.safetensors",
        )
        new = files(*old_paths, sha="c" * 64)
        new += files("split_files/text_encoders/te_int8.safetensors", sha="d" * 64)
        new += files("README_strata.md", sha="e" * 64)
        result, _, edits = self.move(None, old_paths=old_paths, new_inventory=new)
        assert result.reasons == []
        assert result.items[0].noop
        assert not edits

        # A kit-declared revision keeps the kit's strict shape rule: a review.
        result, _, edits = self.move(
            None, old_paths=old_paths, new_inventory=new, kit_pin=True
        )
        assert result.reasons
        assert not result.items[0].noop
        assert not edits

    def test_an_added_chat_template_still_moves_a_recipe_mounting_everything(
        self,
    ) -> None:
        old_paths = ("config.json", "model-00001-of-00001.safetensors")
        new = files(*old_paths, sha="c" * 64) + files("chat_template.jinja")
        result, _, edits = self.move(None, old_paths=old_paths, new_inventory=new)
        assert result.reasons == []
        assert not result.items[0].noop
        assert edits

    def test_the_models_display_moves_with_it(self) -> None:
        _, _, edits = self.move(None)
        # The Model's display moves with it: same text, the new Model's identity.
        written_displays = dict(edits)
        shown = written_displays.pop(f"display/models/demo-{NEW[:8]}.json")
        assert shown is not None
        moved_display = ModelDisplay.model_validate_json(shown)
        assert moved_display.identity.slug == f"demo-{NEW[:8]}"
        assert moved_display.description == "A demo model."
        assert written_displays[f"display/models/demo-{OLD[:8]}.json"] is None

    def test_a_named_mounted_file_gone_stays_a_review_naming_it(self) -> None:
        result, _, edits = self.move(
            ("config.json", "model-00036-of-00036.safetensors")
        )
        assert (
            "file this recipe mounts is gone upstream: "
            "`model-00036-of-00036.safetensors`" in result.reasons[0]
        )
        assert not edits

    def test_a_kit_declared_revision_keeps_its_file_set(self) -> None:
        old = model_with(("config.json", *self.shards(2)))
        new = files("config.json", *self.shards(3))
        assert refresh_classification.model_problems(old, new, None, declared=True)
        assert (
            refresh_classification.model_problems(old, new, None, declared=False) == []
        )


class KitScopeTests(unittest.TestCase):
    """A kit is the subdirectory the adapter cites or the files it vendors, not a monorepo."""

    def assess(self, files: tuple[str, ...], *, notice: str = "", dockerfile: str = ""):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        adapter = root / "adapters/x"
        adapter.mkdir(parents=True)
        (adapter / "Dockerfile").write_text("FROM scratch\n" + dockerfile)
        (adapter / "NOTICE").write_text(notice)
        recipe = {
            "identity": {"slug": "recipe"},
            "provenance": {
                "attribution": ["Example"],
                "source_reference": f"https://github.com/example/recipes/tree/{OLD}",
            },
            "release": {"version": "1.0.0", "released_at": "2026-01-01"},
            "models": [],
            "execution": {"build": {"context": {"path": "adapters/x"}}},
            "runtime": {},
        }
        catalog = refresh_catalog_view.Catalog(root, recipes={"recipe": recipe})
        assessor = refresh_assessment.Assessor(catalog, object(), None, {})  # pyright: ignore[reportArgumentType]
        assessor.target = lambda _provider, _repo: refresh_model.Target(  # pyright: ignore[reportAttributeAccessIssue]
            "github", "example/recipes", NEW, "2026-10-13", None, None, "u"
        )
        with patch(
            "recipe_tooling.refresh_upstream.items.compare_commits",
            return_value=comparison(files=files),
        ):
            return assessor.assess("recipe")

    def cite(self, subdirectory: str = "") -> str:
        return f"Profiles from https://github.com/example/recipes/tree/{OLD}{subdirectory}\n"

    def test_a_citation_scopes_the_kit_to_its_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "NOTICE").write_text(self.cite("/models/vllm-sr"))
            assert refresh_catalog_view.kit_citation_scope(
                Path(tmp), "example/recipes"
            ) == ("models/vllm-sr",)
            (Path(tmp) / "NOTICE").write_text(self.cite())
            assert (
                refresh_catalog_view.kit_citation_scope(Path(tmp), "example/recipes")
                == ()
            )
            (Path(tmp) / "NOTICE").write_text("nothing\n")
            assert (
                refresh_catalog_view.kit_citation_scope(Path(tmp), "example/recipes")
                is None
            )

    def test_changes_outside_the_cited_subdirectory_are_not_a_refresh(self) -> None:
        result = self.assess(
            ("models/other/serve.yaml", "recipes/x.json"),
            notice=self.cite("/models/vllm-sr"),
        )
        assert not result.drifted, result.reasons

    def test_changes_inside_the_cited_subdirectory_need_review(self) -> None:
        result = self.assess(
            ("models/other/serve.yaml", "models/vllm-sr/serve.yaml"),
            notice=self.cite("/models/vllm-sr"),
        )
        assert not result.mechanical
        reasons = " ".join(result.reasons)
        assert "models/vllm-sr/serve.yaml" in reasons
        assert "models/other" not in reasons

    def test_a_citation_of_the_whole_repository_still_judges_everything(self) -> None:
        result = self.assess(("models/other/serve.yaml",), notice=self.cite())
        assert "models/other/serve.yaml" in " ".join(result.reasons)

    def test_a_pip_installed_library_ignores_examples_and_its_own_docker(self) -> None:
        install = (
            f"RUN pip install \\\n    git+https://github.com/example/recipes@{OLD}\n"
        )
        harmless = ("examples/text/requirements.txt", "docker/cuda/Dockerfile")
        result = self.assess(harmless, dockerfile=install)
        assert not result.drifted, result.reasons
        result = self.assess((*harmless, "src/serve.sh"), dockerfile=install)
        assert "src/serve.sh" in " ".join(result.reasons)
        assert "examples/" not in " ".join(result.reasons)
        # Without a pip install of the library those files stay judged.
        assert refresh_model.is_blocking("examples/text/requirements.txt")
        assert refresh_model.is_blocking("docker/cuda/Dockerfile")

    def used(self, files: dict[str, str], manifest: dict[str, Any], trees=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for name, text in files.items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(text)
        (directory / "vendored-upstream.json").write_text(json.dumps(manifest))
        item = refresh_model.Item(
            "source",
            "github",
            "example/recipes",
            OLD,
            refresh_model.Target(
                "github", "example/recipes", NEW, "d", None, None, "u"
            ),
        )
        assessment = refresh_model.Assessment("recipe", items=[item])
        pin = refresh_catalog_view.AdapterPin("example/recipes", OLD, directory)
        assessor = refresh_assessment.Assessor(
            refresh_catalog_view.Catalog(directory),
            object(),  # pyright: ignore[reportArgumentType]
            None,
            {},
        )
        recipe = {"execution": {"build": {"context": {"path": "."}}}}
        with patch.object(
            vendored_upstream,
            "fetch_tree",
            side_effect=lambda _repo, commit: (trees or {}).get(
                commit, vendored_upstream.Tree({}, True)
            ),
        ):
            return assessor._used_changes(item, recipe, pin, set(), assessment)

    MAPPED: ClassVar[dict[str, Any]] = {
        "source": {"repo": "example/recipes", "commit": OLD},
        "renamed": {"launch.sh": "runs/a/launch/launch.sh"},
    }

    def test_a_mapped_subset_does_not_use_the_whole_repository(self) -> None:
        # A bare git checkout beside a mention of the repository is not a clone of it.
        text = "# https://github.com/example/recipes\ngit checkout abc\n"
        assert self.used({"Dockerfile": text}, self.MAPPED) == []
        unmapped = {"source": self.MAPPED["source"]}
        assert self.used({"Dockerfile": text}, unmapped) is None

    def test_a_genuine_clone_beats_the_mapped_subset(self) -> None:
        clone = "RUN git clone https://github.com/example/recipes /src\n"
        assert self.used({"Dockerfile": clone}, self.MAPPED) is None

    def test_a_mod_the_creator_yaml_activates_but_the_adapter_lacks_is_used(
        self,
    ) -> None:
        yaml = "name: x\nmods:\n  - mods/nemotron-nano\n  - mods/vendored\ncommand: |\n  x\n"
        old = vendored_upstream.Tree({"mods/nemotron-nano/run.sh": "a" * 40}, True)
        new = vendored_upstream.Tree({"mods/nemotron-nano/run.sh": "b" * 40}, True)
        trees = {OLD: old, NEW: new}
        files = {"recipes/r.yaml": yaml, "mods/vendored/run.sh": "x"}
        changed = self.used(files, {"source": self.MAPPED["source"]}, trees)
        assert changed == ["mods/nemotron-nano/run.sh"]
        # Unchanged upstream: nothing used changed; a vendored mod is not double counted.
        same = {OLD: old, NEW: old}
        assert self.used(files, {"source": self.MAPPED["source"]}, same) == []
        assert refresh_catalog_view.active_mods(None) == set()

    def test_every_spelling_of_a_mod_list_counts(self) -> None:
        spellings = {
            "indented": "mods:\n  - mods/a\n",
            "column zero": "mods:\n- mods/a\n",
            "flow": "mods: [mods/a, mods/b]\n",
            "dot slash": "mods:\n  - ./mods/a\n",
            "quoted": "mods:\n  - 'mods/a'\n",
            "commented": "mods:\n  # - mods/a\n",
            "trailing slash": "mods:\n  - mods/a/\n",
        }
        for label, text in spellings.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "r.yaml").write_text(text)
                assert "mods/a" in refresh_catalog_view.active_mods(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "r.yaml").write_text("mods: []\n")
            assert refresh_catalog_view.active_mods(Path(tmp)) == set()
