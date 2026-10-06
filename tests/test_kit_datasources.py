"""Kit-declared Hugging Face revisions and container images (tools/kit-pins datasources).

``git-ref`` is covered by ``test_kit_pins.py``. A kit may also declare the model it serves
(``hf-revision``: repository and revision) and the image it runs on (``oci-image``: a tag
or digest of a repository). Like a git ref they are locked once, the recipe follows the
lock, and the refresh scanner never proposes an independent newest value for them. The
fixture for the model is the switch that exposed the gap: MiaAI-Lab's GLM kit pinned
``GLM-5.3-Flash-EXL3-TR3-4bpw`` and, from v1.5, pins ``GLM-5.3-Flash-EXL3-4bpw-TensorFold``
(the TR3 repository now answers HTTP 401).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_kit_pins import (
    BASE,
    KIT,
    FakeApi,
    assessor_for,
    kit_pins,
    refresh,
    upstream,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads(
    (
        ROOT / "contracts/src/vonk_forge_contracts/examples/model-definition.json"
    ).read_text()
)

TR3 = "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
TR3_REV = "9eaebb7c4e96d983dcd538e18624622ba5b820a8"
NEW = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold"
NEW_REV = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
NEW_REV2 = (
    "41ba3c82eff6ef5b93debff02f47ae4681d824ab"  # a later commit of the same repository
)
DRAFT = "incoai/GLM-5.3-Flash-DFlash2"
DRAFT_REV = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
KIT_TR3 = "1" * 40  # declares the TR3 checkpoint
KIT_NEW = "2" * 40  # declares the 4bpw-TensorFold checkpoint
KIT_NEXT = "3" * 40  # declares a later revision of it


def config(model: str, revision: str, drafter_revision: str = DRAFT_REV) -> str:
    """The shape of the kit's ``scripts/config.sh``: a case on the model, a test on the drafter."""
    return (
        'MODEL_ID="${MODEL_ID:-x}"\n'
        'case "$MODEL_ID" in\n'
        f"  {model}) _rev={revision} ;;\n"
        '  *) _rev="" ;;\n'
        "esac\n"
        'DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"\n'
        f'_rev=""; [[ "$DFLASH2_ID" == {DRAFT} ]] && _rev={drafter_revision}\n'
    )


CONFIGS = {
    KIT_TR3: config(TR3, TR3_REV),
    KIT_NEW: config(NEW, NEW_REV),
    KIT_NEXT: config(NEW, NEW_REV2),
}
MODEL_PATTERN = r"^\s*(?P<repo>[\w.-]+/[\w.-]+)\) _rev=(?P<version>[0-9a-f]{40}) ;;$"
DRAFTER_PATTERN = (
    r'^_rev=""; \[\[ "\$DFLASH2_ID" == (?P<repo>[\w.-]+/[\w.-]+) \]\] '
    r"&& _rev=(?P<version>[0-9a-f]{40})$"
)
HF_MANIFEST = {
    "kit": {
        "dependencies": {
            "model": {
                "datasource": "hf-revision",
                "file": "scripts/config.sh",
                "pattern": MODEL_PATTERN,
                "binds": "primary",
            },
            "drafter": {
                "datasource": "hf-revision",
                "file": "scripts/config.sh",
                "pattern": DRAFTER_PATTERN,
                "binds": "drafter",
            },
        }
    },
    "ours": ["licenses/", "patches/"],
}


class FakeHub:
    """Hugging Face's API as far as the kit tools use it: revisions, refs and heads."""

    def __init__(
        self,
        revisions: dict[str, list[str]],
        heads: dict[str, str] | None = None,
        private: tuple[str, ...] = (),
    ) -> None:
        self.revisions = revisions
        self.heads = heads or {}
        self.private = private
        self.calls: list[str] = []

    def __call__(self, path: str) -> Any:
        self.calls.append(path)
        parts = path.split("/")
        repo = "/".join(parts[1:3])
        if repo in self.private:
            raise RuntimeError(f"HTTP 401 for {path}")
        if parts[3:4] == ["revision"]:
            ref = urllib.parse.unquote(parts[4])
            known = self.revisions.get(repo, [])
            full = next((r for r in known if r.startswith(ref)), None)
            if full is None and ref in self.heads:
                full = self.heads[ref]
            if full is None:
                raise RuntimeError(f"HTTP 404 for {path}")
            return {"sha": full, "lastModified": "2026-10-02T13:04:40.000Z"}
        if parts[3:4] == ["refs"]:
            return {"branches": [{"name": "main"}], "tags": []}
        if len(parts) == 3:
            return {
                "sha": self.heads.get(repo, self.revisions[repo][-1]),
                "lastModified": "2026-10-03T00:00:00.000Z",
                "private": False,
                "gated": False,
            }
        raise RuntimeError(f"unrouted {path}")


def hub_for(**private: Any) -> FakeHub:
    return FakeHub(
        {TR3: [TR3_REV], NEW: [NEW_REV, NEW_REV2], DRAFT: [DRAFT_REV]}, **private
    )


def slug_name(repo: str) -> str:
    return repo.split("/")[1].lower().replace(".", "-")


def model_document(
    repo: str, revision: str, files: dict[str, tuple[str, str]], size: int = 10
) -> dict[str, Any]:
    """A Model document of ``repo`` at ``revision``: path -> (role, sha256)."""
    document = json.loads(json.dumps(EXAMPLE))
    name = slug_name(repo)
    document["identity"]["slug"] = f"{name}-{revision[:8]}"
    document["identity"]["model"]["slug"] = name
    document["source"] = {
        "repository": f"https://huggingface.co/{repo}",
        "revision": revision,
    }
    document["files"] = [
        {
            "id": refresh.file_id(path, sha),
            "path": path,
            "roles": [role],
            "sha256": sha,
            "size_bytes": size,
        }
        for path, (role, sha) in sorted(files.items())
    ]
    return document


TR3_FILES = {
    "config.json": ("configuration", "a" * 64),
    "model-00001-of-00002.safetensors": ("weights", "b" * 64),
    "model-00002-of-00002.safetensors": ("weights", "c" * 64),
    "README.md": ("metadata", "d" * 64),
}
NEW_FILES = {
    "config.json": ("configuration", "e" * 64),
    "model-00001-of-00001.safetensors": ("weights", "f" * 64),
    "README.md": ("metadata", "1" * 64),
}
DRAFT_FILES = {"model.safetensors": ("weights", "9" * 64)}


def recipe_document(
    models: dict[str, dict[str, Any]], kit_commit: str
) -> dict[str, Any]:
    return {
        "identity": {"publisher": "vonk-forge", "slug": "mia"},
        "provenance": {
            "attribution": ["x"],
            "source_reference": f"https://github.com/{KIT}/tree/{kit_commit}",
        },
        "release": {"version": "1.4.0", "released_at": "2026-10-01"},
        "execution": {"build": {"context": {"path": "adapters/mia"}}},
        "models": [
            {
                "id": model_id,
                "model": {
                    "publisher": "mia-ai-lab",
                    "slug": document["identity"]["slug"],
                    "content_sha256": kit_pins.model_digest(document),
                },
                "files": [
                    {
                        "file_id": item["id"],
                        "id": f"{model_id}-{item['id']}",
                        "mount": {"target": f"/models/{model_id}"},
                        "roles": ["entrypoint", "worker"],
                    }
                    for item in document["files"]
                ],
            }
            for model_id, document in models.items()
        ],
        "topology": {
            "roles": [
                {"name": "entrypoint", "resources": {"disk": {"artifact_bytes": 1000}}},
                {"name": "worker", "resources": {"disk": {"artifact_bytes": 1000}}},
            ]
        },
    }


class World:
    """A temporary repository: the catalogued Models, one recipe and its kit adapter."""

    def __init__(
        self,
        root: Path,
        *,
        kit_commit: str,
        models: dict[str, dict[str, Any]],
        locked: dict[str, tuple[str, str]],
        manifest: dict[str, Any] | None = None,
        extra_models: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.root = root
        self.kit_commit = kit_commit
        self.directory = root / "adapters/mia"
        self.directory.mkdir(parents=True)
        (root / "models").mkdir()
        (root / "recipes").mkdir()
        for document in [*models.values(), *extra_models]:
            self.write_model(document)
        self.recipe = recipe_document(models, kit_commit)
        (root / "recipes/mia.json").write_text(kit_pins.recipe_text(self.recipe))
        (self.directory / "vendored-upstream.json").write_text(
            json.dumps(manifest or HF_MANIFEST)
        )
        (self.directory / "Dockerfile").write_text("FROM scratch\n")
        dependencies = {
            name: {
                "datasource": "hf-revision",
                "kind": "commit",
                "ref": revision,
                "repo": repo,
                "revision": revision,
            }
            for name, (repo, revision) in locked.items()
        }
        (self.directory / "kit-lock.json").write_text(
            kit_pins.lock_text(
                {
                    "dependencies": dependencies,
                    "kit": {"commit": kit_commit, "repo": KIT},
                }
            )
        )

    def write_model(self, document: dict[str, Any]) -> None:
        path = self.root / "models" / f"{document['identity']['slug']}.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    def check(self, hub: FakeHub | None = None) -> list[str]:
        return kit_pins.check_recipe(
            self.root,
            "mia",
            self.recipe,
            upstream(FakeApi(CONFIGS), hub=kit_pins.HuggingFace(hub or hub_for())),
        )

    def lock(self, hub: FakeHub | None = None) -> list[str]:
        return kit_pins.apply_lock(
            self.root,
            "mia",
            self.recipe,
            upstream(FakeApi(CONFIGS), hub=kit_pins.HuggingFace(hub or hub_for())),
        )


def tr3_world(root: Path, kit_commit: str, *, locked: str = "tr3") -> World:
    models = {
        "primary": model_document(TR3, TR3_REV, TR3_FILES),
        "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
    }
    pins = {
        "tr3": {"model": (TR3, TR3_REV), "drafter": (DRAFT, DRAFT_REV)},
        "new": {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
    }[locked]
    return World(root, kit_commit=kit_commit, models=models, locked=pins)


class HfDeclarationTests(unittest.TestCase):
    def declaration(self, **override: Any) -> Any:
        entry = dict(HF_MANIFEST["kit"]["dependencies"]["model"], **override)
        return kit_pins.declarations({"kit": {"dependencies": {"model": entry}}})[0]

    def test_a_case_line_pairs_the_repository_with_its_revision(self) -> None:
        wanted = self.declaration()
        for commit, repo, revision in (
            (KIT_TR3, TR3, TR3_REV),
            (KIT_NEW, NEW, NEW_REV),
        ):
            text = CONFIGS[commit]
            self.assertEqual(kit_pins.parse_repo(text, wanted), repo)
            self.assertEqual(kit_pins.parse_declared(text, wanted), revision)

    def test_the_drafter_is_a_second_declaration_in_the_same_file(self) -> None:
        wanted = kit_pins.declarations(HF_MANIFEST)
        self.assertEqual([d.name for d in wanted], ["drafter", "model"])
        drafter = wanted[0]
        self.assertEqual(kit_pins.parse_repo(CONFIGS[KIT_NEW], drafter), DRAFT)
        self.assertEqual(kit_pins.parse_declared(CONFIGS[KIT_NEW], drafter), DRAFT_REV)

    def test_one_place_names_the_repository_and_the_recipe_model_is_named(self) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "give one"):
            self.declaration(repo=NEW)
        with self.assertRaisesRegex(kit_pins.KitError, "lacks repo"):
            self.declaration(pattern=r"^_rev=(?P<version>[0-9a-f]{40})$")
        with self.assertRaisesRegex(kit_pins.KitError, "binds"):
            kit_pins.declarations(
                {
                    "kit": {
                        "dependencies": {
                            "model": {
                                k: v
                                for k, v in HF_MANIFEST["kit"]["dependencies"][
                                    "model"
                                ].items()
                                if k != "binds"
                            }
                        }
                    }
                }
            )
        with self.assertRaisesRegex(kit_pins.KitError, "binds is for an hf-revision"):
            kit_pins.declarations(
                {
                    "kit": {
                        "dependencies": {
                            "base": {
                                "datasource": "git-ref",
                                "repo": BASE,
                                "file": "f",
                                "pattern": "(?P<version>x)",
                                "binds": "primary",
                            }
                        }
                    }
                }
            )

    def test_several_models_in_the_file_are_an_error_not_a_guess(self) -> None:
        text = CONFIGS[KIT_TR3] + CONFIGS[KIT_NEW]
        with self.assertRaisesRegex(kit_pins.KitError, "several"):
            kit_pins.parse_repo(text, self.declaration())

    def test_a_declared_repository_is_a_huggingface_owner_name(self) -> None:
        self.assertEqual(
            kit_pins._repository("hf-revision", "https://huggingface.co/a/b/"), "a/b"
        )
        with self.assertRaises(kit_pins.KitError):
            kit_pins._repository("hf-revision", "not-a-repo")


class HfDerivationTests(unittest.TestCase):
    def derive(self, commit: str, hub: FakeHub | None = None, **override: Any) -> Any:
        manifest = json.loads(json.dumps(HF_MANIFEST))
        manifest["kit"]["dependencies"]["model"].update(override)
        return kit_pins.derive(
            KIT,
            commit,
            kit_pins.declarations(manifest),
            upstream(FakeApi(CONFIGS), hub=kit_pins.HuggingFace(hub or hub_for())),
        )

    def test_the_kit_locks_a_revision_of_the_repository_it_declares(self) -> None:
        drafter, model = self.derive(KIT_NEW)
        self.assertEqual(
            (model.repo, model.ref, model.immutable, model.kind, model.datasource),
            (NEW, NEW_REV, NEW_REV, "commit", "hf-revision"),
        )
        self.assertEqual((drafter.repo, drafter.immutable), (DRAFT, DRAFT_REV))

    def test_the_tr3_to_4bpw_tensorfold_switch_changes_repository_and_revision(
        self,
    ) -> None:
        old = self.derive(KIT_TR3)[1]
        new = self.derive(KIT_NEW)[1]
        self.assertEqual((old.repo, old.immutable), (TR3, TR3_REV))
        self.assertEqual((new.repo, new.immutable), (NEW, NEW_REV))

    def test_the_lock_names_the_datasource_and_keeps_the_revision_under_its_own_key(
        self,
    ) -> None:
        pins = self.derive(KIT_NEW)
        lock = kit_pins.lock_document(KIT, KIT_NEW, pins)
        self.assertEqual(
            lock["dependencies"]["model"],
            {
                "datasource": "hf-revision",
                "repo": NEW,
                "ref": NEW_REV,
                "kind": "commit",
                "revision": NEW_REV,
            },
        )

    def test_an_abbreviated_revision_locks_the_full_commit(self) -> None:
        manifest = json.loads(json.dumps(HF_MANIFEST))
        manifest["kit"]["dependencies"] = {
            "model": dict(
                manifest["kit"]["dependencies"]["model"],
                pattern=r"^REVISION=(?P<version>[0-9a-f]{7,40})$",
                repo=NEW,
            )
        }
        pin = kit_pins.derive(
            KIT,
            "4" * 40,
            kit_pins.declarations(manifest),
            upstream(
                FakeApi({"4" * 40: f"REVISION={NEW_REV[:8]}\n"}),
                hub=kit_pins.HuggingFace(hub_for()),
            ),
        )[0]
        self.assertEqual(
            (pin.ref, pin.kind, pin.immutable), (NEW_REV[:8], "commit", NEW_REV)
        )

    def test_a_branch_the_kit_declares_is_followed_like_a_git_branch(self) -> None:
        hub = hub_for(heads={"main": NEW_REV2})
        manifest = json.loads(json.dumps(HF_MANIFEST))
        manifest["kit"]["dependencies"] = {
            "model": dict(
                manifest["kit"]["dependencies"]["model"],
                pattern=r"^BRANCH=(?P<version>\S+)$",
                repo=NEW,
            )
        }
        pin = kit_pins.derive(
            KIT,
            "4" * 40,
            kit_pins.declarations(manifest),
            upstream(
                FakeApi({"4" * 40: "BRANCH=main\n"}), hub=kit_pins.HuggingFace(hub)
            ),
        )[0]
        self.assertEqual(
            (pin.ref, pin.kind, pin.immutable), ("main", "branch", NEW_REV2)
        )

    def test_a_repository_that_is_not_anonymously_readable_cannot_be_locked(
        self,
    ) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "cannot resolve .*TR3"):
            self.derive(KIT_TR3, hub=hub_for(private=(TR3,)))

    def test_an_override_replaces_the_declared_repository_and_revision(self) -> None:
        pin = self.derive(
            KIT_NEW,
            override={"repo": TR3, "revision": TR3_REV, "reason": "why"},
        )[1]
        self.assertEqual(
            (pin.repo, pin.immutable, pin.overridden), (TR3, TR3_REV, "why")
        )


class HfCheckAndLockTests(unittest.TestCase):
    def test_a_recipe_on_the_kits_model_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW, locked="new")
            world.recipe = recipe_document(
                {
                    "primary": model_document(NEW, NEW_REV, NEW_FILES),
                    "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
                },
                KIT_NEW,
            )
            world.write_model(model_document(NEW, NEW_REV, NEW_FILES))
            self.assertEqual(world.check(), [])

    def test_a_recipe_on_another_repository_than_the_kits_is_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW, locked="new")
            problems = world.check()
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("'primary'", problems[0])
        self.assertIn(f"{TR3}@{TR3_REV[:12]}", problems[0])
        self.assertIn(f"{NEW}@{NEW_REV[:12]}", problems[0])
        self.assertIn("tools/catalog-hf-model", problems[0])

    def test_a_hand_pinned_revision_of_the_same_repository_is_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = {
                "primary": model_document(NEW, NEW_REV2, NEW_FILES),
                "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
            }
            world = World(
                Path(tmp),
                kit_commit=KIT_NEW,
                models=models,
                locked={"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            )
            problems = world.check()
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(f"{NEW}@{NEW_REV2[:12]}", problems[0])
        self.assertIn(f"kit locks {NEW}@{NEW_REV[:12]}", problems[0])

    def test_a_stale_lock_names_the_kit_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW)  # the lock still says TR3
            problems = world.check()
        self.assertTrue(any("stale" in p for p in problems), problems)

    def test_the_check_resolves_nothing_it_has_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_TR3)
            hub = hub_for()
            self.assertEqual(world.check(hub), [])
        self.assertEqual(hub.calls, [])

    def test_lock_rebinds_the_recipe_to_the_catalogued_new_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW)
            new = model_document(NEW, NEW_REV, NEW_FILES, size=7)
            # a public copy by another publisher: the reference follows the document
            new["identity"]["publisher"] = "public-copy"
            world.write_model(new)
            changed = world.lock()
            self.assertIn("recipes/mia.json", changed)
            recipe = json.loads((world.root / "recipes/mia.json").read_text())
            primary = next(m for m in recipe["models"] if m["id"] == "primary")
            self.assertEqual(primary["model"]["slug"], new["identity"]["slug"])
            self.assertEqual(primary["model"]["publisher"], "public-copy")
            self.assertEqual(
                primary["model"]["content_sha256"], kit_pins.model_digest(new)
            )
            # a repository switch has no file in common: the whole Model stays mounted
            self.assertEqual(
                [f["file_id"] for f in primary["files"]],
                [f["id"] for f in new["files"]],
            )
            self.assertTrue(
                all(f["id"].startswith("primary-") for f in primary["files"])
            )
            # the disk envelope follows the size of what is selected (4 files -> 3 files)
            self.assertEqual(
                [
                    r["resources"]["disk"]["artifact_bytes"]
                    for r in recipe["topology"]["roles"]
                ],
                [1000 - 10 * 4 + 7 * 3] * 2,
            )
            # no recipe uses the TR3 document any more: the newest-only rule removes it
            self.assertFalse(
                (
                    world.root / "models" / f"{slug_name(TR3)}-{TR3_REV[:8]}.json"
                ).exists()
            )
            lock = json.loads((world.directory / "kit-lock.json").read_text())
            self.assertEqual(lock["dependencies"]["model"]["repo"], NEW)
            world.recipe = recipe
            self.assertEqual(world.check(), [])
            self.assertEqual(world.lock(), [])

    def test_lock_keeps_a_model_another_recipe_still_uses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW)
            world.write_model(model_document(NEW, NEW_REV, NEW_FILES))
            other = recipe_document(
                {"primary": model_document(TR3, TR3_REV, TR3_FILES)}, KIT_TR3
            )
            other["identity"]["slug"] = "other"
            (world.root / "recipes/other.json").write_text(kit_pins.recipe_text(other))
            changed = world.lock()
            self.assertFalse(any("removed" in c for c in changed), changed)
            self.assertTrue(
                (
                    world.root / "models" / f"{slug_name(TR3)}-{TR3_REV[:8]}.json"
                ).exists()
            )

    def test_lock_asks_for_another_repository_to_be_catalogued_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW)
            with self.assertRaisesRegex(
                kit_pins.KitError, "another repository"
            ) as caught:
                world.lock()
        self.assertIn(f"--repository {NEW} --revision {NEW_REV}", str(caught.exception))

    def test_lock_catalogues_another_revision_of_the_same_repository_itself(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = {
                "primary": model_document(NEW, NEW_REV, NEW_FILES),
                "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
            }
            world = World(
                Path(tmp),
                kit_commit=KIT_NEXT,
                models=models,
                locked={"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            )
            moved = {
                "config.json": ("configuration", "e" * 64),
                "model-00001-of-00002.safetensors": ("weights", "3" * 64),
                "model-00002-of-00002.safetensors": ("weights", "4" * 64),
                "README.md": ("metadata", "5" * 64),
                "MODEL_CARD.md": ("metadata", "6" * 64),
            }
            hub = kit_pins.HuggingFace(
                hub_for(),
                inventory=lambda repo, revision: (
                    inventory(moved, size=9)
                    if (repo, revision) == (NEW, NEW_REV2)
                    else []
                ),
            )
            changed = kit_pins.apply_lock(
                world.root,
                "mia",
                world.recipe,
                upstream(FakeApi(CONFIGS), hub=hub),
            )
            new_slug = f"{slug_name(NEW)}-{NEW_REV2[:8]}"
            self.assertIn(f"models/{new_slug}.json (catalogued)", changed)
            document = json.loads(
                (world.root / "models" / f"{new_slug}.json").read_text()
            )
            self.assertEqual(document["source"]["revision"], NEW_REV2)
            # the old curation moved along: the weights are the new snapshot's (two
            # shards, not one), a new metadata file is left out, the README is kept
            self.assertEqual(
                sorted(f["path"] for f in document["files"]),
                [
                    "README.md",
                    "config.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            )
            self.assertEqual(
                {f["path"]: f["sha256"] for f in document["files"]}["README.md"],
                "5" * 64,
            )
            recipe = json.loads((world.root / "recipes/mia.json").read_text())
            primary = next(m for m in recipe["models"] if m["id"] == "primary")
            self.assertEqual(primary["model"]["slug"], new_slug)
            self.assertEqual(
                [f["file_id"] for f in primary["files"]],
                [f["id"] for f in document["files"]],
            )
            world.recipe = recipe
            self.assertEqual(world.check(), [])
            self.assertFalse(
                (
                    world.root / "models" / f"{slug_name(NEW)}-{NEW_REV[:8]}.json"
                ).exists()
            )

    def test_the_file_ids_are_the_ones_the_catalog_tool_gives(self) -> None:
        tree = [
            {"type": "file", "path": path, "size": 3, "lfs": {"oid": "a" * 64}}
            for path in (
                "model-00001-of-00002.safetensors",
                "README.md",
                "THIRD_PARTY_LICENSES/B12X-APACHE-2.0.txt",
                "1.json",
                ".gitattributes",
            )
        ]
        for item in refresh.catalog.inventory("o/n", NEW_REV, tree):
            self.assertEqual(kit_pins.file_id(item["path"], item["sha256"]), item["id"])
            self.assertEqual(refresh.file_id(item["path"], item["sha256"]), item["id"])

    def test_a_revision_moved_within_the_repository_keeps_the_selected_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_files = dict(NEW_FILES)
            models = {
                "primary": model_document(NEW, NEW_REV, old_files),
                "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
            }
            world = World(
                Path(tmp),
                kit_commit=KIT_NEXT,
                models=models,
                locked={"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            )
            # the recipe selects only some files; the new revision changes a README
            recipe = json.loads(json.dumps(world.recipe))
            primary = next(m for m in recipe["models"] if m["id"] == "primary")
            primary["files"] = [
                f for f in primary["files"] if not f["file_id"].startswith("readme")
            ]
            world.recipe = recipe
            (world.root / "recipes/mia.json").write_text(kit_pins.recipe_text(recipe))
            moved = dict(old_files, **{"README.md": ("metadata", "2" * 64)})
            new = model_document(NEW, NEW_REV2, moved)
            world.write_model(new)
            world.lock()
            result = json.loads((world.root / "recipes/mia.json").read_text())
            selected = next(m for m in result["models"] if m["id"] == "primary")
            self.assertEqual(selected["model"]["slug"], new["identity"]["slug"])
            self.assertEqual(
                sorted(f["file_id"] for f in selected["files"]),
                sorted(f["id"] for f in new["files"] if f["path"] != "README.md"),
            )


def refresh_world(
    root: Path,
    kit_pinned: str,
    models: dict[str, dict[str, Any]],
    locked: dict[str, tuple[str, str]],
) -> Any:
    world = World(root, kit_commit=kit_pinned, models=models, locked=locked)
    catalog = refresh.Catalog(
        root,
        recipes={"mia": world.recipe},
        models={m["identity"]["slug"]: m for m in models.values()},
    )
    return world, catalog


def inventory(
    files: dict[str, tuple[str, str]], size: int = 10
) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "sha256": sha,
            "installed_bytes": size,
            "roles": [role],
        }
        for path, (role, sha) in sorted(files.items())
    ]


class HubApi(FakeApi):
    """FakeApi plus the Hugging Face routes the refresh tool's ``Http.json`` serves."""

    def __init__(self, config: dict[str, str], hub: FakeHub) -> None:
        super().__init__(config)
        self.hub = hub

    def json(self, url: str, provider: str, *, anonymous: bool = False) -> Any:
        assert provider == "huggingface" and anonymous, (provider, anonymous)
        return self.hub(url.removeprefix("https://huggingface.co/api/"))


class KitModelAssessmentTests(unittest.TestCase):
    """The scanner follows the kit's model declaration; the newest revision is information."""

    def assess(
        self,
        kit_pinned: str,
        kit_head: str,
        models: dict[str, dict[str, Any]],
        locked: dict[str, tuple[str, str]],
        inventories: dict[tuple[str, str], list[dict[str, Any]]],
        hub: FakeHub | None = None,
    ) -> tuple[Any, Any]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        world, catalog = refresh_world(Path(tmp.name), kit_pinned, models, locked)
        api = HubApi(CONFIGS, hub or hub_for())
        api.compare[(kit_pinned, kit_head)] = {
            "status": "ahead",
            "commits": [],
            "files": [{"filename": "CHANGELOG.md", "status": "modified"}],
            "html_url": "u",
        }
        assessor = assessor_for(catalog, api, kit_head)
        assessor._inventory = lambda repo, revision: inventories[(repo, revision)]  # type: ignore[method-assign]
        with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
            return assessor.assess("mia"), world

    def models(self, repo: str, revision: str, files: dict[str, tuple[str, str]]):
        return {
            "primary": model_document(repo, revision, files),
            "drafter": model_document(DRAFT, DRAFT_REV, DRAFT_FILES),
        }

    def test_a_current_declaration_is_not_drift_and_the_newest_revision_is_information(
        self,
    ) -> None:
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEW,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {},
            hub=hub_for(),
        )
        self.assertFalse(result.drifted, result.reasons)
        self.assertFalse([i for i in result.items if i.kind == "model"])
        self.assertTrue(
            any(
                "information" in e and NEW_REV2[:12] in e and "follows the kit" in e
                for e in result.evidence
            ),
            result.evidence,
        )

    def test_the_newest_revision_of_a_kit_declared_model_is_never_proposed(
        self,
    ) -> None:
        hub = FakeHub(
            {TR3: [TR3_REV], NEW: [NEW_REV, NEW_REV2], DRAFT: [DRAFT_REV]},
            heads={NEW: NEW_REV2},
        )
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEW,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {},
            hub=hub,
        )
        self.assertFalse(result.drifted)
        self.assertFalse(result.items)

    def test_a_kit_that_switches_model_repository_is_a_review(self) -> None:
        result, _ = self.assess(
            KIT_TR3,
            KIT_NEW,
            self.models(TR3, TR3_REV, TR3_FILES),
            {"model": (TR3, TR3_REV), "drafter": (DRAFT, DRAFT_REV)},
            {},
        )
        self.assertFalse(result.mechanical)
        self.assertTrue(
            any(
                "another model repository is a review" in r and NEW in r and TR3 in r
                for r in result.reasons
            ),
            result.reasons,
        )
        self.assertTrue(any("(model) moved" in e for e in result.evidence))

    def test_a_declared_revision_with_the_same_files_is_a_mechanical_move(self) -> None:
        moved = dict(NEW_FILES, **{"README.md": ("metadata", "2" * 64)})
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEXT,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {(NEW, NEW_REV2): inventory(moved)},
        )
        self.assertTrue(result.mechanical, result.reasons)
        new_slug = f"{slug_name(NEW)}-{NEW_REV2[:8]}"
        old_slug = f"{slug_name(NEW)}-{NEW_REV[:8]}"
        self.assertIn(f"models/{new_slug}.json", result.edits)
        self.assertIsNone(result.edits[f"models/{old_slug}.json"])
        recipe = json.loads(result.edits["recipes/mia.json"])
        self.assertEqual(
            next(m for m in recipe["models"] if m["id"] == "primary")["model"]["slug"],
            new_slug,
        )
        lock = json.loads(result.edits["adapters/mia/kit-lock.json"])
        self.assertEqual(lock["dependencies"]["model"]["revision"], NEW_REV2)
        self.assertEqual(lock["dependencies"]["model"]["datasource"], "hf-revision")
        self.assertEqual(lock["kit"]["commit"], KIT_NEXT)
        # the repository's own newest revision played no part: the kit's revision was used
        self.assertEqual(
            [i.target.sha for i in result.items if i.kind == "model"], [NEW_REV2]
        )

    def test_a_declared_revision_with_other_weights_is_a_review(self) -> None:
        changed = dict(
            NEW_FILES, **{"model-00001-of-00001.safetensors": ("weights", "7" * 64)}
        )
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEXT,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {(NEW, NEW_REV2): inventory(changed)},
        )
        self.assertFalse(result.mechanical)
        self.assertTrue(
            any(refresh.KIT_MODEL_FILES_CHANGED in r for r in result.reasons),
            result.reasons,
        )

    def test_a_declared_revision_with_a_changed_configuration_is_a_review(self) -> None:
        changed = dict(NEW_FILES, **{"config.json": ("configuration", "8" * 64)})
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEXT,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {(NEW, NEW_REV2): inventory(changed)},
        )
        self.assertFalse(result.mechanical)

    def test_a_declared_model_that_stopped_being_readable_is_a_review(self) -> None:
        result, _ = self.assess(
            KIT_NEW,
            KIT_NEW,
            self.models(NEW, NEW_REV, NEW_FILES),
            {"model": (NEW, NEW_REV), "drafter": (DRAFT, DRAFT_REV)},
            {},
            hub=hub_for(private=(NEW,)),
        )
        self.assertFalse(result.mechanical)
        self.assertTrue(
            any("cannot be read" in r and "HTTP 401" in r for r in result.reasons),
            result.reasons,
        )


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

IMAGE = "nvcr.io/nvidia/tensorrt-llm/release"
ARM_RC13 = "sha256:" + "4f" * 32
ARM_RC12 = "sha256:" + "82" * 32
INDEX_RC13 = "sha256:" + "2f" * 32
AMD = "sha256:" + "aa" * 32
ATTEST = "sha256:" + "bb" * 32
KIT_RC13 = "5" * 40
KIT_RC12 = "6" * 40
README = {
    KIT_RC13: (
        'export DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"\n'
        "docker run nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13 x\n"
        "# multi-node: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5\n"
    ),
    KIT_RC12: 'export DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"\n',
}
IMAGE_PATTERN = (
    r'^export DOCKER_IMAGE="(?P<repo>nvcr\.io/nvidia/tensorrt-llm/release)'
    r':(?P<version>[^"\s]+)"$'
)
OCI_MANIFEST = {
    "kit": {
        "dependencies": {
            "image": {
                "datasource": "oci-image",
                "file": "README.md",
                "pattern": IMAGE_PATTERN,
            }
        }
    },
    "ours": [],
}


def index(*children: tuple[str, str, str]) -> bytes:
    return json.dumps(
        {
            "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
            "manifests": [
                {"digest": digest, "platform": {"os": os_, "architecture": arch}}
                for digest, os_, arch in children
            ],
        }
    ).encode()


class FakeRegistry:
    """An OCI registry with a bearer-token challenge, serving manifests by tag or digest."""

    def __init__(self, manifests: dict[str, bytes]) -> None:
        self.manifests = manifests
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(
        self, url: str, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((url, headers))
        if url.startswith("https://nvcr.io/proxy_auth"):
            return 200, {}, b'{"token": "t0k"}'
        if headers.get("Authorization") != "Bearer t0k":
            challenge = (
                'Bearer realm="https://nvcr.io/proxy_auth",service="registry",'
                'scope="repository:nvidia/tensorrt-llm/release:pull"'
            )
            return 401, {"www-authenticate": challenge}, b""
        reference = url.rsplit("/manifests/", 1)[1]
        body = self.manifests.get(urllib.parse.unquote(reference))
        if body is None:
            return 404, {}, b""
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        return 200, {"docker-content-digest": digest}, body


def registry_for() -> FakeRegistry:
    multi = index(
        (ARM_RC13, "linux", "arm64"),
        (AMD, "linux", "amd64"),
        (ATTEST, "unknown", "unknown"),
    )
    older = index((ARM_RC12, "linux", "arm64"), (AMD, "linux", "amd64"))
    single = json.dumps({"schemaVersion": 2, "config": {}}).encode()
    return FakeRegistry(
        {
            "1.3.0rc13": multi,
            "1.3.0rc12": older,
            INDEX_RC13: multi,
            ARM_RC13: single,
            "single": single,
        }
    )


class RegistryTests(unittest.TestCase):
    def test_a_tag_resolves_to_the_arm64_manifest_of_its_index(self) -> None:
        registry = kit_pins.Registry(registry_for())
        self.assertEqual(registry.resolve(IMAGE, "1.3.0rc13"), ARM_RC13)
        self.assertEqual(registry.resolve(IMAGE, "1.3.0rc12"), ARM_RC12)

    def test_a_digest_the_kit_names_resolves_to_the_same_arm64_manifest(self) -> None:
        registry = kit_pins.Registry(registry_for())
        self.assertEqual(registry.resolve(IMAGE, INDEX_RC13), ARM_RC13)

    def test_a_single_platform_manifest_is_its_own_digest(self) -> None:
        fake = registry_for()
        digest = "sha256:" + hashlib.sha256(fake.manifests["single"]).hexdigest()
        self.assertEqual(kit_pins.Registry(fake).resolve(IMAGE, "single"), digest)

    def test_the_anonymous_token_is_fetched_once_per_request(self) -> None:
        fake = registry_for()
        kit_pins.Registry(fake).resolve(IMAGE, "1.3.0rc13")
        self.assertEqual(
            [u.split("?")[0] for u, _ in fake.calls],
            [
                f"https://nvcr.io/v2/{IMAGE.removeprefix('nvcr.io/')}/manifests/1.3.0rc13",
                "https://nvcr.io/proxy_auth",
                f"https://nvcr.io/v2/{IMAGE.removeprefix('nvcr.io/')}/manifests/1.3.0rc13",
            ],
        )

    def test_a_missing_tag_and_an_index_without_arm64_are_errors(self) -> None:
        fake = registry_for()
        fake.manifests["amd-only"] = index((AMD, "linux", "amd64"))
        registry = kit_pins.Registry(fake)
        with self.assertRaisesRegex(kit_pins.KitError, "HTTP 404"):
            registry.resolve(IMAGE, "gone")
        with self.assertRaisesRegex(kit_pins.KitError, "no single linux/arm64"):
            registry.resolve(IMAGE, "amd-only")

    def test_docker_hub_names_resolve_to_its_registry(self) -> None:
        self.assertEqual(
            kit_pins.Registry.split("vllm/vllm-openai"),
            ("registry-1.docker.io", "vllm/vllm-openai"),
        )
        self.assertEqual(
            kit_pins.Registry.split("ubuntu"),
            ("registry-1.docker.io", "library/ubuntu"),
        )
        self.assertEqual(
            kit_pins.Registry.split("ghcr.io/ggml-org/llama.cpp"),
            ("ghcr.io", "ggml-org/llama.cpp"),
        )


def image_recipe(kit_commit: str, digest: str) -> dict[str, Any]:
    return {
        "identity": {"publisher": "vonk-forge", "slug": "trt"},
        "provenance": {
            "attribution": ["x"],
            "source_reference": f"https://github.com/{KIT}/tree/{kit_commit}",
        },
        "release": {"version": "1.3.0", "released_at": "2026-10-01"},
        "models": [],
        "execution": {
            "build": {
                "context": {"path": "adapters/trt"},
                "base_image": {
                    "repository": IMAGE,
                    "digest": digest.removeprefix("sha256:"),
                },
            }
        },
    }


class ImageWorld:
    def __init__(
        self,
        root: Path,
        kit_commit: str,
        digest: str,
        *,
        locked: str | None,
        locked_repo: str = IMAGE,
    ) -> None:
        self.root = root
        self.directory = root / "adapters/trt"
        self.directory.mkdir(parents=True)
        (root / "recipes").mkdir()
        self.recipe = image_recipe(kit_commit, digest)
        (root / "recipes/trt.json").write_text(kit_pins.recipe_text(self.recipe))
        (self.directory / "vendored-upstream.json").write_text(json.dumps(OCI_MANIFEST))
        (self.directory / "Dockerfile").write_text(
            f'FROM {IMAGE}@{digest}\nLABEL org.opencontainers.image.version="1.3.0rc13"\n'
            "FROM scratch AS other\n"
        )
        if locked is not None:
            (self.directory / "kit-lock.json").write_text(
                kit_pins.lock_text(
                    {
                        "dependencies": {
                            "image": {
                                "datasource": "oci-image",
                                "repo": locked_repo,
                                "ref": locked,
                                "kind": "tag",
                                "digest": digest,
                            }
                        },
                        "kit": {"commit": kit_commit, "repo": KIT},
                    }
                )
            )

    def source(self, registry: FakeRegistry | None = None) -> Any:
        return upstream(
            FakeApi({}),
            registry=kit_pins.Registry(registry or registry_for()),
        )


class ImageKitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def world(self, kit: str, digest: str, locked: str | None) -> ImageWorld:
        world = ImageWorld(self.root, kit, digest, locked=locked)
        self.api = FakeApi({})
        return world

    def check(self, world: ImageWorld, kit: str) -> list[str]:
        api = _ReadmeApi(README)
        return kit_pins.check_recipe(
            self.root,
            "trt",
            world.recipe,
            upstream(api, registry=kit_pins.Registry(registry_for())),
        )

    def test_the_pattern_picks_the_declared_image_among_several_tags(self) -> None:
        wanted = kit_pins.declarations(OCI_MANIFEST)[0]
        self.assertEqual(kit_pins.parse_repo(README[KIT_RC13], wanted), IMAGE)
        self.assertEqual(kit_pins.parse_declared(README[KIT_RC13], wanted), "1.3.0rc13")

    def test_the_lock_holds_the_arm64_digest_of_the_declared_tag(self) -> None:
        pin = kit_pins.derive(
            KIT,
            KIT_RC13,
            kit_pins.declarations(OCI_MANIFEST),
            upstream(_ReadmeApi(README), registry=kit_pins.Registry(registry_for())),
        )[0]
        self.assertEqual(
            (pin.repo, pin.ref, pin.kind, pin.immutable, pin.datasource),
            (IMAGE, "1.3.0rc13", "tag", ARM_RC13, "oci-image"),
        )
        self.assertEqual(
            kit_pins.lock_document(KIT, KIT_RC13, [pin])["dependencies"]["image"],
            {
                "datasource": "oci-image",
                "repo": IMAGE,
                "ref": "1.3.0rc13",
                "kind": "tag",
                "digest": ARM_RC13,
            },
        )

    def test_a_recipe_on_the_locked_digest_passes_without_asking_the_registry(
        self,
    ) -> None:
        world = self.world(KIT_RC13, ARM_RC13, "1.3.0rc13")
        registry = registry_for()
        api = _ReadmeApi(README)
        problems = kit_pins.check_recipe(
            self.root,
            "trt",
            world.recipe,
            upstream(api, registry=kit_pins.Registry(registry)),
        )
        self.assertEqual(problems, [])
        self.assertEqual(registry.calls, [])

    def test_a_dockerfile_on_another_digest_than_the_lock_is_a_problem(self) -> None:
        world = self.world(KIT_RC13, ARM_RC13, "1.3.0rc13")
        (world.directory / "Dockerfile").write_text(f"FROM {IMAGE}@{ARM_RC12}\n")
        problems = self.check(world, KIT_RC13)
        self.assertTrue(
            any("not the kit's locked image" in p for p in problems), problems
        )

    def test_a_base_image_digest_that_is_not_the_lock_is_a_problem(self) -> None:
        world = self.world(KIT_RC13, ARM_RC13, "1.3.0rc13")
        world.recipe["execution"]["build"]["base_image"]["digest"] = "0" * 64
        problems = self.check(world, KIT_RC13)
        self.assertTrue(any("base_image digest" in p for p in problems), problems)

    def test_a_dockerfile_without_the_kits_image_is_a_problem(self) -> None:
        world = self.world(KIT_RC13, ARM_RC13, "1.3.0rc13")
        (world.directory / "Dockerfile").write_text("FROM scratch\n")
        problems = self.check(world, KIT_RC13)
        self.assertTrue(
            any("no FROM of the kit's image" in p for p in problems), problems
        )

    def test_lock_moves_the_from_and_the_base_image_to_the_tag_the_kit_names(
        self,
    ) -> None:
        world = self.world(KIT_RC12, ARM_RC13, "1.3.0rc13")  # the kit moved to rc12
        changed = kit_pins.apply_lock(
            self.root,
            "trt",
            world.recipe,
            upstream(_ReadmeApi(README), registry=kit_pins.Registry(registry_for())),
        )
        self.assertIn("Dockerfile (FROM digest)", changed)
        text = (world.directory / "Dockerfile").read_text()
        self.assertIn(f"FROM {IMAGE}@{ARM_RC12}\n", text)
        self.assertIn("FROM scratch AS other", text)
        recipe = json.loads((self.root / "recipes/trt.json").read_text())
        self.assertEqual(
            recipe["execution"]["build"]["base_image"]["digest"],
            ARM_RC12.removeprefix("sha256:"),
        )
        world.recipe = recipe
        self.assertEqual(self.check(world, KIT_RC12), [])
        self.assertEqual(
            kit_pins.apply_lock(
                self.root,
                "trt",
                recipe,
                upstream(
                    _ReadmeApi(README), registry=kit_pins.Registry(registry_for())
                ),
            ),
            [],
        )

    def test_a_tag_the_dockerfile_names_keeps_being_named(self) -> None:
        world = self.world(KIT_RC12, ARM_RC13, "1.3.0rc13")
        (world.directory / "Dockerfile").write_text(
            f"FROM {IMAGE}:1.3.0rc13@{ARM_RC13}\n"
        )
        kit_pins.apply_lock(
            self.root,
            "trt",
            world.recipe,
            upstream(_ReadmeApi(README), registry=kit_pins.Registry(registry_for())),
        )
        self.assertEqual(
            (world.directory / "Dockerfile").read_text(),
            f"FROM {IMAGE}:1.3.0rc12@{ARM_RC12}\n",
        )


class _ReadmeApi(FakeApi):
    """FakeApi serving ``README.md`` of a kit commit."""

    def __init__(self, readmes: dict[str, str]) -> None:
        super().__init__({})
        self.readmes = readmes

    def __call__(self, path: str) -> Any:
        import base64

        route = path.split("?")[0].split("/")
        if route[:1] == ["repos"] and route[3] == "contents":
            text = self.readmes.get(path.split("ref=")[1])
            if text is None or "/".join(route[4:]) != "README.md":
                raise RuntimeError(f"HTTP 404 for {path}")
            return {
                "encoding": "base64",
                "content": base64.b64encode(text.encode()).decode(),
            }
        return super().__call__(path)


class ImageAssessmentTests(unittest.TestCase):
    """A kit's image that moves (or a tag that now names another digest) moves the recipe."""

    def assess(
        self,
        kit_pinned: str,
        kit_head: str,
        registry: FakeRegistry,
        locked_repo: str = IMAGE,
    ) -> Any:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        world = ImageWorld(
            root, kit_pinned, ARM_RC13, locked="1.3.0rc13", locked_repo=locked_repo
        )
        catalog = refresh.Catalog(root, recipes={"trt": world.recipe})
        api = _ReadmeApi(README)
        api.compare[(kit_pinned, kit_head)] = {
            "status": "ahead",
            "commits": [],
            "files": [{"filename": "CHANGELOG.md", "status": "modified"}],
            "html_url": "u",
        }
        assessor = refresh.Assessor(catalog, api, None, {})
        assessor.targets[("github", KIT.lower())] = refresh.Target(
            "github", KIT, kit_head, "2026-10-02", None, None, "u"
        )
        with (
            patch.object(refresh, "pin_is_newer_by_date", return_value=False),
            patch.object(
                refresh,
                "kit_upstream",
                lambda http: upstream(api, registry=kit_pins.Registry(registry)),
            ),
        ):
            return assessor.assess("trt")

    def test_a_current_image_declaration_is_not_drift(self) -> None:
        result = self.assess(KIT_RC13, KIT_RC13, registry_for())
        self.assertFalse(result.drifted, result.reasons)

    def test_a_kit_that_names_another_tag_moves_the_image(self) -> None:
        result = self.assess(KIT_RC13, KIT_RC12, registry_for())
        self.assertTrue(result.mechanical, result.reasons)
        self.assertEqual([i.kind for i in result.items if i.kind == "image"], ["image"])
        dockerfile = result.edits["adapters/trt/Dockerfile"].decode()
        self.assertIn(f"{IMAGE}@{ARM_RC12}", dockerfile)
        self.assertNotIn(ARM_RC13, dockerfile)
        lock = json.loads(result.edits["adapters/trt/kit-lock.json"])
        self.assertEqual(lock["dependencies"]["image"]["digest"], ARM_RC12)
        recipe = json.loads(result.edits["recipes/trt.json"])
        self.assertEqual(
            recipe["execution"]["build"]["base_image"]["digest"],
            ARM_RC12.removeprefix("sha256:"),
        )

    def test_a_tag_that_moved_to_a_new_digest_moves_the_image(self) -> None:
        registry = registry_for()
        moved = "sha256:" + "cc" * 32
        registry.manifests["1.3.0rc13"] = index((moved, "linux", "arm64"))
        result = self.assess(KIT_RC13, KIT_RC13, registry)
        self.assertTrue(result.mechanical, result.reasons)
        self.assertIn(moved, result.edits["adapters/trt/Dockerfile"].decode())
        lock = json.loads(result.edits["adapters/trt/kit-lock.json"])
        self.assertEqual(lock["dependencies"]["image"]["digest"], moved)

    def test_another_image_repository_is_a_review(self) -> None:
        registry = registry_for()
        moved = "sha256:" + "dd" * 32
        registry.manifests["1.3.0rc13"] = index((moved, "linux", "arm64"))
        result = self.assess(
            KIT_RC13, KIT_RC13, registry, locked_repo="ghcr.io/other/engine"
        )
        self.assertFalse(result.mechanical)
        self.assertTrue(
            any("another image repository" in r for r in result.reasons), result.reasons
        )


class KitOnlyManifestTests(unittest.TestCase):
    """A manifest that only says where the kit declares things claims no files and no base."""

    def adapter(self, root: Path, manifest: dict[str, Any]) -> Path:
        directory = root / "adapters/mia"
        directory.mkdir(parents=True)
        (directory / "vendored-upstream.json").write_text(json.dumps(manifest))
        (directory / "Dockerfile").write_text(
            'LABEL org.opencontainers.image.source="https://github.com/o/engine" \\\n'
            f'      org.opencontainers.image.revision="{"a" * 40}"\n'
        )
        (directory / "wrapper.py").write_text("print()\n")
        return directory

    def test_a_kit_without_a_base_leaves_the_adapters_own_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self.adapter(Path(tmp), HF_MANIFEST)
            self.assertEqual(
                refresh.vendored._source(directory, HF_MANIFEST), ("o/engine", "a" * 40)
            )
            declared = dict(HF_MANIFEST, source={"repo": KIT, "commit": "b" * 40})
            self.assertEqual(
                refresh.vendored._source(directory, declared), (KIT, "b" * 40)
            )

    def test_a_kit_with_a_base_takes_it_from_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                "kit": {
                    "dependencies": {
                        "base": {
                            "datasource": "git-ref",
                            "repo": BASE,
                            "file": "f",
                            "pattern": "(?P<version>x)",
                        }
                    }
                }
            }
            directory = self.adapter(Path(tmp), manifest)
            self.assertIsNone(refresh.vendored._source(directory, manifest))

    def test_the_manifest_alone_does_not_make_the_adapter_a_vendored_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self.adapter(root, {"kit": HF_MANIFEST["kit"]})
            tree = refresh.vendored.Tree({"README.md": "1" * 40}, True)
            report = refresh.vendored.check_adapter(
                directory, root, lambda _repo, _commit: tree
            )
        self.assertEqual(report.problems, [])
        self.assertIn("no upstream-derived files", report.note)

    def test_a_hand_authored_source_beside_a_model_declaration_is_the_kits_own_pin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_NEW, locked="new")
            manifest = json.loads(
                (world.directory / "vendored-upstream.json").read_text()
            )
            manifest["source"] = {"repo": KIT, "commit": KIT_NEW}
            (world.directory / "vendored-upstream.json").write_text(
                json.dumps(manifest)
            )
            problems = world.check()
        self.assertFalse(any("hand-authored" in p for p in problems), problems)

    def test_the_scanner_follows_the_adapter_source_unless_the_kit_declares_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = self.adapter(root, HF_MANIFEST)
            self.assertFalse(refresh.declares_adapter_source(models))
            self.assertTrue(refresh.whole_kit(models) is True)
            (models / "vendored-upstream.json").write_text(json.dumps(OCI_MANIFEST))
            self.assertTrue(refresh.declares_adapter_source(models))
            (models / "vendored-upstream.json").write_text(
                json.dumps(
                    {
                        "kit": {
                            "whole": False,
                            "dependencies": {
                                "base": {
                                    "datasource": "git-ref",
                                    "repo": BASE,
                                    "file": "f",
                                    "pattern": "(?P<version>x)",
                                }
                            },
                        }
                    }
                )
            )
            self.assertTrue(refresh.declares_adapter_source(models))
            self.assertFalse(refresh.whole_kit(models))
        self.assertFalse(refresh.declares_adapter_source(None))

    def test_a_kit_move_updates_the_lock_even_when_the_adapter_owns_the_source_pin(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            world = tr3_world(Path(tmp), KIT_TR3)
            catalog = refresh.Catalog(world.root, recipes={"mia": world.recipe})
            assessor = refresh.Assessor(catalog, object(), None, {})
            item = refresh.Item(
                "source",
                "github",
                KIT,
                KIT_TR3,
                refresh.Target("github", KIT, KIT_NEW, "2026-10-02", None, None, "u"),
            )
            edits: dict[str, bytes | None] = {}
            assessor._retarget_lock(world.directory, item, edits)
            key = "adapters/mia/kit-lock.json"
            pending = edits[key]
            assert pending is not None
            self.assertEqual(json.loads(pending)["kit"]["commit"], KIT_NEW)
            # a later edit to the same lock (a model moved) merges with it
            lock = assessor._lock(world.directory, edits)
            lock["dependencies"]["model"]["ref"] = "moved"
            assessor._write_lock(world.directory, lock, edits)
            written = edits[key]
            assert written is not None
            merged = json.loads(written)
            self.assertEqual(merged["kit"]["commit"], KIT_NEW)
            self.assertEqual(merged["dependencies"]["model"]["ref"], "moved")


class RecipeCatalogDeclarationTests(unittest.TestCase):
    """Every recipe whose kit declares a dependency has a lock, and the recipe follows it."""

    def test_each_declared_hf_revision_and_image_is_bound_to_its_lock(self) -> None:
        catalog = refresh.Catalog.load(ROOT)
        seen = 0
        for slug, recipe in sorted(catalog.recipes.items()):
            directory = refresh.kit_directory(ROOT, recipe)
            if directory is None:
                continue
            wanted = kit_pins.declarations(
                json.loads((directory / "vendored-upstream.json").read_text())
            )
            lock = kit_pins.read_lock(directory)["dependencies"]
            for declaration in wanted:
                with self.subTest(recipe=slug, dependency=declaration.name):
                    entry = lock[declaration.name]
                    self.assertEqual(entry["datasource"], declaration.datasource)
                    key = kit_pins.IMMUTABLE_KEYS[declaration.datasource]
                    self.assertTrue(entry[key])
                    seen += 1
                    if declaration.datasource == "hf-revision":
                        document = kit_pins.bound_model(
                            ROOT, recipe, declaration.binds
                        )[1]
                        self.assertTrue(
                            kit_pins.model_matches(
                                document, entry["repo"], entry["revision"]
                            )
                        )
                    if declaration.datasource == "oci-image":
                        base = recipe["execution"]["build"]["base_image"]
                        self.assertEqual(
                            base["digest"], entry["digest"].removeprefix("sha256:")
                        )
        self.assertGreater(seen, 0)


if __name__ == "__main__":
    unittest.main()
