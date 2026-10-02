"""Kit-declared base pins (tools/kit-pins) and how the refresh tool uses them.

The fixture is the case that exposed the flaw: MiaAI-Lab's GLM kit declared TensorFold
v0.5.0 at ``ed026ef9`` and declares v0.6.0 from ``v1.3`` on, while TensorFold's own newest
tag moved on to v0.6.2. The recipe pins the kit; the base follows the kit's declaration.
"""

from __future__ import annotations

import base64
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "contracts" / "src"))


def load(name: str, module: str):
    loader = importlib.machinery.SourceFileLoader(module, str(ROOT / "tools" / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = loaded
    loader.exec_module(loaded)
    return loaded


kit_pins = load("kit-pins", "kit_pins_under_test")
refresh = load("refresh-upstream", "refresh_upstream_for_kit_tests")
vendored = refresh.vendored

KIT = "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
BASE = "ashhart/TensorFold"
KIT_OLD = "ed026ef92d1650120dada1294a112acb6c8f2f48"  # declared v0.5.0
KIT_NEW = "4bbf2f3d83f8b5363832368021c5b56b4ced3d92"  # declares v0.6.0
BASE_050 = "9cd52ab4daba68ddd09be89be8f23ad43175e821"
BASE_060 = "c4646171139ee8a3c38103eaa1699dad226ec12b"
BASE_061 = "17c73e189f5e6a5304cda7ea37f086f9c49b4788"
BASE_062 = "6" * 40
PATTERN = r'^TF_VERSION="\$\{TF_VERSION:-(?P<version>[^}"]+)\}"'
CONFIG = {
    KIT_OLD: 'MODEL_ID="x"\nTF_VERSION="${TF_VERSION:-v0.5.0}"\nTF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"\n',
    KIT_NEW: 'MODEL_ID="x"\nTF_VERSION="${TF_VERSION:-v0.6.0}"\nTF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"\n',
}
TAGS = {"v0.5.0": BASE_050, "v0.6.0": BASE_060, "v0.6.1": BASE_061, "v0.6.2": BASE_062}
BRANCH_OLD = "b" * 40
BRANCH_NEW = "c" * 40
MANIFEST = {
    "kit": {
        "dependencies": {
            "base": {
                "repo": BASE,
                "datasource": "git-ref",
                "file": "scripts/config.sh",
                "pattern": PATTERN,
            }
        }
    },
    "ours": ["licenses/", "patches/", "vendor/"],
}


class FakeApi:
    """The GitHub API as far as the kit tools use it."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = CONFIG if config is None else config
        self.branches: dict[str, str] = {}
        self.calls: list[str] = []
        self.compare: dict[tuple[str, str], dict[str, Any]] = {}

    def __call__(self, path: str) -> Any:
        self.calls.append(path)
        route = path.split("?")[0].split("/")
        if route[:1] == ["repos"] and route[3] == "contents":
            commit = path.split("ref=")[1]
            text = self.config.get(commit)
            if text is None or "/".join(route[4:]) != "scripts/config.sh":
                raise RuntimeError(f"HTTP 404 for {path}")
            return {
                "encoding": "base64",
                "content": base64.b64encode(text.encode()).decode(),
            }
        if route[3:6] == ["git", "ref", "heads"]:
            if "/".join(route[6:]) not in self.branches:
                raise RuntimeError(f"HTTP 404 for {path}")
            return {"ref": "refs/heads/" + "/".join(route[6:])}
        if route[3] == "commits":
            ref = "/".join(route[4:])
            commit = self.branches.get(ref) or TAGS.get(ref, ref)
            return {
                "sha": commit,
                "html_url": f"https://github.com/{route[1]}/{route[2]}/commit/{commit}",
                "commit": {"committer": {"date": "2026-10-01T12:00:00Z"}},
            }
        if route[3] == "tags":
            return [
                {"name": name, "commit": {"sha": sha}} for name, sha in TAGS.items()
            ]
        if route[3] == "compare":
            old, new = route[4].split("...")
            return self.compare[(old, new)]
        raise RuntimeError(f"unrouted {path}")

    # the refresh tool's Http face
    def github(self, path: str) -> Any:
        return self(path)


def declaration(**override: Any) -> Any:
    entry = dict(MANIFEST["kit"]["dependencies"]["base"], **override)
    return kit_pins.declarations({"kit": {"dependencies": {"base": entry}}})[0]


class DeclarationTests(unittest.TestCase):
    def test_the_mia_declaration_reads_both_kit_versions(self) -> None:
        wanted = declaration()
        self.assertEqual(kit_pins.parse_declared(CONFIG[KIT_OLD], wanted), "v0.5.0")
        self.assertEqual(kit_pins.parse_declared(CONFIG[KIT_NEW], wanted), "v0.6.0")

    def test_nothing_matching_or_several_values_is_an_error(self) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "nothing in scripts/config.sh"):
            kit_pins.parse_declared("TF_VERSION=none\n", declaration())
        text = CONFIG[KIT_OLD] + CONFIG[KIT_NEW]
        with self.assertRaisesRegex(kit_pins.KitError, "several values"):
            kit_pins.parse_declared(text, declaration())
        # the same value twice is still one declaration
        kit_pins.parse_declared(CONFIG[KIT_OLD] * 2, declaration())

    def test_a_pattern_needs_the_version_group_and_a_known_datasource(self) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "version"):
            declaration(pattern="^TF_VERSION=(.*)$")
        with self.assertRaisesRegex(kit_pins.KitError, "datasource"):
            declaration(datasource="oci-image")
        with self.assertRaisesRegex(kit_pins.KitError, "lacks"):
            kit_pins.declarations({"kit": {"dependencies": {"base": {"repo": BASE}}}})

    def test_an_override_needs_a_reason(self) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "override needs"):
            declaration(override={"ref": "v0.6.1"})
        self.assertIsNotNone(declaration(override={"ref": "v0.6.1", "reason": "x"}))

    def test_a_manifest_without_a_kit_block_declares_nothing(self) -> None:
        self.assertEqual(kit_pins.declarations({"ours": []}), [])


class DerivationTests(unittest.TestCase):
    def derive(self, commit: str, **override: Any) -> Any:
        return kit_pins.derive(
            KIT, commit, [declaration(**override)], kit_pins.GitHub(FakeApi())
        )[0]

    def test_mia_v050_to_v060_follows_the_kit_not_the_newest_tag(self) -> None:
        old, new = self.derive(KIT_OLD), self.derive(KIT_NEW)
        self.assertEqual((old.ref, old.commit), ("v0.5.0", BASE_050))
        self.assertEqual((new.ref, new.commit), ("v0.6.0", BASE_060))
        self.assertEqual(new.version, "0.6.0")
        self.assertNotIn(new.commit, {BASE_061, BASE_062})
        self.assertEqual(new.date, "2026-10-01")

    def test_the_newest_release_is_information_only(self) -> None:
        self.assertEqual(
            kit_pins.newest_release(kit_pins.GitHub(FakeApi()), BASE), "v0.6.2"
        )

    def test_an_override_replaces_the_declaration_and_is_recorded(self) -> None:
        pin = self.derive(KIT_NEW, override={"ref": "v0.6.1", "reason": "fix"})
        self.assertEqual((pin.commit, pin.overridden), (BASE_061, "fix"))
        lock = kit_pins.lock_document(KIT, KIT_NEW, [pin])
        self.assertEqual(lock["dependencies"]["base"]["overridden"], "fix")

    def test_an_unreadable_kit_file_names_the_commit(self) -> None:
        with self.assertRaisesRegex(kit_pins.KitError, "cannot read"):
            kit_pins.derive(KIT, "f" * 40, [declaration()], kit_pins.GitHub(FakeApi()))

    def test_the_lock_is_deterministic(self) -> None:
        pin = self.derive(KIT_NEW)
        text = kit_pins.lock_text(kit_pins.lock_document(KIT, KIT_NEW, [pin]))
        self.assertEqual(
            json.loads(text),
            {
                "dependencies": {
                    "base": {
                        "commit": BASE_060,
                        "kind": "tag",
                        "ref": "v0.6.0",
                        "repo": BASE,
                    }
                },
                "kit": {"commit": KIT_NEW, "repo": KIT},
            },
        )
        self.assertEqual(
            text, json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
        )


KIT_BRANCH = "d" * 40  # declares the branch glm-long-context
CONFIG_BRANCH = {
    KIT_BRANCH: "TF_REF=glm-long-context\n",
}
BRANCH_PATTERN = r"^TF_REF=(?P<version>\S+)$"
BRANCH_MANIFEST = {
    "kit": {
        "dependencies": {
            "base": {
                "repo": "taussoe/TensorFold",
                "datasource": "git-ref",
                "file": "scripts/config.sh",
                "pattern": BRANCH_PATTERN,
            }
        }
    },
    "ours": ["licenses/", "patches/", "vendor/"],
}


class BranchDeclarationTests(unittest.TestCase):
    """A kit that declares a branch is followed like a Nix branch input."""

    def api(self, head: str = BRANCH_OLD) -> FakeApi:
        api = FakeApi(CONFIG_BRANCH)
        api.branches["glm-long-context"] = head
        return api

    def derive(self, api: FakeApi) -> Any:
        wanted = kit_pins.declarations(BRANCH_MANIFEST)
        return kit_pins.derive(KIT, KIT_BRANCH, wanted, kit_pins.GitHub(api))[0]

    def test_a_declared_branch_is_resolved_to_its_head_commit_and_locked_as_a_branch(
        self,
    ) -> None:
        pin = self.derive(self.api())
        self.assertEqual(
            (pin.ref, pin.kind, pin.commit),
            ("glm-long-context", "branch", BRANCH_OLD),
        )
        lock = kit_pins.lock_document(KIT, KIT_BRANCH, [pin])
        self.assertEqual(
            lock["dependencies"]["base"],
            {
                "repo": "taussoe/TensorFold",
                "ref": "glm-long-context",
                "kind": "branch",
                "commit": BRANCH_OLD,
            },
        )

    def test_tags_and_commits_keep_their_own_kind(self) -> None:
        source = kit_pins.GitHub(FakeApi())
        self.assertEqual(source.kind(BASE, "v0.6.0"), "tag")
        self.assertEqual(source.kind(BASE, "191188075bca"), "commit")
        self.assertEqual(source.kind(BASE, BASE_060), "commit")

    def test_a_branch_with_a_slash_in_its_name_is_a_branch(self) -> None:
        api = FakeApi()
        api.branches["integration/0.6.1"] = BRANCH_OLD
        self.assertEqual(kit_pins.GitHub(api).kind(BASE, "integration/0.6.1"), "branch")

    def test_a_moved_branch_does_not_make_the_committed_lock_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_adapter(
                root,
                kit_commit=KIT_BRANCH,
                base_commit=BRANCH_OLD,
                base_ref="glm-long-context",
                base_kind="branch",
                manifest=BRANCH_MANIFEST,
                base_repo="taussoe/TensorFold",
            )
            api = self.api(BRANCH_NEW)
            problems = kit_pins.check_recipe(
                root, "mia", recipe_for(KIT_BRANCH), kit_pins.GitHub(api)
            )
        self.assertEqual([p for p in problems if "stale" in p], [])
        self.assertFalse(any("heads" in call for call in api.calls), api.calls)


def recipe_for(kit_commit: str, context: str = "adapters/mia") -> dict[str, Any]:
    return {
        "identity": {"slug": "mia"},
        "provenance": {
            "attribution": ["x"],
            "source_reference": f"https://github.com/{KIT}/tree/{kit_commit}",
        },
        "release": {"version": "0.5.0", "released_at": "2026-10-01"},
        "models": [],
        "execution": {"build": {"context": {"path": context}}},
    }


def write_adapter(
    root: Path,
    *,
    kit_commit: str,
    base_commit: str,
    base_ref: str,
    archive: bytes = b"archive",
    manifest: dict[str, Any] | None = None,
    lock: bool = True,
    dockerfile_commit: str | None = None,
    base_kind: str = "tag",
    base_repo: str = BASE,
) -> Path:
    import hashlib

    directory = root / "adapters/mia"
    (directory / "vendor").mkdir(parents=True)
    (directory / "patches").mkdir()
    (directory / "patches/0001-a.patch").write_text("--- a\n+++ b\n")
    (directory / "tensorfold-serve").write_text("#!/usr/bin/env python3\n")
    (directory / "vendored-upstream.json").write_text(
        json.dumps(manifest if manifest is not None else MANIFEST)
    )
    name = f"tensorfold-{base_commit}.tar.gz"
    (directory / "vendor" / name).write_bytes(archive)
    digest = hashlib.sha256(archive).hexdigest()
    cited = dockerfile_commit or base_commit
    (directory / "Dockerfile").write_text(
        f'LABEL org.opencontainers.image.revision="{cited}" \\\n'
        f'      io.vonk.recipe-patches.source="https://github.com/{KIT}/tree/{kit_commit}"\n'
        f"COPY vendor/{name} /tmp/tensorfold.tar.gz\n"
        f'RUN echo "{digest}  /tmp/tensorfold.tar.gz" | sha256sum -c -\n'
    )
    if lock:
        (directory / "kit-lock.json").write_text(
            kit_pins.lock_text(
                {
                    "dependencies": {
                        "base": {
                            "commit": base_commit,
                            "kind": base_kind,
                            "ref": base_ref,
                            "repo": base_repo,
                        }
                    },
                    "kit": {"commit": kit_commit, "repo": KIT},
                }
            )
        )
    return directory


class LockCheckTests(unittest.TestCase):
    def check(self, root: Path, kit_commit: str) -> list[str]:
        return kit_pins.check_recipe(
            root, "mia", recipe_for(kit_commit), kit_pins.GitHub(FakeApi())
        )

    def test_a_current_lock_passes_and_costs_one_kit_file_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp), kit_commit=KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            api = FakeApi()
            self.assertEqual(
                kit_pins.check_recipe(
                    Path(tmp), "mia", recipe_for(KIT_NEW), kit_pins.GitHub(api)
                ),
                [],
            )
            self.assertEqual(
                [c for c in api.calls if "/commits/" in c or "/tags" in c], []
            )  # no tag lookups: the lock's commit is trusted, the declaration is re-read

    def test_a_kit_that_moved_its_base_makes_the_lock_stale(self) -> None:
        # The recipe's kit pin moved to v1.3 (base v0.6.0); the lock still says v0.5.0.
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp), kit_commit=KIT_OLD, base_commit=BASE_050, base_ref="v0.5.0"
            )
            problems = self.check(Path(tmp), KIT_NEW)
        self.assertTrue(any("kit-lock.json is stale" in p for p in problems), problems)
        self.assertTrue(any("tools/kit-pins lock mia" in p for p in problems))

    def test_a_hand_authored_pin_beside_the_declaration_is_rejected(self) -> None:
        manifest = dict(MANIFEST, source={"repo": BASE, "commit": BASE_050})
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp),
                kit_commit=KIT_NEW,
                base_commit=BASE_060,
                base_ref="v0.6.0",
                manifest=manifest,
            )
            problems = self.check(Path(tmp), KIT_NEW)
        self.assertTrue(
            any("hand-authored 'source' pin" in p for p in problems), problems
        )

    def test_consumers_must_follow_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp),
                kit_commit=KIT_NEW,
                base_commit=BASE_060,
                base_ref="v0.6.0",
                dockerfile_commit=BASE_050,
            )
            problems = self.check(Path(tmp), KIT_NEW)
        self.assertTrue(any("image revision label" in p for p in problems), problems)
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_adapter(
                Path(tmp), kit_commit=KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            (directory / "vendor" / f"tensorfold-{BASE_060}.tar.gz").write_bytes(
                b"edited"
            )
            problems = self.check(Path(tmp), KIT_NEW)
        self.assertTrue(any("does not verify" in p for p in problems), problems)

    def test_a_missing_lock_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp),
                kit_commit=KIT_NEW,
                base_commit=BASE_060,
                base_ref="v0.6.0",
                lock=False,
            )
            self.assertTrue(
                any("no kit-lock.json" in p for p in self.check(Path(tmp), KIT_NEW))
            )

    def test_a_recipe_without_a_declaration_is_not_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(
                Path(tmp),
                kit_commit=KIT_NEW,
                base_commit=BASE_060,
                base_ref="v0.6.0",
                manifest={"source": {"repo": BASE, "commit": BASE_060}},
                lock=False,
            )
            self.assertEqual(self.check(Path(tmp), KIT_NEW), [])


class ApplyLockTests(unittest.TestCase):
    def test_moving_the_kit_moves_the_lock_the_archive_and_the_dockerfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = write_adapter(
                root,
                kit_commit=KIT_OLD,
                base_commit=BASE_050,
                base_ref="v0.5.0",
                manifest=dict(MANIFEST, source={"repo": BASE, "commit": BASE_050}),
                lock=False,
            )
            fetched: list[tuple[str, str]] = []

            def fetch(repo: str, commit: str) -> bytes:
                fetched.append((repo, commit))
                return b"new archive"

            changed = kit_pins.apply_lock(
                root, "mia", recipe_for(KIT_NEW), kit_pins.GitHub(FakeApi()), fetch
            )
            self.assertEqual(fetched, [(BASE, BASE_060)])
            self.assertTrue(any(c.startswith("kit-lock.json") for c in changed))
            lock = json.loads((directory / "kit-lock.json").read_text())
            self.assertEqual(lock["dependencies"]["base"]["commit"], BASE_060)
            self.assertNotIn(
                "source", json.loads((directory / "vendored-upstream.json").read_text())
            )
            self.assertEqual(
                [p.name for p in (directory / "vendor").iterdir()],
                [f"tensorfold-{BASE_060}.tar.gz"],
            )
            text = (directory / "Dockerfile").read_text()
            self.assertNotIn(BASE_050, text)
            self.assertNotIn(KIT_OLD, text)
            self.assertIn(KIT_NEW, text)
            # the result is exactly what the lock check asks for, and a second run is a no-op
            self.assertEqual(
                kit_pins.check_recipe(
                    root, "mia", recipe_for(KIT_NEW), kit_pins.GitHub(FakeApi())
                ),
                [],
            )
            self.assertEqual(
                kit_pins.apply_lock(
                    root, "mia", recipe_for(KIT_NEW), kit_pins.GitHub(FakeApi()), fetch
                ),
                [],
            )
            self.assertEqual(len(fetched), 1)

    def test_the_manifest_reader_of_the_base_follows_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_adapter(
                Path(tmp), kit_commit=KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            manifest = json.loads((directory / "vendored-upstream.json").read_text())
            self.assertEqual(vendored._source(directory, manifest), (BASE, BASE_060))
            (directory / "kit-lock.json").unlink()
            self.assertIsNone(vendored._source(directory, manifest))


def comparison(*names: str, status: str = "ahead", truncated: bool = False) -> Any:
    return refresh.Comparison(
        status,
        [],
        [{"filename": n, "status": "modified"} for n in names],
        truncated,
        "u",
    )


class KitChangeClassificationTests(unittest.TestCase):
    def judge(
        self, *names: str, mirrored: frozenset[str] = frozenset(), **kwargs: Any
    ) -> list[str]:
        return refresh.classify_changes(
            comparison(*names, **kwargs),
            patched=set(),
            vendor_problems=[],
            used_changes=None,
            mirrored=mirrored,
        )

    def test_docs_and_new_client_helpers_are_not_runtime(self) -> None:
        self.assertEqual(
            self.judge(
                "README.md",
                "CHANGELOG.md",
                "CREDITS.md",
                "NOTICE",
                "tools/end_of_turn.py",
            ),
            [],
        )

    def test_patches_launch_scripts_and_configuration_are_runtime(self) -> None:
        for name in (
            "patches/0054-new.patch",
            "scripts/config.sh",
            "start.sh",
            "scripts/nodes.sh",
            "Dockerfile",
            "config.yaml",
        ):
            with self.subTest(name=name):
                self.assertTrue(self.judge(name))

    def test_a_file_the_adapter_copies_is_runtime_whatever_it_is_called(self) -> None:
        self.assertEqual(self.judge("tools/data.bin"), [])
        self.assertTrue(
            self.judge("tools/data.bin", mirrored=frozenset({"tools/data.bin"}))
        )

    def test_truncated_or_diverged_comparisons_are_not_trusted(self) -> None:
        self.assertTrue(self.judge("README.md", truncated=True))
        self.assertTrue(self.judge("README.md", status="diverged"))


def kit_catalog(root: Path, kit_commit: str, **adapter: Any) -> Any:
    write_adapter(root, kit_commit=kit_commit, **adapter)
    recipe = recipe_for(kit_commit)
    return refresh.Catalog(root, recipes={"mia": recipe})


def assessor_for(catalog: Any, api: FakeApi, kit_head: str) -> Any:
    assessor = refresh.Assessor(catalog, api, None, {})
    assessor.targets[("github", KIT.lower())] = refresh.Target(
        "github",
        KIT,
        kit_head,
        "2026-10-02",
        None,
        None,
        "u",
    )

    # the base repository's own newest release must never be consulted as a pin
    def refuse(provider: str, repo: str) -> Any:
        raise AssertionError(f"{repo} was resolved independently")

    original = assessor.target

    def target(provider: str, repo: str) -> Any:
        if repo.lower() == BASE.lower():
            refuse(provider, repo)
        return original(provider, repo)

    assessor.target = target  # type: ignore[method-assign]
    return assessor


class AssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def assess(
        self,
        kit_pinned: str,
        kit_head: str,
        *,
        base: tuple[str, str],
        files: tuple[str, ...] = (),
    ) -> Any:
        catalog = kit_catalog(
            self.root, kit_pinned, base_commit=base[1], base_ref=base[0]
        )
        api = FakeApi()
        api.compare[(kit_pinned, kit_head)] = {
            "status": "ahead",
            "commits": [],
            "files": [{"filename": n, "status": "modified"} for n in files],
            "html_url": "u",
        }
        with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
            return assessor_for(catalog, api, kit_head).assess("mia")

    def test_a_current_kit_with_its_declared_base_is_not_drift_even_though_the_base_has_a_newer_release(
        self,
    ) -> None:
        result = self.assess(KIT_NEW, KIT_NEW, base=("v0.6.0", BASE_060))
        self.assertFalse(result.drifted)
        self.assertTrue(
            any("v0.6.2" in e and "information" in e for e in result.evidence)
        )

    def test_the_v050_to_v060_move_is_a_review_naming_the_declared_base(self) -> None:
        result = self.assess(
            KIT_OLD, KIT_NEW, base=("v0.5.0", BASE_050), files=("patches/0002.patch",)
        )
        self.assertTrue(result.drifted)
        self.assertFalse(result.mechanical)
        self.assertEqual(
            [i.kind for i in result.items], ["source"]
        )  # no adapter candidate
        self.assertTrue(
            any(
                "kit declares" in r and "v0.6.0" in r and "v0.5.0" in r
                for r in result.reasons
            ),
            result.reasons,
        )
        self.assertFalse(any("v0.6.2" in r or "0.6.1" in r for r in result.reasons))

    def test_a_docs_and_tools_only_kit_move_is_mechanical_and_moves_every_citation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = kit_catalog(
                root, KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            newer = "5" * 40
            api = FakeApi(dict(CONFIG, **{newer: CONFIG[KIT_NEW]}))
            api.compare[(KIT_NEW, newer)] = {
                "status": "ahead",
                "commits": [],
                "html_url": "u",
                "files": [
                    {"filename": n, "status": "modified"}
                    for n in ("README.md", "tools/end_of_turn.py", "CHANGELOG.md")
                ],
            }
            with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
                result = assessor_for(catalog, api, newer).assess("mia")
            self.assertTrue(result.mechanical, result.reasons)
            self.assertIn("adapters/mia/Dockerfile", result.edits)
            self.assertIn(newer, result.edits["adapters/mia/Dockerfile"].decode())
            self.assertNotIn(KIT_NEW, result.edits["adapters/mia/Dockerfile"].decode())
            lock = json.loads(result.edits["adapters/mia/kit-lock.json"])
            self.assertEqual(lock["kit"]["commit"], newer)
            self.assertEqual(lock["dependencies"]["base"]["commit"], BASE_060)
            recipe = json.loads(result.edits["recipes/mia.json"])
            self.assertTrue(recipe["provenance"]["source_reference"].endswith(newer))

    def test_a_kit_move_that_changes_patches_or_launch_configuration_is_a_review(
        self,
    ) -> None:
        for name in ("patches/0054-new.patch", "scripts/config.sh"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                catalog = kit_catalog(
                    root, KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
                )
                newer = "5" * 40
                api = FakeApi(dict(CONFIG, **{newer: CONFIG[KIT_NEW]}))
                api.compare[(KIT_NEW, newer)] = {
                    "status": "ahead",
                    "commits": [],
                    "html_url": "u",
                    "files": [{"filename": name, "status": "modified"}],
                }
                with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
                    result = assessor_for(catalog, api, newer).assess("mia")
                self.assertFalse(result.mechanical)
                self.assertTrue(any(name in r for r in result.reasons), result.reasons)

    def test_a_file_the_adapter_copies_from_the_kit_is_a_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = kit_catalog(
                root, KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            newer = "5" * 40
            api = FakeApi(dict(CONFIG, **{newer: CONFIG[KIT_NEW]}))
            api.compare[(KIT_NEW, newer)] = {
                "status": "ahead",
                "commits": [],
                "html_url": "u",
                "files": [{"filename": "tensorfold-serve", "status": "modified"}],
            }
            with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
                result = assessor_for(catalog, api, newer).assess("mia")
            self.assertTrue(
                any("tensorfold-serve" in r for r in result.reasons), result.reasons
            )

    def test_an_overridden_base_is_reported_not_changed(self) -> None:
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["kit"]["dependencies"]["base"]["override"] = {
            "ref": "v0.6.1",
            "reason": "security fix",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = kit_catalog(
                root,
                KIT_NEW,
                base_commit=BASE_061,
                base_ref="v0.6.1",
                manifest=manifest,
            )
            result = assessor_for(catalog, FakeApi(), KIT_NEW).assess("mia")
        self.assertFalse(result.drifted)
        self.assertTrue(
            any("overridden" in e and "security fix" in e for e in result.evidence)
        )

    def test_a_kit_whose_declaration_stops_matching_is_a_review_not_a_guess(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = kit_catalog(
                root, KIT_NEW, base_commit=BASE_060, base_ref="v0.6.0"
            )
            newer = "5" * 40
            api = FakeApi(dict(CONFIG, **{newer: "nothing here\n"}))
            api.compare[(KIT_NEW, newer)] = {
                "status": "ahead",
                "commits": [],
                "html_url": "u",
                "files": [],
            }
            with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
                result = assessor_for(catalog, api, newer).assess("mia")
        self.assertFalse(result.mechanical)
        self.assertTrue(
            any("cannot be read" in r for r in result.reasons), result.reasons
        )


class BranchAssessmentTests(unittest.TestCase):
    def assess(self, locked: str, head: str) -> Any:
        api = FakeApi(CONFIG_BRANCH)
        api.branches["glm-long-context"] = head
        with tempfile.TemporaryDirectory() as tmp:
            catalog = kit_catalog(
                Path(tmp),
                KIT_BRANCH,
                base_commit=locked,
                base_ref="glm-long-context",
                base_kind="branch",
                manifest=BRANCH_MANIFEST,
                base_repo="taussoe/TensorFold",
            )
            return assessor_for(catalog, api, KIT_BRANCH).assess("mia")

    def test_a_branch_at_its_locked_commit_is_not_drift(self) -> None:
        result = self.assess(BRANCH_OLD, BRANCH_OLD)
        self.assertFalse(result.drifted)

    def test_a_moved_branch_head_is_a_kit_dependency_change_for_review(self) -> None:
        result = self.assess(BRANCH_OLD, BRANCH_NEW)
        self.assertTrue(result.drifted)
        self.assertFalse(result.mechanical)
        self.assertEqual(result.items, [])
        self.assertTrue(
            any(
                "branch `glm-long-context`" in r
                and BRANCH_OLD[:12] in r
                and BRANCH_NEW[:12] in r
                for r in result.reasons
            ),
            result.reasons,
        )


class CitedKitTests(unittest.TestCase):
    """A kit that declares nothing but is cited by the adapter is judged all the same."""

    def test_the_cited_kit_is_judged_by_what_changed(self) -> None:
        for names, mechanical in (
            (("README.md", "tools/needle.py"), True),
            (("patches/0001-a.patch",), False),
        ):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_adapter(
                    root,
                    kit_commit=KIT_NEW,
                    base_commit=BASE_060,
                    base_ref="v0.6.0",
                    manifest={"source": {"repo": BASE, "commit": BASE_060}},
                    lock=False,
                )
                catalog = refresh.Catalog(root, recipes={"mia": recipe_for(KIT_NEW)})
                newer = "5" * 40
                api = FakeApi()
                api.compare[(KIT_NEW, newer)] = {
                    "status": "ahead",
                    "commits": [],
                    "html_url": "u",
                    "files": [{"filename": n, "status": "modified"} for n in names],
                }
                assessor = refresh.Assessor(catalog, api, None, {})
                assessor.targets[("github", KIT.lower())] = refresh.Target(
                    "github", KIT, newer, "2026-10-02", None, None, "u"
                )
                assessor.targets[("github", BASE.lower())] = refresh.Target(
                    "github", BASE, BASE_060, "2026-10-01", "v0.6.0", "0.6.0", "u"
                )
                with patch.object(refresh, "pin_is_newer_by_date", return_value=False):
                    result = assessor.assess("mia")
                self.assertEqual(result.mechanical, mechanical, result.reasons)


class ModelPayloadTests(unittest.TestCase):
    def test_a_readme_only_revision_is_not_a_move(self) -> None:
        old = {
            "files": [
                {"path": "model.safetensors", "roles": ["weights"], "sha256": "a" * 64},
                {"path": "README.md", "roles": ["metadata"], "sha256": "b" * 64},
            ]
        }
        same = [
            {"path": "model.safetensors", "roles": ["weights"], "sha256": "a" * 64},
            {"path": "README.md", "roles": ["metadata"], "sha256": "c" * 64},
        ]
        changed = [dict(same[0], sha256="d" * 64), same[1]]
        self.assertTrue(refresh.payload_identical(old, same))
        self.assertFalse(refresh.payload_identical(old, changed))


def issue_assessment(heads: list[str]) -> Any:
    result = refresh.Assessment("mia")
    for head in heads:
        result.items.append(
            refresh.Item(
                "source",
                "github",
                KIT,
                KIT_OLD,
                refresh.Target(
                    "github",
                    KIT,
                    head,
                    "2026-10-02",
                    None,
                    None,
                    "u",
                    committed_at="2026-10-02T10:17:45Z",
                ),
            )
        )
    result.reasons.append("something changed")
    return result


class IssueFreshnessTests(unittest.TestCase):
    def test_the_body_records_the_head_and_its_timestamp(self) -> None:
        body = refresh.issue_body(issue_assessment([KIT_NEW]))
        self.assertIn(f"head `{KIT_NEW[:12]}`, committed 2026-10-02T10:17:45Z", body)
        self.assertNotIn("Upstream moved", body)

    def test_a_later_run_says_when_upstream_moved_and_keeps_saying_it(self) -> None:
        first = refresh.issue_body(issue_assessment([KIT_NEW]))
        second = refresh.issue_body(issue_assessment(["5" * 40]), first)
        self.assertIn(
            f"> Upstream moved since the last check: {KIT}@{KIT_NEW[:12]} -> {KIT}@{'5' * 12}",
            second,
        )
        third = refresh.issue_body(issue_assessment(["5" * 40]), second)
        self.assertIn("Upstream moved since the last check", third)
        self.assertEqual(
            refresh.MOVED_LINE.search(third).group(1),
            refresh.MOVED_LINE.search(second).group(1),
        )

    def test_an_unchanged_upstream_keeps_the_body_stable(self) -> None:
        first = refresh.issue_body(issue_assessment([KIT_NEW]))
        self.assertEqual(refresh.issue_body(issue_assessment([KIT_NEW]), first), first)


FOLLOWING_RECIPES = {
    "glm-5-3-flash-exl3-dflash2-tensorfold-mia-dual": ("tag", "v0.6.0"),
    "qwen3-8-flash-next-tensorfold-single": ("tag", "v0.6.1"),
    "qwen3-8-27b-taussoe-tensorfold-single": ("commit", "191188075bca"),
    "swift-1-5-flash-next-nvfp4-tournierjc-tensorfold-single": (
        "commit",
        "191188075bca",
    ),
    "qwen3-8-flash-next-taussoe-tensorfold-single": (
        "branch",
        "glm-long-context",
    ),
}


class CatalogKitTests(unittest.TestCase):
    """Recipes whose kit declares its base follow the declaration, not the newest tag."""

    def test_each_listed_recipe_declares_and_locks_its_base_without_a_second_pin(
        self,
    ) -> None:
        catalog = refresh.Catalog.load(ROOT)
        for slug, (kind, ref) in FOLLOWING_RECIPES.items():
            with self.subTest(recipe=slug):
                directory = refresh.kit_directory(ROOT, catalog.recipes[slug])
                self.assertIsNotNone(directory)  # so consider("adapter") is never asked
                manifest = json.loads(
                    (directory / "vendored-upstream.json").read_text()
                )
                self.assertNotIn("source", manifest)
                locked = kit_pins.read_lock(directory)["dependencies"]["base"]
                self.assertEqual(locked["kind"], kind)
                self.assertTrue(locked["ref"].startswith(ref), locked)
                self.assertRegex(locked["commit"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
