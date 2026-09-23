from __future__ import annotations

import ast
import hashlib
import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/moss-vl-realtime"
CONTRACT_PATH = ADAPTER_ROOT / "input_contract.py"
RECIPE_PATH = ROOT / "recipes/moss-vl-realtime-11b-pytorch-single.json"


def _contract_module():
    module = types.ModuleType("moss_input_contract")
    module.__file__ = str(CONTRACT_PATH)
    exec(  # noqa: S102 - load the input contract without importing adapter dependencies.
        compile(CONTRACT_PATH.read_text(encoding="utf-8"), str(CONTRACT_PATH), "exec"),
        module.__dict__,
    )
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class MossRealtimeJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _contract_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.inputs = Path(self.temporary.name)

    def _write_inputs(self) -> None:
        session = self.inputs / "events.json"
        frame = self.inputs / "opening.png"
        session.write_text('{"schema_version":1,"events":[]}', encoding="utf-8")
        frame.write_bytes(b"png-fixture")
        files = [
            {
                "slot": "session",
                "name": session.name,
                "media_type": "application/json",
                "size_bytes": session.stat().st_size,
                "sha256": _digest(session),
            },
            {
                "slot": "frames",
                "name": frame.name,
                "media_type": "image/png",
                "size_bytes": frame.stat().st_size,
                "sha256": _digest(frame),
            },
        ]
        files.sort(key=lambda item: item["name"])
        (self.inputs / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "total_bytes": sum(item["size_bytes"] for item in files),
                    "files": files,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def test_recipe_declares_truthful_typed_slots(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        interface = recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "artifact-job")
        contract = interface["input"]
        self.assertEqual(contract["max_bytes"], 249 * 1024 * 1024)
        slots = {slot["id"]: slot for slot in contract["slots"]}
        self.assertEqual(set(slots), {"frames", "session"})
        self.assertEqual(slots["session"]["max_files"], 1)
        self.assertEqual(slots["session"]["max_file_bytes"], 1024 * 1024)
        self.assertEqual(slots["frames"]["max_files"], 31)
        self.assertEqual(slots["frames"]["max_file_bytes"], 8 * 1024 * 1024)
        outputs = {slot["id"]: slot for slot in interface["output"]["slots"]}
        self.assertEqual(set(outputs), {"replay", "transcript"})
        self.assertEqual(outputs["replay"]["media_types"], ["video/mp4"])
        self.assertEqual(outputs["transcript"]["media_types"], ["application/x-ndjson"])

    def test_transcript_and_notice_follow_current_model_sources(self) -> None:
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        model_slug = recipe["models"][0]["model"]["slug"]
        model = json.loads(
            (ROOT / "models" / f"{model_slug}.json").read_text(encoding="utf-8")
        )
        source_revision = model["source"]["revision"]
        source_url = model["source"]["repository"]
        notice = (ADAPTER_ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(f"{source_url}/tree/{source_revision}", notice)

        qualification = json.loads(
            (ROOT / "qualification/definitions.json").read_text(encoding="utf-8")
        )["recipes"]["vonk-forge/moss-vl-realtime-11b-pytorch-single"]
        expected_revisions = [
            assertion["model_revision"]
            for assertion in qualification["assertions"]
            if assertion.get("kind") == "realtime-transcript"
        ]
        for case in qualification.get("cases", []):
            expected_revisions.extend(
                assertion["model_revision"]
                for assertion in case.get("assertions", [])
                if assertion.get("kind") == "realtime-transcript"
            )
        self.assertTrue(expected_revisions)
        self.assertEqual(len(set(expected_revisions)), 1)
        expected_revision = expected_revisions[0]

        module = ast.parse((ADAPTER_ROOT / "run.py").read_text(encoding="utf-8"))
        start_records = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "record"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "session-start"
        ]
        self.assertEqual(len(start_records), 1)
        revision_values = [
            keyword.value
            for keyword in start_records[0].keywords
            if keyword.arg == "model_revision"
        ]
        self.assertEqual(len(revision_values), 1)
        revision_value = revision_values[0]
        if isinstance(revision_value, ast.Name):
            revision_assignments = [
                ast.literal_eval(node.value)
                for node in module.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == revision_value.id
            ]
            self.assertEqual(len(revision_assignments), 1)
            actual_revision = revision_assignments[0]
        else:
            actual_revision = ast.literal_eval(revision_value)
        self.assertEqual(actual_revision, expected_revision)

    def test_adapter_discovers_arbitrary_names_from_authenticated_slots(self) -> None:
        self._write_inputs()
        session, frames = self.module.resolve_moss_inputs(self.inputs)
        self.assertEqual(session.name, "events.json")
        self.assertEqual(frames, frozenset({"opening.png"}))

    def test_adapter_rejects_unmanifested_and_changed_inputs(self) -> None:
        self._write_inputs()
        (self.inputs / "extra.png").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "outside the authenticated manifest"):
            self.module.resolve_moss_inputs(self.inputs)
        (self.inputs / "extra.png").unlink()
        (self.inputs / "opening.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.module.resolve_moss_inputs(self.inputs)

    def test_signed_source_bundle_matches_recipe_package(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))["execution"][
            "build"
        ]["context"]
        self.assertEqual(context["path"], "adapters/video/moss-vl-realtime")
        self.assertTrue(digest and archive)

        index = json.loads((ROOT / "catalog-index.json").read_text(encoding="utf-8"))
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{RECIPE_PATH.name}"
        )
        self.assertEqual(
            entry["package"]["recipe_content_sha256"], _canonical_digest(RECIPE_PATH)
        )
