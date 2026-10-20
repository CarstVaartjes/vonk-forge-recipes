"""S2 fault matrix: optional failures cannot replace required work or grant deletion."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep
from test_prebuilt_images import INDEX
from test_refresh_upstream import refresh

from spark_sweep.catalog import parse_recipe, recipe_listing
from spark_sweep.image_evidence import load_images
from spark_sweep.policy import classify
from spark_sweep.report import average_test_seconds, build_report, render_report
from spark_sweep.state import State
from spark_sweep.tolerant import InputReason, read_optional
from spark_sweep.vonkctl import Reply, Vonkctl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize(
    "field",
    ["interfaces", "models", "topology", "usage", "model_selectors", "validation"],
)
def test_poison_row_advances_cursor_retains_cache_and_cannot_authorize_removal(
    tmp_path: Path, position: int, field: str
) -> None:
    sweep, fleet, _ = make_sweep(tmp_path, [FakeRecipe("good")], [FakeModel("m1")])
    healthy = fleet._recipe_row(FakeRecipe("good"))
    bad = copy.deepcopy(healthy)
    bad["selector"] = "vonk-forge/poison"
    (bad if field in ("usage", "model_selectors") else bad["document"])[field] = None
    rows = [healthy, healthy]
    rows.insert(position, bad)
    pages = iter(
        [
            {"recipes": rows, "next_cursor": "page2"},
            {"recipes": [healthy]},
        ]
    )

    def runner(argv, timeout):
        return 0, json.dumps(next(pages)), ""

    listing = recipe_listing(Vonkctl("fake", runner=runner))
    cached = parse_recipe(healthy)
    assert cached is not None
    listing.rows["cached"] = cached
    listing.begin(0)
    assert not listing.step() and listing.cursor == "page2"
    assert listing.step(1)
    assert listing.invalid_rows == 1
    assert listing.diagnostics[0].reason is InputReason.INVALID_ROW
    assert "cached" in listing.rows
    sweep.recipe_list = listing
    sweep.model_list.passes = 1
    sweep.model_remover.tick = Mock()
    sweep.remove_finished_models()
    sweep.model_remover.tick.assert_not_called()


@pytest.mark.parametrize("fault", ["mkdir", "write", "command", "nonzero"])
def test_optional_export_preserves_verdict_identity_and_failure_proof(
    tmp_path: Path, fault: str
) -> None:
    sweep, _, _ = make_sweep(tmp_path, [FakeRecipe("bad")], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.state.entry("vonk-forge/bad")["diagnostics"] = "damaged old optional evidence"
    failure = classify("start", "recipe.runtime_exit", "container exited")
    if fault == "mkdir":
        (tmp_path / "evidence").write_text("collision")
        context = patch.object(sweep.vk, "run", wraps=sweep.vk.run)
    elif fault == "write":
        context = patch.object(
            sweep.vk, "run", side_effect=PermissionError("evidence only")
        )
    elif fault == "nonzero":
        context = patch.object(
            sweep.vk, "run", return_value=Reply((), 1, None, "export failed")
        )
    else:
        context = patch.object(
            sweep.vk, "run", side_effect=RuntimeError("export failed")
        )
    with context:
        sweep._record_failure("vonk-forge/bad", failure, "operation-1", final=True)
    entry = sweep.state.entry("vonk-forge/bad")
    assert entry["status"] == "failed"
    assert entry["content_sha256"] == "c1"
    assert entry["evidence"]["operation_id"] == "operation-1"
    assert entry["diagnostics"][0]["reason"] == InputReason.EVIDENCE_UNAVAILABLE
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["evidence"] == entry["evidence"]


@pytest.mark.parametrize("field", ["perf", "timings", "quality"])
@pytest.mark.parametrize(
    "damage", [["damaged"], 3, "bad", {"total_s": "bad", "ttft_ms": "bad"}]
)
def test_damaged_persisted_evidence_preserves_readable_core_and_original_bytes(
    tmp_path: Path, field: str, damage
) -> None:
    state = State(tmp_path / "state.json")
    state.recipes["bad"] = {"status": "passed", "content_sha256": "c1", field: damage}
    state.recipes["good"] = {"status": "passed", "timings": {"total_s": 10}}
    state.save()
    loaded = State.load(state.path)
    before = copy.deepcopy(loaded.data)
    report = build_report(loaded.recipes)
    assert report["summary"] == {"passed": 2}
    assert report["recipes"][0]["content_sha256"] == "c1"
    assert (
        report["recipes"][0]["diagnostics"][0]["reason"] == InputReason.INVALID_OPTIONAL
    )
    assert "bad" in render_report(report)
    assert average_test_seconds(loaded.recipes) == 10
    assert loaded.data == before


@pytest.mark.parametrize(
    "damage",
    [
        "{",
        "null",
        "3",
        '"bad"',
        "[]",
        '{"a":{}}',
        json.dumps({"a" * 64: {"build_key": "b" * 64}}),
        json.dumps({"a" * 64: {"reference": "image:latest", "build_key": "b" * 64}}),
    ],
)
def test_invalid_image_map_keeps_source_build_publication_path(
    tmp_path: Path, damage: str
) -> None:
    path = tmp_path / "images.json"
    path.write_text(damage)
    diagnostics = []
    images = load_images(path, diagnostics.append)
    rows = [{"content_sha256": "a" * 64}]
    INDEX["pin_prebuilt_images"](rows, images)
    if damage != "{":
        INDEX["pin_prebuilt_images"](rows, json.loads(damage))
    assert rows == [{"content_sha256": "a" * 64}]
    assert diagnostics and diagnostics[0].reason is InputReason.INVALID_OPTIONAL


@pytest.mark.parametrize(
    "fault",
    [
        "arm",
        "load",
        "manifest",
        "inventory",
        "heads",
        "assess",
        "save",
        "issues",
        "stale",
        "update",
        "publication",
    ],
)
def test_publication_repair_is_independent_of_assessment_and_housekeeping(
    fault: str,
) -> None:
    render_summary = refresh.write_summary
    names = {
        "arm": (refresh, "arm_open_mechanical_prs"),
        "load": (refresh.Catalog, "load"),
        "manifest": (refresh.drift, "load_manifest"),
        "inventory": (refresh.drift, "embedded_inventory"),
        "heads": (refresh.drift, "observe_embedded_heads"),
        "assess": (refresh.Assessor, "assess"),
        "save": (refresh.Http, "save"),
        "issues": (refresh, "close_current_issues"),
        "stale": (refresh, "close_stale_refresh_prs"),
        "update": (refresh, "update_stale_prs"),
        "publication": (refresh, "publish_if_stale"),
    }
    from contextlib import ExitStack

    with ExitStack() as stack:
        stack.enter_context(patch.object(refresh, "git", return_value="head"))
        stack.enter_context(
            patch.object(refresh.Catalog, "load", return_value=Mock(recipes={"r": {}}))
        )
        stack.enter_context(
            patch.object(refresh.drift, "load_manifest", return_value=({}, {}))
        )
        for name in (
            "discover_watches",
            "embedded_inventory",
            "observe_embedded_heads",
        ):
            stack.enter_context(
                patch.object(refresh.drift, name, return_value={"coverage": {}})
            )
        stack.enter_context(
            patch.object(
                refresh.Assessor,
                "assess",
                return_value=Mock(
                    recipe_id="r", drifted=False, unreachable=[], notes=[]
                ),
            )
        )
        stack.enter_context(patch.object(refresh.Http, "save"))
        for name in (
            "arm_open_mechanical_prs",
            "ensure_labels",
            "foreign_open_prs",
            "sharing_groups",
        ):
            stack.enter_context(patch.object(refresh, name, return_value={}))
        steps = {
            name: stack.enter_context(patch.object(refresh, name))
            for name in (
                "close_current_issues",
                "close_stale_refresh_prs",
                "update_stale_prs",
                "publish_if_stale",
                "write_summary",
            )
        }
        stack.enter_context(
            patch.object(*names[fault], side_effect=RuntimeError("fault injection"))
        )
        # Exercise the real summary renderer even when assessment is unavailable.
        steps["write_summary"].side_effect = render_summary
        # The publication mock must survive the fault patch for call inspection.
        publication = refresh.publish_if_stale
        assert (
            refresh.command_run(argparse.Namespace(cache_dir=None, dry_run=False)) == 1
        )
        publication.assert_called_once_with()
        steps["write_summary"].assert_called_once()
        if fault in {"publication", "save", "issues", "stale", "update", "arm"}:
            assert steps["write_summary"].call_args.args[0].current == 1
        if fault == "issues":
            steps["close_stale_refresh_prs"].assert_called_once()
            steps["update_stale_prs"].assert_called_once()


def test_optional_boundary_never_swallows_interrupts() -> None:
    with pytest.raises(KeyboardInterrupt):
        read_optional(
            "interrupt", lambda: (_ for _ in ()).throw(KeyboardInterrupt()), Mock()
        )


def test_scheduled_publication_recovery_owner_remains_wired() -> None:
    workflow = (ROOT / ".github/workflows/refresh-upstream.yml").read_text()
    assert 'cron: "23 * * * *"' in workflow
    assert "tools/refresh-upstream" in workflow


def bypassed_optional_reads(source: str, fields: set[str]) -> list[int]:
    """Optional fields must be consumed by the shared typed projection."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        key = None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            key = node.args[0].value
        if key in fields and isinstance(node, (ast.Call, ast.Subscript)):
            bad.append(node.lineno)
    return bad


def test_new_optional_report_fields_use_the_shared_typed_reader() -> None:
    from spark_sweep.evidence_contract import ResultEvidence

    fields = set(ResultEvidence.model_fields) - {"core", "diagnostics"}
    assert not bypassed_optional_reads(
        "\n".join(
            ast.unparse(node)
            for node in ast.parse((ROOT / "spark_sweep/report.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("render_")
        ),
        fields,
    )
    for field in fields:
        assert bypassed_optional_reads(f'entry.get("{field}")', fields)
        assert bypassed_optional_reads(f'entry["{field}"]', fields)


def test_optional_validators_and_exports_cannot_bypass_tolerant_boundary() -> None:
    for path in (
        "spark_sweep/evidence_contract.py",
        "spark_sweep/image_evidence.py",
        "spark_sweep/run.py",
        "spark_sweep/catalog.py",
    ):
        tree = ast.parse((ROOT / path).read_text())
        protected = {
            id(node)
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "read_optional"
            for node in ast.walk(call)
        }
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(
                call.func, ast.Attribute
            ):
                continue
            name = ast.unparse(call.func)
            optional = name in {"self._evidence", "self.parse"} or (
                path not in {"spark_sweep/run.py", "spark_sweep/catalog.py"}
                and call.func.attr in {"model_validate", "validate_python", "read_text"}
                and name != "ResultCore.model_validate"
            )
            if optional:
                assert id(call) in protected, (path, call.lineno, name)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_poison_refresh_keeps_a_healthy_lane_and_status_advancing(
    tmp_path: Path, position: int
) -> None:
    from sweep_fakes import Gateway

    gateway = Gateway()
    try:
        sweep, fleet, _ = make_sweep(
            tmp_path, [FakeRecipe("good")], [FakeModel("m1")], gateway=gateway
        )
        real = fleet.runner

        def corrupt(argv, timeout):
            code, stdout, stderr = real(argv, timeout)
            document = json.loads(stdout)
            if isinstance(document, dict) and "recipes" in document:
                bad = copy.deepcopy(fleet._recipe_row(FakeRecipe("good")))
                bad["selector"] = "vonk-forge/poison"
                bad["document"]["interfaces"] = None
                document["recipes"].insert(position, bad)
            return code, json.dumps(document), stderr

        sweep.vk.runner = corrupt
        assert sweep.run() == 0
        assert sweep.state.recipes["vonk-forge/good"]["status"] == "passed"
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["counts"]["passed"] == 1
        assert status["catalog_diagnostics"][0]["reason"] == InputReason.INVALID_ROW
    finally:
        gateway.stop()


def test_bad_image_entry_preserves_valid_sibling_and_release_bytes() -> None:
    digest = "a" * 64
    valid = {
        "reference": "ghcr.io/owner/image@sha256:" + "b" * 64,
        "build_key": "c" * 64,
    }
    rows = [{"content_sha256": digest}, {"content_sha256": "d" * 64}]
    INDEX["pin_prebuilt_images"](
        rows, {digest: valid, "d" * 64: {}, "invalid-key": valid}
    )
    assert rows == [
        {"content_sha256": digest, "prebuilt_image": valid},
        {"content_sha256": "d" * 64},
    ]
    assert INDEX["canonical"]({"recipes": rows})
