"""The command line, the plan, the report, the evidence files and the issue filing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sweep_fakes import (
    GB,
    FakeClock,
    FakeFleet,
    FakeModel,
    FakeRecipe,
    Gateway,
    make_definitions,
)

from spark_sweep.cli import main
from spark_sweep.issues import LABEL, file_issues, title_for
from spark_sweep.report import export_evidence
from spark_sweep.state import ResultsLog
from spark_sweep.vonkctl import Vonkctl

ROOT = Path(__file__).resolve().parents[1]


def _fleet(recipes, models, root="http://127.0.0.1:1"):
    return FakeFleet(FakeClock(), recipes, models, root)


def _run(argv, fleet, recipes):
    return main(
        argv,
        vonkctl_factory=lambda exe: Vonkctl(exe, runner=fleet.runner),
        clock=fleet.clock,
        definitions=make_definitions(recipes),
    )


@pytest.fixture
def gateway():
    gateway = Gateway()
    yield gateway
    gateway.stop()


def test_plan_reports_unique_bytes_and_download_time_without_touching_the_fleet(
    tmp_path: Path, capsys
) -> None:
    recipes = [
        FakeRecipe("a", ("m1",)),
        FakeRecipe("b", ("m1",)),
        FakeRecipe("c", ("m2",)),
        FakeRecipe("d", ("m3",), local="cached"),
        FakeRecipe("e", ("m4",), node_count=4),
    ]
    models = [
        FakeModel("m1"),
        FakeModel("m2"),
        FakeModel("m3", local="cached"),
        FakeModel("m4"),
    ]
    fleet = _fleet(recipes, models)
    assert (
        _run(
            ["plan", "--json", "--state-dir", str(tmp_path), "--rate-mb-s", "400"],
            fleet,
            recipes,
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["testable_recipes"] == 4 and plan["skipped_recipes"] == ["vonk-forge/e"]
    assert plan["unique_models"] == 3  # m1 is shared by two recipes: downloaded once
    assert (
        plan["unique_model_bytes"] == 3 * 20 * GB
        and plan["bytes_to_download"] == 2 * 20 * GB
    )
    assert plan["download_eta_seconds"]["measured"] == pytest.approx(40 * GB / 400e6)
    assert (
        plan["next_models"][0]["cached"] is True
    )  # cached first, then most recipes per new byte
    assert plan["next_models"][1]["recipes"] == ["vonk-forge/a", "vonk-forge/b"]
    mutating = [
        c
        for _, c in fleet.calls
        if c[:2] in (("profile", "add"), ("profile", "load"), ("recipe", "download"))
    ]
    assert mutating == []


def test_plan_text_names_the_unique_bytes_and_a_rate_table(
    tmp_path: Path, capsys
) -> None:
    recipes = [FakeRecipe("a", ("m1",))]
    fleet = _fleet(recipes, [FakeModel("m1")])
    assert _run(["plan", "--state-dir", str(tmp_path)], fleet, recipes) == 0
    out = capsys.readouterr().out
    assert "unique models      1" in out and "download time" in out and "MB s" in out


def test_run_refuses_without_yes_and_never_accepts_owner_profile_numbers(
    tmp_path: Path, capsys
) -> None:
    recipes = [FakeRecipe("a")]
    fleet = _fleet(recipes, [FakeModel("m1")])
    assert _run(["run", "--state-dir", str(tmp_path)], fleet, recipes) == 2
    assert "stops whatever the Sparks currently run" in capsys.readouterr().err
    for flags in (
        ["--sweep-profile", "2"],
        ["--pin-profile", "3"],
        ["--sweep-profile", "10", "--pin-profile", "10"],
    ):
        with pytest.raises(SystemExit):
            _run(["run", "--yes", "--state-dir", str(tmp_path), *flags], fleet, recipes)
    assert fleet.calls == []  # not one vonkctl call before the arguments were accepted


def test_a_full_run_writes_state_results_status_and_report(
    tmp_path: Path, gateway: Gateway
) -> None:
    recipes = [
        FakeRecipe("a", ("m1",)),
        FakeRecipe("b", ("m2",), fail_load="driver mismatch", fail_phase="start"),
    ]
    fleet = _fleet(recipes, [FakeModel("m1"), FakeModel("m2")], gateway.start())
    code = _run(
        ["run", "--yes", "--state-dir", str(tmp_path), "--restore-owner", "2"],
        fleet,
        recipes,
    )
    assert code == 0
    for name in (
        "state.json",
        "results.jsonl",
        "status.json",
        "status.md",
        "report.json",
        "report.md",
    ):
        assert (tmp_path / name).exists(), name
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["summary"] == {"passed": 1, "failed": 1}
    row = next(r for r in report["recipes"] if r["recipe"] == "vonk-forge/a")
    assert (
        row["tokens_per_second"] > 0
        and row["ttft_ms"] is not None
        and row["load_s"]
        and row["download_s"]
    )
    assert [c for p, c in fleet.calls if c[:2] == ("profile", "load")][
        -1
    ] and fleet.calls[-1][0] == 2
    # Rerunning resumes: nothing left to do and nothing is tested again.
    before = len(fleet.calls)
    assert _run(["run", "--yes", "--state-dir", str(tmp_path)], fleet, recipes) == 0
    assert not [
        c
        for _, c in fleet.calls[before:]
        if c[:2] in (("profile", "add"), ("recipe", "download"))
    ]


def test_status_and_report_commands_read_the_state_dir(tmp_path: Path, capsys) -> None:
    recipes = [FakeRecipe("a")]
    fleet = _fleet(recipes, [FakeModel("m1")])
    assert _run(["status", "--state-dir", str(tmp_path)], fleet, recipes) == 0
    assert "no status yet" in capsys.readouterr().out
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "recipes": {
                    "vonk-forge/a": {
                        "status": "passed",
                        "smoke": "service",
                        "timings": {"load_s": 90},
                    }
                },
            }
        )
    )
    assert _run(["report", "--state-dir", str(tmp_path)], fleet, recipes) == 0
    assert "| vonk-forge/a | passed |" in capsys.readouterr().out
    assert (tmp_path / "report.md").exists()


def test_the_entry_point_is_executable() -> None:
    tool = ROOT / "tools" / "sweep-recipes"
    assert tool.stat().st_mode & 0o111
    done = subprocess.run(
        [sys.executable, str(tool), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "plan" in done.stdout and "export-evidence" in done.stdout


# -- evidence --------------------------------------------------------------------------


def _real_recipe() -> str:
    return f"vonk-forge/{min((ROOT / 'recipes').glob('*.json')).stem}"


def test_evidence_files_carry_the_latest_outcome_per_recipe_document(
    tmp_path: Path,
) -> None:
    log = ResultsLog(tmp_path / "results.jsonl", "auth")
    real = _real_recipe()
    log.append(
        batch="sweep",
        recipe=real,
        step="smoke",
        status="failed",
        content_sha256="old",
        error="x",
    )
    log.append(
        batch="sweep",
        recipe=real,
        step="smoke",
        status="passed",
        content_sha256="old",
        result={"cases": []},
    )
    log.append(
        batch="sweep",
        recipe=real,
        step="smoke",
        status="passed",
        content_sha256="new",
        result={"cases": []},
    )
    log.append(
        batch="sweep",
        recipe="vonk-forge/not-a-recipe",
        step="smoke",
        status="passed",
        content_sha256="x",
    )
    log.append(
        batch="sweep", recipe=real, step="stop", status="passed", content_sha256="new"
    )
    written = export_evidence(log.entries(), tmp_path / "evidence", [real])
    assert [p.name for p in written] == [f"{real.partition('/')[2]}.jsonl"]
    lines = [json.loads(x) for x in written[0].read_text().splitlines()]
    assert [(x["content_sha256"], x["status"]) for x in lines] == [
        ("old", "passed"),
        ("new", "passed"),
    ]


def test_committed_hardware_evidence_names_real_recipes_and_digests() -> None:
    directory = ROOT / "qualification" / "hardware-evidence"
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else []:
        assert (ROOT / "recipes" / f"{path.stem}.json").exists(), path
        for line in path.read_text().splitlines():
            record = json.loads(line)
            assert (
                record["recipe"] == f"vonk-forge/{path.stem}"
                and record["step"] == "smoke"
            )
            assert record["status"] in ("passed", "failed") and record["content_sha256"]


# -- issues --------------------------------------------------------------------------------


def test_one_issue_per_failure_with_the_hardware_test_label_and_no_duplicates() -> None:
    recipes = {
        "vonk-forge/bad": {
            "status": "failed",
            "phase": "start",
            "failure_class": "oom",
            "error": "out of memory",
            "cluster": "c1",
            "signature": "s",
        },
        "vonk-forge/known": {
            "status": "failed",
            "phase": "smoke",
            "failure_class": "smoke",
            "error": "x",
        },
        "vonk-forge/done": {
            "status": "failed",
            "phase": "start",
            "failure_class": "oom",
            "issue": "https://github.com/o/r/issues/9",
        },
        "vonk-forge/fine": {"status": "passed"},
    }
    calls: list[list[str]] = []

    def gh(argv):
        calls.append(list(argv))
        if argv[:2] == ["issue", "list"]:
            existing = [
                {
                    "title": title_for("vonk-forge/known", recipes["vonk-forge/known"]),
                    "url": "https://github.com/o/r/issues/5",
                }
            ]
            return 0, json.dumps(existing)
        if argv[:2] == ["issue", "create"]:
            return 0, "https://github.com/o/r/issues/7"
        return 0, ""

    filed = file_issues(recipes, "o/r", gh)
    assert {f["recipe"]: f["url"] for f in filed} == {
        "vonk-forge/bad": "https://github.com/o/r/issues/7",
        "vonk-forge/known": "https://github.com/o/r/issues/5",
    }
    creates = [c for c in calls if c[:2] == ["issue", "create"]]
    assert len(creates) == 1 and creates[0][creates[0].index("--label") + 1] == LABEL
    assert recipes["vonk-forge/bad"]["issue"].endswith("/7")
    assert file_issues(recipes, "o/r", gh) == []  # a rerun files nothing new


def test_dry_run_files_nothing() -> None:
    called = []
    filed = file_issues(
        {
            "vonk-forge/x": {
                "status": "failed",
                "phase": "start",
                "failure_class": "oom",
            }
        },
        "o/r",
        lambda a: called.append(a) or (0, ""),
        dry_run=True,
    )
    assert called == [] and filed[0]["url"] == "(dry run)"
