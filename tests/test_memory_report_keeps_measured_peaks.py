"""The catalog report applies the keep rule a move was checked by: its pre-move content."""

import runpy
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from vonk_forge_contracts import (
    QualificationCauseCode,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationOutcome,
    QualificationPlatform,
    QualificationSummary,
    QualificationTopology,
    RecipeQualificationEvidence,
)

from qualification import documents, memory_need
from qualification.memory_need import ADMISSION_MAXIMUM_PEAK_BYTES

ROOT = Path(__file__).resolve().parents[1]
MOVED = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"  # vLLM, four Sparks
AT_ENVELOPE = "muse-spark-open-nvfp4-vllm-eight"  # vLLM, peaks at the cap
SGLANG = "deepseek-v4-1-flash-1017-mia-sglang-triple"  # no vLLM envelope
UNDERSTATES = "peak understates vLLM budget"


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def write(root: Path, slug: str, peak_bytes: int) -> None:
    recipe = documents.recipe(slug)
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes = peak_bytes
        role.resources.memory.load_peak_bytes = None
        role.resources.memory.minimum_swap_bytes = 0
    (root / "recipes" / f"{slug}.json").write_text(recipe.model_dump_json())


def catalog(tmp_path: Path, *, history: bool) -> Path:
    """Three recipes; only MOVED has a peak below its vLLM envelope (a move)."""
    (tmp_path / "recipes").mkdir()
    (tmp_path / "models").symlink_to(ROOT / "models", target_is_directory=True)
    for slug in (MOVED, SGLANG):  # authored bytes: the predecessor's digest
        shutil.copyfile(
            ROOT / "recipes" / f"{slug}.json", tmp_path / "recipes" / f"{slug}.json"
        )
    write(tmp_path, AT_ENVELOPE, ADMISSION_MAXIMUM_PEAK_BYTES)
    if history:
        git(tmp_path, "init", "-q", "-b", "main")
        git(tmp_path, "add", "-A")
        git(tmp_path, "commit", "-q", "-m", "measured content")
    write(tmp_path, MOVED, 100_000_000_000)
    if history:
        git(tmp_path, "commit", "-q", "-am", "same-budget move")
    return tmp_path


def seed_pass(monkeypatch: pytest.MonkeyPatch, content_sha256: str) -> None:
    """MOVED's evidence: one hardware pass on ``content_sha256``."""
    key = f"vonk-forge/{MOVED}"
    sparks = documents.recipe(MOVED).topology.node_count
    record = QualificationEvidenceRecord(
        identity=QualificationIdentity(
            content_sha256=content_sha256, required_sparks=sparks
        ),
        outcome=QualificationOutcome.PASSED,
        cause_code=QualificationCauseCode.HARDWARE_PASS,
        cause_summary=QualificationSummary.PASSED,
        tested_at=datetime(2026, 11, 1, tzinfo=UTC),
        platform=QualificationPlatform(
            controller_source_sha="0" * 40,
            spark_count=sparks,
            topology=QualificationTopology.MULTI,
        ),
    )

    def evidence(recipe_key: str) -> RecipeQualificationEvidence:
        records = (record,) if recipe_key == key else ()
        return RecipeQualificationEvidence(recipe_key=recipe_key, records=records)

    monkeypatch.setattr(memory_need, "_evidence", evidence)


def listed(body: str) -> bool:
    """Whether the issue body lists MOVED's envelope finding."""
    return any(MOVED in line and UNDERSTATES in line for line in body.splitlines())


def report(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str]]:
    """The issue body and the recipes whose predecessor was looked up."""
    main = runpy.run_path(str(ROOT / "tools/check-recipe-memory-need"))["main"]
    monkeypatch.setitem(main.__globals__, "ROOT", root)
    looked_up: list[str] = []
    lookup = main.__globals__["previous_document"]

    def counted(root: Path, base: str | None, path: Path) -> dict | None:
        looked_up.append(path.stem)
        return lookup(root, base, path)

    monkeypatch.setitem(main.__globals__, "previous_document", counted)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    body = root / "report.md"
    monkeypatch.setattr("sys.argv", ["check-recipe-memory-need", "--report", str(body)])
    assert main() == 0
    return body.read_text(), looked_up


def test_same_budget_move_of_measured_content_is_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_pass(monkeypatch, documents.recipe_digest(MOVED))
    body, looked_up = report(catalog(tmp_path, history=True), monkeypatch)
    assert not listed(body)
    # Only the recipe the keep rule can change pays for a git lookup.
    assert looked_up == [MOVED]


def test_evidence_on_other_content_keeps_the_move_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_pass(monkeypatch, "0" * 64)
    body, _ = report(catalog(tmp_path, history=True), monkeypatch)
    assert listed(body)


def test_without_history_the_move_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_pass(monkeypatch, documents.recipe_digest(MOVED))
    body, looked_up = report(catalog(tmp_path, history=False), monkeypatch)
    assert listed(body)
    assert looked_up == [MOVED]
