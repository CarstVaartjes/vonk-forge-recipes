"""The admission maximum caps the vLLM envelope; peaks measured under no smaller budget stay."""

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import (
    QualificationBlame,
    QualificationCauseCode,
    QualificationEvidenceRecord,
    QualificationIdentity,
    QualificationMeasuredMemory,
    QualificationOutcome,
    QualificationPlatform,
    QualificationRoleMemory,
    QualificationSummary,
    QualificationTopology,
    RecipeDefinition,
    RecipeQualificationEvidence,
)
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from warning_asserts import assert_not_warns, assert_warns, declare_peak

from qualification import documents, memory_need
from qualification.memory_need import (
    ADMISSION_IDLE_USE_BYTES,
    ADMISSION_MAXIMUM_PEAK_BYTES,
    PLATFORM_MEMORY_FLOOR_BYTES,
    SPARK_IDLE_AVAILABLE_BYTES,
    SPARK_MEMORY_BYTES,
    declarable_vllm_envelope,
    load_inputs,
    memory_estimate_warnings,
    raise_peaks_to_envelope,
    recipe_envelope,
    vllm_utilization_envelope,
)

INPUTS = load_inputs()
GLM = "glm-5-3-flash-exl3-k2-0xsero-vllm-single"
PROVEN = "qwen3-6-35b-a3b-fp8-eugr-vllm-single"
UNPROVEN = "muse-spark-open-nvfp4-vllm-eight"
LLAMA = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-four"


def test_maximum_derives_from_the_admission_rule() -> None:
    assert (
        ADMISSION_MAXIMUM_PEAK_BYTES
        == SPARK_IDLE_AVAILABLE_BYTES - PLATFORM_MEMORY_FLOOR_BYTES
        == 124_452_516_352
    )


def test_maximum_is_the_edge_of_the_offline_admission_rule() -> None:
    def admissible(peak: int) -> bool:
        return evaluate_memory_admission(
            peak, SPARK_MEMORY_BYTES, occupied_bytes=ADMISSION_IDLE_USE_BYTES
        ).admissible

    assert admissible(ADMISSION_MAXIMUM_PEAK_BYTES)
    assert not admissible(ADMISSION_MAXIMUM_PEAK_BYTES + 1)


@pytest.mark.parametrize("slug", [GLM, LLAMA])
def test_peak_at_the_admission_maximum_is_not_understated(slug: str) -> None:
    recipe = documents.recipe(slug)
    declare_peak(recipe, ADMISSION_MAXIMUM_PEAK_BYTES)
    warnings = memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest="", touched=True
    )
    assert_not_warns(warnings, "peak understates (vLLM budget|the vLLM)")


@pytest.mark.parametrize("slug", [GLM, LLAMA])
def test_envelope_never_exceeds_the_admission_maximum(slug: str) -> None:
    recipe = documents.recipe(slug)
    for item in recipe.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = "0.95"
    uncapped = vllm_utilization_envelope(recipe)
    assert uncapped is not None
    assert uncapped > ADMISSION_MAXIMUM_PEAK_BYTES
    assert declarable_vllm_envelope(recipe) == ADMISSION_MAXIMUM_PEAK_BYTES
    declare_peak(recipe, 1_000_000_000)
    raised = raise_peaks_to_envelope(recipe)
    assert raised is not None
    assert all(
        r.resources.memory.peak_bytes <= ADMISSION_MAXIMUM_PEAK_BYTES
        for r in raised.topology.roles
    )


def with_utilization(recipe: RecipeDefinition, value: str) -> RecipeDefinition:
    moved = recipe.model_copy(deep=True)
    for item in moved.runtime.arguments:
        if item.name == "gpu-memory-utilization":
            item.value = value
    return moved


NOW = datetime(2026, 11, 4, tzinfo=UTC)


def passed(
    recipe: RecipeDefinition,
    steady_bytes: int | None,
    *,
    budget_bytes: int | None = None,
    content: str = "a" * 64,
    roles: tuple[str, ...] | None = None,
    outcome: QualificationOutcome = QualificationOutcome.PASSED,
    day: int = 0,
) -> QualificationEvidenceRecord:
    """A hardware record of earlier content of ``recipe``; it measured ``steady_bytes``."""
    sparks = recipe.topology.node_count
    failed = outcome == QualificationOutcome.FAILED
    return QualificationEvidenceRecord(
        identity=QualificationIdentity(content_sha256=content, required_sparks=sparks),
        outcome=outcome,
        blame=QualificationBlame.RECIPE if failed else None,
        cause_code=QualificationCauseCode.RUNTIME_OOM
        if failed
        else QualificationCauseCode.HARDWARE_PASS,
        cause_summary=QualificationSummary.RUNTIME_OOM
        if failed
        else QualificationSummary.PASSED,
        tested_at=NOW + timedelta(days=day),
        platform=QualificationPlatform(
            controller_version="2.0.0",
            spark_count=sparks,
            topology=QualificationTopology.SINGLE
            if sparks == 1
            else QualificationTopology.MULTI,
        ),
        memory=None
        if steady_bytes is None or failed
        else QualificationMeasuredMemory(
            budget_bytes=budget_bytes
            if budget_bytes is not None
            else memory_need.recipe_budget(recipe),
            roles=tuple(
                QualificationRoleMemory(role=name, steady_bytes=steady_bytes)
                for name in sorted(roles or {r.name for r in recipe.topology.roles})
            ),
        ),
    )


def evidence(
    monkeypatch: pytest.MonkeyPatch,
    recipe: RecipeDefinition,
    *records: QualificationEvidenceRecord,
) -> None:
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    monkeypatch.setattr(
        memory_need,
        "_evidence",
        lambda _key: RecipeQualificationEvidence(recipe_key=key, records=records),
    )


def test_measured_move_that_raises_the_budget_raises_to_the_capped_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real identity: util 0.80 measured, upstream moves it to 0.92.
    previous = documents.recipe(PROVEN)
    evidence(monkeypatch, previous, passed(previous, 1))
    moved = with_utilization(previous, "0.92")
    raised = raise_peaks_to_envelope(moved)
    assert raised is not None
    assert {r.resources.memory.peak_bytes for r in raised.topology.roles} == {
        declarable_vllm_envelope(moved)
    }
    assert all(
        r.resources.memory.peak_bytes <= ADMISSION_MAXIMUM_PEAK_BYTES
        for r in raised.topology.roles
    )


def test_measured_move_that_keeps_the_budget_keeps_the_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = documents.recipe(PROVEN)
    declare_peak(previous, 1_000_000_000)
    same = previous.model_copy(deep=True)
    assert raise_peaks_to_envelope(same) is not None  # not measured
    evidence(monkeypatch, previous, passed(previous, 2_000_000_000))
    assert raise_peaks_to_envelope(same) is None
    assert raise_peaks_to_envelope(with_utilization(previous, "0.5")) is None


@pytest.mark.parametrize(
    "records",
    [
        # Old evidence: a pass with no measurement is "not measured".
        lambda r: (passed(r, None),),
        # Hardware used more than the declaration allows.
        lambda r: (passed(r, 1_000_000_000 + PLATFORM_MEMORY_FLOOR_BYTES + 1),),
        # Measured under a smaller budget than the launch has now.
        lambda r: (passed(r, 1, budget_bytes=(memory_need.recipe_budget(r) or 0) - 1),),
        # The measured content failed afterwards: it is no longer proven.
        lambda r: (
            passed(r, 1),
            passed(r, None, outcome=QualificationOutcome.FAILED, day=1),
        ),
        # A role the run did not measure.
        lambda r: (passed(r, 1, roles=("other",)),),
    ],
)
def test_peaks_not_covered_by_measured_evidence_are_raised(
    monkeypatch: pytest.MonkeyPatch, records
) -> None:
    previous = documents.recipe(PROVEN)
    declare_peak(previous, 1_000_000_000)
    evidence(monkeypatch, previous, *records(previous))
    assert not memory_need.keeps_measured_peaks(previous)
    assert raise_peaks_to_envelope(previous.model_copy(deep=True)) is not None


def test_unmeasured_estimate_heals_on_a_same_budget_move() -> None:
    # muse-spark declares 2.4 GB/rank below today's envelope and has no evidence.
    previous = documents.recipe(UNPROVEN)
    same = previous.model_copy(deep=True)
    raised = raise_peaks_to_envelope(same)
    assert raised is not None
    assert {r.resources.memory.peak_bytes for r in raised.topology.roles} == {
        recipe_envelope(same)
    }


def test_warning_follows_the_keep_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = documents.recipe(LLAMA)
    moved = previous.model_copy(deep=True)
    declare_peak(moved, 100_000_000_000)

    def warns() -> list[str]:
        return memory_estimate_warnings(
            moved, catalog_models(), INPUTS, document_digest=""
        )

    assert_warns(warns(), "peak understates vLLM")
    evidence(monkeypatch, previous, passed(previous, 99_000_000_000))
    assert_not_warns(warns(), "peak understates vLLM")
    evidence(monkeypatch, previous, passed(previous, 103_000_000_000))
    assert_warns(warns(), "peak understates vLLM")


ALLOWED = {
    ("qualification/memory_need.py", "declarable_vllm_envelope"),
    ("qualification/memory_need.py", "bounded_kv_warnings"),
}
NAME = "vllm_utilization_envelope"


def uses(source: str, relative: str) -> list[str]:
    """Every read of the uncapped envelope outside the allowlist, by scope."""
    tree = ast.parse(source)
    aliases = {NAME}
    for node in ast.walk(tree):
        for imported in getattr(node, "names", []):
            if getattr(imported, "name", "") == NAME and imported.asname:
                aliases.add(imported.asname)
    found: list[str] = []

    def visit(node: ast.AST, scope: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            scope = node.name
        hit = (
            (isinstance(node, ast.Name) and node.id in aliases)
            or (isinstance(node, ast.Attribute) and node.attr == NAME)
            or (
                isinstance(node, ast.ImportFrom)
                and any(a.name == NAME for a in node.names)
            )
        )
        if hit and (relative, scope) not in ALLOWED:
            found.append(f"{relative}:{getattr(node, 'lineno', 0)} in {scope}")
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "<module>")
    return found


def python_sources(root: Path) -> list[Path]:
    paths: list[Path] = []
    for folder in ("qualification", "spark_sweep", "scripts", "tools"):
        for path in (root / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix == ".py":
                paths.append(path)
            elif not path.suffix:
                head = path.read_bytes()[:80]
                if head.startswith(b"#!") and b"python" in head.split(b"\n")[0]:
                    paths.append(path)
    return paths


def test_only_the_admissibility_check_reads_the_uncapped_envelope() -> None:
    root = Path(__file__).resolve().parents[1]
    found = [
        item
        for path in python_sources(root)
        for item in uses(path.read_text(), str(path.relative_to(root)))
    ]
    assert found == []


@pytest.mark.parametrize(
    "source",
    [
        "def f(r):\n    return vllm_utilization_envelope(r)\n",
        "async def f(r):\n    return vllm_utilization_envelope(r)\n",
        "import m\nx = m.vllm_utilization_envelope(1)\n",
        "from m import vllm_utilization_envelope as v\nx = v(1)\n",
        "def f(m):\n    return m.vllm_utilization_envelope(1)\n",
        "x = vllm_utilization_envelope(1)\n",
    ],
)
def test_the_guard_catches_every_form(source: str) -> None:
    assert uses(source, "tools/x.py")


def test_the_guard_allows_the_listed_scopes() -> None:
    source = (
        "def declarable_vllm_envelope(r):\n    return vllm_utilization_envelope(r)\n"
    )
    assert uses(source, "qualification/memory_need.py") == []
    assert uses(source, "tools/other.py")


@pytest.mark.live_evidence
def test_a_real_proven_record_keeps_peaks_once_it_carries_a_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The committed pass records shapes as the sweep writes them; measured, they keep."""
    proven = None
    for path in sorted((memory_need.ROOT / "recipes").glob("*.json")):
        recipe = documents.recipe(path.stem)
        digest = documents.recipe_digest(path.stem)
        if (
            recipe.runtime.engine == "vllm"
            and recipe_envelope(recipe)
            and memory_need.hardware_proven(recipe, digest)
        ):
            proven = (recipe, digest)
            break
    assert proven, "no hardware-proven vLLM recipe in the committed evidence"
    previous, digest = proven
    key = f"{previous.identity.publisher}/{previous.identity.slug}"
    real = memory_need._evidence(key)
    moved = previous.model_copy(deep=True)
    declare_peak(moved, 1_000_000_000)
    measurement = passed(previous, 1).memory
    records = tuple(
        record.model_copy(update={"memory": measurement})
        if record.identity.content_sha256 == digest
        and record.outcome == QualificationOutcome.PASSED
        else record
        for record in real.records
    )
    unmeasured = [r for r in real.records if r.memory is None]
    monkeypatch.setattr(
        memory_need,
        "_evidence",
        lambda _key: real.model_copy(update={"records": tuple(unmeasured)}),
    )
    assert raise_peaks_to_envelope(moved) is not None  # not measured: raised
    monkeypatch.setattr(
        memory_need,
        "_evidence",
        lambda _key: real.model_copy(update={"records": records}),
    )
    assert raise_peaks_to_envelope(moved) is None
    warnings = memory_estimate_warnings(
        moved, catalog_models(), INPUTS, document_digest=""
    )
    assert_not_warns(warnings, "peak understates vLLM")
