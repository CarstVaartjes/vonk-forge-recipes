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
    QualificationMeasuredRoleMemory,
    QualificationOutcome,
    QualificationPlatform,
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
    vllm_budget_bytes,
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
GEMMA = "gemma-4-31b-it-nvfp4-vllm-mia-single"  # option util-085 raises the budget


def record(
    recipe: RecipeDefinition, content: str, *, failed: bool = False, day: int = 0
) -> QualificationEvidenceRecord:
    sparks = recipe.topology.node_count
    return QualificationEvidenceRecord(
        identity=QualificationIdentity(content_sha256=content, required_sparks=sparks),
        outcome=QualificationOutcome.FAILED if failed else QualificationOutcome.PASSED,
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
    )


def measured(
    monkeypatch: pytest.MonkeyPatch,
    recipe: RecipeDefinition,
    steady_bytes: int | None,
    *,
    budget_bytes: int | None = None,
    roles: tuple[str, ...] | None = None,
    failed_after: bool = False,
    content: str = "a" * 64,
) -> None:
    """Earlier content of ``recipe`` passed; its default launch (what the sweep runs)
    measured ``steady_bytes`` per role (``None``: the pass carries no measurement)."""
    key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
    records = (record(recipe, content),) + (
        (record(recipe, content, failed=True, day=1),) if failed_after else ()
    )
    budget = budget_bytes if budget_bytes is not None else vllm_budget_bytes(recipe)
    rows = (
        ()
        if steady_bytes is None
        else tuple(
            QualificationMeasuredRoleMemory(
                content_sha256=content,
                role=name,
                steady_bytes=steady_bytes,
                budget_bytes=budget,
            )
            for name in sorted(roles or {r.name for r in recipe.topology.roles})
        )
    )
    evidence = RecipeQualificationEvidence(
        recipe_key=key, records=records, measured_memory=rows
    )
    monkeypatch.setattr(memory_need, "_evidence", lambda _key: evidence)


def test_measured_move_that_raises_the_budget_raises_to_the_capped_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real identity: util 0.80 measured, upstream moves it to 0.92.
    previous = documents.recipe(PROVEN)
    measured(monkeypatch, previous, 1)
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
    measured(monkeypatch, previous, 2_000_000_000)
    assert raise_peaks_to_envelope(same) is None
    assert raise_peaks_to_envelope(with_utilization(previous, "0.5")) is None


@pytest.mark.parametrize(
    "measurement",
    [
        # Old evidence: a pass without a measurement is "not measured".
        {"steady_bytes": None},
        # Hardware used more than the declaration allows.
        {"steady_bytes": 1_000_000_000 + PLATFORM_MEMORY_FLOOR_BYTES + 1},
        # Measured under a smaller budget than the launch has now.
        {"steady_bytes": 1, "budget_bytes": 1},
        # The measured content failed afterwards: it is no longer proven.
        {"steady_bytes": 1, "failed_after": True},
        # A role the run did not measure.
        {"steady_bytes": 1, "roles": ("other",)},
    ],
)
def test_peaks_not_covered_by_measured_evidence_are_raised(
    monkeypatch: pytest.MonkeyPatch, measurement: dict
) -> None:
    previous = documents.recipe(PROVEN)
    declare_peak(previous, 1_000_000_000)
    measured(monkeypatch, previous, **measurement)
    assert raise_peaks_to_envelope(previous.model_copy(deep=True)) is not None


def test_only_the_measured_launch_variant_keeps_its_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep measures the default launch; a larger option budget is unmeasured."""
    recipe = documents.recipe(GEMMA)
    declare_peak(recipe, 90_000_000_000)
    measured(monkeypatch, recipe, 90_000_000_000)
    larger = recipe.with_option_choices({"gpu-memory": "util-085"})
    raised = raise_peaks_to_envelope(recipe)
    assert raised is not None
    assert {r.resources.memory.peak_bytes for r in raised.topology.roles} == {
        declarable_vllm_envelope(larger)
    }
    warnings = memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest=""
    )
    assert_warns(
        warnings, f"vLLM budget plus overhead {declarable_vllm_envelope(larger)}"
    )
    assert_not_warns(
        warnings, f"vLLM budget plus overhead {declarable_vllm_envelope(recipe)}"
    )
    measured(
        monkeypatch, recipe, 90_000_000_000, budget_bytes=vllm_budget_bytes(larger)
    )
    assert raise_peaks_to_envelope(recipe) is None


def test_a_proven_replaced_document_without_rows_keeps_its_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh hands the document it replaces (never git): proven, it vouches."""
    previous = documents.recipe(PROVEN)
    declare_peak(previous, 1_000_000_000)
    measured(monkeypatch, previous, None)  # Working, but no measured rows
    replaced = (previous, "a" * 64)
    same = previous.model_copy(deep=True)
    assert raise_peaks_to_envelope(same) is not None
    assert raise_peaks_to_envelope(same, replaced) is None
    assert raise_peaks_to_envelope(with_utilization(previous, "0.5"), replaced) is None
    # A move that raises the budget, or lowers a peak, is not vouched for.
    assert raise_peaks_to_envelope(with_utilization(previous, "0.92"), replaced)
    lower = previous.model_copy(deep=True)
    declare_peak(lower, 999_999_999)
    assert raise_peaks_to_envelope(lower, replaced) is not None
    # A measured overrun of the replaced content decides over the fallback.
    measured(monkeypatch, previous, 2 * 1_000_000_000 + PLATFORM_MEMORY_FLOOR_BYTES)
    assert raise_peaks_to_envelope(same, replaced) is not None


def test_complete_rows_retire_the_replaced_document_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured content is judged by its rows per launch variant, refresh or not."""
    recipe = documents.recipe(GEMMA)
    declare_peak(recipe, 90_000_000_000)
    measured(monkeypatch, recipe, 90_000_000_000)  # every role, in peak, default launch
    larger = recipe.with_option_choices({"gpu-memory": "util-085"})
    envelope = {declarable_vllm_envelope(larger)}
    for replaced in (None, (recipe, "a" * 64)):
        raised = raise_peaks_to_envelope(recipe.model_copy(deep=True), replaced)
        assert raised is not None
        assert {
            r.resources.memory.peak_bytes for r in raised.topology.roles
        } == envelope
    warnings = memory_estimate_warnings(
        recipe, catalog_models(), INPUTS, document_digest=""
    )
    assert_warns(
        warnings, f"vLLM budget plus overhead {declarable_vllm_envelope(larger)}"
    )


def test_partial_rows_of_a_proven_replaced_dual_keep_its_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass that measured one role only (the other Spark not alone) costs nothing."""
    previous = documents.recipe(LLAMA)
    declare_peak(previous, 1_000_000_000)
    replaced = (previous, "a" * 64)
    same = previous.model_copy(deep=True)
    measured(monkeypatch, previous, 1, roles=("entrypoint",))
    assert raise_peaks_to_envelope(same) is not None  # rows alone: not measured
    assert raise_peaks_to_envelope(same, replaced) is None
    over = 1_000_000_000 + PLATFORM_MEMORY_FLOOR_BYTES + 1
    measured(monkeypatch, previous, over, roles=("entrypoint",))
    assert raise_peaks_to_envelope(same, replaced) is not None


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
    measured(monkeypatch, previous, 99_000_000_000)
    assert_not_warns(warns(), "peak understates vLLM")
    measured(monkeypatch, previous, 103_000_000_000)
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
def test_a_real_proven_content_keeps_peaks_once_it_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The committed pass records join the table as the sweep writes it."""
    proven = None
    for path in sorted((memory_need.ROOT / "recipes").glob("*.json")):
        recipe = documents.recipe(path.stem)
        digest = documents.recipe_digest(path.stem)
        if (
            recipe.runtime.engine == "vllm"
            and not recipe.options
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
    unmeasured = real.model_copy(update={"measured_memory": ()})
    monkeypatch.setattr(memory_need, "_evidence", lambda _key: unmeasured)
    assert raise_peaks_to_envelope(moved) is not None  # not measured: raised
    rows = tuple(
        QualificationMeasuredRoleMemory(
            content_sha256=digest,
            role=name,
            steady_bytes=1,
            budget_bytes=vllm_budget_bytes(previous),
        )
        for name in sorted({r.name for r in previous.topology.roles})
    )
    table = real.model_copy(update={"measured_memory": rows})
    monkeypatch.setattr(memory_need, "_evidence", lambda _key: table)
    assert raise_peaks_to_envelope(moved) is None
    warnings = memory_estimate_warnings(
        moved, catalog_models(), INPUTS, document_digest=""
    )
    assert_not_warns(warnings, "peak understates vLLM")


@pytest.mark.live_evidence
def test_a_refresh_keeps_every_real_unmeasured_proven_vllm_recipes_peaks() -> None:
    """Proven content without rows: the replaced document keeps what main kept.

    Content the sweep has measured is judged by its rows instead, so evidence
    batches that add rows never turn this red.
    """
    for path in sorted((memory_need.ROOT / "recipes").glob("*.json")):
        recipe = documents.recipe(path.stem)
        digest = documents.recipe_digest(path.stem)
        key = f"{recipe.identity.publisher}/{recipe.identity.slug}"
        if (
            recipe.runtime.engine != "vllm"
            or any(
                row.content_sha256 == digest
                for row in memory_need._evidence(key).measured_memory
            )
            or not memory_need.hardware_proven(recipe, digest)
        ):
            continue
        same = recipe.model_copy(deep=True)
        assert raise_peaks_to_envelope(same, (recipe, digest)) is None, path.stem
