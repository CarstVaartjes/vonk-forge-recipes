"""The scheduling policy as pure functions."""

from __future__ import annotations

import pytest

from spark_sweep import policy
from spark_sweep.catalog import Recipe

GB = 10**9
CAP = 130_663_231_488
RESERVE = 8 * GB


def recipe(slug: str, digests: tuple[str, ...] = ("m",), **kw) -> Recipe:
    fields = {
        "key": f"vonk-forge/{slug}",
        "title": slug,
        "node_count": 1,
        "usage": ("chat",),
        "engine": "vllm",
        "creator": "",
        "model_digests": digests,
        "model_selectors": digests,
        "memory_bytes": 40 * GB,
        "image_bytes": 0,
        "artifact_bytes": 0,
        "ports": (8000,),
        "adapter": "openai",
        "aliases": (slug,),
        "content_sha256": "c1",
        "revision_id": "r1",
        "local": "cached",
        "updated_at": "",
        "base_image": "b",
    }
    fields.update(kw)
    return Recipe(**fields)  # type: ignore[arg-type]


def keys(plans: list[policy.GroupPlan]) -> list[list[str]]:
    return [[k.split("/")[1] for k in p.recipes] for p in plans]


# -- value per byte --------------------------------------------------------------


def test_cached_models_come_first_small_first_then_most_recipes_per_new_byte() -> None:
    pending = (
        [recipe(f"x{i}", ("X",)) for i in range(4)]  # 100 GB for 4 recipes: 0.04 per GB
        + [recipe("y", ("Y",))]  # 10 GB for 1: 0.10 per GB
        + [recipe(f"z{i}", ("Z",)) for i in range(6)]  # 30 GB for 6: 0.20 per GB
        + [recipe("big-cached", ("C1",)), recipe("small-cached", ("C2",))]
    )
    sizes = {"X": 100 * GB, "Y": 10 * GB, "Z": 30 * GB, "C1": 90 * GB, "C2": 5 * GB}
    plans = policy.plan_groups(pending, sizes, present={"C1", "C2"})
    assert [sorted(p.digests) for p in plans] == [["C2"], ["C1"], ["Z"], ["Y"], ["X"]]
    assert keys(plans) == [
        ["small-cached"],
        ["big-cached"],
        [f"z{i}" for i in range(6)],
        ["y"],
        [f"x{i}" for i in range(4)],
    ]
    assert [p.new_bytes for p in plans] == [0, 0, 30 * GB, 10 * GB, 100 * GB]
    assert [p.cached for p in plans] == [True, True, False, False, False]


def test_a_recipe_sharing_part_of_a_model_set_is_unlocked_by_the_shared_part() -> None:
    pending = [
        recipe("only-b", ("B",)),
        recipe("a-and-b", ("A", "B")),
        recipe("far", ("F",)),
    ]
    sizes = {"A": 5 * GB, "B": 50 * GB, "F": 60 * GB}
    plans = policy.plan_groups(pending, sizes, present={"A"})
    assert keys(plans)[0] == [
        "only-b",
        "a-and-b",
    ]  # B alone unlocks both: two recipes for 50 GB
    assert plans[0].new_bytes == 50 * GB


def test_owner_priority_boost_outranks_a_cheaper_group() -> None:
    pending = [recipe("glm-5-3-flash", ("G",)), recipe("cheap", ("C",))]
    sizes = {"G": 80 * GB, "C": 10 * GB}
    plain = policy.plan_groups(pending, sizes, set())
    boosted = policy.plan_groups(
        pending, sizes, set(), policy.keyword_boost(["glm-5-3"], factor=100)
    )
    assert keys(plain)[0] == ["cheap"] and keys(boosted)[0] == ["glm-5-3-flash"]


def test_build_sizes_prefers_the_model_size_and_falls_back_to_declared_artifact_bytes() -> (
    None
):
    from spark_sweep.catalog import Model

    recipes = [
        recipe("a", ("A",), artifact_bytes=7 * GB),
        recipe("b", ("B",), artifact_bytes=9 * GB),
    ]
    sizes = policy.build_sizes(recipes, {"A": Model("m/a", "A", 11 * GB, "cached")})
    assert sizes == {"A": 11 * GB, "B": 9 * GB}


def test_in_flight_and_cached_models_count_as_present() -> None:
    from spark_sweep.catalog import Model

    models = {
        "A": Model("a", "A", 1, "cached"),
        "B": Model("b", "B", 1, "preparing"),
        "C": Model("c", "C", 1, "not_cached"),
    }
    assert policy.present_models(models) == {"A", "B"}


# -- order -------------------------------------------------------------------------


def test_group_members_are_back_to_back_with_engine_and_base_image_locality() -> None:
    members = [
        recipe("v1", engine="vllm", base_image="b1"),
        recipe("s1", engine="sglang", base_image="b2"),
        recipe("v2", engine="vllm", base_image="b1"),
        recipe("v3", engine="vllm", base_image="b3"),
    ]
    other = recipe("o", ("O",))
    by_key = {r.key: r for r in members + [other]}
    plans = policy.plan_groups(members + [other], {"m": GB, "O": 2 * GB}, {"m", "O"})
    order = [k.split("/")[1] for k in policy.order_queue(plans, by_key)]
    assert order == [
        "s1",
        "v1",
        "v2",
        "v3",
        "o",
    ]  # model group together; engine, then base image, inside it


def test_revalidation_goes_after_new_work_and_suspect_siblings_last() -> None:
    rs = [recipe("new", ("N",)), recipe("old", ("O",)), recipe("sib", ("S",))]
    by_key = {r.key: r for r in rs}
    plans = policy.plan_groups(rs, {"N": GB, "O": GB, "S": GB}, {"N", "O", "S"})
    order = policy.order_queue(
        plans, by_key, revalidate={"vonk-forge/old"}, deprioritised={"vonk-forge/sib"}
    )
    assert [k.split("/")[1] for k in order] == ["new", "old", "sib"]


def test_variants_last_tests_one_recipe_per_model_first() -> None:
    rs = [
        recipe("a1", ("A",)),
        recipe("a2", ("A",)),
        recipe("b1", ("B",)),
        recipe("b2", ("B",)),
    ]
    by_key = {r.key: r for r in rs}
    plans = policy.plan_groups(rs, {"A": GB, "B": 2 * GB}, {"A", "B"})
    order = policy.order_queue(plans, by_key, variants_last=True)
    assert [k.split("/")[1] for k in order] == ["a1", "b1", "a2", "b2"]


# -- placement -------------------------------------------------------------------------


def bins(*used: int) -> list[policy.SparkBin]:
    return [policy.SparkBin(f"s{i}", f"spark-{i}", CAP, u) for i, u in enumerate(used)]


def alias(r: Recipe) -> str:
    return r.slug


def test_two_lanes_fill_before_a_spark_takes_a_second_recipe() -> None:
    cands = [
        recipe("a", ports=(8001,)),
        recipe("b", ports=(8002,)),
        recipe("c", ports=(8003,)),
    ]
    placed = policy.place_singles(cands, bins(0, 0), RESERVE, alias)
    assert [(p.recipe.slug, p.spark_ids) for p in placed] == [
        ("a", ("s0",)),
        ("b", ("s1",)),
        ("c", ("s0",)),
    ]


def test_two_small_recipes_pack_on_one_spark_when_memory_fits_and_ports_differ() -> (
    None
):
    cands = [
        recipe("a", memory_bytes=40 * GB, ports=(8001,)),
        recipe("b", memory_bytes=40 * GB, ports=(8002,)),
    ]
    placed = policy.place_singles(cands, bins(0), RESERVE, alias)
    assert [p.spark_ids for p in placed] == [("s0",), ("s0",)]


def test_a_port_clash_or_too_little_memory_prevents_packing() -> None:
    clash = [recipe("a"), recipe("b")]  # both on 8000
    assert len(policy.place_singles(clash, bins(0), RESERVE, alias)) == 1
    big = [
        recipe("a", memory_bytes=70 * GB, ports=(1,)),
        recipe("b", memory_bytes=70 * GB, ports=(2,)),
    ]
    assert len(policy.place_singles(big, bins(0), RESERVE, alias)) == 1
    assert len(policy.place_singles(big, bins(0, 0), RESERVE, alias)) == 2


def test_first_fit_decreasing_places_the_biggest_first_within_the_window() -> None:
    cands = [
        recipe("small", memory_bytes=30 * GB, ports=(1,)),
        recipe("huge", memory_bytes=100 * GB, ports=(2,)),
    ]
    placed = policy.place_singles(cands, bins(0, 0), RESERVE, alias)
    assert {p.recipe.slug: p.spark_ids for p in placed} == {
        "huge": ("s0",),
        "small": ("s1",),
    }
    assert [p.recipe.slug for p in placed] == [
        "small",
        "huge",
    ]  # but queue order is kept for the result


def test_running_recipes_are_respected_and_aliases_cannot_collide() -> None:
    running = [
        policy.SparkBin(
            "s0", "spark-0", CAP, 100 * GB, frozenset({8000}), frozenset({"a"})
        )
    ]
    assert (
        policy.place_singles(
            [recipe("b", ports=(9000,), memory_bytes=40 * GB)], running, RESERVE, alias
        )
        == []
    )
    assert (
        policy.place_singles(
            [recipe("a", ports=(9000,), memory_bytes=1 * GB)], running, RESERVE, alias
        )
        == []
    )


def test_a_dual_needs_every_spark_empty() -> None:
    d = recipe("d", node_count=2, memory_bytes=60 * GB)
    assert policy.place_dual([d], bins(0, 1), RESERVE) is None
    placed = policy.place_dual([recipe("single"), d], bins(0, 0), RESERVE)
    assert (
        placed is not None
        and placed.recipe.slug == "d"
        and placed.spark_ids == ("s0", "s1")
    )


def test_dual_windows_drain_both_sparks_once_instead_of_interleaving() -> None:
    assert policy.choose_mode("single", singles_ready=5, duals_ready=1) == "single"
    assert policy.choose_mode("single", singles_ready=5, duals_ready=2) == "dual"
    assert (
        policy.choose_mode("dual", singles_ready=5, duals_ready=1) == "dual"
    )  # stay until the duals are done
    assert policy.choose_mode("dual", singles_ready=5, duals_ready=0) == "single"
    assert policy.choose_mode("single", singles_ready=0, duals_ready=1) == "dual"


def test_never_fits_flags_declared_demand_beyond_an_empty_spark() -> None:
    assert policy.never_fits(recipe("x", memory_bytes=128 * GB), CAP, RESERVE)
    assert not policy.never_fits(recipe("x", memory_bytes=100 * GB), CAP, RESERVE)
    assert not policy.never_fits(recipe("x", memory_bytes=0), CAP, RESERVE)


# -- timeouts -------------------------------------------------------------------------------


def test_timeout_is_a_flat_first_start_allowance_until_loads_have_been_measured() -> (
    None
):
    assert policy.load_timeout({}, "vllm", 30 * GB) == 3600


def test_timeout_learns_per_engine_and_scales_with_model_size() -> None:
    samples: dict[str, list[list[float]]] = {}
    policy.learn(samples, "vllm", 20 * policy.GIB, 600)  # 30 s per GiB
    small = policy.load_timeout(samples, "vllm", 20 * policy.GIB)
    large = policy.load_timeout(samples, "vllm", 40 * policy.GIB)
    assert small == pytest.approx(1800) and large == pytest.approx(3600)  # 3 x expected
    # An engine never seen falls back to what the others taught.
    assert policy.load_timeout(samples, "sglang", 20 * policy.GIB) == pytest.approx(
        1800
    )
    # A fast engine does not drag the timeout under the floor, a slow one is capped.
    fast: dict[str, list[list[float]]] = {}
    policy.learn(fast, "x", 20 * policy.GIB, 20)
    assert policy.load_timeout(fast, "x", 20 * policy.GIB) == 1200
    slow: dict[str, list[list[float]]] = {}
    policy.learn(slow, "x", policy.GIB, 5000)
    assert policy.load_timeout(slow, "x", 10 * policy.GIB) == 7200


def test_learning_keeps_only_recent_samples() -> None:
    samples: dict[str, list[list[float]]] = {}
    for i in range(30):
        policy.learn(samples, "e", GB, float(i + 1))
    assert len(samples["e"]) == 20 and samples["e"][0][1] == 11.0


# -- failures --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "code", "detail", "klass"),
    [
        ("start", "", "CUDA error: out of memory", "oom"),
        ("start", "", "container exited with code 137", "oom"),
        (
            "download",
            "controller.unavailable",
            "dockerfile.heredoc_forbidden: Dockerfile heredocs are not accepted",
            "build-policy",
        ),
        (
            "review",
            "run-switch.resource.insufficient_capacity",
            "needs more memory",
            "capacity",
        ),
        ("download", "", "no space left on device", "capacity"),
        (
            "download",
            "",
            "sha256 mismatch for model-00001.safetensors",
            "model-integrity",
        ),
        ("download", "", "connection reset by peer", "network"),
        ("smoke", "case.M0", "HTTP 503", "smoke-timeout"),
        ("install", "", "something unexpected", "install"),
        ("timeout", "load.timeout", "not serving after 3600s", "timeout"),
        ("readiness", "run.not_published", "timed out waiting", "readiness"),
    ],
)
def test_failures_are_classified_by_phase_and_cause(
    phase: str, code: str, detail: str, klass: str
) -> None:
    assert policy.classify(phase, code, detail).klass == klass


def test_only_network_like_failures_are_retried_and_only_integrity_failures_are_model_level() -> (
    None
):
    assert policy.classify("download", "", "connection reset by peer").transient
    assert policy.classify("smoke", "case.x", "HTTP 503", "smoke-timeout").transient
    assert not policy.classify("timeout", "load.timeout", "x").transient
    assert not policy.classify("start", "", "out of memory").transient
    assert policy.classify("download", "", "digest mismatch").model_level
    assert not policy.classify("start", "", "out of memory").model_level


def test_signatures_ignore_ids_digits_and_paths_so_one_cause_is_one_cluster() -> None:
    a = policy.classify(
        "start",
        "x",
        "kernel 123 failed on /var/lib/a/b/c at 0123456789abcdef0123 run 11111111-2222-3333-4444-555555555555",
    )
    b = policy.classify(
        "start",
        "x",
        "kernel 98 failed on /opt/z/y/x at fedcba9876543210fedc run aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    )
    other = policy.classify("start", "x", "driver mismatch")
    assert a.cluster == b.cluster != other.cluster


def test_clusters_group_failed_recipes_biggest_first() -> None:
    entries = {
        "a": {"status": "failed", "cluster": "c1", "signature": "s1"},
        "b": {"status": "failed", "cluster": "c1", "signature": "s1"},
        "c": {"status": "failed", "cluster": "c2", "signature": "s2"},
        "d": {"status": "passed", "cluster": "c1", "signature": "s1"},
    }
    clusters = policy.cluster(entries)
    assert [(c.id, c.recipes) for c in clusters] == [("c1", ("a", "b")), ("c2", ("c",))]


def test_child_phases_map_to_failure_phases() -> None:
    assert policy.child_phase("container-build") == "build"
    assert policy.child_phase("runtime-install") == "install"
    assert policy.child_phase("final-verify") == "start"
    assert policy.child_phase("unknown") is None


# -- fixes feed back ---------------------------------------------------------------------------


def test_a_changed_document_digest_invalidates_a_recorded_result() -> None:
    r = recipe("x", content_sha256="new")
    assert (
        policy.requeue_reason({"status": "failed", "content_sha256": "old"}, r)
        == "failed-on-older-revision"
    )
    assert (
        policy.requeue_reason({"status": "passed", "content_sha256": "old"}, r)
        == "passed-on-older-revision"
    )
    assert (
        policy.requeue_reason({"status": "passed", "content_sha256": "new"}, r) is None
    )
    assert policy.requeue_reason({"status": "pending"}, r) is None
    assert policy.requeue_reason(None, r) is None


# -- throughput ----------------------------------------------------------------------------------


def test_rate_is_smoothed_ignores_idle_samples_and_gives_an_eta() -> None:
    rate = policy.RateTracker(alpha=0.5)
    assert rate.eta_seconds(10 * GB) is None
    rate.add(0)
    assert rate.samples == 0
    rate.add(100e6)
    rate.add(300e6)
    assert rate.rate == pytest.approx(200e6)
    assert rate.eta_seconds(20 * GB) == pytest.approx(100)


def test_planning_the_whole_catalog_is_fast_enough_to_run_every_tick() -> None:
    import time

    recipes = [
        recipe(
            f"r{i}", (f"m{i % 200}",) if i % 7 else (f"m{i % 200}", f"m{(i + 1) % 200}")
        )
        for i in range(300)
    ]
    sizes = {f"m{i}": (i + 1) * GB for i in range(200)}
    started = time.monotonic()
    plans = policy.plan_groups(recipes, sizes, {f"m{i}" for i in range(0, 200, 9)})
    assert time.monotonic() - started < 1.0
    assert sorted(k for p in plans for k in p.recipes) == sorted(
        r.key for r in recipes
    )  # every recipe exactly once
