"""S3 fault injection: retained owners, semantic progress and saved-state recovery."""

from __future__ import annotations

import importlib.util
import io
import itertools
import json
import sys
import threading
import time
from contextlib import AbstractContextManager
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep import policy, verdict
from spark_sweep.lifecycle import LifecycleState
from spark_sweep.smoke import HttpConfig, stream_probe
from spark_sweep.state import State

pytest_plugins = ["sweep_bounds"]
pytestmark = pytest.mark.usefixtures("bounded_clock")

ROOT = Path(__file__).resolve().parents[1]
SAVED = json.loads((ROOT / "tests/fixtures/sweep-s3-recovery.json").read_text())


class Stream(AbstractContextManager):
    """A byte stream that bounds an otherwise infinite heartbeat reproducer."""

    def __init__(self, chunks, clock):
        self.chunks, self.clock = iter(chunks), clock
        self.reads = 0

    def read1(self, _size=0):
        self.reads += 1
        assert self.reads < 30, (
            "the reader kept consuming bytes without semantic output"
        )
        self.clock[0] += 0.25
        return next(self.chunks, b"")

    def __iter__(self):
        while data := self.read1():
            yield data

    def __exit__(self, *_args):
        return False


@pytest.mark.parametrize("chunk", [b": heartbeat\n", b"data: ", b"x"])
def test_heartbeat_and_partial_line_streams_end_on_semantic_inactivity(
    monkeypatch, chunk
):
    clock = [0.0]
    response = Stream(itertools.repeat(chunk), clock)
    monkeypatch.setattr("spark_sweep.smoke.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("spark_sweep.smoke.open_with_body", lambda *_args: response)
    with pytest.raises(TimeoutError, match="semantic output"):
        stream_probe("http://fixture/v1", "a", HttpConfig(case_timeout=1))
    assert response.reads <= 4


def test_a_slow_useful_stream_outlives_many_inactivity_windows(monkeypatch):
    clock = [0.0]
    token = b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
    response = Stream([token] * 12 + [b"data: [DONE]\n"], clock)
    progress = []
    monkeypatch.setattr("spark_sweep.smoke.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("spark_sweep.smoke.open_with_body", lambda *_args: response)
    result = stream_probe(
        "http://fixture/v1",
        "a",
        HttpConfig(case_timeout=1, on_progress=lambda: progress.append(clock[0])),
    )
    assert result["tokens"] == len(progress) == 12
    assert result["total_ms"] >= 3000


def test_typed_download_rejection_is_not_replayed_after_saved_state_reload(tmp_path):
    recipe = FakeRecipe("a")
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    fleet.observations.append((("recipe", "download"), 2, SAVED["refusal"]))
    sweep.prefetcher._request(sweep.recipes[recipe.key], "model")
    sweep.state.save()
    restored, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    restored.refresh_catalog()
    for _ in range(5):
        clock.sleep(600)
        restored.prefetcher._poll()
        assert not restored.prefetcher._can_request(
            restored.recipes[recipe.key], clock.now()
        )
    assert restored.state.status(recipe.key) == LifecycleState.FAILED
    record = restored.state.downloads[recipe.key]
    assert record["refusal"] == SAVED["refusal"]
    assert (
        len([call for _, call in fleet.calls if call[:2] == ("recipe", "download")])
        == 1
    )
    old_request = record["request_key"]
    restored.requeue_failed()  # a new explicit owner request authorizes one fresh retest
    assert restored.prefetcher._can_request(restored.recipes[recipe.key], clock.now())
    restored.prefetcher._request(restored.recipes[recipe.key], "model")
    assert restored.state.downloads[recipe.key]["request_key"] != old_request


def test_uncertain_recipe_recovery_after_the_old_attempt_limit_and_restart(tmp_path):
    recipe = FakeRecipe("a")
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.state.entry(recipe.key).update(SAVED["uncertain_recipe"])
    sweep.state.save()
    restored, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    restored.refresh_catalog()
    for _ in range(5):
        restored._record_failure(
            recipe.key,
            policy.classify("load", "controller.transport_timeout", "connection reset"),
            None,
        )
        entry = restored.state.entry(recipe.key)
        assert entry["status"] == "pending" and restored._open(recipe.key)
        assert clock.now() < entry["not_before"] <= clock.now() + 600
        clock.sleep(600)
    # A definite later outcome is still recorded, even after the former limit.
    restored._record_failure(
        recipe.key, policy.classify("start", "recipe.runtime_oom", "OOM"), None
    )
    assert restored.state.status(recipe.key) == LifecycleState.FAILED


def test_missing_cache_recovery_is_evidence_based_after_many_rerequests(tmp_path):
    recipe = FakeRecipe("a")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.state.downloads[recipe.key] = dict(SAVED["cache_recovery"])
    sweep.state.save()
    sweep.prefetcher.state = State.load(tmp_path / "state.json", clock.now)
    sweep.prefetcher.library_fresh_since = clock.now()
    assert sweep.prefetcher._can_request(sweep.recipes[recipe.key], clock.now())


def test_prefetch_uncertainty_remains_retryable_beyond_old_limit(tmp_path):
    recipe = FakeRecipe("a")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[recipe.key], "model")
    record = sweep.state.downloads[recipe.key]
    document = {
        "id": record["operation_id"],
        "state": LifecycleState.FAILED,
        "failure": {
            "code": "controller.transport_timeout",
            "detail": "connection reset",
        },
    }
    for _ in range(8):
        sweep.prefetcher._failed(recipe.key, record, document)
        assert record["retry_at"] > clock.now()
        assert sweep.state.status(recipe.key) == "pending"
        clock.sleep(600)


def test_catalog_read_timeout_does_not_hold_live_observation_or_status(
    tmp_path, monkeypatch
):
    recipe = FakeRecipe("a", local="cached")
    sweep, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.refresh_catalog()
    from spark_sweep.run import DaemonExecutor

    sweep.catalog_executor = DaemonExecutor()
    entered, release = threading.Event(), threading.Event()
    original = sweep.vk.runner

    def page(argv, timeout):
        if "recipe" in argv and "library" in argv:
            entered.set()
            if not release.wait(3):
                return 124, "", "fixture catalog page timed out"
        return original(argv, timeout)

    observed, statuses = [], []
    monkeypatch.setattr(sweep.vk, "runner", page)
    monkeypatch.setattr(sweep, "observe_lanes", lambda: observed.append(True))
    monkeypatch.setattr(
        "spark_sweep.run.write_status", lambda *_a, **_k: statuses.append(True)
    )
    sweep.recipe_list.pass_done_at = -1e18
    started = time.monotonic()
    try:
        sweep.tick()
        assert time.monotonic() - started < 1
        assert entered.wait(1)
        future = sweep.catalog_future
        for _ in range(5):
            sweep.tick()
            assert sweep.catalog_future is future  # one owner, no thread fan-out
        assert observed and statuses
        assert time.monotonic() - started < 1
    finally:
        release.set()
        future = getattr(sweep, "catalog_future", None)
        if future is not None:
            future.result(timeout=2)


def test_held_wait_age_and_typed_next_check_survive_json_roundtrips():
    signal = verdict.application_signal(SAVED["held_application"])
    watch = {}
    for hour in range(48):
        assert (
            verdict.progress_verdict(signal, watch, window=600, now=hour * 3600)
            is verdict.Progress.WAITING
        )
        watch = json.loads(json.dumps(watch))
        assert "wait" in watch
        wait = watch["wait"]
        assert wait["first_seen_at"] == wait["last_progress_at"] == 0
        assert wait["observed_at"] == hour * 3600 < wait["next_check_at"]
        assert wait["owner"] and wait["reason"] and wait["next_check"]
    assert watch["wait"]["reconcile_at"] == 600
    assert watch["wait"]["next_check"] == "reconcile-request"


ADAPTERS = [
    *sorted((ROOT / "adapters/media").glob("comfyui-*/comfyui_job.py")),
    ROOT / "adapters/video/drowzeys-fasth3-tensorfold/comfyui_job.py",
]


@pytest.fixture(params=ADAPTERS, ids=lambda path: path.parent.name)
def adapter(request):
    path = request.param
    wheel = next(path.parent.glob("*.whl"), None)
    if wheel is not None:
        sys.path.insert(0, str(wheel))
    try:
        spec = importlib.util.spec_from_file_location("s3_comfy_job", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if wheel is not None:
            sys.path.remove(str(wheel))


def test_transient_history_read_recovers_the_same_accepted_prompt(adapter, monkeypatch):
    clock = [0.0]
    reads = []

    def request(path):
        reads.append(path)
        if len(reads) <= 4:
            raise HTTPError(path, 503, "unavailable", Message(), io.BytesIO())
        return SAVED["history_complete"]

    monkeypatch.setattr(adapter, "request_json", request)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        adapter.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    adapter.wait_for_prompt("accepted-prompt", SimpleNamespace(poll=lambda: None), 60)
    assert reads == ["/history/accepted-prompt"] * 5
    assert clock[0] > 0


@pytest.mark.parametrize("status", [401, 422])
def test_deterministic_history_rejection_is_prompt(adapter, monkeypatch, status):
    def request(path):
        raise HTTPError(path, status, "refused", Message(), io.BytesIO())

    monkeypatch.setattr(adapter, "request_json", request)
    with pytest.raises(HTTPError):
        adapter.wait_for_prompt(
            "accepted-prompt", SimpleNamespace(poll=lambda: None), 60
        )


@pytest.mark.parametrize("restore", [None, 2])
@pytest.mark.parametrize("accepted", [False, True])
def test_final_fleet_intent_keeps_one_request_across_loss_and_restart(
    tmp_path, restore, accepted
):
    recipe = FakeRecipe("a", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")], restore_owner=restore
    )
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.data["dirty"] = True
    native = fleet.runner
    attempts = []

    def runner(argv, timeout):
        if "load" in argv and "--yes" in argv:
            attempts.append(argv[argv.index("--request-key") + 1])
            if len(attempts) == 1:
                if accepted:
                    native(argv, timeout)
                return (
                    2,
                    json.dumps(
                        {"code": "controller.transport_timeout", "retryable": True}
                    ),
                    "lost reply",
                )
        return native(argv, timeout)

    def crash(_now):
        raise KeyboardInterrupt

    sweep.vk.runner = runner
    clock.hooks.append(crash)
    with pytest.raises(KeyboardInterrupt):
        sweep._leave_fleet()
    assert sweep.state.data["dirty"]
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["finalization"]["request_key"] == attempts[0]
    clock.hooks.clear()
    resumed, _, _ = make_sweep(
        tmp_path,
        [recipe],
        [FakeModel("m1", local="cached")],
        fleet=fleet,
        restore_owner=restore,
    )
    resumed.vk.runner = runner
    resumed.refresh_catalog()
    resumed._leave_fleet()
    assert not resumed.state.data["dirty"]
    assert "finalization" not in resumed.state.data
    assert len(attempts) == (1 if accepted else 2)
    assert set(attempts) == {saved["finalization"]["request_key"]}
    own = next(
        row
        for row in resumed.state.data["own_loads"]
        if row["request_key"] == attempts[0]
    )
    assert own["terminal_receipt"]["state"] == LifecycleState.SUCCEEDED


def test_canonical_acceptance_outranks_a_refusal_shaped_lost_reply(tmp_path):
    recipe = FakeRecipe("a")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    native = fleet.runner

    def runner(argv, timeout):
        reply = native(argv, timeout)
        if "download" in argv:
            return 2, json.dumps(SAVED["refusal"]), "lost acceptance reply"
        return reply

    sweep.vk.runner = runner
    sweep.prefetcher._request(sweep.recipes[recipe.key], "model")
    assert sweep.state.status(recipe.key) == "pending"
    sweep.prefetcher._poll()
    record = sweep.state.downloads[recipe.key]
    assert record["operation_id"] and record["state"] == LifecycleState.RUNNING
    assert "refusal" not in record
    assert sweep.state.status(recipe.key) == "pending"


def test_lane_supervision_retries_a_stuck_stream_without_a_total_duration_cap(
    tmp_path, monkeypatch
):
    from concurrent.futures import Future

    from spark_sweep.smoke import SemanticProgress

    recipe = FakeRecipe("a", local="cached")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    slot["phase"] = "smoking"
    sweep.futures[recipe.key] = Future()
    observation = SemanticProgress(0)
    sweep.smoke_progress[recipe.key] = observation
    monotonic = [0.0]
    monkeypatch.setattr("spark_sweep.smoke.time.monotonic", lambda: monotonic[0])
    for _ in range(8):
        monotonic[0] += sweep.cfg.http.case_timeout
        sweep._note_smoke_progress(slot, observation)
        sweep._advance_smoking(recipe.key, slot)
        assert slot["phase"] == "smoking"
    assert slot["smoke_watch"]["wait"]["next_check"] == "observe-output"
    monotonic[0] += 2 * sweep.cfg.http.case_timeout + 1
    sweep._advance_smoking(recipe.key, slot)
    assert slot["phase"] == "finished"
    assert sweep.state.status(recipe.key) == "pending"
    assert recipe.key not in sweep.futures


def test_an_unchanged_held_lane_escalates_the_same_request_and_publishes_its_age(
    tmp_path,
):
    from spark_sweep.report import build_status

    recipe = FakeRecipe("a", local="cached")
    sweep, _, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.queue = list(sweep.recipes)
    sweep.cached_models = sweep._cached_models()
    sweep.schedule(clock.now())
    slot = sweep.state.slots[recipe.key]
    request = slot["request_key"]
    sweep.state.data["loads"][request]["signal"] = {
        "signature": ["held"],
        "waiting_on": "dependency",
    }
    first_seen = clock.now()
    sweep._observe_load(slot, first_seen)
    clock.sleep(sweep._load_window(slot) + 1)
    sweep._observe_load(slot, clock.now())
    wait = slot["watch"]["wait"]
    assert wait["first_seen_at"] == wait["last_progress_at"] == first_seen
    assert wait["last_reconciled_at"] == clock.now() < wait["reconcile_at"]
    assert f"held:{request}" in sweep.state.data["infra"]
    sweep.state.save()
    saved = State.load(tmp_path / "state.json", clock.now)
    assert saved.slots[recipe.key]["watch"]["wait"] == wait
    lanes = [lane for items in build_status(sweep)["lanes"].values() for lane in items]
    assert lanes and all(lane["wait"] == wait for lane in lanes)
    assert slot["request_key"] == request


@pytest.mark.parametrize("ending", ["budget", "process", "workflow"])
def test_history_retry_respects_the_job_owner_and_actual_workflow_failure(
    adapter, monkeypatch, ending
):
    clock = [0.0]
    reads = []

    def request(path):
        reads.append(path)
        if ending == "workflow":
            return {
                "accepted-prompt": {
                    "status": {"status_str": "error", "messages": ["bad workflow"]}
                }
            }
        raise HTTPError(path, 503, "unavailable", Message(), io.BytesIO())

    def poll():
        return 7 if ending == "process" and reads else None

    monkeypatch.setattr(adapter, "request_json", request)
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        adapter.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    expected = TimeoutError if ending == "budget" else RuntimeError
    with pytest.raises(expected):
        adapter.wait_for_prompt(
            "accepted-prompt", SimpleNamespace(poll=poll, returncode=7), 10
        )
    assert 1 <= len(reads) <= 4
    assert all(path == "/history/accepted-prompt" for path in reads)


def test_a_previously_exhausted_uncertainty_resumes_automatically_from_saved_state(
    tmp_path,
):
    recipe = FakeRecipe("a")
    sweep, fleet, clock = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    sweep.prefetcher._request(sweep.recipes[recipe.key], "model")
    sweep.state.downloads[recipe.key].update(
        state=LifecycleState.FAILED, observe_failures=17
    )
    sweep.state.entry(recipe.key).update(SAVED["deferred_uncertain"])
    sweep.state.save()
    resumed, _, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")], fleet=fleet)
    resumed.refresh_catalog()
    original_key = resumed.state.downloads[recipe.key]["request_key"]
    resumed.requeue_for_rules()
    assert resumed._open(recipe.key)
    assert resumed.state.entry(recipe.key)["attempts"] == 17
    assert resumed.state.downloads[recipe.key]["request_key"] == original_key
    assert (
        clock.now()
        < resumed.state.downloads[recipe.key]["retry_at"]
        <= clock.now() + 600
    )


@pytest.mark.parametrize("bad_receipt", ["wrong-id", "missing-id", "absent"])
def test_finalization_needs_its_exact_receipt_after_acceptance(tmp_path, bad_receipt):
    recipe = FakeRecipe("a", local="cached")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.data["dirty"] = True
    native = fleet.runner
    submissions, observations = [], []

    def runner(argv, timeout):
        reply = native(argv, timeout)
        if "load" in argv and "--yes" in argv:
            submissions.append(argv[-1])
        if "progress" in argv and "--request-key" in argv:
            observations.append(argv[-1])
            if len(observations) == 1:
                if bad_receipt == "absent":
                    return 2, json.dumps({"code": "controller.not_found"}), ""
                document = json.loads(reply[1])
                if bad_receipt == "missing-id":
                    document.pop("id")
                else:
                    document["id"] = "another-application"
                document["state"] = LifecycleState.SUCCEEDED
                return 0, json.dumps(document), ""
        return reply

    sweep.vk.runner = runner
    sweep._leave_fleet()
    assert len(submissions) == 1 and len(observations) >= 2
    assert set(observations) == set(submissions)
    assert not sweep.state.data["dirty"]


def test_a_new_owner_load_supersedes_final_stop_without_another_write(
    tmp_path, monkeypatch
):
    from spark_sweep.owner import OwnerStatus

    recipe = FakeRecipe("a", local="cached")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1", local="cached")])
    sweep.preflight()
    sweep.refresh_catalog()
    sweep.state.data["dirty"] = True
    monkeypatch.setattr(
        sweep.guard, "check", lambda _now: OwnerStatus(new_activity=True)
    )
    sweep._leave_fleet()
    assert not sweep.state.data["dirty"]
    assert not any(call[:3] == ("profile", "load", "--yes") for _, call in fleet.calls)


def test_final_stop_keeps_owning_a_queued_offline_effect_until_recovery(tmp_path):
    recipe = FakeRecipe("a", local="cached")
    sweep, fleet, clock = make_sweep(
        tmp_path, [recipe], [FakeModel("m1", local="cached")]
    )
    sweep.preflight()
    sweep.refresh_catalog()
    fleet.runs.append(
        {
            "alias": "external",
            "recipe": "external",
            "run_id": "offline-run",
            "node_ids": ["spk_b"],
            "ready": True,
        }
    )
    fleet.offline_nodes.add("spk_b")
    sweep.state.data["dirty"] = True
    waiting = []

    def recover(_now):
        final = sweep.state.data.get("finalization")
        if final:
            assert sweep.state.data["dirty"]
            waiting.append(final["request_key"])
            if len(waiting) >= 3:
                fleet.offline_nodes.clear()

    clock.hooks.append(recover)
    sweep._leave_fleet()
    assert len(waiting) >= 3 and len(set(waiting)) == 1
    assert not sweep.state.data["dirty"]
    assert not any(run["run_id"] == "offline-run" for run in fleet.runs)
    submissions = [
        call for _, call in fleet.calls if call[:3] == ("profile", "load", "--yes")
    ]
    assert len(submissions) == 1


def test_a_refusal_of_sweep_owned_input_settles_without_recipe_blame(tmp_path):
    recipe = FakeRecipe("a")
    sweep, fleet, _ = make_sweep(tmp_path, [recipe], [FakeModel("m1")])
    sweep.refresh_catalog()
    refusal = {**SAVED["refusal"], "detail": "request is invalid: $.force"}
    fleet.observations.append((("recipe", "download"), 2, refusal))
    sweep.prefetcher._request(sweep.recipes[recipe.key], "model")
    sweep.prefetcher._poll()  # canonical absence proves rejection before acceptance
    entry = sweep.state.entry(recipe.key)
    assert entry["status"] == "deferred"
    assert entry["code"] == verdict.RecoveryReason.REQUEST_REFUSED
    assert entry["evidence"]["refusal"] == refusal
    assert not sweep.prefetcher._can_request(
        sweep.recipes[recipe.key], sweep.clock.now()
    )
