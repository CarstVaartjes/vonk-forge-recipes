"""Saved verdicts drive content-bound fleet removal, independent of recipe work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sweep_fakes import FakeModel, FakeRecipe, make_sweep

from spark_sweep.lifecycle import LifecycleState
from spark_sweep.removal_contract import RecipeVerdict, RemovalOutcome, RemovalRecord
from spark_sweep.report import build_status, render_status
from spark_sweep.state import State
from spark_sweep.vonkctl import Vonkctl

FIXTURES = Path(__file__).parent / "fixtures/sweep-removal"
MAX_BACKOFF_SECONDS = 3600.0
A, B, C, D = (letter * 64 for letter in "abcd")


def fixture(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class RemovalPeer:
    def __init__(self, sweep, fleet):
        self.sweep = sweep
        self.fleet = fleet
        self.calls = []
        self.state = LifecycleState.SUCCEEDED
        self.profiles = [
            {
                "number": 20,
                "definition": {
                    "assignments": [{"recipe_selector": "vonk-forge/owner"}]
                },
            }
        ]
        self.lost = False
        self.crash = False
        self.blocked = False
        self.mismatch = False

    def runner(self, argv, timeout):
        args = list(argv)[3:]
        if args == ["profile", "list"]:
            return 0, json.dumps({"profiles": self.profiles}), ""
        if args[:2] != ["model", "remove"]:
            return self.fleet.runner(argv, timeout)
        self.calls.append(args)
        digest = args[2]
        if "--review" in args:
            review = fixture("review")
            review.update(
                selector=digest, target_identity=C if self.mismatch else digest
            )
            if self.blocked:
                review["blockers"] = [fixture("blocker")]
            return 2 if self.blocked else 0, json.dumps(review), ""
        request = args[args.index("--request-key") + 1]
        persisted = State.load(self.sweep.state.path).data["model_removals"][digest]
        assert persisted["request_key"] == request  # write-ahead before effects
        if self.crash:
            self.crash = False
            raise KeyboardInterrupt
        if self.lost:
            self.lost = False
            return 124, "", "connection lost after acceptance"
        receipt = fixture("receipt")
        receipt.update(
            request_key=request, model_content_sha256=digest, state=self.state
        )
        receipt["result"]["removed_entries"] = [digest]
        if self.state != LifecycleState.SUCCEEDED:
            receipt["result"] = None
            receipt["failure"] = fixture("failure")
        return 0, json.dumps(receipt), ""

    @property
    def submissions(self):
        return [args for args in self.calls if "--review" not in args]


def scenario(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(fixture("state")))
    sweep, fleet, clock = make_sweep(
        tmp_path,
        [
            FakeRecipe("a", digests=(A, B)),
            FakeRecipe("b", digests=(A,)),
            FakeRecipe("owner", digests=(C,)),
            FakeRecipe("later", digests=(D,)),
        ],
        [FakeModel(digest, local="cached") for digest in (A, B, C, D)],
    )
    sweep.refresh_catalog()
    peer = RemovalPeer(sweep, fleet)
    sweep.vk = Vonkctl(runner=peer.runner)
    if hasattr(sweep, "model_remover"):
        sweep.model_remover.vk = sweep.vk
    return sweep, peer, clock


def advance(sweep, clock):
    clock.t += MAX_BACKOFF_SECONDS + 1
    sweep.remove_finished_models()


def test_shared_and_secondary_models_wait_for_every_current_verdict(tmp_path):
    sweep, peer, clock = scenario(tmp_path)
    sweep.state.recipes["vonk-forge/b"]["status"] = RecipeVerdict.PENDING
    sweep.remove_finished_models()
    assert [args[2] for args in peer.submissions] == [B]
    sweep.state.recipes["vonk-forge/b"]["status"] = RecipeVerdict.FAILED
    advance(sweep, clock)
    assert [args[2] for args in peer.submissions] == [B, A]
    assert C not in [args[2] for args in peer.submissions]
    assert D not in sweep.state.data["model_removals"]
    assert sweep.state.data["model_removals"][C]["outcome"] == RemovalOutcome.OWNER


@pytest.mark.parametrize(
    "change",
    [
        {"status": RecipeVerdict.PENDING},
        {"status": RecipeVerdict.DEFERRED},
        {"status": RecipeVerdict.SKIPPED},
        {"content_sha256": "old-content"},
        {"not_before": 1e20},
        {"defer_until": 1e20},
    ],
)
def test_nonfinal_or_old_content_never_authorizes_removal(tmp_path, change):
    sweep, peer, _ = scenario(tmp_path)
    sweep.state.recipes["vonk-forge/a"].update(change)
    sweep.remove_finished_models()
    assert not peer.submissions


@pytest.mark.parametrize("number", [1, 2, 3, 20, 87])
def test_all_owner_profile_numbers_protect_every_role(tmp_path, number):
    sweep, peer, _ = scenario(tmp_path)
    peer.profiles = [
        {
            "number": number,
            "definition": {"assignments": [{"recipe_selector": "vonk-forge/a"}]},
        }
    ]
    sweep.remove_finished_models()
    assert all(args[2] not in (A, B) for args in peer.submissions)


@pytest.mark.parametrize("owned", [False, True])
def test_only_a_still_owned_sweep_profile_is_excluded(tmp_path, owned):
    sweep, peer, _ = scenario(tmp_path)
    peer.profiles = [
        {
            "number": sweep.cfg.sweep_profile,
            "definition": {
                "labels": {"purpose": "hardware-sweep"} if owned else {},
                "assignments": [{"recipe_selector": "vonk-forge/a"}],
            },
        }
    ]
    sweep.remove_finished_models()
    assert any(args[2] in (A, B) for args in peer.submissions) is owned


def test_owner_unknown_blocks_only_removal_and_heals(tmp_path):
    sweep, peer, clock = scenario(tmp_path)
    peer.profiles = [{"number": 20, "definition": None}]
    sweep.remove_finished_models()
    assert not peer.calls
    assert "unavailable" in sweep.state.data["model_removals"][A]["message"]
    assert sweep.pending()  # unfinished recipe work remains schedulable
    peer.profiles = []
    advance(sweep, clock)
    assert peer.submissions


@pytest.mark.parametrize("fault", ["lost", "crash"])
def test_restart_reconnects_to_the_write_ahead_request(tmp_path, fault):
    sweep, peer, clock = scenario(tmp_path)
    setattr(peer, fault, True)
    if fault == "crash":
        with pytest.raises(KeyboardInterrupt):
            sweep.remove_finished_models()
    else:
        sweep.remove_finished_models()
    original = sweep.state.data["model_removals"][A]["request_key"]
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.model_remover.state = sweep.state
    advance(sweep, clock)
    assert len(peer.submissions) == 2
    assert all(
        args[args.index("--request-key") + 1] == original for args in peer.submissions
    )
    record = sweep.state.data["model_removals"][A]
    assert record["outcome"] == RemovalOutcome.RECEIPT
    assert record["receipt"]["request_key"] == original
    status = build_status(sweep)
    assert status["model_removals"][A]["receipt"] == record["receipt"]
    assert original in render_status(status)


def test_terminal_failure_replans_but_backoff_retains_the_request(tmp_path):
    sweep, peer, clock = scenario(tmp_path)
    peer.state = LifecycleState.BACKOFF
    sweep.remove_finished_models()
    original = sweep.state.data["model_removals"][A]["request_key"]
    advance(sweep, clock)
    assert sweep.state.data["model_removals"][A]["request_key"] == original
    peer.state = LifecycleState.FAILED
    advance(sweep, clock)
    record = sweep.state.data["model_removals"][A]
    assert record["history"] and record["request_key"] is None
    peer.state = LifecycleState.SUCCEEDED
    advance(sweep, clock)
    assert sweep.state.data["model_removals"][A]["request_key"] != original


def test_a_new_request_never_adopts_the_previous_failure_receipt(tmp_path):
    sweep, peer, clock = scenario(tmp_path)
    peer.state = LifecycleState.FAILED
    sweep.remove_finished_models()
    previous = sweep.state.data["model_removals"][A]["receipt"]
    peer.lost = True
    advance(sweep, clock)
    record = sweep.state.data["model_removals"][A]
    assert record["receipt"] is None
    assert record["history"] == [previous]
    assert record["request_key"] != previous["request_key"]


@pytest.mark.parametrize(
    "fault", ["blocked", "mismatch", "terminal", "lost", "running"]
)
def test_repeated_failures_and_uncertainty_always_schedule_a_next_check(
    tmp_path, fault
):
    sweep, peer, clock = scenario(tmp_path)
    if fault == "terminal":
        peer.state = LifecycleState.FAILED
    elif fault == "running":
        peer.state = LifecycleState.RUNNING
    else:
        setattr(peer, fault, True)
    keys = []
    for _ in range(12):
        peer.lost = fault == "lost"
        clock.t += 24 * 3600  # Controller deadlines never end local reconciliation.
        advance(sweep, clock)
        record = RemovalRecord.model_validate(sweep.state.data["model_removals"][A])
        assert clock.now() < record.next_check <= clock.now() + MAX_BACKOFF_SECONDS
        assert record.message
        status = render_status(build_status(sweep))
        assert str(record.outcome) in status and record.message in status
        assert str(record.next_check) in status
        keys = [args[args.index("--request-key") + 1] for args in peer.submissions]
        sweep.state = State.load(sweep.state.path, clock.now)
        sweep.model_remover.state = sweep.state
    if fault in ("blocked", "mismatch"):
        assert not peer.submissions
        assert record.outcome == RemovalOutcome.RETRY
    elif fault == "terminal":
        assert len(set(keys)) == 12
        assert len([args for args in peer.calls if "--review" in args]) >= 12
    else:
        assert len(keys) == 12 and len(set(keys)) == 1
        assert record.outcome == RemovalOutcome.WAIT
        assert record.request_key is not None
        assert record.request_key in status
    assert sweep.pending()


@pytest.mark.parametrize("change", ["pending", "retrying", "owner"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_eligibility_loss_drops_pending_removal_before_its_next_check(
    tmp_path, change, uncertain
):
    sweep, peer, clock = scenario(tmp_path)
    peer.state = LifecycleState.RUNNING
    peer.lost = uncertain
    sweep.remove_finished_models()
    original = sweep.state.data["model_removals"][A]["request_key"]
    submissions = len([args for args in peer.submissions if args[2] == A])
    if change == "owner":
        peer.profiles[0]["definition"]["assignments"] = [
            {"recipe_selector": "vonk-forge/a"}
        ]
    elif change == "retrying":
        sweep.state.recipes["vonk-forge/a"]["not_before"] = clock.now() + 1e6
    else:
        sweep.state.recipes["vonk-forge/a"]["status"] = RecipeVerdict.PENDING
    sweep.remove_finished_models()
    record = sweep.state.data["model_removals"].get(A)
    assert record is None or record["request_key"] is None
    advance(sweep, clock)
    assert len([args for args in peer.submissions if args[2] == A]) == submissions
    peer.profiles = []
    sweep.state.recipes["vonk-forge/a"].update(
        status=RecipeVerdict.PASSED, not_before=0
    )
    peer.state = LifecycleState.SUCCEEDED
    advance(sweep, clock)
    assert sweep.state.data["model_removals"][A]["request_key"] != original


def test_partial_catalog_and_live_slots_never_authorize_removal(tmp_path):
    sweep, peer, _ = scenario(tmp_path)
    sweep.recipe_list.in_pass = True
    sweep.remove_finished_models()
    assert not peer.calls
    sweep.recipe_list.in_pass = False
    sweep.state.slots["vonk-forge/a"] = {}
    sweep.remove_finished_models()
    assert not peer.submissions


def test_the_scheduler_requests_removal_from_saved_verdicts(tmp_path):
    sweep, peer, clock = scenario(tmp_path)
    # Keep the remaining recipe queued while the normal tick runs all its work.
    sweep.state.entry("vonk-forge/later")["not_before"] = clock.now() + 10000
    sweep.tick()
    assert peer.submissions


def test_final_pass_submits_every_eligible_model_without_waiting(tmp_path):
    sweep, peer, _ = scenario(tmp_path)
    peer.state = LifecycleState.RUNNING
    sweep.remove_finished_models(final=True)
    assert {args[2] for args in peer.submissions} == {A, B}
    assert all(
        item["outcome"] != RemovalOutcome.RECEIPT
        for item in sweep.state.data["model_removals"].values()
    )


def test_retesting_rearms_cleanup_with_new_identity_and_keeps_prior_receipt(tmp_path):
    sweep, _peer, clock = scenario(tmp_path)
    sweep.remove_finished_models()
    original = sweep.state.data["model_removals"][A]["request_key"]
    sweep.state.recipes["vonk-forge/a"]["status"] = RecipeVerdict.PENDING
    advance(sweep, clock)
    sweep.state.recipes["vonk-forge/a"]["status"] = RecipeVerdict.PASSED
    advance(sweep, clock)
    record = sweep.state.data["model_removals"][A]
    assert record["request_key"] != original
    assert record["history"][0]["request_key"] == original


@pytest.mark.parametrize(
    "retryable,http_status",
    [(False, 500), (True, 503), (True, None), (True, "omitted")],
)
def test_review_error_keeps_typed_cause_and_backoff_then_heals(
    tmp_path, retryable, http_status
):
    """Catches parsing a Controller error as a review and losing its cause."""
    from spark_sweep.removal_contract import RemovalReplyError

    sweep, peer, clock = scenario(tmp_path)
    document = fixture("error")
    document["retryable"] = retryable
    if http_status == "omitted":
        del document["http_status"]
    else:
        document["http_status"] = http_status
    fault = RemovalReplyError.model_validate(document)
    original_runner = peer.runner

    def runner(argv, timeout):
        if "--review" in argv:
            peer.calls.append(list(argv)[3:])
            return 2, json.dumps(document), ""
        return original_runner(argv, timeout)

    sweep.model_remover.vk = Vonkctl(runner=runner)
    sweep.remove_finished_models()
    record = RemovalRecord.model_validate(sweep.state.data["model_removals"][A])
    assert record.outcome == RemovalOutcome.RETRY
    assert record.request_key is None
    assert fault.code in record.message
    assert f"http_status={fault.http_status}" in record.message
    assert f"retryable={retryable}" in record.message
    assert fault.detail in record.message
    assert "validation error" not in record.message
    assert not peer.submissions
    assert clock.now() < record.next_check <= clock.now() + record.backoff_seconds
    calls = sum(args[2] == A for args in peer.calls)
    sweep.remove_finished_models()
    assert sum(args[2] == A for args in peer.calls) == calls
    sweep.state = State.load(sweep.state.path, clock.now)
    sweep.model_remover.state = sweep.state
    sweep.model_remover.vk = Vonkctl(runner=original_runner)
    advance(sweep, clock)
    assert peer.submissions
    assert sweep.state.data["model_removals"][A]["outcome"] == RemovalOutcome.RECEIPT
