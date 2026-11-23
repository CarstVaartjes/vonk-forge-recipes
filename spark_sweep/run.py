"""The sweep itself: lanes over one sweep profile, fed by the prefetcher.

Profiles cover the whole fleet (an unassigned Spark is idle and its workload is
stopped by a load), so the lanes are *assignments of one sweep profile*, not
one profile per lane. A lane swap edits that profile and loads it; the planner
keeps every unchanged assignment (``--review`` is checked to prove it).
Unchanged running and cleanup effects are carried by canonical adoption links,
so an independent lane can progress while another exact child remains pending.

``Sweep.tick`` is a small state machine, so the whole sweep is deterministic
under a fake clock and a fake ``vonkctl``:

    refresh catalog -> owner guard -> advance slots/receipts -> schedule -> prefetch -> status
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from recipe_tooling.qualification.evidence_store import merge_evidence, read_evidence
from vonk_forge_contracts import (
    QualificationIdentity,
    QualificationStatusView,
    RecipeDefinition,
    RecipeQualificationEvidence,
)

from . import policy, recovery
from .catalog import (
    Fleet,
    Listing,
    Model,
    Recipe,
    model_listing,
    recipe_listing,
)
from .definitions import Definitions
from .evidence_publisher import EvidencePublisher
from .fanout import DaemonExecutor, Outcome
from .owner import OwnerGuard, OwnerStatus
from .prefetch import Prefetcher
from .qualification import (
    history_evidence,
    local_status,
)
from .removal import ModelRemover
from .smoke import (
    SmokeResult,
)
from .source_serving import (
    SourceServing,
    Unserved,
)
from .state import ResultsLog, State
from .sweep_adopt import SweepAdopt
from .sweep_apply import SweepApply
from .sweep_cleanup import SweepCleanup
from .sweep_contract import (
    SweepSlotPhase,
)
from .sweep_exit import SweepExit
from .sweep_failure import SweepFailure
from .sweep_fleet import SweepFleet
from .sweep_loading import SweepLoading
from .sweep_platform import SweepPlatform
from .sweep_reconcile import SweepReconcile
from .sweep_schedule import SweepSchedule
from .sweep_smoke import SweepSmoke
from .sweep_support import (
    Clock,
    RealClock,
    SweepConfig,
)
from .sweep_takeover import SweepTakeover
from .sweep_tick import SweepTick
from .vonkctl import Reply, Vonkctl


class Sweep(
    SweepFleet,
    SweepReconcile,
    SweepTick,
    SweepPlatform,
    SweepFailure,
    SweepLoading,
    SweepSmoke,
    SweepSchedule,
    SweepCleanup,
    SweepTakeover,
    SweepApply,
    SweepAdopt,
    SweepExit,
):
    def __init__(
        self,
        config: SweepConfig,
        vk: Vonkctl,
        state: State,
        log: ResultsLog,
        definitions: Definitions,
        clock: Clock | None = None,
        executor: Any = None,
        *,
        authored: Mapping[str, str],
    ) -> None:
        self.cfg = config
        # The recipe content this sweep's source authored (key to document digest).
        self.authored = dict(authored)
        self.source_serving = SourceServing.UNREAD
        self.unserved_recipes: tuple[Unserved, ...] = ()
        self.unserved_keys: frozenset[str] = frozenset()
        # Served recipes this older source does not author (named once in a results row).
        self.source_missing_keys: frozenset[str] = frozenset()
        # The bounded wait's watch lives in the state: a restart continues the budget.
        self.source_watch: dict[str, Any] = state.data["source_watch"]
        # The wait is for this authored content: a restart with other content waits again.
        self.authored_digest = hashlib.sha256(
            json.dumps(sorted(self.authored.items())).encode()
        ).hexdigest()
        self.sweep_source = recovery.sweep_source_digest()
        # The status alert while the source is not served (cleared once it is).
        self.source_alert = ""
        self.vk = vk
        self.state = state
        self.log = log
        self.defs = definitions
        self.clock = clock or RealClock()
        self.executor = executor or DaemonExecutor()
        self.futures: dict[str, Future[SmokeResult]] = {}
        self.smoke_stops: dict[str, threading.Event] = {}
        self.catalog_executor = DaemonExecutor()
        self.read_executor = DaemonExecutor()
        self.catalog_future: Future[Reply] | None = None
        self.catalog_active: Listing[Recipe] | Listing[Model] | None = None
        # The cached listings: scheduling reads these, a refresh updates them page by page.
        self.recipe_list = recipe_listing(vk)
        self.model_list = model_listing(vk)
        self.recipes: dict[str, Recipe] = self.recipe_list.rows
        self.models: dict[str, Model] = self.model_list.rows
        self.fleet = Fleet((), ())
        self.release_at = self.clock.now()  # check_client has just read it
        self.run_started_at = self.clock.now()  # --limit counts verdicts reached since
        self.status_at = -1e18
        self.owner_status = OwnerStatus()
        # Visible application alarms never reserve unrelated lanes.
        self.stall_reason = ""
        self.stalled_applications: set[str] = set()
        self.idle_since: float | None = None
        self.blind = False  # this tick could not observe the Controller
        self.unobserved_loads: set[str] = set()
        # This pass's free Sparks whose placement review did not adopt a loading lane.
        self.adoption_holds: dict[str, str] = {}
        # This pass's free Sparks that no ready recipe was placed on, and why.
        self.fit_holds: dict[str, str] = {}
        self._skew_checked_at = float("-inf")
        self.vk.observer = self
        # Written by worker threads (through ``Vonkctl.run``), read by the main loop only.
        self._contract_lock = threading.Lock()
        self._mismatched: set[tuple[str, ...]] = set()
        self._mismatch_pending = False
        self._mismatch_healed = False
        self.library_commit = ""
        self.rate = policy.RateTracker()
        self.rate.rate = float(state.data["rate"].get("ema", 0.0))
        self.rate.samples = int(state.data["rate"].get("samples", 0))
        self.guard = OwnerGuard(vk, state, config.owner_hold_seconds)
        self._reads: dict[str, Outcome] = {}
        self.model_remover = ModelRemover(vk, state)
        # This is a projection of live catalog/owner observations, never a
        # persisted authorization or a gate to carry into a new process.
        self.state.data.pop("model_removal_wait", None)
        self.prefetcher = Prefetcher(
            vk,
            state,
            config.prefetch,
            self.rate,
            self.clock.now,
            self._download_failed,
            self._download_done,
        )
        self.prefetcher.on_infra = self.note_infra
        self.prefetcher.on_ok = self.clear_infra
        self.prefetcher.release = lambda: self.release_sha
        self.prefetcher.online_lanes = lambda: (
            len(self.sparks()) if self.fleet.sparks else None  # None: not read yet
        )
        self.boost = self._build_boost()
        self.queue: list[str] = []
        self.sizes: dict[str, int] = {}
        self.last_prefetch: dict[str, Any] = {}
        self.cached_models: set[str] = set()
        self.idle_ticks = 0
        self._assigned: dict[str, str] = {}
        self._wait_event_at = -1e18
        # The latest Controller document of each load (by request key): evidence for its rows.
        self.documents: dict[str, Mapping[str, Any]] = {}
        # The recipe documents that ran, by content: written by smoke threads, read on the tick.
        self.ran_recipes: dict[str, RecipeDefinition] = {}
        self.retry_failed_pending = config.retry_failed
        self.qualification_evidence = {
            key: merge_evidence(
                read_evidence(Path(__file__).resolve().parents[1], key),
                RecipeQualificationEvidence.model_validate(value),
            )
            for key, value in state.data.get("qualification_evidence", {}).items()
        }
        self.evidence_publisher = EvidencePublisher(
            log.path,
            config.state_dir / "evidence-publisher",
            Path(__file__).resolve().parents[1] / "tools/sync-qualification-evidence",
            state.path,
        )
        for row in log.entries():
            self._remember_qualification(row)

    def _remember_qualification(self, row: Mapping[str, Any]) -> None:
        try:
            evidence = history_evidence(row)
            if evidence is None:
                return
            key = evidence.recipe_key
            previous = self.qualification_evidence.get(key) or read_evidence(
                Path(__file__).resolve().parents[1], key
            )
            merged = merge_evidence(previous, evidence)
            self.qualification_evidence[key] = merged
            self.state.data.setdefault("qualification_evidence", {})[key] = (
                merged.model_dump(mode="json")
            )
        except (ValueError, TypeError, OSError) as error:
            self.state.event(
                f"qualification evidence unavailable: {type(error).__name__}"
            )

    def _recipe_evidence(self, recipe: Recipe) -> RecipeQualificationEvidence | None:
        try:
            return self.qualification_evidence.get(recipe.key) or read_evidence(
                Path(__file__).resolve().parents[1], recipe.key
            )
        except ValueError, TypeError, OSError:
            return None

    def qualification_view(
        self,
        recipe: Recipe,
        evidence: RecipeQualificationEvidence | None = None,
        identity: QualificationIdentity | None = None,
    ) -> QualificationStatusView | None:
        try:
            evidence = evidence or self._recipe_evidence(recipe)
            if evidence is None:
                return None
            return local_status(
                evidence,
                recipe,
                len(self.enrolled()),
                recipe.key in self.state.slots
                and self.state.slots[recipe.key].get("phase")
                != SweepSlotPhase.FINISHED,
                identity,
            )
        except ValueError, TypeError, OSError:
            return None  # malformed observed identity cannot certify public proof

    # ------------------------------------------------------------------ setup

    def _build_boost(self) -> policy.Boost:
        return policy.make_boost(
            self.cfg.boost, self.cfg.seed, self.clock.now(), self.cfg.recent_days
        )

    def _placement_priority(self) -> policy.Boost:
        return policy.make_priority(self.cfg.boost, self.cfg.seed)
