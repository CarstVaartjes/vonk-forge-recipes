"""The run loop and recording failures and passes (methods of ``Sweep``)."""

# pyright: reportAttributeAccessIssue=false
# The methods reach the state set in ``Sweep.__init__`` and the sibling mixins through ``self``.
from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from vonk_forge_contracts import (
    QualificationCauseCode,
    RecipeDefinition,
)

from recipe_tooling.qualification.tolerant import InputReason, read_optional

from . import memory, policy, recovery, verdict
from .catalog import (
    Recipe,
)
from .identity import (
    AcceptedRequest,
    IdentityReason,
    content_identity,
    verdict_matches,
)
from .lifecycle import ACTIVE as ACTIVE_APP
from .lifecycle import (
    LifecycleState,
    ReconciliationStatus,
    ReleaseAuthority,
)
from .report import write_status
from .smoke import (
    SmokeResult,
)
from .source_serving import (
    SourceServing,
)
from .state import StateLock
from .sweep_contract import (
    ReviewInfrastructureSource,
    SweepFailureClass,
)
from .sweep_support import (
    CATALOG_SKIP,
    IDENTICAL_EXIT_LIMIT,
    SweepProfileOwnershipError,
    _bounded_document,
    _install_signal_handlers,
    _restore_signal_handlers,
)
from .verdict import Disposition


class SweepFailure:
    def run(self) -> int:
        """Run the sweep. One sweep per state directory; SIGINT and SIGTERM end it cleanly."""
        lock = StateLock(self.cfg.state_dir)
        lock.acquire()
        previous = _install_signal_handlers()
        try:
            try:
                return self._run()
            finally:
                self._flush_evidence()
        finally:
            _restore_signal_handlers(previous)
            lock.release()

    def _flush_evidence(self) -> None:
        """Whatever ends the sweep, its last results are published once, not stranded.

        Optional transport: a failure is logged, never raised over the sweep's result.
        """
        if not (self.cfg.publish_evidence and self.qualification_evidence):
            return
        try:  # a late stop signal must not change the exit code or skip the release
            try:
                self.evidence_publisher.flush(self.clock.now())
            except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - publication never changes the exit code
                self.evidence_publisher.message = f"evidence final batch failed ({type(error).__name__}); results stay in history"
            print(f"sweep: {self.evidence_publisher.message}", flush=True)
            write_status(
                self,  # pyright: ignore[reportArgumentType]
                final=True,
            )  # the last status shows the flush outcome
        except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
            print(
                f"sweep: final status not written ({type(error).__name__})", flush=True
            )

    def _run(self) -> int:
        # Adoption parses no remote state and writes nothing during read-only
        # loads. Publish its evidence only under this run's exclusive lock.
        for row in self.state.adoptions:
            if not self.record_result(
                row.recipe,
                "requeue",
                ReconciliationStatus.REQUEUED,
                reason=row.reason,
                previous=row.previous,
            ):
                return 1
        if self.state.adoptions:
            self.state.save()
            self.state.adoptions.clear()
        self.state.data.pop("end", None)
        self._begin_run()
        # Retired sources may be forgotten, but restarting does not prove that a
        # review/adoption dependency recovered. Its first sighting and budget survive.
        self.state.data["infra"] = {
            source: item
            for source, item in self.state.data["infra"].items()
            if source in ("review", "catalog")
            or source.startswith(f"{ReviewInfrastructureSource.TRANSPORT}:")
        }
        stuck = 0
        try:
            self.check_client()
            if self.state.data.get("finalization"):
                self._leave_fleet()
            self.requeue_uncertain()
            started = False
            unmatched = False
            for _ in itertools.count():
                round_start = self.clock.now()
                try:
                    if not started:
                        if not self._startup(self.preflight):
                            continue
                        self.adopt_in_flight()
                        started = True
                    self.tick()
                    self.observe_source_wait()
                    if self.done():
                        self.state.data.pop("pending_idle", None)
                        self.state.data["alerts"] = []
                        if self.cfg.watch_seconds <= 0:
                            unmatched = self._only_matches_nothing()
                            break
                        self.clock.sleep(self.cfg.watch_seconds)
                        self.recipe_list.pass_done_at = (
                            self.model_list.pass_done_at
                        ) = -1e18
                        continue
                    idle = self._nothing_moves()
                    self.clear_infra("loop")
                    stuck = stuck + 1 if idle else 0
                    if not idle:
                        self.state.data.pop("pending_idle", None)
                    if stuck >= 30:
                        self.state.event(
                            "IDLE: pending work has no runnable lane; observing and retrying"
                        )
                        self.state.data["pending_idle"] = (
                            "IDLE: pending work; sweep remains active"
                        )
                        self.state.data["alerts"] = [self.state.data["pending_idle"]]
                        self._save_quietly()
                        stuck = 0
                except SweepProfileOwnershipError:
                    raise  # authorization boundary: never adopt foreign saved assignments
                except Exception as error:  # noqa: BLE001 - only a stop or a signal ends the loop
                    self.note_infra("loop", f"{type(error).__name__}: {error}")
                    self.state.event(
                        f"unexpected error in the loop, continuing: {type(error).__name__}: {str(error)[:150]}"
                    )
                self.clock.sleep(
                    min(
                        self.cfg.poll_seconds,
                        max(
                            0.0, round_start + self.cfg.poll_seconds - self.clock.now()
                        ),
                    )
                )
        except KeyboardInterrupt:
            self.interrupt()
            self.finish(interrupted=True)
            return 130
        self.finish()
        if unmatched:
            raise RuntimeError(
                f"--only {' '.join(self.cfg.only)} matches no recipe in the catalog"
            )
        return 0

    def _only_matches_nothing(self) -> bool:
        """A complete catalog pass in which ``--only`` selected nothing: a typo, not an empty sweep."""
        return (
            bool(self.cfg.only)
            and bool(self.recipe_list.passes)
            and not self.recipe_list.in_pass
            and not any(self._selected_key(key) for key in self.recipes)
        )

    def _model_preparing(self, recipe: Recipe) -> bool:
        return any(
            (m := self.models.get(d)) is not None and m.local == "preparing"
            for d in recipe.model_digests
        )

    def _nothing_moves(self) -> bool:
        if not self.catalog_loaded:
            return False  # the Controller has not answered yet: keep trying
        busy = bool(self.state.slots) or any(
            r.get("state") in ACTIVE_APP
            or (r.get("retry_at") and r.get("state") == LifecycleState.FAILED)
            for r in self.state.downloads.values()
        )
        external = any(
            self.recipes[k].local == "preparing"
            or self._model_preparing(self.recipes[k])
            for k in self.queue
            if k in self.recipes
        )  # downloads started by someone else still deliver
        waiting = (
            external
            or self.paused
            or self.source_serving is SourceServing.UNSERVED  # a typed bounded wait
            or self.clock.now()
            < max(
                self.prefetcher.pressure_until,
                self.prefetcher.pause_until,
            )
        )
        return not busy and not waiting and not self.ready_candidates()

    # --------------------------------------------------------------- failures

    def _download_failed(
        self, key: str, failure: policy.Failure, op_id: str | None, model_level: bool
    ) -> None:
        accepted = AcceptedRequest.read(self.state.downloads.get(key))
        recipe = accepted.recipe if accepted else None
        self._record_failure(key, failure, op_id, from_download=True)
        if (
            model_level
            and failure.code in policy.RECIPE_FAILURE_CODES
            and recipe is not None
        ):
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster

    def _unplaced_recipe(self, key: str) -> Recipe | None:
        """Only pre-placement failures may describe the currently requested document."""
        return self.recipes.get(key)

    def _skewed(
        self, key: str, failure: policy.Failure, accepted: AcceptedRequest | None
    ) -> policy.Failure:
        """A smoke create/upload or missing-definition refusal of content the source did not author.

        The tested content (the slot's accepted request, else the catalog's) differs from this
        process's fixed authored digest, so the refusal is deterministic for the pair and no
        verdict on the recipe: it ends deferred (platform blame), reopened by a restart that
        re-reads the source or by a content change.
        """
        authored = self.authored.get(key)
        tested = accepted.recipe if accepted else self.recipes.get(key)
        if (
            failure.klass != "recipe-data"
            or failure.phase != "smoke"
            or authored is None
            or tested is None
            or tested.content_sha256 == authored
        ):
            return failure
        return policy.classify(
            "smoke",
            verdict.SKEWED,
            f"the catalog served {tested.content_sha256}, the source authored {authored}; "
            f"refused: {failure.describe()}",
            verdict.SKEWED,
            {
                **dict(failure.evidence),
                "source_sha256": authored,
                "served_sha256": tested.content_sha256,
            },
        )

    def _record_failure(
        self,
        key: str,
        failure: policy.Failure,
        op_id: str | None,
        *,
        inherited_from: str | None = None,
        from_download: bool = False,
        document: Mapping[str, Any] | None = None,
    ) -> None:
        """Record what an attempt came to; only an OBSERVE outcome spends an attempt.

        WAIT (the platform or a dependency is not ready) puts the recipe back without an
        attempt; OBSERVE retains ownership with capped backoff; FAIL records at once. Every outcome
        writes a results row with its evidence.
        """
        now = self.clock.now()
        entry = self.state.entry(key)
        slot = None if from_download else self.state.slots.get(key)
        accepted = AcceptedRequest.read(
            self.state.downloads.get(key) if from_download else slot
        )
        recipe = (
            accepted.recipe
            if accepted
            else (None if slot or from_download else self._unplaced_recipe(key))
        )
        failure = self._skewed(key, failure, accepted)
        outcome = verdict.disposition(failure)
        if slot is not None and accepted is None and outcome is not Disposition.SKIP:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        if (
            slot is not None
            and failure.phase != "review"
            and self.owner_overlapped(slot)
        ):
            self.requeue(
                key,
                f"{failure.phase} failure overlapped an owner profile load",
                failure=failure,
            )
            return
        if slot is not None:
            slot["phase"] = "finished"
        if outcome is Disposition.WAIT:
            entry.update(
                status="pending",
                not_before=now + self.cfg.retry_delay,
                last_failure=failure.signature,
            )
            self.record_result(
                key,
                failure.phase,
                "waiting",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"wait {key}: {failure.klass} ({failure.phase})")
            return
        if outcome is Disposition.SKIP:
            entry.update(
                status="skipped",
                reason=failure.describe(),
                reason_kind=CATALOG_SKIP,
            )
            self.record_result(
                key,
                failure.phase,
                "skipped",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"skipped {key}: {failure.describe()[:160]}")
            return
        if outcome is Disposition.OBSERVE and recipe is not None:
            failure = self._bound_identical_exit(entry, failure, recipe)
            outcome = verdict.disposition(failure)
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        if outcome is Disposition.OBSERVE:
            entry.update(
                status="pending",
                not_before=now
                + verdict.retry_delay(self.cfg.retry_delay, entry["attempts"]),
                last_failure=failure.signature,
            )
            self.record_result(
                key,
                failure.phase,
                "retry",
                reason=failure.describe(),
                disposition=outcome.value,
                failure_class=failure.klass,
                code=failure.code,
                evidence=dict(failure.evidence),
                slot=slot,
                document=document,
            )
            self.state.event(f"retry {key}: {failure.klass} ({failure.phase})")
            return
        entry["evidence_seq"] = int(entry.get("evidence_seq", 0)) + 1
        bundle = (
            read_optional(
                f"evidence/{key}",
                lambda: self._evidence(key, op_id, entry["evidence_seq"]),
                lambda diagnostic: entry.update(
                    diagnostics=[diagnostic.model_dump(mode="json")]
                ),
                InputReason.EVIDENCE_UNAVAILABLE,
            )
            if inherited_from is None
            else None
        )
        proof = dict(failure.evidence)
        if op_id:
            proof.setdefault("operation_id", op_id)
            proof["evidence_command"] = f"vonkctl fleet evidence {op_id}"
        # The release the work started on (a load's slot or a download's record), not the
        # one now deployed; none recorded (platform unreadable then) heals once on the next.
        started = self.state.downloads.get(key) if from_download else slot
        release = (
            (started or {}).get("release")
            if from_download or slot is not None
            else self.release_sha
        )
        attributable = (
            verdict.blame(failure.klass, failure.code) is verdict.Blame.RECIPE
        )
        status = "failed" if attributable else "deferred"
        entry.update(
            status=status,
            phase=failure.phase,
            failure_class=failure.klass,
            code=failure.code,
            error=failure.describe(),
            evidence=proof,
            signature=failure.signature,
            cluster=failure.cluster,
            finished_at=now,
            release=release,
            release_authority=ReleaseAuthority.DEPLOYED_CONTROLLER if release else None,
            recovery_basis=recovery.failure_basis(
                failure.code, self._component_observations()
            ),
            rules=policy.LOAD_RULES,
            smoke_rules=policy.SMOKE_RULES,
            content_sha256=recipe.content_sha256
            if recipe
            else entry.get("content_sha256"),
            revision_id=recipe.revision_id if recipe else entry.get("revision_id"),
            verdict_identity=content_identity(recipe).model_dump(mode="json")
            if recipe
            else entry.get("verdict_identity"),
        )
        if inherited_from:
            entry["inherited_from"] = inherited_from
        if bundle:
            entry["evidence_bundle"] = bundle
        if failure.model_level and attributable and recipe is not None:
            for digest in recipe.model_digests:
                self.state.data["model_failures"][digest] = failure.cluster
        self.record_result(
            key,
            "smoke",
            status,
            error=f"{failure.phase}/{failure.klass}: {failure.describe()}"[:1536],
            disposition=outcome.value,
            slot=slot,
            document=document,
        )
        self.state.event(f"{status.upper()} {key}: {failure.phase}/{failure.klass}")
        self._reconcile_completed(key)

    def _bound_identical_exit(
        self, entry: dict[str, Any], failure: policy.Failure, recipe: Recipe
    ) -> policy.Failure:
        """The same content identity exiting the same way three times in a row is its verdict.

        The run of identical facts lives on the recipe's entry: a platform exit's parsed cause and
        trigger, or the class and code of an outcome no producer named. Another cause, another class
        or a changed content identity restarts the count.
        """
        cause = failure.evidence.get("exit_cause")
        if failure.klass == SweepFailureClass.PLATFORM_EXIT and cause:
            # Parsed exit facts are compared; free text (a long Controller prefix) is not.
            facts = {
                "exit_cause": cause,
                "exit_trigger": failure.evidence.get("exit_trigger"),
            }
        elif failure.klass not in verdict.DECLARED_CLASSES:
            # A class nobody named is observed; the same one on the same identity ends too.
            facts = {"klass": failure.klass, "code": failure.code}
            if failure.klass == verdict.SWEEP_CRASH:
                # One code covers every exception: the type and message tell crashes apart.
                facts["crash"] = failure.detail
        else:
            entry.pop("exit_repeat", None)
            return failure
        key = {"identity": content_identity(recipe).model_dump(mode="json"), **facts}
        seen = entry.get("exit_repeat") or {}
        count = int(seen.get("count", 0)) + 1 if seen.get("key") == key else 1
        if count < IDENTICAL_EXIT_LIMIT:
            entry["exit_repeat"] = {"key": key, "count": count}
            return failure
        # The outcome is written: a retest starts at zero.
        entry.pop("exit_repeat", None)
        if "exit_cause" not in facts:
            # An unnamed class keeps its own class and code (no recipe cause is invented): it is
            # recorded as the platform's, deferred, and reopened by a Controller release change (and, for
            # a sweep crash, by a change of the sweep's own source).
            return replace(
                failure, evidence={**failure.evidence, "identical_exits": count}
            )
        return policy.classify(
            failure.phase,
            QualificationCauseCode.RUNTIME_EXIT,
            failure.detail,
            evidence={**failure.evidence, "identical_exits": count},
        )

    def _reconcile_completed(self, key: str) -> None:
        # Re-plan now: a one-shot sweep may exit before another catalog tick.
        recipe = self.recipes.get(key)
        if recipe is not None:
            self._reconcile_entry(recipe, self.fleet_capacity())

    def record_result(
        self,
        key: str | None,
        step: str,
        status: str,
        *,
        slot: Mapping[str, Any] | None = None,
        document: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> bool:
        """The one writer of ``results.jsonl``: every lane exit leaves a row.

        The row carries the recipe's facts and the Controller document its load was last
        seen in, so the evidence of a wait, a retry or a requeue is never only in the
        bounded event ring. A lane's own row (``key`` None) names its Spark instead.
        """
        row: dict[str, Any] = {"batch": "sweep", "step": step, "status": status}
        if key is not None:
            entry = self.state.entry(key)
            slot = slot if slot is not None else self.state.slots.get(key)
            if document is None and slot is not None:
                document = self.documents.get(str(slot.get("request_key", "")))
            row.update(recipe=key, **self._facts(key, entry, slot))
        row.update({k: v for k, v in fields.items() if v is not None})
        if document:
            row["controller_document"] = _bounded_document(document)
        try:
            recorded = self.log.append(**row)
            self._remember_qualification(recorded)
        except OSError as error:
            # The log is the evidence; a full disk is an infrastructure fact to show.
            self.note_infra("results-log", f"{type(error).__name__}: {error}")
            return False
        else:
            self.clear_infra("results-log")
            return True

    def _facts(
        self, key: str, entry: Mapping[str, Any], slot: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        accepted = AcceptedRequest.read(
            slot if slot is not None else self.state.downloads.get(key)
        )
        if (
            slot is None
            and accepted is not None
            and not verdict_matches(entry, accepted.recipe)
        ):
            accepted = None
        recipe = accepted.recipe if accepted else None
        facts: dict[str, Any] = {
            "phase": entry.get("phase"),
            "failure_class": entry.get("failure_class"),
            "code": entry.get("code"),
            "signature": entry.get("signature"),
            "cluster": entry.get("cluster"),
            "attempt": entry.get("attempts"),
            "evidence": entry.get("evidence"),
            "evidence_bundle": entry.get("evidence_bundle"),
            "diagnostics": entry.get("diagnostics"),
            "content_sha256": (recipe.content_sha256 if recipe else None)
            if slot
            else entry.get("content_sha256"),
            "recipe_revision_id": (recipe.revision_id if recipe else None)
            if slot
            else entry.get("revision_id"),
            "library_commit": accepted.library_commit if accepted else None,
            "controller_release": entry.get("release"),
            "controller_version": self.state.data["release"].get("version"),
            "node_count": recipe.node_count if recipe else None,
            "engine": recipe.engine if recipe else None,
            "verdict_identity": content_identity(recipe).model_dump(mode="json")
            if slot and recipe is not None
            else entry.get("verdict_identity"),
        }
        if slot:
            # The revision the Controller's review bound to this load; evidence readers
            # accept the row only when it is the requested one.
            facts["ran_revision"] = slot.get("ran_revision")
            facts["node_ids"] = list(slot.get("node_ids", []))
            facts["sparks"] = list(slot.get("spark_names", []))
            facts["run_id"] = slot.get("run_id")
        if accepted is not None:
            facts["request_key"] = accepted.request_key
            facts["model_digests"] = list(accepted.recipe.model_digests)
        return {k: v for k, v in facts.items() if v is not None}

    def _evidence(self, key: str, op_id: str | None, attempt: int) -> str | None:
        if not op_id:
            return None
        directory = self.cfg.state_dir / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{key.partition('/')[2]}-{attempt}.json"
        reply = self.vk.run("fleet", "evidence", op_id, "--output", str(target))
        if not reply.ok:
            raise ValueError(reply.error_text)
        return str(target)

    def _ran_recipe(self, recipe: Recipe) -> RecipeDefinition | None:
        """The document of exactly the content that ran; None if unreadable.

        Read once per content (the smoke's facts and the pass's budget share it) and held
        until the pass is recorded.
        """
        held = self.ran_recipes.get(recipe.content_sha256)
        if held is not None:
            return held
        reply = self.vk.run("recipe", "detail", recipe.key, "--technical", timeout=60)
        definition = (
            memory.ran_recipe(reply.document, recipe.content_sha256)
            if reply.ok
            else None
        )
        if definition is not None:
            self.ran_recipes[recipe.content_sha256] = definition
        return definition

    def _record_pass(self, key: str, slot: dict[str, Any], result: SmokeResult) -> None:
        now = self.clock.now()
        entry = self.state.entry(key)
        accepted = AcceptedRequest.read(slot)
        if accepted is None:
            self.requeue(key, IdentityReason.UNBOUND)
            return
        recipe = accepted.recipe
        download = self.state.downloads.get(key, {})
        downloaded = AcceptedRequest.read(download)
        if downloaded is None or content_identity(
            downloaded.recipe
        ) != content_identity(recipe):
            download = {}
        timings = {
            "download_s": round(
                float(download["done_at"]) - float(download["started_at"]), 1
            )
            if download.get("done_at") and download.get("started_at")
            else None,
            "load_s": round(float(slot["ready_at"]) - float(slot["started_at"]), 1),
            "smoke_s": result.seconds,
            "total_s": round(now - float(slot["started_at"]), 1),
        }
        entry.update(
            status="passed",
            exit_repeat=None,
            attempts=int(entry.get("attempts", 0)) + 1,
            finished_at=now,
            release=self.release_sha,
            release_authority=ReleaseAuthority.DEPLOYED_CONTROLLER
            if self.release_sha
            else None,
            content_sha256=recipe.content_sha256,
            revision_id=recipe.revision_id,
            verdict_identity=content_identity(recipe).model_dump(mode="json"),
            timings=timings,
            smoke=result.kind,
            perf=result.perf,
            smoke_rules=policy.SMOKE_RULES,
            revalidate=False,
        )
        fleet = self.vk.run("fleet")
        observed = memory.observe(
            fleet.document if fleet.ok else None,
            slot["node_ids"],
            str(slot["alias"]),
            recipe.memory_bytes,
            recipe.load_peak_bytes,
            recipe.minimum_swap_bytes,
        )
        measured = memory.measurement(
            observed,
            self.documents.get(str(slot.get("request_key", ""))),
            str(slot["alias"]),
            self._ran_recipe(recipe) if observed else None,
            recipe.content_sha256,
        )
        self.ran_recipes.pop(recipe.content_sha256, None)
        quality = result.quality_notes + memory.notes(observed)
        if quality:
            entry["quality"] = quality
        else:
            entry.pop("quality", None)
        for stale in (
            "phase",
            "failure_class",
            "signature",
            "cluster",
            "error",
            "evidence",
            "evidence_bundle",
            "inherited_from",
        ):
            entry.pop(stale, None)
        slot["phase"] = "finished"
        self.record_result(
            key,
            "smoke",
            "passed",
            result={"endpoint_alias": slot["alias"], **result.record()},
            memory=observed,
            measured_memory=[row.model_dump(mode="json") for row in measured] or None,
            timings=timings,
            slot=slot,
        )
        notes = len(quality)
        self.state.event(
            f"passed {key}"
            + (f" ({notes} quality note{'s' if notes != 1 else ''})" if notes else "")
        )

        self._reconcile_completed(key)
