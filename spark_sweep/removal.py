"""Best-effort, request-led fleet removal after the last current recipe verdict."""

from collections.abc import Mapping

from pydantic import ValidationError

from .catalog import Recipe
from .lifecycle import TERMINAL, LifecycleState
from .prefetch import SWEEP_LABEL
from .removal_contract import (
    ProfileList,
    RecipeVerdict,
    RemovalOutcome,
    RemovalReceipt,
    RemovalRecord,
    RemovalReview,
)
from .state import State
from .vonkctl import Vonkctl, request_key

MAX_ATTEMPTS = 3
MAX_OBSERVATIONS = 144
RETRY_SECONDS = 60.0
OBSERVE_SECONDS = 600.0
REQUEST_SECONDS = 24 * 3600.0


class ModelRemover:
    def __init__(self, vk: Vonkctl, state: State) -> None:
        self.vk = vk
        self.state = state

    def _store(self, digest: str, record: RemovalRecord) -> None:
        previous = self.state.data.setdefault("model_removals", {}).get(digest)
        value = record.model_dump(mode="json")
        self.state.data["model_removals"][digest] = value
        if previous != value:
            self.state.event(
                f"model removal {digest}: {record.outcome}: {record.message}"
            )
        self.state.save()

    def _owners(self, recipes: Mapping[str, Recipe], own: set[int]) -> set[str]:
        reply = self.vk.run("profile", "list", timeout=30)
        if not reply.ok:
            raise ValueError(f"owner profiles unreadable: {reply.error_text}")
        profiles = ProfileList.model_validate(reply.document)
        protected: set[str] = set()
        for profile in profiles.profiles:
            if (
                profile.number in own
                and profile.definition is not None
                and profile.definition.labels.get(SWEEP_LABEL[0]) == SWEEP_LABEL[1]
            ):
                continue
            if profile.definition is None:
                raise ValueError(
                    f"owner profile {profile.number} definition unavailable"
                )
            for view in profile.assignments:
                if view.model.content_sha256:
                    protected.add(view.model.content_sha256)
            for assignment in profile.definition.assignments:
                recipe = recipes.get(assignment.recipe_selector)
                if recipe is None or not recipe.model_digests:
                    raise ValueError(
                        f"owner profile {profile.number} recipe unavailable"
                    )
                if assignment.model_variant and not any(
                    view.recipe_selector == assignment.recipe_selector
                    and view.model.content_sha256
                    for view in profile.assignments
                ):
                    raise ValueError(
                        f"owner profile {profile.number} variant unavailable"
                    )
                protected.update(recipe.model_digests)
        return protected

    def tick(
        self,
        recipes: Mapping[str, Recipe],
        own: set[int],
        now: float,
        *,
        final: bool = False,
    ) -> None:
        consumers: dict[str, list[Recipe]] = {}
        for recipe in recipes.values():
            for digest in recipe.model_digests:
                consumers.setdefault(digest, []).append(recipe)
        eligible = {
            digest
            for digest, users in consumers.items()
            if all(
                (entry := self.state.recipes.get(recipe.key, {})).get("status")
                in (RecipeVerdict.PASSED, RecipeVerdict.FAILED)
                and bool(recipe.content_sha256)
                and entry.get("content_sha256") == recipe.content_sha256
                and recipe.key not in self.state.slots
                and now >= float(entry.get("not_before", 0))
                and now >= float(entry.get("defer_until", 0))
                for recipe in users
            )
        }
        records = self.state.data.setdefault("model_removals", {})
        for digest in consumers.keys() - eligible:
            if digest in records:
                record = RemovalRecord.model_validate(records[digest])
                if not record.needed_again:
                    record.needed_again = True
                    self._store(digest, record)
        if not eligible:
            return
        try:
            protected = self._owners(recipes, own)
            owner_error = ""
        except (ValueError, ValidationError) as error:
            protected, owner_error = set(), str(error)
        # One model per tick bounds CLI latency; rotate by the persisted next check.
        advanced = False
        for digest in sorted(eligible):
            record = RemovalRecord.model_validate(
                self.state.data.setdefault("model_removals", {}).get(digest, {})
            )
            scope = {recipe.key: recipe.content_sha256 for recipe in consumers[digest]}
            if (
                record.outcome == RemovalOutcome.RECEIPT
                or (
                    record.outcome == RemovalOutcome.EXHAUSTED
                    and record.request_key is None
                )
            ) and (record.needed_again or record.consumers != scope):
                history = [*record.history]
                if record.receipt:
                    history.append(record.receipt)
                record = RemovalRecord(cycle=record.cycle + 1, history=history)
            record.consumers = scope
            if record.outcome in (RemovalOutcome.RECEIPT, RemovalOutcome.EXHAUSTED):
                continue
            if now < record.next_check:
                continue
            if digest in protected or owner_error:
                record.outcome = RemovalOutcome.OWNER
                record.message = owner_error or "referenced by an owner profile"
                record.next_check = now + RETRY_SECONDS
                self._store(digest, record)
                continue
            if advanced and not final:
                continue
            self._advance(digest, record, now)
            advanced = True

    def _retry(self, record: RemovalRecord, message: str, now: float) -> None:
        record.message = message
        record.outcome = (
            RemovalOutcome.EXHAUSTED
            if record.attempt_count >= MAX_ATTEMPTS
            else RemovalOutcome.RETRY
        )
        record.next_check = now + RETRY_SECONDS * max(1, record.attempt_count)

    def _advance(self, digest: str, record: RemovalRecord, now: float) -> None:
        if record.request_key is None:
            record.attempt_count += 1
            review = self.vk.run(
                "model", "remove", digest, "--fleet", "--review", timeout=30
            )
            try:
                parsed = RemovalReview.model_validate(review.document)
                if parsed.target_identity != digest:
                    record.outcome = RemovalOutcome.EXHAUSTED
                    record.message = (
                        "review names different model content; no effects requested"
                    )
                    record.last_response = review.document
                    self._store(digest, record)
                    return
                record.review = review.document
                if parsed.blockers:
                    if any(not blocker.retryable for blocker in parsed.blockers):
                        record.outcome = RemovalOutcome.EXHAUSTED
                        record.message = "; ".join(
                            item.detail for item in parsed.blockers
                        )
                        self._store(digest, record)
                        return
                    raise ValueError("; ".join(item.detail for item in parsed.blockers))
                if not review.ok:
                    raise ValueError(review.error_text)
            except (ValueError, ValidationError) as error:
                self._retry(record, f"review refused: {error}", now)
                record.last_response = review.document
                self._store(digest, record)
                return
            record.request_key = request_key(
                "fleet-model-removal",
                self.state.nonce,
                digest,
                record.cycle,
                record.attempt_count,
            )
            record.receipt = (
                None  # prior terminal receipts belong to history, not this request
            )
            record.deadline = now + REQUEST_SECONDS
            record.observations = 0
        if now >= record.deadline or record.observations >= MAX_OBSERVATIONS:
            record.outcome = RemovalOutcome.EXHAUSTED
            record.message = "observation budget ended; original request may still run; receipt retained"
            self._store(digest, record)
            return
        record.observations += 1
        record.outcome = RemovalOutcome.WAIT
        record.message = "request submitted or reconnecting; waiting for typed receipt"
        record.next_check = now + OBSERVE_SECONDS
        # Write-ahead: even a process death inside run() reconnects to this key.
        self._store(digest, record)
        reply = self.vk.run(
            "model",
            "remove",
            digest,
            "--fleet",
            "--yes",
            "--detach",
            "--request-key",
            record.request_key,
            timeout=30,
        )
        record.last_response = reply.document
        try:
            receipt = RemovalReceipt.model_validate(reply.document)
            if (
                receipt.request_key != record.request_key
                or receipt.model_content_sha256 != digest
            ):
                record.outcome = RemovalOutcome.EXHAUSTED
                record.message = (
                    "receipt does not bind the original request and model content"
                )
                self._store(digest, record)
                return
        except (ValueError, ValidationError) as error:
            # A transport loss or refusal can follow acceptance. Never mint a new key
            # from prose or an HTTP status: reconnect to the same owner, bounded.
            record.message = f"receipt unavailable: {reply.error_text}; {error}"[:1024]
            if record.receipt is None and record.observations >= MAX_ATTEMPTS:
                record.outcome = RemovalOutcome.EXHAUSTED
                record.message += "; reconnect budget ended; original request retained"
        else:
            record.receipt = reply.document
            record.deadline = min(
                record.deadline, receipt.fleet.deadline_at.timestamp()
            )
            record.message = f"operation {receipt.operation_id}: {receipt.state}"
            if receipt.state == LifecycleState.SUCCEEDED and receipt.result is not None:
                record.outcome = RemovalOutcome.RECEIPT
                self.state.data["models_done"] = [
                    item for item in self.state.data["models_done"] if item != digest
                ]
            elif receipt.state in TERMINAL:
                record.history.append(record.receipt or {})
                record.request_key = None
                self._retry(
                    record,
                    receipt.failure.detail if receipt.failure else record.message,
                    now,
                )
                if receipt.failure and not receipt.failure.retryable:
                    record.outcome = RemovalOutcome.EXHAUSTED
        self._store(digest, record)
