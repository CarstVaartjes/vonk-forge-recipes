"""Best-effort, request-led fleet removal after the last current recipe verdict."""

from collections.abc import Mapping
from random import uniform
from uuid import uuid4

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
from .vonkctl import Vonkctl

RETRY_SECONDS = 60.0
MAX_BACKOFF_SECONDS = 3600.0


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
        for digest in records.keys() - eligible:
            record = RemovalRecord.model_validate(records[digest])
            if record.outcome == RemovalOutcome.RECEIPT:
                if not record.needed_again:
                    record.needed_again = True
                    self._store(digest, record)
            else:
                del records[digest]
                self.state.event(
                    f"model removal {digest}: dropped; consumers need it again"
                )
                self.state.save()
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
            if record.outcome == RemovalOutcome.RECEIPT and (
                record.needed_again or record.consumers != scope
            ):
                history = [*record.history]
                if record.receipt:
                    history.append(record.receipt)
                record = RemovalRecord(history=history)
            record.consumers = scope
            if record.outcome == RemovalOutcome.RECEIPT:
                continue
            if digest in protected:
                if record.request_key is not None:
                    self.state.event(
                        f"model removal {digest}: dropped request {record.request_key}; owner needs it"
                    )
                    record = RemovalRecord(consumers=scope, history=record.history)
                if now < record.next_check:
                    continue
                record.outcome = RemovalOutcome.OWNER
                record.message = "referenced by an owner profile"
                self._schedule(record, now)
                self._store(digest, record)
                continue
            if now < record.next_check:
                continue
            if owner_error:
                record.outcome = RemovalOutcome.OWNER
                record.message = owner_error
                self._schedule(record, now)
                self._store(digest, record)
                continue
            if advanced and not final:
                continue
            self._advance(digest, record, now)
            advanced = True

    def _schedule(self, record: RemovalRecord, now: float) -> None:
        delay = min(MAX_BACKOFF_SECONDS, max(RETRY_SECONDS, record.backoff_seconds))
        record.next_check = now + uniform(delay / 2, delay)
        record.backoff_seconds = min(MAX_BACKOFF_SECONDS, delay * 2)

    def _retry(self, record: RemovalRecord, message: str, now: float) -> None:
        record.message = message
        record.outcome = RemovalOutcome.RETRY
        self._schedule(record, now)

    def _advance(self, digest: str, record: RemovalRecord, now: float) -> None:
        if record.request_key is None:
            review = self.vk.run(
                "model", "remove", digest, "--fleet", "--review", timeout=30
            )
            try:
                parsed = RemovalReview.model_validate(review.document)
                if parsed.target_identity != digest:
                    raise ValueError(
                        "review names different model content; no effects requested"
                    )
                record.review = review.document
                if parsed.blockers:
                    raise ValueError("; ".join(item.detail for item in parsed.blockers))
                if not review.ok:
                    raise ValueError(review.error_text)
            except (ValueError, ValidationError) as error:
                self._retry(record, f"review refused: {error}", now)
                record.last_response = review.document
                self._store(digest, record)
                return
            record.request_key = str(uuid4())
            record.receipt = (
                None  # prior terminal receipts belong to history, not this request
            )
        record.outcome = RemovalOutcome.WAIT
        record.message = "request submitted or reconnecting; waiting for typed receipt"
        self._schedule(record, now)
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
                raise ValueError(
                    "receipt does not bind the original request and model content"
                )
        except (ValueError, ValidationError) as error:
            # A transport loss or refusal can follow acceptance. Never mint a new key
            # from prose or an HTTP status: reconnect to the same owner after backoff.
            record.message = f"receipt unavailable: {reply.error_text}; {error}"[:1024]
        else:
            record.receipt = reply.document
            record.message = f"operation {receipt.operation_id}: {receipt.state}"
            if receipt.state == LifecycleState.SUCCEEDED and receipt.result is not None:
                record.outcome = RemovalOutcome.RECEIPT
                self.state.data["models_done"] = [
                    item for item in self.state.data["models_done"] if item != digest
                ]
            elif receipt.state in TERMINAL:
                record.history.append(record.receipt or {})
                record.request_key = None
                record.outcome = RemovalOutcome.RETRY
                record.message = (
                    receipt.failure.detail if receipt.failure else record.message
                )
        self._store(digest, record)
