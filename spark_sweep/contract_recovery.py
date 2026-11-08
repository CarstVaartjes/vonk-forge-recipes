"""A Controller on another control contract: update the client once, retry, else say so and wait.

vonkctl answers ``client.contract_mismatch`` (decision ``exit``) when the Controller runs a newer
control contract than the installed client (vonk-forge #2170). The typed code is the only input;
no message is matched. The state machine has three states and one transition function:

    ALIGNED --mismatch, update available, apply ok--> UPDATED   (retry the failed call once)
    ALIGNED --mismatch, no update / apply failed----> STUCK     (alert, no second command)
    UPDATED --mismatch again------------------------> STUCK     (the one apply per process is spent)
    STUCK   --the failed command succeeds-----------> ALIGNED   (someone fixed it; alert cleared)
    STUCK   --mismatch, ``RECHECK_SECONDS`` later----> re-read ``vonkctl update`` (read-only);
              an update that appeared is applied only if this process has not applied one yet.

While STUCK the callers get the mismatch reply, which ``verdict.reply_disposition`` treats as a
wait (never a recipe verdict); the alert is the sweep's infrastructure incident with escalation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum

from .vonkctl import Reply

#: A stuck mismatch looks again (read-only) no more often than this.
RECHECK_SECONDS = 300.0


class ContractState(StrEnum):
    ALIGNED = "aligned"
    UPDATED = "updated"
    STUCK = "stuck"


class ContractRecovery:
    def __init__(
        self,
        now: Callable[[], float],
        alert: Callable[[str], None],
        clear: Callable[[], None],
        updated: Callable[[], None],
    ) -> None:
        self._now = now
        self._alert = alert
        self._clear = clear
        self._updated = updated
        self.state = ContractState.ALIGNED
        self.applied = False
        self._checked_at = float("-inf")
        #: The commands (first two words) that met a mismatch and have not succeeded since.
        self._failed: set[tuple[str, ...]] = set()

    def recover(self, run_once: Callable[..., Reply], command: tuple[str, ...]) -> bool:
        """The one transition for a typed mismatch; True means "retry the failed call"."""
        self._failed.add(command)
        if self.state is ContractState.UPDATED:
            return self._stuck(
                "vonkctl was updated once and the Controller still names another contract"
            )
        now = self._now()
        if (
            self.state is ContractState.STUCK
            and now < self._checked_at + RECHECK_SECONDS
        ):
            return False
        self._checked_at = now
        check = run_once("update", timeout=60)  # read-only
        document = check.document if isinstance(check.document, Mapping) else {}
        if not check.ok or document.get("update_available") is not True:
            return self._stuck(
                "no vonkctl update is available (this vonkctl may be newer than the Controller, "
                f"or the check failed: {check.error_text[:120] if not check.ok else 'up to date'})"
            )
        if self.applied:
            return self._stuck(
                "vonkctl update was already applied by this sweep process"
            )
        self.applied = True
        upgraded = run_once("update", "--apply", timeout=120)
        result = upgraded.document if isinstance(upgraded.document, Mapping) else {}
        if not (upgraded.ok and result.get("updated") is True):
            return self._stuck(
                f"vonkctl update --apply did not apply: {upgraded.error_text[:120]}"
            )
        self.state = ContractState.UPDATED
        self._clear()
        self._updated()
        return True

    def observed_ok(self, command: tuple[str, ...]) -> None:
        self._failed.discard(command)
        if self.state is ContractState.STUCK and not self._failed:
            self.state = ContractState.ALIGNED
            self._clear()

    def _stuck(self, why: str) -> bool:
        self.state = ContractState.STUCK
        self._alert(
            "the Controller runs another control contract than this vonkctl "
            f"(client.contract_mismatch): {why}. Not a recipe failure: work waits; "
            "run `vonkctl update --apply` or restart the sweep after the client matches."
        )
        return False
