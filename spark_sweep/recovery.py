"""Recovery decisions from observed component identities, never publication pointers.

The installed CLI validates the authenticated platform contract. These small
consumer projections only decide whether a recorded, typed fault has a proven
changed owner; they do not create another deployment or workload authority.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Owner = Literal["client", "api", "worker", "agent"]
_SOURCE = re.compile(r"[0-9a-f]{40}\Z")
_CONTRACT = re.compile(r"[0-9a-f]{64}\Z")
OBSERVATION_SECONDS = 60.0

# This is deliberately closed. ``arguments`` is produced by the local CLI's
# _argument_error, not the API or an agent. Generic application failures,
# transport errors, capacity and preparation waits do not identify a software
# owner. Add a code only with evidence from its actual canonical producer.
FAULT_OWNERS: Mapping[str, Owner] = {"arguments": "client"}


@dataclass(frozen=True)
class Fingerprint:
    owner: Owner
    source_sha: str
    contract_sha256: str

    def record(self) -> dict[str, str]:
        return {
            "owner": self.owner,
            "source_sha": self.source_sha,
            "contract_sha256": self.contract_sha256,
        }


def _identity(owner: Owner, source: Any, contract: Any) -> Fingerprint | None:
    if (
        not isinstance(source, str)
        or not _SOURCE.fullmatch(source)
        or not isinstance(contract, str)
        or not _CONTRACT.fullmatch(contract)
    ):
        return None
    return Fingerprint(owner, source, contract)


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except ValueError:
        return None


def fingerprints(
    platform: Mapping[str, Any], client: Mapping[str, Any], now: float
) -> dict[Owner, Fingerprint | None]:
    """Project the current contract; missing/stale/mixed provenance stays unknown.

    Worker process instances prove which completed loop made an observation,
    but restarting the same code is not a software-fix fingerprint. Agent
    identity is intentionally unknown without its own authenticated contract.
    """
    result: dict[Owner, Fingerprint | None] = {
        "client": None,
        "api": None,
        "worker": None,
        "agent": None,
    }
    observed = _timestamp(platform.get("observed_at"))
    if observed is None or not -5.0 <= now - observed <= OBSERVATION_SECONDS:
        return result
    api = platform.get("api")
    if isinstance(api, Mapping):
        result["api"] = _identity(
            "api", api.get("source_sha"), api.get("control_contract_sha256")
        )
    local = _identity(
        "client", client.get("source_sha"), client.get("control_contract_sha256")
    )
    api_identity = result["api"]
    if (
        local is not None
        and api_identity is not None
        and local.contract_sha256 == api_identity.contract_sha256
    ):
        result["client"] = local
    workers = platform.get("workers")
    if (
        platform.get("worker_issue") is not None
        or not isinstance(workers, list)
        or not workers
    ):
        return result
    identities: set[Fingerprint] = set()
    for worker in workers:
        if not isinstance(worker, Mapping):
            return result
        completed = _timestamp(worker.get("completed_at"))
        instance = worker.get("process_instance_id")
        identity = _identity(
            "worker", worker.get("source_sha"), worker.get("worker_contract_sha256")
        )
        if (
            completed is None
            or not 0 <= observed - completed <= 30.0
            or not isinstance(instance, str)
            or not _CONTRACT.fullmatch(instance)
            or type(worker.get("loop_sequence")) is not int
            or worker["loop_sequence"] < 1
            or identity is None
        ):
            return result
        identities.add(identity)
    if len(identities) == 1:
        result["worker"] = identities.pop()
    return result


def failure_basis(
    code: str, observations: Mapping[Owner, Fingerprint | None]
) -> dict[str, str] | None:
    owner = FAULT_OWNERS.get(code)
    identity = observations.get(owner) if owner is not None else None
    return identity.record() if identity is not None else None


def changed_owner(
    entry: Mapping[str, Any], observations: Mapping[Owner, Fingerprint | None]
) -> dict[str, str] | None:
    """One retest may be scheduled only for a proven change to that fault's owner."""
    code = entry.get("code")
    owner = FAULT_OWNERS.get(code) if isinstance(code, str) else None
    previous = entry.get("recovery_basis")
    if owner is None or not isinstance(previous, Mapping):
        return None
    if previous.get("owner") != owner:
        return None
    old = _identity(owner, previous.get("source_sha"), previous.get("contract_sha256"))
    current = observations.get(owner)
    if old is None or current is None or old == current:
        return None
    basis = current.record()
    history = entry.get("recovery_history", [])
    if not isinstance(history, list):
        return None
    if any(
        isinstance(item, Mapping) and item.get("basis") == basis for item in history
    ):
        return None
    return basis
