"""Memory admission shared by offline declarations and live Controller review."""

from pydantic import BaseModel, ConfigDict

# 2 GiB: creators run with 2-3 GiB of steady free memory (0xSero DeepSeek V4.1
# "2-3 GiB available"; Mia DeepSeek V4.1 EXL3 low-water 2.1 GiB at 601k context),
# and the swap-aware host memory guard (vonk-forge #1936, R1) stops a workload
# only below 1 GiB MemAvailable, so this keeps 1 GiB above its trigger.
PLATFORM_MEMORY_FLOOR_BYTES = 2 * 1024**3


class MemoryAdmission(BaseModel):
    """One calculation supplies the decision and its explanatory headroom."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    free_after_bytes: int
    commitment_headroom_bytes: int
    declaration_admissible: bool
    memory_floor_bytes: int

    @property
    def admissible(self) -> bool:
        return (
            self.declaration_admissible
            and self.free_after_bytes >= self.memory_floor_bytes
            and self.commitment_headroom_bytes >= 0
        )


def evaluate_memory_admission(
    demand_bytes: int,
    pool_bytes: int,
    *,
    memory_floor_bytes: int = PLATFORM_MEMORY_FLOOR_BYTES,
    occupied_bytes: int = 0,
    reserved_bytes: int = 0,
    unmaterialized_bytes: int = 0,
    unknown_residual_bytes: int = 0,
    release_bytes: int = 0,
) -> MemoryAdmission:
    """Check observed free memory and accepted commitments independently.

    With only demand and pool, check a declaration on an otherwise empty pool.
    Host occupancy is observed input, never an assumed offline baseline. Demand
    is the resolved peak, not peak plus informational recipe reserve. The floor
    is separate from host occupancy and from the sustained-pressure runtime
    guard. A release credits both live use and commitments; it cannot make a
    declaration larger than the physical pool admissible. Callers validate
    evidence freshness and reconcile exact claim ownership.
    """
    values = (
        demand_bytes,
        pool_bytes,
        memory_floor_bytes,
        occupied_bytes,
        reserved_bytes,
        unmaterialized_bytes,
        unknown_residual_bytes,
        release_bytes,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("memory admission requires nonnegative integer bytes")
    return MemoryAdmission(
        declaration_admissible=demand_bytes + memory_floor_bytes <= pool_bytes,
        free_after_bytes=(
            pool_bytes
            - occupied_bytes
            - unmaterialized_bytes
            - unknown_residual_bytes
            - demand_bytes
            + release_bytes
        ),
        commitment_headroom_bytes=(
            pool_bytes
            - reserved_bytes
            - demand_bytes
            - memory_floor_bytes
            + release_bytes
        ),
        memory_floor_bytes=memory_floor_bytes,
    )
