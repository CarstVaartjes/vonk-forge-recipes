"""Typed local sweep reconciliation records; no Controller success is inferred."""

from enum import IntEnum, StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue


class WorkloadExitCause(StrEnum):
    """Controller exit facts from the helper runtime_logs::ExitCause vocabulary."""

    OOM_KILLED = "oom_killed"
    HOST_MEMORY_EXHAUSTED = "host_memory_exhausted"
    BAD_ARGUMENTS = "bad_arguments"
    MISSING_FILE = "missing_file"
    UNSUPPORTED_GPU_ARCH = "unsupported_gpu_arch"
    IMAGE_ARCH_MISMATCH = "image_arch_mismatch"
    START_REFUSED = "start_refused"
    CRASHED_BY_SIGNAL = "crashed_by_signal"
    PYTHON_EXCEPTION = "python_exception"
    UNCLASSIFIED = "unclassified"


class WorkloadMemoryTrigger(StrEnum):
    NVIDIA_ALLOCATION_FAILURE = "nvidia_allocation_failure"
    SUSTAINED_FULL_PSI = "sustained_full_psi"
    SUSTAINED_MEMAVAILABLE_EXHAUSTION = "sustained_memavailable_exhaustion"


class WorkloadExitFacts(BaseModel):
    """Only fixed summary facts, never the exception/log text that follows them."""

    exit_cause: WorkloadExitCause
    trigger: WorkloadMemoryTrigger | None = None


class WorkloadFailureCode(StrEnum):
    APPLICATION_FAILED = "application.failed"
    RECIPE_START_FAILED = "recipe_start_failed"
    RUNTIME_PROCESS_EXITED = "runtime_process_exited"
    HOST_MEMORY_EXHAUSTED = "workload.host_memory_exhausted"


class SweepFailureClass(StrEnum):
    PLATFORM_EXIT = "platform-exit"


class CliObservationStatus(StrEnum):
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class CliObservation(BaseModel):
    """Projection of vonkctl's local observation, not a Controller receipt."""

    status: CliObservationStatus
    error: str | None = None


class CliObservationReply(BaseModel):
    observation: CliObservation
    result: dict[str, JsonValue]


class CleanupEndReason(StrEnum):
    TARGET_ABSENT = "sweep.cleanup_target_absent"


class CleanupEnd(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: CleanupEndReason
    observed_at: float


class CleanupObservationBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 0
    next_check: float = 0.0


class SweepSlotPhase(StrEnum):
    LOADING = "loading"
    SMOKING = "smoking"
    FINISHED = "finished"


class LoadedRunObservation(BaseModel):
    run_id: str


class FleetNodeObservation(BaseModel):
    id: str
    connection: "FleetConnectionObservation"
    loaded: list[LoadedRunObservation]


class FleetOnlineState(StrEnum):
    ONLINE = "online"


class FleetConnectionObservation(BaseModel):
    # External wire vocabulary is validated by the installed CLI. Only ONLINE
    # certifies absence; other values remain observed, without authorizing it.
    online_state: str


class CleanupFleetObservation(BaseModel):
    nodes: list[FleetNodeObservation]


class ReviewedCleanupStop(BaseModel):
    run_id: str
    node_ids: list[str]


class ReviewInfrastructureSource(StrEnum):
    TRANSPORT = "review-transport"


class InfrastructureIncident(BaseModel):
    """Durable local retry history; escalation never invents a Controller ending."""

    model_config = ConfigDict(extra="forbid")

    message: str = ""
    since: float
    count: int = 0
    until: float = 0.0
    dependency: str | None = None
    escalate_at: float | None = None


class SweepStorageCode(StrEnum):
    """Controller run-switch storage decisions consumed by the sweep."""

    INSUFFICIENT_DISK = "run-switch.insufficient-disk"
    DISK_EVICTION_PLANNED = "run-switch.disk-eviction-planned"


class SweepStorageReason(BaseModel):
    code: str
    detail: str
    node_ids: list[str] = Field(default_factory=list)


class SweepStorageAssessment(BaseModel):
    blockers: list[SweepStorageReason] = Field(default_factory=list)


class SweepAssignmentStorageAssessment(BaseModel):
    assessment: SweepStorageAssessment


class SweepStorageReview(BaseModel):
    reasons: list[SweepStorageReason] = Field(default_factory=list)
    assessments: list[SweepAssignmentStorageAssessment] = Field(default_factory=list)

    @property
    def insufficient(self) -> list[SweepStorageReason]:
        return [
            reason
            for reasons in [
                self.reasons,
                *(item.assessment.blockers for item in self.assessments),
            ]
            for reason in reasons
            if reason.code == SweepStorageCode.INSUFFICIENT_DISK
        ]


class SweepStorageWait(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasons: list[SweepStorageReason]
    content_sha256: str
    next_check: float

    @property
    def node_ids(self) -> set[str]:
        return {node for reason in self.reasons for node in reason.node_ids}


class SweepWaitReason(StrEnum):
    APPLICATION_QUEUED = "application queued"
    APPLICATION_BACKOFF = "application backoff"
    OPERATOR = "operator"
    ADMISSION = "admission"
    BLOCKER = "controller blocker"
    CONTROLLER_RETRY = "controller retry"
    CONTROLLER_WAIT = "controller wait"
    OPERATION_QUEUED = "operation queued"
    OPERATION_BACKOFF = "operation backoff"
    NAS_SPACE = "nas space"
    PIN_CAPACITY = "pin capacity"
    NO_ONLINE_LANES = "no online lanes"
    OBSERVATION = "observation unavailable"
    ROUTE_PUBLICATION = "route publication"
    CONTROLLER_STALLED = "controller stalled"
    ENDPOINT_PUBLICATION = "endpoint publication"
    CLIENT_KEY = "client key renewal"
    SMOKE_BUSY = "smoke busy"
    SMOKE_RESPONSE = "smoke response"


class SweepWaitCode(StrEnum):
    BUDGET_EXHAUSTED = "sweep.wait_budget_exhausted"


class ControllerWaitOperation(BaseModel):
    """The Controller operation a wait is behind (its ``BlockingOperation``)."""

    kind: str
    #: The Controller always sends it; state written before it was read has none.
    id: str | None = None


class ControllerWaitBlocker(BaseModel):
    code: str
    detail: str = ""
    operation: ControllerWaitOperation | None = None


class ControllerWait(BaseModel):
    """Projection of the Controller's declared dependency, irrespective of severity.

    Reason and operation kind are external wire vocabulary, not sweep states.
    Only the same reason behind the same Controller operations (kind and id)
    correlates waits across recipes; the same reason text alone never does.
    """

    reason: str
    detail: str = ""
    blockers: list[ControllerWaitBlocker] = Field(default_factory=list)
    dependency_deadline: AwareDatetime | None = None

    @property
    def identity(self) -> tuple[str, frozenset[tuple[str, str]]] | None:
        operations = frozenset(
            (b.operation.kind, b.operation.id)
            for b in self.blockers
            if b.operation is not None and b.operation.id is not None
        )
        return (self.reason, operations) if operations else None

    @property
    def description(self) -> str:
        details = self.detail or next(
            (b.detail for b in self.blockers if b.code == self.reason), ""
        )
        return (
            f"{self.reason}: {details}"
            if details and details != self.reason
            else self.reason
        )


class SweepWait(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: SweepWaitReason
    detail: str = ""
    controller: ControllerWait | None = None

    @property
    def description(self) -> str:
        if self.controller is not None:
            return self.controller.description
        return f"{self.reason}: {self.detail}" if self.detail else self.reason.value


class SweepWaitExpiry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: SweepWaitCode = SweepWaitCode.BUDGET_EXHAUSTED
    wait: SweepWait
    started_at: float
    deadline: float
    observed_at: float


class SweepProgress(StrEnum):
    PROGRESSING = "progressing"
    WAITING = "waiting"
    STALLED = "stalled"
    WAIT_EXPIRED = "wait-expired"


class SweepEvidenceExit(IntEnum):
    SYNCED = 0
    WAITING = 10


class SweepEvidenceMergeState(StrEnum):
    BEHIND = "BEHIND"
    BLOCKED = "BLOCKED"
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    DRAFT = "DRAFT"
    HAS_HOOKS = "HAS_HOOKS"
    UNKNOWN = "UNKNOWN"
    UNSTABLE = "UNSTABLE"


class SweepAdoptionReason(StrEnum):
    LOAD_RULES = "load-timeout rules changed: copying no longer counts"
    SMOKE_RULES = "smoke rules changed: answer quality no longer fails a recipe"
    SUPERSESSION = "load-superseded: not a recipe failure"
    CAPACITY = "review capacity refusal: the fleet was not cleared first"


class SweepAdoptionResult(BaseModel):
    """One normalized saved verdict, awaiting the run's locked result writer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recipe: str
    reason: SweepAdoptionReason
    previous: dict[str, JsonValue]
