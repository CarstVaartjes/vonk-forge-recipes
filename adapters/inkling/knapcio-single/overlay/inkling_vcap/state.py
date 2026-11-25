"""Process-wide runtime state of the verify cap (one DSpark worker per process).

Invariant that keeps output exact: a row is routed as "dead" by the MoE remap only
when the accept of the same step is capped at that row's request live length. Both
read the same device buffers (``live_lens`` and ``src_index``, rewritten before each
target verify), and both skip together when a step's shape exceeds ``capacity``.
Accept cap without remap is always exact (it only gives up acceptance).
"""

from __future__ import annotations

import sys
from typing import Optional

import torch

from inkling_vcap import fastprobs, fused, logic
from inkling_vcap.config import VcapConfig

PREFIX = "[inkling-vcap]"


def log(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def _capturing(t: torch.Tensor) -> bool:
    if t.device.type != "cuda":
        return False
    return bool(torch.cuda.is_current_stream_capturing())


class _Cutoff:
    """Duck-typed stand-in for RaggedVerifyLayout: accept_draft_tokens only reads
    ``cutoff_layout.verify_lens``."""

    __slots__ = ("verify_lens",)

    def __init__(self, verify_lens: torch.Tensor) -> None:
        self.verify_lens = verify_lens


class VcapState:
    def __init__(self) -> None:
        self.config: Optional[VcapConfig] = None
        self.applied: set[str] = set()
        # Set by attach(): the hooks act only from then on.
        self.enabled = False
        self.worker = None
        self.stride = 0
        self.capacity = 0
        self.live_lens: Optional[torch.Tensor] = None
        self.src_index: Optional[torch.Tensor] = None
        self.layout: Optional[logic.RowLayout] = None
        self.live_lens_source = ""
        # Per-step bookkeeping (host side).
        self.in_verify_moe = False
        self.last_proposal = None
        self.step_planned = False
        self.planned_bs = 0
        self._src_key = None
        self._epi_state = None
        self.epilogue = None
        self.use_fused = False
        # False when the config never cuts (fixed:8, cost with every T = 0): then the
        # verify is left exactly as the engine runs it (no remap, cutoff, accept swap).
        self.capping = False
        self._validated_first_step = False
        # Diagnostics (Python-side counts: eager calls + capture-time recordings).
        self.remap_recorded = 0
        self.remap_skipped = 0
        self.cutoff_injected = 0
        self.static_accept_recorded = 0
        # Stats (device accumulators, read every log_every steps).
        self._acc: Optional[torch.Tensor] = None
        self._steps = 0
        self._req_steps = 0
        self._host_live = 0
        self._union_slots: dict[int, int] = {}
        self._union_acc: Optional[torch.Tensor] = None
        self._union_names: list[str] = []
        # policy: mode device tensors (table [2, gamma, nb], edges, row_cost, u1)
        self._pol = None
        # Block verification (--speculative-use-block-verification): the accept wrapper
        # sets bv_lens for the duration of one accept call; _draft_probs masks the dead
        # drafts with it (logic.mask_dead_draft_rows).
        self.block_verify = False
        self.bv_lens: Optional[torch.Tensor] = None
        self.bv_masked = 0
        self.bv_skipped = 0
        # INKLING_VERIFY_CAP_FASTPROBS: set by attach() after the device self-check.
        self.use_fastprobs = False
        self.fastprobs_calls = 0
        # INKLING_VERIFY_CAP_FUSED_ACCEPT: primitives object after the self-check, or None.
        self.fused_accept = None
        self.fused_accept_calls = 0
        # DFLASH (dflash.py): "dspark" or "dflash" once a worker attached; the walked
        # selector-row features written by the sample_path wrapper (device buffers).
        self.algorithm = ""
        self.df = None
        self.df_planned = None     # dfcap.Cutoff of the current DFLASH step (exactness guard in _accept_block)
        self.df_min_live = None    # [bs] int64 live-row minimum for this step (inkling_pcopy: forced copy drafts + 1)
        self.df_policy = None      # dfcap.DfPolicyDevice (dfpolicy: mode)
        self.src_identity = True   # src_index currently the identity (dfcap.plan)
        # INKLING_VERIFY_CAP_STEPLOG: the full-width capture (dfsteplog.DfStepLog).
        self.steplog = None

    # ------------------------------------------------------------------ setup
    def allocate(self, *, stride: int, capacity: int, device, live_lens=None) -> None:
        self.stride = int(stride)
        self.capacity = int(capacity)
        if live_lens is not None:
            if live_lens.numel() < self.capacity or live_lens.dtype != torch.int64:
                raise RuntimeError(
                    f"{PREFIX} live-length buffer must be int64 with >= {self.capacity} "
                    f"rows, got {live_lens.dtype} x {live_lens.numel()}"
                )
            self.live_lens = live_lens
        else:
            self.live_lens = torch.zeros(self.capacity, dtype=torch.int64, device=device)
        self.layout = logic.RowLayout(self.capacity, self.stride, device)
        # Identity until the first planned step (graph capture records against it).
        self.src_index = self.layout.rows.clone()
        self._acc = torch.zeros(3, dtype=torch.int64, device=device)
        pf = None if self.config is None else self.config.policy_file
        if pf is not None:
            self._pol = {
                "table": torch.tensor(pf.calib, dtype=torch.float32, device=device),
                "edges": torch.tensor(pf.edges, dtype=torch.float32, device=device),
                "row_cost": torch.tensor(pf.row_cost, dtype=torch.float32, device=device),
                "u1": torch.tensor(pf.u1, dtype=torch.float32, device=device),
            }
        if self.config is not None and self.config.union:
            # [slot, (sum distinct, visits)]; slots = MoE modules, 64 is plenty.
            self._union_acc = torch.zeros((64, 2), dtype=torch.float64, device=device)

    # ------------------------------------------------------------- per step
    def plan_step(self, *, bs: int, proposal) -> None:
        """Write this step's live lengths and remap index; called right before the
        target verify (same stream). No host sync. A constant-length step whose
        buffers already hold the right values launches nothing."""
        cfg = self.config
        self.step_planned = False
        if bs > self.capacity or bs <= 0:
            return
        stride = self.stride
        pol = cfg.policy(bs, stride)
        # begin_static_step refills verify_lens_buf (= live_lens) when its (bs, armed)
        # state changes; watching that state tells whether our values survived.
        epi_state = getattr(self.epilogue, "_static_step_state", None)
        prev_bs = None if self._src_key is None else self._src_key[2]
        if pol.const_len:
            key = ("const", pol.const_len, bs)
            if key != self._src_key or epi_state != self._epi_state:
                self.live_lens[:bs].fill_(pol.const_len)
                if key != self._src_key:
                    if bs != prev_bs:
                        # Padding requests: L = 0 (dead, remapped to row 0, capped to 0).
                        self.live_lens[bs:].zero_()
                    logic.build_src_index(self.live_lens, self.layout, out=self.src_index)
        else:
            key = ("dyn", 0, bs)
            block = getattr(proposal, "draft_block", None)
            logits = getattr(block, "corrected_logits", None)
            if logits is None:
                raise RuntimeError(
                    f"{PREFIX} {cfg.kind} mode needs the draft's kept step logits, but "
                    "this step's proposal has none (SGLANG_DSPARK_USE_GREEDY_DRAFT=1, "
                    "or a draft path that drops them). Use fixed:K or unset the flag."
                )
            if self.use_fastprobs and logits.is_cuda and logits.stride(-1) == 1:
                # Same max probabilities from one read of the logits (fastprobs.py).
                probs = fastprobs.max_prob(
                    logits[:bs],
                    temperatures=getattr(block, "temperatures", None),
                    greedy_mask=getattr(block, "greedy_mask", None),
                )
            else:
                probs = logic.draft_token_probs(
                    logits[:bs],
                    block.draft_tokens[:bs],
                    temperatures=getattr(block, "temperatures", None),
                    greedy_mask=getattr(block, "greedy_mask", None),
                    # Max probability: a function of the draft distribution at position
                    # j (context + earlier drafts), never of the token d_j it verifies.
                    use_max=True,
                )
            if cfg.kind == "policy":
                lens = logic.calibrated_live_lens(
                    probs, self._sampled_rows(block, bs, probs.device),
                    table=self._pol["table"], edges=self._pol["edges"],
                    row_cost=self._pol["row_cost"], theta=pol.threshold,
                    min_len=cfg.min_len, max_len=cfg.resolved_max(stride),
                    u1=self._pol["u1"], pool=pol.pool,
                    rounds=cfg.policy_file.rounds if pol.pool > 0 else 0,
                )
            else:
                lens = logic.live_lens_from_probs(
                    probs, threshold=pol.threshold, min_len=cfg.min_len,
                    max_len=cfg.resolved_max(stride),
                )
            self.live_lens[:bs].copy_(lens)
            if bs != prev_bs:
                self.live_lens[bs:].zero_()
                logic.build_src_index(self.live_lens, self.layout, out=self.src_index)
            else:
                logic.build_src_index(
                    self.live_lens, self.layout, num_reqs=bs, out=self.src_index
                )
        self._src_key = key
        self._epi_state = epi_state
        self.step_planned = True
        self.planned_bs = bs

    @staticmethod
    def _sampled_rows(block, bs: int, device) -> torch.Tensor:
        """[bs] bool: the request samples (its draft max-probabilities are at its
        temperature, so it uses the sampled calibration table). Same rule as
        draft_token_probs: greedy_mask wins; without one, every row is at its
        temperature when temperatures exist (sampled), else at temperature 1."""
        mask = getattr(block, "greedy_mask", None)
        if mask is None:
            temps = getattr(block, "temperatures", None)
            return torch.full((bs,), temps is not None, dtype=torch.bool, device=device)
        return ~mask.reshape(-1)[:bs].to(device=device, dtype=torch.bool)

    def cutoff_for(self, n: int) -> Optional[_Cutoff]:
        if not (self.enabled and self.capping) or n > self.capacity:
            return None
        self.cutoff_injected += 1
        return _Cutoff(self.live_lens[:n])

    def block_verify_lens(self, verify_lens: torch.Tensor, n: int) -> Optional[torch.Tensor]:
        """The live lengths to mask the draft with before a block-verification accept,
        or None when every row of this step is live (the stock accept is then exact as
        it is). The skip is a host decision, so it is taken only on an eager step whose
        lengths are our own constant full-width buffer; a graph capture always records
        the mask (it reads the buffer at replay)."""
        key = self._src_key
        if (
            key is not None and key[0] == "const" and key[1] >= self.stride and key[2] == n
            and self.live_lens is not None
            and verify_lens.data_ptr() == self.live_lens.data_ptr()
            and not _capturing(verify_lens)
        ):
            self.bv_skipped += 1
            return None
        return verify_lens[:n]

    def remap(self, weights: torch.Tensor, ids: torch.Tensor):
        if ids.dim() != 2 or weights.shape[0] != ids.shape[0]:
            raise RuntimeError(
                f"{PREFIX} unexpected Inkling2Gate output: weights "
                f"{tuple(weights.shape)}, ids {tuple(ids.shape)}"
            )
        m = ids.shape[0]
        if m % self.stride or m > self.src_index.numel():
            self.remap_skipped += 1
            return weights, ids
        self.remap_recorded += 1
        if self.use_fused and fused.supports(weights, ids):
            return fused.remap_rows_fused(weights, ids, self.src_index)
        return logic.remap_rows(weights, ids, self.src_index)

    def count_union(self, module, ids: torch.Tensor, num_experts: int) -> None:
        if self._union_acc is None:
            return
        key = id(module)
        slot = self._union_slots.get(key)
        if slot is None:
            slot = len(self._union_slots)
            if slot >= self._union_acc.shape[0]:
                return
            self._union_slots[key] = slot
            self._union_names.append(getattr(module, "_vcap_name", f"moe{slot}"))
        m = ids.shape[0]
        if m % self.stride == 0 and m <= self.src_index.numel():
            # Count real requests only: graph-padding rows (L = 0) read as row 0.
            # The live-row remap is already applied when capping; this gather only
            # feeds the counter.
            ids = ids.index_select(0, self._pad_index(m))
        add = torch.stack(
            [logic.distinct_experts(ids, num_experts).to(torch.float64),
             torch.ones((), dtype=torch.float64, device=ids.device)]
        )
        self._union_acc[slot].add_(add)

    def _pad_index(self, m: int) -> torch.Tensor:
        # Rows of real requests map to themselves (or their remap source); padding
        # requests' rows to row 0. src_index already encodes exactly that.
        return self.src_index[:m]

    # ---------------------------------------------------------------- stats
    def observe(self, result) -> None:
        if not self.step_planned:
            return
        self.step_planned = False
        cfg = self.config
        if cfg.log_every <= 0:
            return
        bs = self.planned_bs
        commit = getattr(result, "accept_lens", None)
        block = getattr(result, "block_accept_lens", None)
        if commit is None:
            return
        commit = commit[:bs]
        block = commit if block is None else block[:bs]
        if self._src_key is not None and self._src_key[0] == "const":
            # Constant L: counted on the host; two sums on the device (3 launches).
            self._host_live += self._src_key[1] * bs
            self._acc[1:].add_(torch.stack([commit, block.to(commit.dtype)]).sum(dim=1))
        else:
            rows = torch.stack([self.live_lens[:bs], commit.to(torch.int64),
                                block.to(torch.int64)])
            self._acc.add_(rows.sum(dim=1))
        self._steps += 1
        self._req_steps += bs
        if self._steps % cfg.log_every == 0:
            self.flush_stats()

    def flush_stats(self) -> None:
        if self._req_steps == 0:
            return
        sum_l, sum_commit, sum_block = self._acc.tolist()
        sum_l += self._host_live
        n = float(self._req_steps)
        mean_l = sum_l / n
        msg = (
            f"steps={self._steps} req_steps={self._req_steps} mode={self.config.kind} "
            f"mean_bs={self._req_steps / max(1, self._steps):.2f} "
            f"mean_live={mean_l:.3f} mean_dead={self.stride - mean_l:.3f} "
            f"mean_commit={sum_commit / n:.3f} mean_uncapped={sum_block / n:.3f} "
            f"(uncapped = commit + cap-trimmed drafts)"
        )
        if self.fused_accept is not None:
            msg += f" fused_accept_calls={self.fused_accept_calls}"
        if self.block_verify:
            msg += f" block_verify masked_calls={self.bv_masked} skipped_full={self.bv_skipped}"
        if self._union_acc is not None and self._union_slots:
            u = self._union_acc[: len(self._union_slots)].tolist()
            visits = sum(row[1] for row in u)
            total = sum(row[0] for row in u)
            per = " ".join(
                f"{name}={row[0] / row[1]:.1f}" if row[1] else f"{name}=-"
                for name, row in zip(self._union_names, u)
            )
            msg += (
                f" union_mean={total / visits:.2f} experts/visit over {int(visits)} visits;"
                f" per-layer {per}" if visits else ""
            )
            self._union_acc.zero_()
        log(msg)
        self._acc.zero_()
        self._host_live = 0
        self._steps = 0
        self._req_steps = 0


STATE = VcapState()
