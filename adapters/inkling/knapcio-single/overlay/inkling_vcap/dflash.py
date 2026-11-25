"""inkling_vcap on the DFLASH worker (SPEC=dflash2): hook points and the walked-row features.

Hook points (sglang.srt.speculative.dflash_worker_v2, the pinned image after the DFlash2 compatibility patch;
checked by signature and source text, never by line):

  DFlashWorkerV2.__init__              -> attach_dflash(worker) after the stock init. The scheduler captures the
                                          target verify graphs only after both workers exist
                                          (Scheduler.init_all_cuda_graphs), so the MoE gate hook is armed first.
  worker.selector.sample_path          -> instance-level wrapper (the class attribute is rebound and sha-pinned by
                                          inkling_rsfix): after the stock walk, the walked selector row of every
                                          slot, its effective temperature and max / top-2 gap probabilities are
                                          written to static buffers. It runs inside the draft CUDA graph
                                          (_SelectorDraftSampler.__call__ via the capture tail hook) and on the eager
                                          path (_propose_selector_block); fixed shapes, no host sync.
  DFlashWorkerV2._stage_draft_sampler  -> steplog: timing event before the draft forward.
  DFlashWorkerV2._plan_verify          -> right before the target verify. steplog: per-request metadata snapshot,
                                          route-journal cursor reset, timing event. (The verify cap, milestone 3,
                                          writes its live lengths / src_index here and returns them as the layout,
                                          which the stock code hands to _accept_block as cutoff_verify_lens.)
  DFlashWorkerV2._accept_block         -> steplog: the full-width accepted draft count, timing event, hand-off to
                                          the writer thread.

Features (walked row r_j over the K = 16 candidates of slot j; slot 0 reads scores[:, 0, 0], slot j >= 1 reads
scores[:, j, path[j-1]]): q_j = softmax(r_j / T_eff) with T_eff = 1 for greedy rows and the request temperature (the
sampler's clamped one) for sampled rows; m_j = max q_j, gap_j = q_j(top1) - q_j(top2). r_j is selected by d_{j-1},
never by d_j, so any width rule on (m, gap)_0..j is a stopping rule (decided by d_0..d_{j-1}).
"""

from __future__ import annotations

import functools
import importlib
import inspect

import torch

from inkling_vcap import logic
from inkling_vcap.state import PREFIX, STATE, log

MOD_DFLASH = "sglang.srt.speculative.dflash_worker_v2"
P_DF_INIT = "DFlashWorkerV2.__init__"
P_DF_STAGE = "DFlashWorkerV2._stage_draft_sampler"
P_DF_PLAN = "DFlashWorkerV2._plan_verify"
P_DF_ACCEPT = "DFlashWorkerV2._accept_block"
P_DF_FBG = "DFlashWorkerV2.forward_batch_generation"
P_DF_SAMPLED = "DFlashWorkerV2._selector_sampling_accept"
REQUIRED_DFLASH = ("Inkling2MoE.forward", "Inkling2Gate.forward", P_DF_INIT, P_DF_STAGE, P_DF_PLAN, P_DF_ACCEPT,
                   P_DF_FBG, P_DF_SAMPLED)
_MARK = "__inkling_vcap__"

# forward_batch_generation must call the hooked methods in this order, and hand the planned layout to the accept.
FBG_NEEDLES = [
    "self._stage_draft_sampler(bs=bs, sampling_info=batch.sampling_info)",
    "draft_out = self.draft_model_runner.forward(forward_batch)",
    "= self._plan_verify(",
    "target_out = self.target_worker.forward_batch_generation(",
    ") = self._accept_block(",
    "cutoff_verify_lens=None if layout is None else layout.verify_lens,",
]
SAMPLE_PATH_PARAMS = ["self", "candidate_ids", "scores", "uniforms", "temperatures", "greedy_mask"]


class DflashPatchError(RuntimeError):
    pass


def _marked(fn) -> bool:
    return bool(getattr(fn, _MARK, False))


def _mark(fn, original):
    functools.update_wrapper(fn, original)
    setattr(fn, _MARK, True)
    return fn


def _params(fn) -> list:
    return list(inspect.signature(fn).parameters)


def _require_params(fn, expected, where, keyword_only=()):
    got = _params(fn)
    if got != list(expected):
        raise DflashPatchError(f"{PREFIX} signature changed for {where}: expected {list(expected)}, got {got}")
    sig = inspect.signature(fn)
    for name in keyword_only:
        if sig.parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY:
            raise DflashPatchError(f"{PREFIX} {where}: {name} is no longer keyword-only")


def _require_source(fn, needles, where, ordered=True):
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise DflashPatchError(f"{PREFIX} cannot read the source of {where}") from exc
    last = -1
    for needle in needles:
        idx = src.find(needle)
        if idx < 0:
            raise DflashPatchError(f"{PREFIX} {where} no longer contains {needle!r}; re-check the patch")
        if ordered and idx < last:
            raise DflashPatchError(f"{PREFIX} {where}: {needle!r} is out of the expected order")
        last = idx


# ------------------------------------------------------------------ features (pure torch, graph-safe)
def walked_rows(candidate_ids: torch.Tensor, scores: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """[bs, G, K] selector rows the walk drew each slot from (as the stock _walked_confidence recovers them).
    candidate_ids [bs, G, K] are distinct vocabulary ids per slot (a top-k), so the token's lane is unique."""
    bs, gamma, k = candidate_ids.shape
    tok = tokens.reshape(bs, gamma)
    path = (candidate_ids == tok.unsqueeze(-1)).to(torch.int32).argmax(dim=-1)  # [bs, G]
    rows = scores[:, 1:].gather(2, path[:, :-1, None, None].expand(-1, -1, 1, k))[:, :, 0]
    return torch.cat((scores[:, :1, 0], rows), dim=1)


def effective_temperatures(temperatures: torch.Tensor, greedy_mask: torch.Tensor) -> torch.Tensor:
    t = temperatures.reshape(-1).to(torch.float32).clamp_min(1e-5)
    return torch.where(greedy_mask.reshape(-1).to(torch.bool), torch.ones_like(t), t)


def row_features(rows: torch.Tensor, t_eff: torch.Tensor):
    """(m, gap), each [bs, G]: max and top-1 minus top-2 of softmax(rows / T_eff) over the K candidates."""
    q = torch.softmax(rows.to(torch.float32) / t_eff[:, None, None], dim=-1)
    top = q.topk(2, dim=-1).values
    top = torch.nan_to_num(top, nan=0.0)
    return top[..., 0], top[..., 0] - top[..., 1]


class DflashFeatures:
    """Static device buffers of the walked-row features, [capacity, G(, K)]. ``seq`` counts the walks recorded (graph
    replays included): the capture checks that exactly one walk ran between two logged verify steps."""

    def __init__(self, *, capacity: int, gamma: int, top_k: int, device, keep_rows: bool) -> None:
        self.capacity, self.gamma, self.top_k = int(capacity), int(gamma), int(top_k)
        f32 = dict(dtype=torch.float32, device=device)
        self.m = torch.zeros((self.capacity, self.gamma), **f32)
        self.gap = torch.zeros((self.capacity, self.gamma), **f32)
        self.t_eff = torch.ones((self.capacity,), **f32)
        self.greedy = torch.ones((self.capacity,), dtype=torch.bool, device=device)
        self.rows = torch.zeros((self.capacity, self.gamma, self.top_k), **f32) if keep_rows else None
        self.seq = torch.zeros((1,), dtype=torch.int64, device=device)
        self.skipped = 0  # host count of walks above capacity (eager only: no graph is that large)
        self.failed = None
        # Capture-only lattice fields (keep_rows): candidate ids, full lattice scores (fp16), drafted tokens, and from
        # compute_candidates (flat [capacity * G] rows) the unary logits and the log-sum-exp of the unary transform
        # over the whole candidate plane (tail mass = 1 - sum_k exp(unary_k - lse)); NaN when not produced.
        self.lattice = bool(keep_rows)
        if self.lattice:
            self.cand = torch.zeros((self.capacity, self.gamma, self.top_k), dtype=torch.int64, device=device)
            self.scores = torch.zeros((self.capacity, self.gamma, self.top_k, self.top_k), dtype=torch.float16,
                                      device=device)
            self.drafted = torch.zeros((self.capacity, self.gamma), dtype=torch.int64, device=device)
            self.unary = torch.full((self.capacity * self.gamma, self.top_k), float("nan"), **f32)
            self.lse = torch.full((self.capacity * self.gamma,), float("nan"), **f32)
        self._logits = None  # projected candidate logits of the current compute_candidates call (stash)

    def record(self, *, candidate_ids, scores, tokens, temperatures, greedy_mask) -> None:
        bs = int(candidate_ids.shape[0])
        if bs > self.capacity or tuple(candidate_ids.shape[1:]) != (self.gamma, self.top_k):
            self.skipped += 1
            return
        rows = walked_rows(candidate_ids, scores, tokens)
        t_eff = effective_temperatures(temperatures[:bs], greedy_mask[:bs])
        m, gap = row_features(rows, t_eff)
        self.m[:bs].copy_(m)
        self.gap[:bs].copy_(gap)
        self.t_eff[:bs].copy_(t_eff)
        self.greedy[:bs].copy_(greedy_mask.reshape(-1)[:bs])
        if self.rows is not None:
            self.rows[:bs].copy_(rows)
        if self.lattice:
            self.cand[:bs].copy_(candidate_ids)
            self.scores[:bs].copy_(scores)
            self.drafted[:bs].copy_(tokens.reshape(bs, self.gamma))
        self.seq.add_(1)

    def record_candidates(self, *, n_rows, ids, unary, transform, chunk=16) -> None:
        """compute_candidates side write (rows = bs * G, flat): unary logits and the log-sum-exp of transform(logits)
        over the whole projected candidate plane, from the stashed logits (chunked FP32: no [N, V] FP32 temp)."""
        logits, self._logits = self._logits, None
        if not self.lattice or n_rows > self.unary.shape[0]:
            return
        self.unary[:n_rows].copy_(unary.reshape(n_rows, -1)[:, :self.top_k])
        if logits is None or logits.dim() != 2 or logits.shape[0] != n_rows:
            self.lse[:n_rows].fill_(float("nan"))
            return
        for start in range(0, n_rows, chunk):
            end = min(n_rows, start + chunk)
            x = logits[start:end]
            x = x.float() if x.dtype != torch.float32 else x.clone()  # transform works in place
            self.lse[start:end].copy_(torch.logsumexp(transform(x), dim=-1))


def make_sample_path(orig):
    """Wrap a bound CandidateSelector.sample_path: the walk is the stock one (whatever rsfix bound), its outputs are
    returned unchanged; the features are a side write."""

    def sample_path(*, candidate_ids, scores, uniforms, temperatures, greedy_mask):
        tokens, q_rows = orig(candidate_ids=candidate_ids, scores=scores, uniforms=uniforms,
                              temperatures=temperatures, greedy_mask=greedy_mask)
        feats = STATE.df
        if STATE.enabled and feats is not None and not feats.failed:
            try:
                feats.record(candidate_ids=candidate_ids, scores=scores, tokens=tokens,
                             temperatures=temperatures, greedy_mask=greedy_mask)
            except Exception as exc:  # noqa: BLE001 - a feature side write must never break the draft
                feats.failed = repr(exc)
                log(f"walked-row features DISABLED ({exc!r}); the draft is unaffected")
                if STATE.steplog is not None:
                    STATE.steplog.fail("features", exc)
        return tokens, q_rows

    sample_path.__wrapped__ = orig
    setattr(sample_path, _MARK, True)
    return sample_path


def make_project_logits(orig):
    """models.dflash._project_candidate_logits: stash the projected candidate plane for the lse side output (the
    returned tensor is the stock one, unchanged)."""

    def _project_candidate_logits(*args, **kwargs):
        out = orig(*args, **kwargs)
        feats = STATE.df
        if STATE.enabled and feats is not None and feats.lattice and not feats.failed:
            feats._logits = out
        return out

    _project_candidate_logits.__wrapped__ = orig
    setattr(_project_candidate_logits, _MARK, True)
    return _project_candidate_logits


def make_compute_candidates(orig, draft_model):
    def compute_candidates(hidden):
        ids, unary = orig(hidden)
        feats = STATE.df
        if STATE.enabled and feats is not None and feats.lattice and not feats.failed:
            try:
                feats.record_candidates(n_rows=int(ids.shape[0]), ids=ids, unary=unary,
                                        transform=draft_model._transform_unary_logits)
            except Exception as exc:  # noqa: BLE001 - a side write must never break the draft
                feats.failed = repr(exc)
                log(f"candidate-plane features DISABLED ({exc!r}); the draft is unaffected")
                if STATE.steplog is not None:
                    STATE.steplog.fail("candidates", exc)
        else:
            if feats is not None:
                feats._logits = None
        return ids, unary

    compute_candidates.__wrapped__ = orig
    setattr(compute_candidates, _MARK, True)
    return compute_candidates


# ------------------------------------------------------------------ worker patch
def patch_worker(mod) -> None:
    cls = getattr(mod, "DFlashWorkerV2", None)
    sampler_cls = getattr(mod, "_SelectorDraftSampler", None)
    if cls is None or sampler_cls is None:
        raise DflashPatchError(f"{PREFIX} {MOD_DFLASH} lacks DFlashWorkerV2 / _SelectorDraftSampler")
    for name in ("__init__", "_stage_draft_sampler", "_plan_verify", "_accept_block", "forward_batch_generation",
                 "_propose_selector_block"):
        if getattr(cls, name, None) is None:
            raise DflashPatchError(f"{PREFIX} patch target missing: DFlashWorkerV2.{name}")
    _require_source(cls.forward_batch_generation, FBG_NEEDLES, "DFlashWorkerV2.forward_batch_generation")
    _require_source(sampler_cls.__call__, ["self.selector.sample_path("], "_SelectorDraftSampler.__call__")
    _require_source(cls._propose_selector_block, ["self.selector.sample_path("],
                    "DFlashWorkerV2._propose_selector_block")
    if not _marked(cls._plan_verify):
        _require_params(cls._plan_verify, ["self", "batch", "draft_input", "prefix_lens"], P_DF_PLAN,
                        keyword_only=("batch", "draft_input", "prefix_lens"))
        _require_source(cls._plan_verify, ["if self._verify_planner is None:", "return None, None, None"], P_DF_PLAN)
    if not _marked(cls._accept_block):
        _require_params(cls._accept_block,
                        ["self", "candidates", "next_token_logits", "sampling_info", "draft_input", "prefix_lens",
                         "bs", "target_predict", "cutoff_verify_lens", "verify_shard"], P_DF_ACCEPT,
                        keyword_only=("candidates", "cutoff_verify_lens"))
        _require_source(cls._accept_block,
                        ["return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, cap_trim_lens"],
                        P_DF_ACCEPT)
    if not _marked(cls._stage_draft_sampler):
        _require_params(cls._stage_draft_sampler, ["self", "bs", "sampling_info"], P_DF_STAGE,
                        keyword_only=("bs", "sampling_info"))

    if not _marked(cls.__init__):
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if STATE.config is not None and getattr(self, "_hosts_draft", False):
                attach_dflash(self)

        cls.__init__ = _mark(__init__, orig_init)

    if not _marked(cls._stage_draft_sampler):
        orig_stage = cls._stage_draft_sampler

        def _stage_draft_sampler(self, *, bs, sampling_info):
            orig_stage(self, bs=bs, sampling_info=sampling_info)
            if STATE.enabled and STATE.worker is self and STATE.steplog is not None:
                STATE.steplog.mark_draft_start(bs)

        cls._stage_draft_sampler = _mark(_stage_draft_sampler, orig_stage)

    if not _marked(cls._plan_verify):
        orig_plan = cls._plan_verify

        def _plan_verify(self, *, batch, draft_input, prefix_lens):
            out = orig_plan(self, batch=batch, draft_input=draft_input, prefix_lens=prefix_lens)
            if STATE.enabled and STATE.worker is self:
                if out[2] is not None:
                    raise RuntimeError(f"{PREFIX} the stock verify planner produced a layout; vcap needs "
                                       "SGLANG_RAGGED_VERIFY_MODE=static")
                if STATE.steplog is not None:
                    STATE.steplog.start(batch=batch, prefix_lens=prefix_lens)
                elif STATE.capping:
                    cut = plan_cap(int(prefix_lens.numel()), batch, prefix_lens)
                    STATE.df_planned = cut
                    if cut is not None:
                        # The stock code hands layout.verify_lens to _accept_block as cutoff_verify_lens; with the
                        # planner absent (static mode) nothing else reads the layout except the SPS dumper
                        # (graph_num_tokens) and the result's cap_lens (scheduler stats).
                        return out[0], out[1], cut
            return out

        cls._plan_verify = _mark(_plan_verify, orig_plan)

    if not _marked(cls._accept_block):
        orig_accept = cls._accept_block

        def _accept_block(self, *args, **kwargs):
            if args:
                raise DflashPatchError(f"{PREFIX} _accept_block called positionally; re-check the patch")
            if STATE.enabled and STATE.worker is self and STATE.capping:
                planned, STATE.df_planned = STATE.df_planned, None
                if planned is not None and (kwargs.get("cutoff_verify_lens") is not planned.verify_lens
                                            or kwargs.get("verify_shard") is not None):
                    # Exactness: the remap of this step's verify is only valid with this step's capped accept.
                    raise RuntimeError(f"{PREFIX} a planned (capped) step reached _accept_block without its cutoff")
            out = orig_accept(self, **kwargs)
            if STATE.enabled and STATE.worker is self and STATE.steplog is not None:
                if kwargs.get("cutoff_verify_lens") is not None:
                    # Under a cutoff, AcceptGreedy counts matches past the live length (dead-row predictions) before
                    # clamping: such a count must never enter the capture (it is full-width only).
                    STATE.steplog.fail("finish", RuntimeError("capped accept in a full-width capture"))
                else:
                    STATE.steplog.finish(accept_len=out[0], commit_lens=out[1], out_tokens=out[3],
                                         logits=kwargs.get("next_token_logits"))
            return out

        cls._accept_block = _mark(_accept_block, orig_accept)

    if not _marked(cls.forward_batch_generation):
        orig_fbg = cls.forward_batch_generation

        def forward_batch_generation(self, *args, **kwargs):
            try:
                result = orig_fbg(self, *args, **kwargs)
            finally:
                if STATE.enabled and STATE.worker is self and STATE.steplog is not None:
                    STATE.steplog.end_step()
            if STATE.enabled and STATE.worker is self and STATE.df_planned is not None:
                STATE.df_planned = None
                # A capped step that left without _accept_block (a megakernel tail / accept path) would have committed
                # tokens verified under the remap: refuse (attach also refuses those paths while capping).
                raise RuntimeError(f"{PREFIX} a capped step bypassed _accept_block")
            return result

        cls.forward_batch_generation = _mark(forward_batch_generation, orig_fbg)

    sa = getattr(cls, "_selector_sampling_accept", None)
    if sa is None:
        raise DflashPatchError(f"{PREFIX} patch target missing: DFlashWorkerV2._selector_sampling_accept")
    if not _marked(sa):
        _require_params(sa, ["self", "candidates", "next_token_logits", "candidate_ids", "q_rows", "sampling_info",
                             "draft_input", "cutoff_verify_lens"], P_DF_SAMPLED,
                        keyword_only=("candidates", "q_rows", "cutoff_verify_lens"))
        _require_source(sa, ["draft_probs.scatter_(-1, candidate_ids, q_rows.float())",
                             "block_verification=self._block_verification",
                             "draft_probs.scatter_(-1, candidate_ids, 0.0)"], P_DF_SAMPLED)

        def _selector_sampling_accept(self, *args, **kwargs):
            if args:
                raise DflashPatchError(f"{PREFIX} _selector_sampling_accept called positionally")
            cut = kwargs.get("cutoff_verify_lens")
            if (STATE.enabled and STATE.worker is self and STATE.capping and cut is not None
                    and STATE.block_verify and STATE.config.bv_mask):
                # Block verification under the cap: the dead drafts become a stopped draft (logic.py); a masked
                # COPY of the sparse q (the draft graph owns q_out). The scatter / unscatter keep the dense
                # buffer's all-zero invariant: only candidate lanes are ever written.
                STATE.bv_masked += 1
                kwargs["q_rows"] = logic.mask_dead_q_rows_sparse(
                    kwargs["q_rows"], kwargs["candidate_ids"], kwargs["candidates"][:, 1:], cut)
            return sa(self, **kwargs)

        cls._selector_sampling_accept = _mark(_selector_sampling_accept, sa)

    STATE.applied.update((P_DF_INIT, P_DF_STAGE, P_DF_PLAN, P_DF_ACCEPT, P_DF_FBG, P_DF_SAMPLED))


def plan_cap(bs: int, batch=None, prefix_lens=None):
    """This step's live lengths / remap (dfcap.plan) under the configured DFLASH mode; None = uncapped step (full
    width, identity src_index)."""
    from inkling_vcap import dfcap

    cfg = STATE.config
    lo, hi = cfg.min_len, cfg.resolved_max(STATE.stride)
    uncapped = lambda: dfcap.plan(STATE, bs=0, mode="fixed", arg=STATE.stride)  # noqa: E731
    if cfg.bsmax and bs > cfg.bsmax:
        return uncapped()
    si = getattr(batch, "sampling_info", None)
    if si is not None and getattr(si, "has_custom_logit_processor", False):
        # A custom logit processor sees the whole verify block and may mix rows: dead rows must not reach live ones.
        return uncapped()
    if cfg.kind == "fixed":
        return dfcap.plan(STATE, bs=bs, mode="fixed", arg=max(lo, min(cfg.fixed_len, hi)), lo=lo, hi=hi)
    feats = STATE.df
    if feats is None or feats.failed:
        return uncapped()  # adaptive rules need this step's features: full width when they are unavailable
    if cfg.kind == "conf":
        return dfcap.plan(STATE, bs=bs, mode="conf", arg=cfg.threshold, lo=lo, hi=hi)
    return dfcap.plan(STATE, bs=bs, mode="policy", arg=STATE.df_policy, lo=lo, hi=hi, context=prefix_lens)


def _graph_capacity() -> int:
    """The largest captured decode batch (the draft sampler's and the verify graphs' bound); eager batches above it
    are neither featured nor captured. Falls back to the running-request limit, then 64."""
    from inkling_vcap.patches import _runtime_context

    rc = _runtime_context()
    if rc is not None:
        try:
            return int(max(rc.get_exec().graph.cuda_graph_config.decode.bs))
        except Exception:  # noqa: BLE001
            pass
        try:
            n = int(rc.get_schedule().max_running_requests or 0)
            if n > 0:
                return n
        except Exception:  # noqa: BLE001
            pass
    return 64


def attach_dflash(worker) -> None:
    """Validate the DFLASH setup, allocate the feature buffers (and the capture), arm the hooks."""
    cfg = STATE.config
    moe_mod = importlib.import_module("sglang.srt.models.inkling_common.inkling2_moe")
    missing = [p for p in REQUIRED_DFLASH if p not in STATE.applied]
    if missing:
        raise DflashPatchError(f"{PREFIX} patches not applied: {missing}")
    if STATE.enabled:
        raise RuntimeError(f"{PREFIX} a second speculative worker attached in one process")
    if not cfg.steplog and cfg.kind not in ("fixed", "conf", "dfpolicy"):
        raise RuntimeError(f"{PREFIX} INKLING_VERIFY_CAP={cfg.describe()}: the DFLASH worker takes fixed:K, conf:T or "
                           "dfpolicy:/path (the DSpark policy: / cost: formats score draft logits DFLASH has not)")
    stride = int(worker.block_size)
    if stride != 8:
        raise RuntimeError(f"{PREFIX} DFLASH steplog expects verify width 8, got {stride}")
    selector = getattr(worker, "selector", None)
    if selector is None or getattr(worker, "_is_domino", False) or getattr(worker, "lilicorr", None) is not None:
        raise RuntimeError(f"{PREFIX} the DFLASH port needs a selector draft (DFlash2)")
    if getattr(worker, "_verify_planner", None) is not None:
        raise RuntimeError(f"{PREFIX} needs SGLANG_RAGGED_VERIFY_MODE=static on DFLASH")
    wmod = importlib.import_module(MOD_DFLASH)
    if float(getattr(wmod, "SIMULATE_ACC_LEN", 0) or 0) > 0:
        raise RuntimeError(f"{PREFIX} incompatible with SGLANG_SIMULATE_ACC_LEN")
    from inkling_vcap.patches import _runtime_context

    rc = _runtime_context()
    if rc is not None:
        par = rc.get_parallel()
        for name in ("tp_size", "pp_size"):
            try:
                if int(getattr(par, name, 1) or 1) > 1:
                    raise RuntimeError(f"{PREFIX} the DFLASH port is validated for {name}=1 only")
            except AttributeError:
                pass
    if getattr(worker, "_megakernel_accept", None) is not None:
        log("WARNING: the DFLASH megakernel greedy accept / tail is present; its steps bypass _accept_block and are "
            "dropped from the capture (counted as no_accept)")

    target = getattr(worker, "_target_worker", None) or worker.target_worker
    n_moe = 0
    for name, module in target.model_runner.model.named_modules():
        if isinstance(module, moe_mod.Inkling2MoE):
            short = name.replace("model.", "").replace("layers.", "L").replace(".mlp", "")
            module._vcap_name = short
            module.gate._vcap_name = short
            n_moe += 1
    if n_moe == 0:
        raise RuntimeError(f"{PREFIX} the target model has no Inkling2MoE layers")

    capacity = _graph_capacity()
    STATE.allocate(stride=stride, capacity=capacity, device=worker.device)
    STATE.capping = not cfg.never_caps(stride)
    STATE.df_planned = None
    STATE.src_identity = True
    if cfg.steplog and STATE.capping:
        raise RuntimeError(f"{PREFIX} internal: steplog must run full width")
    fused_note = "off"
    if STATE.capping:
        cfg.validate_for_stride(stride)
        if getattr(worker, "_megakernel_accept", None) is not None or getattr(worker, "_megakernel_sampled", None):
            raise RuntimeError(f"{PREFIX} the DFLASH megakernel accept paths ignore the cutoff; refused while capping")
        tlp = getattr(worker, "_target_logits_processor", None)
        if tlp is not None and (getattr(tlp, "do_tensor_parallel_all_gather", False)
                                or getattr(tlp, "do_tensor_parallel_all_gather_dp_attn", False)):
            # The processor exists on every Inkling target; only an actual vocab-sharded (TP > 1) gather defers.
            raise RuntimeError(f"{PREFIX} the deferred vocab-sharded verify gather is not supported while capping")
        if cfg.kind == "dfpolicy":
            from inkling_vcap import dfcap

            pol = dfcap.load_df_policy(cfg.dfpolicy_path)
            if pol.gamma != stride - 1 or any(len(v) != stride - 1 for v in pol.rowcost.values()):
                raise RuntimeError(f"{PREFIX} {cfg.dfpolicy_path}: policy width {pol.gamma + 1} != verify width {stride}")
            STATE.df_policy = dfcap.DfPolicyDevice(pol, worker.device)
        STATE.block_verify = bool(getattr(worker, "_block_verification", False))
        if cfg.remap and cfg.fused and torch.device(worker.device).type == "cuda":
            from inkling_vcap import fused

            err = fused.selfcheck(worker.device)  # inkling_noexp hooks its phantom-mask self-check here
            STATE.use_fused = err is None
            fused_note = "on (self-check passed)" if err is None else f"off ({err})"

    sp = getattr(selector, "sample_path", None)
    if sp is None:
        raise DflashPatchError(f"{PREFIX} the selector has no sample_path")
    if not _marked(sp):
        _require_params(type(selector).sample_path, SAMPLE_PATH_PARAMS, "CandidateSelector.sample_path",
                        keyword_only=SAMPLE_PATH_PARAMS[1:])
        selector.sample_path = make_sample_path(sp)  # instance attribute: the class one stays rsfix's
    top_k = int(selector.top_k)
    STATE.df = DflashFeatures(capacity=capacity, gamma=stride - 1, top_k=top_k, device=worker.device,
                              keep_rows=bool(cfg.steplog))
    if cfg.steplog:
        # Lattice side outputs (capture only): the projected candidate plane (module function, looked up by
        # candidate_topk at call time) and compute_candidates on the draft-model instance.
        mdf = importlib.import_module("sglang.srt.models.dflash")
        pcl = getattr(mdf, "_project_candidate_logits", None)
        draft_model = getattr(worker, "draft_model", None)
        cc = getattr(draft_model, "compute_candidates", None)
        tr = getattr(draft_model, "_transform_unary_logits", None)
        if pcl is None or cc is None or tr is None:
            raise DflashPatchError(f"{PREFIX} the DFlash2 draft model lacks compute_candidates / "
                                   "_transform_unary_logits / _project_candidate_logits")
        _require_source(getattr(mdf, "candidate_topk"), ["logits = _project_candidate_logits("],
                        "models.dflash.candidate_topk")
        _require_source(type(draft_model).compute_candidates, ["candidate_topk(", "_transform_unary_logits(vals)"],
                        "DFlash2DraftModel.compute_candidates")
        if not _marked(pcl):
            mdf._project_candidate_logits = make_project_logits(pcl)
        if not _marked(cc):
            draft_model.compute_candidates = make_compute_candidates(cc, draft_model)
    if cfg.steplog:
        from inkling_vcap.dfsteplog import DfStepLog

        STATE.steplog = DfStepLog(cfg.steplog, stride=stride, top_k=top_k, capacity=capacity, device=worker.device,
                                  features=STATE.df)
    STATE.worker = worker
    STATE.algorithm = "dflash"
    STATE.enabled = True
    run = STATE.steplog.run if STATE.steplog is not None else "-"
    log(f"DFLASH attached: mode={cfg.describe()} capping={'on' if STATE.capping else 'off'} "
        f"block_verification={'on' if STATE.block_verify else 'off'} bv_mask={'on' if cfg.bv_mask else 'off'} "
        f"fused_remap={fused_note} stride={stride} capacity={capacity} top_k={top_k} moe_layers={n_moe} "
        f"steplog_run={run}; patched: " + ", ".join(REQUIRED_DFLASH) + ", selector.sample_path (instance)")
