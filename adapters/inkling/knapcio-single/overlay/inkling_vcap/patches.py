"""The import-time patches. Each hook validates its targets (presence, signature,
and a few source-level invariants) and raises on any mismatch, so a changed engine
fails at startup instead of silently running without the cap."""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Callable, Iterable

import torch

from inkling_vcap import fastprobs, logic
from inkling_vcap.state import PREFIX, STATE, log

MOD_MOE = "sglang.srt.models.inkling_common.inkling2_moe"
MOD_VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
MOD_SAG = "sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph"
MOD_DRAFT = "sglang.srt.speculative.dspark_components.dspark_draft"
MOD_WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"

P_MOE = "Inkling2MoE.forward"
P_GATE = "Inkling2Gate.forward"
P_STATIC_ACCEPT = "DsparkVerifyEpilogue._accept_static_greedy"
P_RUN_NON_COMPACT = "TargetVerifyExecutor.run_non_compact"
P_ACCEPT_VERIFY = "dspark_verify.accept_draft_tokens"
P_ACCEPT_SAG = "dspark_sampling_accept_graph.accept_draft_tokens"
P_DRAFT_PROBS = "dspark_verify._draft_probs"  # required only with block verification
P_PROPOSE = "DraftBlockProposer.propose"
P_WORKER_INIT = "DSparkWorkerV2.__init__"
P_FORWARD_DECODE = "DSparkWorkerV2._forward_decode"

REQUIRED = (
    P_MOE, P_GATE, P_STATIC_ACCEPT, P_RUN_NON_COMPACT, P_ACCEPT_VERIFY,
    P_ACCEPT_SAG, P_PROPOSE, P_WORKER_INIT, P_FORWARD_DECODE,
)
_MARK = "__inkling_vcap__"


class PatchTargetError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def _attr(owner, name: str, where: str):
    value = getattr(owner, name, None)
    if value is None:
        raise PatchTargetError(f"{PREFIX} patch target missing: {where}.{name}")
    return value


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def _require_params(fn, expected: list[str], where: str) -> None:
    got = _params(fn)
    if got != expected:
        raise PatchTargetError(
            f"{PREFIX} signature changed for {where}: expected {expected}, got {got}"
        )


def _require_subset(fn, names: Iterable[str], where: str) -> None:
    got = set(_params(fn))
    missing = [n for n in names if n not in got]
    if missing:
        raise PatchTargetError(
            f"{PREFIX} signature changed for {where}: missing parameters {missing}"
        )


def _require_source(fn, needles: list[str], where: str, ordered: bool = False) -> None:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise PatchTargetError(f"{PREFIX} cannot read the source of {where}") from exc
    last = -1
    for needle in needles:
        idx = src.find(needle)
        if idx < 0:
            raise PatchTargetError(
                f"{PREFIX} {where} no longer contains {needle!r}; re-check the patch"
            )
        if ordered and idx < last:
            raise PatchTargetError(f"{PREFIX} {where}: unexpected order of {needles}")
        last = idx


def _mark(fn: Callable, original: Callable) -> Callable:
    functools.update_wrapper(fn, original)
    setattr(fn, _MARK, True)
    return fn


def _is_marked(fn) -> bool:
    return bool(getattr(fn, _MARK, False))


# ------------------------------------------------------------------ MoE
def patch_moe(mod) -> None:
    moe_cls = _attr(mod, "Inkling2MoE", MOD_MOE)
    gate_cls = _attr(mod, "Inkling2Gate", MOD_MOE)
    if _is_marked(moe_cls.forward) and _is_marked(gate_cls.forward):
        STATE.applied.update((P_MOE, P_GATE))
        return
    _require_params(moe_cls.forward, ["self", "x", "forward_batch", "combine"], P_MOE)
    _require_params(gate_cls.forward, ["self", "x"], P_GATE)
    _require_source(moe_cls.forward, ["self.gate(x)", "self.experts("], P_MOE, ordered=True)

    orig_moe = moe_cls.forward
    orig_gate = gate_cls.forward

    def moe_forward(self, x, forward_batch=None, combine=True):
        prev = STATE.in_verify_moe
        mode = getattr(forward_batch, "forward_mode", None)
        STATE.in_verify_moe = bool(
            STATE.enabled and mode is not None and mode.is_target_verify()
        )
        try:
            return orig_moe(self, x, forward_batch, combine)
        finally:
            STATE.in_verify_moe = prev

    def gate_forward(self, x):
        out = orig_gate(self, x)
        if not (STATE.enabled and STATE.in_verify_moe):
            return out
        weights, ids = out
        cfg = STATE.config
        if STATE.capping and cfg.remap:
            weights, ids = STATE.remap(weights, ids)
        if cfg.union:
            # Observational only: never changes what the experts receive.
            STATE.count_union(self, ids, int(self.weight.shape[0]))
        return weights, ids

    moe_cls.forward = _mark(moe_forward, orig_moe)
    gate_cls.forward = _mark(gate_forward, orig_gate)
    STATE.applied.update((P_MOE, P_GATE))


# ------------------------------------------------------------------ verify / accept
def _wrap_accept_draft_tokens(orig):
    _require_subset(orig, ["candidates", "cutoff_layout", "block_verification"],
                    "accept_draft_tokens")
    sig = inspect.signature(orig)
    if sig.parameters["candidates"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise PatchTargetError(f"{PREFIX} accept_draft_tokens is no longer keyword-only")

    def accept_draft_tokens(**kwargs):
        if not STATE.enabled:
            return orig(**kwargs)
        n = int(kwargs["candidates"].shape[0])
        if kwargs.get("cutoff_layout") is None:
            cutoff = STATE.cutoff_for(n)
            if cutoff is not None:
                kwargs["cutoff_layout"] = cutoff
        cutoff = kwargs.get("cutoff_layout")
        if STATE.fused_accept is not None:
            fused = _fused_sampled_accept(orig, kwargs, cutoff)
            if fused is not None:
                return fused
        bv_lens = None
        if kwargs.get("block_verification") and cutoff is not None and STATE.config.bv_mask:
            # Block verification reads every row: the dead drafts must look stopped
            # (README, "Block verification under the cap"); _draft_probs masks them.
            bv_lens = STATE.block_verify_lens(cutoff.verify_lens, n)
        if bv_lens is None:
            return orig(**kwargs)
        STATE.bv_lens = bv_lens
        try:
            return orig(**kwargs)
        finally:
            STATE.bv_lens = None

    return _mark(accept_draft_tokens, orig)


def _fused_sampled_accept(orig, kwargs, cutoff):
    """The sampled branch of accept_draft_tokens through fused_accept.accept_core, or
    None when this call is not eligible (the stock path then runs)."""
    from inkling_vcap import fused_accept

    si = kwargs.get("sampling_info")
    block = kwargs.get("draft_block")
    cand = kwargs["candidates"]
    logits = kwargs.get("target_logits")
    if (
        si is None or si.is_all_greedy or si.need_top_k_sampling or si.need_top_p_sampling
        or block is None or getattr(block, "greedy_draft", True)
        or getattr(block, "vocab_shard", None) is not None
        or kwargs.get("rejection_noise") is not None
        or logits is None or not logits.is_cuda or torch.cuda.is_current_stream_capturing()
    ):
        return None
    bs, stride = cand.shape
    if bs < STATE.config.fused_accept_min_bs:
        return None
    dlog = block.corrected_logits
    vocab = logits.shape[-1]
    if (
        dlog is None or dlog.dim() != 3 or tuple(dlog.shape[:2]) != (bs, stride - 1)
        or dlog.shape[-1] != vocab or dlog.stride(-1) != 1 or logits.stride(-1) != 1
        or logits.shape[0] != bs * stride
    ):
        return None
    t = si.temperatures.reshape(-1)[:bs].to(torch.float32)
    t_temps = torch.where(t > 0, t, torch.ones_like(t))
    d_temps = block.temperatures.reshape(-1)[:bs].to(torch.float32)
    coins = torch.rand((bs, stride - 1), dtype=torch.float32, device=logits.device)
    STATE.fused_accept_calls += 1
    sampled = fused_accept.accept_core(
        STATE.fused_accept, candidates=cand, target_logits=logits, draft_logits=dlog,
        t_temps=t_temps, d_temps=d_temps,
        live_lens=None if cutoff is None else cutoff.verify_lens,
        block_verification=bool(kwargs.get("block_verification")), coins=coins,
    )
    if not si.is_any_greedy:
        return sampled
    mod = _ACCEPT_MODULE["mod"]
    greedy_len, greedy_bonus, greedy_trim = mod.AcceptGreedy.execute(
        candidates=cand, target_logits=logits,
        verify_num_draft_tokens=kwargs["verify_num_draft_tokens"],
        cutoff_verify_lens=None if cutoff is None else cutoff.verify_lens,
    )
    sampling_len, sampling_bonus, sampling_trim = sampled
    sel = mod.SelectMixedAccept.execute(
        greedy_mask=block.greedy_mask, greedy_len=greedy_len, greedy_bonus=greedy_bonus,
        greedy_trim=greedy_trim, sampling_len=sampling_len.to(greedy_len.dtype),
        sampling_bonus=sampling_bonus.to(greedy_bonus.dtype),
        sampling_trim=sampling_trim.to(greedy_trim.dtype),
    )
    return sel.correct_len, sel.bonus, sel.cap_trim_lens


_ACCEPT_MODULE: dict = {}


def _wrap_draft_probs(orig):
    _require_params(orig, ["draft_block", "vocab", "device"], P_DRAFT_PROBS)

    def _draft_probs(draft_block, vocab, device):
        probs = None
        if STATE.enabled and STATE.use_fastprobs and not draft_block.greedy_draft:
            logits = draft_block.corrected_logits
            if (
                logits is not None and logits.is_cuda and logits.dim() == 3
                and logits.shape[-1] == vocab and logits.stride(-1) == 1
                and getattr(draft_block, "vocab_shard", None) is None
            ):
                bs, gamma, _ = logits.shape
                STATE.fastprobs_calls += 1
                probs = fastprobs.softmax_temp(
                    logits.reshape(bs * gamma, vocab), draft_block.temperatures, gamma
                ).view(bs, gamma, vocab)
        if probs is None:
            probs = orig(draft_block, vocab, device)
        lens = STATE.bv_lens
        if lens is not None:
            STATE.bv_masked += 1
            logic.mask_dead_draft_rows(probs, draft_block.draft_tokens, lens)
        return probs

    return _mark(_draft_probs, orig)


def patch_verify(mod) -> None:
    epi = _attr(mod, "DsparkVerifyEpilogue", MOD_VERIFY)
    execu = _attr(mod, "TargetVerifyExecutor", MOD_VERIFY)
    adt = _attr(mod, "accept_draft_tokens", MOD_VERIFY)

    if not _is_marked(epi._accept_static_greedy):
        _require_params(
            epi._accept_static_greedy,
            ["self", "io", "verify_lens", "gate", "gated"],
            P_STATIC_ACCEPT,
        )
        _require_subset(epi._accept, ["candidates", "logits", "draft_tokens",
                                      "seq_lens", "cutoff_verify_lens"],
                        "DsparkVerifyEpilogue._accept")
        _require_source(epi._static_epilogue, ["self._accept_static_greedy"],
                        "DsparkVerifyEpilogue._static_epilogue")
        _require_source(epi.run_static, ["verify_lens = self.verify_lens_buf[:bs]",
                                         "accept(io, verify_lens=verify_lens"],
                        "DsparkVerifyEpilogue.run_static")
        orig_static = epi._accept_static_greedy

        def _accept_static_greedy(self, io, *, verify_lens, gate, gated):
            if not (STATE.enabled and STATE.capping):
                return orig_static(self, io, verify_lens=verify_lens, gate=gate,
                                   gated=gated)
            # The greedy accept with the cutoff, recorded into the verify graph; the
            # fused accept tail (which never caps) is bypassed while the cap is on.
            STATE.static_accept_recorded += 1
            commit_lens = self._accept(
                candidates=io.candidates,
                logits=io.logits,
                draft_tokens=io.candidates[:, 1:].contiguous(),
                seq_lens=io.seq_lens,
                cutoff_verify_lens=verify_lens,
            )
            if not gated:
                return None
            return torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate

        epi._accept_static_greedy = _mark(_accept_static_greedy, orig_static)

    if not _is_marked(execu.run_non_compact):
        _require_params(
            execu.run_non_compact,
            ["self", "batch", "draft_input", "verify_ids_2d", "verify_window",
             "sampling_info"],
            P_RUN_NON_COMPACT,
        )
        orig_rnc = execu.run_non_compact

        def run_non_compact(self, *args, **kwargs):
            if args:
                raise PatchTargetError(
                    f"{PREFIX} run_non_compact called positionally; re-check the patch"
                )
            if STATE.enabled:
                _before_verify(int(kwargs["verify_ids_2d"].shape[0]))
            return orig_rnc(self, **kwargs)

        execu.run_non_compact = _mark(run_non_compact, orig_rnc)

    if not _is_marked(adt):
        mod.accept_draft_tokens = _wrap_accept_draft_tokens(adt)
    _ACCEPT_MODULE["mod"] = mod
    dp = getattr(mod, "_draft_probs", None)
    if dp is not None:
        if not _is_marked(dp):
            mod._draft_probs = _wrap_draft_probs(dp)
        STATE.applied.add(P_DRAFT_PROBS)
    STATE.applied.update((P_STATIC_ACCEPT, P_RUN_NON_COMPACT, P_ACCEPT_VERIFY))


def patch_sampling_accept_graph(mod) -> None:
    adt = _attr(mod, "accept_draft_tokens", MOD_SAG)
    _require_source(_attr(mod, "sampling_accept_commit_lens", MOD_SAG),
                    ["accept_draft_tokens("], "sampling_accept_commit_lens")
    if not _is_marked(adt):
        mod.accept_draft_tokens = _wrap_accept_draft_tokens(adt)
    STATE.applied.add(P_ACCEPT_SAG)


def _before_verify(bs: int) -> None:
    if not STATE._validated_first_step:
        _first_step_checks()
        STATE._validated_first_step = True
    proposal, STATE.last_proposal = STATE.last_proposal, None
    if proposal is None and STATE.config.uses_draft_probs:
        raise RuntimeError(f"{PREFIX} target verify without a stashed draft proposal")
    STATE.plan_step(bs=bs, proposal=proposal)


def _first_step_checks() -> None:
    worker = STATE.worker
    cfg = STATE.config
    sampler = getattr(worker, "_draft_sampler", None)
    if cfg.uses_draft_probs and sampler is not None:
        strategy = type(getattr(sampler, "_draft_strategy", None)).__name__
        if strategy != "DsparkDraftStrategy" or getattr(sampler, "_markov_walker", None):
            raise RuntimeError(
                f"{PREFIX} {cfg.kind} mode reads every row of the draft step logits, "
                f"but the draft sampler runs {strategy} / int8 markov walk, which may "
                "leave greedy rows unwritten. Use fixed:K, or disable that path."
            )
    epilogue = getattr(worker, "_verify_epilogue", None)
    captured = bool(getattr(epilogue, "static_io", None))
    if (
        STATE.capping and epilogue is not None and captured and cfg.remap
        and STATE.remap_recorded > 0 and STATE.static_accept_recorded == 0
    ):
        raise RuntimeError(
            f"{PREFIX} remap was recorded in the verify graphs but the in-graph accept "
            "cap was not; refusing to run (output would not be exact)."
        )
    if not STATE.capping:
        log("full width for every batch size (e.g. fixed:8): verification is "
            "untouched (no remap, no cutoff, stock accept tail); stats/union only")
    elif cfg.remap and STATE.remap_recorded == 0:
        log("WARNING: no MoE remap recorded before the first decode step "
            "(verify graphs captured before the overlay attached?); the cap stays "
            "exact but saves nothing.")
    log(
        f"first decode step: remap recorded in {STATE.remap_recorded} MoE calls, "
        f"in-graph capped accept recorded {STATE.static_accept_recorded}x, "
        f"captured static verify bs={sorted(getattr(epilogue, 'static_io', {}) or [])}"
    )
    if STATE._union_acc is not None:
        STATE._union_acc.zero_()  # drop warmup / capture counts


# ------------------------------------------------------------------ draft
def patch_draft(mod) -> None:
    proposer = _attr(mod, "DraftBlockProposer", MOD_DRAFT)
    if not _is_marked(proposer.propose):
        _require_subset(proposer.propose, ["batch", "draft_input", "bs",
                                           "sampling_info"], P_PROPOSE)
        orig = proposer.propose

        def propose(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            if STATE.enabled:
                STATE.last_proposal = out
            return out

        proposer.propose = _mark(propose, orig)
    STATE.applied.add(P_PROPOSE)


# ------------------------------------------------------------------ worker
def patch_worker(mod) -> None:
    worker_cls = _attr(mod, "DSparkWorkerV2", MOD_WORKER)
    _require_source(
        worker_cls._forward_decode,
        ["self._proposer.propose(", "begin_static_step(", "run_non_compact(",
         "accept_and_finalize("],
        P_FORWARD_DECODE,
        ordered=True,
    )
    if not _is_marked(worker_cls.__init__):
        orig_init = worker_cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if getattr(self, "_hosts_draft", False):
                attach(self)

        worker_cls.__init__ = _mark(__init__, orig_init)
    if not _is_marked(worker_cls._forward_decode):
        orig_fd = worker_cls._forward_decode

        def _forward_decode(self, *args, **kwargs):
            result = orig_fd(self, *args, **kwargs)
            if STATE.enabled and STATE.worker is self:
                STATE.observe(result)
            return result

        worker_cls._forward_decode = _mark(_forward_decode, orig_fd)
    STATE.applied.update((P_WORKER_INIT, P_FORWARD_DECODE))


HOOKS = {
    MOD_MOE: patch_moe,
    MOD_VERIFY: patch_verify,
    MOD_SAG: patch_sampling_accept_graph,
    MOD_DRAFT: patch_draft,
    MOD_WORKER: patch_worker,
}


def _runtime_context():
    try:
        return importlib.import_module("sglang.srt.runtime_context")
    except ImportError:
        return None


def _capacity_without_epilogue() -> int:
    rc = _runtime_context()
    sizes = [1]
    if rc is not None:
        try:
            sizes.append(max(rc.get_exec().graph.cuda_graph_config.decode.bs))
        except Exception:
            pass
        try:
            sizes.append(int(rc.get_schedule().max_running_requests or 0))
        except Exception:
            pass
    return max(sizes)


def fastprobs_selfcheck(device) -> "str | None":
    """None when fastprobs matches the torch reference on this device; else why not."""
    try:
        if not fastprobs.available(device):
            return "not CUDA or the image's split softmax kernels are missing"
        gen = torch.Generator(device="cpu").manual_seed(0)
        bs, gamma, vocab = 3, 7, 201024
        logits = (torch.randn((bs, gamma, vocab), generator=gen) * 4).to(torch.bfloat16)
        logits[:, :, 150000:] = float("-inf")  # hot-vocabulary style rows
        logits = logits.to(device)
        temps = torch.tensor([0.8, 0.0, 1.3], device=device)
        greedy = torch.tensor([False, True, False], device=device)
        ref_m = logic.draft_token_probs(logits, torch.zeros(bs, gamma, dtype=torch.long,
                                        device=device), temperatures=temps,
                                        greedy_mask=greedy, use_max=True)
        got_m = fastprobs.max_prob(logits, temperatures=temps, greedy_mask=greedy)
        err_m = float((ref_m - got_m).abs().max())
        clamped = temps.clamp_min(1e-5)
        x = logits.float().reshape(bs * gamma, vocab) / clamped.repeat_interleave(gamma)[:, None]
        ref_q = torch.softmax(x, dim=-1)
        got_q = fastprobs.softmax_temp(logits.reshape(bs * gamma, vocab), clamped, gamma)
        err_q = float((ref_q - got_q).abs().max())
        sum_err = float((got_q.sum(-1) - 1).abs().max())
        if not (err_m < 1e-5 and err_q < 1e-6 and sum_err < 1e-4):
            return f"self-check mismatch: m {err_m:.2e}, q {err_q:.2e}, sum {sum_err:.2e}"
        return None
    except Exception as exc:  # noqa: BLE001
        return f"self-check failed: {exc!r}"[:200]


def fused_accept_selfcheck(device) -> "str | None":
    """Install the Triton primitives in STATE.fused_accept when they agree with the dense
    torch primitives on this device; else return why not (the stock accept stays)."""
    STATE.fused_accept = None
    try:
        from inkling_vcap import fused_accept as fa

        mod = _ACCEPT_MODULE.get("mod")
        src = inspect.getsource(mod.accept_draft_tokens.__wrapped__)
        for needle in ("AcceptGreedy.execute(", "SelectMixedAccept.execute(",
                       "sampling_info.is_any_greedy", "AcceptSampling.execute("):
            if needle not in src:
                return f"accept_draft_tokens no longer contains {needle!r}"
        if torch.device(device).type != "cuda" or fa.triton is None or not fastprobs.available(device):
            return "not CUDA, or Triton / the split softmax kernels are missing"
        tp, rp = fa.TritonPrims(), fa.TorchPrims()
        gen = torch.Generator(device="cpu").manual_seed(0)
        bs, S, V = 3, 8, 201024
        G = S - 1
        tl2 = (torch.randn((bs * S, V), generator=gen) * 3).to(device)
        dl2 = (torch.randn((bs * G, V), generator=gen) * 3).to(torch.bfloat16)
        dl2[:, 150000:] = float("-inf")
        dl2 = dl2.to(device)
        tt = torch.tensor([0.8, 1.0, 0.5], device=device)
        td = tt * 0.7
        need_t = torch.ones(bs * S, dtype=torch.bool, device=device)
        need_d = torch.ones(bs * G, dtype=torch.bool, device=device)
        st_ref = rp.stats(tl2, tt, S, need_t) + rp.stats(dl2, td, G, need_d)
        st_got = tp.stats(tl2, tt, S, need_t) + tp.stats(dl2, td, G, need_d)
        e_lse = max(float((a_ - b_).abs().max()) for a_, b_ in zip(st_ref, st_got))
        st = (st_got[0].view(bs, S), st_got[1].view(bs, S), st_got[2].view(bs, G),
              st_got[3].view(bs, G))
        pp = torch.rand((bs, G), generator=gen).to(device)
        need = torch.ones((bs, G), dtype=torch.bool, device=device)
        need[:, G - 1] = False
        m_ref = rp.masses(tl2, dl2, st, tt, td, pp, need)
        m_got = tp.masses(tl2, dl2, st, tt, td, pp, need)
        e_mass = float((m_ref - m_got).abs().max())
        # the race only picks tokens with positive residual weight
        tau = torch.tensor([0, 3, 7], device=device)
        scale = torch.tensor([1.0, 0.6, 1.0], device=device)
        use_q = torch.tensor([True, True, False], device=device)
        pick = tp.sample(tl2, dl2, st, tt, td, tau, scale, use_q)
        P, Q = rp._dense(tl2, dl2, st, tt, td)
        rows = torch.arange(bs, device=device)
        w = (scale * P[rows, tau, pick] - Q[rows, tau.clamp(max=G - 1), pick] * use_q).clamp(min=0)
        if not (e_lse < 1e-4 and e_mass < 1e-5 and bool((w > 0).all())):
            return f"self-check mismatch: lse {e_lse:.1e}, mass {e_mass:.1e}, race w {w.tolist()}"
        STATE.fused_accept = tp
        return None
    except Exception as exc:  # noqa: BLE001
        return f"self-check failed: {exc!r}"[:200]


def attach(worker) -> None:
    """Validate the serving setup, allocate the buffers, and enable the hooks."""
    cfg = STATE.config
    for name in HOOKS:
        importlib.import_module(name)  # triggers the hook if not yet imported
    missing = [p for p in REQUIRED if p not in STATE.applied]
    if missing:
        raise PatchTargetError(f"{PREFIX} patches not applied: {missing}")

    stride = int(worker.verify_num_draft_tokens)
    cfg.validate_for_stride(stride)
    planner = getattr(worker, "_verify_planner", None)
    if getattr(planner, "mode_value", None) != "static":
        raise RuntimeError(
            f"{PREFIX} needs SGLANG_RAGGED_VERIFY_MODE=static (got "
            f"{getattr(planner, 'mode_value', None)!r}); the cap replaces ragged verify"
        )
    if float(getattr(worker, "_simulate_acc_len", 0.0) or 0.0) > 0:
        raise RuntimeError(f"{PREFIX} incompatible with SGLANG_SIMULATE_ACC_LEN")
    rc = _runtime_context()
    block_verify = False
    if rc is not None:
        spec = rc.get_spec()
        block_verify = bool(getattr(spec, "speculative_use_block_verification", False))
        if block_verify and P_DRAFT_PROBS not in STATE.applied:
            raise PatchTargetError(
                f"{PREFIX} --speculative-use-block-verification needs the "
                f"{P_DRAFT_PROBS} patch (dead drafts masked as a stopped draft); "
                "the target is missing in this engine"
            )
        par = rc.get_parallel()

        def _p(name, default):
            try:
                return getattr(par, name)
            except Exception:
                return default

        if int(_p("pp_size", 1) or 1) > 1:
            raise RuntimeError(f"{PREFIX} pipeline parallelism is not supported")
        if cfg.uses_draft_probs and (
            int(_p("tp_size", 1) or 1) > 1 or bool(_p("attn_dp_enabled", False))
        ):
            raise RuntimeError(
                f"{PREFIX} {cfg.kind} mode is validated for tp=1 only (rank-invariant "
                "live lengths are not synced); use fixed:K with tp>1"
            )
    if cfg.uses_draft_probs and getattr(worker, "_greedy_draft", False):
        raise RuntimeError(
            f"{PREFIX} {cfg.kind} mode needs draft step logits; "
            "SGLANG_DSPARK_USE_GREEDY_DRAFT=1 drops them (use fixed:K)"
        )
    if getattr(worker, "_sharded_accept", False):
        raise RuntimeError(f"{PREFIX} the vocab-sharded accept is not supported")

    moe_mod = importlib.import_module(MOD_MOE)
    target = getattr(worker, "target_worker", None) or worker._target_worker
    model = target.model_runner.model
    n_moe = 0
    for name, module in model.named_modules():
        if isinstance(module, moe_mod.Inkling2MoE):
            short = name.replace("model.", "").replace("layers.", "L").replace(".mlp", "")
            module._vcap_name = short
            module.gate._vcap_name = short
            n_moe += 1
    if n_moe == 0:
        raise RuntimeError(
            f"{PREFIX} the target model has no Inkling2MoE layers; the overlay only "
            "supports Inkling2 targets"
        )

    epilogue = getattr(worker, "_verify_epilogue", None)
    if epilogue is not None:
        if getattr(epilogue, "simulate_fold", False):
            raise RuntimeError(f"{PREFIX} incompatible with SGLANG_DSPARK_SIMULATE_FOLD")
        if int(epilogue.stride) != stride:
            raise RuntimeError(f"{PREFIX} epilogue stride {epilogue.stride} != {stride}")
        capacity = int(epilogue.max_bs)
        STATE.allocate(stride=stride, capacity=capacity, device=worker.device,
                       live_lens=epilogue.verify_lens_buf)
        STATE.live_lens_source = "DsparkVerifyEpilogue.verify_lens_buf"
    else:
        capacity = max(_capacity_without_epilogue(), 256)
        STATE.allocate(stride=stride, capacity=capacity, device=worker.device)
        STATE.live_lens_source = "own buffer (no static epilogue: eager accept)"
    STATE.epilogue = epilogue
    STATE.capping = not cfg.never_caps(stride)
    fused_note = "off"
    if STATE.capping and cfg.remap and cfg.fused:
        from inkling_vcap import fused

        if torch.device(worker.device).type != "cuda":
            fused_note = "off (not CUDA)"
        else:
            err = fused.selfcheck(worker.device)
            STATE.use_fused = err is None
            fused_note = "on (self-check passed)" if err is None else f"off ({err})"
    fa_note = "off"
    if cfg.fused_accept:
        err = fused_accept_selfcheck(worker.device)
        fa_note = "on (self-check passed)" if err is None else f"off ({err})"
    fast_note = "off"
    if cfg.fastprobs:
        err = fastprobs_selfcheck(worker.device)
        STATE.use_fastprobs = err is None
        fast_note = "on (self-check passed)" if err is None else f"off ({err})"
    STATE.worker = worker
    STATE.block_verify = block_verify
    STATE.enabled = True
    log(
        f"enabled mode={cfg.describe()} capping={'on' if STATE.capping else 'off (inert)'} "
        f"block_verification={('on, ' + ('dead drafts masked as a stopped draft' if cfg.bv_mask else 'stock kernel + cap')) if block_verify else 'off'} "
        f"stride={stride} capacity={capacity} reqs "
        f"moe_layers={n_moe} live_lens={STATE.live_lens_source} "
        f"fused_remap={fused_note} fastprobs={fast_note} fused_accept={fa_note}; patched: "
        + ", ".join(REQUIRED)
    )
