"""SGLang patch points of inkling_cert (README.md has the table and the argument).

  inkling2       Inkling2LogitsProcessor._get_logits: marks target-verify calls (the only ones the certificate takes)
                 Inkling2ForConditionalGeneration.post_load_weights: builds the 6-bit copy and the bound tables after
                 inkling_pico12 coded the head, then runs the start-up self-test
  unquant        sm12x_fp32_lm_head: target-verify calls that pico's `lmhead` takes (1-16 rows) go through the
                 certified head; everything else to the wrapped function (pico's, which wraps the stock one)
  dspark_verify  DsparkVerifyEpilogue.begin_static_step / begin_step: remember whether the step arms the in-graph
                 greedy accept; TargetVerifyExecutor.run_non_compact: writes the device flag `on` before the replay
"""

import functools
import os
import sys
import time

import torch

from . import build, cert_math, ops, quant

K = cert_math.K
UNQUANT = "sglang.srt.layers.quantization.unquant"
MODEL = "sglang.srt.models.inkling2"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"

# Debug observers that read the verify logits after a folded greedy accept (dspark_observability.py)
OBSERVER_ENVS = ("SGLANG_DSPARK_STS_COLLECT_PATH", "SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS",
                 "SGLANG_DSPARK_DEBUG_DUMP", "SGLANG_DSPARK_ENABLE_SPS_RECORD")

STATE = {"mode": "1", "cfg": None, "log": None, "patched": [], "cert": None, "disabled": None, "calls": 0}
CTX = {"lp": None}  # the Inkling2LogitsProcessor of the target-verify call in progress
EXIT = os._exit


def _log(msg):
    (STATE["log"] or (lambda m: sys.stderr.write(m + "\n")))(msg)


def _need(owner, names, where):
    missing = [n for n in names if getattr(owner, n, None) is None]
    if missing:
        raise ops.CertFatal(f"inkling_cert: patch target {where} lacks {missing}")


def _guarded(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001
            _log(f"FATAL: {exc!r}")
            EXIT(78)

    return run


def _disable(why):
    if STATE["disabled"] is None:
        STATE["disabled"] = why
        _log(f"disabled: {why}; the head runs unchanged")


def _pico():
    """inkling_pico12's patches / ops modules, or None when its `lmhead` lever is not armed."""
    pp = sys.modules.get("inkling_pico12.patches")
    po = sys.modules.get("inkling_pico12.ops")
    if pp is None or po is None or "lmhead" not in tuple(pp.STATE.get("levers", ())):
        return None, None
    return pp, po


# ---- inkling2: verify context and table build ------------------------------------------------------------------


def make_get_logits(inner):
    def _get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        mode = getattr(logits_metadata, "forward_mode", None)
        verify = mode is not None and mode.is_target_verify()
        if verify:
            result = _wide_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs)
            if result is not None:
                return result
        prev = CTX["lp"]
        CTX["lp"] = self if verify else None
        try:
            return inner(self, hidden_states, lm_head, logits_metadata, *args, **kwargs)
        finally:
            CTX["lp"] = prev

    _get_logits.__wrapped__ = inner
    _get_logits._inkling_cert = True
    return _get_logits


def _wide_modules():
    return sys.modules.get("inkling_wide.patches"), sys.modules.get("inkling_wide.ops")


def _wide_logits(lp, x, head, meta, embedding_bias=None, use_logits_buffer=True):
    cert = STATE["cert"]
    wide = getattr(cert, "wide", None)
    if (wide is None or STATE["disabled"] is not None or STATE["mode"] != "1"
            or not x.is_cuda or not torch.cuda.is_current_stream_capturing()):
        return None
    from . import certwide

    wp, wo = _wide_modules()
    weight = getattr(head, "weight", None)
    entry = wide.entry
    m = wo.rows_of(x)
    if (not certwide.MIN_ROWS <= m <= min(certwide.MAX_ROWS, STATE["cfg"].get("wide_max_rows", certwide.MAX_ROWS))
            or x.dim() != 2
            or not isinstance(weight, torch.Tensor) or wp.HEADS.get(weight.data_ptr()) is not entry
            or tuple(weight.shape) != (entry.n, entry.k) or not wo.eligible(x, entry)
            or wo.choose_split(entry.n, entry.k, m, entry.kind) != 1):
        return None
    if (embedding_bias is not None or not getattr(lp, "use_fp32_lm_head", False)
            or getattr(lp, "shard_softcap", None) != cert_math.CAP
            or getattr(lp, "logit_scale", None) is not None
            or getattr(lp, "final_logit_softcapping", None) is not None
            or getattr(lp, "do_tensor_parallel_all_gather", False)
            or getattr(lp, "do_tensor_parallel_all_gather_dp_attn", False)
            or getattr(lp, "vocab_size", None) != entry.n
            or getattr(meta, "extend_return_logprob", False)
            or hasattr(head, "set_lora")):
        return None
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        return None
    v_real = int(lp.config.vocab_size)
    if not 0 < v_real <= entry.n:
        return None
    out = meta.next_token_logits_buffer if use_logits_buffer else None
    # The stock implementation discards a buffer with a different shape.
    if out is not None and tuple(out.shape) != (m, entry.n):
        out = None
    if out is None:
        out = torch.empty((m, entry.n), dtype=torch.float32, device=x.device)
    if (not wo.out_ok(out, m, entry, torch.float32, x.device) or out.data_ptr() % 32
            or not out.is_contiguous()):
        return None
    # The tensor is the graph's existing fallback buffer. Its attribute is
    # consumed at capture by the folded accept, which records the ID pointer.
    out = out.view_as(out)  # keep the attribute off the graph's shared buffer object
    out._inkling_certwide_ids = wide.run(wo.as_rows(x, entry.k), out, cert.on, v_real, wo.SET)
    from . import dflash

    dflash.note_wide_ids(out, out._inkling_certwide_ids)  # DFLASH: the eager accept looks them up after replay
    return out


def _prepare_wide(cert, w, lp, tables):
    from . import certwide

    wp, wo = _wide_modules()
    entry = wp.HEADS.get(w.data_ptr()) if wp is not None else None
    if entry is None or entry.fmt != wo.BF16X or entry.pack is not cert.pack:
        raise ops.CertFatal("inkling_certwide: needs the wide head registered with pico's packed weights")
    if wo.choose_split(entry.n, entry.k, certwide.MAX_ROWS, entry.kind) != 1:
        raise ops.CertFatal("inkling_certwide: needs the unsplit packed wide head")
    if getattr(cert, "wide", None) is None:
        cert.wide = certwide.CertWide(cert.ext, wo.EXT, entry, tables)
    else:
        cert.wide.refresh(entry, tables)

    cert.wide.counting = False  # the self-tests below are not served steps
    with torch.inference_mode():
        # Warm the CUDA/Triton paths before capture. Use peaked BF16 rows so
        # the sparse arithmetic and the forced full fallback are both checked.
        for m in (17, 33, 64, 96, 128, 192):
            x = (w[:m].float() * 60).to(torch.bfloat16)
            out = torch.empty((m, entry.n), dtype=torch.float32, device=w.device)
            ref = torch.empty_like(out)
            wo.gemm(x, entry, ref, m)
            v_real = int(lp.config.vocab_size)
            expected = _softcap_argmax(ref, v_real)
            for on in (cert.force, cert.off):
                ids = cert.wide.run(x, out, on, v_real, wo.SET)
                if not torch.equal(ids, expected):
                    raise ops.CertFatal(f"inkling_certwide: M={m} self-test token mismatch")
                if on is cert.off and not torch.equal(out.view(torch.int32), ref.view(torch.int32)):
                    raise ops.CertFatal(f"inkling_certwide: M={m} fallback logit mismatch")
    cert.wide.counting = True
    if STATE["cfg"].get("wide_stats") and cert.wide.stats is None:  # once: graphs capture the counter address
        cert.wide.enable_stats()
    _log(f"wide certified packed head armed (17–{STATE['cfg'].get('wide_max_rows', certwide.MAX_ROWS)} rows, "
         f"compact capacity {cert.wide.capacity} tiles{', outcome stats on' if cert.wide.stats is not None else ''})")


def prepare(model):
    if STATE["mode"] == "0":
        return
    for env in OBSERVER_ENVS:
        if os.environ.get(env, "").strip() not in ("", "0", "false", "False"):
            _disable(f"{env} is set (that observer reads the verify logits)")
            return
    pp, po = _pico()
    if pp is None:
        raise ops.CertFatal("inkling_cert: needs INKLING_PICO12 with `lmhead` (pico's GEMV is the exact pass)")
    lm_head = getattr(getattr(model, "llm", None), "lm_head", None)
    w = getattr(lm_head, "weight", None)
    pack = pp.HEADS.get(w.data_ptr()) if isinstance(w, torch.Tensor) else None
    if pack is None:
        raise ops.CertFatal("inkling_cert: inkling_pico12 has not coded the head (hook order: pico12 must come first)")
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001
        tp = 1
    if tp != 1:
        _disable(f"tensor parallel size {tp} (TP1 only)")
        return
    unq = sys.modules.get(UNQUANT)
    if unq is None or not getattr(unq.sm12x_fp32_lm_head, "_inkling_cert", False):
        raise ops.CertFatal("inkling_cert: unquant.sm12x_fp32_lm_head is not ours (another overlay wrapped it after us)")
    t0 = time.monotonic()
    ext = build.load(STATE["cfg"]["cache_dir"])
    t_ext = time.monotonic() - t0
    old = STATE["cert"]
    _require_wide_replay_stream()
    tables = quant.build_tables(w.data)
    if old is not None:  # a reload: refresh in place (captured graphs hold these pointers)
        for name in ("q", "sc", "rt", "nv"):
            getattr(old, name).copy_(tables[name])
        old.pack = pack
        cert = old
    else:
        cert = ops.Cert(ext, pack, tables, po.SET.wk, w.device)
    lp = getattr(getattr(model, "llm", None), "logits_processor", None)
    if STATE["cfg"].get("wide"):
        _prepare_wide(cert, w, lp, tables)
    del tables
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selftest(cert, po, v_real=int(getattr(getattr(lp, "config", None), "vocab_size", 0) or 0) or None)
    STATE["cert"] = cert
    info = build.INFO
    _log(f"active (mode {STATE['mode']}): 6-bit screen of the [{cert.V}x{K}] head, {cert.screen_bytes / 1e6:.1f} MB "
         f"read per certified step (pico's coded head: {pack.nbytes / 1e6:.1f} MB), eps {cert.eps * 100:.2f} %; "
         f"patched {', '.join(STATE['patched'])}; ext {'reused' if info.get('cached') else 'built'} in {t_ext:.1f} s; "
         f"tables {time.monotonic() - t0 - t_ext:.1f} s; wk={cert.wk}")


@torch.inference_mode()
def selftest(cert, po, v_real=None, n=4):
    """Start-up check on random rows (M = 8 and 16): the not-certified path equals pico's full head bit for bit,
    the certified path gives the full head's softcapped argmax. Also loads every kernel before graph capture."""
    dev = cert.q.device
    g = torch.Generator(device=dev)
    g.manual_seed(1234)
    v_real = v_real or cert.V
    for m in (8, 16):
        for i in range(n):
            x = (torch.randn((m, K), generator=g, device=dev) * (1.0 + i)).to(torch.bfloat16)
            ref = torch.empty((m, cert.V), dtype=torch.float32, device=dev)
            po.gemv(x, cert.pack, ref)
            a = torch.empty_like(ref)
            cert.run(x, a, cert.off, v_real, cert_math.CAP)
            if not torch.equal(a.view(torch.int32), ref.view(torch.int32)):
                raise ops.CertFatal(f"inkling_cert: self-test M={m}: the full fallback differs from pico's head")
            b = torch.empty_like(ref)
            cert.run(x, b, cert.force, v_real, cert_math.CAP)
            ta = _softcap_argmax(b, v_real)
            tb = _softcap_argmax(ref.clone(), v_real)
            if not torch.equal(ta, tb):
                raise ops.CertFatal(f"inkling_cert: self-test M={m}: certified argmax differs from the full head")
    cert.stats.zero_()
    torch.cuda.synchronize()


def _softcap_argmax(logits, v_real, cap=cert_math.CAP):
    sc, am = _stock_fns()
    sc(logits, cap)
    logits[:, v_real:] = float("-inf")
    return am(logits)


_FNS = []


def _stock_fns():
    if not _FNS:
        from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
        from sglang.kernels.ops.sampling import argmax as sampling_argmax

        _FNS.extend([softcap_inplace_logits, sampling_argmax])
    return _FNS[0], _FNS[1]


def patch_model(mod):
    _need(mod, ["Inkling2LogitsProcessor", "Inkling2ForConditionalGeneration"], MODEL)
    lp_cls = mod.Inkling2LogitsProcessor
    _need(lp_cls, ["_get_logits", "_compute_lm_head"], f"{MODEL}.Inkling2LogitsProcessor")
    if not getattr(lp_cls._get_logits, "_inkling_cert", False):
        lp_cls._get_logits = make_get_logits(lp_cls._get_logits)
        STATE["patched"].append("Inkling2LogitsProcessor._get_logits")
    cls = mod.Inkling2ForConditionalGeneration
    _need(cls, ["post_load_weights"], f"{MODEL}.Inkling2ForConditionalGeneration")
    orig = cls.post_load_weights
    if getattr(orig, "_inkling_cert", False):
        return

    @functools.wraps(orig)
    def post_load_weights(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)  # inkling_pico12 (if armed before us) codes the head in here
        _guarded(prepare)(self)
        return result

    post_load_weights._inkling_cert = True
    cls.post_load_weights = post_load_weights
    STATE["patched"].append("Inkling2ForConditionalGeneration.post_load_weights")


# ---- unquant: the head ---------------------------------------------------------------------------------------


def _eligible(lp, x, weight, out, cert):
    """Our call iff pico's `lmhead` would take it (the reference bits are pico's) and the processor is as assumed."""
    pp, po = _pico()
    if pp is None or cert is None:
        return None
    pack = pp.HEADS.get(weight.data_ptr()) if isinstance(weight, torch.Tensor) else None
    if pack is None or pack is not cert.pack or tuple(weight.shape) != (pack.n, pack.k):
        return None
    m = po.rows_of(x)
    if not (1 <= m <= ops.MAX_ROWS) or not po.eligible(x, pack):
        return None
    if out is not None and not po.out_ok(out, m, pack, torch.float32, x.device):
        return None
    cap = getattr(lp, "shard_softcap", None)
    if cap is None or float(cap) != cert_math.CAP or getattr(lp, "logit_scale", None) is not None:
        _disable(f"logits processor softcap {cap!r} / logit_scale {getattr(lp, 'logit_scale', None)!r} not as assumed")
        return None
    if getattr(lp, "do_tensor_parallel_all_gather", False) or getattr(lp, "do_tensor_parallel_all_gather_dp_attn", False):
        _disable("logits gather across ranks")
        return None
    v_real = int(getattr(getattr(lp, "config", None), "vocab_size", 0) or 0)
    if not 0 < v_real <= pack.n:
        _disable(f"vocab_size {v_real} not in (0, {pack.n}]")
        return None
    return po, pack, m, v_real


def make_lm_head(inner):
    def sm12x_fp32_lm_head(x, weight, out):
        lp = CTX["lp"]
        cert = STATE["cert"]
        if lp is None or cert is None or STATE["disabled"] is not None or STATE["mode"] == "0":
            return inner(x, weight, out)
        got = _eligible(lp, x, weight, out, cert)
        if got is None:
            return inner(x, weight, out)
        po, pack, m, v_real = got
        capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing:
            # eager target verify (warm-up, or a batch without a graph): pico's full head; the first call per row
            # bucket goes through the kernels (full fallback, the same bits) so every launch is loaded before capture
            bucket = 8 if m <= 8 else 16
            if bucket in cert.warm:
                return inner(x, weight, out)
            cert.warm.add(bucket)
        lead = x.shape[:-1]
        x2 = po.as_rows(x, K)
        if out is None:
            out = torch.empty((m, pack.n), dtype=torch.float32, device=x.device)
        on = cert.on if capturing else cert.off
        if capturing:
            from . import dflash

            x2 = dflash.dead_rows_to_anchor(x2, cert)  # DFLASH + verify cap only; identity otherwise
        if STATE["mode"] == "check":
            sc, am = _stock_fns()
            cert.run_check(x2, out, on, v_real, cert_math.CAP, lambda xx, oo: po.gemv(xx, pack, oo), sc, am)
        else:
            cert.run(x2, out, on, v_real, cert_math.CAP)
        return out.view(*lead, pack.n)

    sm12x_fp32_lm_head.__wrapped__ = inner
    sm12x_fp32_lm_head._inkling_cert = True
    # keep pico's marker visible so that its own hook (if it runs again) does not wrap a second time
    sm12x_fp32_lm_head._pico12 = getattr(inner, "_pico12", False) or _over_pico(inner)
    sm12x_fp32_lm_head._wide = getattr(inner, "_wide", False)  # inkling_wide's hook must not wrap again
    return sm12x_fp32_lm_head


def _over_pico(fn):
    """True if fn is pico12's head function, or inkling_wide's wrapper directly over it (wide takes only
    17..256 rows; every call cert serves or falls back on has <= 16 rows and reaches pico's function)."""
    if getattr(fn, "_pico12", False):
        return True
    return bool(getattr(fn, "_wide", False)) and getattr(getattr(fn, "__wrapped__", None), "_pico12", False)


def patch_unquant(mod):
    _need(mod, ["sm12x_fp32_lm_head", "sm12x_fp32_lm_head_applies"], UNQUANT)
    if getattr(mod.sm12x_fp32_lm_head, "_inkling_cert", False):
        return
    if not _over_pico(mod.sm12x_fp32_lm_head):
        raise ops.CertFatal("inkling_cert: unquant.sm12x_fp32_lm_head is not inkling_pico12's (arm INKLING_PICO12 with "
                            "lmhead, before INKLING_CERT_HEAD)")
    mod.sm12x_fp32_lm_head = make_lm_head(mod.sm12x_fp32_lm_head)
    STATE["patched"].append("unquant.sm12x_fp32_lm_head (outside inkling_pico12)")


# ---- dspark_verify: the per-step device flag -------------------------------------------------------------------


def _sampling_ok(si):
    if si is None:
        return True
    return (bool(getattr(si, "is_all_greedy", False)) and not getattr(si, "has_watermark_candidates", False)
            and getattr(si, "sampling_mask_batch_indices", None) is None)


def batch_ok(batch, sampling_info):
    """Host-side conditions beyond the epilogue's own arming (fold_eligible: all greedy, no logit adjustments, no
    grammar): nothing after the folded accept may read the verify logits. Both the sampling info the worker passes
    and the batch's own are checked (simulated acceptance can pass None while the batch samples)."""
    if getattr(batch, "return_logprob", False) or getattr(batch, "has_grammar", False):
        return False
    return _sampling_ok(sampling_info) and _sampling_ok(getattr(batch, "sampling_info", None))


def _patch_wide_accept(mod):
    _need(mod, ["AcceptGreedy", "sampling_argmax"], VERIFY)
    original_argmax = mod.sampling_argmax
    original_accept = mod.AcceptGreedy

    def sampling_argmax(logits):
        ids = getattr(logits, "_inkling_certwide_ids", None)
        return original_argmax(logits) if ids is None else ids

    class AcceptGreedy(original_accept):
        @classmethod
        def execute(cls, *args, **kwargs):
            logits = kwargs.get("target_logits")
            ids = getattr(logits, "_inkling_certwide_ids", None)
            if ids is not None and kwargs.get("target_predict") is None:
                kwargs["target_predict"] = ids
            return original_accept.execute(*args, **kwargs)

    # Both the fused static tail and the ordinary accept (also used by vcap)
    # get their predictions from the same capture output tensor.
    mod.sampling_argmax = sampling_argmax
    mod.AcceptGreedy = AcceptGreedy


def _require_wide_replay_stream():
    wide = getattr(STATE["cert"], "wide", None)
    if wide is not None:
        wide.require_replay_stream()


def patch_graph_runner(mod):
    cls = mod.DecodeCudaGraphRunner
    orig = cls.execute
    if getattr(orig, "_inkling_certwide", False):
        return

    @functools.wraps(orig)
    def execute(self, *args, **kwargs):
        if self.capture_forward_mode.is_target_verify() and not self.model_runner.is_draft_worker:
            _require_wide_replay_stream()
        return orig(self, *args, **kwargs)

    execute._inkling_certwide = True
    cls.execute = execute
    STATE["patched"].append("DecodeCudaGraphRunner.execute")
    from . import dflash

    dflash.patch_runner(mod)


def patch_verify(mod):
    _need(mod, ["DsparkVerifyEpilogue", "TargetVerifyExecutor"], VERIFY)
    ep = mod.DsparkVerifyEpilogue
    ex = mod.TargetVerifyExecutor
    _need(ep, ["begin_static_step", "begin_step"], f"{VERIFY}.DsparkVerifyEpilogue")
    _need(ex, ["run_non_compact"], f"{VERIFY}.TargetVerifyExecutor")
    if getattr(ex.run_non_compact, "_inkling_cert", False):
        return
    if STATE["cfg"].get("wide"):
        _patch_wide_accept(mod)
    b_static, b_step, run = ep.begin_static_step, ep.begin_step, ex.run_non_compact

    @functools.wraps(b_static)
    def begin_static_step(self, bs, armed):
        _require_wide_replay_stream()
        self.__dict__["_inkling_cert_armed"] = bool(armed)
        return b_static(self, bs, armed)

    @functools.wraps(b_step)
    def begin_step(self, verify_lens, armed):
        _require_wide_replay_stream()
        self.__dict__["_inkling_cert_armed"] = False
        cert = STATE["cert"]
        if cert is not None:
            cert.set_on(0)
        return b_step(self, verify_lens, armed)

    @functools.wraps(run)
    def run_non_compact(self, *, batch, sampling_info, **kwargs):
        cert = STATE["cert"]
        if cert is not None:
            epi = getattr(self, "verify_epilogue", None)
            armed = epi is not None and bool(epi.__dict__.get("_inkling_cert_armed", False))
            # SGLANG_SIMULATE_ACC_LEN (a benchmark mode; its folded bonus may be sampled): never certify
            simulated = getattr(self, "_simulated_correct_len", None) is not None or bool(getattr(epi, "simulate_fold", False))
            _require_wide_replay_stream()
            cert.set_on(armed and not simulated and STATE["disabled"] is None and batch_ok(batch, sampling_info))
            STATE["calls"] += 1
            every = STATE["cfg"]["log_every"]
            if every and STATE["calls"] % every == 0:
                _report(cert)
        return run(self, batch=batch, sampling_info=sampling_info, **kwargs)

    for f in (begin_static_step, begin_step, run_non_compact):
        f._inkling_cert = True
    ep.begin_static_step, ep.begin_step, ex.run_non_compact = begin_static_step, begin_step, run_non_compact
    STATE["patched"].append("DsparkVerifyEpilogue.begin_static_step/begin_step, TargetVerifyExecutor.run_non_compact")


_PINNED = {}


def _report(cert):
    """Print the counters copied at the previous report (no sync), then start the next copy."""
    prev = _PINNED.get("buf")
    if prev is not None and _PINNED.get("ev") is not None and _PINNED["ev"].query():
        s = prev.tolist()
        cert_steps = max(s[1], 1)
        line = (f"steps certified {s[1]} (tiles/step {s[0] / cert_steps:.1f}), not certifiable {s[2]}, "
                f"not eligible {s[4]}")
        if STATE["mode"] == "check":
            line += f"; check: {s[7]} steps, {s[5]} rows, mismatches {s[6]}"
        _log(line)
        wprev = _PINNED.get("wbuf")
        if wprev is not None:
            _log("wide outcomes " + getattr(cert, "wide").describe_stats(wprev))
    if prev is None:
        _PINNED["buf"] = torch.zeros((8,), dtype=torch.int64, pin_memory=True)
        wide = getattr(cert, "wide", None)
        if wide is not None and wide.stats is not None:
            _PINNED["wbuf"] = torch.zeros(wide.stats.shape, dtype=torch.int64, pin_memory=True)
    _PINNED["buf"].copy_(cert.stats, non_blocking=True)
    if _PINNED.get("wbuf") is not None:
        _PINNED["wbuf"].copy_(cert.wide.stats, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    _PINNED["ev"] = ev


def patch_dflash_worker(mod):
    from . import dflash

    dflash.patch_worker(mod)


def patch_dflash_tp_worker(mod):
    from . import dflash

    dflash.patch_tp_worker(mod)
