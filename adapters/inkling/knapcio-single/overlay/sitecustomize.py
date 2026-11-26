"""Combined loader for the Inkling-2-Mini SGLang overlays (verify cap, draft bytes, kernels, pico12, prefill).
Each overlay activates only when its environment switch is set; with none set nothing is imported.
A failing overlay stops the process (site.py would otherwise only print a warning and run the stock path)."""
import os
import sys


def _on(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "none", "false", "no")


try:
    if _on("INKLING_KV12") and _on("INKLING_KV12S"):
        raise RuntimeError("INKLING_KV12S and INKLING_KV12 are mutually exclusive")
    if _on("INKLING_VERIFY_CAP"):
        from inkling_vcap import activate
        activate()
    if _on("INKLING_DRAFT_BYTES"):
        import inkling_draftbytes
        inkling_draftbytes.install()
    if _on("INKLING_KERNELS"):
        import inkling_kernels
        inkling_kernels.install()
    if _on("INKLING_MOE2"):  # after inkling_kernels: per-token tile tables over moe_tiles' _tiles wrapper
        import inkling_moe2
        inkling_moe2.install()
    if _on("INKLING_SCODES"):  # NVFP4 expert scale codes (exact; wraps nvfp4_expert_matmul)
        import inkling_scodes
        inkling_scodes.install()
    if _on("INKLING_PICO12"):  # after inkling_kernels: its _project_sources wrapper goes outside qkvr_gemv's
        import inkling_pico12
        inkling_pico12.install()
    if _on("INKLING_PREFILL"):  # last: its _project_sources wrapper (large M only) goes outermost
        import inkling_prefill
        inkling_prefill.install()
    if _on("INKLING_PFUSE"):  # prefill glue fusion: Inkling2DecoderLayer._forward, extend only
        import inkling_pfuse
        inkling_pfuse.install()
    if _on("INKLING_PROLOGUE"):  # SWA prefill: K conv + L2 clip, V conv + value norm fused (FP32-order class)
        import inkling_prologue
        inkling_prologue.install()
    if _on("INKLING_PFUSE_HOIST"):  # prefill loop-bank hoist over pfuse's loop_mix (FP32-order class; needs PFUSE)
        import inkling_phoist
        inkling_phoist.install()
    if _on("INKLING_WIDE"):  # after pico12 (reads its packs) and prefill: outermost _project_sources for 17..256 rows
        import inkling_wide
        inkling_wide.install()
    if _on("INKLING_CERT_HEAD"):  # after pico12 and wide: outermost lm-head wrapper (<= 16 rows), reads pico's coded head
        import inkling_cert
        inkling_cert.install()
    if _on("INKLING_ATTN") and not _on("INKLING_KV12S"):  # fused relative-bias prefill attention (falls back to the stock FA4 path per call)
        import inkling_attn
        inkling_attn.install()
    if _on("INKLING_KV12S_VATTN_TRACE"):  # diagnostic: log compact verify attention arguments
        from inkling_kv12s.patch import register_vattn_trace
        register_vattn_trace()
    if _on("INKLING_VATTN"):  # target-verify attention: split-KV relative-bias kernel for all batch sizes
        import inkling_vattn
        inkling_vattn.install()
    if _on("INKLING_KV12"):  # after vattn: global-layer KV read from a packed 12-bit side pool (exact)
        import inkling_kv12
        inkling_kv12.install()
    if _on("INKLING_KV12S"):
        import inkling_kv12s
        inkling_kv12s.install()
    elif _on("INKLING_DSPARK_EAGER_COMMIT"):  # diagnostic: KV12S's eager draft commit on stock pools
        from inkling_kv12s.patch import register_eager_commit_only
        register_eager_commit_only()
    if _on("INKLING_NOEXP"):  # dead verify rows get no experts (exact for live rows; needs INKLING_VERIFY_CAP)
        import inkling_noexp
        inkling_noexp.install()
    if _on("INKLING_PADX"):  # after noexp: CUDA-graph padding rows route to the phantom expert (exact for live rows)
        import inkling_padx
        inkling_padx.install()
    if _on("INKLING_DFLASH_BYTES") or _on("INKLING_DFLASH_CTX8"):  # DFlash2 drafter byte cuts / MXFP8 context projections (draft-only; SPEC=dflash2)
        import inkling_dfbytes
        inkling_dfbytes.install()
    if _on("INKLING_DFLASH_FUSED_ACCEPT"):  # DFLASH sampled accept from the logits with sparse q (same law)
        import inkling_dfaccept
        inkling_dfaccept.install()
    if _on("INKLING_DFLASH_PCOPY"):  # DFLASH: prompt-copy forced drafts (draft proposals only; after the verify cap)
        import inkling_pcopy
        inkling_pcopy.install()
    if _on("INKLING_DFLASH_PREFIX_TAIL") or _on("INKLING_DFLASH_TAIL_SNAPSHOTS"):  # DFLASH draft ring on prefix hits: (A) re-prefill the last N input tokens, (B) restore a prompt-end draft window snapshot
        import inkling_dftail
        inkling_dftail.install()
    if _on("INKLING_PROMPT_TOUCH"):  # cache LRU order only: refresh a finished request's prompt-end checkpoint
        import inkling_ptouch
        inkling_ptouch.install()
    if _on("INKLING_RSFIX"):  # exact residual / bonus lane rule (+ SGLang PR #51681 / #51692 fixes)
        import inkling_rsfix
        inkling_rsfix.install()
except BaseException as exc:  # noqa: BLE001
    sys.stderr.write("[inkling-overlay pid=%d] FATAL at startup: %r\n" % (os.getpid(), exc))
    sys.stderr.flush()
    os._exit(78)
