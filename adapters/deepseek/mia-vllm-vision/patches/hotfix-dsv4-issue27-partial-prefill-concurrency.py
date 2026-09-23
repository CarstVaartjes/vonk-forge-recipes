#!/usr/bin/env python3
"""Hotfix: enforce SchedulerConfig.max_num_partial_prefills in the v1 scheduler.

Upstream vLLM 0.25.2.dev0 (ghcr.io/anemll/dspark-vllm-gx10:0.1.1) defines
``max_num_partial_prefills`` / ``max_long_partial_prefills`` on SchedulerConfig
but the v1 ``Scheduler.schedule`` admission loop never reads them — only
``max_num_seqs`` and ``token_budget`` gate new admissions. With chunked prefill
+ async scheduling + max_num_seqs>=8 and long_prefill_token_threshold=0
(default), multiple already-admitted-but-still-prefilling requests at the
front of ``self.running`` each consume up to ``max_num_batched_tokens`` per
step; decode-active requests later in the running list get
``num_new_tokens == 0`` and are skipped with ``continue`` (NOT preempted) —
producing severe, cold-only, zero-preemption decode lane starvation that
grows with prompt length. (Issue #27.)

Fix: at the top of the waiting-admission loop, break (don't admit a new
prefill request) once the number of in-flight partial prefills has reached
the cap. The cap is ``DSPARK_MAX_INFLIGHT_PREFILLS`` (1-3, default 1 via
compose) because this image rejects ``--max-num-partial-prefills``. It is
parsed once during ``Scheduler`` construction; unset, blank, nonpositive, or
malformed values fall back to ``SchedulerConfig.max_num_partial_prefills``
(stock 1), and malformed values emit one warning. The admission count is
derived from ``self.running`` in exact parity with the stock
``_inflight_prefills`` set: already-running requests count while their prompt
remains unfinished; requests admitted this step count only when their
scheduled tokens leave more prompt work. This avoids counting async-KV loads
that are not running requests and prevents stale/lost set entries from
allowing too many prefills. The set remains owned by vLLM for async-KV
reservation bookkeeping. The resolved cap is logged once at construction and
undercount drift is warned.

Idempotent only for r3. An older issue27 marker is refused rather than
silently treating its different admission predicate as current.

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""

import sys
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
MARK = "# [issue27-hotfix] enforce max_num_partial_prefills on admission"
R2_MARK = "# [issue27-r2]"
R3_MARK = "# [issue27-r3]"
if len(sys.argv) > 1 and sys.argv[1] == "--status":
    status_src = P.read_text() if P.is_file() else ""
    if MARK in status_src and R3_MARK in status_src:
        status = "APPLIED (r3)"
    elif MARK in status_src and R2_MARK in status_src:
        status = "APPLIED (r2, stale)"
    elif MARK in status_src:
        status = "APPLIED (pre-r2, stale)"
    else:
        status = "NOT APPLIED"
    print(
        "issue27 partial-prefill cap        :",
        status,
    )
    raise SystemExit(0)
src = P.read_text()
if MARK in src:
    if R3_MARK in src:
        print(f"[issue27-hotfix] already applied to {P}")
        raise SystemExit(0)
    print("[issue27-hotfix] older issue27 gate present (pre-r3); refusing to patch")
    raise SystemExit(1)

INIT_ANCHOR = (
    "        # In-flight requests still prefilling (prefill chunks + in-progress\n"
    "        # async KV loads). Their remaining-block reservation gates async loads.\n"
    "        self._inflight_prefills: set[Request] = set()\n"
)
ADMISSION_ANCHOR = (
    "                num_running = len(self.running) + self.num_waiting_for_streaming_input\n"
    "                if num_running >= self.max_num_running_reqs:\n"
    "                    break\n"
)
assert INIT_ANCHOR in src, "scheduler init anchor not found; refusing to patch"
assert ADMISSION_ANCHOR in src, "admission guard anchor not found; refusing to patch"

INIT_INJECT = INIT_ANCHOR + (
    "\n"
    "        # [issue27-hotfix] parse the admission cap once, outside schedule().\n"
    "        _pp_cap_raw = __import__('os').environ.get(\n"
    "            'DSPARK_MAX_INFLIGHT_PREFILLS', ''\n"
    "        ).strip()\n"
    "        try:\n"
    "            _pp_cap = int(_pp_cap_raw, 10) if _pp_cap_raw else 0\n"
    "        except ValueError:\n"
    "            logger.warning(\n"
    "                'Invalid DSPARK_MAX_INFLIGHT_PREFILLS; using '\n"
    "                'SchedulerConfig.max_num_partial_prefills'\n"
    "            )\n"
    "            _pp_cap = 0\n"
    "        if _pp_cap <= 0:\n"
    "            _pp_cap = self.scheduler_config.max_num_partial_prefills\n"
    "        self._dspark_max_inflight_prefills = min(_pp_cap, 3)\n"
    "        # [issue27-hotfix] boot evidence and undercount tripwire state.\n"
    "        self._dspark_inflight_diag = __import__('os').environ.get(\n"
    "            'DSPARK_ISSUE43_SCHED_DIAG', '0'\n"
    "        ) not in ('0', '', 'false', 'False')\n"
    "        self._dspark_inflight_mismatches = 0\n"
    "        logger.info(\n"
    "            '[issue27-hotfix] in-flight prefill cap=%d env=%r sched=%x',\n"
    "            self._dspark_max_inflight_prefills,\n"
    "            _pp_cap_raw,\n"
    "            id(self),\n"
    "        )\n"
)

INJECT = ADMISSION_ANCHOR + (
    "\n"
    "                # [issue27-hotfix] enforce max_num_partial_prefills on admission.\n"
    "                # Upstream defines this field but the v1 scheduler never reads\n"
    "                # it, so without this gate N already-admitted-but-still-prefilling\n"
    "                # requests at the front of self.running consume the whole\n"
    "                # max_num_batched_tokens each step; decode-active requests behind\n"
    "                # them get num_new_tokens==0 and are skipped (continue, not preempt)\n"
    "                # -> zero-preemption decode starvation (issue #27). Admission\n"
    "                # is counted directly from self.running (exact parity with the\n"
    "                # _inflight_prefills set, see [issue27-r3] below), not from the\n"
    "                # set itself, whose add/discard bookkeeping is shared with\n"
    "                # async-KV loads and is kept only for\n"
    "                # _inflight_prefill_reserved_blocks.\n"
    "                # DSPARK_MAX_INFLIGHT_PREFILLS is parsed and cached once\n"
    "                # during Scheduler construction, never in this hot loop.\n"
    "                if self._dspark_max_inflight_prefills > 0:\n"
    "                    _pp_running = 0\n"
    "                    # [issue27-r3] Match the stock set's membership and\n"
    "                    # discard timing. Earlier running entries count only\n"
    "                    # while their prompt is unfinished; decoders never\n"
    "                    # qualify, regardless of output placeholders. Requests\n"
    "                    # admitted this step use the stock set's add predicate.\n"
    "                    _pp_old = (\n"
    "                        len(self.running)\n"
    "                        - len(scheduled_new_reqs)\n"
    "                        - len(scheduled_resumed_reqs)\n"
    "                    )\n"
    "                    for _i, _r in enumerate(self.running):\n"
    "                        if _i < _pp_old:\n"
    "                            if _r.num_computed_tokens < _r.num_prompt_tokens:\n"
    "                                _pp_running += 1\n"
    "                        elif (\n"
    "                            _r.num_computed_tokens\n"
    "                            + num_scheduled_tokens.get(_r.request_id, 0)\n"
    "                            < _r.num_tokens\n"
    "                        ):\n"
    "                            _pp_running += 1\n"
    "                    _pp_tracked = len(self._inflight_prefills)\n"
    "                    if _pp_tracked < _pp_running:\n"
    "                        self._dspark_inflight_mismatches += 1\n"
    "                        if self._dspark_inflight_mismatches <= 16:\n"
    "                            logger.warning(\n"
    "                                '[issue27-hotfix] in-flight prefill undercount: '\n"
    "                                'tracked=%d running=%d cap=%d step=%d (n=%d)',\n"
    "                                _pp_tracked,\n"
    "                                _pp_running,\n"
    "                                self._dspark_max_inflight_prefills,\n"
    "                                self.current_step,\n"
    "                                self._dspark_inflight_mismatches,\n"
    "                            )\n"
    "                    if self._dspark_inflight_diag:\n"
    "                        logger.info(\n"
    "                            '[issue27-adm] step=%d tracked=%d running=%d cap=%d '\n"
    "                            'waiting=%d',\n"
    "                            self.current_step,\n"
    "                            _pp_tracked,\n"
    "                            _pp_running,\n"
    "                            self._dspark_max_inflight_prefills,\n"
    "                            len(self.waiting),\n"
    "                        )\n"
    "                    if _pp_running >= self._dspark_max_inflight_prefills:\n"
    "                        break\n"
)
src = src.replace(INIT_ANCHOR, INIT_INJECT, 1)
src = src.replace(ADMISSION_ANCHOR, INJECT, 1)
P.write_text(src)
print(f"[issue27-hotfix] patched {P}")
