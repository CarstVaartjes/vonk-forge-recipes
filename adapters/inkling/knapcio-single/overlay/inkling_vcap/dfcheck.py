"""Smoke / drain check of a DFLASH capture directory (numpy only; run inside the serving container):

  python3 -m inkling_vcap.dfcheck DIR [--run RUN] [--min-steps N] [--visits 48] [--wait-drain SECONDS]

Prints one JSON line (run, status, steps, fraction of steps that are complete = walks == 1 AND visits == --visits,
accepted range, greedy / sampled request-steps, batch sizes). Exit 0 only when the newest (or the given) run has
>= --min-steps steps, >= 99 % of them complete, no capture failure, no writer error, and the writer is drained: a
status written AFTER --after (epoch seconds; e.g. the end of the last workload) that reports idle with
written == logged. --wait-drain waits up to SECONDS for that acknowledgement. --since-step judges only the steps of
one phase (step >= the 'step_ids' count read before it; at least one new step is always required); completeness is
complete steps / every decode step id of the phase (dropped steps count as incomplete).
--require-sampled / --require-batched also need sampled request-steps / a step with bs > 1 among the COMPLETE
steps."""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def _status(directory, run):
    paths = sorted(glob.glob(os.path.join(directory, f"status-{run or '*'}.json")), key=os.path.getmtime)
    if not paths:
        return None
    with open(paths[-1]) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--run")
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--visits", type=int, default=48)
    ap.add_argument("--wait-drain", type=float, default=0.0)
    ap.add_argument("--after", type=float, default=0.0)
    ap.add_argument("--since-step", type=int, default=0, help="judge only steps with step >= this (a phase's new steps)")
    ap.add_argument("--require-sampled", action="store_true")
    ap.add_argument("--require-batched", action="store_true")
    a = ap.parse_args(argv)
    def drained(st):
        return bool(st and st.get("idle") and st.get("written") == st.get("logged")
                    and float(st.get("time", 0)) > a.after)

    st = _status(a.dir, a.run)
    deadline = time.time() + a.wait_drain
    while a.wait_drain > 0 and time.time() < deadline and not drained(st):
        time.sleep(1.0)
        st = _status(a.dir, a.run)
    run = a.run or (st or {}).get("run")
    files = sorted(glob.glob(os.path.join(a.dir, f"dflash1-{run}-*.npz"))) if run else []
    steps = walks_ok = visits_ok = complete = 0
    acc_min, acc_max, greedy, sampled = 99, -1, 0, 0
    bs_seen = {}
    for f in files:
        z = np.load(f)
        sel = z["step"] >= a.since_step                        # per step
        req_sel = np.repeat(sel, z["bs"])                       # per request-step
        comp = (z["walks"] == 1) & (z["visits"] == a.visits)
        req_comp = np.repeat(comp, z["bs"]) & req_sel
        steps += int(sel.sum())
        walks_ok += int((z["walks"][sel] == 1).sum())
        visits_ok += int((z["visits"][sel] == a.visits).sum())
        complete += int((comp & sel).sum())
        acc = z["accepted"][req_sel]
        if len(acc):
            acc_min, acc_max = min(acc_min, int(acc.min())), max(acc_max, int(acc.max()))
        # coverage of the request kinds / batch sizes counts complete steps only
        greedy += int((z["greedy"] & req_comp).sum())
        sampled += int((~z["greedy"] & req_comp).sum())
        for b in z["bs"][comp & sel]:
            bs_seen[int(b)] = bs_seen.get(int(b), 0) + 1
    frac_w = walks_ok / steps if steps else 0.0
    frac_v = visits_ok / steps if steps else 0.0
    # completeness over every decode step id of the phase (dropped steps consume ids but write no row)
    ids_end = int((st or {}).get("step_ids", 0)) if st else 0
    expected = max(steps, ids_end - a.since_step)
    frac_c = complete / expected if expected else 0.0
    is_drained = drained(st)
    ok = (steps >= max(1, a.min_steps) and frac_c >= 0.99 and 0 <= acc_min and acc_max <= 7 and is_drained
          and st is not None and not st.get("failed") and not st.get("writer_error")
          and (sampled > 0 or not a.require_sampled) and (max(bs_seen, default=0) > 1 or not a.require_batched))
    print(json.dumps(dict(ok=ok, run=run, files=len(files), since_step=a.since_step, step_ids=ids_end,
                          expected_steps=expected, steps=steps, complete=round(frac_c, 4),
                          walks_ok=round(frac_w, 4), visits_ok=round(frac_v, 4), accepted=[acc_min, acc_max],
                          greedy_req_steps=greedy, sampled_req_steps=sampled, bs=bs_seen, drained=is_drained,
                          status=st)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
