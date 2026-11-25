"""Source pinning for inkling_dftail: every stock function the tail (A) or the draft-tail snapshots (B) wrap or rely on
is hashed (sha256 of its exact source lines, decorators included) and compared with the text of the pinned image
(lmsysorg/sglang@sha256:efac853b...). Any difference refuses the overlay at start-up.

Pure Python (ast + hashlib): runs before sglang is imported, and on a machine without it (the CPU tests)."""

import ast
import hashlib
import importlib.util
import os

SCHEDULE_BATCH = "srt/managers/schedule_batch.py"
DRAFT_RING = "srt/speculative/dflash_draft_ring.py"
DFLASH_WORKER = "srt/speculative/dflash_worker_v2.py"
KV_CONFIG = "srt/mem_cache/kv_cache_configurator.py"
WEIGHT_UPDATER = "srt/managers/scheduler_components/weight_updater.py"

# file -> qualified name -> sha256 of the stock source segment in the pinned image.
EXPECTED = {
    SCHEDULE_BATCH: {
        # (A) wrapped: the bound it returns is the one the tail tightens
        "Req._compute_max_prefix_len": "28ac28ced09b6cb7bbf0f69febbcba4103cd96b90c362a9ae354093a58686a74",
        # (A) the radix lookup: passes the bound as max_prefix_len (and the streaming-session logprob rule before it)
        "Req.init_next_round_input": "fb0049c4f3441b785757230a32cd3dcca7eaef1b67bee49fbbc7109d223649af",
    },
    DRAFT_RING: {
        # (B) relied on: min_pos / _floor_generation / window_left / span / paged layout
        "DFlashDraftKVRing.__init__": "e349383f60b4b8f176c847e1fa7bc8223a421c6b1fe49d1972c926a583268ef0",
        # (B) relied on: a position's ring row (relative to min_pos when paged)
        "DFlashDraftKVRing.locs": "e5d328a819a22091b9f3ad959b6eb194d3ef02abbe5d923568c4ba51fae0e467",
        # (B) wrapped: the floor of a fresh slot (lowered to the restored range) and the restore point
        "DFlashDraftKVRing.record_floors": "42c97ddee5be95f746b4bdae6033b7482f07c46044dd7c0f8dac7dee420f440b",
    },
    DFLASH_WORKER: {
        # (B) wrapped: allocates the snapshot store next to the ring
        "DFlashWorkerV2._init_draft_kv_ring": "5c5810b43eb01f3c67038f5b9e3cba2b07c8df01f32510b6c2d63ecd08d96b05",
        # (B) wrapped: captures after the forward; relied on: record_floors -> extend_locs -> append order in prefill
        "DFlashWorkerV2.forward_batch_generation": "0f7249cbf179270fc14d96e6e0daa13ed13123171eb8dc3865985bdabba1bdb6",
        # (B) wrapped: flush_cache drops every snapshot
        "DFlashWorkerV2.clear_cache_pool": "9e451b5b28a8323737d706590318ea22a2d5390dd0d51bc82461088f5886ac61",
    },
    KV_CONFIG: {
        # (B) wrapped: the ring's charge to the target KV budget; the snapshot pool is charged with it
        "KVCacheConfigurator._dflash_draft_kv_ring_bytes": "40c31b2ba28694b9c39b0f27ae14595294e8042998009656b21f88350a1ed471",
        # (B) relied on: subtracts that charge from the budget the KV pool is sized from
        "KVCacheConfigurator._profile_available_bytes": "0c6db2e3a004f038aec54f9b5feeea41a421430bde9873dcbfed6d28446c2477",
    },
    WEIGHT_UPDATER: {
        # (B) wrapped: every weight load selects its runners here; snapshots are dropped
        "SchedulerWeightUpdaterManager._select_runners": "778803c5ed26e88f4390274145ff68707c215b97bf2a77e829544eb90abc111b",
    },
}
PART_A = (SCHEDULE_BATCH,)
PART_B = (DRAFT_RING, DFLASH_WORKER, KV_CONFIG, WEIGHT_UPDATER)


def _find(tree, qualname):
    node = tree
    for part in qualname.split("."):
        hits = [
            n for n in getattr(node, "body", [])
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part
        ]
        if len(hits) != 1:
            return None
        node = hits[0]
    return node


def segment(text, qualname):
    """Exact source lines of a top-level function / class method, decorators included, or None when the name is
    missing or defined more than once."""
    node = _find(ast.parse(text), qualname)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines(keepends=True)
    return "".join(lines[start - 1: node.end_lineno])


def digest(text, qualname):
    seg = segment(text, qualname)
    return None if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()


def mismatches(text, rel=SCHEDULE_BATCH):
    """[(qualname, why)] for every pinned segment of file ``rel`` that differs in ``text``."""
    bad = []
    for name, want in EXPECTED[rel].items():
        got = digest(text, name)
        if got is None:
            bad.append((name, "missing or defined twice"))
        elif got != want:
            bad.append((name, f"sha256 {got[:12]} != pinned {want[:12]}"))
    return bad


def sglang_root():
    spec = importlib.util.find_spec("sglang")  # a top-level find_spec imports nothing
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_dftail: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check(root=None, off_hint="", files=PART_A):
    """Raise RuntimeError unless every installed file in ``files`` matches the pinned text."""
    base = root or sglang_root()
    for rel in files:
        path = os.path.join(base, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_dftail: patch target {path} is missing{off_hint}") from exc
        bad = mismatches(text, rel)
        if bad:
            raise RuntimeError(f"inkling_dftail: {path} differs from the pinned image: {bad}{off_hint}")
