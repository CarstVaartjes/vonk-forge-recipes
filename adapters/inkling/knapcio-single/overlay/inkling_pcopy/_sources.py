"""Source pinning for inkling_pcopy: every stock function it wraps or relies on is hashed (sha256 of its exact source
lines) and compared with the text of the pinned image (lmsysorg/sglang@sha256:efac853b...). Any difference refuses the
overlay at start-up. Pure Python (ast + hashlib): runs before sglang is imported and in the CPU tests."""

import ast
import hashlib
import importlib.util
import os

DFLASH_WORKER = "srt/speculative/dflash_worker_v2.py"
KV_CONFIG = "srt/mem_cache/kv_cache_configurator.py"

EXPECTED = {
    DFLASH_WORKER: {
        # wrapped: the history is allocated after the stock ring
        "DFlashWorkerV2._init_draft_kv_ring": "5c5810b43eb01f3c67038f5b9e3cba2b07c8df01f32510b6c2d63ecd08d96b05",
        # wrapped (extend: seed the history) and relied on: draft_tokens is _draft_block_tokens_buf[:bs], the verify
        # input is draft_tokens.reshape(-1), _plan_verify runs after both and before the target verify, the grammar
        # tree is built before it, the accept reads draft_tokens and _selector_sample
        "DFlashWorkerV2.forward_batch_generation": "0f7249cbf179270fc14d96e6e0daa13ed13123171eb8dc3865985bdabba1bdb6",
        # wrapped: the step hook (outermost, before the verify cap's own wrapper)
        "DFlashWorkerV2._plan_verify": "cc266f15a2c3b14a74e7cf117ec22b8ad37fc600fa88b6b2156ed37089e1fc69",
        # wrapped: forced rows grade no drafts
        "DFlashWorkerV2._observe_acceptance": "99dc2e1b67c829a917ecc9926f37030af6f7b31a0d8dcadc02397447499c9921",
    },
    KV_CONFIG: {
        # wrapped: the history is charged to the KV budget with the ring
        "KVCacheConfigurator._dflash_draft_kv_ring_bytes": "40c31b2ba28694b9c39b0f27ae14595294e8042998009656b21f88350a1ed471",
        # relied on: subtracts that charge from the budget the KV pool is sized from
        "KVCacheConfigurator._profile_available_bytes": "0c6db2e3a004f038aec54f9b5feeea41a421430bde9873dcbfed6d28446c2477",
    },
}


def _find(tree, qualname):
    node = tree
    for part in qualname.split("."):
        hits = [n for n in getattr(node, "body", [])
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part]
        if len(hits) != 1:
            return None
        node = hits[0]
    return node


def digest(text, qualname):
    node = _find(ast.parse(text), qualname)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    seg = "".join(text.splitlines(keepends=True)[start - 1: node.end_lineno])
    return hashlib.sha256(seg.encode("utf-8")).hexdigest()


def mismatches(text, rel):
    bad = []
    for name, want in EXPECTED[rel].items():
        got = digest(text, name)
        if got is None:
            bad.append((name, "missing or defined twice"))
        elif got != want:
            bad.append((name, f"sha256 {got[:12]} != pinned {want[:12]}"))
    return bad


def sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_pcopy: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check(root=None, off_hint=""):
    base = root or sglang_root()
    for rel in EXPECTED:
        path = os.path.join(base, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_pcopy: patch target {path} is missing{off_hint}") from exc
        bad = mismatches(text, rel)
        if bad:
            raise RuntimeError(f"inkling_pcopy: {path} differs from the pinned image: {bad}{off_hint}")
