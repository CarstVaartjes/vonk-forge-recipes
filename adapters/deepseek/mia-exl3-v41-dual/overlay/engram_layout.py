"""vLLM Engram hash-head layout (must match vLLM EngramLayout + TP shard).

vLLM shards complete hash-head buckets, not an even row split. pack_engram
and the file-backed lookup have to use the same [lo, hi) or packed shards
fail the row_store header check / serve the wrong rows.
"""
from __future__ import annotations

from typing import Sequence


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def head_sizes(
    *,
    vocab_size: int,
    n_heads: int,
    max_ngram_size: int,
    seen: set[int] | None = None,
) -> tuple[int, ...]:
    """Flat head bucket sizes for one Engram layer, in vLLM order."""
    if seen is None:
        seen = set()
    sizes: list[int] = []
    for _ in range(max_ngram_size - 1):
        current = vocab_size - 1
        for _ in range(n_heads):
            current = find_next_prime(current, seen)
            seen.add(current)
            sizes.append(current)
    return tuple(sizes)


def shard_range(
    sizes: Sequence[int],
    rank: int,
    tp: int,
    dp_size: int = 1,
    dp_rank: int = 0,
) -> tuple[int, int, int, int]:
    """Return (vocab_start, vocab_end, head_start, local_heads) for one rank.

    TP-major, matching vLLM `engram_head_shard_rank`.
    """
    n_hash_cols = len(sizes)
    num_shards = tp * dp_size
    part_n_hash_cols = (n_hash_cols + num_shards - 1) // num_shards
    head_start = (rank * dp_size + dp_rank) * part_n_hash_cols
    head_end = min(head_start + part_n_hash_cols, n_hash_cols)
    vocab_start = int(sum(sizes[:head_start]))
    vocab_end = int(sum(sizes[:head_end]))
    return vocab_start, vocab_end, head_start, head_end - head_start


def layer_head_sizes_from_config(text_config: dict, layer_index: int) -> tuple[int, ...]:
    """Head sizes for `engram_layer_ids[layer_index]`, walking prior layers' primes."""
    seen: set[int] = set()
    last: tuple[int, ...] = ()
    for _ in range(layer_index + 1):
        last = head_sizes(
            vocab_size=int(text_config["engram_vocab_size"]),
            n_heads=int(text_config["engram_n_heads"]),
            max_ngram_size=int(text_config["engram_max_ngram_size"]),
            seen=seen,
        )
    return last
