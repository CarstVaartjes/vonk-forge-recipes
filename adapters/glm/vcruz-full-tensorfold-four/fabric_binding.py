"""Translate the Controller's primary NCCL selector for b12x's port-1 proxy."""
import re


def primary_hca(selector):
    parts = selector.removeprefix("=").split(",")
    parsed = [re.fullmatch(r"([A-Za-z0-9_.-]+):([1-9][0-9]*)", part) for part in parts]
    if not parsed or any(part is None for part in parsed):
        raise ValueError("full GLM requires Controller device:port fabric selectors")
    primary = parsed[0]
    if primary.group(2) != "1":
        raise ValueError("the pinned b12x proxy supports primary fabric port 1 only")
    return primary.group(1)
