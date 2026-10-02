"""The read-only plan: how many unique bytes, how long to fetch them, in which order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import policy
from .catalog import Model, Recipe

RATES_MB_S = (100, 250, 500, 1000)


def make_plan(
    recipes: Sequence[Recipe],
    models: Mapping[str, Model],
    usable_sparks: int,
    done: Mapping[str, Mapping[str, Any]],
    boost: policy.Boost,
    measured_rate: float = 0.0,
    top: int = 25,
) -> dict[str, Any]:
    testable = [r for r in recipes if r.node_count <= usable_sparks]
    skipped = [r.key for r in recipes if r.node_count > usable_sparks]
    pending = [
        r
        for r in testable
        if done.get(r.key, {}).get("status") not in ("passed", "failed")
    ]
    sizes = policy.build_sizes(testable, models)
    present = policy.present_models(models)
    plans = policy.plan_groups(pending, sizes, present, boost)
    queue = policy.order_queue(plans, {r.key: r for r in recipes})
    unique = {d for r in testable for d in r.model_digests}
    unique_bytes = sum(sizes.get(d, 0) for d in unique)
    cached_bytes = sum(sizes.get(d, 0) for d in unique if d in present)
    to_download = unique_bytes - cached_bytes
    eta = (
        {"measured": to_download / measured_rate}
        if measured_rate > 0
        else {f"{rate}_MB_s": to_download / (rate * 1e6) for rate in RATES_MB_S}
    )
    return {
        "testable_recipes": len(testable),
        "skipped_recipes": skipped,
        "pending_recipes": len(pending),
        "unique_models": len(unique),
        "unique_model_bytes": unique_bytes,
        "cached_model_bytes": cached_bytes,
        "bytes_to_download": to_download,
        "measured_rate_bytes_per_second": measured_rate or None,
        "download_eta_seconds": eta,
        "next_models": [
            {
                "recipes": list(plan.recipes),
                "new_bytes": plan.new_bytes,
                "cached": plan.cached,
                "bytes_per_recipe": plan.new_bytes // max(len(plan.recipes), 1),
            }
            for plan in plans[:top]
        ],
        "queue_head": queue[:top],
    }


def render_plan(plan: Mapping[str, Any]) -> str:
    gib = 1024**3

    def hours(seconds: float) -> str:
        return f"{seconds / 3600:,.1f} h"

    lines = [
        f"testable recipes   {plan['testable_recipes']} ({len(plan['skipped_recipes'])} need more Sparks than the fleet has)",
        f"pending            {plan['pending_recipes']}",
        (
            f"unique models      {plan['unique_models']}, {plan['unique_model_bytes'] / 1e12:,.2f} TB "
            f"({plan['unique_model_bytes'] / gib:,.0f} GiB; an upper bound if variants share files)"
        ),
        f"already on the NAS {plan['cached_model_bytes'] / 1e12:,.2f} TB",
        f"to download        {plan['bytes_to_download'] / 1e12:,.2f} TB",
        "download time      "
        + ", ".join(
            f"{name.replace('_', ' ')}: {hours(seconds)}"
            for name, seconds in plan["download_eta_seconds"].items()
        ),
        "",
        "next models (fewest new bytes per recipe first, cached first):",
    ]
    for item in plan["next_models"]:
        where = "cached" if item["cached"] else f"{item['new_bytes'] / gib:,.1f} GiB"
        lines.append(
            f"  {where:>12}  {len(item['recipes']):>3} recipes  {item['recipes'][0]}"
        )
    return "\n".join(lines) + "\n"
