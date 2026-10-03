"""Live status and final report, from the state file alone.

``build_status`` is the live view (lanes, queue, throughput, ETA, failure
clusters) written as JSON plus a short markdown page on every cycle.
``build_report`` is the end-of-run summary: pass or fail per recipe with phase,
reason, timings and tokens per second. ``export_evidence`` turns the results
log into per-recipe evidence files for the repository.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import policy
from .state import write_atomic

if TYPE_CHECKING:
    from .run import Sweep

DEFAULT_TEST_SECONDS = 1200.0


def _gib(value: float) -> str:
    return f"{value / 1024**3:,.1f} GiB"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def counts(recipes: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = defaultdict(int)
    for entry in recipes.values():
        tally[str(entry.get("status", "pending"))] += 1
    return dict(tally)


def average_test_seconds(recipes: Mapping[str, Mapping[str, Any]]) -> float:
    totals = [
        float(e["timings"]["total_s"])
        for e in recipes.values()
        if isinstance(e.get("timings"), dict) and e["timings"].get("total_s")
    ]
    return statistics.mean(totals) if totals else DEFAULT_TEST_SECONDS


def build_status(sweep: Sweep) -> dict[str, Any]:
    data = sweep.state.data
    now = sweep.clock.now()
    lanes: dict[str, list[dict[str, Any]]] = {s.name: [] for s in sweep.sparks()}
    for key, slot in data["slots"].items():
        for name in slot.get("spark_names", []):
            lanes.setdefault(name, []).append(
                {
                    "recipe": key,
                    "phase": slot["phase"],
                    "elapsed_s": round(now - slot["started_at"]),
                    "limit_s": round(slot["deadline"] - slot["started_at"]),
                }
            )
    present = policy.present_models(sweep.models)
    remaining_models = {
        d
        for key in sweep.queue
        if key in sweep.recipes
        for d in sweep.recipes[key].model_digests
        if d not in present
    }
    remaining_bytes = sum(sweep.sizes.get(d, 0) for d in remaining_models)
    pending = counts(data["recipes"]).get("pending", 0)
    lane_count = max(1, len(sweep.sparks()))
    eta_tests = pending * average_test_seconds(data["recipes"]) / lane_count
    eta_download = sweep.rate.eta_seconds(remaining_bytes)
    ready = {k for k in sweep.queue if sweep.ready(sweep.recipes[k])}
    return {
        "updated_at": now,
        "paused": sweep.owner_status.reason or None,
        "mode": data["mode"],
        "counts": counts(data["recipes"]),
        "lanes": lanes,
        "queue": [{"recipe": k, "cached": k in ready} for k in sweep.queue[:15]],
        "queue_length": len(sweep.queue),
        "downloads": {
            "in_flight": [
                {
                    "recipe": k,
                    "kind": r.get("kind"),
                    "done": r.get("bytes_done"),
                    "total": r.get("bytes_total"),
                    "bps": r.get("bps"),
                }
                for k, r in data["downloads"].items()
                if r.get("state") in ("queued", "running", "partial", "accepted")
            ],
            "rate_bps": round(sweep.rate.rate),
            "remaining_unique_bytes": remaining_bytes,
            "pinned": len(data["pins"]),
            "pin_error": sweep.prefetcher.pin_error,
            "nas_pressure": bool(sweep.last_prefetch.get("pressure")),
        },
        "eta": {
            "downloads_s": eta_download,
            "tests_s": eta_tests,
            "overall_s": max(eta_tests, eta_download or 0.0),
        },
        "failure_clusters": [
            {
                "id": c.id,
                "signature": c.signature,
                "count": len(c.recipes),
                "recipes": list(c.recipes[:8]),
            }
            for c in policy.cluster(data["recipes"])
        ],
        "infrastructure": {
            source: {**item, "retry_in_s": max(0, round(item["until"] - now))}
            for source, item in data["infra"].items()
        },
        "client": data.get("client"),
        "events": [e["message"] for e in data["events"][-10:]],
    }


def render_status(status: Mapping[str, Any]) -> str:
    c = status["counts"]
    lines = [
        "# Hardware sweep status",
        "",
        f"passed {c.get('passed', 0)} - failed {c.get('failed', 0)} - pending {c.get('pending', 0)} - skipped {c.get('skipped', 0)}",
        f"mode {status['mode']}"
        + (f" - PAUSED: {status['paused']}" if status["paused"] else ""),
        "",
        "## Lanes",
    ]
    for spark, items in status["lanes"].items():
        text = ", ".join(
            f"{i['recipe']} [{i['phase']} {_duration(i['elapsed_s'])}/{_duration(i['limit_s'])}]"
            for i in items
        )
        lines.append(f"- {spark}: {text or 'idle'}")
    if status.get("infrastructure"):
        lines += ["", "## Infrastructure problems (not recipe failures; retrying)"]
        lines += [
            f"- {source} x{item['count']}, next try in {_duration(item['retry_in_s'])}: {item['message']}"
            for source, item in status["infrastructure"].items()
        ]
    d = status["downloads"]
    lines += [
        "",
        "## Downloads",
        f"- rate {d['rate_bps'] / 1e6:,.0f} MB/s - {len(d['in_flight'])} in flight - {d['pinned']} recipes pinned"
        + (" - NAS pressure, new model downloads paused" if d["nas_pressure"] else "")
        + (f" - pin profile problem: {d['pin_error']}" if d["pin_error"] else ""),
        f"- remaining unique model bytes {_gib(d['remaining_unique_bytes'])}",
        "",
        "## ETA",
        f"- downloads {_duration(status['eta']['downloads_s'])}, tests {_duration(status['eta']['tests_s'])}, overall {_duration(status['eta']['overall_s'])}",
        "",
        f"## Queue (next {len(status['queue'])} of {status['queue_length']})",
    ]
    lines += [
        f"- {q['recipe']}{'' if q['cached'] else ' (downloading)'}"
        for q in status["queue"]
    ]
    if status["failure_clusters"]:
        lines += ["", "## Failure clusters"]
        lines += [
            f"- {k['count']} x {k['signature']}" for k in status["failure_clusters"]
        ]
    if status["events"]:
        lines += ["", "## Recent"] + [f"- {m}" for m in status["events"]]
    return "\n".join(lines) + "\n"


def build_report(recipes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for key, entry in sorted(recipes.items()):
        timings = entry.get("timings") or {}
        perf = entry.get("perf") or {}
        rows.append(
            {
                "recipe": key,
                "status": entry.get("status", "pending"),
                "phase": entry.get("phase"),
                "failure_class": entry.get("failure_class"),
                "reason": entry.get("error") or entry.get("reason"),
                "cluster": entry.get("cluster"),
                "smoke": entry.get("smoke"),
                "download_s": timings.get("download_s"),
                "load_s": timings.get("load_s"),
                "smoke_s": timings.get("smoke_s"),
                "ttft_ms": perf.get("ttft_ms"),
                "tokens_per_second": perf.get("tokens_per_second"),
                "content_sha256": entry.get("content_sha256"),
                "evidence": entry.get("evidence"),
            }
        )
    return {
        "summary": counts(recipes),
        "recipes": rows,
        "failure_clusters": [
            {"id": c.id, "signature": c.signature, "recipes": list(c.recipes)}
            for c in policy.cluster(recipes)
        ],
    }


def render_report(report: Mapping[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Hardware sweep report",
        "",
        f"passed {s.get('passed', 0)}, failed {s.get('failed', 0)}, not yet tested {s.get('pending', 0)}, skipped {s.get('skipped', 0)}",
        "",
        "| recipe | result | phase / class | reason | download | load | TTFT ms | tok/s |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def seconds(value: Any) -> str:
        return _duration(value) if value is not None else ""

    for row in report["recipes"]:
        if row["status"] == "skipped":
            continue
        where = (
            f"{row['phase']} / {row['failure_class']}"
            if row["phase"]
            else (row["smoke"] or "")
        )
        reason = str(row["reason"] or "").replace("|", "/").replace("\n", " ")[:140]
        lines.append(
            f"| {row['recipe']} | {row['status']} | {where} | {reason} | {seconds(row['download_s'])} | "
            f"{seconds(row['load_s'])} | {row['ttft_ms'] or ''} | {row['tokens_per_second'] or ''} |"
        )
    if report["failure_clusters"]:
        lines += ["", "## Failure clusters (one root cause each)"]
        for cluster in report["failure_clusters"]:
            lines.append(
                f"- {len(cluster['recipes'])} x `{cluster['signature']}`: {', '.join(cluster['recipes'][:6])}"
            )
    skipped = [r for r in report["recipes"] if r["status"] == "skipped"]
    if skipped:
        lines += ["", f"## Not testable on this fleet ({len(skipped)})"]
        lines += [f"- {r['recipe']}: {r['reason']}" for r in skipped]
    return "\n".join(lines) + "\n"


def write_status(sweep: Sweep, final: bool = False) -> None:
    directory = sweep.cfg.state_dir
    status = build_status(sweep)
    write_atomic(
        directory / "status.json",
        json.dumps(status, indent=1, sort_keys=True, default=str),
    )
    write_atomic(directory / "status.md", render_status(status))
    if final:
        report = build_report(sweep.state.recipes)
        write_atomic(
            directory / "report.json", json.dumps(report, indent=1, sort_keys=True)
        )
        write_atomic(directory / "report.md", render_report(report))


# -- repository evidence ------------------------------------------------------


def export_evidence(
    entries: Iterable[Mapping[str, Any]], out_dir: Path, known_recipes: Iterable[str]
) -> list[Path]:
    """One ``<slug>.jsonl`` per recipe: the latest recorded outcome for each recipe document digest."""
    known = set(known_recipes)
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        recipe, digest = entry.get("recipe"), entry.get("content_sha256")
        if (
            entry.get("step") == "smoke"
            and isinstance(recipe, str)
            and isinstance(digest, str)
            and recipe in known
        ):
            latest[(recipe, digest)] = entry
    by_recipe: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (recipe, _), entry in sorted(
        latest.items(), key=lambda item: str(item[1].get("recorded_at"))
    ):
        by_recipe[recipe].append(entry)
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for recipe, items in sorted(by_recipe.items()):
        path = out_dir / f"{recipe.partition('/')[2]}.jsonl"
        write_atomic(
            path,
            "".join(json.dumps(i, sort_keys=True, default=str) + "\n" for i in items),
        )
        written.append(path)
    return written
