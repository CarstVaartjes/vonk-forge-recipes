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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import policy
from .lifecycle import ACTIVE
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


def quality_notes(recipes: Mapping[str, Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Recipes that ran but whose model missed an expected answer: information, not failures."""
    return {
        key: list(entry["quality"])
        for key, entry in sorted(recipes.items())
        if entry.get("status") == "passed" and entry.get("quality")
    }


def average_test_seconds(recipes: Mapping[str, Mapping[str, Any]]) -> float:
    totals = [
        float(e["timings"]["total_s"])
        for e in recipes.values()
        if isinstance(e.get("timings"), dict) and e["timings"].get("total_s")
    ]
    return statistics.mean(totals) if totals else DEFAULT_TEST_SECONDS


def sample_evidence(
    entries: Mapping[str, Mapping[str, Any]], recipes: Sequence[str]
) -> str:
    """One line of what the first failed recipe of a cluster saw: case, status, operation, response."""
    for key in recipes:
        proof = policy.evidence_of(entries.get(key) or {})
        parts = [
            f"{name}={proof[name]}"
            for name in ("case", "http_status", "operation_id", "reason")
            if proof.get(name) not in (None, "")
        ]
        if proof.get("body"):
            parts.append("body=" + " ".join(str(proof["body"]).split())[:200])
        if parts:
            return f"{key}: " + ", ".join(parts)
    return ""


def failures_by_release(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Failed recipes grouped by the Controller release they failed under, newest release first."""
    versions = {h["sha"]: h.get("version") for h in data.get("release_history", [])}
    current = data.get("release", {}).get("sha")
    groups: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for key, entry in data["recipes"].items():
        if entry.get("status") == "failed":
            sha = (
                entry.get("release")
                if entry.get("release_authority") == "deployed-controller"
                else None
            )
            groups[str(sha or "unknown")][key] = entry
    order = [h["sha"] for h in reversed(data.get("release_history", []))]
    ranked = sorted(
        groups, key=lambda sha: order.index(sha) if sha in order else len(order)
    )
    return [
        {
            "release": sha,
            "version": versions.get(sha),
            "current": sha == current,
            "failed": len(groups[sha]),
            "clusters": [
                {
                    "id": c.id,
                    "signature": c.signature,
                    "count": len(c.recipes),
                    "recipes": list(c.recipes[:6]),
                    "evidence": sample_evidence(data["recipes"], c.recipes),
                }
                for c in policy.cluster(groups[sha])
            ],
        }
        for sha in ranked
    ]


def build_status(sweep: Sweep) -> dict[str, Any]:
    data = sweep.state.data
    now = sweep.clock.now()
    lanes: dict[str, list[dict[str, Any]]] = {s.name: [] for s in sweep.sparks()}
    for key, slot in data["slots"].items():
        for name in slot.get("spark_names", []):
            lanes.setdefault(name, []).append(
                {
                    "recipe": key,
                    "phase": "loading (copying)"
                    if slot.get("copying")
                    else slot["phase"],
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
        "quality_notes": {
            key: len(notes) for key, notes in quality_notes(data["recipes"]).items()
        },
        "lanes": lanes,
        "alerts": list(data.get("alerts") or []),
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
                if r.get("state") in ACTIVE
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
                "evidence": sample_evidence(data["recipes"], c.recipes),
            }
            for c in policy.cluster(data["recipes"])
        ],
        "release": data["release"].get("sha"),
        "failures_by_release": failures_by_release(data),
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
        f"passed {c.get('passed', 0)}"
        + (
            f" ({len(status['quality_notes'])} with quality notes)"
            if status.get("quality_notes")
            else ""
        )
        + f" - failed {c.get('failed', 0)} - deferred {c.get('deferred', 0)} - pending {c.get('pending', 0)} - skipped {c.get('skipped', 0)}",
        f"mode {status['mode']}"
        + (f" - PAUSED: {status['paused']}" if status["paused"] else ""),
        "",
    ]
    if status.get("alerts"):
        lines += ["## ALERTS", *(f"- {a}" for a in status["alerts"]), ""]
    lines.append("## Lanes")
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
    if status.get("failures_by_release"):
        lines += ["", "## Failures by Controller release"]
        for group in status["failures_by_release"]:
            tag = " (current)" if group["current"] else ""
            version = f" {group['version']}" if group["version"] else ""
            lines.append(
                f"- release {str(group['release'])[:12]}{version}{tag}: {group['failed']} failed"
            )
            for c in group["clusters"]:
                lines.append(f"  - {c['count']} x {c['signature']}")
                if c.get("evidence"):
                    lines.append(f"    - evidence: {c['evidence']}")
    if status["failure_clusters"]:
        lines += ["", "## Failure clusters"]
        for k in status["failure_clusters"]:
            lines.append(f"- {k['count']} x {k['signature']}")
            if k.get("evidence"):
                lines.append(f"  - evidence: {k['evidence']}")
    if status.get("quality_notes"):
        lines += ["", "## Quality notes (ran; the answer differed, not failures)"]
        lines += [
            f"- {key}: {count} quality note{'s' if count != 1 else ''}"
            for key, count in status["quality_notes"].items()
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
                "evidence": policy.evidence_of(entry) or None,
                "evidence_bundle": entry.get("evidence_bundle"),
                "quality": quality_notes({key: entry}).get(key, []),
            }
        )
    return {
        "summary": counts(recipes),
        "recipes": rows,
        "failure_clusters": [
            {
                "id": c.id,
                "signature": c.signature,
                "recipes": list(c.recipes),
                "evidence": sample_evidence(recipes, c.recipes),
            }
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
        status = (
            f"ran; {len(row['quality'])} quality note"
            f"{'s' if len(row['quality']) != 1 else ''}"
            if row["status"] == "passed" and row.get("quality")
            else row["status"]
        )
        lines.append(
            f"| {row['recipe']} | {status} | {where} | {reason} | {seconds(row['download_s'])} | "
            f"{seconds(row['load_s'])} | {row['ttft_ms'] or ''} | {row['tokens_per_second'] or ''} |"
        )
    if report["failure_clusters"]:
        lines += ["", "## Failure clusters (one root cause each)"]
        for cluster in report["failure_clusters"]:
            lines.append(
                f"- {len(cluster['recipes'])} x `{cluster['signature']}`: {', '.join(cluster['recipes'][:6])}"
            )
    noted = [r for r in report["recipes"] if r.get("quality")]
    if noted:
        lines += ["", "## Quality notes (the recipe ran; information, not failures)"]
        for row in noted:
            for note in row["quality"]:
                lines.append(
                    f"- {row['recipe']} {note['case']}: expected {note['expected']}, "
                    f"got {note['got']}"
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
