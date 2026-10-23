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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vonk_forge_contracts import (
    QualificationDerivationInput,
    QualificationStatus,
    QualificationStatusView,
    RecipeQualificationEvidence,
    derive_status,
)
from vonk_forge_contracts.sweep import SweepSlotPhase, SweepWait

from . import policy
from .evidence_contract import read_result
from .identity import recorded_identity
from .lifecycle import ACTIVE
from .qualification import current_identity, from_history, merge_evidence, read_evidence
from .removal_contract import RecipeVerdict, RemovalWait
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
    notes = {}
    for key, entry in sorted(recipes.items()):
        result = read_result(entry, key)
        if result.core.status is RecipeVerdict.PASSED and result.quality:
            notes[key] = [note.model_dump(exclude_none=True) for note in result.quality]
    return notes


def average_test_seconds(recipes: Mapping[str, Mapping[str, Any]]) -> float:
    totals = [
        total
        for key, entry in recipes.items()
        if (total := read_result(entry, key).timings.total_s) is not None
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
    # Every enrolled Spark keeps its lane; an offline one is shown waiting, never dropped.
    lanes: dict[str, list[dict[str, Any]]] = {s.name: [] for s in sweep.enrolled()}
    offline = {
        s.name: {"reason": s.offline_reason, "last_seen_at": s.last_seen_at or None}
        for s in sweep.enrolled()
        if not s.online
    }
    for key, slot in data["slots"].items():
        smoking = slot["phase"] == "smoking"
        # Show the enforced phase budget, including elapsed time during dependency waits.
        budget = slot.get("smoke_budget_s") if smoking else None
        since = (
            slot.get("smoke_started_at", slot.get("ready_at", slot["started_at"]))
            if smoking
            else slot["started_at"]
        )
        limit = budget if budget is not None else slot["deadline"] - slot["started_at"]
        # The current phase's own wait; the load's last wait ended with the load.
        watch = (
            (slot.get("smoke_watch") or {})
            if smoking
            else max(
                (slot.get("watch") or {}, slot.get("ready_watch") or {}),
                key=lambda item: float(
                    item.get("observed_at", item.get("progressed_at", 0))
                ),
            )
        )
        if watch.get("waiting_on"):
            limit = min(limit, float(watch.get("wait_deadline", since + limit)) - since)
        if not smoking and slot.get("copying") and not watch.get("waiting_on"):
            # Copying has an inactivity window, not the engine startup budget.
            since = watch.get("progressed_at", slot["started_at"])
            limit = sweep.cfg.copy_stall_seconds
        for name in slot.get("spark_names", []):
            lanes.setdefault(name, []).append(
                {
                    "recipe": key,
                    "phase": "loading (copying)"
                    if slot.get("copying")
                    else slot["phase"],
                    "elapsed_s": round(now - since),
                    "limit_s": round(limit),
                    "waiting_on": watch.get("waiting_on")
                    if watch
                    else slot.get("waiting_on"),
                    "wait": watch.get("wait"),
                    "silent_s": round(
                        max(0.0, now - float(watch.get("progressed_at", now)))
                    ),
                    "over_budget_s": max(
                        0,
                        round(
                            now
                            - min(
                                since + limit,
                                float(watch.get("wait_deadline", since + limit)),
                            )
                        ),
                    ),
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
        "finalization": data.get("finalization"),
        "paused": sweep.pause_reason or None,
        "controller_wait_issue": sweep.controller_wait_issue(),
        "controller_waits": {
            key: hold.model_dump(mode="json")
            for key, hold in sweep.controller_wait_holds().items()
        },
        "mode": data["mode"],
        "counts": counts(data["recipes"]),
        "catalog_diagnostics": data.get("catalog_diagnostics", []),
        "model_removals": data.get("model_removals", {}),
        "model_removal_wait": data.get("model_removal_wait"),
        "quality_notes": {
            key: len(notes) for key, notes in quality_notes(data["recipes"]).items()
        },
        "lanes": lanes,
        "offline_lanes": offline,
        "waiting_for_lanes": sum(
            1
            for k in sweep.queue
            if k in sweep.recipes and sweep.lanes_short(sweep.recipes[k])
        ),
        "alerts": list(data.get("alerts") or [])
        + ([sweep.stall_reason] if sweep.stall_reason else [])
        + [
            f"ESCALATION: {source} has not recovered since {item['since']}; "
            f"{item['dependency']}"
            for source, item in data["infra"].items()
            if item.get("escalate_at") is not None and now >= item["escalate_at"]
        ],
        "queue": [
            {
                "recipe": k,
                "cached": k in ready,
                "lanes_short": sweep.lanes_short(sweep.recipes[k])
                if k in sweep.recipes
                else 0,
            }
            for k in sweep.queue[:15]
        ],
        "queue_length": len(sweep.queue),
        "evidence_publication": sweep.evidence_publisher.message,
        "qualification": {
            key: view.model_dump(mode="json", exclude_none=True)
            for key, recipe in sweep.recipes.items()
            if (view := sweep.qualification_view(recipe)) is not None
        },
        "verdict_identities": {
            key: entry.get("verdict_identity") for key, entry in data["recipes"].items()
        },
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
            source: {
                **item,
                "retry_in_s": max(0, round(item["until"] - now)),
                **(
                    {"escalate_in_s": max(0, round(item["escalate_at"] - now))}
                    if item.get("escalate_at") is not None
                    else {}
                ),
            }
            for source, item in data["infra"].items()
        },
        "client": data.get("client"),
        "events": [e["message"] for e in data["events"][-10:]],
    }


def _qualification_summary(views: Iterable[Mapping[str, Any]]) -> str:
    tally = Counter(
        QualificationStatusView.model_validate(view).status for view in views
    )
    return ", ".join(
        f"{status.value} {tally[status]}"
        for status in QualificationStatus
        if status in tally
    )


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
    if status.get("qualification"):
        lines[2] = _qualification_summary(status["qualification"].values())
    if status.get("alerts"):
        lines += ["## ALERTS", *(f"- {a}" for a in status["alerts"]), ""]
    if status.get("model_removal_wait"):
        wait = RemovalWait.model_validate(status["model_removal_wait"])
        age = max(0, status["updated_at"] - wait.since)
        retry = max(0, wait.next_check - status["updated_at"])
        lines += [
            (
                f"Fleet model removal: {wait.reason}: {wait.detail}"
                f"; age {_duration(age)}; next observation in {_duration(retry)}"
            ),
            "",
        ]
    if status.get("model_removals"):
        lines += ["## Fleet model removal"]
        for digest, item in status["model_removals"].items():
            lines.append(
                f"- {digest}: {item['outcome']}: {item['message']}"
                f"; next check {item['next_check']}; request {item['request_key'] or 'none'}"
            )
        lines.append("")
    lines.append("## Lanes")
    for spark, items in status["lanes"].items():
        text = ", ".join(
            f"{i['recipe']} [{i['phase']} {_duration(i['elapsed_s'])}/{_duration(i['limit_s'])}]"
            + (
                f" OVER BUDGET by {_duration(i['over_budget_s'])}"
                + ("; re-plan due" if i.get("waiting_on") else "")
                if i.get("over_budget_s")
                else ""
            )
            + (f" waiting on {i['waiting_on']}" if i.get("waiting_on") else "")
            for i in items
        )
        down = status.get("offline_lanes", {}).get(spark)
        if down and not text:
            text = (
                f"offline ({down['reason']}, last seen {down['last_seen_at'] or 'unknown'}): "
                "lane kept, waiting for the Spark"
            )
        lines.append(f"- {spark}: {text or 'idle'}")
    if status.get("waiting_for_lanes"):
        lines.append(
            f"- {status['waiting_for_lanes']} queued recipe(s) waiting for more lanes online"
        )
    if status.get("controller_waits"):
        lines += ["", "## Controller dependency waits (no recipe attempts)"]
        if status.get("controller_wait_issue"):
            lines.append(str(status["controller_wait_issue"]))
        lines += [
            f"- {key}: {SweepWait.model_validate(hold['wait']).description}; "
            f"next observation in {_duration(max(0, hold['next_check'] - status['updated_at']))}"
            + (
                f"; next allowed retry at {datetime.fromtimestamp(hold['retry_at'], UTC).isoformat()}"
                if hold.get("retry_at") is not None
                else ""
            )
            for key, hold in status["controller_waits"].items()
        ]
    if status.get("infrastructure"):
        lines += ["", "## Infrastructure problems (not recipe failures; retrying)"]
        lines += [
            f"- {source} x{item['count']}, next try in {_duration(item['retry_in_s'])}: {item['message']}"
            + (
                f"; escalation in {_duration(item['escalate_in_s'])}: {item['dependency']}"
                if "escalate_in_s" in item
                else ""
            )
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
        + (
            f" (waiting for {q['lanes_short']} more lane{'s' if q['lanes_short'] > 1 else ''})"
            if q.get("lanes_short")
            else ""
        )
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
    if status.get("verdict_identities"):
        lines += ["", "## Verdict content identities"]
        for key, value in sorted(status["verdict_identities"].items()):
            lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    if status.get("qualification"):
        lines += [
            "",
            "## Qualification",
            "",
            "| Recipe | Status and proof |",
            "|---|---|",
        ]
        lines += [
            f"| {key} | {_qualification_text(view)} |"
            for key, view in sorted(status["qualification"].items())
        ]
        lines += ["", str(status.get("evidence_publication", ""))]
    return "\n".join(lines) + "\n"


def _qualification_text(view: Mapping[str, Any]) -> str:
    text = str(view["status"])
    if view.get("changed"):
        text += "; changed: " + ", ".join(view["changed"])
    if view.get("last_pass"):
        proof = view["last_pass"]
        text += "; last pass " + str(proof["tested_at"])
        release = proof["platform"].get("controller_source_sha") or proof[
            "platform"
        ].get("controller_version")
        if release:
            text += " on " + release
    return text


def build_report(
    recipes: Mapping[str, Mapping[str, Any]],
    qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for key, entry in sorted(recipes.items()):
        result = read_result(entry, key)
        timings = result.timings
        perf = result.perf
        rows.append(
            {
                "recipe": key,
                "qualification": (qualification or {}).get(key),
                "status": entry.get("status", "pending"),
                "phase": entry.get("phase"),
                "failure_class": entry.get("failure_class"),
                "reason": entry.get("error") or entry.get("reason"),
                "cluster": entry.get("cluster"),
                "smoke": entry.get("smoke"),
                "download_s": timings.download_s,
                "load_s": timings.load_s,
                "smoke_s": timings.smoke_s,
                "ttft_ms": perf.ttft_ms,
                "tokens_per_second": perf.tokens_per_second,
                "content_sha256": entry.get("content_sha256"),
                "verdict_identity": entry.get("verdict_identity"),
                "evidence": policy.evidence_of(entry) or None,
                "evidence_bundle": entry.get("evidence_bundle"),
                "quality": [
                    note.model_dump(exclude_none=True) for note in result.quality
                ],
                "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics],
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

    qualified = [
        row["qualification"] for row in report["recipes"] if row.get("qualification")
    ]
    if qualified:
        lines[2] = _qualification_summary(qualified)

    def seconds(value: Any) -> str:
        return _duration(value) if value is not None else ""

    for row in report["recipes"]:
        if row["status"] == "skipped" and not row.get("qualification"):
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
        if row.get("qualification"):
            status = _qualification_text(row["qualification"])
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
    lines += ["", "## Verdict content identities"]
    for row in report["recipes"]:
        identity = recorded_identity(row)
        image = identity.runtime_image if identity is not None else None
        lines.append(
            f"- {row['recipe']}: recipe `{row['content_sha256']}`; "
            + (
                f"image `{image.image_digest}`; build `{image.build_key}`"
                if image is not None
                else "unknown image"
            )
            + (f"; models {', '.join(identity.model_digests)}" if identity else "")
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
        lines += ["", f"## Skipped ({len(skipped)})"]
        lines += [f"- {r['recipe']}: {r['reason']}" for r in skipped]
    return "\n".join(lines) + "\n"


def saved_qualification(inputs: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    views = {}
    for key, raw in inputs.items():
        observed = QualificationDerivationInput.model_validate(raw)
        evidence = merge_evidence(observed.evidence, read_evidence(root, key))
        views[key] = derive_status(
            evidence,
            observed.current_identity,
            observed.fleet_size,
            observed.testing_now,
        ).model_dump(mode="json", exclude_none=True)
    return views


def write_status(sweep: Sweep, final: bool = False) -> None:
    directory = sweep.cfg.state_dir
    status = build_status(sweep)
    inputs = {}
    for key, recipe in sweep.recipes.items():
        if key not in status["qualification"]:
            continue
        evidence = sweep.qualification_evidence.get(key) or read_evidence(
            Path(__file__).resolve().parents[1], key
        )
        inputs[key] = QualificationDerivationInput(
            evidence=evidence,
            current_identity=current_identity(recipe),
            fleet_size=len(sweep.enrolled()),
            testing_now=key in sweep.state.slots
            and sweep.state.slots[key].get("phase") != SweepSlotPhase.FINISHED,
        ).model_dump(mode="json", exclude_none=True)
    sweep.state.data["qualification_inputs"] = inputs
    write_atomic(
        directory / "status.json",
        json.dumps(status, indent=1, sort_keys=True, default=str),
    )
    write_atomic(directory / "status.md", render_status(status))
    if final:
        report = build_report(sweep.state.recipes, status.get("qualification"))
        write_atomic(
            directory / "report.json", json.dumps(report, indent=1, sort_keys=True)
        )
        write_atomic(directory / "report.md", render_report(report))


# -- repository evidence ------------------------------------------------------


def export_evidence(
    entries: Iterable[Mapping[str, Any]], out_dir: Path, known_recipes: Iterable[str]
) -> list[Path]:
    """Export the canonical public evidence model; raw diagnostics stay in state."""
    known = set(known_recipes)
    grouped: dict[str, RecipeQualificationEvidence] = {}
    for entry in entries:
        key = entry.get("recipe")
        if not isinstance(key, str) or key not in known:
            continue
        record = from_history(entry)
        if record is None:
            continue
        incoming = RecipeQualificationEvidence(recipe_key=key, records=(record,))
        previous = grouped.get(key, RecipeQualificationEvidence(recipe_key=key))
        grouped[key] = merge_evidence(previous, incoming)
    written = []
    for key, evidence in sorted(grouped.items()):
        path = out_dir / f"{key}.json"
        if path.exists():
            previous = RecipeQualificationEvidence.model_validate_json(path.read_text())
            evidence = merge_evidence(previous, evidence)
        write_atomic(path, evidence.model_dump_json(indent=2, exclude_none=True) + "\n")
        written.append(path)
    return written
