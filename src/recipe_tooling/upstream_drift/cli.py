"""Compare immutable catalog source pins with explicitly selected upstream channels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.upstream_drift.embedded import embedded_inventory
from recipe_tooling.upstream_drift.heads import observe_embedded_heads
from recipe_tooling.upstream_drift.inventory import discover_watches, load_manifest
from recipe_tooling.upstream_drift.issues import file_issues, render_table
from recipe_tooling.upstream_drift.model import (
    DEFAULT_MANIFEST,
    CreatorSignals,
    DriftInputError,
    SignalError,
)
from recipe_tooling.upstream_drift.observe import ProviderClient, observe, report
from recipe_tooling.upstream_drift.signals import scan_creator_signals

ROOT = REPO_ROOT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--output", type=Path, help="also write the complete JSON report"
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="exit 2 when any watched source has advanced",
    )
    parser.add_argument(
        "--file-issues",
        type=Path,
        metavar="REPORT",
        help="file one issue per creator finding of a saved report, then exit",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.file_issues:
        document = json.loads(arguments.file_issues.read_text(encoding="utf-8"))
        signals = CreatorSignals.model_validate(document["creator_signals"])
        # Refused writes are warnings, not a failed run: this job is report-only.
        file_issues(signals)
        return 0
    root = arguments.root.resolve()
    manifest = (
        arguments.manifest or root / DEFAULT_MANIFEST.relative_to(ROOT)
    ).resolve()
    try:
        defaults, overrides = load_manifest(manifest)
        watches = discover_watches(root, defaults, overrides)
    except (DriftInputError, OSError, json.JSONDecodeError) as error:
        print(f"upstream drift input error: {error}", file=sys.stderr)
        return 1
    client = ProviderClient(
        github_token=os.environ.get("GITHUB_TOKEN"),
        huggingface_token=os.environ.get("HF_TOKEN"),
    )
    observations = [observe(watch, client) for watch in watches]
    checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        embedded = embedded_inventory(root, watches)
        observe_embedded_heads(embedded, client)
    except (DriftInputError, OSError, json.JSONDecodeError) as error:
        print(f"embedded source inventory error: {error}", file=sys.stderr)
        return 1
    # After the pin observations, not beside them: both share one rate-limited
    # GITHUB_TOKEN (P21), so the creator pool must not starve the pin checks.
    # Any failure here is a signal_error; it never costs the drift report.
    try:
        signals = scan_creator_signals(root, watches, client, datetime.now(UTC))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        signals = CreatorSignals(
            signal_errors=[
                SignalError(subject="creator-signals", detail=str(error)[:256])
            ]
        )
    document = report(observations, checked_at, embedded, signals)
    rendered_json = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered_json, encoding="utf-8")
    print(
        rendered_json if arguments.format == "json" else render_table(observations),
        end="",
    )
    if any(observation.status in {"missing", "error"} for observation in observations):
        return 1
    if arguments.fail_on_drift and any(
        observation.status == "advanced" for observation in observations
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
