"""Command line: ``plan``, ``run``, ``status``, ``report``, ``export-evidence``, ``file-issues``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import policy
from .catalog import fetch_fleet, fetch_models, fetch_recipes
from .definitions import Definitions
from .issues import file_issues
from .plan import make_plan, render_plan
from .prefetch import TIB, PrefetchConfig
from .report import build_report, export_evidence, render_report
from .run import Clock, Sweep, SweepConfig
from .smoke import HttpConfig
from .state import ResultsLog, State
from .vonkctl import OWNER_PROFILES, Vonkctl, VonkctlError

# Kept equal to AUTHORITY_ID in tools/build-qualification-authority (a test checks it), so
# the results log is read by `vonk-fleet-qualify-campaign status` for that authority.
AUTHORITY_ID = "nl-family-aware-20260924"
DEFAULT_STATE_DIR = Path.home() / ".vonk-sweep"
REPO = "CarstVaartjes/vonk-forge-recipes"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sweep-recipes", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
        sub.add_argument("--vonkctl", default="vonkctl", help="the vonkctl executable")
        sub.add_argument(
            "--spark",
            action="append",
            default=[],
            help="Spark name or id (default: every online Spark)",
        )
        sub.add_argument(
            "--only",
            action="append",
            default=[],
            help="only recipes whose selector contains this",
        )
        sub.add_argument(
            "--boost",
            action="append",
            default=None,
            help="prioritise recipes matching this word (default: glm-5-3)",
        )
        sub.add_argument(
            "--seed-list",
            type=Path,
            help="JSON list of recipe slugs to test first (already prefetched)",
        )
        sub.add_argument("--authority-id", default=AUTHORITY_ID)

    plan = commands.add_parser(
        "plan", help="read-only: unique bytes, download time and order"
    )
    common(plan)
    plan.add_argument(
        "--rate-mb-s",
        type=float,
        default=0.0,
        help="measured download rate for the ETA",
    )
    plan.add_argument("--json", action="store_true")

    run = commands.add_parser(
        "run", help="run the sweep on the real fleet (stops what runs on the Sparks)"
    )
    common(run)
    run.add_argument(
        "--yes",
        action="store_true",
        help="confirm that the sweep may stop whatever the Sparks run",
    )
    run.add_argument("--sweep-profile", type=int, default=10)
    run.add_argument("--pin-profile", type=int, default=13)
    run.add_argument(
        "--no-pin",
        action="store_true",
        help="do not keep a never-loaded profile that protects downloads from eviction",
    )
    run.add_argument("--client-key-file", type=Path)
    run.add_argument("--ca-file", type=Path)
    run.add_argument("--insecure-tls", action="store_true")
    run.add_argument(
        "--nas-budget-tib",
        type=float,
        default=2.0,
        help="bytes of downloaded-but-untested models to hold",
    )
    run.add_argument("--max-model-downloads", type=int, default=3)
    run.add_argument("--max-image-pulls", type=int, default=2)
    run.add_argument(
        "--first-start-minutes",
        type=float,
        default=60.0,
        help="load timeout before any load has been measured",
    )
    run.add_argument("--timeout-multiplier", type=float, default=3.0)
    run.add_argument("--max-load-minutes", type=float, default=120.0)
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after this many tested recipes (0 = all)",
    )
    run.add_argument(
        "--variants-last",
        action="store_true",
        help="test one recipe per model first, variants afterwards",
    )
    run.add_argument(
        "--no-revalidate",
        action="store_true",
        help="do not retest passed recipes whose document changed",
    )
    run.add_argument(
        "--restore-owner",
        type=int,
        metavar="PROFILE",
        help="load this owner profile (for example 2) when the sweep ends",
    )
    run.add_argument(
        "--leave-running",
        action="store_true",
        help="do not stop the last recipes at the end",
    )
    run.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="keep polling the library for changed recipes after the queue drains",
    )
    run.add_argument("--poll-seconds", type=float, default=10.0)
    run.add_argument(
        "--file-issues",
        action="store_true",
        help=f"file one {REPO} issue per failure (label hardware-test)",
    )

    for name, text in (
        ("status", "print the live status"),
        ("report", "write and print the report"),
    ):
        common(commands.add_parser(name, help=text))
    export = commands.add_parser(
        "export-evidence",
        help="write qualification/hardware-evidence/<slug>.jsonl files",
    )
    common(export)
    export.add_argument(
        "--output-dir", type=Path, default=Path("qualification/hardware-evidence")
    )
    issues = commands.add_parser(
        "file-issues", help="file the missing hardware-test issues for failed recipes"
    )
    common(issues)
    issues.add_argument("--repo", default=REPO)
    issues.add_argument("--dry-run", action="store_true")
    return parser


def _boost(args: argparse.Namespace) -> policy.Boost:
    keywords = tuple(args.boost) if args.boost is not None else ("glm-5-3",)
    keyword = policy.keyword_boost(keywords)
    seed = _seed(args)
    return lambda recipe: (
        keyword(recipe) * (1000.0 if recipe.slug in seed or recipe.key in seed else 1.0)
    )


def _seed(args: argparse.Namespace) -> frozenset[str]:
    if not args.seed_list:
        return frozenset()
    return frozenset(
        str(item) for item in json.loads(args.seed_list.read_text(encoding="utf-8"))
    )


def _config(args: argparse.Namespace) -> SweepConfig:
    if args.sweep_profile in OWNER_PROFILES or (
        not args.no_pin and args.pin_profile in OWNER_PROFILES
    ):
        raise SystemExit(
            "profiles 1-3 are the owner's: choose other numbers for --sweep-profile and --pin-profile"
        )
    if not args.no_pin and args.pin_profile == args.sweep_profile:
        raise SystemExit("--pin-profile must differ from --sweep-profile")
    return SweepConfig(
        state_dir=args.state_dir,
        authority_id=args.authority_id,
        sweep_profile=args.sweep_profile,
        pin_profile=None if args.no_pin else args.pin_profile,
        sparks=tuple(args.spark),
        poll_seconds=args.poll_seconds,
        http=HttpConfig(args.client_key_file, args.ca_file, args.insecure_tls),
        prefetch=PrefetchConfig(
            max_model_downloads=args.max_model_downloads,
            max_image_pulls=args.max_image_pulls,
            nas_budget_bytes=int(args.nas_budget_tib * TIB),
            pin_profile=None if args.no_pin else args.pin_profile,
        ),
        timeouts=policy.TimeoutPolicy(
            first_start_seconds=args.first_start_minutes * 60,
            multiplier=args.timeout_multiplier,
            cap_seconds=args.max_load_minutes * 60,
        ),
        boost=tuple(args.boost) if args.boost is not None else ("glm-5-3",),
        seed=_seed(args),
        only=tuple(args.only),
        limit=args.limit,
        variants_last=args.variants_last,
        revalidate_passes=not args.no_revalidate,
        restore_owner=args.restore_owner,
        stop_at_end=not args.leave_running,
        watch_seconds=args.watch,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    vonkctl_factory: Callable[[str], Vonkctl] = lambda exe: Vonkctl(exe),
    clock: Clock | None = None,
    definitions: Definitions | None = None,
) -> int:
    args = _parser().parse_args(argv)
    state_path = args.state_dir / "state.json"
    results = ResultsLog(args.state_dir / "results.jsonl", args.authority_id)

    if args.command == "status":
        status = args.state_dir / "status.md"
        print(
            status.read_text(encoding="utf-8") if status.exists() else "no status yet"
        )
        return 0
    if args.command in ("report", "export-evidence", "file-issues"):
        state = State.load(state_path)
        if args.command == "report":
            report = build_report(state.recipes)
            args.state_dir.mkdir(parents=True, exist_ok=True)
            (args.state_dir / "report.json").write_text(
                json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
            )
            text = render_report(report)
            (args.state_dir / "report.md").write_text(text, encoding="utf-8")
            print(text)
        elif args.command == "export-evidence":
            known = [
                f"vonk-forge/{p.stem}"
                for p in (Path(__file__).resolve().parents[1] / "recipes").glob(
                    "*.json"
                )
            ]
            for path in export_evidence(results.entries(), args.output_dir, known):
                print(path)
        else:
            for item in file_issues(state.recipes, args.repo, dry_run=args.dry_run):
                print(f"{item['recipe']}: {item['url']}")
            state.save()
        return 0

    vk = vonkctl_factory(args.vonkctl)
    defs = definitions or Definitions.load()
    if args.command == "plan":
        recipes, _ = fetch_recipes(vk)
        fleet = fetch_fleet(vk, 130_663_231_488)
        usable = (
            len([s for s in fleet.sparks if s.online])
            if not args.spark
            else len(args.spark)
        )
        state = State.load(state_path)
        rate = args.rate_mb_s * 1e6 or float(state.data["rate"].get("ema", 0.0))
        plan = make_plan(
            recipes, fetch_models(vk), usable, state.recipes, _boost(args), rate
        )
        print(json.dumps(plan, indent=1) if args.json else render_plan(plan))
        return 0

    if not args.yes:
        print(
            "run stops whatever the Sparks currently run (a profile load replaces it).\n"
            "Review `plan`, then pass --yes. Use --restore-owner 2 to load the owner's profile 2 at the end.",
            file=sys.stderr,
        )
        return 2
    config = _config(args)
    state = State.load(state_path)
    try:
        code = Sweep(config, vk, state, results, defs, clock).run()
    except (RuntimeError, VonkctlError) as error:
        print(f"sweep-recipes: {error}", file=sys.stderr)
        return 2
    if args.file_issues:
        for item in file_issues(state.recipes, REPO):
            print(f"{item['recipe']}: {item['url']}")
        state.save()
    return code
