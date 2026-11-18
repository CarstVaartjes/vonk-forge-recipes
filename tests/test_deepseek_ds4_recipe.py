"""Recipe-structure checks for the DS4 recipes.

The upstream engine's own command line is deliberately not exercised here. The
pinned ds4 source has its own release cadence, and a flag it renames or drops
surfaces when the container runs, not in this producer suite. A check that
compiled the vendored archive and asked its parser about our argument names
would fail on upstream drift while proving nothing about the recipe document.
These tests pin what this repository authors: the declared topology, argument
names and release metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from documents import recipe_display
from generated_catalog import GENERATED

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/deepseek-v4-flash-0731-ds4-single.json"


WRAPPER = ROOT / "adapters/deepseek/ds4-tp2/ds4-serve"
SERVER_ARGUMENTS = [
    "--model",
    "/models/m.gguf",
    "--ctx",
    "32768",
    "--tensor-parallel",
    "--transport",
    "rdma",
    "--batched-session",
    "8",
    "--host",
    "0.0.0.0",
    "--port",
    "8080",
]


def run_wrapper(rank: str, arguments: list[str], **env: str) -> list[str]:
    """Run the wrapper with the two ds4 binaries replaced by echoing stubs."""

    with tempfile.TemporaryDirectory() as folder:
        for name in ("ds4", "ds4-server"):
            stub = Path(folder) / name
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$0" "$@"\n')
            stub.chmod(0o755)
        script = Path(folder) / "ds4-serve"
        script.write_text(WRAPPER.read_text().replace("/opt/vonk/bin", folder))
        script.chmod(0o755)
        placement = {
            "VONK_WORLD_SIZE": "2",
            "VONK_RANK": rank,
            "VONK_MASTER_ADDR": "10.0.0.1",
            "VONK_MASTER_PORT": "29500",
        } | env
        done = subprocess.run(
            [str(script), *arguments],
            env={"PATH": os.environ["PATH"]} | placement,
            capture_output=True,
            text=True,
            check=False,
        )
    if done.returncode:
        raise subprocess.CalledProcessError(
            done.returncode, "ds4-serve", stderr=done.stderr
        )
    return [Path(done.stdout.splitlines()[0]).name, *done.stdout.splitlines()[1:]]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    body = json.dumps(
        load(path), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(body.encode()).hexdigest()


class DeepseekDs4RecipeTests(unittest.TestCase):
    def test_cuda_profile_describes_ordered_two_session_fallback(self) -> None:
        recipe = load(RECIPE)
        self.assertIn(
            "two-session concurrency", recipe_display(recipe["identity"]["slug"]).title
        )
        self.assertEqual(
            next(
                a["setting"]
                for a in recipe["runtime"]["arguments"]
                if a["name"] == "batched-session"
            ),
            "concurrency",
        )
        self.assertEqual(recipe["settings"]["kind"], "generation")
        self.assertEqual(recipe["topology"]["node_count"], 1)
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "ctx", "batched-session"])

    def test_dspark_declares_target_drafter_and_context_arguments(self) -> None:
        recipe = load(
            ROOT / "recipes/deepseek-v4-flash-0731-ds4-dspark-latency-single.json"
        )
        names = [argument["name"] for argument in recipe["runtime"]["arguments"]]
        self.assertEqual(names, ["model", "dspark", "mtp-model", "ctx"])

    def test_dual_wrapper_serves_on_rank_zero_and_joins_on_rank_one(self) -> None:
        self.assertEqual(
            run_wrapper("0", SERVER_ARGUMENTS),
            [
                "ds4-server",
                *SERVER_ARGUMENTS,
                "--role",
                "coordinator",
                "--listen",
                "10.0.0.1",
                "29500",
            ],
        )
        self.assertEqual(
            run_wrapper("1", SERVER_ARGUMENTS),
            [
                "ds4",
                "--model",
                "/models/m.gguf",
                "--ctx",
                "32768",
                "--tensor-parallel",
                "--transport",
                "rdma",
                "--role",
                "worker",
                "--coordinator",
                "10.0.0.1",
                "29500",
            ],
        )

    def test_release_binds_the_current_recipe_digest(self) -> None:
        index = load(GENERATED / "catalog-index.json")
        entry = next(
            item
            for item in index["recipes"]
            if item["source_path"] == f"recipes/{RECIPE.name}"
        )
        self.assertEqual(entry["package"]["recipe_content_sha256"], digest(RECIPE))


if __name__ == "__main__":
    unittest.main()
