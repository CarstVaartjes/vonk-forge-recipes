"""A vLLM recipe that declares the Ray backend must ship a Ray-aware launcher.

The platform launches one process per rank and appends ``--nnodes``,
``--node-rank`` and (for a worker) ``--headless``. vLLM rejects ``--nnodes`` > 1
with ``--distributed-executor-backend ray`` ("nnodes > 1 can only be set when
distributed executor backend is mp, uni or external_launcher"), so the rank-0
process dies seconds after launch unless the image's entrypoint consumes those
options and starts the Ray cluster itself.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_405B = "llama-3-1-405b-instruct-awq-int4-nvidia-vllm-dual"


def ray_vllm_recipes() -> list[tuple[str, dict]]:
    found = []
    for path in sorted((ROOT / "recipes").glob("*.json")):
        recipe = json.loads(path.read_text())
        topology = recipe["topology"]
        if (
            recipe["runtime"]["engine"] == "vllm"
            and topology.get("node_count", 1) > 1
            and topology["parallelism"]["backend"] == "ray"
        ):
            found.append((path.stem, recipe))
    return found


def context_files(recipe: dict) -> list[Path]:
    context = ROOT / recipe["execution"]["build"]["context"]["path"]
    return [path for path in context.rglob("*") if path.is_file()]


class RayVllmLauncherTest(unittest.TestCase):
    def test_the_class_is_not_empty(self) -> None:
        self.assertTrue(ray_vllm_recipes())

    def test_every_ray_recipe_ships_a_launcher_that_consumes_the_placement_options(
        self,
    ) -> None:
        for slug, recipe in ray_vllm_recipes():
            with self.subTest(recipe=slug):
                launchers = [
                    path
                    for path in context_files(recipe)
                    if path.name != "Dockerfile"
                    and (text := path.read_text(errors="ignore"))
                    and "nnodes" in text
                    and "ray" in text
                    and "start" in text
                ]
                self.assertTrue(
                    launchers,
                    f"{slug} declares the Ray backend but ships no launcher that "
                    "consumes --nnodes and starts the Ray cluster; vLLM exits at "
                    "configuration time with --nnodes and the Ray executor",
                )

    def test_the_405b_context_fits_the_bounded_chat_check(self) -> None:
        recipe = json.loads((ROOT / "recipes" / f"{NVIDIA_405B}.json").read_text())
        context = recipe["settings"]["context_tokens"]["value"]
        arguments = {item["name"]: item for item in recipe["runtime"]["arguments"]}
        check = next(
            item
            for item in recipe["validation"]["serving"]["checks"]
            if item["name"] == "bounded-chat"
        )
        # The check asks for up to max_tokens completion tokens on top of the
        # chat-templated prompt (the playbook's 64-token context cannot hold both).
        self.assertGreater(context, check["request"]["body"]["max_tokens"] + 32)
        self.assertGreaterEqual(arguments["max-num-batched-tokens"]["value"], context)


class Llama405bNativeLaunchTest(unittest.TestCase):
    """The NGC vLLM image ships no Ray, so the 405B pair must not ask for it.

    The image's layers hold no ray package and no ray executable; a recipe that
    declares the Ray backend there fails its build-time image check (and, before
    that check existed, died at launch).  The native multi-node launch needs no
    Ray: the platform's --nnodes/--node-rank options go straight to vLLM.
    """

    def test_it_uses_the_native_multi_node_launch(self) -> None:
        recipe = json.loads((ROOT / "recipes" / f"{NVIDIA_405B}.json").read_text())
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "mp")
        self.assertNotIn(NVIDIA_405B, {slug for slug, _ in ray_vllm_recipes()})

    def test_its_image_check_does_not_require_ray(self) -> None:
        context = ROOT / "adapters/nvidia" / NVIDIA_405B
        self.assertNotIn("ray", (context / "verify-runtime.py").read_text())
        self.assertIn(
            "COPY --chmod=0755 vllm /opt/vonk/bin/vllm",
            (context / "Dockerfile").read_text(),
        )
        self.assertTrue(os.access(context / "vllm", os.X_OK))


if __name__ == "__main__":
    unittest.main()
