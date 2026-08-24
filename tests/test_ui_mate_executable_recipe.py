from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model-versions/ui-mate-27b-3ade2378.json"
RUNTIME = ROOT / "runtime-distributions/ui-mate-vllm-0-27-1-arm64.json"
RECIPE = ROOT / "recipes/ui-mate-27b-vllm-single.json"
ADAPTER = ROOT / "adapters/llm/ui-mate-vllm"

MODEL_REVISION = "3ade2378fc84032d5017c1a9c93c4eaa77d65e57"
HARNESS_REVISION = "d2b2e0aede83eeacfb1bc86f66503acbc4a6738a"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            _read(path),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arguments(recipe: dict[str, object]) -> dict[str, object]:
    runtime = recipe["runtime"]
    assert isinstance(runtime, dict)
    arguments = runtime["arguments"]
    assert isinstance(arguments, list)
    return {str(item["name"]): item["value"] for item in arguments}


class UIMateExecutableRecipeTests(unittest.TestCase):
    def test_model_snapshot_is_exact_and_complete(self) -> None:
        model = _read(MODEL)
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(model["parameters"]["total"], 27_356_728_560)
        self.assertEqual(model["limits"]["context_tokens"], 262_144)
        self.assertEqual(model["sizes"]["download_bytes"], 54_733_724_149)
        artifacts = model["artifacts"]
        self.assertEqual(len(artifacts), 22)
        self.assertTrue(all(item["revision"] == MODEL_REVISION for item in artifacts))
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            model["sizes"]["download_bytes"],
        )
        self.assertEqual(
            {item["path"] for item in artifacts if "weights" in item["roles"]},
            {f"model-{index:05d}-of-00012.safetensors" for index in range(1, 13)},
        )

    def test_official_agent_sources_are_vendored_byte_for_byte(self) -> None:
        self.assertEqual(
            _sha256(ADAPTER / "agents/__init__.py"),
            "e7cd2a9ff9e93cde70d9c0cbe6d475a9707cf6c9849effb27285739b6cb77c46",
        )
        self.assertEqual(
            _sha256(ADAPTER / "agents/ui_mate_agent.py"),
            "60e1f6924745333dcb0d363e7187d9045d04c06924a1150f02d334311fc38d77",
        )
        self.assertEqual(
            _sha256(ADAPTER / "agents/demo_workflow.py"),
            "4939d179bc15035661819efa149c3e1a7a5075a2f2334478e3c53e49991fd0a4",
        )
        runtime = _read(RUNTIME)
        self.assertEqual(runtime["source"]["revision"], HARNESS_REVISION)
        self.assertEqual(
            runtime["source"]["archive_sha256"],
            "98d13c2f66cc128d5b96b3c574243d02adb89b27f9a4dab4aae5ef0a7bbf9f28",
        )

    def test_recipe_uses_official_protocol_with_bounded_spark_resources(self) -> None:
        recipe = _read(RECIPE)
        arguments = _arguments(recipe)
        self.assertEqual(recipe["model"]["content_sha256"], _canonical_digest(MODEL))
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            _canonical_digest(RUNTIME),
        )
        self.assertEqual(arguments["served-model-name"], "UI_Mate")
        self.assertIs(arguments["trust-remote-code"], True)
        self.assertEqual(arguments["chat-template-content-format"], "openai")
        self.assertEqual(
            json.loads(str(arguments["limit-mm-per-prompt"])),
            {"image": 6, "video": 0},
        )
        self.assertEqual(arguments["mm-encoder-tp-mode"], "data")
        self.assertEqual(arguments["max-model-len"], 32_768)
        self.assertEqual(arguments["max-num-seqs"], 1)
        self.assertEqual(recipe["topology"]["node_count"], 1)
        self.assertEqual(recipe["topology"]["parallelism"]["tensor"], 1)
        self.assertLessEqual(
            recipe["topology"]["roles"][0]["resources"]["memory"]["startup_peak_bytes"],
            100_000_000_000,
        )

    def test_adapter_build_is_offline_and_never_executes_actions(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "FROM docker.io/vllm/vllm-openai@sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c",
            dockerfile,
        )
        for forbidden in ("curl ", "wget ", "git clone", "pip install", "uv pip"):
            self.assertNotIn(forbidden, dockerfile)
        wrapper = (ADAPTER / "ui-mate-step.py").read_text(encoding="utf-8")
        self.assertNotIn("exec(", wrapper)
        self.assertNotIn("pyautogui.", wrapper)

    def test_vendored_parser_scales_official_actions_without_actuating(self) -> None:
        pil = types.ModuleType("PIL")
        pil.Image = object()
        previous_pil = sys.modules.get("PIL")
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.modules["PIL"] = pil
        sys.path.insert(0, str(ADAPTER))
        try:
            spec = importlib.util.spec_from_file_location(
                "ui_mate_agent_test", ADAPTER / "agents/ui_mate_agent.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            response = """<think>Inspect.</think><action>Click.</action>
<tool_call>
<function=computer_use>
<parameter=action>
left_click
</parameter>
<parameter=coordinate>
[500,250]
</parameter>
</function>
</tool_call>"""
            action, code = module.parse_response(response, 1920, 1080)
            self.assertEqual(action, "Click.")
            self.assertEqual(code, ["pyautogui.click(960, 270)"])
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            sys.path.remove(str(ADAPTER))
            if previous_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = previous_pil


if __name__ == "__main__":
    unittest.main()
