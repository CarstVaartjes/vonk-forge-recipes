from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from repo_root import ROOT

from recipe_tooling import catalog_index
from vonk_forge_contracts import document_sha256

MODEL = ROOT / "models/ui-mate-27b-3ade2378.json"
RECIPE = ROOT / "recipes/ui-mate-27b-vllm-single.json"
ADAPTER = ROOT / "adapters/llm/ui-mate-vllm"
SOURCE_BUNDLE = vars(catalog_index)["source_bundle"]

MODEL_REVISION = "3ade2378fc84032d5017c1a9c93c4eaa77d65e57"
HARNESS_REVISION = "d185dc9d74cfcab3a890d7ffb2bb011ecdd64c64"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    return document_sha256(_read(path))


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
        assert model["source"]["revision"] == MODEL_REVISION  # pyright: ignore[reportIndexIssue]
        artifacts = model["files"]
        assert len(artifacts) == 22  # pyright: ignore[reportArgumentType]
        assert (
            {item["path"] for item in artifacts if "weights" in item["roles"]}  # pyright: ignore[reportGeneralTypeIssues]
            == {f"model-{index:05d}-of-00012.safetensors" for index in range(1, 13)}
        )

    def test_official_agent_sources_are_vendored_byte_for_byte(self) -> None:
        assert (
            _sha256(ADAPTER / "agents/__init__.py")
            == "e7cd2a9ff9e93cde70d9c0cbe6d475a9707cf6c9849effb27285739b6cb77c46"
        )
        assert (
            _sha256(ADAPTER / "agents/ui_mate_agent.py")
            == "60e1f6924745333dcb0d363e7187d9045d04c06924a1150f02d334311fc38d77"
        )
        assert (
            _sha256(ADAPTER / "agents/demo_workflow.py")
            == "4939d179bc15035661819efa149c3e1a7a5075a2f2334478e3c53e49991fd0a4"
        )
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        assert f'org.opencontainers.image.revision="{HARNESS_REVISION}"' in dockerfile
        assert (
            'io.vonk.vllm.build-commit="ced6857afa0ea7b2e3f0846a62e1394e90f15607"'
            in dockerfile
        )
        assert (
            'io.vonk.ui-mate.upstream-archive-sha256="12046e80b390539417bfc803d1effcd7b867a4bb95ac8a20a5631ce60db9ab4d"'
            in dockerfile
        )

    def test_recipe_uses_official_protocol_with_bounded_spark_resources(self) -> None:
        recipe = _read(RECIPE)
        arguments = _arguments(recipe)
        assert (
            recipe["models"][0]["model"]["content_sha256"]  # pyright: ignore[reportIndexIssue]
            == _canonical_digest(MODEL)
        )
        assert (
            recipe["execution"]["build"]["base_image"]["digest"]  # pyright: ignore[reportIndexIssue]
            == "e06f024cf1dd864a2f4b6e033b496807c55952404c14cf8c1b08e4ca9888597e"
        )
        assert arguments["served-model-name"] == "UI_Mate"
        assert arguments["trust-remote-code"] is True
        assert arguments["chat-template-content-format"] == "openai"
        assert json.loads(str(arguments["limit-mm-per-prompt"])) == {
            "image": 6,
            "video": 0,
        }
        assert arguments["mm-encoder-tp-mode"] == "data"
        assert recipe["settings"]["context_tokens"]["value"] == 32768  # pyright: ignore[reportIndexIssue]
        assert recipe["settings"]["concurrency"]["value"] == 1  # pyright: ignore[reportIndexIssue]
        assert arguments["max-num-batched-tokens"] == 8192
        assert arguments["max-cudagraph-capture-size"] == 4
        assert recipe["topology"]["node_count"] == 1  # pyright: ignore[reportIndexIssue]
        assert recipe["topology"]["parallelism"]["tensor"] == 1  # pyright: ignore[reportIndexIssue]

    def test_adapter_build_is_offline_and_never_executes_actions(self) -> None:
        dockerfile = (ADAPTER / "Dockerfile").read_text(encoding="utf-8")
        assert (
            "FROM docker.io/vllm/vllm-openai@sha256:e06f024cf1dd864a2f4b6e033b496807c55952404c14cf8c1b08e4ca9888597e"
            in dockerfile
        )
        for cache in ("HOME", "HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR"):
            assert cache in dockerfile
        runtime_wrapper = (ADAPTER / "vllm-wrapper.sh").read_text(encoding="utf-8")
        assert 'mkdir -p \\\n  "$HOME"' in runtime_wrapper
        for forbidden in ("curl ", "wget ", "git clone", "pip install", "uv pip"):
            assert forbidden not in dockerfile
        wrapper = (ADAPTER / "ui-mate-step.py").read_text(encoding="utf-8")
        assert "exec(" not in wrapper
        assert "pyautogui." not in wrapper

    def test_recipe_binds_the_exact_offline_adapter_bundle(self) -> None:
        recipe = _read(RECIPE)
        context = recipe["execution"]["build"]["context"]  # pyright: ignore[reportIndexIssue]
        assert context["path"] == "adapters/llm/ui-mate-vllm"
        assert len(SOURCE_BUNDLE(ADAPTER)[0]) > 0

    def test_vendored_parser_scales_official_actions_without_actuating(self) -> None:
        pil = types.ModuleType("PIL")
        pil.Image = object()  # pyright: ignore[reportAttributeAccessIssue]
        previous_pil = sys.modules.get("PIL")
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.modules["PIL"] = pil
        sys.path.insert(0, str(ADAPTER))
        try:
            spec = importlib.util.spec_from_file_location(
                "ui_mate_agent_test", ADAPTER / "agents/ui_mate_agent.py"
            )
            assert spec is not None
            assert spec.loader is not None
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
            assert action == "Click."
            assert code == ["pyautogui.click(960, 270)"]
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            sys.path.remove(str(ADAPTER))
            if previous_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = previous_pil


if __name__ == "__main__":
    unittest.main()
