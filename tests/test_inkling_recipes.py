from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class InklingSmallRecipeTests(unittest.TestCase):
    def test_complete_model_inventory_and_two_spark_profile(self) -> None:
        model, recipe = (
            load("models/inkling-small-nvfp4.json"),
            load("recipes/inkling-small-nvfp4-sglang-dual.json"),
        )
        self.assertEqual(
            model["source"]["revision"], "b6a99534467840620d411e4cd4ad5819b2610d9c"
        )
        self.assertEqual(
            len({item["id"] for item in model["files"]}), len(model["files"])
        )
        self.assertEqual(recipe["topology"]["node_count"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["tensor"], 2)
        self.assertEqual(recipe["topology"]["parallelism"]["backend"], "native")

    def test_runtime_is_offline_and_multimodal_api_is_named(self) -> None:
        recipe = load("recipes/inkling-small-nvfp4-sglang-dual.json")
        interface = recipe["interfaces"][0]
        self.assertEqual(interface["adapter"], "openai")
        self.assertEqual(interface["model_aliases"], ["inkling-small"])
        self.assertEqual(recipe["runtime"]["engine"], "sglang")
        self.assertTrue(recipe["models"])

    def test_wrapper_resolves_only_the_compiler_rendezvous_sentinel(self) -> None:
        wrapper = ROOT / "adapters/inkling/small-dual/sglang-serve"
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "sglang/launch_server.py"
            launcher.parent.mkdir()
            launcher.write_text(
                "import json, os, sys\nopen(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))"
            )
            capture = Path(directory) / "arguments.json"
            env = {
                **os.environ,
                "PYTHONPATH": directory,
                "CAPTURE": str(capture),
                "VONK_MASTER_ADDR": "192.0.2.10",
                "VONK_MASTER_PORT": "29500",
                "VONK_LOCAL_ADDR": "192.0.2.11",
                "NCCL_SOCKET_IFNAME": "eth0",
                "NCCL_IB_HCA": "roce0",
                "NCCL_IB_GID_INDEX": "3",
                "TP_SOCKET_IFNAME": "eth0",
                "GLOO_SOCKET_IFNAME": "eth0",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(wrapper),
                    "--nnodes",
                    "2",
                    "--node-rank",
                    "1",
                    "--dist-init-addr",
                    "VONK_MASTER_ADDR:VONK_MASTER_PORT",
                ],
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(capture.read_text())[-1], "192.0.2.10:29500")


if __name__ == "__main__":
    unittest.main()


def _detector_class(source: str):
    """Execute the pinned detector without importing its GPU-dependent package."""
    import ast
    import json
    import logging
    from dataclasses import dataclass, field

    @dataclass
    class Result:
        normal_text: str = ""
        calls: list = field(default_factory=list)

    @dataclass
    class Call:
        tool_index: int
        name: str
        parameters: str

    class Base:
        def __init__(self):
            self._buffer = ""

        def get_structural_tag(self, *args):
            return None

        def _get_tool_indices(self, tools):
            return {tool.function.name: i for i, tool in enumerate(tools)}

        @staticmethod
        def _ends_with_partial_token(text, token):
            return max(
                (i for i in range(1, len(token)) if text.endswith(token[:i])),
                default=0,
            )

    class Tag:
        @staticmethod
        def model_validate(value):
            return value

    tree = ast.parse(source)
    detector = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *[node for node in tree.body if isinstance(node, ast.FunctionDef)],
            detector,
        ],
        type_ignores=[],
    )
    namespace = {
        "BaseFormatDetector": Base,
        "StructuralTag": Tag,
        "StreamingParseResult": Result,
        "ToolCallItem": Call,
        "Mapping": dict,
        "json": json,
        "logger": logging.getLogger(__name__),
    }
    tokens = {
        "CONTENT_INVOKE_TOOL_JSON": "<|content_invoke_tool_json|>",
        "CONTENT_INVOKE_TOOL_TEXT": "<|content_invoke_tool_text|>",
        "END_MESSAGE": "<|end_message|>",
        "MESSAGE_MODEL": "<|message_model|>",
    }
    namespace.update(tokens)
    namespace["INKLING_CONTROL_TOKENS"] = tuple(tokens.values())
    exec(  # noqa: S102 - execute the pinned detector without GPU imports.
        compile(ast.fix_missing_locations(module), "inkling_detector.py", "exec"),
        namespace,
    )
    return namespace["InklingDetector"]


def test_named_inkling_choice_constrains_one_call_and_preserves_parser(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import jsonschema

    source = ROOT / "tests/fixtures/inkling-detector.py.txt"
    target = tmp_path / "inkling_detector.py"
    target.write_bytes(source.read_bytes())
    tool = SimpleNamespace(
        function=SimpleNamespace(
            name="report_temperature",
            parameters={
                "type": "object",
                "$defs": {"City": {"type": "string"}},
                "properties": {"city": {"$ref": "#/$defs/City"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        )
    )
    choice = SimpleNamespace(function=SimpleNamespace(name="report_temperature"))
    original = _detector_class(target.read_text())()
    # The pinned image inherited no native named-choice grammar and fell back
    # to a legacy repeating tag. The historical T_REPORT response had two calls.
    assert original.get_structural_tag([tool], choice) is None
    patcher = ROOT / "adapters/inkling/small-dual/patch-named-tool.py"
    subprocess.run([sys.executable, str(patcher), str(target)], check=True)
    detector = _detector_class(target.read_text())()
    tag = detector.get_structural_tag([tool], choice)
    frame = tag["format"]
    assert frame["type"] == "tag"  # one tag, no repetition or triggered tags
    assert (
        frame["begin"]
        == '<|message_model|>report_temperature<|content_invoke_tool_json|>{"name":"report_temperature","args":'
    )
    assert frame["end"] == "}<|end_message|>"
    schema = frame["content"]["json_schema"]
    payload = {"city": "Amsterdam"}
    jsonschema.validate(payload, schema)
    import pytest

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([payload, payload], schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"city": 123}, schema)
    text = frame["begin"] + json.dumps(payload) + frame["end"]
    result = detector.detect_and_parse(text, [tool])
    assert len(result.calls) == 1
    assert result.calls[0].name == "report_temperature"
    assert json.loads(result.calls[0].parameters) == {"city": "Amsterdam"}
    # Token splits through the header, JSON, and terminator produce the same
    # one call. Neither parser path drops or deduplicates a legitimate call.
    calls = []
    streaming = _detector_class(target.read_text())()
    for character in text:
        calls.extend(streaming.parse_streaming_increment(character, [tool]).calls)
    assert [(call.name, call.parameters) for call in calls] == [
        (call.name, call.parameters) for call in result.calls
    ]
    assert (
        detector.get_structural_tag([tool], "required", parallel_tool_calls=True)
        is None
    )
    assert detector.get_structural_tag([tool], "auto") is None
    assert (
        "COPY patch-named-tool.py"
        in (ROOT / "adapters/inkling/small-dual/Dockerfile").read_text()
    )
    patched_bytes = target.read_bytes()
    drift = subprocess.run(
        [sys.executable, str(patcher), str(target)],
        capture_output=True,
        check=False,
    )
    assert drift.returncode != 0
    assert b"detector grammar changed" in drift.stderr
    assert target.read_bytes() == patched_bytes
