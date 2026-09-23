from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_ADAPTERS = (
    ROOT / "adapters/deepseek/mia-vllm",
    ROOT / "adapters/deepseek/mia-vllm-vision",
)
ISSUE27_MARK = "# [issue27-hotfix] enforce max_num_partial_prefills on admission"
ISSUE27_SCRIPT_PATH = 'P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")'


@dataclass(eq=False)
class _Request:
    request_id: str
    num_computed_tokens: int
    num_prompt_tokens: int
    num_tokens: int
    num_output_placeholders: int = 0


ISSUE27_FIXTURE = """\
class Request:
    def __init__(self, request_id, num_computed_tokens, num_tokens,
                 num_output_placeholders=0, num_prompt_tokens=None):
        self.request_id = request_id
        self.num_computed_tokens = num_computed_tokens
        self.num_tokens = num_tokens
        self.num_prompt_tokens = num_tokens if num_prompt_tokens is None else num_prompt_tokens
        self.num_output_placeholders = num_output_placeholders


class _Logger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    @staticmethod
    def _fmt(message, args):
        return message % args if args else message

    def warning(self, message, *args):
        self.warnings.append(self._fmt(message, args))

    def info(self, message, *args):
        self.infos.append(self._fmt(message, args))


logger = _Logger()


class Scheduler:
    def __init__(self, running=(), tracked=(), config_cap=1):
        self.scheduler_config = type(
            "SchedulerConfig", (), {"max_num_partial_prefills": config_cap}
        )()
        self.max_num_running_reqs = 8
        self.num_waiting_for_streaming_input = 0
        self.running = list(running)
        self.waiting = [Request("waiting", 0, 512)]
        self.current_step = 0
        self.step_scheduled = {}
        self.step_new = []
        self.step_resumed = []
        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()
        self._inflight_prefills.update(tracked)

    def schedule(self):
        token_budget = 1
        admitted = 0
        can_schedule_waiting = True
        num_scheduled_tokens = dict(self.step_scheduled)
        scheduled_new_reqs = list(self.step_new)
        scheduled_resumed_reqs = list(self.step_resumed)
        if can_schedule_waiting:
            while self.waiting and token_budget > 0:
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                request = self.waiting.pop(0)
                num_new_tokens = min(
                    1024, request.num_tokens - request.num_computed_tokens
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                self.running.append(request)
                scheduled_new_reqs.append(request)
                if request.num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                admitted += 1
        return admitted
"""


def _request(
    name: str,
    computed: int,
    prompt: int,
    tokens: int | None = None,
    placeholders: int = 0,
):
    return _Request(
        name,
        computed,
        prompt,
        prompt if tokens is None else tokens,
        placeholders,
    )


def _apply_issue27(patch_path: Path, scheduler_path: Path) -> None:
    source = patch_path.read_text(encoding="utf-8").replace(
        ISSUE27_SCRIPT_PATH,
        f"P = Path({str(scheduler_path)!r})",
    )
    with (
        patch.object(sys, "argv", [str(patch_path)]),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        try:
            # Execute the patcher's CLI source under controlled argv and output.
            exec(compile(source, str(patch_path), "exec"), {"__name__": "__main__"})  # noqa: S102
        except SystemExit as exc:
            if exc.code != 0:
                raise


def _load_patched_scheduler(
    patch_path: Path, running=(), tracked=(), config_cap: int = 1
):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scheduler.py"
        path.write_text(ISSUE27_FIXTURE, encoding="utf-8")
        _apply_issue27(patch_path, path)
        patched = path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    # Load the patched scheduler fixture without importing the vLLM package.
    exec(compile(patched, str(path), "exec"), namespace)  # noqa: S102
    scheduler = namespace["Scheduler"](running, tracked, config_cap)
    return scheduler, namespace["logger"], patched


class MiaDeepSeekRecentPatchTests(unittest.TestCase):
    def test_recipe_revisions_retain_source_and_runtime_pins_with_verified_tree_hashes(
        self,
    ) -> None:
        expected = (
            (
                ROOT / "recipes/deepseek-v4-flash-0731-mia-dual.json",
                ROOT / "adapters/deepseek/mia-vllm/Dockerfile",
                "2.3.3",
                "0107cef1835a56d1a2bcdabf7d9e1a085b70338b",
                "28e0db557f1c23af34f005d9734aba1bc7b04810973b410d011ee657fed29ffd",
            ),
            (
                ROOT / "recipes/deepseek-v4-flash-vision-exp-mia-dual.json",
                ROOT / "adapters/deepseek/mia-vllm-vision/Dockerfile",
                "1.1.7",
                "c444d7032957f5a5437261d5366fd06b27a01760",
                "4aca0b546cc6b981e085eb2e9f8d9e87380d03891acea2b9a40e84ac4446e962",
            ),
        )
        for recipe_path, dockerfile_path, version, source_pin, patch_hash in expected:
            with self.subTest(recipe=recipe_path.name):
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
                dockerfile = dockerfile_path.read_text(encoding="utf-8")
                self.assertEqual(recipe["release"]["version"], version)
                self.assertTrue(
                    recipe["provenance"]["source_reference"].endswith(source_pin)
                )
                self.assertEqual(
                    recipe["execution"]["build"]["base_image"]["digest"],
                    "a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8",
                )
                self.assertIn(f'io.vonk.patch-tree-sha256="{patch_hash}"', dockerfile)
        vanilla_apply = (DEEPSEEK_ADAPTERS[0] / "apply-build-patches.py").read_text()
        self.assertLess(
            vanilla_apply.index('run("hotfix-gb10-spin-wait.sh"'),
            vanilla_apply.index('run("hotfix-vllm-issue117-shm-ring-buffer.py")'),
        )
        issue117 = (
            DEEPSEEK_ADAPTERS[0] / "patches/hotfix-vllm-issue117-shm-ring-buffer.py"
        ).read_text()
        self.assertIn(
            'UPSTREAM_MERGE = "10c75477b07c2f1a361f54b7357af1019bba5fd8"', issue117
        )
        self.assertIn("SHM_READER_RECHECK_INTERVAL_MS = 5000", issue117)
        issue136_patch = "patches/hotfix-vllm-issue136-xgrammar-termination.py"
        self.assertEqual(
            (DEEPSEEK_ADAPTERS[0] / issue136_patch).read_bytes(),
            (DEEPSEEK_ADAPTERS[1] / issue136_patch).read_bytes(),
        )
        for adapter in DEEPSEEK_ADAPTERS:
            self.assertIn(
                'run("hotfix-vllm-issue136-xgrammar-termination.py")',
                (adapter / "apply-build-patches.py").read_text(),
            )

    def test_issue27_r3_counts_running_prefills_not_the_async_tracking_set(
        self,
    ) -> None:
        for adapter in DEEPSEEK_ADAPTERS:
            patch_path = (
                adapter / "patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
            )
            with self.subTest(adapter=adapter.name):
                # The set can miss a still-running prefill. Admission must still
                # stop at the configured cap and report the bookkeeping drift.
                scheduler, logger, patched = _load_patched_scheduler(
                    patch_path,
                    running=[_request("prefill", 1024, 22829)],
                )
                self.assertIn("# [issue27-r3]", patched)
                self.assertEqual(scheduler.schedule(), 0)
                self.assertEqual(len(logger.warnings), 1)
                self.assertIn("tracked=0 running=1", logger.warnings[0])

                # An async KV reservation can leave a tracked object that is not
                # in the running list; it must not consume an admission slot.
                scheduler, _, _ = _load_patched_scheduler(
                    patch_path,
                    tracked=[object()],
                )
                self.assertEqual(scheduler.schedule(), 1)

    def test_issue27_r3_does_not_count_decoders_or_output_placeholders(self) -> None:
        for adapter in DEEPSEEK_ADAPTERS:
            patch_path = (
                adapter / "patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
            )
            with self.subTest(adapter=adapter.name):
                decoder = _request("decoder", 1024, 1024, tokens=1028, placeholders=4)
                scheduler, _, _ = _load_patched_scheduler(
                    patch_path,
                    running=[decoder],
                    tracked=[decoder],
                )
                self.assertEqual(scheduler.schedule(), 1)

    def test_issue27_r3_is_idempotent_and_refuses_an_older_gate(self) -> None:
        for adapter in DEEPSEEK_ADAPTERS:
            patch_path = (
                adapter / "patches/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
            )
            with (
                self.subTest(adapter=adapter.name),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "scheduler.py"
                path.write_text(ISSUE27_FIXTURE, encoding="utf-8")
                _apply_issue27(patch_path, path)
                once = path.read_bytes()
                _apply_issue27(patch_path, path)
                self.assertEqual(path.read_bytes(), once)

                stale = Path(directory) / "stale-scheduler.py"
                stale_bytes = (ISSUE27_MARK + "\n" + ISSUE27_FIXTURE).encode()
                stale.write_bytes(stale_bytes)
                source = patch_path.read_text(encoding="utf-8").replace(
                    ISSUE27_SCRIPT_PATH,
                    f"P = Path({str(stale)!r})",
                )
                with (
                    patch.object(sys, "argv", [str(patch_path)]),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    # Execute the CLI entrypoint to assert a stale patch exits closed.
                    exec(  # noqa: S102
                        compile(source, str(patch_path), "exec"),
                        {"__name__": "__main__"},
                    )
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(stale.read_bytes(), stale_bytes)

    def test_issue55_truncation_keeps_empty_lists_serializable_and_upgrades_old_patch(
        self,
    ) -> None:
        valid = SimpleNamespace(function=SimpleNamespace(arguments='{"ok":true}'))
        invalid = SimpleNamespace(function=SimpleNamespace(arguments='{"unfinished":'))
        no_function = SimpleNamespace(function=None)
        cases = (
            ("length", [invalid, no_function], []),
            ("length", [invalid, valid], [valid]),
            ("length", [valid], [valid]),
            ("length", [], []),
            ("stop", [invalid, valid], [invalid, valid]),
        )
        for adapter in DEEPSEEK_ADAPTERS:
            patch_path = adapter / "patches/hotfix-dsv4-issue55-tool-truncation.py"
            with self.subTest(adapter=adapter.name):
                hotfix = runpy.run_path(str(patch_path), run_name="issue55_patch")
                for streaming in (True, False):
                    for reason, calls, expected in cases:
                        message = SimpleNamespace(tool_calls=list(calls))
                        namespace = {
                            "output": SimpleNamespace(finish_reason=reason, index=0),
                            "delta_message": message,
                            "message": message,
                            "tools_streamed": [True],
                            "i": 0,
                            "tool_choice_function_name": None,
                            "is_finish_reason_tool_calls": True,
                            "logprobs": None,
                            "ChatCompletionResponseChoice": SimpleNamespace,
                        }
                        # Evaluate the exact helper body assembled by the patcher.
                        exec(hotfix["HELPER_NEW"], namespace)  # noqa: S102
                        source = (
                            hotfix["STREAMING_NEW"]
                            if streaming
                            else hotfix["NOSTREAM_NEW"] + "\n            )"
                        )
                        # Evaluate the generated replacement in its fixture namespace.
                        exec(textwrap.dedent(source), namespace)  # noqa: S102
                        self.assertEqual(message.tool_calls, expected)
                        finish = (
                            namespace["finish_reason_"]
                            if streaming
                            else namespace["choice_data"].finish_reason
                        )
                        self.assertEqual(
                            finish,
                            "length" if reason == "length" else "tool_calls",
                        )
                        data = {"tool_calls": message.tool_calls}
                        if len(data.get("tool_calls", [])) == 0:
                            data.pop("tool_calls", None)
                        encoded = json.dumps(data, default=vars)
                        self.assertNotIn('"tool_calls": null', encoded)

                # Upgrade every mixed streaming/non-streaming predecessor without
                # leaving the old `or None` forms behind, then remain idempotent.
                for old_stream, old_chat in (
                    (True, True),
                    (True, False),
                    (False, True),
                ):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        serving = root / hotfix["SERVING"]
                        serving.parent.mkdir(parents=True)
                        source = (
                            hotfix["HELPER_NEW"]
                            + hotfix["STREAMING_OLD"]
                            + "\n"
                            + hotfix["NOSTREAM_OLD"]
                        )
                        source = source.replace(
                            hotfix["STREAMING_OLD"],
                            hotfix["STREAMING_PREVIOUS"]
                            if old_stream
                            else hotfix["STREAMING_NEW"],
                        )
                        source = source.replace(
                            hotfix["NOSTREAM_OLD"],
                            hotfix["NOSTREAM_PREVIOUS"]
                            if old_chat
                            else hotfix["NOSTREAM_NEW"],
                        )
                        serving.write_text(source, encoding="utf-8")
                        with patch.object(sys, "argv", [str(patch_path), str(root)]):
                            self.assertEqual(hotfix["main"](), 0)
                            upgraded = serving.read_text(encoding="utf-8")
                            self.assertNotIn(hotfix["STREAMING_PREVIOUS"], upgraded)
                            self.assertNotIn(hotfix["NOSTREAM_PREVIOUS"], upgraded)
                            self.assertIn(hotfix["STREAMING_NEW"], upgraded)
                            self.assertIn(hotfix["NOSTREAM_NEW"], upgraded)
                            self.assertEqual(hotfix["main"](), 0)
                            self.assertEqual(
                                serving.read_text(encoding="utf-8"), upgraded
                            )


if __name__ == "__main__":
    unittest.main()
