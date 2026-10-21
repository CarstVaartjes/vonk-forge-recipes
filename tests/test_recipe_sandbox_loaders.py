"""Catalog loader policy: io_uring is unavailable in the workload sandbox."""

from __future__ import annotations

import ast
import json
import os
import runpy
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.recipe import (
    RecipeOption,
    RecipeOptionChoice,
    RecipeRuntimeArgument,
    RecipeRuntimeEnvironment,
)

ROOT = Path(__file__).resolve().parents[1]
QWEN = "qwen3-8-flash-next-nvfp4-eugr-vllm-single"


def recipe(slug: str = QWEN) -> RecipeDefinition:
    return RecipeDefinition.model_validate_json(
        (ROOT / "recipes" / f"{slug}.json").read_text()
    )


def require_sandbox_loaders(
    document: RecipeDefinition, *, image_policy: bool = True
) -> None:
    """Check base launch and every authored option through the real merger."""
    launches = [document]
    launches.extend(
        document.with_option_choices({option.name: choice.value})
        for option in document.options
        for choice in option.choices
    )
    for launch in launches:
        context = ROOT / launch.execution.build.context.path
        dockerfile = (ROOT / launch.execution.build.dockerfile).read_text()
        env: dict[str, str] = {}
        for line in dockerfile.replace("\\\n", " ").splitlines():
            if line.strip().upper().startswith("ENV "):
                values = shlex.split(line.strip()[4:])
                if values and "=" not in values[0]:
                    env[values[0]] = " ".join(values[1:])
                else:
                    for value in values:
                        key, separator, setting = value.partition("=")
                        if separator:
                            env[key] = setting
        env.update((item.name, str(item.value)) for item in launch.runtime.environment)
        # Image launchers enforce the sandbox transport before exec, including
        # when an upstream recipe still declares the old backend. Exec handoffs
        # are exercised separately with a fake vLLM that refuses io_uring.
        if image_policy:
            for wrapper in (context / "vllm-wrapper.py", context / "vllm-serve"):
                if not wrapper.is_file():
                    continue
                source = wrapper.read_text()
                for node in ast.parse(source).body:
                    if (
                        isinstance(node, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "SANDBOX_LOADER_ENV"
                            for target in node.targets
                        )
                        and "os.environ.update(SANDBOX_LOADER_ENV)" in source
                    ):
                        env.update(ast.literal_eval(node.value))
        args = {item.name: item.value for item in launch.runtime.arguments}
        load_format = str(args.get("load-format", "auto")).lower()
        assert load_format not in {"io_uring", "uring", "uring_buffered"}, (
            "load-format requires io_uring"
        )
        # InstantTensor's defaults try URING then AIO; BUFFERED tries
        # URING_BUFFERED then MMAP. Only an exclusive URING selection is unsafe.
        candidates = env.get("INSTANTTENSOR_BACKEND", "AUTO").upper().split(",")
        assert any(
            value.strip() not in {"URING", "URING_BUFFERED", ""} for value in candidates
        ), "InstantTensor requires io_uring"
        assert env.get("B12X_DISK_BACKEND") != "io_uring", (
            "b12x explicitly requires io_uring"
        )
        additional = args.get("additional-config", {})
        if isinstance(additional, str):
            additional = json.loads(additional)
        assert isinstance(additional, dict)
        engram = args.get("engram-config", {})
        if isinstance(engram, str):
            engram = json.loads(engram)
        assert isinstance(engram, dict)
        ple = additional.get("ple_table_memory", env.get("VLLM_PLE_TABLE_MEMORY"))
        # b12x Qwen Flash Next defaults BF16 PLE to disk, even without an env.
        implicit_ple_disk = (
            load_format == "b12x"
            and any("flash-next" in item.model.slug for item in launch.models)
            and ple is None
            and "VLLM_PLE_CPU_OFFLOAD" not in env
        )
        if (
            ple in {"disk", "io_uring"}
            or implicit_ple_disk
            or engram.get("table_memory") in {"disk", "io_uring"}
        ):
            posix = context / "_ple_reader.c"
            # Older b12x has no backend switch. Its image must install the
            # single POSIX transport, which the native behaviour tests exercise.
            posix_image = (
                posix.is_file()
                and "COPY _ple_reader.c" in dockerfile
                and "pread(" in posix.read_text()
                and "io_uring" not in posix.read_text()
            )
            assert posix_image or (
                env.get("B12X_DISK_BACKEND") == "gds"
                and env.get("CUFILE_FORCE_COMPAT_MODE") == "true"
            ), "disk PLE/Engram defaults to io_uring; select compatibility I/O"


def test_catalog_loaders_do_not_require_io_uring() -> None:
    for path in sorted((ROOT / "recipes").glob("*.json")):
        try:
            require_sandbox_loaders(recipe(path.stem))
        except AssertionError as error:
            pytest.fail(f"{path.name}: {error}")


@pytest.mark.parametrize(
    "environment",
    [
        [],  # Original eugr failure: omitted backend silently chooses io_uring.
        [RecipeRuntimeEnvironment(name="B12X_DISK_BACKEND", value="io_uring")],
        [RecipeRuntimeEnvironment(name="INSTANTTENSOR_BACKEND", value="URING")],
        [
            RecipeRuntimeEnvironment(
                name="INSTANTTENSOR_BACKEND", value="URING,URING_BUFFERED"
            )
        ],
    ],
)
def test_guard_rejects_uring_only_launches(
    environment: list[RecipeRuntimeEnvironment],
) -> None:
    document = recipe()
    document.runtime.environment = environment
    with pytest.raises(AssertionError, match="io_uring"):
        require_sandbox_loaders(document, image_policy=False)


@pytest.mark.parametrize("backend", ["AIO", "BUFFERED", "MMAP", "URING,AIO"])
def test_guard_preserves_documented_fallbacks(backend: str) -> None:
    document = recipe()
    document.runtime.environment.append(
        RecipeRuntimeEnvironment(name="INSTANTTENSOR_BACKEND", value=backend)
    )
    require_sandbox_loaders(document)


def test_guard_checks_nondefault_options() -> None:
    document = recipe()
    document.options = [
        RecipeOption(
            name="loader",
            label="Loader",
            help="Select a loader",
            choices=[
                RecipeOptionChoice(
                    value="standard", label="Standard", help="Standard", default=True
                ),
                RecipeOptionChoice(
                    value="uring",
                    label="Uring",
                    help="Requires io_uring",
                    args=[RecipeRuntimeArgument(name="load-format", value="io_uring")],
                ),
            ],
        )
    ]
    with pytest.raises(AssertionError, match="io_uring"):
        require_sandbox_loaders(document)


def test_guard_checks_implicit_bf16_ple_and_additional_config() -> None:
    document = recipe()
    document.runtime.environment = []
    with pytest.raises(AssertionError, match="defaults to io_uring"):
        require_sandbox_loaders(document, image_policy=False)

    document.runtime.environment = [
        RecipeRuntimeEnvironment(name="VLLM_PLE_TABLE_MEMORY", value="ram")
    ]
    require_sandbox_loaders(document, image_policy=False)
    document.runtime.arguments.append(
        RecipeRuntimeArgument(
            name="additional-config", value='{"ple_table_memory":"disk"}'
        )
    )
    with pytest.raises(AssertionError, match="defaults to io_uring"):
        require_sandbox_loaders(document, image_policy=False)


@pytest.mark.parametrize(
    "wrapper",
    [
        "adapters/eugr/qwen3-8-flash-next-nvfp4/vllm-wrapper.py",
        "adapters/qwen/vcruz-fp4-local-vllm-single/vllm-serve",
    ],
)
def test_launcher_replaces_uring_even_when_requested(
    monkeypatch: pytest.MonkeyPatch, wrapper: str
) -> None:
    monkeypatch.setenv("B12X_DISK_BACKEND", "io_uring")
    monkeypatch.setenv("CUFILE_FORCE_COMPAT_MODE", "false")
    monkeypatch.setattr(sys, "argv", [wrapper, "serve", "/models"])
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: "/fake/vllm")
    calls: list[list[str]] = []

    def exec_without_uring(
        executable: str, arguments: list[str] | tuple[str, ...]
    ) -> None:
        assert executable
        # Inject the sandbox's hard refusal at the process handoff.
        if os.environ.get("B12X_DISK_BACKEND") != "gds":
            raise RuntimeError(
                "io_uring initialization failed: Operation not permitted"
            )
        assert os.environ["CUFILE_FORCE_COMPAT_MODE"] == "true"
        calls.append(list(arguments))

    monkeypatch.setattr(os, "execv", exec_without_uring)
    runpy.run_path(str(ROOT / wrapper), run_name="__main__")
    assert len(calls) == 1
    assert calls[0][-2:] == ["serve", "/models"]
