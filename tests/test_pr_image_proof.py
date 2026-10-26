"""PR builds and release promotion must use one executable image proof path."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/recipe-images.yml"
KEY = "a" * 64


def _step(name: str) -> str:
    step = WORKFLOW.read_text().split(f"      - name: {name}\n", 1)[1]
    body = step.split("        run: |\n", 1)[1]
    return textwrap.dedent(re.split(r"\n(?= {0,9}\S)", body, maxsplit=1)[0])


def test_pr_and_publish_share_planner_build_and_smokes():
    validate = (ROOT / ".github/workflows/validate.yml").read_text()
    publish = (ROOT / ".github/workflows/publish.yml").read_text()
    for caller in (validate, publish):
        assert "uses: ./.github/workflows/recipe-images.yml" in caller
        assert "scripts/recipe-image-plan" not in caller
        assert "podman" not in caller
    assert "reuse_only" not in publish
    assert "--reuse-only" not in WORKFLOW.read_text()
    producer = publish.split("\n  producer-contracts:\n", 1)[1].split(
        "\n  catalog-index:", 1
    )[0]
    # GitHub validates nested workflow permissions even for skipped PR jobs.
    assert "packages: write" in producer and "actions: write" in producer
    gate = validate.split("\n  catalog-index:\n", 1)[1].split(
        "\n  producer-contracts:", 1
    )[0]
    assert "needs: images" in gate and "always()" in gate
    assert 'test "$IMAGE_RESULT" = success' in gate
    workflow = WORKFLOW.read_text()
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert '--base-plan "$RUNNER_TEMP/base-plan.json"' in workflow
    assert "python tools/prepare-remote-code-smoke" in workflow
    assert workflow.index(
        "Build the recipe image with the Spark build flags"
    ) < workflow.index("Push to every recipe that uses this image")
    build = _step("Build the recipe image with the Spark build flags")
    # The recipe Dockerfile executes its native build/import checks before the
    # separate remote-module smoke and the final platform adapter build.
    assert build.index('"${recipe_args[@]}"') < build.index("remote-code-smoke.py")
    assert build.index("remote-code-smoke.py") < build.index('"${adapter_args[@]}"')


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_workflow_plan_files_exclude_catalog_progress(tmp_path: Path, event: str):
    workspace = tmp_path / "workspace"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(ROOT), str(workspace)],
        check=True,
        capture_output=True,
        timeout=15,
    )
    scripts = workspace / ".vonk-forge/scripts"
    scripts.mkdir(parents=True)
    builder = scripts / "build-recipe-library"
    builder.write_text('#!/bin/sh\necho "wrote catalog-index.json for $1"\n')
    builder.chmod(0o755)
    tools = workspace / "tools"
    tools.mkdir()
    shutil.copyfile(ROOT / "tools/prebuilt-images", tools / "prebuilt-images")
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    # Replace environment setup and the external platform planner only. Keep
    # noisy catalog stdout and run the real downstream plan consumer.
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
if any(arg.endswith('/scripts/recipe-image-plan') for arg in args):
    library = args[args.index('--library-root') + 1]
    key = ('a' if library == os.environ['GITHUB_WORKSPACE'] else 'b') * 64
    Path(args[args.index('--output-dir') + 1]).mkdir(parents=True)
    print('planner diagnostic', file=sys.stderr)
    print(json.dumps({'images': [{'build_key': key, 'recipes': []}]}))
else:
    script = args.index('python') + 1
    raise SystemExit(subprocess.run(
        [sys.executable, os.environ['SMOKE_TOOL'], *args[script + 1:]],
        check=False, timeout=10,
    ).returncode)
"""
    )
    uv.chmod(0o755)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
    ).strip()
    platform_ref = subprocess.check_output(
        [sys.executable, str(tools / "prebuilt-images"), "platform-ref"],
        cwd=ROOT,
        text=True,
        timeout=5,
    ).strip()
    script = _step("Plan one image per distinct build key").replace(
        "${{ steps.platform.outputs.ref }}", platform_ref
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=workspace,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(tmp_path),
            "EVENT_NAME": event,
            "BASE_SHA": head,
            "SMOKE_TOOL": str(ROOT / "tools/prepare-remote-code-smoke"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plans = [("plan.json", KEY)]
    if event == "pull_request":
        plans.append(("base-plan.json", "b" * 64))
    for filename, key in plans:
        assert json.loads((tmp_path / filename).read_text()) == {
            "images": [{"build_key": key, "recipes": []}]
        }
    assert result.stdout.count("wrote catalog-index.json") == len(plans)


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_required_check_rejects_every_non_success_image_result(result: str):
    workflow = (ROOT / ".github/workflows/validate.yml").read_text()
    gate = workflow.split("- name: Require PR image proof", 1)[1]
    script = gate.split("        run: ", 1)[1].splitlines()[0]
    checked = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "IMAGE_RESULT": result},
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (checked.returncode == 0) == (result == "success")


@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("fault", ["", "native", "remote", "adapter"])
def test_build_or_smoke_failure_cannot_publish_an_image(
    tmp_path: Path, fault: str, event: str
):
    directory = tmp_path / "plan" / KEY
    directory.mkdir(parents=True)
    (directory / "remote-code-smoke.py").touch()
    (tmp_path / "plan/plan.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "build_key": KEY,
                        "base_images": [],
                        "recipe_build_arguments": [],
                        "adapter_containerfile": "FROM scratch",
                        "adapter_build_arguments": [],
                        "adapter_build_argument": "VONK_RECIPE_IMAGE",
                    }
                ]
            }
        )
    )
    (tmp_path / "plan/selection.json").write_text(
        json.dumps(
            {
                "builds": [
                    {
                        "build_key": KEY,
                        "recipes": [{"repository": "ghcr.io/o/recipe", "version": "1"}],
                    }
                ]
            }
        )
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    # Execute the workflow shell against a fake container boundary. No host
    # privilege command or actual container build runs in this test.
    podman = binary / "podman"
    podman.write_text("""#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(' '.join(args) + '\\n')
fault = os.environ['FAULT']
stage = 'remote' if 'run' in args else ('native' if args[-1:] == ['context'] else 'adapter')
if fault and stage == fault and ('build' in args or 'run' in args):
    raise SystemExit(17)
if 'push' in args:
    Path(args[args.index('--digestfile') + 1]).write_text('sha256:' + 'd' * 64)
if 'inspect' in args:
    print(123)
""")
    podman.chmod(0o755)
    for name in ("skopeo", "sleep"):
        path = binary / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    log = tmp_path / "commands"
    environment = {
        **os.environ,
        "PATH": f"{binary}:{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path),
        "KEY": KEY,
        "WORK": str(tmp_path),
        "GH_TOKEN": "",
        "GITHUB_ENV": str(tmp_path / "env"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "EVENT_NAME": event,
        "WORKFLOW_NAME": (
            "Publish recipe library release" if event == "push" else "Validate recipes"
        ),
        "COMMAND_LOG": str(log),
        "FAULT": fault,
    }
    result = None
    for name in (
        "Build the recipe image with the Spark build flags",
        "Push to every recipe that uses this image",
    ):
        script = _step(name).replace("sudo ", "env ")
        script = script.replace("${{ github.actor }}", "test").replace(
            "${{ github.repository }}", "o/repo"
        )
        result = subprocess.run(
            ["bash", "-c", script],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode:
            break
        for line in (tmp_path / "env").read_text().splitlines():
            name, value = line.split("=", 1)
            environment[name] = value
    assert result is not None
    assert (result.returncode == 0) == (not fault), result.stderr
    commands = log.read_text()
    assert ("push --digestfile" in commands) == (not fault)
    if not fault:
        prefix = "candidate" if event == "pull_request" else "build"
        assert f"docker://ghcr.io/o/recipe:{prefix}-{KEY}" in commands
        summary = (tmp_path / "summary").read_text()
        assert ("built at publish: no PR candidate" in summary) == (event == "push")


def test_bounded_proof_checks_the_planned_dockerfile_and_retries(tmp_path: Path):
    # Run the real step with a docker double: the first check fails as a
    # transient manifest lookup would; the Dockerfile comes from the plan.
    plan = tmp_path / "plan"
    (plan / KEY / "context").mkdir(parents=True)
    (plan / "plan.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "build_key": KEY,
                        "recipe_build_arguments": [
                            "--no-cache",
                            "--file",
                            "context/Dockerfile.spark",
                            "--jobs=1",
                        ],
                    }
                ]
            }
        )
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    (binary / "docker").write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        f'[ "$(wc -l < "{calls}")" -gt 1 ] || exit 1\n'
    )
    (binary / "sleep").write_text("#!/bin/sh\n")
    for tool in binary.iterdir():
        tool.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _step("Check the Dockerfile and its base images without building"),
        ],
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "KEY": KEY,
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    attempts = calls.read_text().splitlines()
    assert len(attempts) == 2
    check = attempts[-1]
    assert check.startswith("buildx build --check --platform linux/arm64 ")
    # Lint rules podman never applies cannot fail the gate; parsing and FROM
    # resolution still can.
    assert "--build-arg BUILDKIT_DOCKERFILE_CHECK=skip=all" in check
    assert "BUILDKIT_SYNTAX=docker/dockerfile:1.12@sha256:" in check
    assert check.endswith("--file context/Dockerfile.spark context")
    assert "--jobs" not in check and "--no-cache" not in check
