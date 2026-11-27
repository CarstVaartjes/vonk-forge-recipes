"""PR builds and release promotion must use one executable image proof path."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from repo_root import ROOT

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
        "\n  memory-estimates:", 1
    )[0]
    # GitHub validates nested workflow permissions even for skipped PR jobs.
    assert "packages: write" in producer
    assert "actions: write" in producer
    gate = validate.split("\n  catalog-index:\n", 1)[1].split(
        "\n  producer-contracts:", 1
    )[0]
    assert "needs: [images, catalog]" in gate
    assert "always()" in gate
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
    # separate remote-module smoke; the recipe image is the pushed image.
    assert build.index('"${recipe_args[@]}"') < build.index("remote-code-smoke.py")
    assert "adapter_args" not in build


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_workflow_plan_files_exclude_catalog_progress(tmp_path: Path, event: str):
    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    for tool in ("build-plan", "prepare-remote-code-smoke"):
        shutil.copyfile(ROOT / "tools" / tool, workspace / "tools" / tool)
    for tree in ("src", "policy"):
        shutil.copytree(
            ROOT / tree, workspace / tree, ignore=shutil.ignore_patterns("__pycache__")
        )
    binary = tmp_path / "bin"
    binary.mkdir()
    # Replace the CI edge (the commit and its time) and environment setup only.
    # Keep noisy catalog stdout and run the real planner and plan consumer.
    commit = binary / "git"
    commit.write_text("#!/bin/sh\necho 0\n")
    commit.chmod(0o755)
    uv = binary / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
script = args.index('python') + 1
if args[script].endswith('/tools/build-catalog-index'):
    root = Path(args[script]).parents[1]
    root.mkdir(parents=True, exist_ok=True)
    (root / 'catalog-index.json').write_text(json.dumps({'source_commit': 'c', 'recipes': []}))
    print('wrote catalog-index.json for', root)
else:
    raise SystemExit(subprocess.run(
        [sys.executable, *args[script:]], check=False, timeout=10
    ).returncode)
"""
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", _step("Plan one image per distinct build key")],
        cwd=workspace,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(tmp_path),
            "EVENT_NAME": event,
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    plans = ["plan.json"] + (["base-plan.json"] if event == "pull_request" else [])
    for filename in plans:
        assert json.loads((tmp_path / filename).read_text()) == {
            "schema_version": 2,
            "source_commit": "c",
            "images": [],
            "skipped": [],
        }
    assert result.stdout.count("wrote catalog-index.json") == len(plans)


@pytest.mark.parametrize("catalog", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "skipped"])
def test_required_check_rejects_every_non_success_image_or_catalog_result(
    result: str, catalog: str
):
    workflow = (ROOT / ".github/workflows/validate.yml").read_text()
    gate = workflow.split("- name: Require PR image proof", 1)[1]
    script = gate.split("        run: ", 1)[1].splitlines()[0]
    checked = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "IMAGE_RESULT": result, "CATALOG_RESULT": catalog},
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (checked.returncode == 0) == (result == catalog == "success")


@pytest.mark.parametrize("event", ["pull_request", "push"])
@pytest.mark.parametrize("fault", ["", "native", "remote"])
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
stage = 'remote' if 'run' in args else 'native'
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
        "JOB_START": str(int(time.time())),
        "GH_TOKEN": "",
        "RECIPES_REVISION": "0" * 40,
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
            variable, value = line.split("=", 1)
            environment[variable] = value
    assert result is not None
    assert (result.returncode == 0) == (not fault), result.stderr
    commands = log.read_text()
    assert ("push --digestfile" in commands) == (not fault)
    if not fault:
        prefix = "candidate-v2" if event == "pull_request" else "build-v2"
        assert f"docker://ghcr.io/o/recipe:{prefix}-{KEY}" in commands
        assert f"ai.vonkforge.recipes-revision={'0' * 40}" in commands
        summary = (tmp_path / "summary").read_text()
        assert ("built at publish: no PR candidate" in summary) == (event == "push")
