"""Fault injection: a draft stranded before undrafting never blocks publication.

The publish steps run as written against a fake ``gh`` that keeps GitHub's
release semantics: the tags endpoint never returns a draft, and a draft's
tag is created only when it is undrafted.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "owner/recipes"
TAG = "v3.0.0"
SHA = "a" * 40
OLD_SHA = "b" * 40

FAKE_GH = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

state_path = Path(os.environ["FAKE_GH_STATE"])
state = json.loads(state_path.read_text())
args = sys.argv[1:]
prefix = "repos/" + os.environ["GITHUB_REPOSITORY"] + "/releases"


def save():
    state_path.write_text(json.dumps(state))


def option(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def by_tag(tag):
    # gh resolves a tag to its published release, else falls back to a draft.
    published = [r for r in state["releases"] if r["tag_name"] == tag and not r["draft"]]
    drafts = [r for r in state["releases"] if r["tag_name"] == tag and r["draft"]]
    return (published or drafts or [None])[0]


if args[0] == "api":
    path = next(a for a in args[1:] if a.startswith("repos/"))
    if option("--method") == "DELETE":
        release_id = int(path.rsplit("/", 1)[1])
        state["releases"] = [r for r in state["releases"] if r["id"] != release_id]
        save()
        sys.exit(0)
    if path.startswith(prefix + "/tags/"):
        tag = path.rsplit("/", 1)[1]
        found = [r for r in state["releases"] if r["tag_name"] == tag and not r["draft"]]
        if not found:
            print("HTTP 404: Not Found", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(found[0]))
        sys.exit(0)
    if path == prefix:
        payload = json.dumps(state["releases"])
        jq = option("--jq")
        if jq is None:
            print(payload)
        else:
            sys.stdout.write(
                subprocess.run(["jq", "-r", jq], input=payload, text=True,
                               check=True, capture_output=True).stdout
            )
        sys.exit(0)
elif args[:2] == ["release", "create"]:
    tag = args[2]
    if by_tag(tag) is not None and not by_tag(tag)["draft"]:
        print("a release with the same tag name already exists", file=sys.stderr)
        sys.exit(1)
    state["next_id"] += 1
    state["releases"].append({
        "id": state["next_id"], "tag_name": tag, "draft": "--draft" in args,
        "target_commitish": option("--target"), "name": option("--title"),
        "assets": [Path(args[-1]).name],
    })
    save()
    sys.exit(0)
elif args[:2] == ["release", "edit"]:
    release = by_tag(args[2])
    if release is None:
        print("release not found", file=sys.stderr)
        sys.exit(1)
    if "--draft=false" in args:
        release["draft"] = False
        state["tags"].setdefault(release["tag_name"], release["target_commitish"])
    if "--title" in args:
        release["name"] = option("--title")
    save()
    sys.exit(0)
print("unexpected gh call: " + " ".join(args), file=sys.stderr)
sys.exit(2)
"""

# Stands in for the release Python: builds a bundle file, and the asset sync
# (already covered by its own tests) succeeds without touching the release.
FAKE_PYTHON = """#!/usr/bin/env bash
if [ "$1" = tools/build-release-bundle ]; then : > "$3"; fi
"""


def step_script(name: str) -> str:
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    step = workflow.split(f"      - name: {name}\n")[1]
    return textwrap.dedent(
        # The step ends at the next step or the next job.
        re.split(r"\n      - name:|\n\n  \S", step.split("        run: |\n")[1])[0]
    )


def run_step(name: str, tmp_path: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        ["bash", "-c", step_script(name)],
        cwd=ROOT,
        env={**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", **env},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def publish(tmp_path: Path, releases: list[dict[str, object]]) -> dict[str, Any]:
    (tmp_path / "bin").mkdir()
    for name, body in (("gh", FAKE_GH), ("release-python", FAKE_PYTHON)):
        (tmp_path / "bin" / name).write_text(body)
        (tmp_path / "bin" / name).chmod(0o755)
    state = tmp_path / "state.json"
    tags = {r["tag_name"]: r["target_commitish"] for r in releases if not r["draft"]}
    state.write_text(json.dumps({"releases": releases, "tags": tags, "next_id": 10}))
    (tmp_path / "release").mkdir()
    (tmp_path / "attestation.json").write_text("{}")
    output = tmp_path / "output"
    output.write_text("")
    env = {
        "FAKE_GH_STATE": str(state),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_SHA": SHA,
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(tmp_path),
        "RELEASE_TAG": TAG,
        "RELEASE_TEST_PYTHON": "/usr/bin/false",
    }
    run_step("Compare with the published release", tmp_path, env)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    run_step(
        "Publish or update the contract-version release",
        tmp_path,
        {
            **env,
            "RELEASE_TEST_PYTHON": str(tmp_path / "bin" / "release-python"),
            "RELEASE_EXISTS": outputs["exists"],
            "UPDATED_AT": "2026-11-02T00:00:00Z",
            "ATTESTATION_BUNDLE": str(tmp_path / "attestation.json"),
        },
    )
    return json.loads(state.read_text())


def draft(release_id: int) -> dict[str, object]:
    return {
        "id": release_id,
        "tag_name": TAG,
        "draft": True,
        "target_commitish": OLD_SHA,
        "name": "stranded",
        "assets": [],
    }


@pytest.mark.skipif(shutil.which("jq") is None, reason="gh's --jq is emulated with jq")
@pytest.mark.parametrize("stranded", [[], [draft(1)], [draft(1), draft(2)]])
def test_a_stranded_draft_is_replaced_by_a_published_release(
    tmp_path: Path, stranded: list[dict[str, object]]
) -> None:
    other = {**draft(3), "tag_name": "v2.8.1"}
    state = publish(tmp_path, [*stranded, other])
    releases = [r for r in state["releases"] if r["tag_name"] == TAG]
    assert len(releases) == 1
    assert releases[0]["draft"] is False
    assert releases[0]["target_commitish"] == SHA
    assert releases[0]["assets"] == ["recipe-library.tar"]
    assert state["tags"] == {TAG: SHA}
    assert other in state["releases"]


def test_a_published_release_is_updated_in_place(tmp_path: Path) -> None:
    published = {**draft(1), "draft": False, "name": "old"}
    state = publish(tmp_path, [published])
    assert [r["id"] for r in state["releases"]] == [1]
    assert state["releases"][0]["name"].startswith(f"Vonk Forge recipes {TAG}")
    assert state["tags"] == {TAG: OLD_SHA}
