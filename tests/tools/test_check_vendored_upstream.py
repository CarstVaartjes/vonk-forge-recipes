from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from repo_root import ROOT

from recipe_tooling import vendored_upstream as checker

COMMIT = "a" * 40
LABELS = (
    'LABEL org.opencontainers.image.source="https://github.com/o/r" \\\n'
    f'      org.opencontainers.image.revision="{COMMIT}"\n'
)


def sha(data: bytes) -> str:
    return checker.git_blob_sha(data)


class VendoredUpstreamTest(unittest.TestCase):
    def _run(
        self,
        upstream: dict[str, bytes],
        local: dict[str, bytes],
        manifest: dict[str, object] | None = None,
        dockerfile: str = "FROM scratch\n" + LABELS,
        complete: bool = True,
        at: str | None = None,
    ):
        tree = checker.Tree({p: sha(b) for p, b in upstream.items()}, complete)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapters/x"
            adapter.mkdir(parents=True)
            (adapter / "Dockerfile").write_text(dockerfile)
            for name, data in local.items():
                target = adapter / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if manifest is not None:
                (adapter / "vendored-upstream.json").write_text(json.dumps(manifest))
            return checker.check_adapter(adapter, root, lambda _r, _c: tree, at=at)

    def test_an_unreadable_upstream_is_a_note_not_a_problem(self) -> None:
        def down(_repo: str, _commit: str) -> Any:
            raise OSError("connection reset")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapters/x"
            adapter.mkdir(parents=True)
            (adapter / "Dockerfile").write_text("FROM scratch\n" + LABELS)
            report = checker.check_adapter(adapter, root, down, lambda *_: None)
            gone = checker.check_adapter(adapter, root, down, lambda *_: False)
        assert report.problems == []
        assert report.note
        assert gone.problems  # GitHub answers "not there": ours to fix

    def test_identical_vendored_file_passes(self) -> None:
        report = self._run(
            {"p/a.py": b"1"},
            {"p/a.py": b"1", "wrapper.sh": b"w"},
            {"ours": ["wrapper.sh"]},
        )
        assert report.problems == []
        assert report.checked == 1

    def test_drifted_file_names_adapter_and_file(self) -> None:
        report = self._run({"a.py": b"1"}, {"a.py": b"2"}, {"ours": []})
        assert len(report.problems) == 1
        assert "adapters/x: a.py: differs from upstream a.py" in report.problems[0]

    def test_messages_name_the_commit_the_tree_is(self) -> None:
        report = self._run({"a.py": b"1"}, {"a.py": b"2"}, {"ours": []}, at="c" * 40)
        assert report.problems[0].endswith("at " + "c" * 12)
        assert "a" * 12 not in report.problems[0]

    def test_unlisted_platform_file_fails_in_vendoring_adapter(self) -> None:
        report = self._run({"a.py": b"1"}, {"a.py": b"1", "extra.py": b"x"})
        assert len(report.problems) == 1
        assert "extra.py" in report.problems[0]
        assert "not listed under 'ours'" in report.problems[0]

    def test_wrapper_only_adapter_is_reported_not_compared(self) -> None:
        report = self._run({"z.py": b"1"}, {"wrapper.sh": b"w"})
        assert report.problems == []
        assert "no upstream-derived files" in report.note

    def test_identical_but_moved_file_needs_rename_entry(self) -> None:
        report = self._run({"files/a.py": b"1"}, {"a.py": b"1"}, {"ours": []})
        assert "add it to 'renamed'" in report.problems[0]
        fixed = self._run(
            {"files/a.py": b"1"}, {"a.py": b"1"}, {"renamed": {"a.py": "files/a.py"}}
        )
        assert fixed.problems == []

    def test_renamed_and_drifted_file_fails(self) -> None:
        report = self._run(
            {"files/unique_name.py": b"1"}, {"unique_name.py": b"2"}, {"ours": []}
        )
        assert "renamed and drifted" in report.problems[0]

    def test_renamed_dirs_and_provenance_spelling(self) -> None:
        report = self._run(
            {"image-patch/a.py": b"1", "LICENSE": b"L"},
            {"overlays/a.py": b"1", "LICENSE.upstream": b"L"},
            {"renamed_dirs": {"overlays/": "image-patch/"}},
        )
        assert report.problems == []
        assert report.checked == 2

    def test_stale_manifest_entry_fails(self) -> None:
        report = self._run({"a.py": b"1"}, {"a.py": b"1"}, {"ours": ["gone.py"]})
        assert "'gone.py', which is not in the adapter" in report.problems[0]

    def test_adapter_without_pin_is_skipped_with_a_note(self) -> None:
        report = self._run({}, {"a.py": b"1"}, dockerfile="FROM scratch\n")
        assert report.problems == []
        assert "skipped" in report.note

    def test_non_github_source_is_skipped(self) -> None:
        dockerfile = (
            'LABEL org.opencontainers.image.source="https://huggingface.co/o/r" '
            f'org.opencontainers.image.revision="{COMMIT}"\n'
        )
        report = self._run({}, {"a.py": b"1"}, dockerfile=dockerfile)
        assert "skipped" in report.note

    def test_manifest_source_overrides_labels(self) -> None:
        report = self._run(
            {"a.py": b"1"},
            {"a.py": b"1"},
            {"source": {"repo": "o/other", "commit": "b" * 40}},
            dockerfile="FROM scratch\n",
        )
        assert report.source == "o/other@" + "b" * 12
        assert report.problems == []

    def test_ours_directory_prefix(self) -> None:
        report = self._run(
            {"a.py": b"1"}, {"a.py": b"1", "mine/x.py": b"x"}, {"ours": ["mine/"]}
        )
        assert report.problems == []

    def _run_also(
        self,
        other: dict[str, bytes],
        local: dict[str, bytes],
        also: object | None = None,
    ):
        trees = {
            ("o/r", COMMIT): checker.Tree({"a.py": sha(b"1")}, True),
            ("o/other", "b" * 40): checker.Tree(
                {p: sha(b) for p, b in other.items()}, True
            ),
        }
        manifest = {
            "ours": [],
            "also": [{"repo": "o/other", "commit": "b" * 40, "files": ["other/"]}]
            if also is None
            else also,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapters/x"
            adapter.mkdir(parents=True)
            (adapter / "Dockerfile").write_text("FROM scratch\n" + LABELS)
            (adapter / "vendored-upstream.json").write_text(json.dumps(manifest))
            for name, data in {"a.py": b"1", **local}.items():
                target = adapter / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return checker.check_adapter(
                adapter, root, lambda repo, commit: trees[(repo, commit)]
            )

    def test_also_source_files_match_their_own_upstream(self) -> None:
        report = self._run_also({"other/b.py": b"2"}, {"other/b.py": b"2"})
        assert report.problems == []
        assert report.checked == 2

    def test_also_source_drift_and_missing_files_fail(self) -> None:
        drifted = self._run_also({"other/b.py": b"2"}, {"other/b.py": b"3"})
        assert (
            "other/b.py: differs from upstream o/other other/b.py at " + "b" * 12
            in drifted.problems[0]
        )
        missing = self._run_also({}, {"other/b.py": b"2"})
        assert "differs from upstream o/other" in missing.problems[0]

    def test_also_prefix_without_files_fails(self) -> None:
        report = self._run_also({}, {})
        assert "'other/', which is not in the adapter" in report.problems[0]

    def test_malformed_also_entries_are_guard_problems(self) -> None:
        good = {"repo": "o/other", "commit": "b" * 40, "files": ["other/"]}
        cases = {
            "not a list": ({"repo": "o/other"}, "must be a list"),
            "entry not an object": (["o/other"], "entry 0 must be an object, not str"),
            "missing repo": ([{**good, "repo": None}], "needs a repo string"),
            "repo wrong type": ([{**good, "repo": 3}], "needs a repo string"),
            "missing commit": ([{"repo": "o/other", "files": ["x/"]}], "40-hex commit"),
            "short commit": ([{**good, "commit": "abc"}], "40-hex commit"),
            "missing files": ([{"repo": "o/other", "commit": "b" * 40}], "needs files"),
            "files not a list": ([{**good, "files": "other/"}], "needs files"),
            "files empty": ([{**good, "files": []}], "needs files"),
            "files item wrong type": ([{**good, "files": [1]}], "needs files"),
        }
        for name, (also, expected) in cases.items():
            with self.subTest(name):
                report = self._run_also({}, {}, also=also)
                assert len(report.problems) == 1
                assert expected in report.problems[0]
                assert "vendored-upstream.json also" in report.problems[0]


class GhApiRetryTest(unittest.TestCase):
    """.github/scripts/gh-api-retry with a stub gh (and a sleep that returns at once)."""

    SCRIPT = ROOT / ".github/scripts/gh-api-retry"

    def _run(self, gh: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            for name, body in (("gh", gh), ("sleep", "exit 0")):
                stub = Path(tmp) / name
                stub.write_text(f"#!/bin/bash\n{body}\n")
                stub.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{tmp}:{os.environ['PATH']}",
                "COUNT": f"{tmp}/n",
            }
            return subprocess.run(
                ["bash", str(self.SCRIPT), "x"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

    def test_a_failed_attempt_leaves_only_the_success_body(self) -> None:
        done = self._run(
            'n=$(cat "$COUNT" 2>/dev/null || echo 0); echo $((n+1)) >"$COUNT"; '
            '[ "$n" -ge 1 ] && { echo ahead; exit 0; }; '
            'echo "error body" ; echo "gh: Bad Gateway (HTTP 502)" >&2; exit 1'
        )
        assert (done.returncode, done.stdout) == (0, "ahead\n")

    def test_a_404_fails_at_once_and_an_outage_exits_75(self) -> None:
        assert self._run('echo "gh: Not Found (HTTP 404)" >&2; exit 1').returncode == 1
        assert self._run('echo "timeout" >&2; exit 1').returncode == 75


if __name__ == "__main__":
    unittest.main()
