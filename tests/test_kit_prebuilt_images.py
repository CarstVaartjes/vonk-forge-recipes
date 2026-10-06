"""A kit that provides a prebuilt image is run as published (tools/kit-pins images rule)."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any

from test_kit_pins import KIT, kit_pins

COMMIT = "a" * 40
IMAGE = "ghcr.io/creator/kit-image"
DIGEST = "sha256:" + "b" * 64


def archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"kit-{COMMIT[:7]}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def scan(files: dict[str, str], scope: str = "") -> list[str]:
    return [f["reference"] for f in kit_pins.scan_archive(archive(files), scope)]


class ScanTests(unittest.TestCase):
    def test_finds_the_images_a_kit_runs_or_pulls(self) -> None:
        found = scan(
            {
                "Dockerfile": f"FROM {IMAGE}@{DIGEST}\n",
                "scripts/config.sh": 'IMG="${IMG:-ghcr.io/creator/other:v1}"\n',
                "start.sh": "docker run --rm --gpus all -v /a/b:/c creator/hub-image:tag python\n",
                "compose.yaml": "services:\n  x:\n    image: quay.io/creator/third:1\n",
                "README.md": "docker pull ghcr.io/creator/readme:latest\nsee ghcr.io/creator/prose:1\n",
            }
        )
        self.assertEqual(
            sorted(found),
            sorted(
                [
                    f"{IMAGE}@{DIGEST}",
                    "ghcr.io/creator/other:v1",
                    "creator/hub-image:tag",
                    "quay.io/creator/third:1",
                    "ghcr.io/creator/readme:latest",
                ]
            ),
        )

    def test_engines_bases_and_documentation_are_not_creator_images(self) -> None:
        found = scan(
            {
                "Dockerfile": "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n",
                "a/Dockerfile": "FROM nvcr.io/nvidia/pytorch:26.07-py3\n",
                "b/Dockerfile": "FROM lmsysorg/sglang:v1\n",
                "docs/run.sh": f"docker run {IMAGE}:x\n",
                "tests/run.sh": f"docker run {IMAGE}:x\n",
                "run.sh": "# docker run ghcr.io/creator/commented:x\n",
                "paths.sh": "docker run -v models/foo:/data -e HF=org/model:rev img\n",
                "model.sh": "MODEL_IMAGE_ID=Qwen/Qwen-Image:file\n",
            }
        )
        self.assertEqual(found, [], found)

    def test_a_monorepo_kit_is_scoped_to_the_cited_directory(self) -> None:
        files = {
            "one/start.sh": f"docker run {IMAGE}:a\n",
            "two/start.sh": "docker run ghcr.io/creator/two:a\n",
        }
        self.assertEqual(scan(files, "two/"), ["ghcr.io/creator/two:a"])
        recipe = {
            "provenance": {
                "source_reference": f"https://github.com/{KIT}/blob/{COMMIT}/two/README.md"
            }
        }
        self.assertEqual(kit_pins.kit_scope(recipe), "two/")
        recipe["provenance"]["source_reference"] = (
            f"https://github.com/{KIT}/tree/{COMMIT}"
        )
        self.assertEqual(kit_pins.kit_scope(recipe), "")


def recipe(root: Path, dockerfile: str, base: str | None = None) -> dict[str, Any]:
    adapter = root / "adapters/x"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "Dockerfile").write_text(dockerfile)
    build: dict[str, Any] = {"context": {"path": "adapters/x"}}
    if base:
        build["base_image"] = {"repository": base, "digest": "b" * 64}
    return {
        "identity": {"slug": "x"},
        "provenance": {"source_reference": f"https://github.com/{KIT}/tree/{COMMIT}"},
        "execution": {"build": build},
    }


class RuleTests(unittest.TestCase):
    def problems(
        self,
        dockerfile: str,
        referenced: list[str],
        exceptions: dict[str, str] | None = None,
        base: str | None = None,
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = recipe(root, dockerfile, base)
            found = [
                {
                    "repo": kit_pins.image_name(r.split(":")[0].split("@")[0]),
                    "reference": r,
                    "file": "start.sh",
                    "line": 1,
                }
                for r in referenced
            ]
            return kit_pins.image_problems(root, "x", document, found, exceptions or {})

    def test_a_recipe_that_builds_from_the_kits_image_passes(self) -> None:
        self.assertEqual(
            self.problems(f"FROM {IMAGE}@{DIGEST}\n", [f"{IMAGE}:tag"]), []
        )

    def test_the_recorded_base_image_counts_as_running_it(self) -> None:
        self.assertEqual(
            self.problems("FROM scratch\n", [f"{IMAGE}:tag"], base=IMAGE), []
        )

    def test_a_self_build_beside_a_published_image_is_a_problem(self) -> None:
        problems = self.problems(
            "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n", [f"{IMAGE}:tag"]
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("kit provides prebuilt image", problems[0])
        self.assertIn("kit-image-exceptions.json", problems[0])

    def test_a_reasoned_exception_allows_it(self) -> None:
        self.assertEqual(
            self.problems(
                "FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n",
                [f"{IMAGE}:tag"],
                {"x": "the kit's image is another engine lane than this recipe"},
            ),
            [],
        )

    def test_an_exception_needs_a_reason(self) -> None:
        self.assertIn(
            "no reason",
            self.problems("FROM scratch\n", [f"{IMAGE}:tag"], {"x": " "})[0],
        )

    def test_a_kit_without_an_image_needs_nothing(self) -> None:
        self.assertEqual(
            self.problems("FROM docker.io/vllm/vllm-openai@" + DIGEST + "\n", []), []
        )


class CheckTests(unittest.TestCase):
    def test_checks_every_recipe_once_per_kit_and_names_unknown_exceptions(
        self,
    ) -> None:
        calls: list[tuple[str, str, str]] = []

        def images(repo: str, commit: str, scope: str) -> list[dict[str, Any]]:
            calls.append((repo, commit, scope))
            return [
                {
                    "repo": IMAGE,
                    "reference": f"{IMAGE}:t",
                    "file": "start.sh",
                    "line": 3,
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = recipe(root, f"FROM {IMAGE}@{DIGEST}\n")
            second = json.loads(json.dumps(first))
            second["identity"]["slug"] = "y"
            second["execution"]["build"]["context"]["path"] = "adapters/y"
            (root / "adapters/y").mkdir()
            (root / "adapters/y/Dockerfile").write_text("FROM scratch\n")
            (root / kit_pins.EXCEPTIONS).write_text(
                json.dumps({"gone": "reason", "x": "was needed once"})
            )
            notices: list[str] = []
            found = kit_pins.check_images(
                root,
                {"x": first, "y": second},
                kit_pins.Upstream(kit_pins.GitHub(lambda path: {}), images=images),
                notice=notices.append,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(found), 2, found)
        self.assertEqual(
            notices,
            [
                "remove the kit-image-exceptions.json entry of x: it runs the kit's image ghcr.io/creator/kit-image"
            ],
        )
        self.assertTrue(any("'gone' is not a recipe" in p for p in found), found)
        self.assertTrue(any(p.startswith("y: the kit provides") for p in found), found)


if __name__ == "__main__":
    unittest.main()
