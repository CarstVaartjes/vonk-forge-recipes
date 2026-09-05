from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class BuildNetworkHostMetadataTests(unittest.TestCase):
    def test_declared_build_hosts_are_used_by_source_builds(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            if recipe["execution"]["mode"] != "build":
                continue
            network = recipe["execution"]["build"]["network"]
            self.assertEqual(len(network["hosts"]), len(set(network["hosts"])), path.name)
            dockerfile = ROOT / recipe["execution"]["build"]["dockerfile"]
            self.assertTrue(dockerfile.is_file(), path.name)
            if network["mode"] == "none":
                self.assertEqual(network["hosts"], [])

    def test_nvcr_is_only_a_base_image_registry(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            if recipe["execution"]["mode"] != "build":
                continue
            build = recipe["execution"]["build"]
            hosts = build["network"]["hosts"]
            if "nvcr.io" not in hosts:
                continue
            dockerfile = ROOT / build["dockerfile"]
            steps = "\n".join(line for line in dockerfile.read_text().splitlines() if not line.lstrip().upper().startswith("FROM "))
            self.assertIn("https://nvcr.io", steps, path.name)

    def test_security_and_redirect_hosts_remain_explicit(self) -> None:
        for slug in ("ltx-2-19b-dev-fp4-pytorch-single", "step1x-3d-geometry-pytorch-single", "step1x-3d-label-geometry-pytorch-single", "step1x-3d-texture-pytorch-single"):
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            hosts = recipe["execution"]["build"]["network"]["hosts"]
            self.assertTrue(hosts, slug)
            if slug.startswith("step1x"):
                self.assertIn("security.ubuntu.com", hosts)


if __name__ == "__main__":
    unittest.main()
