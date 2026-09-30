from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vllm-pr41797-base"
EXPECTED_PATCH_SHA256 = (
    "01ace84baef35fea8d337f5b44623286fd150367140ae383cef846cfdd028a83"
)
MOD_PATHS = (
    Path("adapters/mimo/tonyd2wild-v25-nvfp4kv-dual/mods/fix-mimo-v2-vllm"),
    Path(
        "adapters/mimo/tonyd2wild-v25-nvfp4kv-triple/mods-external/"
        "MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark@6dae0a60/fix-mimo-v2-vllm"
    ),
    Path("adapters/mimo/mia-vllm-dual/recipe/mods/fix-mimo-v2-vllm"),
)
BACKEND = Path("vllm/v1/attention/backends/triton_attn_diffkv.py")
PATCH_PATHS = (
    Path("vllm/model_executor/models/mimo_v2.py"),
    Path("vllm/v1/attention/backends/flash_attn_diffkv.py"),
    Path("vllm/v1/attention/backends/registry.py"),
    BACKEND,
    Path("vllm/v1/attention/ops/triton_unified_attention_diffkv.py"),
)


def create_package_initializers(site: Path) -> None:
    for parent in (
        site / "vllm",
        site / "vllm/v1",
        site / "vllm/v1/attention",
        site / "vllm/v1/attention/backends",
    ):
        parent.mkdir(parents=True, exist_ok=True)
        (parent / "__init__.py").touch(exist_ok=True)


def unpack_base_fixture(site: Path) -> None:
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    with tarfile.open(FIXTURE / "base-source.tar.gz", "r:gz") as archive:
        archive.extractall(site, filter="data")
    for relative, expected in manifest["files"].items():
        actual = hashlib.sha256((site / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"fixture source hash mismatch: {relative}")
    create_package_initializers(site)


def run_helper(helper: Path, site: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SITE_PACKAGES"] = str(site)
    env["PYTHONPATH"] = str(site)
    return subprocess.run(
        ["bash", str(helper)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(tempfile.gettempdir()),
    )


class MimoPr41797BuildPatchTests(unittest.TestCase):
    def test_all_three_build_contexts_vendor_the_same_immutable_patch(self) -> None:
        digests = set()
        helper_digests = set()
        for relative in MOD_PATHS:
            mod = ROOT / relative
            patch = mod / "pr41797-vllm.diff"
            helper = mod / "apply-pr41797.sh"
            metadata = mod / "pr41797-source.txt"
            self.assertTrue(helper.is_file())
            self.assertTrue(os.access(helper, os.X_OK))
            self.assertTrue(metadata.is_file())
            helper_digests.add(hashlib.sha256(helper.read_bytes()).hexdigest())
            self.assertIn(
                "ab10addb603ac17611d55aa88b64acbf1da71373", metadata.read_text()
            )
            digest = hashlib.sha256(patch.read_bytes()).hexdigest()
            digests.add(digest)
            self.assertEqual(digest, EXPECTED_PATCH_SHA256)
            self.assertNotIn(
                "patch-diff.githubusercontent.com", (mod / "run.sh").read_text()
            )
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(helper_digests), 1)

    def test_complete_postimage_skips_without_changing_any_patch_file(self) -> None:
        for relative in MOD_PATHS:
            with (
                self.subTest(context=relative.as_posix()),
                tempfile.TemporaryDirectory(prefix="vonk-pr41797-complete-") as temp,
            ):
                site = Path(temp) / "site"
                site.mkdir()
                unpack_base_fixture(site)
                helper = ROOT / relative / "apply-pr41797.sh"
                first = run_helper(helper, site)
                self.assertEqual(first.returncode, 0, first.stderr)
                postimage = {path: (site / path).read_bytes() for path in PATCH_PATHS}

                second = run_helper(helper, site)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertIn(
                    "Complete pinned PR #41797 patch is already applied",
                    second.stdout,
                )
                self.assertEqual(
                    postimage,
                    {path: (site / path).read_bytes() for path in PATCH_PATHS},
                )

    def test_backend_module_only_is_partial_and_fails_closed(self) -> None:
        for relative in MOD_PATHS:
            with (
                self.subTest(context=relative.as_posix()),
                tempfile.TemporaryDirectory(prefix="vonk-pr41797-partial-") as temp,
            ):
                temp_root = Path(temp)
                partial = temp_root / "partial-site"
                partial.mkdir()
                unpack_base_fixture(partial)

                complete = temp_root / "complete-site"
                complete.mkdir()
                unpack_base_fixture(complete)
                helper = ROOT / relative / "apply-pr41797.sh"
                applied = run_helper(helper, complete)
                self.assertEqual(applied.returncode, 0, applied.stderr)
                partial_backend = partial / BACKEND
                partial_backend.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(complete / BACKEND, partial_backend)
                before = {
                    path: (partial / path).read_bytes()
                    for path in PATCH_PATHS
                    if (partial / path).is_file()
                }

                result = run_helper(helper, partial)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("neither the exact patch base", result.stderr)
                self.assertEqual(
                    before,
                    {
                        path: (partial / path).read_bytes()
                        for path in PATCH_PATHS
                        if (partial / path).is_file()
                    },
                )

    def test_missing_backend_applies_exact_patch_to_pinned_base_preimages(self) -> None:
        for relative in MOD_PATHS:
            with (
                self.subTest(context=relative.as_posix()),
                tempfile.TemporaryDirectory(prefix="vonk-pr41797-missing-") as temp,
            ):
                site = Path(temp) / "site"
                site.mkdir()
                unpack_base_fixture(site)
                result = run_helper(ROOT / relative / "apply-pr41797.sh", site)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((site / BACKEND).is_file())
                registry = (site / "vllm/v1/attention/backends/registry.py").read_text()
                model = (site / "vllm/model_executor/models/mimo_v2.py").read_text()
                self.assertIn("TRITON_ATTN_DIFFKV", registry)
                self.assertIn(
                    "backend_enum = AttentionBackendEnum.TRITON_ATTN_DIFFKV", model
                )
                self.assertIn("Auto-pick FA-DiffKV", model)

    def test_incompatible_patch_anchor_fails_before_applying_any_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vonk-pr41797-anchor-") as temp:
            site = Path(temp) / "site"
            site.mkdir()
            unpack_base_fixture(site)
            registry_path = site / "vllm/v1/attention/backends/registry.py"
            registry = registry_path.read_text()
            registry_path.write_text(
                registry.replace('TRITON_ATTN = "', 'TRITON_ATTN = "broken-', 1)
            )
            result = run_helper(ROOT / MOD_PATHS[0] / "apply-pr41797.sh", site)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("neither the exact patch base", result.stderr)
            self.assertFalse((site / BACKEND).exists())
            self.assertNotIn("TRITON_ATTN_DIFFKV", registry_path.read_text())

    def test_changed_vendored_patch_bytes_fail_before_applying(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vonk-pr41797-bytes-") as temp:
            temp_root = Path(temp)
            mod = temp_root / "mod"
            shutil.copytree(ROOT / MOD_PATHS[0], mod)
            patch = mod / "pr41797-vllm.diff"
            patch.write_bytes(patch.read_bytes() + b"# modified bytes\n")
            site = temp_root / "site"
            site.mkdir()
            unpack_base_fixture(site)
            result = run_helper(mod / "apply-pr41797.sh", site)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertFalse((site / BACKEND).exists())


if __name__ == "__main__":
    unittest.main()
