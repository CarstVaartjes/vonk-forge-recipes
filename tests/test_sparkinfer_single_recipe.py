from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/deepseek/sparkinfer-single"
MODEL_PATH = ROOT / "model-versions/deepseek-v4-flash-0731-sparkinfer-exl3-k216.json"
RECIPE_PATH = ROOT / "recipes/deepseek-v4-flash-0731-sparkinfer-single.json"
RELEASE_PATH = ROOT / "recipe-releases/deepseek-v4-flash-0731-sparkinfer-single.json"
RUNTIME_PATH = ROOT / "runtime-distributions/sparkinfer-dsv4-single.json"
MODEL_REVISION = "ce5ff0f1efb2e184aafc759d281bfae47d3a359c"
EXECUTABLE_PAYLOAD_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
RUNTIME_REVISION = "590d2172394dd83c1f36ff29f0dc9ec6032ea9e2"
IMAGE_DIGEST = "2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"


def _document(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    payload = json.dumps(
        _document(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class SparkInferSingleRecipeTests(unittest.TestCase):
    def test_complete_immutable_authority_closure(self) -> None:
        recipe = _document(RECIPE_PATH)
        model = _document(MODEL_PATH)
        runtime = _document(RUNTIME_PATH)

        self.assertEqual(
            recipe["model"]["content_sha256"], _canonical_digest(MODEL_PATH)
        )
        self.assertEqual(
            recipe["runtime"]["distribution"]["content_sha256"],
            _canonical_digest(RUNTIME_PATH),
        )
        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        self.assertEqual(runtime["source"]["revision"], RUNTIME_REVISION)
        self.assertEqual(
            runtime["image"],
            f"ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:{IMAGE_DIGEST}",
        )
        self.assertEqual(
            runtime["source"]["archive_sha256"],
            "aa5dca81afd568a9b2f8b492387462637cfe9ed8a9d76eaa0e9afa015e8bebb8",
        )
        self.assertEqual(runtime["image_manifest"]["digest"], IMAGE_DIGEST)
        self.assertEqual(runtime["image_manifest"]["size"], 12691)
        self.assertEqual(
            runtime["image_manifest"]["compressed_layers_bytes"], 11125732014
        )
        self.assertEqual(
            runtime["build"], {"network_hosts": [], "offline_after_installation": True}
        )

        artifacts = model["artifacts"]
        self.assertEqual(len(artifacts), 190)
        self.assertEqual(len({item["path"] for item in artifacts}), len(artifacts))
        self.assertEqual(
            sum(item["download_bytes"] for item in artifacts),
            model["sizes"]["download_bytes"],
        )
        self.assertEqual(
            sum(item["installed_bytes"] for item in artifacts),
            model["sizes"]["installed_bytes"],
        )
        self.assertTrue(all(item["revision"] == MODEL_REVISION for item in artifacts))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in artifacts))

    def test_adapter_is_offline_and_uses_the_published_launch_path(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        wrapper = (ADAPTER_ROOT / "vllm-wrapper.sh").read_text(encoding="utf-8")
        recipe_text = RECIPE_PATH.read_text(encoding="utf-8")
        release_text = RELEASE_PATH.read_text(encoding="utf-8")

        self.assertIn(f"@sha256:{IMAGE_DIGEST}", dockerfile)
        self.assertIn(
            f'org.opencontainers.image.revision="{RUNTIME_REVISION}"', dockerfile
        )
        self.assertIn("ENTRYPOINT []", dockerfile)
        self.assertIn("/opt/recipe/scripts/coalesce_rank_sliced_exl3.py", wrapper)
        self.assertIn("/opt/recipe/scripts/verify_tp1_manifest.py", wrapper)
        self.assertIn("/opt/recipe/scripts/build_dspark_draft.py", wrapper)
        self.assertIn("/opt/recipe/scripts/selftest.py", wrapper)
        self.assertIn("exec /opt/vllm/serve-ds4-flash.sh", wrapper)
        self.assertIn(
            f"readonly executable_payload_revision={EXECUTABLE_PAYLOAD_REVISION}",
            wrapper,
        )
        self.assertIn(
            "readonly state_root=/outputs/${executable_payload_revision}", wrapper
        )

        for forbidden in (
            "snapshot_download",
            "huggingface_hub",
            "curl ",
            "wget ",
            "git clone",
        ):
            self.assertNotIn(forbidden, wrapper)
        for forbidden in (
            "metadata-only",
            "non-executable",
            "integration-required",
            "/bin/false",
            "exit 78",
        ):
            self.assertNotIn(forbidden, recipe_text)
        for forbidden in (
            "non-executable",
            "integration-required",
            "/bin/false",
            "exit 78",
        ):
            self.assertNotIn(forbidden, release_text)

        subprocess.run(
            ["bash", "-n", str(ADAPTER_ROOT / "vllm-wrapper.sh")],
            check=True,
        )

    def test_source_bundle_and_release_digests_match(self) -> None:
        recipe = _document(RECIPE_PATH)
        release = _document(RELEASE_PATH)
        index_tool = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
        archive, _files, digest = index_tool["source_bundle"](ADAPTER_ROOT)
        context = recipe["build"]["context"]
        self.assertEqual(context["sha256"], digest)
        self.assertEqual(context["expected_bytes"], len(archive))
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"],
            _canonical_digest(RECIPE_PATH),
        )


if __name__ == "__main__":
    unittest.main()
