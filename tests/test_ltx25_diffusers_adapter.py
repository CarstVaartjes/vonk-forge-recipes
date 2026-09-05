from __future__ import annotations

import hashlib
import json
import runpy
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "adapters/video/ltx25-diffusers"
ADAPTER_PATH = ADAPTER_ROOT / "run.py"
PREFLIGHT_PATH = ADAPTER_ROOT / "preflight.py"
MODEL_VERSION = ROOT / "models/ltx-2-5-22b-distilled-bf16-diffusers.json"
RECIPE = ROOT / "recipes/ltx-2-5-22b-distilled-bf16-diffusers-single.json"
RELEASE = ROOT / "recipe-releases/ltx-2-5-22b-distilled-bf16-diffusers-single.json"
MODEL_REVISION = "426936f8b22dc28e4def61e515478b0b7e4a53cc"
DIFFUSERS_REVISION = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(
        json.dumps(
            _document(path),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _adapter_module():
    module = types.ModuleType("ltx25_diffusers_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(  # noqa: S102 - load the adapter without importing its heavy dependencies.
        compile(ADAPTER_PATH.read_text(encoding="utf-8"), str(ADAPTER_PATH), "exec"),
        module.__dict__,
    )
    return module


def _preflight_module():
    module = types.ModuleType("ltx25_diffusers_preflight")
    module.__file__ = str(PREFLIGHT_PATH)
    exec(  # noqa: S102 - exercise the standalone preflight module in isolation.
        compile(PREFLIGHT_PATH.read_text(encoding="utf-8"), str(PREFLIGHT_PATH), "exec"),
        module.__dict__,
    )
    return module


class Ltx25CatalogTests(unittest.TestCase):
    def test_authorities_and_catalog_bindings_are_exact(self) -> None:
        model = _document(MODEL_VERSION)
        recipe = _document(RECIPE)
        release = _document(RELEASE)

        self.assertEqual(model["source"]["revision"], MODEL_REVISION)
        selection = recipe["models"][0]
        self.assertEqual([item["id"] for item in selection["files"]], ["primary-filtered-snapshot", "primary-filtered-snapshot-2"])
        self.assertTrue(model["license"]["operator_acceptance_required"])
        self.assertEqual(model["license"]["spdx"], "LicenseRef-LTX-2-Community")
        from vonk_forge_contracts import ModelDefinition
        model_digest = hashlib.sha256(json.dumps(ModelDefinition.model_validate(model).model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(selection["model"]["content_sha256"], model_digest)
        self.assertEqual(
            release["history"][0]["recipe_content_sha256"], _digest(RECIPE)
        )
        self.assertEqual(model["files"][0]["size_bytes"], 70_090_051_372)
        role = recipe["topology"]["roles"][0]
        disk = role["resources"]["disk"]
        self.assertGreaterEqual(disk["artifact_bytes"], model["files"][0]["size_bytes"])
        memory = role["resources"]["memory"]
        required = max(
            memory["startup_peak_bytes"],
            memory["steady_state_bytes"] + memory["runtime_growth_bytes"],
        ) + memory["system_reserve_bytes"]
        self.assertEqual(required, 128_000_000_000)
        input_contract = recipe["interfaces"][0]["input"]
        self.assertEqual(input_contract["max_bytes"], 81_920)
        input_slots = {slot["id"]: slot for slot in input_contract["slots"]}
        self.assertEqual(set(input_slots), {"prompt", "request"})
        self.assertEqual(input_slots["prompt"]["media_types"], ["text/plain"])
        self.assertEqual(input_slots["prompt"]["min_files"], 1)
        self.assertEqual(input_slots["request"]["media_types"], ["application/json"])
        self.assertEqual(input_slots["request"]["min_files"], 0)
        output = recipe["interfaces"][0]["output"]
        self.assertEqual(output["max_total_bytes"], 1024**3 + 1024**2)
        slots = {slot["id"]: slot for slot in output["slots"]}
        self.assertEqual(set(slots), {"video", "receipt"})
        self.assertEqual(slots["video"]["media_types"], ["video/mp4"])
        self.assertEqual(slots["receipt"]["media_types"], ["application/json"])
        self.assertEqual(slots["video"]["min_files"], 1)
        self.assertEqual(slots["receipt"]["min_files"], 1)

    def test_filtered_snapshot_matches_adapter_closure(self) -> None:
        adapter = _adapter_module()
        selected = "\n".join(item["file_id"] for item in _document(RECIPE)["models"][0]["files"])
        self.assertIn("filtered-snapshot", selected)
        for excluded in adapter.FORBIDDEN_PATHS:
            self.assertNotIn(excluded, selected)
        self.assertIn("filtered-snapshot", selected)

    def test_signed_source_bundle_matches_recipe(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        archive, _, digest = source_bundle(ADAPTER_ROOT)
        context = _document(RECIPE)["execution"]["build"]["context"]
        self.assertEqual(context["path"], "adapters/video/ltx25-diffusers")
        self.assertTrue(digest)

    def test_container_and_preflight_are_immutable_and_offline(self) -> None:
        dockerfile = (ADAPTER_ROOT / "Dockerfile").read_text(encoding="utf-8")
        readme = (ADAPTER_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(MODEL_REVISION, dockerfile)
        self.assertIn(DIFFUSERS_REVISION, dockerfile)
        self.assertIn("HF_HUB_OFFLINE=1", dockerfile)
        self.assertIn("TRANSFORMERS_OFFLINE=1", dockerfile)
        self.assertIn("a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2_x", readme)
        self.assertIn("Hugging Face read token", readme)
        self.assertIn("be75acae5c99b0fb16ed6cfbf8f731e5121a729bef112d20337699407e796451", readme)
        self.assertIn("505-byte", readme)
        self.assertIn("generic managed-artifact", readme)
        self.assertIn("error; rerun this preflight", readme)
        self.assertEqual(_adapter_module().MODEL_ROOT, Path("/models/target"))
        self.assertIn("NVFP4 is intentionally not an install profile", readme)
        self.assertIn("SM121", readme)


class Ltx25PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = _preflight_module()

    def test_license_acknowledgement_is_bound_to_exact_pinned_text(self) -> None:
        self.preflight._verify_license_acknowledgement(
            self.preflight.LICENSE_SHA256
        )
        with self.assertRaisesRegex(
            self.preflight.PreflightError, "pinned LTX-2 Community License"
        ):
            self.preflight._verify_license_acknowledgement("0" * 64)

    def test_token_file_must_be_private_and_is_never_returned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("hf_private_test_value\n", encoding="ascii")
            path.chmod(0o600)
            self.assertEqual(
                self.preflight._token_from_file(path), "hf_private_test_value"
            )
            path.chmod(0o644)
            with self.assertRaisesRegex(
                self.preflight.PreflightError, "permissions.*0600"
            ):
                self.preflight._token_from_file(path)

    def test_access_probe_is_bounded_and_verifies_immutable_blob(self) -> None:
        payload = b"probe"
        blob_sha1 = hashlib.sha1(b"blob 5\0" + payload).hexdigest()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def geturl() -> str:
                return self.preflight.PROBE_URL

            @staticmethod
            def read(limit: int) -> bytes:
                self.assertEqual(limit, 6)
                return payload

        opener = mock.Mock()
        opener.open.return_value = Response()
        with mock.patch.object(self.preflight, "PROBE_BYTES", 5), mock.patch.object(
            self.preflight, "PROBE_GIT_BLOB_SHA1", blob_sha1
        ):
            digest = self.preflight._verify_gated_access(
                "hf_private_test_value", opener=opener
            )
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer hf_private_test_value")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)

    def test_access_failures_are_clear_and_do_not_echo_token(self) -> None:
        opener = mock.Mock()
        response_error = urllib.error.HTTPError(
            self.preflight.PROBE_URL, 401, "Unauthorized", {}, None
        )
        opener.open.side_effect = response_error
        try:
            with self.assertRaises(self.preflight.PreflightError) as captured:
                self.preflight._verify_gated_access(
                    "hf_private_test_value", opener=opener
                )
        finally:
            response_error.close()
        message = str(captured.exception)
        self.assertIn("401 GatedRepo", message)
        self.assertIn("No model weights were downloaded", message)
        self.assertNotIn("hf_private_test_value", message)


class Ltx25AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = _adapter_module()

    def _write_closure(self, root: Path) -> None:
        for relative in self.adapter.REQUIRED_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"weights")
        for relative, shards in self.adapter.EXPECTED_SHARDS.items():
            weight_map = {
                f"layer.{index}": shard for index, shard in enumerate(sorted(shards))
            }
            (root / relative).write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )

    def test_import_does_not_require_heavy_runtime(self) -> None:
        self.assertEqual(self.adapter.DEFAULT_PROFILE, "bf16-model-offload")
        self.assertNotIn("diffusers", self.adapter.__dict__)
        self.assertNotIn("torch", self.adapter.__dict__)

    def test_model_closure_and_index_maps_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_closure(root)
            self.adapter.MODEL_ROOT = root
            self.adapter._validate_model_closure()

            index = root / "transformer/diffusion_pytorch_model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"x": "wrong.safetensors"}}))
            with self.assertRaisesRegex(SystemExit, "shard index changed"):
                self.adapter._validate_model_closure()

            self._write_closure(root)
            (root / "transformer_full").mkdir()
            with self.assertRaisesRegex(SystemExit, "excluded component"):
                self.adapter._validate_model_closure()

    def test_request_profiles_and_seed_are_bounded(self) -> None:
        self.assertEqual(self.adapter._profile(None), "bf16-model-offload")
        self.assertEqual(
            self.adapter._profile("fp8-cast-sequential-offload"),
            "fp8-cast-sequential-offload",
        )
        with self.assertRaises(ValueError):
            self.adapter._profile("nvfp4")
        self.assertEqual(self.adapter._seed(42, 0), 42)
        for invalid in (-1, 2**63, True, "42"):
            with self.assertRaises(ValueError):
                self.adapter._seed(invalid, 0)
    def test_prompt_file_is_required_and_options_json_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.adapter.INPUT_ROOT = Path(directory)
            self.assertEqual(self.adapter._load_request(), {})
            with self.assertRaisesRegex(ValueError, "exactly one UTF-8"):
                self.adapter._load_prompt()
            (self.adapter.INPUT_ROOT / "prompt.txt").write_text(
                "  Operator supplied synchronized scene.  ", encoding="utf-8"
            )
            self.assertEqual(
                self.adapter._load_prompt(), "Operator supplied synchronized scene."
            )

    def test_generation_contract_hashes_raw_tensors_before_mux(self) -> None:
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for expected in (
            "sigmas=DISTILLED_SIGMA_VALUES",
            "guidance_scale=1.0",
            "audio_guidance_scale=1.0",
            "stg_scale=0.0",
            "audio_stg_scale=0.0",
            "guidance_rescale=0.0",
            "audio_guidance_rescale=0.0",
            "enable_prompt_enhancement=False",
            "max_sequence_length=1024",
            'torch.Generator("cuda").manual_seed(seed)',
            "num_frames=65",
            'output_type="np"',
            '"tensors": tensor_receipt',
        ):
            self.assertIn(expected, source)
        self.assertIn("enable_layerwise_casting", source)
        self.assertIn("storage_dtype=torch.float8_e4m3fn", source)
        self.assertIn('enable_sequential_cpu_offload(device="cuda")', source)

        class Array:
            dtype = "float32"
            shape = (1, 2)

            def __init__(self, payload: bytes):
                self.payload = payload

            def tobytes(self, *, order: str) -> bytes:
                self.assert_order = order
                return self.payload

        fake_numpy = types.SimpleNamespace(ascontiguousarray=lambda value: value)
        with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
            first = self.adapter._array_receipt(Array(b"one"))
            second = self.adapter._array_receipt(Array(b"one"))
            changed = self.adapter._array_receipt(Array(b"two"))
        self.assertEqual(first, second)
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_joint_audio_video_output_is_verified(self) -> None:
        video_stream = types.SimpleNamespace(
            width=768,
            height=512,
            average_rate=24,
            codec_context=types.SimpleNamespace(name="h264"),
        )
        audio_stream = types.SimpleNamespace(
            codec_context=types.SimpleNamespace(
                name="aac",
                sample_rate=48_000,
                layout=types.SimpleNamespace(name="stereo"),
            )
        )

        class Container:
            streams = types.SimpleNamespace(video=[video_stream], audio=[audio_stream])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def decode(self, *, video=None, audio=None):
                if video == 0:
                    return iter([object()] * 65)
                if audio == 0:
                    frame = types.SimpleNamespace(
                        samples=65_000,
                        to_ndarray=lambda: types.SimpleNamespace(size=1, values=[1.0]),
                    )
                    return iter([frame, frame])
                return iter(())

        fake_av = types.SimpleNamespace(open=lambda *_args, **_kwargs: Container())
        fake_numpy = types.SimpleNamespace(
            abs=lambda value: value.values,
            max=max,
        )
        with mock.patch.dict(sys.modules, {"av": fake_av, "numpy": fake_numpy}):
            receipt = self.adapter._verify_joint_av(
                Path("unused.mp4"),
                width=768,
                height=512,
                frame_count=65,
                sample_rate=48_000,
            )
        self.assertEqual(receipt["frames"], 65)
        self.assertEqual(receipt["audio_samples"], 130_000)
        self.assertEqual(receipt["video_codec"], "h264")
        self.assertEqual(receipt["audio_codec"], "aac")
        self.assertEqual(receipt["fps"], 24)
        self.assertEqual(receipt["duration_seconds"], 65 / 24)


if __name__ == "__main__":
    unittest.main()
