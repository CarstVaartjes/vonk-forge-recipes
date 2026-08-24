from __future__ import annotations

import hashlib
import json
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "nano": ROOT / "recipes/nemotron-3-nano-30b-a3b-vllm-single.json",
    "omni": ROOT / "recipes/nemotron-3-nano-omni-30b-a3b-vllm-single.json",
    "super": ROOT / "recipes/nemotron-3-super-120b-a12b-vllm-single.json",
}
RUNTIME_PATHS = {
    "nano": ROOT / "runtime-distributions/vllm-0-20-0-nvidia-arm64.json",
    "omni": ROOT / "runtime-distributions/vllm-0-20-0-nvidia-arm64.json",
    "super": ROOT / "runtime-distributions/vllm-0-27-1-nvidia-arm64.json",
}
ADAPTER_ROOTS = {
    "nano": ROOT / "adapters/nvidia/nemotron-vllm-0-20-0",
    "omni": ROOT / "adapters/nvidia/nemotron-vllm-0-20-0",
    "super": ROOT / "adapters/llm/vllm-openai",
}
RELEASES = {
    name: ROOT / f"recipe-releases/{path.stem}.json"
    for name, path in RECIPES.items()
}
MODELS = {
    "nano": ROOT / "model-versions/nemotron-3-nano-30b-a3b-nvfp4.json",
    "omni": ROOT
    / "model-versions/nemotron-3-nano-omni-30b-a3b-reasoning-nvfp4.json",
    "super": ROOT / "model-versions/nemotron-3-super-120b-a12b-nvfp4.json",
}
SUPER_MTPV2 = ROOT / "model-versions/nemotron-3-super-120b-a12b-bf16-mtpv2.json"
REVISIONS = {
    "nano": "ce1b118ae66ec705d02c241525192832eb045fd3",
    "omni": "dc5f0b0bfddf8b6e0f5891475be9af05b80126fe",
    "super": "445d56f38229f7a37ae5207734f7e8af0fa9a2c8",
}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_digest(path: Path) -> str:
    payload = json.dumps(
        _read(path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arguments(recipe: dict[str, object]) -> dict[str, object]:
    runtime = recipe["runtime"]
    assert isinstance(runtime, dict)
    arguments = runtime["arguments"]
    assert isinstance(arguments, list)
    return {str(item["name"]): item["value"] for item in arguments}


class NemotronExecutableRecipeTests(unittest.TestCase):
    def test_release_changelogs_bind_version_2_to_the_exact_recipes(self) -> None:
        release_validator = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "recipe_release"
        ]
        for name, recipe_path in RECIPES.items():
            with self.subTest(recipe=name):
                recipe = _read(recipe_path)
                identity = recipe["identity"]
                release = release_validator(
                    RELEASES[name],
                    publisher=identity["publisher"],
                    slug=identity["slug"],
                    recipe_digest=_canonical_digest(recipe_path),
                )
                expected_version = "2.1.0" if name == "super" else "2.0.1"
                self.assertEqual(release["version"], expected_version)
                expected_effect = "rebuild" if name == "super" else "metadata-only"
                self.assertEqual(release["history"][0]["upgrade_effect"], expected_effect)

    def test_all_recipes_use_exact_current_models_and_runtime(self) -> None:
        for name, recipe_path in RECIPES.items():
            with self.subTest(recipe=name):
                recipe = _read(recipe_path)
                model = _read(MODELS[name])
                self.assertEqual(model["source"]["revision"], REVISIONS[name])
                self.assertEqual(
                    recipe["model"]["content_sha256"],
                    _canonical_digest(MODELS[name]),
                )
                self.assertEqual(recipe["artifacts"][0]["revision"], REVISIONS[name])
                self.assertEqual(
                    recipe["runtime"]["distribution"]["content_sha256"],
                    _canonical_digest(RUNTIME_PATHS[name]),
                )
                expected_runtime = (
                    "vllm-0-27-1-nvidia-arm64"
                    if name == "super"
                    else "vllm-0-20-0-nvidia-arm64"
                )
                self.assertEqual(
                    recipe["runtime"]["distribution"]["slug"],
                    expected_runtime,
                )
                self.assertIn("candidate", recipe["metadata"]["tags"])
                self.assertNotIn("metadata-only", recipe["metadata"]["tags"])
                self.assertNotIn("non-executable", recipe["metadata"]["tags"])

    def test_custom_parsers_are_snapshot_owned_and_wired(self) -> None:
        expected = {
            "nano": (
                "nano_v3",
                "nano_v3_reasoning_parser.py",
                "aafb12208054504f619cbdd01837e1532a482ad937ed987bfe9a13fb812ae2b7",
            ),
            "super": (
                "super_v3",
                "super_v3_reasoning_parser.py",
                "f7fc71d1697ed79931787cf7485ee297365a41c3442cc1e05e6ebab43e0a39c1",
            ),
        }
        for name, (parser, filename, digest) in expected.items():
            with self.subTest(recipe=name):
                arguments = _arguments(_read(RECIPES[name]))
                self.assertEqual(arguments["reasoning-parser"], parser)
                mount = "/models/target" if name == "super" else "/models"
                self.assertEqual(
                    arguments["reasoning-parser-plugin"], f"{mount}/{filename}"
                )
                artifacts = _read(MODELS[name])["artifacts"]
                plugin = next(item for item in artifacts if item["path"] == filename)
                self.assertEqual(plugin["sha256"], digest)
                self.assertEqual(plugin["revision"], REVISIONS[name])
                self.assertIn("runtime", plugin["roles"])

    def test_super_contract_includes_required_fp4_mamba_and_mtp_settings(self) -> None:
        recipe = _read(RECIPES["super"])
        arguments = _arguments(recipe)
        self.assertEqual(arguments["quantization"], "modelopt_fp4")
        self.assertEqual(arguments["moe-backend"], "marlin")
        self.assertEqual(arguments["kv-cache-dtype"], "fp8")
        self.assertEqual(arguments["mamba-ssm-cache-dtype"], "float16")
        self.assertEqual(
            json.loads(str(arguments["speculative-config"])),
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
                "model": "/models/drafter",
                "moe_backend": "triton",
            },
        )
        companion = _read(SUPER_MTPV2)
        self.assertEqual(
            companion["source"]["revision"],
            "c929f8a55d0527fea9f58b4cedc9e0c855cfc421",
        )
        self.assertEqual(
            recipe["dependencies"][0]["content_sha256"],
            _canonical_digest(SUPER_MTPV2),
        )
        drafter = next(item for item in recipe["artifacts"] if item["id"] == "drafter")
        self.assertEqual(drafter["mount"]["target"], "/models/drafter")
        self.assertEqual(drafter["download_bytes"], companion["sizes"]["download_bytes"])

    def test_super_model_inventories_are_exact_and_complete(self) -> None:
        target = _read(MODELS["super"])
        target_artifacts = target["artifacts"]
        self.assertEqual(len(target_artifacts), 36)
        self.assertTrue(
            all(item["revision"] == REVISIONS["super"] for item in target_artifacts)
        )
        self.assertEqual(
            sum(item["download_bytes"] for item in target_artifacts),
            target["sizes"]["download_bytes"],
        )
        target_readme = next(
            item for item in target_artifacts if item["path"] == "README.md"
        )
        self.assertEqual(target_readme["download_bytes"], 81392)
        self.assertEqual(
            target_readme["sha256"],
            "4e07131b1a37d311cc220487cb7731801ea57effd69a7b9eefc605231c76435c",
        )

        companion = _read(SUPER_MTPV2)
        expected = {
            ".gitattributes": (
                1519,
                "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
            ),
            "README.md": (
                13736,
                "505b2b8d7626f14865e70de71d5c49025e9137af23082ce531108ad7c5fe49e7",
            ),
            "config.json": (
                1924,
                "699f34f0fc645d29ebffa5767fb59e6ae6ec98e3a4605485eb9913256d0df7e6",
            ),
            "configuration_nemotron_h.py": (
                13669,
                "2b3ce37ae17d9d2eb78594d9cc6becc57037245d828e69727d6f3cd45125a197",
            ),
            "generation_config.json": (
                150,
                "dc5b4dbdff26634eaab9c50adc4b58d01aaed0d1648214f1694665373c905516",
            ),
            "model.safetensors": (
                5884780728,
                "f0114c95f8a85fb19de1d05a8733abfffcd9a15203512ab715fe75a1d33b5c83",
            ),
            "model.safetensors.index.json": (
                77663,
                "7f36923c963a0f95a331ef25aa9d51dc6e3d701eff53c58596c1c481e1d5e4b7",
            ),
        }
        actual = {
            item["path"]: (item["download_bytes"], item["sha256"])
            for item in companion["artifacts"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(sum(size for size, _sha in actual.values()), 5884889389)

    def test_omni_advertises_an_executable_text_only_contract(self) -> None:
        recipe = _read(RECIPES["omni"])
        arguments = _arguments(recipe)
        self.assertIs(arguments["language-model-only"], True)
        tags = set(recipe["metadata"]["tags"])
        self.assertIn("text", tags)
        self.assertTrue({"audio", "image", "video", "multimodal"}.isdisjoint(tags))
        self.assertEqual(recipe["interfaces"][0]["adapter"], "openai")

    def test_source_bundle_and_runtime_have_no_hidden_download_step(self) -> None:
        source_bundle = runpy.run_path(str(ROOT / "tools/build-catalog-index"))[
            "source_bundle"
        ]
        expected_images = {
            "nano": "sha256:871fc9b75a97dc6f58449bfb2aa30cab685c87a8bf6b4519a75c4760c1f8bcd8",
            "omni": "sha256:871fc9b75a97dc6f58449bfb2aa30cab685c87a8bf6b4519a75c4760c1f8bcd8",
            "super": "sha256:1c8e60a0841b333c700488cb029d3664807249da0c071e862191b00fe34b228c",
        }
        for name, recipe_path in RECIPES.items():
            adapter_root = ADAPTER_ROOTS[name]
            archive, _, digest = source_bundle(adapter_root)
            dockerfile = (adapter_root / "Dockerfile").read_text(encoding="utf-8")
            self.assertNotIn("pip install", dockerfile)
            self.assertNotIn("apt-get", dockerfile)
            self.assertIn(expected_images[name], dockerfile)
            context = _read(recipe_path)["build"]["context"]
            self.assertEqual(context["sha256"], digest)
            self.assertEqual(context["expected_bytes"], len(archive))


if __name__ == "__main__":
    unittest.main()
