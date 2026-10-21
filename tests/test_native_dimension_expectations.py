"""Native bucket assertions must follow fixture aspect, never an old frame size."""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from qualification.definitions_loader import load_definitions
from spark_sweep.job_assertions import Output, evaluate
from spark_sweep.smoke import AssertionFailed

ROOT = Path(__file__).resolve().parents[1]
HUNYUAN = {
    "hunyuan-video-15-distilled-diffusers-single": (848, 480),
    "hunyuan-video-15-i2v-step-distilled-diffusers-single": (640, 640),
    "hunyuan-video-15-t2v-diffusers-single": (1280, 720),
}


def test_imported_bucket_bodies_match_pinned_upstream():
    # AST body hashes of the original functions, fetched at these exact image
    # pins. An image refresh must review its bucket implementation too.
    pins = {
        "adapters/video/hunyuan-video-15-native": "d035dcd7cc7c88e0a154609b62887d50bba9fdc2",
        "adapters/image/qwen-image-edit-2511-diffusers": "c5469b7ceb606edd7ba6570dcd17d38590a18db6",
        "adapters/image/qwen-image-lightning-diffusers": "c5469b7ceb606edd7ba6570dcd17d38590a18db6",
        "adapters/image/qwen-image-layered-diffusers": "c5469b7ceb606edd7ba6570dcd17d38590a18db6",
    }
    for context, pin in pins.items():
        assert pin in (ROOT / context / "Dockerfile").read_text()
    expected = {
        "generate_crop_size_list": "49d5ef08392fe14f6deedf1e9619b974f89d81ca2608f4a3baa1697f2d993dd3",
        "get_closest_ratio": "95ab9c311cb3bdae61225d52ccbcf079220d3cf36850cdb3020af373711633ac",
        "calculate_dimensions": "591cae36575fcb9cbfe0f2b950fc7f7db71627360d6ecab1c6e925b2ba53e041",
    }
    source = ast.parse((ROOT / "qualification/native_buckets.py").read_text())
    actual = {
        node.name: hashlib.sha256(
            ast.dump(ast.Module(body=node.body, type_ignores=[])).encode()
        ).hexdigest()
        for node in source.body
        if isinstance(node, ast.FunctionDef)
    }
    assert actual == expected


@pytest.mark.parametrize("slug,size", HUNYUAN.items())
def test_native_hunyuan_output_passes_and_wrong_dimensions_fail(
    slug, size, monkeypatch
):
    """Feed the measured stream shape through the real smoke evaluator."""
    spec = load_definitions()["recipes"][f"vonk-forge/{slug}"]
    width, height = size
    stream = {
        "width": width,
        "height": height,
        "codec_name": "h264",
        "avg_frame_rate": "24/1",
        "nb_read_frames": "1",
        "duration": 1 / 24,
        "start_time": 0,
    }
    monkeypatch.setattr(
        "spark_sweep.job_media._probe",
        lambda content: {
            "video": stream,
            "audio": [],
            "streams": [stream],
            "format": {"start_time": 0},
        },
    )
    assertion = next(a for a in spec["assertions"] if a["kind"] == "video-metadata")
    output = Output("output.mp4", "video/mp4", b"measured artifact")
    assert evaluate([assertion], [output]) == {}
    for field in ("width", "height"):
        stream[field] += 16
        with pytest.raises(AssertionFailed, match=f"video-metadata {field}"):
            evaluate([assertion], [output])
        stream[field] -= 16


@pytest.mark.parametrize("size", [(64, 64), (192, 108), (108, 192)])
@pytest.mark.parametrize(
    "kind,target", [("hunyuan-video-15", 640), ("qwen-image", 1024)]
)
def test_fixture_aspect_changes_resolved_assertions(tmp_path, size, kind, target):
    from qualification.native_dimensions import resolve_native_dimensions

    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, format="PNG")
    content = buffer.getvalue()
    (tmp_path / "input.b64").write_bytes(base64.b64encode(content))
    fixture = {
        "image": {
            "path": "input.b64",
            "name": "input.png",
            "encoding": "base64",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    }
    case = {
        "inputs": [{"slot": "image", "fixture": "image"}],
        "assertions": [{"native_dimensions": {"kind": kind, "target_size": target}}],
    }
    resolve_native_dimensions(case, fixture, tmp_path)
    assertion = case["assertions"][0]
    width, height = assertion["width"], assertion["height"]
    assert (width > height) == (size[0] > size[1])
    assert (width < height) == (size[0] < size[1])
    assert width % 16 == height % 16 == 0
    assert "native_dimensions" not in assertion
    # A changed fixture cannot silently redefine the expected dimensions.
    (tmp_path / "input.b64").write_bytes(base64.b64encode(b"corrupt"))
    case["assertions"] = [{"native_dimensions": {"kind": kind, "target_size": target}}]
    with pytest.raises(ValueError, match="recorded digest"):
        resolve_native_dimensions(case, fixture, tmp_path)


def test_bucket_selecting_media_contexts_cannot_hard_code_dimensions():
    """Audit every dimension assertion against the pipeline selected by its image.

    These Diffusers contexts let the native pipeline select dimensions from input
    aspect. Other contexts were reviewed separately: Comfy Qwen Edit uses
    FluxKontextImageScale and its square fixtures already match 1024x1024.
    """
    contexts = {
        "adapters/video/hunyuan-video-15-native",
        "adapters/image/qwen-image-edit-2511-diffusers",
        "adapters/image/qwen-image-lightning-diffusers",
        "adapters/image/qwen-image-layered-diffusers",
    }
    seen = set()
    resolved = load_definitions()
    for path in (ROOT / "qualification/recipes").glob("*.json"):
        recipe = json.loads((ROOT / "recipes" / path.name).read_text())
        context = recipe["execution"].get("build", {}).get("context", {}).get("path")
        if context not in contexts:
            continue
        if context == "adapters/image/qwen-image-lightning-diffusers" and not any(
            a.get("value") == "image-to-image" for a in recipe["runtime"]["arguments"]
        ):
            continue
        raw = json.loads(path.read_text())
        spec = raw["recipes"]
        for case in [spec, *spec.get("cases", [])]:
            for assertion in case.get("assertions", []):
                if assertion["kind"] not in {"image-metadata", "video-metadata"}:
                    continue
                assert "native_dimensions" in assertion, path.name
                assert not {"width", "height"} & assertion.keys(), path.name
        seen.add(context)
        for assertion in resolved["recipes"][raw["id"]]["assertions"]:
            assert "native_dimensions" not in assertion
    assert seen == contexts


def test_additional_case_derives_inherited_assertions_from_its_own_input(tmp_path):
    fixtures = {}
    for name, size in (("square", (64, 64)), ("portrait", (108, 192))):
        buffer = io.BytesIO()
        Image.new("RGB", size).save(buffer, format="PNG")
        content = buffer.getvalue()
        (tmp_path / name).write_bytes(content)
        fixtures[name] = {
            "path": name,
            "name": f"{'a' if name == 'portrait' else 'z'}-{name}.png",
            "encoding": "raw",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (tmp_path / "shared.json").write_text(json.dumps({"fixtures": fixtures}))
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes/test.json").write_text(
        json.dumps(
            {
                "id": "p/test",
                "recipes": {
                    "inputs": [{"slot": "image", "fixture": "square"}],
                    "assertions": [
                        {
                            "native_dimensions": {
                                "kind": "qwen-image",
                                "target_size": 1024,
                            }
                        }
                    ],
                    "cases": [
                        {
                            "inputs": [
                                {"slot": "image", "fixture": "square"},
                                {"slot": "image", "fixture": "portrait"},
                            ]
                        }
                    ],
                },
            }
        )
    )
    spec = load_definitions(tmp_path)["recipes"]["p/test"]
    assert spec["assertions"][0] == {"width": 1024, "height": 1024}
    portrait = spec["cases"][0]["assertions"][0]
    assert portrait["height"] > portrait["width"]
