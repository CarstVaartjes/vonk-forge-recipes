"""The offline transform must reject corrupt output and preserve verified work."""

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_glm_view", ROOT / "adapters/glm/vcruz-full-tensorfold-four/prepare_view.py"
)
assert SPEC and SPEC.loader
VIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIEW)


def test_extract_exact_head_reuses_verified_output_and_preserves_source(tmp_path):
    tensor = b"\x01\x00\x02\x00"
    metadata = {
        "lm_head.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [2, 6]}
    }
    header = json.dumps(metadata).encode()
    source = tmp_path / "source"
    original = struct.pack("<Q", len(header)) + header + b"xx" + tensor
    source.write_bytes(original)
    source.chmod(0o444)
    expected_header = json.dumps(
        {"lm_head.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 4]}}
    ).encode()
    expected_header += b" " * (-len(expected_header) % 8)
    expected = struct.pack("<Q", len(expected_header)) + expected_header + tensor
    output = tmp_path / "cache" / "head"
    identity = hashlib.sha256(expected).hexdigest()
    VIEW.extract_head(source, output, identity, len(expected))
    assert output.read_bytes() == expected
    modified = output.stat().st_mtime_ns
    VIEW.extract_head(source, output, identity, len(expected))
    assert output.stat().st_mtime_ns == modified
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="exact identity"):
        VIEW.extract_head(source, output, "0" * 64, len(expected))
    assert output.read_bytes() == expected
    assert not list(output.parent.glob(".head-*"))


def test_invalid_tensor_bounds_never_publish(tmp_path):
    source = tmp_path / "source"
    header = json.dumps(
        {"lm_head.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [0, 999]}}
    ).encode()
    source.write_bytes(struct.pack("<Q", len(header)) + header + b"xx")
    output = tmp_path / "head"
    with pytest.raises(ValueError, match="invalid BF16"):
        VIEW.extract_head(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".head-*"))


def test_cached_view_recovers_corrupt_index_and_missing_link(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "config.json").write_bytes(b"{}")
    (pack / "weights.safetensors").write_bytes(b"immutable weights")
    (pack / "chat_template.jinja").write_bytes(b"before")
    (pack / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "other": "weights.safetensors",
                    "lm_head.weight": "weights.safetensors",
                }
            }
        )
    )
    monkeypatch.setattr(VIEW, "CONFIG_SHA", hashlib.sha256(b"{}").hexdigest())
    monkeypatch.setattr(
        VIEW,
        "extract_head",
        lambda source, output: output.write_bytes(b"verified head"),
    )
    monkeypatch.setitem(
        sys.modules,
        "fix_chat_template",
        SimpleNamespace(
            OFFICIAL_SHA=hashlib.sha256(b"before").hexdigest(),
            FIXED_SHA=hashlib.sha256(b"after").hexdigest(),
            OLD=b"before",
            NEW=b"after",
        ),
    )
    before = {p.name: p.read_bytes() for p in pack.iterdir()}
    view = VIEW.prepare(pack, tmp_path / "source", tmp_path / "cache")
    (view / "model.safetensors.index.json").write_bytes(b"interrupted/corrupt metadata")
    (view / "weights.safetensors").unlink()
    assert VIEW.prepare(pack, tmp_path / "source", tmp_path / "cache") == view
    assert (
        json.loads((view / "model.safetensors.index.json").read_text())["weight_map"][
            "lm_head.weight"
        ]
        == "lm_head.safetensors"
    )
    assert (view / "weights.safetensors").read_bytes() == b"immutable weights"
    assert {p.name: p.read_bytes() for p in pack.iterdir()} == before
