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
