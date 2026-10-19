"""Derive native TensorFold inputs from immutable checkpoints into the managed cache."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

BIAS_SHA256 = "4745da4b5f24c67170ee24e2c95652e45ba13c0ca0abc2be3cc7a2a3545792d0"


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def model_view(source, destination, template):
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "chat_template.jinja":
            continue
        target = destination / item.name
        if target.is_symlink() and target.resolve() == item.resolve():
            continue
        temporary = destination / (".link-" + item.name)
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(item)
        os.replace(temporary, target)
    atomic_bytes(destination / "chat_template.jinja", template.read_bytes())
    sync_dir(destination)


def rank_cache_valid(path, expected):
    try:
        data_end = sum(row[4] for row in expected)
        size = path.stat().st_size
        if size < data_end + 18:
            return False
        with path.open("rb") as stream:
            stream.seek(-16, os.SEEK_END)
            tail = stream.read(16)
            if tail[8:] != b"TFDSRK01" or struct.unpack("<Q", tail[:8])[0] != data_end:
                return False
            stream.seek(data_end)
            body = stream.read(size - data_end - 16)
        digest = hashlib.sha256(body).hexdigest()[:20]
        return json.loads(body) == expected and path.name.endswith(f"-{digest}.bin")
    except (OSError, ValueError, struct.error):
        return False


def prepare_rank(model, cache, rank, world, binary):
    cache.mkdir(parents=True, exist_ok=True)
    command = [str(binary), str(model), str(cache), str(rank), str(world)]
    expected = json.loads(subprocess.check_output([*command, "--index-only"]))
    matches = list(cache.glob(f"rank{rank}of{world}-*.bin"))
    if len(matches) == 1 and rank_cache_valid(matches[0], expected):
        return matches[0]
    # Only this model/rank's derived files live here. Invalid published files
    # cannot poison the next start; the upstream writer atomically replaces its partial.
    for path in matches:
        path.unlink()
    subprocess.run(command, check=True)
    matches = list(cache.glob(f"rank{rank}of{world}-*.bin"))
    if len(matches) != 1 or not rank_cache_valid(matches[0], expected):
        raise ValueError("TensorFold rank-cache output does not match its checkpoint plan")
    sync_dir(cache)
    return matches[0]


def prepare_kit(model, destination, source, tools):
    expected = json.loads((source / "rope.json").read_text())
    hashes = {name: value for name, value in expected.items() if name.endswith(".f32")}
    hashes["vision/gate_bias_vl.safetensors"] = BIAS_SHA256
    receipt = destination / "vonk-generated.json"
    try:
        records = json.loads(receipt.read_text())
        if records.get("source") != (source / "VONK_SOURCE_SHA256").read_text():
            raise ValueError("kernel kit changed")
        if all(records[name]["sha256"] == digest and (destination / name).stat().st_size == records[name]["size"] for name, digest in hashes.items()):
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    subprocess.run([sys.executable, str(tools / "make_rope.py"), str(model), str(temporary)], check=True)
    subprocess.run([sys.executable, str(tools / "make_bias_vl.py"), str(model), str(temporary / "vision/gate_bias_vl.safetensors")], check=True)
    records = {"source": (source / "VONK_SOURCE_SHA256").read_text()}
    for name, digest in hashes.items():
        path = temporary / name
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"generated TensorFold kit digest differs: {name}")
            os.fsync(stream.fileno())
        records[name] = {"sha256": digest, "size": path.stat().st_size}
    atomic_bytes(temporary / "vonk-generated.json", json.dumps(records).encode())
    # There is no live reader before launch. Retain the old kit until the replacement verifies.
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    sync_dir(destination.parent)


def prepare(model, cache, rank, world, engine=Path("/opt/tensorfold"), template=Path("/opt/vonk/bin/chat_template.jinja")):
    cache.mkdir(parents=True, exist_ok=True)
    view = cache / "model"
    model_view(model, view, template)
    prepare_rank(model, cache / "rank-cache", rank, world, engine / "bin/tf-dsv41-rank-cache")
    token_map = cache / "dsv41_token_map.json"
    temporary = cache / ".token-map.partial"
    subprocess.run([sys.executable, str(engine / "tools/make_token_map.py"), str(model), str(temporary)], check=True)
    mapping = json.loads(temporary.read_text())
    if not isinstance(mapping, list) or not mapping or any(type(x) is not int or x < 0 for x in mapping):
        raise ValueError("invalid generated Engram token map")
    atomic_bytes(token_map, temporary.read_bytes())
    temporary.unlink()
    prepare_kit(model, cache / "kit", engine / "kit", engine / "tools")
    return view
