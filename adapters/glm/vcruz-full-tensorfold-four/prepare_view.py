"""Derive the creator's BF16 head and template in managed cache, offline."""
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

HEAD_SHA = "2df0f4a8469cf4a65295107130b8df4e5562e955ba1bd32f75b51f3e80ddb5f3"
HEAD_BYTES = 1903165544
CONFIG_SHA = "f520ce2795fa4f38a835f84ad4d3db17983d63babb7bb16f6e23178b4612ba18"


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_head(source, output, expected_sha=HEAD_SHA, expected_size=HEAD_BYTES):
    source, output = Path(source), Path(output)
    if output.is_file() and output.stat().st_size == expected_size and digest(output) == expected_sha:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".head-", dir=output.parent)
    try:
        with source.open("rb") as stream, os.fdopen(descriptor, "wb") as target:
            n = struct.unpack("<Q", stream.read(8))[0]
            if n > source.stat().st_size - 8:
                raise ValueError("invalid source shard header")
            meta = json.loads(stream.read(n))["lm_head.weight"]
            a, b = meta["data_offsets"]
            if meta["dtype"] != "BF16" or not 0 <= a < b <= source.stat().st_size - 8 - n:
                raise ValueError("invalid BF16 head tensor")
            header = json.dumps({"lm_head.weight": {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [0, b-a]}}).encode()
            header += b" " * ((8-len(header) % 8) % 8)
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            stream.seek(8+n+a)
            remaining = b-a
            while remaining:
                block = stream.read(min(16 << 20, remaining))
                if not block:
                    raise ValueError("truncated source tensor")
                target.write(block)
                remaining -= len(block)
            target.flush()
            os.fsync(target.fileno())
        if Path(temporary).stat().st_size != expected_size or digest(temporary) != expected_sha:
            raise ValueError("derived BF16 head does not match creator's exact identity")
        os.replace(temporary, output)
        sync_directory(output.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_atomic(path, content):
    if path.is_file() and not path.is_symlink() and path.read_bytes() == content:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def link_atomic(path, target):
    if path.is_symlink() and path.readlink() == target:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".link-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        temporary.unlink()
        temporary.symlink_to(target)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(pack, source, cache):
    import fix_chat_template as template
    pack, cache = Path(pack).resolve(), Path(cache).resolve()
    if digest(pack / "config.json") != CONFIG_SHA:
        raise ValueError("target config differs from the pinned creator checkpoint")
    original = (pack / "chat_template.jinja").read_bytes()
    if hashlib.sha256(original).hexdigest() != template.OFFICIAL_SHA or original.count(template.OLD) != 1:
        raise ValueError("chat template differs from the pinned original")
    fixed = original.replace(template.OLD, template.NEW)
    if hashlib.sha256(fixed).hexdigest() != template.FIXED_SHA:
        raise ValueError("derived chat template has an invalid identity")
    index = json.loads((pack / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
    cache.mkdir(parents=True, exist_ok=True)
    head = cache / (HEAD_SHA + ".safetensors")
    extract_head(source, head)
    view = cache / "view-cc64e77b-aca966e4"
    existing = view.is_dir()
    stage = view if existing else Path(tempfile.mkdtemp(prefix=".view-", dir=cache))
    try:
        # Reconcile derived metadata before launching the engine. A prior failed
        # start or corrupt disposable view cannot poison future starts.
        for path in pack.iterdir():
            if path.name not in {"model.safetensors.index.json", "chat_template.jinja"}:
                link_atomic(stage / path.name, path)
        link_atomic(stage / "lm_head.safetensors", head)
        write_atomic(stage / "chat_template.jinja", fixed)
        write_atomic(stage / "model.safetensors.index.json", json.dumps(index, indent=2).encode())
        sync_directory(stage)
        if not existing:
            os.replace(stage, view)
            sync_directory(cache)
    finally:
        if not existing and stage.exists():
            shutil.rmtree(stage)
    return view
