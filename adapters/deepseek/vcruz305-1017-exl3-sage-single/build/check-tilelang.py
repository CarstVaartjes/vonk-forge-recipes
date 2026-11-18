import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import os
import zipfile
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

built = {
    "tilelang": os.environ["TILELANG_VERSION"],
    "tile_kernels": os.environ["TILE_KERNELS_VERSION"],
}
for name, version in built.items():
    installed = metadata.version(name)
    assert installed == version, f"{name} {installed} is installed, expected {version}"
    copies = [d for d in metadata.distributions() if canonicalize_name(d.metadata["Name"] or "") == canonicalize_name(name)]
    assert len(copies) == 1, f"{len(copies)} {name} distributions are installed"
    # Every file of the wheel built above is installed unchanged.
    [wheel] = Path("/tmp/tilelang-wheels").glob(f"{name}-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        for path, digest, _ in csv.reader(io.TextIOWrapper(archive.open(record), "utf-8")):
            if not digest or ".dist-info/" in path or ".data/" in path:
                continue
            algorithm, wanted = digest.split("=", 1)
            data = Path(copies[0].locate_file(path)).read_bytes()
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).rstrip(b"=").decode()
            assert actual == wanted, f"{name}: installed {path} is not the built file"


def check(consumer: str, raw: str, shared: set[str] | None = None) -> None:
    try:
        requirement = Requirement(raw)
    except InvalidRequirement:
        return
    if shared is not None and canonicalize_name(requirement.name) not in shared:
        return
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        return
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"{consumer} requires {requirement}, which is not installed") from None
    if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f"{consumer} requires {requirement}, but {installed} is installed")


for name in built:
    for raw in metadata.requires(name) or ():
        check(name, raw)

# TileLang and B12X's CuTe DSL stack share apache-tvm-ffi, and TileKernels
# needs TileLang: every consumer must accept the installed versions. The
# serving vLLM is the candidate source, so its requirements/cuda.txt pins
# apply rather than the base image's vLLM metadata.
SHARED = {"apache-tvm-ffi", "tilelang"}
for dist in metadata.distributions():
    consumer = canonicalize_name(dist.metadata["Name"] or "")
    if consumer != "vllm":
        for raw in dist.requires or ():
            check(consumer, raw, SHARED)
for line in Path("/opt/spark3/candidate/vllm/requirements/cuda.txt").read_text().splitlines():
    line = line.split("#", 1)[0].strip()
    if line and not line.startswith("-"):
        check("vllm (requirements/cuda.txt)", line, SHARED)
print("TileLang build checks passed")
