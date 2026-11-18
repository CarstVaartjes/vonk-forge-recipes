import importlib.metadata as metadata
import os

installed = metadata.version("dgx-spark-networking")
assert installed == os.environ["SPARKNET_VERSION"], f"sparknet {installed} is installed"
unconditional = [r for r in metadata.requires("dgx-spark-networking") or () if "extra ==" not in r]
assert not unconditional, f"sparknet takes no runtime dependencies: {unconditional}"
from sparknet.oneshot._proxy import load

print("sparknet proxy ABI", load().roce_abi_version())
print("sparknet build checks passed")
