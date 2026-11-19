import importlib.metadata as metadata

from packaging.requirements import Requirement

for raw_requirement in metadata.requires("b12x") or ():
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = metadata.version(requirement.name)
    if requirement.specifier and installed not in requirement.specifier:
        raise RuntimeError(f"B12X requires {requirement}, but {installed} is installed")

# Every package built on CuTe DSL must accept the installed version: a locked
# wheel set older than one consumer's requirement silently breaks that consumer
# (quack-kernels 0.6.5 imports cutlass.base_dsl.enums, new in 4.7).
SHARED = ("nvidia-cutlass-dsl", "quack-kernels")
for consumer in ("vllm", "quack-kernels", "flashinfer-python", "nvidia-cudnn-frontend", "tokenspeed-mla"):
    try:
        requirements = metadata.requires(consumer) or ()
    except metadata.PackageNotFoundError:
        continue
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if not requirement.name.startswith(SHARED):
            continue
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        installed = metadata.version(requirement.name)
        if requirement.specifier and installed not in requirement.specifier:
            raise RuntimeError(f"{consumer} requires {requirement}, but {installed} is installed")
