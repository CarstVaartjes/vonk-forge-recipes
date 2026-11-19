import importlib.metadata as metadata

from packaging.requirements import Requirement

for raw_requirement in metadata.requires("b12x") or ():
    requirement = Requirement(raw_requirement)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    installed = metadata.version(requirement.name)
    if requirement.specifier and installed not in requirement.specifier:
        raise RuntimeError(f"B12X requires {requirement}, but {installed} is installed")
