# Consumer dependency proposal

The recipe repository owns `contracts/src/vonk_forge_contracts`; no Controller
module or JSON Schema copy is authoritative. During the greenfield rollout,
the platform and public catalog build jobs should refresh the contracts source
from the latest `vonk-forge-recipes` `main` before running their frozen install.
The build records the resolved recipe repository commit for traceability and
regenerates both schemas from the installed package.

Consumers should not carry a manually edited Git source revision. A contracts
change is a dependency change: it triggers Controller, catalog, schema, and
recipe checks and causes the dependent build to refresh its source snapshot.
The package has no Controller/runtime dependencies and requires
`pydantic>=2.13,<3`; the local wheel is a packaging check only and does not
imply PyPI publication.
