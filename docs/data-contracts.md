# Sweep consumer contracts

The Python consumer model registry is `tools/python-model-registry.json`.
Public Model and Recipe contracts remain documented in `contracts/README.md`.

| Module | Models | Ownership |
|---|---|---|
| `vonk_forge_contracts.sweep` | `CleanupEnd`, `CleanupObservationBudget` | Local durable sweep reconciliation; `CleanupEndReason` supplies the typed end reason. |
| `vonk_forge_contracts.sweep` | `CleanupFleetObservation`, `FleetNodeObservation`, `FleetConnectionObservation`, `LoadedRunObservation`, `ReviewedCleanupStop` | Read projections of authenticated CLI observations and recorded reviewed stop targets. Unrecognized or incomplete observations cannot establish absence. |

These projections ignore external wire extensions, require the fields used for
absence decisions, and retain the Controller as the admission authority.
They add no Controller wire fields or public Model/Recipe document fields.
Rust/TS Controller generation belongs to the Controller repository; this clone
contains no Controller generators or generated consumers for local sweep state.
