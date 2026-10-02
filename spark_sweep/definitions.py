"""The repository's reviewed smoke definitions, as the sweep runs them.

``qualification/shared.json`` and ``qualification/recipes/<slug>.json`` already
say, per recipe, which smoke cases a serving recipe must pass and which fixture
and assertions a job recipe must satisfy. The sweep runs exactly those cases
against the real Sparks; it never invents a second definition.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qualification.definitions_loader import QUALIFICATION_ROOT, load_definitions


@dataclass(frozen=True)
class Fixture:
    name: str
    media_type: str
    content: bytes


class Definitions:
    def __init__(
        self, document: dict[str, Any], root: Path = QUALIFICATION_ROOT
    ) -> None:
        self.document = document
        self.root = root
        self._fixtures: dict[str, Fixture] = {}

    @classmethod
    def load(cls, root: Path = QUALIFICATION_ROOT) -> Definitions:
        return cls(load_definitions(root), root)

    def service(self, key: str) -> dict[str, Any] | None:
        value = self.document["service_recipes"].get(key)
        return value if isinstance(value, dict) else None

    def artifact(self, key: str) -> dict[str, Any] | None:
        value = self.document["recipes"].get(key)
        return value if isinstance(value, dict) else None

    def alias(self, key: str) -> str | None:
        service = self.service(key)
        return str(service["alias"]) if service and service.get("alias") else None

    def fixture(self, fixture_id: str) -> Fixture:
        if fixture_id not in self._fixtures:
            spec = self.document["fixtures"][fixture_id]
            raw = (self.root / spec["path"]).read_bytes()
            content = base64.b64decode(raw) if spec["encoding"] == "base64" else raw
            if hashlib.sha256(content).hexdigest() != spec["sha256"]:
                raise ValueError(
                    f"fixture {fixture_id} does not match its recorded digest"
                )
            self._fixtures[fixture_id] = Fixture(
                spec["name"], spec["media_type"], content
            )
        return self._fixtures[fixture_id]

    def service_cases(self, key: str, alias: str) -> list[dict[str, Any]]:
        """The recipe's reviewed smoke cases with ``$ALIAS`` and fixtures substituted."""
        service = self.service(key)
        if service is None:
            return []
        templates = self.document["service_case_templates"]
        cases: list[dict[str, Any]] = []
        for case_id in service.get("smoke_cases", []):
            template = templates.get(case_id)
            if isinstance(template, dict):
                rendered = self._substitute(template, alias)
                rendered["id"] = case_id
                cases.append(rendered)
        return cases

    def _substitute(self, value: Any, alias: str) -> Any:
        if value == "$ALIAS":
            return alias
        if isinstance(value, list):
            return [self._substitute(item, alias) for item in value]
        if isinstance(value, dict):
            if set(value) in ({"$fixture_data_uri"}, {"$fixture_base64"}):
                marker = next(iter(value))
                fixture = self.fixture(value[marker])
                encoded = base64.b64encode(fixture.content).decode("ascii")
                return (
                    f"data:{fixture.media_type};base64,{encoded}"
                    if marker == "$fixture_data_uri"
                    else encoded
                )
            return {k: self._substitute(v, alias) for k, v in value.items()}
        return value
