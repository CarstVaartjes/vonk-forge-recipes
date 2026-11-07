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

from vonk_forge_contracts.api import RecipeApi

from qualification.definitions_loader import QUALIFICATION_ROOT, load_definitions
from qualification.documents import recipe_api


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
        self._recipe_apis: dict[str, RecipeApi] = {}

    @classmethod
    def load(cls, root: Path = QUALIFICATION_ROOT) -> Definitions:
        """Every definition and fixture, read and digest-checked now.

        A sweep runs for days from a source tree that may change under it: its inputs are
        held in memory from the start, so a smoke never reads a file that has since gone.
        """
        definitions = cls(load_definitions(root), root)
        for fixture_id in definitions.document["fixtures"]:
            definitions.fixture(fixture_id)
        definitions._load_recipe_apis()
        return definitions

    def _load_recipe_apis(self) -> None:
        """The ``recipe-api`` documents beside the qualification tree (``api/recipes``)."""
        for path in sorted((self.root.parent / "api" / "recipes").glob("*.json")):
            self._recipe_apis[path.stem] = recipe_api(path.stem, self.root.parent)

    def recipe_api(self, slug: str) -> RecipeApi | None:
        """The recipe's published API facts, when it has them."""
        return self._recipe_apis.get(slug)

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

    def service_cases(
        self, key: str, alias: str, voice: str | None = None
    ) -> list[dict[str, Any]]:
        """The recipe's reviewed smoke cases with ``$ALIAS``, ``$VOICE`` and fixtures substituted.

        ``voice`` is the recipe's first ``audio_speech`` voice (its api document owns it).
        """
        service = self.service(key)
        if service is None:
            return []
        templates = self.document["service_case_templates"]
        cases: list[dict[str, Any]] = []
        for case_id in service.get("smoke_cases", []):
            template = templates.get(case_id)
            if isinstance(template, dict):
                rendered = self._substitute(template, alias, voice)
                rendered["id"] = case_id
                cases.append(rendered)
        return cases

    def _substitute(self, value: Any, alias: str, voice: str | None = None) -> Any:
        if value == "$ALIAS":
            return alias
        if value == "$VOICE":
            if voice is None:
                raise ValueError("a $VOICE case needs the recipe's audio_speech voices")
            return voice
        if isinstance(value, list):
            return [self._substitute(item, alias, voice) for item in value]
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
            return {k: self._substitute(v, alias, voice) for k, v in value.items()}
        return value
