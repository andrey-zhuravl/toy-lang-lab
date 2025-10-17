"""Grammar DSL loader for Toy Lang Lab."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Grammar:
    """Represents the parsed grammar file."""

    entities: dict[str, list[str]]
    templates: list[str]
    template_weights: list[float] | None
    raw: Mapping[str, object]

    def validate(self) -> None:
        if not self.entities:
            raise GrammarError("Grammar must define at least one entity")
        for name, values in self.entities.items():
            if not values:
                raise GrammarError(f"Entity '{name}' must have values")
        if not self.templates:
            raise GrammarError("Grammar must define at least one template")
        if self.template_weights is not None:
            if len(self.template_weights) != len(self.templates):
                raise GrammarError("templates weights must match number of templates")
            if not _is_probability_vector(self.template_weights):
                raise GrammarError("templates weights must be non-negative and sum to 1")


class GrammarError(ValueError):
    """Raised for invalid grammar files."""


def load_grammar(path: Path) -> Grammar:
    """Load a YAML grammar file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, MutableMapping):
        raise GrammarError("Grammar must be a mapping")

    entities = _parse_entities(data.get("entities"))
    templates = _parse_templates(data.get("templates"))
    weights_section = data.get("weights")
    template_weights: list[float] | None = None
    if weights_section is not None:
        if not isinstance(weights_section, MutableMapping):
            raise GrammarError("weights must be a mapping")
        template_weights = _parse_weights(weights_section.get("templates"), len(templates))

    grammar = Grammar(
        entities=entities,
        templates=templates,
        template_weights=template_weights,
        raw=data,
    )
    grammar.validate()
    return grammar


def _parse_entities(value: object) -> dict[str, list[str]]:
    if not isinstance(value, MutableMapping) or not value:
        raise GrammarError("entities must be a non-empty mapping")
    entities: dict[str, list[str]] = {}
    for name, items in value.items():
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            raise GrammarError(f"Entity '{name}' must map to a list of strings")
        converted = [str(item) for item in items]
        if not converted:
            raise GrammarError(f"Entity '{name}' must not be empty")
        entities[str(name)] = converted
    return entities


def _parse_templates(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise GrammarError("templates must be a list of strings")
    templates = [str(template) for template in value]
    if not templates:
        raise GrammarError("templates must not be empty")
    return templates


def _parse_weights(value: object, expected: int) -> list[float]:
    if value is None:
        raise GrammarError("weights.templates must be provided when weights is specified")
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise GrammarError("weights.templates must be a list of numbers")
    weights = [float(item) for item in value]
    if len(weights) != expected:
        raise GrammarError("weights.templates length must match number of templates")
    if not _is_probability_vector(weights):
        raise GrammarError("weights.templates must be non-negative and sum to 1")
    return weights


def _is_probability_vector(values: Sequence[float]) -> bool:
    if not values:
        return False
    total = sum(values)
    if total <= 0:
        return False
    normalized = [v / total for v in values]
    return all(v >= 0 for v in normalized) and abs(sum(normalized) - 1.0) < 1e-6
