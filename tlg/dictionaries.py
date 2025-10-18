"""Dictionary builder utilities for Stage A2."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tlg.grammar import Grammar


class DictionaryError(RuntimeError):
    """Raised when dictionary generation fails."""


@dataclass(frozen=True)
class DictionaryConfig:
    """Configuration for dictionary generation."""

    name: str
    entries: list[dict[str, object]]
    metadata: dict[str, object]
    shuffle: bool
    seed: int | None
    output_path: Path
    raw: dict[str, object]


def load_dictionary_config(path: Path) -> DictionaryConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise DictionaryError("Dictionary config must be a mapping")

    dictionary_section = raw.get("dictionary")
    if not isinstance(dictionary_section, dict):
        raise DictionaryError("'dictionary' section is required")

    name = dictionary_section.get("name")
    if not isinstance(name, str) or not name:
        raise DictionaryError("dictionary.name must be a non-empty string")

    entries_value = dictionary_section.get("entries")
    if not isinstance(entries_value, list) or not entries_value:
        raise DictionaryError("dictionary.entries must be a non-empty list")

    entries: list[dict[str, object]] = []
    for item in entries_value:
        if isinstance(item, str):
            entries.append({"token": item})
        elif isinstance(item, dict):
            if "token" not in item and "value" not in item:
                raise DictionaryError("dictionary entry mappings must include 'token' or 'value'")
            normalized = {str(key): item[key] for key in item}
            if "token" not in normalized and "value" in normalized:
                normalized["token"] = normalized["value"]
            entries.append(normalized)
        else:
            raise DictionaryError("dictionary entries must be strings or mappings")

    metadata = dictionary_section.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DictionaryError("dictionary.metadata must be a mapping")
    metadata_normalized = {str(key): metadata[key] for key in metadata}

    shuffle = bool(dictionary_section.get("shuffle", False))

    seed_value = raw.get("seed")
    if seed_value is not None and not isinstance(seed_value, int):
        raise DictionaryError("seed must be an integer when provided")

    output_section = raw.get("output")
    if not isinstance(output_section, dict):
        raise DictionaryError("'output' section is required")
    output_path_value = output_section.get("path")
    if not isinstance(output_path_value, str) or not output_path_value:
        raise DictionaryError("output.path must be a non-empty string")

    return DictionaryConfig(
        name=name,
        entries=[dict(entry) for entry in entries],
        metadata=metadata_normalized,
        shuffle=shuffle,
        seed=seed_value,
        output_path=(path.parent / output_path_value).resolve(),
        raw=_canonicalize(raw),
    )


def build_dictionary(config: DictionaryConfig) -> dict[str, object]:
    entries = [dict(item) for item in config.entries]
    if config.shuffle and entries:
        rng = random.Random(config.seed)
        rng.shuffle(entries)
    hash_value = _hash_payload(entries)

    payload = {
        "name": config.name,
        "entries": entries,
        "metadata": config.metadata,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "shuffle": config.shuffle,
        "hash": hash_value,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "name": config.name,
        "entries": len(entries),
        "hash": hash_value,
        "path": str(config.output_path),
        "created_utc": payload["created_utc"],
        "config": config.raw,
    }

    manifest_path = config.output_path.parent / "dict_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _canonicalize(value: Any) -> object:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _hash_payload(payload: list[dict[str, object]]) -> str:
    import hashlib

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_dictionary_from_grammar(
    grammar: Grammar,
    out_path: Path,
    add_specials: list[str] | None = None,
) -> dict[str, object]:
    """Generate a dictionary file from a grammar definition."""

    specials = add_specials or []
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        normalized = token.strip()
        if not normalized:
            return
        if normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)

    for token in specials:
        _add(token)

    for values in grammar.entities.values():
        for value in values:
            _add(value)

    pattern = re.compile(r"[A-Za-z0-9_<>{}/-]+")
    for template in grammar.templates:
        for candidate in pattern.findall(template):
            if candidate.startswith("{") and candidate.endswith("}"):
                continue
            _add(candidate)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    token2id = {token: idx for idx, token in enumerate(tokens)}
    id2token_path = out_path.with_suffix(".id2token.tsv")
    with id2token_path.open("w", encoding="utf-8") as handle:
        for token, idx in token2id.items():
            handle.write(f"{idx}\t{token}\n")
    token2id_path = out_path.with_suffix(".token2id.json")
    token2id_path.write_text(json.dumps(token2id, indent=2, ensure_ascii=False), encoding="utf-8")

    specials_path = out_path.with_suffix(".special_tokens.json")
    specials_path.write_text(
        json.dumps(tokens[: len(specials)], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "vocab_path": str(out_path),
        "token2id_path": str(token2id_path),
        "id2token_path": str(id2token_path),
        "tokens": len(tokens),
        "special_tokens": tokens[: len(specials)],
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "DictionaryConfig",
    "DictionaryError",
    "build_dictionary",
    "generate_dictionary_from_grammar",
    "load_dictionary_config",
]
