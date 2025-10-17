"""Configuration loading and validation for Toy Lang Lab."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ALLOWED_TASKS: list[str] = ["lm", "seq2seq", "cls"]
ALLOWED_FORMATS: list[str] = ["jsonl", "parquet"]
TOKENIZER_TYPES: list[str] = ["char", "byte"]


@dataclass(frozen=True)
class DatasetSplits:
    """Dataset split sizes."""

    train: int
    valid: int
    test: int

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset level configuration."""

    name: str
    format: Literal["jsonl", "parquet"]
    splits: DatasetSplits


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer settings."""

    type: Literal["char", "byte"]
    lowercase: bool = False


@dataclass(frozen=True)
class NoiseConfig:
    """Noise parameters with optional probabilities."""

    drop_prob: float = 0.0
    typo_prob: float = 0.0
    swap_prob: float = 0.0
    synonym_prob: float = 0.0


@dataclass(frozen=True)
class TlgConfig:
    """Top-level configuration."""

    dataset: DatasetConfig
    seed: int
    tasks: list[str]
    grammar_yaml: Path
    tokenizer: TokenizerConfig
    noise: NoiseConfig
    config_path: Path
    raw_dict: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return canonicalize(self.raw_dict)


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


def canonicalize(data: object) -> object:
    """Recursively sort dictionaries for deterministic serialization."""

    if isinstance(data, Mapping):
        return {k: canonicalize(data[k]) for k in sorted(data)}
    if isinstance(data, list):
        return [canonicalize(item) for item in data]
    return data


def _ensure_positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer")
    return value


def _ensure_probability(value: object, key: str) -> float:
    if not isinstance(value, int | float) or not (0.0 <= float(value) <= 1.0):
        raise ConfigError(f"'{key}' must be between 0 and 1")
    return float(value)


def load_config(path: Path) -> TlgConfig:
    """Load a configuration file from YAML and validate it."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, MutableMapping):
        raise ConfigError("Configuration must be a mapping at top level")

    dataset_section = raw.get("dataset")
    if not isinstance(dataset_section, MutableMapping):
        raise ConfigError("'dataset' section must be provided")
    name = dataset_section.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("dataset.name must be a non-empty string")
    fmt = dataset_section.get("format", "jsonl")
    if fmt not in ALLOWED_FORMATS:
        raise ConfigError(f"dataset.format must be one of {ALLOWED_FORMATS}")

    splits_section = dataset_section.get("splits")
    if not isinstance(splits_section, MutableMapping):
        raise ConfigError("dataset.splits must be a mapping")
    train = _ensure_positive_int(splits_section.get("train"), "dataset.splits.train")
    valid = _ensure_positive_int(splits_section.get("valid"), "dataset.splits.valid")
    test = _ensure_positive_int(splits_section.get("test"), "dataset.splits.test")

    dataset = DatasetConfig(
        name=name,
        format=fmt,  # type: ignore[arg-type]
        splits=DatasetSplits(train=train, valid=valid, test=test),
    )

    seed_value = raw.get("seed")
    if not isinstance(seed_value, int):
        raise ConfigError("'seed' must be an integer")

    tasks_value = raw.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise ConfigError("'tasks' must be a non-empty list")
    for task in tasks_value:
        if task not in ALLOWED_TASKS:
            raise ConfigError(f"Unknown task '{task}'")

    tokenizer_section = raw.get("tokenizer", {})
    if not isinstance(tokenizer_section, MutableMapping):
        raise ConfigError("'tokenizer' must be a mapping")
    tokenizer_type = tokenizer_section.get("type", "char")
    if tokenizer_type not in TOKENIZER_TYPES:
        raise ConfigError(f"Unsupported tokenizer.type '{tokenizer_type}'")
    lowercase = bool(tokenizer_section.get("lowercase", False))
    tokenizer = TokenizerConfig(type=tokenizer_type, lowercase=lowercase)  # type: ignore[arg-type]

    noise_section = raw.get("noise", {})
    if not isinstance(noise_section, MutableMapping):
        raise ConfigError("'noise' must be a mapping")
    noise = NoiseConfig(
        drop_prob=_ensure_probability(noise_section.get("drop_prob", 0.0), "noise.drop_prob"),
        typo_prob=_ensure_probability(noise_section.get("typo_prob", 0.0), "noise.typo_prob"),
        swap_prob=_ensure_probability(noise_section.get("swap_prob", 0.0), "noise.swap_prob"),
        synonym_prob=_ensure_probability(
            noise_section.get("synonym_prob", 0.0), "noise.synonym_prob"
        ),
    )

    grammar_path = raw.get("grammar_yaml")
    if not isinstance(grammar_path, str) or not grammar_path:
        raise ConfigError("'grammar_yaml' must be a path string")

    resolved = path.parent / grammar_path

    return TlgConfig(
        dataset=dataset,
        seed=seed_value,
        tasks=list(tasks_value),
        grammar_yaml=resolved,
        tokenizer=tokenizer,
        noise=noise,
        config_path=path,
        raw_dict=canonicalize(raw),
    )


def compute_config_hash(raw: Mapping[str, object]) -> str:
    """Compute SHA256 hash for a canonical configuration mapping."""

    canonical = canonicalize(raw)
    serialized = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return _sha256(serialized.encode("utf-8"))


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
