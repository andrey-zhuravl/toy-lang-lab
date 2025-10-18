"""Fixed embeddings generator for Stage A2."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


class EmbeddingsError(RuntimeError):
    """Raised when embeddings generation fails."""


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Configuration for embeddings generation."""

    name: str
    embedding_type: Literal["char", "word"]
    dim: int
    init: Literal["seeded_random", "uniform"]
    seed: int
    tokens: tuple[str, ...]
    output_path: Path
    raw: dict[str, object]


def load_embeddings_config(path: Path) -> EmbeddingsConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise EmbeddingsError("Embeddings config must be a mapping")

    embeddings_section = raw.get("embeddings")
    if not isinstance(embeddings_section, dict):
        raise EmbeddingsError("'embeddings' section is required")

    name = embeddings_section.get("name")
    if not isinstance(name, str) or not name:
        raise EmbeddingsError("embeddings.name must be a non-empty string")

    embedding_type = embeddings_section.get("type", "char")
    if embedding_type not in {"char", "word"}:
        raise EmbeddingsError("embeddings.type must be 'char' or 'word'")

    dim = embeddings_section.get("dim")
    if not isinstance(dim, int) or dim <= 0:
        raise EmbeddingsError("embeddings.dim must be a positive integer")

    init = embeddings_section.get("init", "seeded_random")
    if init not in {"seeded_random", "uniform"}:
        raise EmbeddingsError("embeddings.init must be 'seeded_random' or 'uniform'")

    seed_value = embeddings_section.get("seed", 0)
    if not isinstance(seed_value, int):
        raise EmbeddingsError("embeddings.seed must be an integer")

    tokens_value = embeddings_section.get("tokens")
    tokens: tuple[str, ...]
    if tokens_value is None:
        tokens = tuple(_default_tokens(embedding_type))
    elif isinstance(tokens_value, list) and all(isinstance(item, str) for item in tokens_value):
        tokens = tuple(tokens_value)
    else:
        raise EmbeddingsError("embeddings.tokens must be a list of strings when provided")

    output_section = raw.get("output")
    if not isinstance(output_section, dict):
        raise EmbeddingsError("'output' section is required")
    output_path_value = output_section.get("path")
    if not isinstance(output_path_value, str) or not output_path_value:
        raise EmbeddingsError("output.path must be a non-empty string")

    return EmbeddingsConfig(
        name=name,
        embedding_type=embedding_type,  # type: ignore[arg-type]
        dim=dim,
        init=init,  # type: ignore[arg-type]
        seed=seed_value,
        tokens=tokens,
        output_path=(path.parent / output_path_value).resolve(),
        raw=_canonicalize(raw),
    )


def build_embeddings(config: EmbeddingsConfig) -> dict[str, object]:
    rng = random.Random(config.seed)
    if not config.tokens:
        raise EmbeddingsError("No tokens provided for embeddings generation")

    vectors: list[list[float]] = []
    for _ in config.tokens:
        vector = [_generate_value(rng, config.init) for _ in range(config.dim)]
        vectors.append(vector)

    table = pa.table(
        {
            "token": list(config.tokens),
            "values": pa.array(vectors, type=pa.list_(pa.float32())),
        }
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, config.output_path)

    hash_value = _hash_file(config.output_path)

    manifest = {
        "name": config.name,
        "type": config.embedding_type,
        "dim": config.dim,
        "init": config.init,
        "seed": config.seed,
        "tokens": len(config.tokens),
        "created_utc": datetime.now(UTC).isoformat(),
        "embeddings_hash": hash_value,
        "path": str(config.output_path),
        "config": config.raw,
    }

    manifest_path = config.output_path.parent / "embeddings_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _generate_value(rng: random.Random, init: str) -> float:
    if init == "uniform":
        return rng.uniform(-1.0, 1.0)
    return rng.random() * 2 - 1.0


def _default_tokens(kind: str) -> list[str]:
    if kind == "word":
        return [
            "the",
            "a",
            "an",
            "to",
            "and",
            "of",
            "in",
            "is",
            "you",
            "it",
        ]
    base_chars = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    punctuation = " .,!?"
    return list(base_chars + digits + punctuation)


def _canonicalize(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _hash_file(path: Path) -> str:
    import hashlib

    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


__all__ = ["EmbeddingsConfig", "EmbeddingsError", "build_embeddings", "load_embeddings_config"]
