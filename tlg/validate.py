"""Dataset validation utilities for Stage A2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    split_hashes: dict[str, str]
    dataset_hash: str | None


def validate_dataset(output_dir: Path) -> ValidationResult:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    errors: list[str] = []
    if not manifest_path.exists():
        errors.append("manifest.json is missing")
        return ValidationResult(False, errors, {}, None)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_hashes_manifest = manifest.get("split_hashes", {})
    if not isinstance(split_hashes_manifest, dict):
        errors.append("manifest.split_hashes must be a mapping")
        split_hashes_manifest = {}

    computed_split_hashes: dict[str, str] = {}
    for split, declared_hash in split_hashes_manifest.items():
        split_path = output_dir / f"{split}.{manifest.get('format', 'jsonl')}"
        try:
            computed = _compute_split_hash(split_path, manifest.get("format", "jsonl"))
        except FileNotFoundError:
            errors.append(f"missing split file: {split_path}")
            continue
        if declared_hash != computed:
            errors.append(f"split hash mismatch for {split}")
        computed_split_hashes[split] = computed

    dataset_hash_manifest = manifest.get("dataset_hash")
    computed_dataset_hash = None
    if computed_split_hashes:
        computed_dataset_hash = _hash_joined(computed_split_hashes.values())
        if dataset_hash_manifest and dataset_hash_manifest != computed_dataset_hash:
            errors.append("dataset hash mismatch")

    if "tokenizer_hash" in manifest and not manifest.get("tokenizer_hash"):
        errors.append("tokenizer_hash present but empty")

    return ValidationResult(not errors, errors, computed_split_hashes, computed_dataset_hash)


def _compute_split_hash(path: Path, fmt: str) -> str:
    if fmt == "jsonl":
        canonical_lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                canonical_lines.append(_canonical_json(record))
        return _hash_joined(canonical_lines)
    if fmt == "parquet":
        table = pq.read_table(path)
        rows = table.to_pylist()
        canonical_lines = [_canonical_json(row) for row in rows]
        return _hash_joined(canonical_lines)
    raise FileNotFoundError(path)


def _canonical_json(record: Any) -> str:
    canonical = _canonicalize(record)
    return json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _hash_joined(lines: Any) -> str:
    import hashlib

    if isinstance(lines, dict):
        iterable = lines.values()
    else:
        iterable = lines
    joined = "\n".join(iterable)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


__all__ = ["ValidationResult", "validate_dataset"]
