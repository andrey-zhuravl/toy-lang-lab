"""Dataset writing and hashing utilities."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .config import TlgConfig, compute_config_hash


@dataclass
class SplitWriteResult:
    path: Path
    num_records: int
    hash: str


@dataclass
class DatasetWriteResult:
    split_results: dict[str, SplitWriteResult]
    dataset_hash: str
    manifest: dict[str, object]


class WriterError(RuntimeError):
    """Raised when writing outputs fails."""


def _canonical_json(record: Mapping[str, object]) -> str:
    def _canonicalize(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: _canonicalize(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [_canonicalize(item) for item in value]
        return value

    canonical = _canonicalize(record)
    return json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _hash_records(records: Iterable[Mapping[str, object]]) -> tuple[str, list[str]]:
    lines: list[str] = []
    for record in records:
        lines.append(_canonical_json(record))
    joined = "\n".join(lines)
    digest = _sha256(joined.encode("utf-8"))
    return digest, lines


def _sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def write_jsonl(records: list[MutableMapping[str, object]], path: Path) -> SplitWriteResult:
    digest, lines = _hash_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    return SplitWriteResult(path=path, num_records=len(records), hash=digest)


def write_parquet(records: list[MutableMapping[str, object]], path: Path) -> SplitWriteResult:
    digest, lines = _hash_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
    return SplitWriteResult(path=path, num_records=len(records), hash=digest)


def write_dataset(
    config: TlgConfig,
    dataset: dict[str, list[MutableMapping[str, object]]],
    grammar_hash: str,
    stats: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> DatasetWriteResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_results: dict[str, SplitWriteResult] = {}
    split_hashes: dict[str, str] = {}
    total_hash_payload: list[str] = []

    for split, records in dataset.items():
        file_path = output_dir / f"{split}.{config.dataset.format}"
        if config.dataset.format == "jsonl":
            result = write_jsonl(records, file_path)
        elif config.dataset.format == "parquet":
            result = write_parquet(records, file_path)
        else:  # pragma: no cover - config validation prevents this
            raise WriterError(f"Unsupported format {config.dataset.format}")
        split_results[split] = result
        split_hashes[split] = result.hash
        total_hash_payload.append(result.hash)

    dataset_hash = _sha256("\n".join(total_hash_payload).encode("utf-8"))

    manifest = _build_manifest(config, grammar_hash, split_results, dataset_hash, stats, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return DatasetWriteResult(
        split_results=split_results,
        dataset_hash=dataset_hash,
        manifest=manifest,
    )


def _build_manifest(
    config: TlgConfig,
    grammar_hash: str,
    split_results: Mapping[str, SplitWriteResult],
    dataset_hash: str,
    stats: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> dict[str, object]:
    config_hash = compute_config_hash(config.to_dict())
    git_rev = _current_git_rev(output_dir)
    code_hash = _sha256(f"{__version__}:{git_rev}".encode())

    row_counts = {name: result.num_records for name, result in split_results.items()}
    split_hashes = {name: result.hash for name, result in split_results.items()}

    return {
        "dataset_name": config.dataset.name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_rev": git_rev,
        "code_hash": code_hash,
        "config_path": str(config.config_path),
        "config_hash": config_hash,
        "splits": config.dataset.splits.as_dict(),
        "format": config.dataset.format,
        "split_hashes": split_hashes,
        "dataset_hash": dataset_hash,
        "seed": config.seed,
        "grammar_file": str(config.grammar_yaml),
        "grammar_hash": grammar_hash,
        "row_counts": row_counts,
        "stats": stats,
    }


def _current_git_rev(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # pragma: no cover - git may not be available in tests
        return "unknown"
    return result.stdout.strip()
