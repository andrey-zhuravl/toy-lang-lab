"""Dataset statistics utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _record_text(record: Mapping[str, object]) -> str:
    if "text" in record:
        return str(record["text"])
    if "src" in record:
        return str(record["src"])
    return ""


def _lengths(records: Iterable[Mapping[str, object]]) -> list[int]:
    return [len(_record_text(record)) for record in records]


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q * 100))


def compute_stats(
    dataset: Mapping[str, list[MutableMapping[str, object]]],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for split, records in dataset.items():
        lengths = _lengths(records)
        corpus = [_record_text(record) for record in records]
        vocab = sorted({char for sample in corpus for char in sample})
        stats[split] = {
            "num_records": float(len(records)),
            "avg_len": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "p50_len": _percentile(lengths, 0.5),
            "p95_len": _percentile(lengths, 0.95),
            "vocab_size_char": float(len(vocab)),
        }
    return stats


def load_split(path: Path) -> list[MutableMapping[str, object]]:
    if path.suffix == ".jsonl":
        records: list[MutableMapping[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
        return records
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        return table.to_pylist()
    raise ValueError(f"Unsupported split file {path}")


def load_dataset(output_dir: Path) -> dict[str, list[MutableMapping[str, object]]]:
    dataset: dict[str, list[MutableMapping[str, object]]] = {}
    for split in ("train", "valid", "test"):
        for ext in (".jsonl", ".parquet"):
            candidate = output_dir / f"{split}{ext}"
            if candidate.exists():
                dataset[split] = load_split(candidate)
                break
        else:
            raise FileNotFoundError(f"Missing split file for {split}")
    return dataset


def dataset_stats_from_dir(output_dir: Path) -> dict[str, dict[str, float]]:
    return compute_stats(load_dataset(output_dir))
