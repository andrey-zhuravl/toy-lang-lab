"""Utilities for loading corpora and computing hashes for tokenizer workflows."""

from __future__ import annotations

import csv
import glob
import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


class CorpusError(RuntimeError):
    """Raised when tokenizer corpus loading fails."""


SUPPORTED_SUFFIXES = {".jsonl", ".csv", ".txt"}


@dataclass(frozen=True)
class CorpusSample:
    """Represents a single text sample loaded from the corpus."""

    text: str
    source: Path


def resolve_input_paths(inputs: Iterable[str]) -> list[Path]:
    """Resolve a sequence of user provided paths/globs to concrete files."""

    paths: list[Path] = []
    for item in inputs:
        expanded = list(sorted(glob.glob(os.fspath(item))))
        if not expanded:
            raise CorpusError(f"No files matched input '{item}'")
        for entry in expanded:
            path = Path(entry).resolve()
            if not path.is_file():
                raise CorpusError(f"Input path is not a file: {path}")
            if path.suffix not in SUPPORTED_SUFFIXES:
                raise CorpusError(
                    f"Unsupported input format '{path.suffix}' for {path}. "
                    "Expected one of .jsonl, .csv, .txt"
                )
            paths.append(path)
    if not paths:
        raise CorpusError("No input files provided")
    return paths


def iter_corpus_text(paths: Iterable[Path], text_key: str) -> Iterator[CorpusSample]:
    """Yield :class:`CorpusSample` items from the provided paths."""

    for path in paths:
        if path.suffix == ".jsonl":
            yield from _iter_jsonl(path, text_key)
        elif path.suffix == ".csv":
            yield from _iter_csv(path, text_key)
        elif path.suffix == ".txt":
            yield from _iter_txt(path)
        else:  # pragma: no cover - safeguarded by resolve_input_paths
            raise CorpusError(f"Unsupported file extension: {path.suffix}")


def compute_dataset_hash(paths: Iterable[Path]) -> str:
    """Compute a deterministic SHA256 hash across the provided files."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path, text_key: str) -> Iterator[CorpusSample]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get(text_key)
            if not isinstance(value, str):
                raise CorpusError(
                    f"Expected string field '{text_key}' in {path} but found {type(value)!r}"
                )
            yield CorpusSample(text=value, source=path)


def _iter_csv(path: Path, text_key: str) -> Iterator[CorpusSample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if text_key not in reader.fieldnames:
            raise CorpusError(
                f"Field '{text_key}' not present in CSV header for {path}"
            )
        for row in reader:
            value = row.get(text_key)
            if value is None:
                continue
            yield CorpusSample(text=str(value), source=path)


def _iter_txt(path: Path) -> Iterator[CorpusSample]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if stripped:
                yield CorpusSample(text=stripped, source=path)


__all__ = [
    "CorpusError",
    "CorpusSample",
    "compute_dataset_hash",
    "iter_corpus_text",
    "resolve_input_paths",
]
