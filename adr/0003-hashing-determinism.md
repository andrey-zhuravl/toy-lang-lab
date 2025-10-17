# ADR 0003: Hashing & Reproducibility

## Status
Accepted

## Context
Deterministic hashing is critical for verifying reproducibility. Hashes must remain stable across
runs and independent of output format.

## Decision
Canonicalize each record by sorting keys recursively and serializing to UTF-8 JSON lines. Split hashes
are computed over newline-joined canonical lines, and the dataset hash is calculated from the split
hashes. The manifest records hashes, configuration info, and grammar revision.

## Consequences
Dataset builds can be compared efficiently. The hashing strategy also ensures Parquet outputs are
aligned with JSONL by using the same canonical lines for hashing.
