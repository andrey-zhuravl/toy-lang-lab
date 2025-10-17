# ADR 0002: Dataset Format (JSONL vs Parquet)

## Status
Accepted

## Context
Toy Lang Lab must emit datasets consumable by downstream tooling. JSONL is widely interoperable,
while Parquet offers efficient columnar storage.

## Decision
Support both JSONL and Parquet outputs. JSONL is the default format, with Parquet enabled through the
Hydra config or CLI override. Both formats share canonical hashing by projecting records to JSON
strings before hashing.

## Consequences
Users can choose the best format for their workflow. The hashing contract ensures deterministic
manifests and equality checks across formats.
