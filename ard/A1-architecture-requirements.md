# Toy Lang Lab A1 Architecture Requirements

## Context & Goals
Stage A1 delivers a deterministic synthetic data generator based on YAML grammars. The focus is on
reproducibility, simple noise injection, and integration with CLI tooling and MLflow.

## Constraints
- Python 3.11 runtime
- Deterministic hashing across runs and formats
- Optional MLflow logging without breaking offline environments

## Interfaces (CLI / DSL)
- `tlg build-data conf=... out=...`: generate splits, manifest, and hashes
- `tlg stats path=...`: compute dataset statistics for existing outputs
- Grammar DSL uses YAML with entities, templates, weights, and noise sections

## Data Contracts
- Records follow task-specific schemas (lm, seq2seq, cls)
- Manifest captures config hash, code hash, split hashes, and counts
- Stats expose record counts, text lengths, and character vocabulary size

## Risks
- Parquet hashing parity with JSONL ensured via canonical JSON serialization
- Config drift mitigated through hash tracking
- Optional MLflow to avoid runtime errors when tracking URI is absent

## Testing & Quality Gates
- Unit, integration, and smoke tests executed via `pytest`
- CI workflow runs linting, mypy, tests, and smoke pipeline
- Coverage target of 80% enforced in CI
