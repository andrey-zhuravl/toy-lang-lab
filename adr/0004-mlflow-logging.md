# ADR 0004: Minimal MLflow Logging

## Status
Accepted

## Context
Stage A1 requires optional MLflow integration to publish build metadata. The integration must be
minimal, optional, and easy to disable when MLflow is unavailable.

## Decision
Implement a thin wrapper (`tlg.mlflow_logger`) that checks for MLflow availability and the
`MLFLOW_TRACKING_URI` environment variable (or an explicit flag). When enabled, it starts a run,
logs parameters, tags, and uploads the manifest and stats JSON artifacts.

## Consequences
Users with MLflow available gain reproducibility tracking without impacting offline workflows. The
wrapper can be extended in later stages for richer logging.
