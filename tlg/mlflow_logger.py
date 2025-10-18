"""Minimal MLflow logging wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping

try:  # pragma: no cover - mlflow optional
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def is_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit and mlflow is not None
    return mlflow is not None and bool(os.getenv("MLFLOW_TRACKING_URI"))


def log_run(
    params: Mapping[str, object],
    tags: Mapping[str, str],
    manifest: Mapping[str, object],
    stats: Mapping[str, Mapping[str, float]],
    run_name: str = "tlg-build",
    enabled: bool | None = None,
) -> str | None:
    if not is_enabled(enabled):  # pragma: no cover - depends on env
        return None
    assert mlflow is not None  # for type-checkers
    with mlflow.start_run(run_name=run_name) as run:  # type: ignore[attr-defined]
        for key, value in params.items():
            mlflow.log_param(key, value)  # type: ignore[attr-defined]
        for key, value in tags.items():
            mlflow.set_tag(key, value)  # type: ignore[attr-defined]
        mlflow.log_dict(manifest, "manifest.json")  # type: ignore[attr-defined]
        mlflow.log_dict(stats, "stats.json")  # type: ignore[attr-defined]
        return run.info.run_id  # type: ignore[return-value]


def log_tokenizer_run(
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifact_dir: Path,
    manifest: Mapping[str, object],
    run_name: str = "tokenizer-B1",
    enabled: bool | None = None,
) -> str | None:
    if not is_enabled(enabled):  # pragma: no cover - depends on env
        return None
    assert mlflow is not None
    with mlflow.start_run(run_name=run_name) as run:  # type: ignore[attr-defined]
        for key, value in params.items():
            mlflow.log_param(key, value)  # type: ignore[attr-defined]
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))  # type: ignore[attr-defined]
        for key, value in tags.items():
            mlflow.set_tag(key, value)  # type: ignore[attr-defined]
        mlflow.log_dict(manifest, "tokenizer_manifest.json")  # type: ignore[attr-defined]
        mlflow.log_artifacts(str(artifact_dir), artifact_path="tokenizer")  # type: ignore[attr-defined]
        return run.info.run_id  # type: ignore[return-value]


def log_encoding_run(
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifact_paths: Mapping[str, str],
    manifest: Mapping[str, object],
    run_name: str = "encode-B1",
    enabled: bool | None = None,
) -> str | None:
    if not is_enabled(enabled):  # pragma: no cover
        return None
    assert mlflow is not None
    with mlflow.start_run(run_name=run_name) as run:  # type: ignore[attr-defined]
        for key, value in params.items():
            mlflow.log_param(key, value)  # type: ignore[attr-defined]
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))  # type: ignore[attr-defined]
        for key, value in tags.items():
            mlflow.set_tag(key, value)  # type: ignore[attr-defined]
        mlflow.log_dict(manifest, "encode_manifest.json")  # type: ignore[attr-defined]
        for name, path in artifact_paths.items():
            mlflow.log_artifact(path, artifact_path=f"encode/{name}")  # type: ignore[attr-defined]
        return run.info.run_id  # type: ignore[return-value]


__all__ = [
    "is_enabled",
    "log_encoding_run",
    "log_run",
    "log_tokenizer_run",
]
