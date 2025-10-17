"""Minimal MLflow logging wrapper."""

from __future__ import annotations

import os
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


__all__ = ["is_enabled", "log_run"]
