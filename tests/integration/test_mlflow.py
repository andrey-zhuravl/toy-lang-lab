from types import SimpleNamespace

import pytest

from tlg.mlflow_logger import log_run


class DummyRun:
    def __init__(self) -> None:
        self.info = SimpleNamespace(run_id="123")

    def __enter__(self) -> "DummyRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


class DummyMlflow:
    def __init__(self) -> None:
        self.params = {}
        self.tags = {}
        self.artifacts = {}

    def start_run(self, run_name: str):  # pragma: no cover - simple pass-through
        self.run_name = run_name
        return DummyRun()

    def log_param(self, key: str, value: object) -> None:
        self.params[key] = value

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def log_dict(self, data, artifact_file: str) -> None:
        self.artifacts[artifact_file] = data


@pytest.fixture
def dummy_mlflow(monkeypatch):
    module = DummyMlflow()
    monkeypatch.setattr("tlg.mlflow_logger.mlflow", module)
    return module


def test_log_run_enabled(dummy_mlflow, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file://mlruns")
    run_id = log_run(
        params={"seed": 1},
        tags={"stage": "A1"},
        manifest={"dataset_hash": "abc"},
        stats={"train": {"num_records": 1.0}},
        enabled=True,
    )
    assert run_id == "123"
    assert dummy_mlflow.params["seed"] == 1
    assert dummy_mlflow.tags["stage"] == "A1"
    assert "manifest.json" in dummy_mlflow.artifacts
