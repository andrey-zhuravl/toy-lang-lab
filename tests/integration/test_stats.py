from pathlib import Path

from typer.testing import CliRunner

from tlg.cli import app
from tlg.stats import dataset_stats_from_dir


def test_dataset_stats_from_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["build-data", "conf=conf/data/toy_small.yaml", f"out={out_dir}"])
    assert result.exit_code == 0
    stats = dataset_stats_from_dir(out_dir)
    assert "train" in stats
    assert stats["train"]["num_records"] > 0
