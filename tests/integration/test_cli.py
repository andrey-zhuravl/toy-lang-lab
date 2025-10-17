from pathlib import Path

from typer.testing import CliRunner

from tlg.cli import app

runner = CliRunner()


def test_build_data_jsonl(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["build-data", "conf=conf/data/toy_small.yaml", f"out={out_dir}"])
    assert result.exit_code == 0, result.output
    manifest = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert "dataset_hash" in manifest


def test_build_data_parquet(tmp_path: Path) -> None:
    out_dir = tmp_path / "parquet"
    result = runner.invoke(
        app,
        [
            "build-data",
            "conf=conf/data/toy_small.yaml",
            f"out={out_dir}",
            "format=parquet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "train.parquet").exists()


def test_stats_command(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    runner.invoke(app, ["build-data", "conf=conf/data/toy_small.yaml", f"out={out_dir}"])
    result = runner.invoke(app, ["stats", f"path={out_dir}"])
    assert result.exit_code == 0
    assert "train" in result.output
