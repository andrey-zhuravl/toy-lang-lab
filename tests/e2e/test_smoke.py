import json
import subprocess
import sys
from pathlib import Path


def run_cli(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([sys.executable, "-m", "tlg.cli", *args], check=True)


def test_smoke_build_and_stats(tmp_path: Path) -> None:
    out_dir = tmp_path / "dev"
    run_cli(["build-data", "conf=conf/data/toy_small.yaml", f"out={out_dir}"])
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ["dataset_name", "dataset_hash", "config_hash", "git_rev", "row_counts"]:
        assert key in manifest
    result = subprocess.run(
        [sys.executable, "-m", "tlg.cli", "stats", f"path={out_dir}"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "train" in result.stdout
