from pathlib import Path

from tlg.config import DatasetConfig, DatasetSplits, NoiseConfig, TlgConfig, TokenizerConfig
from tlg.stats import compute_stats
from tlg.writer import write_dataset


def make_config(tmp_path: Path, fmt: str) -> TlgConfig:
    config_path = tmp_path / "config.yaml"
    grammar_path = tmp_path / "grammar.yaml"
    grammar_path.write_text("entities: {x: [1]}\ntemplates: ['{x}']\n", encoding="utf-8")
    raw = {
        "dataset": {
            "name": "toy",
            "format": fmt,
            "splits": {"train": 1, "valid": 1, "test": 1},
        },
        "seed": 1,
        "tasks": ["lm"],
        "grammar_yaml": str(grammar_path.name),
        "tokenizer": {"type": "char", "lowercase": False},
        "noise": {},
    }
    config_yaml = "\n".join(
        [
            "dataset:",
            "  name: toy",
            f"  format: {fmt}",
            "  splits:",
            "    train: 1",
            "    valid: 1",
            "    test: 1",
            "seed: 1",
            "tasks:",
            "  - lm",
            "grammar_yaml: grammar.yaml",
            "tokenizer:",
            "  type: char",
            "  lowercase: false",
            "noise: {}",
        ]
    )
    config_path.write_text(f"{config_yaml}\n", encoding="utf-8")
    return TlgConfig(
        dataset=DatasetConfig(
            name="toy",
            format=fmt,
            splits=DatasetSplits(train=1, valid=1, test=1),
        ),
        seed=1,
        tasks=["lm"],
        grammar_yaml=grammar_path,
        tokenizer=TokenizerConfig(type="char", lowercase=False),
        noise=NoiseConfig(),
        config_path=config_path,
        raw_dict=raw,
    )


def test_write_dataset_jsonl(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, "jsonl")
    dataset = {
        split: [
            {
                "id": f"{split}-lm-000000",
                "task": "lm",
                "text": f"sample-{split}",
                "meta": {"split": split, "seed": cfg.seed, "grammar_rev": "hash"},
            }
        ]
        for split in ("train", "valid", "test")
    }
    stats = compute_stats(dataset)
    result = write_dataset(cfg, dataset, "hash", stats, tmp_path)
    assert result.dataset_hash
    for split in ("train", "valid", "test"):
        assert (tmp_path / f"{split}.jsonl").exists()
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()


def test_write_dataset_parquet(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, "parquet")
    dataset = {
        split: [
            {
                "id": f"{split}-lm-000000",
                "task": "lm",
                "text": f"sample-{split}",
                "meta": {"split": split, "seed": cfg.seed, "grammar_rev": "hash"},
            }
        ]
        for split in ("train", "valid", "test")
    }
    stats = compute_stats(dataset)
    out_dir = tmp_path / "parquet"
    result = write_dataset(cfg, dataset, "hash", stats, out_dir)
    assert result.dataset_hash
    for split in ("train", "valid", "test"):
        assert (out_dir / f"{split}.parquet").exists()
