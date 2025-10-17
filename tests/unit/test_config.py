from pathlib import Path

import yaml

from tlg.config import ConfigError, compute_config_hash, load_config


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_config_roundtrip(tmp_path: Path) -> None:
    data = {
        "dataset": {
            "name": "toy",
            "format": "jsonl",
            "splits": {"train": 2, "valid": 1, "test": 1},
        },
        "seed": 7,
        "tasks": ["lm", "cls"],
        "grammar_yaml": "grammar.yaml",
        "tokenizer": {"type": "char", "lowercase": True},
        "noise": {"drop_prob": 0.1},
    }
    grammar_path = tmp_path / "grammar.yaml"
    grammar_path.write_text("entities: {x: [1]}\ntemplates: ['{x}']\n", encoding="utf-8")
    config_path = _write_config(tmp_path, data)

    cfg = load_config(config_path)
    assert cfg.dataset.name == "toy"
    assert cfg.dataset.splits.train == 2
    assert cfg.seed == 7
    assert cfg.grammar_yaml == grammar_path
    assert cfg.config_path == config_path
    assert compute_config_hash(cfg.to_dict())


def test_invalid_task(tmp_path: Path) -> None:
    data = {
        "dataset": {
            "name": "toy",
            "format": "jsonl",
            "splits": {"train": 1, "valid": 1, "test": 1},
        },
        "seed": 1,
        "tasks": ["unknown"],
        "grammar_yaml": "grammar.yaml",
    }
    (tmp_path / "grammar.yaml").write_text(
        "entities: {x: [1]}\ntemplates: ['{x}']\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, data)
    try:
        load_config(config_path)
    except ConfigError as exc:
        assert "Unknown task" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ConfigError")
