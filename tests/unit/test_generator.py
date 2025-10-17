from __future__ import annotations

import random
from pathlib import Path

from tlg.config import (
    DatasetConfig,
    DatasetSplits,
    NoiseConfig,
    TlgConfig,
    TokenizerConfig,
)
from tlg.generator import NoiseApplier, compute_grammar_hash, generate_dataset
from tlg.grammar import Grammar


def make_config(tmp_path: Path) -> tuple[TlgConfig, Grammar]:
    grammar_path = tmp_path / "grammar.yaml"
    config_file = tmp_path / "config.yaml"
    grammar_path.write_text(
        """entities:
  person: [Ann]
  item: [book]
  location: [library]
templates:
  - "{person} found a {item} at the {location}."
""",
        encoding="utf-8",
    )
    raw = {
        "dataset": {
            "name": "toy",
            "format": "jsonl",
            "splits": {"train": 1, "valid": 1, "test": 1},
        },
        "seed": 3,
        "tasks": ["lm", "cls", "seq2seq"],
        "grammar_yaml": "grammar.yaml",
        "tokenizer": {"type": "char", "lowercase": False},
        "noise": {},
    }
    config_yaml = "\n".join(
        [
            "dataset:",
            "  name: toy",
            "  format: jsonl",
            "  splits:",
            "    train: 1",
            "    valid: 1",
            "    test: 1",
            "seed: 3",
            "tasks:",
            "  - lm",
            "  - cls",
            "  - seq2seq",
            "grammar_yaml: grammar.yaml",
            "tokenizer:",
            "  type: char",
            "  lowercase: false",
            "noise: {}",
        ]
    )
    config_file.write_text(f"{config_yaml}\n", encoding="utf-8")
    grammar_raw = {
        "entities": {"person": ["Ann"], "item": ["book"], "location": ["library"]},
        "templates": ["{person} found a {item} at the {location}."],
    }
    grammar = Grammar(
        entities=grammar_raw["entities"],
        templates=grammar_raw["templates"],
        template_weights=None,
        raw=grammar_raw,
    )
    config = TlgConfig(
        dataset=DatasetConfig(
            name="toy",
            format="jsonl",
            splits=DatasetSplits(train=1, valid=1, test=1),
        ),
        seed=3,
        tasks=["lm", "cls", "seq2seq"],
        grammar_yaml=grammar_path,
        tokenizer=TokenizerConfig(type="char", lowercase=False),
        noise=NoiseConfig(),
        config_path=config_file,
        raw_dict=raw,
    )
    return config, grammar


def test_noise_applier_deterministic() -> None:
    applier = NoiseApplier(NoiseConfig(drop_prob=1.0))
    text = applier.apply("abc", random.Random(0))
    assert text in {"ab", "bc", "ac"}


def test_generate_dataset(tmp_path: Path) -> None:
    cfg, grammar = make_config(tmp_path)
    dataset, grammar_hash = generate_dataset(cfg, grammar)
    assert grammar_hash == compute_grammar_hash(grammar)
    assert set(dataset) == {"train", "valid", "test"}
    assert len(dataset["train"]) == len(cfg.tasks)
    record = dataset["train"][0]
    assert record["task"] in cfg.tasks
