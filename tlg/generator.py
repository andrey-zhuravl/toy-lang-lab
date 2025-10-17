"""Dataset generation utilities."""

from __future__ import annotations

import json
import random
from collections.abc import MutableMapping
from dataclasses import dataclass
from hashlib import sha256

from .config import NoiseConfig, TlgConfig
from .grammar import Grammar


@dataclass
class NoiseApplier:
    config: NoiseConfig

    def apply(self, text: str, rng: random.Random) -> str:
        updated = text
        if self.config.drop_prob and rng.random() < self.config.drop_prob:
            updated = self._apply_drop(updated, rng)
        if self.config.typo_prob and rng.random() < self.config.typo_prob:
            updated = self._apply_typo(updated, rng)
        if self.config.swap_prob and rng.random() < self.config.swap_prob:
            updated = self._apply_swap(updated, rng)
        if self.config.synonym_prob and rng.random() < self.config.synonym_prob:
            updated = self._apply_synonym(updated)
        return updated

    @staticmethod
    def _apply_drop(text: str, rng: random.Random) -> str:
        if len(text) < 2:
            return text
        index = rng.randrange(len(text))
        return text[:index] + text[index + 1 :]

    @staticmethod
    def _apply_typo(text: str, rng: random.Random) -> str:
        if len(text) < 2:
            return text
        idx = rng.randrange(len(text) - 1)
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    @staticmethod
    def _apply_swap(text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        idx = rng.randrange(len(words) - 1)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)

    @staticmethod
    def _apply_synonym(text: str) -> str:
        # Placeholder for synonym substitution; Stage A1 keeps deterministic text.
        return text


def compute_grammar_hash(grammar: Grammar) -> str:
    canonical = json.dumps(grammar.raw, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _choose_template(grammar: Grammar, rng: random.Random) -> tuple[str, int]:
    if grammar.template_weights:
        choice = rng.choices(
            range(len(grammar.templates)),
            weights=grammar.template_weights,
            k=1,
        )[0]
        return grammar.templates[choice], choice
    index = rng.randrange(len(grammar.templates))
    return grammar.templates[index], index


def _fill_template(template: str, grammar: Grammar, rng: random.Random) -> dict[str, str]:
    values: dict[str, str] = {}
    result = template
    for entity, options in grammar.entities.items():
        placeholder = "{" + entity + "}"
        if placeholder not in template:
            continue
        choice = rng.choice(options)
        result = result.replace(placeholder, choice)
        values[entity] = choice
    return {"text": result, **values}


def generate_dataset(
    config: TlgConfig, grammar: Grammar
) -> tuple[dict[str, list[MutableMapping[str, object]]], str]:
    noise = NoiseApplier(config.noise)
    grammar_hash = compute_grammar_hash(grammar)
    splits = {
        "train": config.dataset.splits.train,
        "valid": config.dataset.splits.valid,
        "test": config.dataset.splits.test,
    }
    split_offsets = {"train": 0, "valid": 1, "test": 2}
    dataset: dict[str, list[MutableMapping[str, object]]] = {split: [] for split in splits}

    for split, count in splits.items():
        for index in range(count):
            sample_seed = config.seed + split_offsets[split] * 1_000_000 + index
            split_rng = random.Random(sample_seed)
            template, template_index = _choose_template(grammar, split_rng)
            filled = _fill_template(template, grammar, split_rng)
            text = noise.apply(filled["text"], split_rng)
            meta = {
                "split": split,
                "seed": config.seed,
                "grammar_rev": grammar_hash,
                "template_index": template_index,
            }
            for task in config.tasks:
                record_id = f"{split}-{task}-{index:06d}"
                if task == "lm":
                    dataset[split].append(
                        {
                            "id": record_id,
                            "task": "lm",
                            "text": text,
                            "meta": meta,
                        }
                    )
                elif task == "seq2seq":
                    dataset[split].append(
                        {
                            "id": record_id,
                            "task": "seq2seq",
                            "src": text,
                            "tgt": _build_seq2seq_target(filled),
                            "meta": meta,
                        }
                    )
                elif task == "cls":
                    dataset[split].append(
                        {
                            "id": record_id,
                            "task": "cls",
                            "text": text,
                            "label": _label_from_template(template),
                            "meta": meta,
                        }
                    )
                else:  # pragma: no cover - config validation prevents this
                    raise ValueError(f"Unsupported task '{task}'")
    return dataset, grammar_hash


def _build_seq2seq_target(filled: dict[str, str]) -> str:
    if "item" in filled and "location" in filled:
        return f"{filled['item']} -> {filled['location']}"
    return filled.get("text", "")


def _label_from_template(template: str) -> str:
    if "found" in template:
        return "found"
    if "lost" in template:
        return "lost"
    if "traded" in template:
        return "traded"
    return "unknown"
