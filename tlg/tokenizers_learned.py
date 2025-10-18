"""Utilities for training and applying learned tokenizers (Stage A2)."""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml


class LearnedTokenizerError(RuntimeError):
    """Raised when learned tokenizer operations fail."""


@dataclass(frozen=True)
class BpeTrainingConfig:
    """Configuration for training a simple BPE tokenizer."""

    name: str
    algorithm: Literal["bpe"]
    vocab_size: int
    min_frequency: int
    unicode_normalize: str | None
    lowercase: bool
    seed: int
    tiebreak: Literal["lexicographic"]
    input_path: Path
    text_field: str
    limit: int | None
    output_dir: Path
    raw: dict[str, object]


@dataclass
class LearnedTokenizerModel:
    """In-memory representation of a trained tokenizer."""

    name: str
    algorithm: str
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    lowercase: bool
    unicode_normalize: str | None
    config: dict[str, object]


@dataclass
class TokenizerManifest:
    """Metadata persisted alongside tokenizer artifacts."""

    name: str
    algorithm: str
    vocab_path: Path
    merges_path: Path
    manifest_path: Path
    tokenizer_hash: str
    config: dict[str, object]


@dataclass
class LearnedTokenizer:
    """Runtime helper for applying a trained tokenizer."""

    name: str
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    lowercase: bool
    unicode_normalize: str | None
    tokenizer_hash: str
    unk_token: str = "<unk>"

    def __post_init__(self) -> None:
        self._merge_ranks: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._unk_id = self.vocab.get(self.unk_token, 0)

    def encode(self, text: str) -> list[str]:
        """Return BPE tokens for the provided text."""

        processed = _preprocess_text(text, lowercase=self.lowercase, normalize=self.unicode_normalize)
        if not processed:
            return []
        tokens: list[str] = []
        for word in processed.split():
            tokens.extend(self._encode_word(word))
        return tokens

    def encode_with_ids(self, text: str) -> tuple[list[str], list[int]]:
        tokens = self.encode(text)
        ids = [self.vocab.get(token, self._unk_id) for token in tokens]
        return tokens, ids

    def _encode_word(self, word: str) -> list[str]:
        if not word:
            return []
        symbols: list[str] = list(word) + ["</w>"]
        while True:
            pairs = _get_pairs(symbols)
            if not pairs:
                break
            best_pair = min(
                pairs,
                key=lambda pair: self._merge_ranks.get(pair, float("inf")),
            )
            if best_pair not in self._merge_ranks:
                break
            symbols = _merge_symbols(symbols, best_pair)
        if symbols and symbols[-1] == "</w>":
            symbols = symbols[:-1]
        return symbols

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self._unk_id)


def load_bpe_training_config(path: Path) -> BpeTrainingConfig:
    """Load a tokenizer training configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, MutableMapping):
        raise LearnedTokenizerError("Tokenizer config must be a mapping")

    tokenizer_section = raw.get("tokenizer")
    if not isinstance(tokenizer_section, MutableMapping):
        raise LearnedTokenizerError("'tokenizer' section is required")

    name = _require_string(tokenizer_section.get("name"), "tokenizer.name")
    algorithm = tokenizer_section.get("algorithm", "bpe")
    if algorithm != "bpe":
        raise LearnedTokenizerError("Only BPE algorithm is supported in Stage A2")
    vocab_size = _require_positive_int(tokenizer_section.get("vocab_size"), "tokenizer.vocab_size")
    min_frequency = _require_positive_int(
        tokenizer_section.get("min_frequency", 1),
        "tokenizer.min_frequency",
    )
    unicode_normalize = tokenizer_section.get("unicode_normalize")
    if unicode_normalize is not None and not isinstance(unicode_normalize, str):
        raise LearnedTokenizerError("tokenizer.unicode_normalize must be a string")
    lowercase = bool(tokenizer_section.get("lowercase", False))
    seed = tokenizer_section.get("seed", 0)
    if not isinstance(seed, int):
        raise LearnedTokenizerError("tokenizer.seed must be an integer")
    tiebreak = tokenizer_section.get("tiebreak", "lexicographic")
    if tiebreak != "lexicographic":
        raise LearnedTokenizerError("Only lexicographic tiebreak is supported")

    train_section = raw.get("train")
    if not isinstance(train_section, MutableMapping):
        raise LearnedTokenizerError("'train' section is required")
    input_value = _require_string(train_section.get("input"), "train.input")
    text_field = _require_string(train_section.get("text_field"), "train.text_field")
    limit_value = train_section.get("limit")
    limit: int | None
    if limit_value is None:
        limit = None
    else:
        limit = _require_positive_int(limit_value, "train.limit")

    output_section = raw.get("output")
    if not isinstance(output_section, MutableMapping):
        raise LearnedTokenizerError("'output' section is required")
    output_dir_value = _require_string(output_section.get("dir"), "output.dir")

    return BpeTrainingConfig(
        name=name,
        algorithm="bpe",
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        unicode_normalize=unicode_normalize,
        lowercase=lowercase,
        seed=seed,
        tiebreak="lexicographic",
        input_path=(path.parent / input_value).resolve(),
        text_field=text_field,
        limit=limit,
        output_dir=(path.parent / output_dir_value).resolve(),
        raw=_canonicalize(raw),
    )


def train_bpe(config: BpeTrainingConfig) -> LearnedTokenizerModel:
    """Train a deterministic BPE tokenizer from the provided config."""

    corpus = list(_iter_training_corpus(config))
    if not corpus:
        raise LearnedTokenizerError("Training corpus is empty")

    vocab_counter: Counter[tuple[str, ...]] = Counter()
    alphabet: set[str] = set()
    for word in corpus:
        processed_word = _preprocess_text(
            word,
            lowercase=config.lowercase,
            normalize=config.unicode_normalize,
        )
        if not processed_word:
            continue
        symbols = tuple(list(processed_word) + ["</w>"])
        alphabet.update(symbols)
        vocab_counter[symbols] += 1

    if not vocab_counter:
        raise LearnedTokenizerError("No valid tokens extracted from corpus")

    alphabet.add("</w>")
    unk_token = "<unk>"
    tokens: list[str] = [unk_token]
    base_symbols = sorted(alphabet)
    seen_tokens = {unk_token}
    for symbol in base_symbols:
        if symbol not in seen_tokens:
            tokens.append(symbol)
            seen_tokens.add(symbol)

    merges: list[tuple[str, str]] = []
    current_vocab = vocab_counter
    while len(tokens) < config.vocab_size:
        pair_stats = _collect_pair_stats(current_vocab)
        best_pair = _select_best_pair(pair_stats, config.min_frequency)
        if best_pair is None:
            break
        merges.append(best_pair)
        current_vocab = _merge_vocab(current_vocab, best_pair)
        merged_token = "".join(best_pair)
        if merged_token not in seen_tokens:
            tokens.append(merged_token)
            seen_tokens.add(merged_token)

    if len(tokens) > config.vocab_size:
        tokens = tokens[: config.vocab_size]

    vocab = {token: index for index, token in enumerate(tokens)}

    return LearnedTokenizerModel(
        name=config.name,
        algorithm=config.algorithm,
        vocab=vocab,
        merges=merges,
        lowercase=config.lowercase,
        unicode_normalize=config.unicode_normalize,
        config=config.raw,
    )


def save_trained_tokenizer(model: LearnedTokenizerModel, output_dir: Path) -> TokenizerManifest:
    """Persist tokenizer artifacts and return manifest metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    manifest_path = output_dir / "tokenizer_manifest.json"

    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(model.vocab, handle, indent=2, sort_keys=True, ensure_ascii=False)

    with merges_path.open("w", encoding="utf-8") as handle:
        for left, right in model.merges:
            handle.write(f"{left} {right}\n")

    tokenizer_hash = _hash_files([vocab_path, merges_path])

    manifest_data = {
        "name": model.name,
        "algorithm": model.algorithm,
        "created_utc": datetime.now(UTC).isoformat(),
        "tokenizer_hash": tokenizer_hash,
        "config": model.config,
        "files": {
            "vocab": vocab_path.name,
            "merges": merges_path.name,
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2, sort_keys=True)

    return TokenizerManifest(
        name=model.name,
        algorithm=model.algorithm,
        vocab_path=vocab_path,
        merges_path=merges_path,
        manifest_path=manifest_path,
        tokenizer_hash=tokenizer_hash,
        config=model.config,
    )


def load_learned_tokenizer(model_path: Path) -> LearnedTokenizer:
    """Load a persisted tokenizer."""

    model_dir = model_path
    if model_dir.is_file():
        model_dir = model_dir.parent
    manifest_path = model_dir / "tokenizer_manifest.json"
    if not manifest_path.exists():
        raise LearnedTokenizerError(f"Missing tokenizer manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    vocab_file = files.get("vocab")
    merges_file = files.get("merges")
    if not isinstance(vocab_file, str) or not isinstance(merges_file, str):
        raise LearnedTokenizerError("Manifest missing vocab/merges references")

    vocab_path = model_dir / vocab_file
    merges_path = model_dir / merges_file
    if not vocab_path.exists() or not merges_path.exists():
        raise LearnedTokenizerError("Tokenizer artifacts are incomplete")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    merges = _load_merges(merges_path)

    config = manifest.get("config", {})
    tokenizer_section = {}
    if isinstance(config, MutableMapping):
        tokenizer_section = config.get("tokenizer", {})  # type: ignore[assignment]

    lowercase = bool(tokenizer_section.get("lowercase", False)) if isinstance(tokenizer_section, MutableMapping) else False
    unicode_normalize = tokenizer_section.get("unicode_normalize") if isinstance(tokenizer_section, MutableMapping) else None
    if unicode_normalize is not None and not isinstance(unicode_normalize, str):
        unicode_normalize = None

    tokenizer_hash = manifest.get("tokenizer_hash")
    if not isinstance(tokenizer_hash, str):
        tokenizer_hash = _hash_files([vocab_path, merges_path])

    return LearnedTokenizer(
        name=str(manifest.get("name", "unknown")),
        vocab={str(k): int(v) for k, v in vocab.items()},
        merges=merges,
        lowercase=lowercase,
        unicode_normalize=unicode_normalize,  # type: ignore[arg-type]
        tokenizer_hash=tokenizer_hash,
    )


def apply_tokenizer_to_dataset(
    dataset: dict[str, list[MutableMapping[str, object]]],
    fields: Sequence[str],
    tokenizer: LearnedTokenizer,
    add_ids: bool,
) -> dict[str, object]:
    """Mutate dataset in-place, adding tokens/ids for the requested fields."""

    tokenized_records = 0
    for split_records in dataset.values():
        for record in split_records:
            for field in fields:
                value = record.get(field)
                if not isinstance(value, str):
                    continue
                tokens = tokenizer.encode(value)
                record[f"{field}_tokens"] = tokens
                if add_ids:
                    record[f"{field}_ids"] = [tokenizer.token_to_id(token) for token in tokens]
                tokenized_records += 1
    return {
        "tokenized_fields": list(fields),
        "tokenized_records": tokenized_records,
        "tokenizer_name": tokenizer.name,
        "tokenizer_hash": tokenizer.tokenizer_hash,
    }


def _iter_training_corpus(config: BpeTrainingConfig) -> Iterable[str]:
    if not config.input_path.exists():
        raise LearnedTokenizerError(f"Training input not found: {config.input_path}")
    count = 0
    with config.input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if config.limit is not None and count >= config.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LearnedTokenizerError("Training corpus must be JSONL") from exc
            value = payload.get(config.text_field)
            if isinstance(value, str):
                yield value
                count += 1


def _collect_pair_stats(vocab: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
    stats: Counter[tuple[str, str]] = Counter()
    for symbols, freq in vocab.items():
        if freq <= 0:
            continue
        for i in range(len(symbols) - 1):
            stats[(symbols[i], symbols[i + 1])] += freq
    return stats


def _select_best_pair(
    stats: Counter[tuple[str, str]],
    min_frequency: int,
) -> tuple[str, str] | None:
    best_pair: tuple[str, str] | None = None
    best_freq = 0
    for pair, freq in stats.items():
        if freq < min_frequency:
            continue
        if freq > best_freq or (freq == best_freq and (best_pair is None or pair < best_pair)):
            best_pair = pair
            best_freq = freq
    return best_pair


def _merge_vocab(
    vocab: Counter[tuple[str, ...]],
    pair: tuple[str, str],
) -> Counter[tuple[str, ...]]:
    merged: Counter[tuple[str, ...]] = Counter()
    for symbols, freq in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        merged[tuple(new_symbols)] += freq
    return merged


def _preprocess_text(text: str, lowercase: bool, normalize: str | None) -> str:
    updated = text
    if normalize:
        updated = unicodedata.normalize(normalize, updated)
    if lowercase:
        updated = updated.lower()
    return updated


def _get_pairs(symbols: Sequence[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for i in range(len(symbols) - 1):
        pairs.add((symbols[i], symbols[i + 1]))
    return pairs


def _merge_symbols(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def _hash_files(paths: Sequence[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(paths):
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _load_merges(path: Path) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise LearnedTokenizerError("Invalid merges file format")
            merges.append((parts[0], parts[1]))
    return merges


def _canonicalize(data: object) -> object:
    if isinstance(data, MutableMapping):
        return {str(key): _canonicalize(data[key]) for key in sorted(data)}
    if isinstance(data, list):
        return [_canonicalize(item) for item in data]
    return data


def _require_string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise LearnedTokenizerError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise LearnedTokenizerError(f"{key} must be a positive integer")
    return value

