"""Tokenizer training workflows for Layer B1 without external dependencies."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal

from tlg.tokenizers_learned import (
    BpeTrainingConfig as LegacyBpeConfig,
    LearnedTokenizer,
    train_bpe as legacy_train_bpe,
)
from tlg.tokenizer_io import CorpusSample


class TokenizerTrainingError(RuntimeError):
    """Raised when tokenizer training fails."""


@dataclass(frozen=True)
class TokenizerTrainingConfig:
    """High level configuration for tokenizer training."""

    kind: Literal["bpe", "unigram"]
    vocab_size: int
    min_frequency: int
    character_coverage: float
    limit_alphabet: int
    normalization: Literal["nfc", "nfkc", "none"]
    pretokenizer: Literal["whitespace", "bytelevel", "basic"]
    special_tokens: list[str]
    seed: int
    output_dir: Path


@dataclass
class TokenizerTrainingResult:
    tokenizer_name: str
    tokenizer_hash: str
    vocab_size: int
    special_tokens: list[str]
    artifacts: dict[str, str]
    report: dict[str, float]
    manifest: dict[str, object]


@dataclass
class _Encoding:
    ids: list[int]


class _BpeTokenizerWrapper:
    def __init__(self, model: LearnedTokenizer, vocab: dict[str, int]) -> None:
        self.model = model
        self.vocab = vocab
        self._id_to_token = {idx: token for token, idx in vocab.items()}

    def encode(self, text: str) -> _Encoding:
        _, ids = self.model.encode_with_ids(text)
        return _Encoding(ids)

    def token_to_id(self, token: str) -> int:
        return self.model.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token.get(idx, "")

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def save(self, path: Path, metadata: dict[str, object]) -> None:
        payload = {
            "format": "toy-bpe",
            "metadata": metadata,
            "vocab": self.vocab,
            "merges": self.model.merges,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class _UnigramTokenizerWrapper:
    def __init__(self, vocab: dict[str, int], special_tokens: list[str]) -> None:
        self.vocab = vocab
        self.special_tokens = special_tokens
        self._id_to_token = {idx: token for token, idx in vocab.items()}
        self.unk_id = self.vocab.get(_find_special_token(special_tokens, {"<unk>", "<unknown>"}) or "<unk>", 0)

    def encode(self, text: str) -> _Encoding:
        tokens = text.split()
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        return _Encoding(ids)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        return self._id_to_token.get(idx, "")

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def save(self, path: Path, metadata: dict[str, object]) -> None:
        payload = {
            "format": "toy-unigram",
            "metadata": metadata,
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def train_tokenizer(
    config: TokenizerTrainingConfig,
    corpus: Iterable[CorpusSample],
    dataset_hash: str,
    dataset_id: str,
) -> TokenizerTrainingResult:
    samples = [sample for sample in corpus]
    if not samples:
        raise TokenizerTrainingError("Training corpus is empty")

    texts = [sample.text for sample in samples]
    special_tokens = _dedupe_tokens(config.special_tokens)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.kind == "bpe":
        tokenizer_wrapper, merges = _train_bpe(texts, config, special_tokens)
    else:
        tokenizer_wrapper, merges = _train_unigram(texts, config, special_tokens)

    metrics = _compute_segmentation_metrics(tokenizer_wrapper, texts, special_tokens)
    artifacts = _persist_artifacts(
        config=config,
        tokenizer=tokenizer_wrapper,
        merges=merges,
        special_tokens=special_tokens,
        metrics=metrics,
        dataset_hash=dataset_hash,
        dataset_id=dataset_id,
    )

    manifest = {
        "kind": config.kind,
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "character_coverage": config.character_coverage,
        "limit_alphabet": config.limit_alphabet,
        "normalization": config.normalization,
        "pretokenizer": config.pretokenizer,
        "seed": config.seed,
        "special_tokens": special_tokens,
        "dataset_hash": dataset_hash,
        "dataset_id": dataset_id,
        "tokenizer_hash": artifacts["tokenizer_hash"],
        "artifacts": {key: value for key, value in artifacts.items() if key != "tokenizer_hash"},
        "metrics": metrics,
    }

    tokenizer_name = f"tokenizer-{config.kind}-v{config.vocab_size}"

    return TokenizerTrainingResult(
        tokenizer_name=tokenizer_name,
        tokenizer_hash=artifacts["tokenizer_hash"],
        vocab_size=config.vocab_size,
        special_tokens=special_tokens,
        artifacts=artifacts,
        report=metrics,
        manifest=manifest,
    )


def _train_bpe(
    texts: list[str],
    config: TokenizerTrainingConfig,
    special_tokens: list[str],
) -> tuple[_BpeTokenizerWrapper, list[tuple[str, str]]]:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp_dir:
        corpus_path = Path(tmp_dir) / "corpus.jsonl"
        with corpus_path.open("w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False))
                handle.write("\n")

        legacy_config = LegacyBpeConfig(
            name="bpe",
            algorithm="bpe",
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            unicode_normalize=None,
            lowercase=False,
            seed=config.seed,
            tiebreak="lexicographic",
            input_path=corpus_path,
            text_field="text",
            limit=None,
            output_dir=Path(tmp_dir),
            raw={},
        )
        model = legacy_train_bpe(legacy_config)

    tokenizer = LearnedTokenizer(
        name=model.name,
        vocab=model.vocab,
        merges=model.merges,
        lowercase=False,
        unicode_normalize=None,
        tokenizer_hash="",
        unk_token=_find_special_token(special_tokens, {"<unk>", "<unknown>"}) or "<unk>",
    )
    wrapper = _BpeTokenizerWrapper(tokenizer, model.vocab)
    return wrapper, model.merges


def _train_unigram(
    texts: list[str],
    config: TokenizerTrainingConfig,
    special_tokens: list[str],
) -> tuple[_UnigramTokenizerWrapper, list[tuple[str, str]]]:
    vocab: dict[str, int] = {}
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())

    for token, _ in counter.most_common():
        if len(vocab) >= config.vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    wrapper = _UnigramTokenizerWrapper(vocab, special_tokens)
    return wrapper, []


def _persist_artifacts(
    *,
    config: TokenizerTrainingConfig,
    tokenizer: _BpeTokenizerWrapper | _UnigramTokenizerWrapper,
    merges: list[tuple[str, str]],
    special_tokens: list[str],
    metrics: dict[str, float],
    dataset_hash: str,
    dataset_id: str,
) -> dict[str, str]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "kind": config.kind,
        "vocab_size": config.vocab_size,
        "special_tokens": special_tokens,
        "metrics": metrics,
    }

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_json, metadata)

    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    vocab_path = output_dir / "vocab.txt"
    with vocab_path.open("w", encoding="utf-8") as handle:
        for token, _ in vocab_items:
            handle.write(f"{token}\n")

    merges_path = None
    if merges:
        merges_path = output_dir / "merges.txt"
        with merges_path.open("w", encoding="utf-8") as handle:
            for left, right in merges:
                handle.write(f"{left} {right}\n")

    token2id = {token: idx for token, idx in vocab_items}
    id2token_path = output_dir / "id2token.tsv"
    with id2token_path.open("w", encoding="utf-8") as handle:
        for token, idx in vocab_items:
            handle.write(f"{idx}\t{token}\n")

    token2id_path = output_dir / "token2id.json"
    token2id_path.write_text(json.dumps(token2id, indent=2, ensure_ascii=False), encoding="utf-8")

    specials_path = output_dir / "special_tokens.json"
    specials_path.write_text(json.dumps(special_tokens, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = output_dir / "segmentation_report.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest_path = output_dir / "tokenizer_run_manifest.json"
    manifest_payload = {
        "dataset_hash": dataset_hash,
        "dataset_id": dataset_id,
        "metrics": metrics,
        "special_tokens": special_tokens,
        "kind": config.kind,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    tokenizer_hash = _hash_file(tokenizer_json)

    artifacts: dict[str, str] = {
        "tokenizer_json": str(tokenizer_json),
        "vocab": str(vocab_path),
        "id2token": str(id2token_path),
        "token2id": str(token2id_path),
        "special_tokens": str(specials_path),
        "segmentation_report": str(report_path),
        "manifest": str(manifest_path),
        "tokenizer_hash": tokenizer_hash,
    }

    if merges_path is not None:
        artifacts["merges"] = str(merges_path)

    if config.kind == "unigram":
        model_path = output_dir / "tokenizer.model"
        tokenizer.save(model_path, metadata)
        artifacts["tokenizer_model"] = str(model_path)

    return artifacts


def _compute_segmentation_metrics(
    tokenizer: _BpeTokenizerWrapper | _UnigramTokenizerWrapper,
    texts: list[str],
    special_tokens: list[str],
) -> dict[str, float]:
    lengths: list[int] = []
    total_tokens = 0
    unk_token = _find_special_token(special_tokens, {"<unk>", "<unknown>"}) or "<unk>"
    unk_id = tokenizer.token_to_id(unk_token)
    unk_tokens = 0
    tokens_per_word: list[float] = []

    vocab = tokenizer.get_vocab()

    for text in texts:
        encoding = tokenizer.encode(text)
        ids = encoding.ids
        lengths.append(len(ids))
        total_tokens += len(ids)
        unk_tokens += sum(1 for idx in ids if idx == unk_id)
        words = text.split()
        if words:
            tokens_per_word.append(len(ids) / len(words))

    unique_chars = {char for text in texts for char in text}
    covered_chars = {
        char
        for char in unique_chars
        if any(char in token for token in vocab)
    }

    return {
        "oov_rate": (unk_tokens / total_tokens) if total_tokens else 0.0,
        "tokens_per_seq_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "tokens_per_seq_p95": _percentile(lengths, 95),
        "tokens_per_word_mean": (sum(tokens_per_word) / len(tokens_per_word)) if tokens_per_word else 0.0,
        "chars_coverage": (len(covered_chars) / len(unique_chars)) if unique_chars else 1.0,
    }


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    index = max(int(round((q / 100.0) * (len(values) - 1))), 0)
    return float(sorted(values)[index])


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _find_special_token(tokens: Iterable[str], candidates: set[str]) -> str | None:
    lowered = {token.lower(): token for token in tokens}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


__all__ = [
    "TokenizerTrainingConfig",
    "TokenizerTrainingError",
    "TokenizerTrainingResult",
    "train_tokenizer",
]
