"""Tokenization utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class TokenizerError(ValueError):
    """Raised when tokenization fails."""


@dataclass
class CharTokenizer:
    lowercase: bool = False

    def tokenize(self, text: str) -> list[str]:
        processed = text.lower() if self.lowercase else text
        return list(processed)

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def vocab(self, corpus: Iterable[str]) -> list[str]:
        chars = set()
        for sample in corpus:
            chars.update(self.tokenize(sample))
        return sorted(chars)


@dataclass
class ByteTokenizer:
    lowercase: bool = False

    def tokenize(self, text: str) -> list[int]:
        processed = text.lower() if self.lowercase else text
        return list(processed.encode("utf-8"))

    def detokenize(self, tokens: Sequence[int]) -> str:
        try:
            data = bytes(tokens)
            decoded = data.decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            raise TokenizerError("Invalid byte sequence") from exc
        if self.lowercase:
            return decoded
        return decoded

    def vocab(self, corpus: Iterable[str]) -> list[int]:
        chars = set()
        for sample in corpus:
            chars.update(self.tokenize(sample))
        return sorted(chars)


def create_tokenizer(tokenizer_type: str, lowercase: bool = False):
    if tokenizer_type == "char":
        return CharTokenizer(lowercase=lowercase)
    if tokenizer_type == "byte":
        return ByteTokenizer(lowercase=lowercase)
    raise TokenizerError(f"Unsupported tokenizer type '{tokenizer_type}'")
