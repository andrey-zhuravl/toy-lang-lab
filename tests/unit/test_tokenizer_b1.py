from __future__ import annotations

from pathlib import Path

from tlg.grammar import load_grammar
from tlg.tokenizer_encoding import EncodeConfig, LoadedTokenizer, encode_dataset
from tlg.tokenizer_io import compute_dataset_hash, iter_corpus_text
from tlg.tokenizer_training import TokenizerTrainingConfig, train_tokenizer
from tlg.dictionaries import generate_dictionary_from_grammar


FIXTURE_CORPUS = Path("fixtures/corpus_toy.jsonl")


def _load_corpus() -> tuple[list, str]:
    samples = list(iter_corpus_text([FIXTURE_CORPUS], "text"))
    dataset_hash = compute_dataset_hash([FIXTURE_CORPUS])
    return samples, dataset_hash


def test_train_bpe(tmp_path: Path) -> None:
    corpus, dataset_hash = _load_corpus()
    output_dir = tmp_path / "bpe"
    config = TokenizerTrainingConfig(
        kind="bpe",
        vocab_size=64,
        min_frequency=1,
        character_coverage=0.9995,
        limit_alphabet=1000,
        normalization="nfc",
        pretokenizer="whitespace",
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
        seed=123,
        output_dir=output_dir,
    )

    result = train_tokenizer(config, corpus, dataset_hash, "fixture")
    assert (output_dir / "tokenizer.json").exists()
    assert "oov_rate" in result.report
    assert result.tokenizer_hash


def test_train_unigram(tmp_path: Path) -> None:
    corpus, dataset_hash = _load_corpus()
    output_dir = tmp_path / "unigram"
    config = TokenizerTrainingConfig(
        kind="unigram",
        vocab_size=64,
        min_frequency=1,
        character_coverage=0.999,
        limit_alphabet=1000,
        normalization="nfc",
        pretokenizer="whitespace",
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
        seed=321,
        output_dir=output_dir,
    )

    result = train_tokenizer(config, corpus, dataset_hash, "fixture")
    assert (output_dir / "tokenizer.model").exists()
    assert result.report["oov_rate"] >= 0.0


def test_encode_arrow(tmp_path: Path) -> None:
    corpus, dataset_hash = _load_corpus()
    output_dir = tmp_path / "bpe"
    config = TokenizerTrainingConfig(
        kind="bpe",
        vocab_size=64,
        min_frequency=1,
        character_coverage=0.9995,
        limit_alphabet=1000,
        normalization="nfc",
        pretokenizer="whitespace",
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
        seed=999,
        output_dir=output_dir,
    )
    train_tokenizer(config, corpus, dataset_hash, "fixture")

    tokenizer = LoadedTokenizer(output_dir / "tokenizer.json")
    encode_config = EncodeConfig(
        tokenizer_path=output_dir / "tokenizer.json",
        input_paths=[FIXTURE_CORPUS],
        text_key="text",
        output_path=tmp_path / "encoded.arrow",
        output_format="arrow",
        max_length=32,
        pad_to_max_length=True,
        prepend_bos=True,
        append_eos=True,
        write_maps=True,
    )
    result = encode_dataset(encode_config, tokenizer, corpus)
    assert result.output_path.exists()
    assert result.report_path.exists()
    assert result.metrics["samples"] == len(corpus)


def test_generate_dictionary_from_grammar(tmp_path: Path) -> None:
    grammar = load_grammar(Path("conf/grammar/basic_world.yaml"))
    out = tmp_path / "dsl.vocab.txt"
    manifest = generate_dictionary_from_grammar(grammar, out, add_specials=["<pad>"])
    assert out.exists()
    assert manifest["tokens"] > 0
    assert manifest["special_tokens"] == ["<pad>"]
