from tlg.tokenizers import ByteTokenizer, CharTokenizer, create_tokenizer


def test_char_tokenizer_roundtrip() -> None:
    tokenizer = CharTokenizer(lowercase=True)
    tokens = tokenizer.tokenize("AbC")
    assert tokens == ["a", "b", "c"]
    assert tokenizer.detokenize(tokens) == "abc"


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    tokens = tokenizer.tokenize("abc")
    assert tokenizer.detokenize(tokens) == "abc"


def test_create_tokenizer_factory() -> None:
    assert isinstance(create_tokenizer("char"), CharTokenizer)
    assert isinstance(create_tokenizer("byte"), ByteTokenizer)
