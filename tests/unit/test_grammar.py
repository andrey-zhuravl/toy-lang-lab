from pathlib import Path

import pytest

from tlg.grammar import GrammarError, load_grammar


def write_grammar(tmp_path: Path) -> Path:
    path = tmp_path / "grammar.yaml"
    path.write_text(
        """entities:
  person: [Ann]
  item: [book]
templates:
  - "{person} has a {item}."
weights:
  templates: [1.0]
""",
        encoding="utf-8",
    )
    return path


def test_load_valid_grammar(tmp_path: Path) -> None:
    path = write_grammar(tmp_path)
    grammar = load_grammar(path)
    assert grammar.entities["person"] == ["Ann"]
    assert grammar.templates
    assert grammar.template_weights == [1.0]


def test_invalid_grammar_missing_templates(tmp_path: Path) -> None:
    path = tmp_path / "grammar.yaml"
    path.write_text("entities: {x: [1]}\n", encoding="utf-8")
    with pytest.raises(GrammarError):
        load_grammar(path)
