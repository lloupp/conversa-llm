from conversa_llm.eval_generation import answer_matches, normalize_text


def test_normalize_removes_accents_and_collapses_spaces():
    assert normalize_text("  Não   ") == "nao"


def test_keyword_match_uses_word_boundaries():
    assert answer_matches("Use len para isso.", "len")
    assert not answer_matches("Use lista.", "is")


def test_symbolic_answer_ignores_spaces():
    assert answer_matches("A resposta é x=10.", "x = 10")
    assert answer_matches("Use lista.append(x).", "lista.append(x)")
