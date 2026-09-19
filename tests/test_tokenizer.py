from conversa_llm.tokenizer import ByteTokenizer


def test_utf8_roundtrip():
    tokenizer = ByteTokenizer()
    text = "Olá, mundo! ação 🚀"
    assert tokenizer.decode_text(tokenizer.encode_text(text)) == text


def test_prompt_has_roles():
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode_prompt("oi")
    assert tokens[:2] == [tokenizer.BOS, tokenizer.USER]
    assert tokens[-2:] == [tokenizer.SEP, tokenizer.ASSISTANT]
