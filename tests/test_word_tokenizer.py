from pathlib import Path

from conversa_llm.word_tokenizer import WordTokenizer


def test_word_tokenizer_train_save_load(tmp_path: Path):
    data = tmp_path / "d.jsonl"
    data.write_text('{"messages":[{"role":"user","content":"qual operador resto"},{"role":"assistant","content":"%"}]}\n', encoding="utf-8")
    tok = WordTokenizer.train_from_jsonl([data])
    assert tok.encode_text("operador") != [tok.UNK]
    out = tmp_path / "vocab.json"
    tok.save(out)
    loaded = WordTokenizer.load(out)
    assert loaded.encode_text("operador") == tok.encode_text("operador")
    assert loaded.vocab_size == tok.vocab_size


def test_word_tokenizer_conversation_roles(tmp_path: Path):
    data = tmp_path / "d.jsonl"
    data.write_text('{"messages":[{"role":"user","content":"oi"},{"role":"assistant","content":"ola"}]}\n', encoding="utf-8")
    tok = WordTokenizer.train_from_jsonl([data])
    tokens, mask = tok.encode_conversation([{"role":"user","content":"oi"},{"role":"assistant","content":"ola"}])
    assert tokens[0] == tok.BOS
    assert tok.USER in tokens and tok.ASSISTANT in tokens
    assert any(mask)
