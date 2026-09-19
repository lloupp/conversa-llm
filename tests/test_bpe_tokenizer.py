import json
from pathlib import Path

from conversa_llm.bpe_tokenizer import ByteBPETokenizer


def test_bpe_roundtrip_preserves_python(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    code = "def f(x):\n    return x + 1\n"
    data.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "codigo"},
                {"role": "assistant", "content": code},
            ]
        }) + "\n"
    )
    tok = ByteBPETokenizer.train_from_jsonl([data], vocab_size=300, min_pair_freq=1)
    assert tok.decode_text(tok.encode_text(code)) == code


def test_bpe_save_load(tmp_path: Path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"aaaa"},'
        '{"role":"assistant","content":"bbbb"}]}\n'
    )
    tok = ByteBPETokenizer.train_from_jsonl([data], vocab_size=280, min_pair_freq=1)
    out = tmp_path / "bpe.json"
    tok.save(out)
    loaded = ByteBPETokenizer.load(out)
    assert loaded.merges == tok.merges
    assert loaded.decode_text(loaded.encode_text("a b\n")) == "a b\n"
