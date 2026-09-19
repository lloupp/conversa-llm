from __future__ import annotations

import json
from pathlib import Path

from .bpe_tokenizer import ByteBPETokenizer
from .tokenizer import ByteTokenizer


def load_tokenizer(path: str | None):
    if not path:
        return ByteTokenizer()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = payload.get("type")
    if kind == "byte_bpe":
        return ByteBPETokenizer.load(path)
    if kind == "word" or (kind is None and "vocab" in payload):
        from .word_tokenizer import WordTokenizer
        return WordTokenizer.load(path)
    raise ValueError(f"tipo de tokenizer não suportado: {kind!r}")
