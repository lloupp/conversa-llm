from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class WordTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    USER = 3
    ASSISTANT = 4
    SEP = 5
    UNK = 6
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<sep>", "<unk>"]

    def __init__(self, vocab: list[str]):
        if vocab[: len(self.SPECIAL_TOKENS)] != self.SPECIAL_TOKENS:
            raise ValueError("vocab deve iniciar com os tokens especiais padrao")
        self.id_to_token = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def train_from_jsonl(
        cls,
        paths: Iterable[str | Path],
        min_freq: int = 1,
        max_vocab: int = 4096,
    ) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for source in paths:
            with Path(source).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    for message in item.get("messages", []):
                        counter.update(_TOKEN_RE.findall(message["content"]))
        words = [token for token, freq in counter.most_common() if freq >= min_freq]
        words = words[: max(0, max_vocab - len(cls.SPECIAL_TOKENS))]
        return cls(cls.SPECIAL_TOKENS + words)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["vocab"])

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"type": "word", "vocab": self.id_to_token}, ensure_ascii=False, indent=2), encoding="utf-8")

    def encode_text(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.UNK) for token in _TOKEN_RE.findall(text)]

    def decode_text(self, token_ids: list[int]) -> str:
        tokens = [self.id_to_token[i] for i in token_ids if 0 <= i < self.vocab_size and i >= len(self.SPECIAL_TOKENS)]
        return " ".join(tokens)

    def encode_prompt(self, text: str) -> list[int]:
        return [self.BOS, self.USER, *self.encode_text(text), self.SEP, self.ASSISTANT]

    def encode_conversation(self, messages: list[dict[str, str]]) -> tuple[list[int], list[bool]]:
        tokens: list[int] = [self.BOS]
        assistant_content_mask: list[bool] = [False]
        for message in messages:
            role = message["role"]
            if role == "user":
                role_token = self.USER
                supervise = False
            elif role == "assistant":
                role_token = self.ASSISTANT
                supervise = True
            else:
                raise ValueError(f"role invalido: {role}")
            content_tokens = self.encode_text(message["content"])
            tokens.append(role_token)
            assistant_content_mask.append(False)
            tokens.extend(content_tokens)
            assistant_content_mask.extend([supervise] * len(content_tokens))
            tokens.append(self.SEP)
            assistant_content_mask.append(supervise)
        tokens.append(self.EOS)
        assistant_content_mask.append(bool(messages and messages[-1]["role"] == "assistant"))
        return tokens, assistant_content_mask
