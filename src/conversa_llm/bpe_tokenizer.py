from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class ByteBPETokenizer:
    """Byte-level BPE treinado do zero e reversível.

    IDs 0..255 são bytes. 256..261 são controles compatíveis com ByteTokenizer.
    Merges BPE começam em 262. Como todo texto vira bytes antes dos merges,
    não existe token desconhecido e espaços/quebras/indentação são preservados.
    """

    PAD = 256
    BOS = 257
    EOS = 258
    USER = 259
    ASSISTANT = 260
    SEP = 261
    FIRST_MERGE_ID = 262

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.vocab_size = self.FIRST_MERGE_ID + len(self.merges)
        self._token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for index, (left, right) in enumerate(self.merges):
            token_id = self.FIRST_MERGE_ID + index
            if left not in self._token_bytes or right not in self._token_bytes:
                raise ValueError(f"merge inválido #{index}: {(left, right)}")
            self._token_bytes[token_id] = self._token_bytes[left] + self._token_bytes[right]

    @classmethod
    def train_from_jsonl(
        cls,
        paths: Iterable[str | Path],
        vocab_size: int = 768,
        min_pair_freq: int = 2,
    ) -> "ByteBPETokenizer":
        if vocab_size < cls.FIRST_MERGE_ID:
            raise ValueError(f"vocab_size deve ser >= {cls.FIRST_MERGE_ID}")
        sequences: list[list[int]] = []
        for source in paths:
            with Path(source).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    for message in item.get("messages", []):
                        content = str(message.get("content", ""))
                        if content:
                            sequences.append(list(content.encode("utf-8")))
        if not sequences:
            raise ValueError("nenhum texto encontrado para treinar BPE")

        merges: list[tuple[int, int]] = []
        target_merges = vocab_size - cls.FIRST_MERGE_ID
        for _ in range(target_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < min_pair_freq:
                break
            new_id = cls.FIRST_MERGE_ID + len(merges)
            merges.append(pair)
            sequences = [cls._replace_pair(sequence, pair, new_id) for sequence in sequences]
        return cls(merges)

    @staticmethod
    def _replace_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        if len(tokens) < 2:
            return tokens
        out: list[int] = []
        i = 0
        left, right = pair
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def encode_text(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for index, pair in enumerate(self.merges):
            tokens = self._replace_pair(tokens, pair, self.FIRST_MERGE_ID + index)
        return tokens

    def decode_text(self, token_ids: list[int]) -> str:
        chunks = [self._token_bytes[token] for token in token_ids if token in self._token_bytes]
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def encode_prompt(self, text: str) -> list[int]:
        return [self.BOS, self.USER, *self.encode_text(text), self.SEP, self.ASSISTANT]

    def encode_conversation(self, messages: list[dict[str, str]]) -> tuple[list[int], list[bool]]:
        tokens: list[int] = [self.BOS]
        mask: list[bool] = [False]
        for message in messages:
            role = message["role"]
            if role == "user":
                role_token = self.USER
                supervise = False
            elif role == "assistant":
                role_token = self.ASSISTANT
                supervise = True
            else:
                raise ValueError(f"role inválido para treino: {role}")
            content_tokens = self.encode_text(message["content"])
            tokens.append(role_token)
            mask.append(False)
            tokens.extend(content_tokens)
            mask.extend([supervise] * len(content_tokens))
            tokens.append(self.SEP)
            mask.append(supervise)
        tokens.append(self.EOS)
        mask.append(bool(messages and messages[-1]["role"] == "assistant"))
        return tokens, mask

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"type": "byte_bpe", "merges": [list(pair) for pair in self.merges]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte_bpe":
            raise ValueError("arquivo não é um tokenizer byte_bpe")
        return cls([tuple(pair) for pair in payload.get("merges", [])])
