from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class ConversationDataset(Dataset):
    def __init__(
        self,
        path: str | Path | Iterable[str | Path],
        context_length: int,
        supervise_all: bool = False,
        tokenizer=None,
    ):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.context_length = context_length
        self.supervise_all = supervise_all
        self.examples: list[tuple[list[int], list[int]]] = []

        paths = [path] if isinstance(path, (str, Path)) else list(path)
        if not paths:
            raise ValueError("ao menos um arquivo de dados é obrigatório")

        for source in paths:
            self._load_path(Path(source))

        if not self.examples:
            raise ValueError("dataset não possui exemplos supervisionáveis")

    def _load_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{path}: linha {line_number}: messages ausente ou vazio")
                tokens, mask = self.tokenizer.encode_conversation(messages)
                tokens = tokens[: self.context_length + 1]
                mask = mask[: self.context_length + 1]
                if len(tokens) < 2:
                    continue
                inputs = tokens[:-1]
                if self.supervise_all:
                    targets = tokens[1:]
                else:
                    targets = [
                        token if supervised else -100
                        for token, supervised in zip(tokens[1:], mask[1:])
                    ]
                if all(target == -100 for target in targets):
                    continue
                self.examples.append((inputs, targets))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def collate_batch(batch: list[tuple[list[int], list[int]]], pad_token: int = ByteTokenizer.PAD):
    max_len = max(len(inputs) for inputs, _ in batch)
    inputs_out = []
    labels_out = []
    for inputs, labels in batch:
        pad = max_len - len(inputs)
        inputs_out.append(inputs + [pad_token] * pad)
        labels_out.append(labels + [-100] * pad)
    return torch.tensor(inputs_out, dtype=torch.long), torch.tensor(labels_out, dtype=torch.long)
