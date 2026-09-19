from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .decision_model import ACTION_TO_ID


class DecisionDataset(Dataset):
    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        tokenizer,
        context_length: int,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.examples: list[tuple[list[int], int]] = []
        sources = [paths] if isinstance(paths, (str, Path)) else list(paths)
        for source in sources:
            self._load(Path(source))
        if not self.examples:
            raise ValueError("dataset de decisões vazio")

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                state = item.get("state")
                label = item.get("label")
                if not isinstance(state, str) or label not in ACTION_TO_ID:
                    raise ValueError(f"{path}:{line_number}: state/label inválidos")
                ids = self.tokenizer.encode_text(state)
                ids = ids[-self.context_length :]
                if not ids:
                    continue
                self.examples.append((ids, ACTION_TO_ID[label]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def collate_decisions(batch, pad_token: int):
    max_len = max(len(ids) for ids, _ in batch)
    inputs, masks, labels = [], [], []
    for ids, label in batch:
        pad = max_len - len(ids)
        inputs.append(ids + [pad_token] * pad)
        masks.append([1] * len(ids) + [0] * pad)
        labels.append(label)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(labels, dtype=torch.long),
    )
