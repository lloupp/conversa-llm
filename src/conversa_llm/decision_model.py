from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import Block


ACTIONS = (
    "answer",
    "web_search",
    "read",
    "write",
    "edit",
    "bash",
    "powershell",
    "grep",
    "find",
    "ls",
    "stop",
)
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}


@dataclass
class DecisionConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    n_actions: int = len(ACTIONS)

    def to_dict(self) -> dict:
        return asdict(self)

    def backbone_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


class DecisionModel(nn.Module):
    """Modelo System-One: uma passagem, uma decisão tipada, probabilidades explícitas."""

    def __init__(self, config: DecisionConfig):
        super().__init__()
        self.config = config
        backbone = config.backbone_config()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(backbone) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_actions)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        batch, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError("sequência maior que context_length")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if attention_mask is None:
            last_index = torch.full(
                (batch,), seq_len - 1, dtype=torch.long, device=input_ids.device
            )
        else:
            last_index = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        pooled = x[torch.arange(batch, device=input_ids.device), last_index]
        logits = self.head(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict:
        self.eval()
        logits, _ = self(input_ids, attention_mask)
        probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        values, indices = probs.max(dim=-1)
        index = int(indices[0].item())
        return {
            "action": ACTIONS[index],
            "confidence": float(values[0].item()),
            "probabilities": {
                action: float(probs[0, i].item()) for i, action in enumerate(ACTIONS)
            },
        }
