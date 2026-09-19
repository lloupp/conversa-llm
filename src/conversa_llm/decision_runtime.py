from __future__ import annotations

import torch

from .decision_model import DecisionConfig, DecisionModel
from .decision_state import render_decision_state
from .tokenizer_loader import load_tokenizer


class DecisionRuntime:
    def __init__(
        self,
        model_path: str,
        tokenizer_file: str,
        device: str = "cpu",
        threshold: float = 0.65,
    ):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        self.model = DecisionModel(DecisionConfig(**checkpoint["config"])).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.tokenizer = load_tokenizer(tokenizer_file)
        self.device = device
        self.threshold = float(threshold)

    def decide(self, messages: list[dict]) -> dict:
        state = render_decision_state(messages)
        ids = self.tokenizer.encode_text(state)[-self.model.config.context_length :]
        if not ids:
            return {
                "action": "answer",
                "confidence": 0.0,
                "probabilities": {},
                "trusted": False,
            }
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(x, dtype=torch.bool)
        decision = self.model.decide(x, mask, temperature=self.temperature)
        decision["trusted"] = decision["confidence"] >= self.threshold
        return decision
