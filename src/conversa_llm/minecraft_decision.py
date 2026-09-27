from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from .decision_model import DecisionConfig, DecisionModel
from .tokenizer_loader import load_tokenizer

MINECRAFT_ACTIONS = (
    "gather",
    "craft",
    "smelt",
    "eat",
    "move",
    "deposit",
    "withdraw",
    "build",
    "fight",
    "wait",
    "stop",
)
MINECRAFT_ACTION_TO_ID = {name: i for i, name in enumerate(MINECRAFT_ACTIONS)}


class MinecraftDecisionModel(DecisionModel):
    """Decision head dedicated to Minecraft; it never reuses coding-action labels."""

    def __init__(self, config: DecisionConfig):
        if config.n_actions != len(MINECRAFT_ACTIONS):
            config = replace(config, n_actions=len(MINECRAFT_ACTIONS))
        super().__init__(config)

    @torch.no_grad()
    def decide(self, input_ids, attention_mask=None, temperature: float = 1.0) -> dict:
        self.eval()
        logits, _ = self(input_ids, attention_mask)
        probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        values, indices = probs.max(dim=-1)
        index = int(indices[0].item())
        return {
            "action": MINECRAFT_ACTIONS[index],
            "confidence": float(values[0].item()),
            "probabilities": {
                action: float(probs[0, i].item())
                for i, action in enumerate(MINECRAFT_ACTIONS)
            },
        }


class MinecraftDecisionRuntime:
    def __init__(self, model_path: str, tokenizer_file: str, device: str = "cpu", threshold: float = 0.70):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        raw_config = dict(checkpoint["config"])
        raw_config["n_actions"] = len(MINECRAFT_ACTIONS)
        self.model = MinecraftDecisionModel(DecisionConfig(**raw_config)).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.tokenizer = load_tokenizer(tokenizer_file)
        self.device = device
        self.threshold = float(threshold)

    def decide_state(self, state: str) -> dict:
        ids = self.tokenizer.encode_text(state)[-self.model.config.context_length:]
        if not ids:
            return {"action": "wait", "confidence": 0.0, "probabilities": {}, "trusted": False}
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(x, dtype=torch.bool)
        decision = self.model.decide(x, mask, temperature=self.temperature)
        decision["trusted"] = decision["confidence"] >= self.threshold
        return decision
