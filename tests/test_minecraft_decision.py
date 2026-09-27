import torch

from conversa_llm.decision_model import DecisionConfig
from conversa_llm.minecraft_decision import MINECRAFT_ACTIONS, MinecraftDecisionModel


def test_minecraft_profile_has_dedicated_actions():
    config = DecisionConfig(vocab_size=128, context_length=16, d_model=16, n_heads=4, n_layers=1, d_ff=32)
    model = MinecraftDecisionModel(config)
    x = torch.randint(0, 128, (1, 8))
    result = model.decide(x, torch.ones_like(x, dtype=torch.bool))
    assert result["action"] in MINECRAFT_ACTIONS
    assert set(result["probabilities"]) == set(MINECRAFT_ACTIONS)
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_minecraft_profile_does_not_expose_code_actions():
    assert "bash" not in MINECRAFT_ACTIONS
    assert "write" not in MINECRAFT_ACTIONS
    assert "powershell" not in MINECRAFT_ACTIONS
