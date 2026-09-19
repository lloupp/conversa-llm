import torch

from conversa_llm.decision_model import ACTIONS, DecisionConfig, DecisionModel


def test_decision_model_outputs_typed_distribution():
    config = DecisionConfig(
        vocab_size=300,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
    )
    model = DecisionModel(config)
    x = torch.randint(0, 300, (2, 8))
    mask = torch.ones_like(x, dtype=torch.bool)
    logits, loss = model(x, mask, torch.tensor([0, 1]))
    assert logits.shape == (2, len(ACTIONS))
    assert loss is not None and torch.isfinite(loss)


def test_decide_probabilities_sum_to_one():
    config = DecisionConfig(
        vocab_size=300,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
    )
    model = DecisionModel(config)
    x = torch.randint(0, 300, (1, 8))
    result = model.decide(x, torch.ones_like(x, dtype=torch.bool))
    assert result["action"] in ACTIONS
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5
