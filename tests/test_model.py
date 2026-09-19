import torch

from conversa_llm.config import ModelConfig
from conversa_llm.model import ConversaGPT


def test_forward_shape_and_loss():
    config = ModelConfig(context_length=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = ConversaGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_generation_grows_sequence():
    config = ModelConfig(context_length=16, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = ConversaGPT(config)
    x = torch.tensor([[257, 259, 111, 105, 261, 260]])
    out = model.generate(x, max_new_tokens=3, top_k=5)
    assert out.shape[1] == x.shape[1] + 3
