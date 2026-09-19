from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 262
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
