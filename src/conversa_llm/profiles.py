from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    d_model: int
    layers: int
    heads: int
    context: int
    batch_size: int
    grad_accum: int
    lr: float


PROFILES: dict[str, TrainingProfile] = {
    "cpu-8gb-python": TrainingProfile(
        d_model=64, layers=2, heads=4, context=128,
        batch_size=4, grad_accum=1, lr=1e-3,
    ),
    "cpu-8gb": TrainingProfile(
        d_model=96, layers=3, heads=4, context=128,
        batch_size=2, grad_accum=8, lr=3e-4,
    ),
    # Mantido para reproduzir checkpoints Pi antigos.
    "cpu-8gb-pi-small": TrainingProfile(
        d_model=96, layers=3, heads=4, context=256,
        batch_size=2, grad_accum=4, lr=8e-4,
    ),
    # BPE byte-level + context=256: ~1.8M params, seguro em 8 GB.
    "cpu-8gb-bpe": TrainingProfile(
        d_model=128, layers=4, heads=4, context=256,
        batch_size=2, grad_accum=8, lr=3e-4,
    ),
    # Novo alvo: ~6-7M parâmetros e 512 tokens, ainda viável em 8 GB.
    "cpu-8gb-pi": TrainingProfile(
        d_model=256, layers=8, heads=8, context=512,
        batch_size=1, grad_accum=8, lr=3e-4,
    ),
    "cpu-16gb": TrainingProfile(
        d_model=192, layers=6, heads=6, context=512,
        batch_size=2, grad_accum=8, lr=3e-4,
    ),
    "gpu-12gb": TrainingProfile(
        d_model=384, layers=10, heads=8, context=1024,
        batch_size=4, grad_accum=4, lr=2e-4,
    ),
}


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"perfil desconhecido: {name}. Opções: {choices}") from exc
