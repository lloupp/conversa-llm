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
    # Perfil do checkpoint Pi v2: ~433 mil parâmetros, BPE e contexto neural 256.
    "cpu-8gb-pi": TrainingProfile(
        d_model=96, layers=3, heads=4, context=256,
        batch_size=2, grad_accum=4, lr=8e-4,
    ),
    "cpu-16gb": TrainingProfile(
        d_model=128, layers=4, heads=4, context=256,
        batch_size=4, grad_accum=8, lr=3e-4,
    ),
    "gpu-12gb": TrainingProfile(
        d_model=256, layers=6, heads=8, context=256,
        batch_size=8, grad_accum=4, lr=3e-4,
    ),
}


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"perfil desconhecido: {name}. Opções: {choices}") from exc
