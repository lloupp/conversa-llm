import argparse

import pytest

from conversa_llm.profiles import get_profile
from conversa_llm.train import resolve_training_args


def test_cpu_8gb_profile_is_conservative():
    profile = get_profile("cpu-8gb")
    assert profile.context <= 128
    assert profile.batch_size <= 2
    assert profile.grad_accum >= 4


def test_profile_populates_training_args():
    args = argparse.Namespace(
        profile="cpu-8gb",
        batch_size=None,
        grad_accum=None,
        lr=None,
        d_model=None,
        layers=None,
        heads=None,
        context=None,
        steps=10,
    )
    resolved = resolve_training_args(args)
    assert resolved.d_model == 96
    assert resolved.batch_size == 2
    assert resolved.grad_accum == 8


def test_unknown_profile_fails():
    with pytest.raises(ValueError):
        get_profile("nao-existe")
