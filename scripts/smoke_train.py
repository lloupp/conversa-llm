"""Treino curto para validar que o pipeline inteiro executa."""

import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "conversa_llm.train",
        "--steps",
        "5",
        "--batch-size",
        "4",
        "--d-model",
        "32",
        "--layers",
        "1",
        "--heads",
        "4",
        "--context",
        "128",
        "--out",
        "checkpoints/smoke.pt",
    ],
    check=True,
)
