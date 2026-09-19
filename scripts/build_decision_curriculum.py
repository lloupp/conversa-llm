from __future__ import annotations

import json
import random
from pathlib import Path

from conversa_llm.decision_state import render_decision_state


def row(messages: list[dict], label: str) -> dict:
    return {"state": render_decision_state(messages), "label": label}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    examples: list[dict] = []

    static_questions = [
        "O que faz len em Python?",
        "Qual operador calcula resto em Python?",
        "Explique uma lista Python.",
        "Como funciona return?",
        "Qual a diferença entre tuple e list?",
        "Escreva uma função simples de soma.",
    ]
    for text in static_questions:
        examples.append(row([{"role": "user", "content": text}], "answer"))

    current_questions = [
        "Qual é a versão atual do Django?",
        "Pesquise a documentação mais recente do FastAPI.",
        "Qual release do Python saiu mais recentemente?",
        "Busque na internet as mudanças recentes do PyTorch.",
        "Quero informação atual sobre uma biblioteca que você não conhece.",
    ]
    for text in current_questions:
        examples.append(row([{"role": "user", "content": text}], "web_search"))

    direct = {
        "read": [
            "Leia README.md.",
            "Abra src/app.py.",
            "Inspecione pyproject.toml.",
        ],
        "write": [
            "Crie status.txt com conteúdo ok.",
            "Escreva pronto em notes.txt.",
        ],
        "edit": [
            "Troque foo por bar em src/app.py.",
            "Altere DEBUG = True para DEBUG = False em config.py.",
        ],
        "bash": [
            "Rode pytest -q no terminal.",
            "Execute python -m compileall src.",
        ],
        "powershell": [
            "Use PowerShell para executar Get-ChildItem.",
            "Rode no PowerShell python --version.",
        ],
        "grep": [
            "Procure TODO em src.",
            "Busque def autenticar no projeto.",
        ],
        "find": [
            "Encontre **/*.py em tests.",
            "Localize arquivos *config*.json.",
        ],
        "ls": [
            "Liste os arquivos em src.",
            "Mostre o conteúdo do diretório docs.",
        ],
    }
    for label, prompts in direct.items():
        for prompt in prompts:
            examples.append(row([{"role": "user", "content": prompt}], label))

    multi_turn = [
        (
            [
                {"role": "user", "content": "Corrija src/calc.py e rode os testes."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{\"path\":\"src/calc.py\"}"},
                    }],
                },
                {
                    "role": "tool",
                    "name": "read",
                    "tool_call_id": "c1",
                    "content": "def soma(a,b):\n    return a-b",
                },
            ],
            "edit",
        ),
        (
            [
                {"role": "user", "content": "Corrija src/calc.py e rode os testes."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "edit", "arguments": "{}"},
                    }],
                },
                {"role": "tool", "name": "edit", "tool_call_id": "c2", "content": "Arquivo atualizado."},
            ],
            "bash",
        ),
        (
            [
                {"role": "user", "content": "Corrija o bug e valide."},
                {"role": "tool", "name": "bash", "tool_call_id": "c3", "content": "1 failed: AssertionError"},
            ],
            "read",
        ),
        (
            [
                {"role": "user", "content": "Corrija o bug e valide."},
                {"role": "tool", "name": "bash", "tool_call_id": "c4", "content": "12 passed in 0.31s"},
            ],
            "stop",
        ),
        (
            [
                {"role": "user", "content": "Descubra onde autenticar está definida."},
                {"role": "tool", "name": "grep", "tool_call_id": "g1", "content": "src/auth.py:12:def autenticar"},
            ],
            "read",
        ),
    ]
    for messages, label in multi_turn:
        examples.append(row(messages, label))

    rng = random.Random(20260919)
    train: list[dict] = []
    for _ in range(40):
        copy = examples[:]
        rng.shuffle(copy)
        train.extend(copy)

    benchmark = [
        row([{"role": "user", "content": "Explique dict.get em Python."}], "answer"),
        row([{"role": "user", "content": "Pesquise qual é a versão mais recente do NumPy."}], "web_search"),
        row([{"role": "user", "content": "Abra docs/setup.md."}], "read"),
        row([{"role": "user", "content": "Procure FIXME em src."}], "grep"),
        row([{"role": "user", "content": "Liste a pasta tests."}], "ls"),
        row([{"role": "user", "content": "Encontre **/*api*.py em tests."}], "find"),
        row([{"role": "tool", "name": "bash", "tool_call_id": "z1", "content": "8 passed"}], "stop"),
    ]

    write_jsonl(root / "data" / "decision_curriculum.jsonl", train)
    write_jsonl(root / "data" / "decision_benchmark.jsonl", benchmark)
    print(f"decision_train={len(train)} decision_benchmark={len(benchmark)}")


if __name__ == "__main__":
    main()
