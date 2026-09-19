from __future__ import annotations

import json
import random
from pathlib import Path

from conversa_llm.tool_protocol import encode_tool_call


READ_PATHS = [
    "README.md", "pyproject.toml", "src/app.py", "src/router.py", "tests/test_app.py",
    "docs/setup.md", "config/settings.json", "scripts/build.py",
]
COMMANDS = [
    "pytest -q", "python -m pytest -q", "python -m compileall src",
    "git status --short", "python -m conversa_llm.chat --help",
]
WRITES = [
    ("notes.txt", "feito"),
    ("tmp/result.txt", "ok"),
    ("docs/status.txt", "validado"),
]
EDITS = [
    ("src/app.py", "alpha", "beta"),
    ("config/settings.json", "false", "true"),
    ("README.md", "antigo", "novo"),
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rng = random.Random(20260919)
    rows: list[dict] = []

    read_templates = [
        "Leia {path}.", "Abra {path} e me mostre o conteúdo.",
        "Inspecione o arquivo {path}.", "Confira {path} antes de continuar.",
    ]
    for path in READ_PATHS:
        answer = encode_tool_call("read", {"path": path})
        for template in read_templates:
            rows.append({"messages": [
                {"role": "user", "content": template.format(path=path)},
                {"role": "assistant", "content": answer},
            ]})

    bash_templates = [
        "Execute no shell o comando `{cmd}`.",
        "Rode `{cmd}` no terminal.",
        "Use bash para executar `{cmd}`.",
    ]
    for cmd in COMMANDS:
        answer = encode_tool_call("bash", {"command": cmd})
        for template in bash_templates:
            rows.append({"messages": [
                {"role": "user", "content": template.format(cmd=cmd)},
                {"role": "assistant", "content": answer},
            ]})

    for path, content in WRITES:
        prompts = [
            f"Crie {path} com exatamente este conteúdo: {content}",
            f"Escreva em {path}: {content}",
            f"Salve o texto {content} no arquivo {path}.",
        ]
        answer = encode_tool_call("write", {"path": path, "content": content})
        for prompt in prompts:
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]})

    for path, old, new in EDITS:
        prompts = [
            f"Substitua {old} por {new} em {path}.",
            f"Troque {old} por {new} dentro de {path}.",
            f"Edite {path}: altere {old} para {new}.",
        ]
        answer = encode_tool_call(
            "edit", {"path": path, "edits": [{"oldText": old, "newText": new}]}
        )
        for prompt in prompts:
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]})

    code_examples = [
        (
            "Escreva uma função Python soma(a, b) que retorna a soma.",
            "def soma(a, b):\n    return a + b\n",
        ),
        (
            "Crie uma função Python eh_par(n) que retorna True quando n é par.",
            "def eh_par(n):\n    return n % 2 == 0\n",
        ),
        (
            "Crie uma função Python maior(a, b) que retorna o maior valor.",
            "def maior(a, b):\n    return a if a > b else b\n",
        ),
        (
            "Crie uma função Python dobro(n) que retorna o dobro de n.",
            "def dobro(n):\n    return n * 2\n",
        ),
        (
            "Crie uma função Python inverter(texto) que devolve a string invertida.",
            "def inverter(texto):\n    return texto[::-1]\n",
        ),
    ]
    for prompt, code in code_examples:
        for prefix in ["", "Sem explicar, ", "Responda apenas com código. "]:
            rows.append({"messages": [
                {"role": "user", "content": prefix + prompt},
                {"role": "assistant", "content": code},
            ]})

    # Repete com variações de ordem para um corpus pequeno mas denso.
    dense = []
    for _ in range(8):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        dense.extend(shuffled)
    write_jsonl(root / "data" / "pi_curriculum.jsonl", dense)

    benchmark = [
        {"request": "Leia src/router.py.", "name": "read", "arguments": {"path": "src/router.py"}},
        {"request": "Abra docs/setup.md e veja o conteúdo.", "name": "read", "arguments": {"path": "docs/setup.md"}},
        {"request": "Execute no shell o comando `pytest tests -q` agora.", "name": "bash", "arguments": {"command": "pytest tests -q"}},
        {"request": "Rode `python -m compileall src` no terminal.", "name": "bash", "arguments": {"command": "python -m compileall src"}},
        {"request": "Crie report.txt com exatamente este conteúdo: pronto", "name": "write", "arguments": {"path": "report.txt", "content": "pronto"}},
        {"request": "Salve o texto ok no arquivo tmp/out.txt.", "name": "write", "arguments": {"path": "tmp/out.txt", "content": "ok"}},
        {"request": "Substitua foo por bar em src/app.py.", "name": "edit", "arguments": {"path": "src/app.py", "edits": [{"oldText": "foo", "newText": "bar"}]}},
        {"request": "Troque off por on dentro de config/settings.json.", "name": "edit", "arguments": {"path": "config/settings.json", "edits": [{"oldText": "off", "newText": "on"}]}},
    ]
    write_jsonl(root / "data" / "pi_tool_benchmark.jsonl", benchmark)

    code_benchmark = [
        {"name": "soma", "prompt": "Crie uma função soma(a, b) que retorna a soma.", "tests": [["soma(2, 3)", 5], ["soma(-1, 1)", 0]]},
        {"name": "eh_par", "prompt": "Crie uma função eh_par(n) que retorna True se n for par.", "tests": [["eh_par(4)", True], ["eh_par(5)", False]]},
        {"name": "maior", "prompt": "Crie uma função maior(a, b) que retorna o maior.", "tests": [["maior(7, 3)", 7], ["maior(2, 9)", 9]]},
        {"name": "dobro", "prompt": "Crie uma função dobro(n) que retorna o dobro.", "tests": [["dobro(8)", 16], ["dobro(-2)", -4]]},
        {"name": "inverter", "prompt": "Crie uma função inverter(texto) que inverte uma string.", "tests": [["inverter('abc')", "cba"], ["inverter('oi')", "io"]]},
    ]
    write_jsonl(root / "data" / "pi_code_benchmark.jsonl", code_benchmark)
    print(f"curriculum={len(dense)} tool_benchmark={len(benchmark)} code_benchmark={len(code_benchmark)}")


if __name__ == "__main__":
    main()
