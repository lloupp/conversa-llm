from __future__ import annotations

import json
import random
from pathlib import Path

from conversa_llm.agent_prompt import render_agent_prompt
from conversa_llm.pi_tools import PI_CORE_TOOLS
from conversa_llm.tool_protocol import encode_tool_call


def tool_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }],
    }


def result_message(call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def add_state(rows: list[dict], messages: list[dict], completion: str) -> None:
    rows.append({
        "prompt": render_agent_prompt(messages, PI_CORE_TOOLS),
        "completion": completion,
    })


def build_fix_trajectory(
    rows: list[dict],
    path: str,
    old: str,
    new: str,
    failing_test: str,
    file_content: str,
) -> None:
    user = {
        "role": "user",
        "content": f"Corrija o bug em {path} e rode os testes. O teste que falha é {failing_test}.",
    }
    messages = [user]

    read_args = {"path": path}
    add_state(rows, messages, encode_tool_call("read", read_args))

    messages += [
        tool_message("c1", "read", read_args),
        result_message("c1", "read", file_content),
    ]
    edit_args = {"path": path, "edits": [{"oldText": old, "newText": new}]}
    add_state(rows, messages, encode_tool_call("edit", edit_args))

    messages += [
        tool_message("c2", "edit", edit_args),
        result_message("c2", "edit", "Arquivo atualizado com sucesso."),
    ]
    bash_args = {"command": f"pytest {failing_test} -q"}
    add_state(rows, messages, encode_tool_call("bash", bash_args))

    messages += [
        tool_message("c3", "bash", bash_args),
        result_message("c3", "bash", "1 passed in 0.08s"),
    ]
    add_state(rows, messages, "Corrigido e validado: o teste passou.")


def build_failed_test_recovery(rows: list[dict]) -> None:
    messages = [{
        "role": "user",
        "content": "Corrija src/calc.py para que dividir(a, b) faça divisão real e valide tests/test_calc.py.",
    }]
    read_args = {"path": "src/calc.py"}
    add_state(rows, messages, encode_tool_call("read", read_args))
    messages += [
        tool_message("f1", "read", read_args),
        result_message("f1", "read", "def dividir(a, b):\n    return a // b\n"),
    ]
    edit1 = {
        "path": "src/calc.py",
        "edits": [{"oldText": "return a // b", "newText": "return a / b"}],
    }
    add_state(rows, messages, encode_tool_call("edit", edit1))
    messages += [
        tool_message("f2", "edit", edit1),
        result_message("f2", "edit", "Arquivo atualizado."),
    ]
    test_args = {"command": "pytest tests/test_calc.py -q"}
    add_state(rows, messages, encode_tool_call("bash", test_args))
    messages += [
        tool_message("f3", "bash", test_args),
        result_message(
            "f3",
            "bash",
            "FAILED tests/test_calc.py::test_zero - ZeroDivisionError: division by zero",
        ),
    ]
    edit2 = {
        "path": "src/calc.py",
        "edits": [{
            "oldText": "def dividir(a, b):\n    return a / b",
            "newText": "def dividir(a, b):\n    if b == 0:\n        return None\n    return a / b",
        }],
    }
    add_state(rows, messages, encode_tool_call("edit", edit2))
    messages += [
        tool_message("f4", "edit", edit2),
        result_message("f4", "edit", "Arquivo atualizado."),
    ]
    add_state(rows, messages, encode_tool_call("bash", test_args))
    messages += [
        tool_message("f5", "bash", test_args),
        result_message("f5", "bash", "2 passed in 0.09s"),
    ]
    add_state(rows, messages, "Correção concluída e os 2 testes passaram.")


def build_exploration(rows: list[dict]) -> None:
    cases = [
        (
            "Descubra onde a função autenticar está definida e leia o arquivo.",
            "grep",
            {"pattern": "def autenticar", "path": "."},
            "src/auth.py:12:def autenticar(usuario):",
            "read",
            {"path": "src/auth.py"},
        ),
        (
            "Encontre os testes Python do módulo de pagamentos e inspecione o principal.",
            "find",
            {"pattern": "**/*payment*.py", "path": "tests"},
            "tests/test_payment.py\ntests/integration/test_payment_api.py",
            "read",
            {"path": "tests/test_payment.py"},
        ),
        (
            "Liste o conteúdo de src para eu saber a estrutura.",
            "ls",
            {"path": "src"},
            "api/\napp.py\nmodels/\nutils.py",
            None,
            None,
        ),
    ]
    for index, (request, first_name, first_args, result, second_name, second_args) in enumerate(cases):
        messages = [{"role": "user", "content": request}]
        add_state(rows, messages, encode_tool_call(first_name, first_args))
        messages += [
            tool_message(f"e{index}a", first_name, first_args),
            result_message(f"e{index}a", first_name, result),
        ]
        if second_name:
            add_state(rows, messages, encode_tool_call(second_name, second_args))
        else:
            add_state(rows, messages, "A estrutura de src foi listada.")


def build_direct_tool_states(rows: list[dict]) -> None:
    direct = [
        ("Leia README.md.", "read", {"path": "README.md"}),
        ("Liste os arquivos em src.", "ls", {"path": "src"}),
        ("Procure `TODO` em src.", "grep", {"pattern": "TODO", "path": "src"}),
        ("Encontre `**/*.py` em tests.", "find", {"pattern": "**/*.py", "path": "tests"}),
        ("Rode `pytest -q` no terminal.", "bash", {"command": "pytest -q"}),
        (
            "Substitua DEBUG = True por DEBUG = False em config.py.",
            "edit",
            {
                "path": "config.py",
                "edits": [{"oldText": "DEBUG = True", "newText": "DEBUG = False"}],
            },
        ),
        (
            "Crie status.txt com exatamente este conteúdo: ok",
            "write",
            {"path": "status.txt", "content": "ok"},
        ),
    ]
    for request, name, args in direct:
        add_state(
            rows,
            [{"role": "user", "content": request}],
            encode_tool_call(name, args),
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base: list[dict] = []
    build_direct_tool_states(base)
    build_exploration(base)
    build_failed_test_recovery(base)

    build_fix_trajectory(
        base,
        "src/math_utils.py",
        "return a - b",
        "return a + b",
        "tests/test_math_utils.py",
        "def soma(a, b):\n    return a - b\n",
    )
    build_fix_trajectory(
        base,
        "src/text.py",
        "return texto",
        "return texto[::-1]",
        "tests/test_text.py",
        "def inverter(texto):\n    return texto\n",
    )
    build_fix_trajectory(
        base,
        "src/paridade.py",
        "return n % 2 == 1",
        "return n % 2 == 0",
        "tests/test_paridade.py",
        "def eh_par(n):\n    return n % 2 == 1\n",
    )

    rng = random.Random(20260919)
    train: list[dict] = []
    for _ in range(20):
        copy = base[:]
        rng.shuffle(copy)
        train.extend(copy)
    write_jsonl(root / "data" / "pi_agent_curriculum.jsonl", train)

    benchmark: list[dict] = []
    heldout_states: list[tuple[list[dict], str, dict | None, str | None]] = [
        (
            [{"role": "user", "content": "Procure `FIXME` em src."}],
            "grep",
            {"pattern": "FIXME", "path": "src"},
            None,
        ),
        (
            [{"role": "user", "content": "Encontre `**/*api*.py` em tests."}],
            "find",
            {"pattern": "**/*api*.py", "path": "tests"},
            None,
        ),
        (
            [{"role": "user", "content": "Liste o conteúdo de docs."}],
            "ls",
            {"path": "docs"},
            None,
        ),
    ]
    for messages, name, args, text in heldout_states:
        row = {"prompt": render_agent_prompt(messages, PI_CORE_TOOLS)}
        if name:
            row.update({"name": name, "arguments": args})
        else:
            row["text_contains"] = text
        benchmark.append(row)

    # Estado multi-turn inédito: depois de editar, deve testar.
    messages = [{
        "role": "user",
        "content": "Corrija src/slug.py e valide tests/test_slug.py.",
    }]
    messages += [
        tool_message("b1", "read", {"path": "src/slug.py"}),
        result_message("b1", "read", "def slug(s):\n    return s.upper()\n"),
        tool_message(
            "b2",
            "edit",
            {
                "path": "src/slug.py",
                "edits": [{"oldText": "return s.upper()", "newText": "return s.lower()"}],
            },
        ),
        result_message("b2", "edit", "Arquivo atualizado."),
    ]
    benchmark.append({
        "prompt": render_agent_prompt(messages, PI_CORE_TOOLS),
        "name": "bash",
        "arguments": {"command": "pytest tests/test_slug.py -q"},
    })

    # Estado após sucesso: deve concluir em texto, não chamar outra tool.
    messages += [
        tool_message("b3", "bash", {"command": "pytest tests/test_slug.py -q"}),
        result_message("b3", "bash", "3 passed in 0.10s"),
    ]
    benchmark.append({
        "prompt": render_agent_prompt(messages, PI_CORE_TOOLS),
        "text_contains": "pass",
    })

    write_jsonl(root / "data" / "pi_agent_benchmark.jsonl", benchmark)
    print(f"agent_train={len(train)} agent_benchmark={len(benchmark)}")


if __name__ == "__main__":
    main()
