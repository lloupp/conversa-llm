from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import torch

from .chat import load_model
from .tokenizer_loader import load_tokenizer

FORBIDDEN = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
    ast.ClassDef, ast.Global, ast.Nonlocal,
)
SAFE_BUILTINS = {
    "range": range, "sum": sum, "len": len, "min": min, "max": max,
    "str": str, "int": int, "float": float, "bool": bool,
    "enumerate": enumerate, "zip": zip,
}


def generate(model, tok, prompt: str, device: str) -> str:
    prefix = tok.encode_prompt(prompt)
    if len(prefix) >= model.config.context_length:
        prefix = prefix[:2] + prefix[-(model.config.context_length - 3):]
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=160, temperature=1.0, top_k=1,
        stop_tokens={tok.EOS, tok.USER, tok.SEP},
    )[0].tolist()
    return tok.decode_text(out[len(prefix):]).strip().replace("```python", "").replace("```", "").strip()


def safe_exec_tests(code: str, tests: list[list]) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax: {exc}"
    if any(isinstance(node, FORBIDDEN) for node in ast.walk(tree)):
        return False, "forbidden AST"
    env = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(compile(tree, "<generated>", "exec"), env, env)
        for expr, expected in tests:
            value = eval(compile(ast.parse(expr, mode="eval"), "<test>", "eval"), env, env)
            if value != expected:
                return False, f"{expr} -> {value!r}, esperado {expected!r}"
    except Exception as exc:
        return False, f"runtime: {exc}"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--benchmark", default="data/pi_code_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    model = load_model(args.model, args.device)
    tok = load_tokenizer(args.tokenizer_file)
    rows = [
        json.loads(line)
        for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    for row in rows:
        code = generate(model, tok, row["prompt"], args.device)
        ok, reason = safe_exec_tests(code, row["tests"])
        passed += int(ok)
        if args.verbose:
            print(f"[{'OK' if ok else '--'}] {row['name']}: {reason}\n{code}\n")
    print(f"code_exec={passed}/{len(rows)}={100*passed/max(1,len(rows)):.1f}%")


if __name__ == "__main__":
    main()
