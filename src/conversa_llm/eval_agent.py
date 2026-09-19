from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .chat import load_model
from .tokenizer_loader import load_tokenizer
from .tool_protocol import parse_tool_calls


def greedy(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 160) -> str:
    prefix = tokenizer.encode_prompt(prompt)
    if len(prefix) >= model.config.context_length:
        prefix = prefix[:2] + prefix[-(model.config.context_length - 3):]
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
        stop_tokens={tokenizer.EOS, tokenizer.USER, tokenizer.SEP},
    )[0].tolist()
    return tokenizer.decode_text(out[len(prefix):]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia estados multi-turn do agente Pi.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--benchmark", default="data/pi_agent_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    tokenizer = load_tokenizer(args.tokenizer_file)
    rows = [
        json.loads(line)
        for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0
    for row in rows:
        output = greedy(model, tokenizer, row["prompt"], args.device)
        calls = parse_tool_calls(output)
        if "name" in row:
            first = calls[0] if calls else None
            ok = bool(
                first
                and first.name == row["name"]
                and first.arguments == row["arguments"]
            )
        else:
            needle = row["text_contains"].lower()
            ok = not calls and needle in output.lower()
        passed += int(ok)
        if args.verbose:
            print(f"[{'OK' if ok else '--'}] {output!r}")

    total = len(rows)
    print(f"agent_state={passed}/{total}={100*passed/max(1,total):.1f}%")


if __name__ == "__main__":
    main()
