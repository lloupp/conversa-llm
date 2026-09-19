from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .chat import load_model
from .tokenizer_loader import load_tokenizer
from .tool_protocol import parse_tool_calls


def greedy(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 128) -> str:
    prefix = tokenizer.encode_prompt(prompt)
    if len(prefix) >= model.config.context_length:
        prefix = prefix[:2] + prefix[-(model.config.context_length - 3):]
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1,
        stop_tokens={tokenizer.EOS, tokenizer.USER, tokenizer.SEP},
    )[0].tolist()
    return tokenizer.decode_text(out[len(prefix):]).strip()


def evaluate(model, tokenizer, benchmark: str | Path, device: str, verbose: bool = False):
    rows = [
        json.loads(line)
        for line in Path(benchmark).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exact = 0
    tool_only = 0
    details = []
    for row in rows:
        output = greedy(model, tokenizer, row["prompt"], device)
        calls = parse_tool_calls(output)
        first = calls[0] if calls else None
        tool_hit = bool(first and first.name == row["name"])
        exact_hit = bool(tool_hit and first.arguments == row["arguments"])
        tool_only += int(tool_hit)
        exact += int(exact_hit)
        details.append({**row, "output": output, "tool_correct": tool_hit, "exact": exact_hit})
        if verbose:
            print(f"[{'OK' if exact_hit else '--'}] {row['request']} -> {output!r}")
    return {"total": len(rows), "tool_correct": tool_only, "exact": exact, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--benchmark", default="data/pi_tool_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    model = load_model(args.model, args.device)
    tokenizer = load_tokenizer(args.tokenizer_file)
    result = evaluate(model, tokenizer, args.benchmark, args.device, args.verbose)
    total = max(1, result["total"])
    print(
        f"tool_name={result['tool_correct']}/{result['total']}={100*result['tool_correct']/total:.1f}% "
        f"exact={result['exact']}/{result['total']}={100*result['exact']/total:.1f}%"
    )


if __name__ == "__main__":
    main()
