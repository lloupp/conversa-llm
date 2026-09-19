from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch

from .chat import load_model
from .tokenizer import ByteTokenizer
from .word_tokenizer import WordTokenizer


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("`", "").strip()
    return re.sub(r"\s+", " ", text)


def answer_matches(output: str, expected: str) -> bool:
    got = normalize_text(output)
    want = normalize_text(expected)
    if not got:
        return False
    symbol_rich = any(ch in want for ch in '=/*%[](){}"\'')
    if symbol_rich:
        return re.sub(r"\s+", "", want) in re.sub(r"\s+", "", got)
    return re.search(rf"(?<!\w){re.escape(want)}(?!\w)", got) is not None


def greedy_answer(model, tokenizer, question: str, device: str, max_new_tokens: int = 30) -> str:
    prefix = tokenizer.encode_prompt(question)
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
        stop_tokens={tokenizer.EOS, tokenizer.USER, tokenizer.SEP},
    )[0].tolist()
    return tokenizer.decode_text(out[len(prefix):]).strip()


def evaluate_generation(
    model,
    benchmark_path: str | Path,
    device: str = "cpu",
    verbose: bool = False,
    tokenizer=None,
):
    tokenizer = tokenizer or ByteTokenizer()
    rows = [json.loads(line) for line in Path(benchmark_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    details = []
    model.eval()
    for row in rows:
        output = greedy_answer(model, tokenizer, row["question"], device)
        hit = answer_matches(output, row["answer"])
        correct += int(hit)
        details.append({**row, "output": output, "correct": hit})
        if verbose:
            print(f"[{'OK' if hit else '--'}] {row.get('skill', '?')}: {output!r} | esperado={row['answer']!r}")
    accuracy = correct / len(rows) if rows else 0.0
    return {"correct": correct, "total": len(rows), "accuracy": accuracy, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia respostas Python por geração livre greedy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="data/python_generation_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tokenizer-file", default=None)
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    tokenizer = WordTokenizer.load(args.tokenizer_file) if args.tokenizer_file else ByteTokenizer()
    result = evaluate_generation(model, args.benchmark, args.device, args.verbose, tokenizer=tokenizer)
    print(
        f"python_generation={result['correct']}/{result['total']} "
        f"accuracy={result['accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
