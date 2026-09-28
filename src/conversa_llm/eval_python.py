from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import load_model
from .tokenizer import ByteTokenizer


def _candidate_example(tokenizer: ByteTokenizer, question: str, candidate: str, context_length: int):
    prefix = tokenizer.encode_prompt(question)
    answer = (tokenizer.encode_text(candidate) + [tokenizer.SEP])[-(context_length - 1):]
    keep_prefix = max(1, context_length + 1 - len(answer))
    prefix = prefix[-keep_prefix:]
    tokens = prefix + answer
    inputs = tokens[:-1]
    start = len(prefix) - 1
    labels = [-100] * len(inputs)
    for i in range(start, len(inputs)):
        labels[i] = tokens[i + 1]
    return inputs, labels


def rank_choices(model, tokenizer: ByteTokenizer, question: str, choices: list[str], device: str):
    examples = [
        _candidate_example(tokenizer, question, choice, model.config.context_length)
        for choice in choices
    ]
    max_len = max(len(x[0]) for x in examples)
    inputs, labels = [], []
    for input_ids, target_ids in examples:
        pad = max_len - len(input_ids)
        inputs.append(input_ids + [tokenizer.PAD] * pad)
        labels.append(target_ids + [-100] * pad)

    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(y.shape)
        mask = y.ne(-100)
        mean_nll = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    order = torch.argsort(mean_nll).tolist()
    return order, mean_nll.tolist()


def evaluate(model, benchmark_path: str | Path, device: str = "cpu", verbose: bool = False):
    tokenizer = ByteTokenizer()
    rows = [json.loads(line) for line in Path(benchmark_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    correct = 0
    details = []
    model.eval()
    for row in rows:
        order, scores = rank_choices(model, tokenizer, row["question"], row["choices"], device)
        predicted = row["choices"][order[0]]
        hit = predicted == row["answer"]
        correct += int(hit)
        detail = {
            "skill": row.get("skill"),
            "question": row["question"],
            "expected": row["answer"],
            "predicted": predicted,
            "correct": hit,
            "best_nll": scores[order[0]],
        }
        details.append(detail)
        if verbose:
            mark = "OK" if hit else "--"
            print(f"[{mark}] {row.get('skill', '?')}: {predicted}")
    accuracy = correct / len(rows) if rows else 0.0
    return {"correct": correct, "total": len(rows), "accuracy": accuracy, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia conhecimento Python por ranking de respostas inéditas.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", default="data/python_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    result = evaluate(model, args.benchmark, args.device, args.verbose)
    print(
        f"python_benchmark={result['correct']}/{result['total']} "
        f"accuracy={result['accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
