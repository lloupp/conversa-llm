from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .decision_model import ACTIONS, ACTION_TO_ID, DecisionConfig, DecisionModel
from .tokenizer_loader import load_tokenizer


def load_decision(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DecisionModel(DecisionConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, float(checkpoint.get("temperature", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--data", default="data/decision_benchmark.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, temperature = load_decision(args.model, args.device)
    tokenizer = load_tokenizer(args.tokenizer_file)
    rows = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct = 0
    nll = 0.0
    brier = 0.0
    confidences = []
    hits = []
    for row in rows:
        ids = tokenizer.encode_text(row["state"])[-model.config.context_length :]
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        mask = torch.ones_like(x, dtype=torch.bool)
        logits, _ = model(x, mask)
        probs = F.softmax(logits / temperature, dim=-1)[0]
        target = ACTION_TO_ID[row["label"]]
        pred = int(probs.argmax().item())
        hit = pred == target
        correct += int(hit)
        nll += float(-torch.log(probs[target].clamp_min(1e-9)).item())
        one_hot = F.one_hot(torch.tensor(target), len(ACTIONS)).float().to(probs.device)
        brier += float(torch.mean((probs - one_hot) ** 2).item())
        confidences.append(float(probs.max().item()))
        hits.append(float(hit))

    # ECE simples com 10 bins.
    ece = 0.0
    for low_i in range(10):
        low, high = low_i / 10, (low_i + 1) / 10
        selected = [i for i, conf in enumerate(confidences) if low <= conf < high or (high == 1 and conf == 1)]
        if not selected:
            continue
        avg_conf = sum(confidences[i] for i in selected) / len(selected)
        avg_acc = sum(hits[i] for i in selected) / len(selected)
        ece += len(selected) / max(1, len(rows)) * abs(avg_conf - avg_acc)

    total = max(1, len(rows))
    print(
        f"decision_accuracy={correct}/{len(rows)}={100*correct/total:.1f}% "
        f"nll={nll/total:.4f} brier={brier/total:.4f} ece={ece:.4f}"
    )


if __name__ == "__main__":
    main()
