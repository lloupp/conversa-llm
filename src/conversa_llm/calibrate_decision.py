from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .decision_data import DecisionDataset, collate_decisions
from .decision_model import DecisionConfig, DecisionModel
from .tokenizer_loader import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra probabilidades do modelo System-One.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    model = DecisionModel(DecisionConfig(**checkpoint["config"])).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer_file)
    dataset = DecisionDataset(
        args.data,
        tokenizer,
        model.config.context_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda batch: collate_decisions(batch, tokenizer.PAD),
    )

    logits_all = []
    labels_all = []
    with torch.no_grad():
        for x, mask, labels in loader:
            logits, _ = model(x.to(args.device), mask.to(args.device))
            logits_all.append(logits.cpu())
            labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in torch.linspace(0.35, 3.0, 107):
        nll = F.cross_entropy(logits / temperature, labels).item()
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature.item())

    output = {
        "config": checkpoint["config"],
        "model": checkpoint["model"],
        "temperature": best_temperature,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out)
    print(f"temperature={best_temperature:.4f} calibration_nll={best_nll:.4f} out={out}")


if __name__ == "__main__":
    main()
