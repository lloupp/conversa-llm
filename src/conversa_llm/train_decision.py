from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .decision_data import DecisionDataset, collate_decisions
from .decision_model import DecisionConfig, DecisionModel
from .tokenizer_loader import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a camada System-One de decisões.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer-file", required=True)
    parser.add_argument("--out", default="checkpoints/conversa-decision.pt")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = load_tokenizer(args.tokenizer_file)
    config = DecisionConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_model * 4,
    )
    dataset = DecisionDataset(args.data, tokenizer, args.context)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_decisions(batch, tokenizer.PAD),
    )
    model = DecisionModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    iterator = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            x, mask, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, mask, labels = next(iterator)
        x, mask, labels = x.to(args.device), mask.to(args.device), labels.to(args.device)
        _, loss = model(x, mask, labels)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss.item():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config.to_dict(),
        "model": model.state_dict(),
        "temperature": 1.0,
    }, out)
    params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint={out} params={params:,}")


if __name__ == "__main__":
    main()
