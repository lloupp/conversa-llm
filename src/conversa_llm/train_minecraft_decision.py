from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .decision_model import DecisionConfig
from .minecraft_decision import MINECRAFT_ACTION_TO_ID, MinecraftDecisionModel
from .tokenizer_loader import load_tokenizer


class MinecraftDataset(Dataset):
    def __init__(self, paths, tokenizer, context_length):
        self.examples = []
        for source in paths:
            with Path(source).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    state, label = row.get("state"), row.get("label")
                    if not isinstance(state, str) or label not in MINECRAFT_ACTION_TO_ID:
                        raise ValueError(f"{source}:{line_number}: state/label inválidos")
                    ids = tokenizer.encode_text(state)[-context_length:]
                    if ids:
                        self.examples.append((ids, MINECRAFT_ACTION_TO_ID[label]))
        if not self.examples:
            raise ValueError("dataset Minecraft vazio")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(batch, pad):
    width = max(len(ids) for ids, _ in batch)
    x, mask, labels = [], [], []
    for ids, label in batch:
        n = width - len(ids)
        x.append(ids + [pad] * n)
        mask.append([1] * len(ids) + [0] * n)
        labels.append(label)
    return torch.tensor(x), torch.tensor(mask, dtype=torch.bool), torch.tensor(labels)


def main():
    p = argparse.ArgumentParser(description="Treina o perfil de decisões Minecraft.")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--tokenizer-file", required=True)
    p.add_argument("--out", default="checkpoints/conversa-minecraft.pt")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(42)
    tok = load_tokenizer(args.tokenizer_file)
    cfg = DecisionConfig(vocab_size=tok.vocab_size, context_length=args.context, d_model=args.d_model,
                         n_heads=args.heads, n_layers=args.layers, d_ff=args.d_model * 4,
                         n_actions=len(MINECRAFT_ACTION_TO_ID))
    data = MinecraftDataset(args.data, tok, args.context)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.PAD))
    model = MinecraftDecisionModel(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            x, mask, labels = next(it)
        except StopIteration:
            it = iter(loader); x, mask, labels = next(it)
        x, mask, labels = x.to(args.device), mask.to(args.device), labels.to(args.device)
        _, loss = model(x, mask, labels)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step:04d} loss={loss.item():.4f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "temperature": 1.0, "profile": "minecraft"}, out)
    print(f"checkpoint={out}")


if __name__ == "__main__":
    main()
