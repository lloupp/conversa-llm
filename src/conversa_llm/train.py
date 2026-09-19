from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import ConversationDataset, collate_batch
from .model import ConversaGPT
from .profiles import PROFILES, get_profile
from .tokenizer_loader import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treina o Conversa LLM do zero.")
    parser.add_argument(
        "--data", nargs="+", default=["data/sample_conversations.jsonl"],
        help="Um ou mais arquivos JSONL de conversa",
    )
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument(
        "--tokenizer-file", default=None,
        help="Tokenizer salvo (word ou byte-BPE); omita para byte-level puro",
    )
    parser.add_argument("--init-from", default=None, help="Checkpoint cujos pesos iniciam este treino")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument(
        "--supervise-all", action="store_true",
        help="Aplica loss a toda a sequência; útil para pré-treino pequeno.",
    )
    return parser


def resolve_training_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "batch_size": 8, "grad_accum": 1, "lr": 3e-4,
        "d_model": 128, "layers": 4, "heads": 4, "context": 256,
    }
    if args.profile:
        profile = get_profile(args.profile)
        defaults.update(
            batch_size=profile.batch_size, grad_accum=profile.grad_accum,
            lr=profile.lr, d_model=profile.d_model, layers=profile.layers,
            heads=profile.heads, context=profile.context,
        )
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.batch_size < 1 or args.grad_accum < 1 or args.steps < 1:
        raise ValueError("steps, batch-size e grad-accum devem ser >= 1")
    return args


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = resolve_training_args(build_parser().parse_args())
    torch.manual_seed(42)
    tokenizer = load_tokenizer(args.tokenizer_file)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_model * 4,
    )
    dataset = ConversationDataset(
        args.data, config.context_length,
        supervise_all=args.supervise_all, tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token=tokenizer.PAD),
    )
    model = ConversaGPT(config).to(args.device)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=args.device, weights_only=True)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("configuração do --init-from difere da configuração atual")
        model.load_state_dict(checkpoint["model"])
        print(f"pesos iniciais carregados de {args.init_from}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    parameter_count = count_parameters(model)
    print(
        f"device={args.device} params={parameter_count:,} examples={len(dataset)} "
        f"batch={args.batch_size} grad_accum={args.grad_accum} "
        f"batch_efetivo={args.batch_size * args.grad_accum} context={args.context}"
    )

    model.train()
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        running_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                input_ids, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_ids, labels = next(iterator)
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            _, loss = model(input_ids, labels)
            assert loss is not None
            (loss / args.grad_accum).backward()
            running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"step={step:04d} loss={running_loss / args.grad_accum:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "training": {
                "profile": args.profile, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "steps": args.steps, "lr": args.lr,
                "supervise_all": args.supervise_all, "init_from": args.init_from,
                "tokenizer_file": args.tokenizer_file,
            },
        },
        out,
    )
    print(f"checkpoint salvo em {out}")


if __name__ == "__main__":
    main()
